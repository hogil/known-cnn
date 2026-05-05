#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DDP (DistributedDataParallel) helper — minimal env-driven setup for cnn_train*.py.

Usage pattern in trainer:

    from _ddp import init_ddp, wrap_ddp, make_sampler, cleanup_ddp, unwrap

    ddp = init_ddp()
    device = ddp["device"] if ddp["enabled"] else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = ddp["is_main"]

    train_sampler = make_sampler(train_set, ddp, shuffle=True)
    train_ld = DataLoader(train_set, sampler=train_sampler,
                          shuffle=(train_sampler is None), ...)
    val_sampler = make_sampler(val_set, ddp, shuffle=False)
    val_ld = DataLoader(val_set, sampler=val_sampler, shuffle=False, ...)

    model = build_model(...).to(device)
    model = wrap_ddp(model, ddp)

    for ep in range(epochs):
        if ddp["enabled"]:
            train_sampler.set_epoch(ep)
        train_one_epoch(model, train_ld, ...)
        if is_main:
            # save / log only on rank 0
            torch.save(unwrap(model).state_dict(), "best.pth")

    cleanup_ddp(ddp)

Launch: torchrun --nproc-per-node=<N_GPU> --standalone cnn_train_chip.py [args...]

Env vars set by torchrun: LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def init_ddp() -> Dict[str, Any]:
    """Init DDP from torchrun env vars. Single-GPU returns enabled=False."""
    if "LOCAL_RANK" not in os.environ:
        return {
            "enabled": False, "local_rank": 0, "global_rank": 0, "world_size": 1,
            "device": None, "is_main": True,
        }
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested but CUDA unavailable")
    torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return {
        "enabled": True, "local_rank": local_rank, "global_rank": global_rank,
        "world_size": world_size, "device": torch.device(f"cuda:{local_rank}"),
        "is_main": (global_rank == 0),
    }


def wrap_ddp(model: torch.nn.Module, ddp_state: Dict[str, Any],
             find_unused_parameters: bool = False) -> torch.nn.Module:
    """Wrap model in DDP if enabled, else return as-is."""
    if not ddp_state["enabled"]:
        return model
    return DDP(model, device_ids=[ddp_state["local_rank"]],
               output_device=ddp_state["local_rank"],
               find_unused_parameters=find_unused_parameters)


def make_sampler(dataset, ddp_state: Dict[str, Any], shuffle: bool = True
                 ) -> Optional[DistributedSampler]:
    """Make DistributedSampler if DDP enabled, else None (caller falls back to shuffle=True/False)."""
    if not ddp_state["enabled"]:
        return None
    return DistributedSampler(dataset, num_replicas=ddp_state["world_size"],
                              rank=ddp_state["global_rank"], shuffle=shuffle, drop_last=False)


def cleanup_ddp(ddp_state: Dict[str, Any]) -> None:
    """Barrier + destroy process group on shutdown."""
    if not ddp_state["enabled"]:
        return
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Get model.module if DDP-wrapped, else return self.
    Use for state_dict / named_parameters access."""
    return model.module if hasattr(model, "module") else model


def all_reduce_mean(value: float, ddp_state: Dict[str, Any]) -> float:
    """All-reduce a scalar across ranks (mean). Single-GPU passthrough."""
    if not ddp_state["enabled"]:
        return value
    t = torch.tensor([float(value)], device=ddp_state["device"])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / ddp_state["world_size"]


def barrier(ddp_state: Dict[str, Any]) -> None:
    """Synchronize all ranks. No-op if single-GPU."""
    if ddp_state["enabled"] and dist.is_initialized():
        dist.barrier()
