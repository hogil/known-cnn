# Reboot Resume - Mega Matrix

Current pushed head:

```text
9305233 Constrain default mega run GPU memory
```

Default run target:

```text
script:   mega_matrix/run.sh
backbone: convnextv2_base.fcmae_ft_in22k_in1k_384
weight:   mega_matrix/weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth
batch:    24
hidden max forward batch: <= 96
GPU mode: run.sh defaults CUDA_VISIBLE_DEVICES=0
DDP/multi-GPU: only via mega_matrix/run_ddp.sh
```

Important cause of prior OOM:

```text
cutmix-mode complement + cutmix-pair masked + cutmix-n-groups 2
expands the actual forward batch up to 4x.
Displayed batch=24 means forward batch can be <=96.
```

After reboot, run:

```bash
cd /path/to/known-cnn
git pull
git rev-parse --short HEAD
ls -lh mega_matrix/weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth
nvidia-smi
bash mega_matrix/run.sh
```

Expected first log line should include:

```text
backbone=convnextv2_base.fcmae_ft_in22k_in1k_384
img=384
batch=24
effective_forward_batch<=96
cuda_visible=0
```

If the weight file is missing, create it on an internet machine:

```bash
python mega_matrix/download.py --allow-download --only convnextv2_base.fcmae_ft_in22k_in1k_384
```

Then copy only the generated `.pth` into:

```text
mega_matrix/weights/
```

Do not commit weights. `mega_matrix/weights/` is ignored by Git.

If OOM happens again, first check whether another old process is already using GPU:

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Then kill old python/torchrun processes before retrying.
