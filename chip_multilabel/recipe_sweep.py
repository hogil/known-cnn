# -*- coding: utf-8 -*-
"""Sequential chip multi-label recipe sweep.

This runner keeps dataset roots separated so results from different image pools
cannot be mistaken for each other.  Each recipe produces:

- train_root evaluation metrics
- eval_root evaluation metrics
- train_root pos/neg probability diagnostics
- eval_root pos/neg probability diagnostics with per-class 4-bit probabilities
- one leaderboard row with explicit train/eval paths
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
WEIGHTS = Path("models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth")
BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
PCLS_ORDER = [
    "bank_boundary",
    "fork",
    "scratch",
    "scratch_rot",
    "bank_boundary+fork",
    "bank_boundary+scratch",
    "bank_boundary+scratch_rot",
    "fork+scratch",
    "fork+scratch_rot",
    "scratch+scratch_rot",
    "Normal",
    "Invalid",
    "CenterDonut",
    "CrossScratch",
    "DiagonalSmear",
    "Starburst",
]
PCLS_RANK = {name: i for i, name in enumerate(PCLS_ORDER)}

# Live pruning from canonical old-eval evidence on 2026-05-29:
# cmp=0.595/0.605/0.61/0.615 around iter116J_T7_g3_p020 all
# collapsed or leaked FAR before reaching the cmp=0.60 baseline.  Skip the
# remaining high-cmp iter116J branch and move compute to p/LS/g4/fcm_margin.
LIVE_PRUNED_TAGS = {
    # v15direct_n2000 high-p matrix: p=0.30/cmp=0.50/g3/grid8 failed to
    # recover POS recall and repeatedly leaked tails (ep01 FAR 58.33%,
    # ep03 FAR 74.67%, ep06 FAR 54.67%; best quick bit_F1 only 0.7941).
    "fcmpm_hpmat_T7_LS02975_g3_grid8_cmp05000_p03000_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 evidence: best_model full eval bit_F1=0.9087/FAR=0.02;
    # ep06 quick was only 0.9388/FAR=0.33, so grid6+cmp-down is not worth
    # additional full eval or transfer repeats.
    "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
    "iter116J_T7_g3_cmp0615_p020_s1_ep10_I10only_far0_edge",
    "iter116J_T7_g3_cmp062_p020_s1_ep10_I10only_far0_edge",
    "iter116J_T7_g3_cmp0625_p020_s1_ep10_I10only_far0_edge",
    "iter116J_T7_g3_cmp063_p020_s1_ep10_I10only_far0_edge",
    "iter116J_T7_g3_cmp064_p020_s1_ep10_I10only_far0_edge",
    "iter116J_T7_g3_cmp0645_p020_s1_ep10_I10only_far0_edge",
    "iter116J_T7_g3_cmp065_p020_s1_ep10_I10only_combo_pos_push",
    "iter116J_T7_g3_cmp065_p0195_s1_ep10_I10only_far0_edge",
    "iter116J_T7_g3_cmp065_p019_s1_ep10_I10only_far0_edge",
    "iter116J_T7_g3_cmp065_p018_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp065_p018_s1_ep10_I10only_combo_pos_push",
    "iter116J_T7_LS028_g3_cmp06_p020_s1_ep10_I10only_scratch_combo_balance",
    "iter116J_T7_LS032_g3_cmp06_p020_s1_ep10_I10only_scratch_combo_balance",
    "iter116J_T7_LS028_g3_cmp065_p020_s1_ep10_I10only_scratch_combo_balance",
    "iter116J_T7_LS032_g3_cmp06_p022_s1_ep10_I10only_scratch_combo_balance",
    "iter116J_T7_g4_cmp06_p018_s1_ep10_I10only_g4_tail_balance",
    "iter116J_T7_g3_cmp06_p025_s1_ep10_I10only_combo_recall_retry",
    "iter116J_T7_g3_cmp07_p025_s1_ep10_I10only_combo_recall_retry",
    "iter116J_T7_g3_cmp06_p020_nopair_s1_ep10_I10only_tail_check",
    "iter116J_T7_g4_cmp05_p020_s1_ep10_I10only_g4_tail_cmp05",
    "iter116J_T7_g4_cmp06_p020_s1_ep10_I10only_g4_tail_cmp06",
    "iter116J_T7_g3_cmp07_p020_s1_ep10_I10only_gap_balance",
    "iter116J_T7_g4_cmp07_p025_s1_ep10_I10only_gap_balance",
    "iter116J_T7_g4_cmp07_p020_s1_ep10_I10only_gap_balance",
    "fcm_margin_g3_cls06_ls30_pair_s42_ep8_I10only",
    "fcm_margin_g3_cls07_ls30_nopair_s42_ep8_I10only",
    "fcm_margin_g3_cls07_ls30_pair_s42_ep8_I10only",
    "fcm_margin_g3_cls07_ls30_pair_s42_ep10",
    "fcm_margin_g4_cls05_ls30_nopair_s42_ep8_I10only",
    "fcm_margin_g3_cls07_ls30_nopair_s42_ep10",
    "fcm_margin_g3_cls065_ls30_nopair_s42_ep10_I10only",
    "fcm_margin_g3_cls065_ls30_pair_s42_ep10_I10only",
    "fcm_margin_g4_cls06_ls30_nopair_s42_ep10_I10only",
    "fcm_margin_g4_cls05_ls30_nopair_s42_ep10",
    "fcm_margin_g4_cls05_ls30_pair_s42_ep10",
    "fcm_margin_g2_cls05_ls30_nopair_s42_ep10",
    "fcm_margin_g2_cls05_ls30_pair_s42_ep10",
    "fcm_margin_g3_cls05_ls40_nopair_s42_ep10",
    "fcm_margin_g3_cls05_ls40_pair_s42_ep10",
    "iter26D_T7_LS040_g4_cmp05_p025_s1_ep8",
    "iter26H_T7_LS067_g3_cmp05_p025_white_s1_ep8",
    "iter21F_T7_LS067_g3_cmp05_p025_s1_ep8",
    "iter21H_T7_LS075_g4_cmp05_p025_s1_ep8",
    "iter21E_T7_LS100_g2_cmp10_p025_s1_ep8",
    "iter25_T7_LS020_g2_cmp05_p025_s1_ep8",
    "iter25_T7_LS020_g2_cmp05_p025_s7_ep8",
    "iter25_T7_LS020_g2_cmp05_p025_s42_ep8",
    "iter25_T7_LS030_g2_cmp05_p025_s1_ep8",
    "iter25_T7_LS030_g2_cmp05_p025_s7_ep8",
    "iter25_T7_LS030_g2_cmp05_p025_s42_ep8",
    "iter116J_exact_T7_LS030_g3_cmp05_p025_s7_ep10",
    "iter116J_exact_T7_LS030_g3_cmp05_p025_s42_ep10",
    "iter116J_exact_T7_LS030_g3_cmp05_p025_s77_ep10",
    "adapt_T7_LS024_g3_cmp070_p025_pair_corner_s1_ep10",
    "fcm_margin_T7_LS030_g3_cmp07_p025_s1_ep10",
    "fcm_margin_nopair_T7_LS032_g3_cmp07_p025_s1_ep10_normal_tail",
    "fcm_margin_nopair_T7_LS035_g3_cmp07_p025_s1_ep10_normal_tail",
    "fcm_margin_nopair_T7_LS030_g3_cmp065_p025_s1_ep10_normal_tail",
    "iter26B_T7_LS050_g3_cmp05_p025_s1_ep8",
    "fcm_margin_pair_T7_LS030_g3_cmp0725_p025_s1_ep10_combo_push",
    "fcm_margin_pair_T7_LS030_g3_cmp075_p025_s1_ep10_combo_push",
    "fcm_margin_pair_T7_LS030_g3_cmp07_p0225_s1_ep10_normal_tail",
    "fcm_margin_pair_T7_LS030_g3_cmp0675_p025_s1_ep10_normal_tail",
    "fcm_margin_pair_T7_LS032_g3_cmp07_p025_s1_ep10_normal_tail",
    "fcm_margin_pair_T7_LS029_g3_cmp07_p025_s1_ep10_mid_balance",
    "fcm_margin_pair_T7_LS030_g4_cmp07_p025_s1_ep10_normal_tail",
    "fcm_margin_pair_T7_LS02925_g3_cmp07_p025_s1_ep10_mid_balance",
    "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s1_ep10_mid_balance",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p0255_s1_ep10_combo_pos_push",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p026_s1_ep10_combo_pos_push",
    "fcm_margin_pair_T7_LS0295_g3_cmp0705_p025_s1_ep10_combo_pos_push",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s77_ep10_seedcheck",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p024_s1_ep10_neg_tail_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp0695_p025_s1_ep10_neg_tail_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp069_p025_s1_ep10_neg_tail_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp0685_p025_s7_ep10_seed7_neg_tail_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp0695_p025_s7_ep10_seed7_neg_tail_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp0705_p02475_s7_ep10_seed7_combo_balance",
    "fcm_margin_pair_T7_LS02925_g3_cmp07_p025_s7_ep10_seed7_ls_combo_balance",
    "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_ep10_seed7_ls_tail_balance",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p02475_s7_ep10_seed7_p_tail_balance",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p02525_s7_ep10_seed7_p_combo_balance",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p0245_s7_ep10_seed7_p_tail_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p0255_s7_ep10_seed7_p_combo_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_fill_white_ep10_tail_test",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_fill_noise_ep10_tail_test",
    "fcm_margin_pair_T7_LS030_g3_cmp07_p025_s7_ep10_seedcheck",
    "fcm_margin_pair_T7_LS030_g3_cmp07_p025_s42_ep10_seedcheck",
    "fcm_margin_pair_T7_LS030_g3_cmp07_p025_s77_ep10_seedcheck",
    "fcm_margin_g4_cls05_ls30_nopair_s42_ep10_retry_old_eval",
    "fcm_margin_g4_cls05_ls30_pair_s42_ep10_retry_old_eval",
    "iter116J_T7_g3_cmp06_p025_s1_ep10_I10only_combo_recall",
    "iter116J_T7_g3_cmp07_p025_s1_ep10_I10only_combo_recall",
    "iter116J_T7_g4_cmp06_p025_s1_ep10_I10only_combo_ood_balance",
    "iter116J_T7_g3_cmp06_p020_s1_ep10_I10only_less_cutmix",
    "iter116J_T7_g3_cmp06_p022_s1_ep10_I10only_combo_pos_push",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p02475_s1_ep10_neg_tail_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp071_p02475_s1_ep10_combo_pos_push",
    "adapt_T7_LS029_g3_cmp070_p025_pair_corner_s1_ep10",
    "adapt_T7_LS034_g3_cmp070_p025_pair_corner_s1_ep10",
    "adapt_T7_LS029_g3_cmp070_p025_pair_s7_ep10",
    "adapt_T7_LS029375_g3_cmp070_p025_pair_s7_ep10",
    "adapt_T7_LS029625_g3_cmp070_p025_pair_s7_ep10",
    "adapt_T7_LS0295_g3_cmp0695_p025_pair_s7_ep10",
    "adapt_T7_LS0295_g3_cmp06975_p025_pair_s7_ep10",
    "adapt_T7_LS0295_g3_cmp07025_p025_pair_s7_ep10",
    "adapt_T7_LS0295_g3_cmp06975_p02525_pair_s7_ep10",
    "adapt_T7_LS0295_g3_cmp07025_p02475_pair_s7_ep10",
    "adapt_T7_LS0295_g3_cmp06975_p02525_pair_s13_ep10",
    "adapt_T7_LS0295_g3_cmp07025_p02475_pair_s13_ep10",
    "adapt_T7_LS0295_g3_cmp06975_p02525_pair_s1_ep10",
    "adapt_T7_LS0295_g3_cmp07025_p02475_pair_s1_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_pt090nt010_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_pt088nt010_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_pt090nt012_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_pt090nt000_ptasyntasy0.10_0.10_0.10_0.08_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_plw050_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_plw075_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_plw125_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_rawmix_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_ab070_072_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_ab072_070_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_ol002_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_grid6_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_grid10_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_rawmix_s13_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_modebisect_rand_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_modebisect_h_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_modebisect_v_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_modegrid_complete_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p025_pair_area_s7_ep10",
    "adapt_T7_LS030_g3_cmp06975_p02525_pair_s1_ep10",
    "adapt_T7_LS030_g3_cmp07025_p02475_pair_s1_ep10",
    "adapt_T7_LS030_g3_cmp070_p025_pair_s7_ep10",
    "adapt_T7_LS0295_g3_cmp06975_p02475_pair_s1_ep10",
    "adapt_T7_LS0295_g3_cmp07025_p02425_pair_s1_ep10",
    "adapt_T7_LS0295_g3_cmp070_p0245_pair_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p0255_pair_s7_ep10",
    "adapt_T7_LS0295_g3_cmp070_p0245_pair_s13_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p025_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p025_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p025_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p025_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p025_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p025_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p020_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p020_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p020_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p020_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p020_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p020_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p020_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p020_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p035_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p035_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p035_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p035_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p035_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p035_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p035_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p035_pair_s77_ep10",
    "iter116J_T7_g3_cmp062_p020_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp0615_p020_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp0625_p020_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp0635_p020_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp0645_p020_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp065_p020_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp0650_p020_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp060_p018_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp060_p019_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp060_p021_s1_ep10_I10only_partner_push",
    "iter116J_T7_g3_cmp060_p022_s1_ep10_I10only_partner_push",
    "iter116J_T7_g4_cmp060_p020_s1_ep10_I10only_partner_push",
    "iter116J_T7_LS032_g3_cmp060_p020_s1_ep10_I10only_partner_push",
    "iter116J_T7_LS028_g3_cmp060_p020_s1_ep10_I10only_partner_push",
    "fcm_margin_pair_T7_LS0295_g3_cmp0701_p0249_s7_ep10_tail_gap",
    "fcm_margin_pair_T7_LS0295_g3_cmp0699_p0251_s7_ep10_combo_gap",
    "fcm_margin_pair_T7_LS02945_g3_cmp07_p025_s7_ep10_combo_gap",
    "fcm_margin_pair_T7_LS02955_g3_cmp07_p025_s7_ep10_tail_gap",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p0249_s7_ep10_tail_gap",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p0251_s7_ep10_combo_gap",
    # EMA=0.95 underperformed the exact seed=7 champion by ep06
    # (quick bit_F1 0.9792/FAR0 vs champion ep06 quick 0.9883/FAR0).
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_ema095_ep10_stability",
    # Deterministic neighbor cmp=0.6975/p=0.2525 did not follow the champion
    # recovery curve by ep03 (quick bit_F1 0.8446, FAR 5.67%).
    "adapt_T7_LS0295_g3_cmp06975_p02525_pair_det_s7_ep10",
    # Deterministic neighbor cmp=0.7025/p=0.2475 collapsed by ep02
    # (quick bit_F1 0.7168, Total FAR 79.33%).
    "adapt_T7_LS0295_g3_cmp07025_p02475_pair_det_s7_ep10",
    # Deterministic seed=13 exact hparams failed to recover by ep04
    # (quick bit_F1 0.8793, Total FAR 29.00%).
    "adapt_T7_LS0295_g3_cmp070_p025_pair_det_s13_ep10",
    # Deterministic seed=1 cmp=0.6975/p=0.2525 collapsed by ep02
    # (quick bit_F1 0.0000, all-zero prediction region).
    "adapt_T7_LS0295_g3_cmp06975_p02525_pair_det_s1_ep10",
    # Deterministic seed=1 cmp=0.7025/p=0.2475 also collapsed by ep02
    # (quick bit_F1 0.0000, all-zero prediction region).
    "adapt_T7_LS0295_g3_cmp07025_p02475_pair_det_s1_ep10",
    # Warmup+eta-min stability variant leaked heavily through ep03
    # (quick bit_F1 0.6015, Total FAR 29.67%).
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_warm1_eta1e6_ep10_stability",
    # EMA+warmup combined did not recover through ep02
    # (quick bit_F1 0.7891, Total FAR 64.00%).
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_ema095_warm1_ep10_stability",
    # Historical fallback p=0.10/no-pair is far behind the champion curve by ep03
    # (quick bit_F1 0.8149/FAR0 vs champion ep02 0.8940/FAR0).
    "histgrid_all_T7_LS030_g3_cmp070_p010_nopair_s1_ep10",
    # p=0.50/no-pair keeps high F1 but raises negative scratch leakage
    # (s1 full eval 0.9986, Total FAR 0.45%, Normal sc max 0.472).
    "histgrid_all_T7_LS030_g3_cmp070_p050_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p050_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p050_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp070_p050_nopair_s77_ep10",
    # grad-clip 0.75 did not follow the champion/gclip05 recovery curve
    # (quick ep05 bit_F1 0.9008/FAR0).
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_gclip075_ep10_stability",
    "fcm_margin_pair_T7_LS0295_g3_cmp07025_p025_s7_gclip05_ep10_combo_balance",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p02525_s7_gclip05_ep10_combo_balance",
    "fcm_margin_pair_T7_LS0295_g3_cmp07025_p02475_s7_gclip05_ep10_tail_balance",
    "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_gclip05_ep10_tail_balance",
    "fcm_margin_pair_T7_LS02925_g3_cmp07_p025_s7_gclip05_ep10_combo_balance",
    # cmp=0.5/no-pair leaks heavily on the canonical old eval from the start.
    "histgrid_all_T7_LS030_g3_cmp050_p025_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p025_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p025_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p025_nopair_s77_ep10",
    # cmp=0.5/p=0.25 pair also has very slow recall recovery
    # (s42 quick ep03 bit_F1 0.7682/FAR0).
    "histgrid_all_T7_LS030_g3_cmp050_p025_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p025_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p025_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p025_pair_s77_ep10",
    # cmp=0.5/p=0.20 no-pair also leaks hard by ep02.
    "histgrid_all_T7_LS030_g3_cmp050_p020_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p020_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p020_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p020_nopair_s77_ep10",
    # cmp=0.5/p=0.20 pair remains far below the champion curve by ep03.
    "histgrid_all_T7_LS030_g3_cmp050_p020_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p020_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p020_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p020_pair_s77_ep10",
    # cmp=0.5/p=0.35 no-pair is still below the champion recovery curve by ep02.
    "histgrid_all_T7_LS030_g3_cmp050_p035_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p035_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p035_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p035_nopair_s77_ep10",
    # cmp=0.5/p=0.35 pair collapses immediately on old eval.
    "histgrid_all_T7_LS030_g3_cmp050_p035_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p035_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p035_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp050_p035_pair_s77_ep10",
    # cmp=0.6/p=0.25 no-pair leaks hard by ep02 on old eval.
    "histgrid_all_T7_LS030_g3_cmp060_p025_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p025_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p025_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p025_nopair_s77_ep10",
    # cmp=0.6/p=0.25 pair also leaks hard by ep02.
    "histgrid_all_T7_LS030_g3_cmp060_p025_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p025_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p025_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p025_pair_s77_ep10",
    # cmp=0.6/p=0.20 no-pair leaks hard by ep02 as well.
    "histgrid_all_T7_LS030_g3_cmp060_p020_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p020_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p020_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p020_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p020_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p020_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p020_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p020_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p035_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p035_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p035_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p035_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p035_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p035_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p035_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp060_p035_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p025_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p025_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p025_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p025_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p025_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p025_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p025_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p025_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p020_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p020_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p020_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p020_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p020_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p020_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p020_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p020_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p035_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p035_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p035_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p035_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p035_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p035_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p035_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp080_p035_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p025_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p025_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p025_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p025_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p025_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p025_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p025_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p025_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p020_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p020_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p020_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p020_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p020_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p020_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p020_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p020_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p035_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p035_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p035_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p035_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p035_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p035_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p035_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp030_p035_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p025_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p025_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p025_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p025_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p025_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p025_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p025_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p025_pair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p020_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p020_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p020_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p020_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p035_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p035_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p035_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p035_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p035_pair_s1_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p035_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p035_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p035_pair_s77_ep10",
    # g=4/cmp=0.7/p=0.25 no-pair is far behind by ep02
    # (quick bit_F1 0.7624, Total FAR 55.67%).
    "histgrid_all_T7_LS030_g4_cmp070_p025_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p025_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p025_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p025_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p025_pair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p025_pair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p025_pair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p025_pair_s77_ep10",
    # g=4/cmp=0.6/p=0.25 no-pair collapsed/leaked immediately:
    # ep02 quick bit_F1=0.7731, Total FAR=81.67%.
    "histgrid_all_T7_LS030_g4_cmp060_p025_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p025_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p025_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p025_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p025_pair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p025_pair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p025_pair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p025_pair_s77_ep10",
    # Same g=4/cmp=0.6 broad branch with higher p; prior g4/cmp0.6 runs
    # collapsed or lagged far behind champion.
    "histgrid_all_T7_LS030_g4_cmp060_p035_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p035_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p035_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p035_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p035_pair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p035_pair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p035_pair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p035_pair_s77_ep10",
    # g=4/cmp=0.6/p=0.20 no-pair is the same collapse region:
    # ep01 quick bit_F1=0.7453, Total FAR=84.33%.
    "histgrid_all_T7_LS030_g4_cmp060_p020_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p020_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p020_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p020_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p020_pair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p020_pair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p020_pair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp060_p020_pair_s77_ep10",
    # g=4/cmp=0.7/p=0.35 no-pair also recovered too slowly:
    # ep04 quick bit_F1=0.9299/FAR0, still far below the champion curve.
    "histgrid_all_T7_LS030_g4_cmp070_p035_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p035_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p035_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p035_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p035_pair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p035_pair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p035_pair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p035_pair_s77_ep10",
    # Do not fall through into broad g=4/cmp=0.70/p=0.20 seed grids.
    # Existing g4 cmp070 evidence is far below the champion curve.
    "histgrid_all_T7_LS030_g4_cmp070_p020_nopair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p020_nopair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p020_nopair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p020_nopair_s77_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p020_pair_s1_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p020_pair_s42_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p020_pair_s7_ep10",
    "histgrid_all_T7_LS030_g4_cmp070_p020_pair_s77_ep10",
    # No-save broad-grid run found ep04 quick 0.9918/FAR0 but then overwrote
    # best_model at ep05; rerun with save-every below to preserve epoch choice.
    "histgrid_all_T7_LS030_g3_cmp100_p020_pair_s1_ep10",
    # Full eval of the save-every rerun showed OOD scratch tail leakage
    # (eval_F1=0.9951, Total FAR=0.15%, DiagonalSmear FAR=5.5%).
    # Do not spend seed repeats on this exact cmp=1.0/p=0.20 pair region.
    "histgrid_all_T7_LS030_g3_cmp100_p020_pair_s42_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p020_pair_s7_ep10",
    "histgrid_all_T7_LS030_g3_cmp100_p020_pair_s77_ep10",
    # Per-bit negative target lowering destabilized calibration on quick eval:
    # sr010 reached bit_F1=0.9994 but FAR=77.67% by epoch 6.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_negasy_sr010_ep10_tail_gap",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_negasy_sc010sr010_ep10_tail_gap",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_maskneg010_ep10_tail_gap",
    # bank_boundary+scratch pair-bias recovered too slowly by ep06
    # (quick bit_F1=0.8869/FAR0, sr remains weak).
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_bank_scratch2_ep10_combo_gap",
    # fork+scratch bias k=1 still leaked heavily once F1 recovered
    # (quick ep07 bit_F1=0.9879, Total FAR=12.33%).
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch1_ep10_far_guard",
    # Lowering p with fork+scratch bias did not fix the leakage/recovery tradeoff:
    # ep05 bit_F1=0.9401/FAR0, ep06 OOD FAR=46.00%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p0245_s7_pairbias_fork_scratch2_ep10_far_guard",
    # Mild fork/scratch pos_weight was too slow and still leaked Invalid by ep07
    # (quick bit_F1=0.9350, Total FAR=2.00%).
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_posw_fork103_scratch103_ep10_combo_gap",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_posw_fork105_scratch105_ep10_combo_gap",
    # Direct scratch+scratch_rot pair bias destroyed negative calibration:
    # epoch10 quick eval bit_F1=0.8071, Total FAR=66.67%, Normal FAR=92%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_scratch_scratchrot2_ep10_sr_combo_gap",
    # bank_boundary+scratch_rot is not the champion bottleneck (min_pos already ~0.658);
    # skip direct pair-bias after scratch/scratch_rot destabilized negatives.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_bank_scratchrot2_ep10_sr_combo_gap",
    # fork+scratch pair-bias plus weaker pair-loss stayed far behind champion:
    # ep05 quick bit_F1=0.9589/FAR0, ep06 bit_F1=0.9571/Total FAR=0.33%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_pairloss075_ep10_far_guard",
    # gclip05 with cmp down / p up did not recover: ep04 bit_F1=0.9402/FAR0.33,
    # ep05 bit_F1=0.8979/Total FAR=23.00%.
    "fcm_margin_pair_T7_LS0295_g3_cmp06975_p02525_s7_gclip05_ep10_combo_balance",
    # fork/scratch pair-bias plus gclip05 still leaked badly:
    # ep05 quick bit_F1=0.9489, Total FAR=38.00%, OOD FAR=55.00%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_gclip05_ep10_far_guard",
    # Even very mild fork/scratch pos_weight plus gclip05 leaked Invalid/NI early:
    # ep03 quick bit_F1=0.8378, Total FAR=14.67%, NI FAR=35.00%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_posw_fork101_scratch101_gclip05_ep10_combo_guard",
    # More aggressive fork+scratch pair-bias did not improve the champion tail:
    # ep04-ep07 quick eval stayed leaky (ep07 bit_F1=0.9586, Total FAR=15.67%).
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch3_ep10_combo_gap",
    # BCE temperature above 1 destabilized negative calibration:
    # bceT105 ep03 quick bit_F1=0.9277, Total FAR=46.33%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_bceT105_ep10_gap",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_bceT110_ep10_gap",
    # Seed 23 collapsed to all-negative on the champion basin:
    # ep02 quick bit_F1=0.0000, Total FAR=0.00%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s23_ep10_seedcheck",
    # BCE temperature below 1 also failed to recover:
    # bceT095 ep03 quick bit_F1=0.8496, Total FAR=6.67%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_bceT095_ep10_invalid_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_bceT095_ep10_combo_guard",
    # Stronger pair counterfactual loss alone was too slow and leaked Invalid:
    # ep03 quick bit_F1=0.9191, Total FAR=1.33%, Invalid FAR=8%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss125_ep10_invalid_guard",
    # Mild drop-path became too conservative:
    # ep03 quick bit_F1=0.8399, Total FAR=0.00%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_dpr002_ep10_tail_guard",
    # masked2 with weaker pair loss leaked Invalid and stayed low-F1:
    # ep02 quick bit_F1=0.8029, Total FAR=1.33%, Invalid FAR=8%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_masked2_w050_ep10_tail_guard",
    # This run was accidentally stripped of --cutmix-mask-pos-target by the
    # sweep sanitizer and duplicated plain pairloss125 evidence.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss125_maskpos070_ep10_tail_guard",
    # Applying maskpos070 produced the same early curve as plain pairloss125:
    # ep02 quick bit_F1=0.8940, Total FAR=0.00%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss125_maskpos070_apply_ep10_tail_guard",
    # raw mix + ab0.72 combo push broke negative calibration:
    # ep02 quick bit_F1=0.8823, Total FAR=10.67%, NI FAR=30%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_rawmix_ab072072_pairloss125_ep10_combo_guard",
    # pairbias fork/scratch2 + cmp0.7025 recovered too slowly and then regressed:
    # ep03 quick bit_F1=0.9313/FAR=0.67%, ep04 quick bit_F1=0.9056/FAR=1.00%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07025_p025_s7_pairbias_fork_scratch2_ep10_combo_lift",
    # pairbias fork/scratch2 + p0.2525 stayed below the recovery gate:
    # ep03 quick bit_F1=0.8891/FAR=0.00%.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p02525_s7_pairbias_fork_scratch2_ep10_combo_lift",
    # The separate cmp and p lifts both underperformed, so skip the combined lift.
    # Mask pos target above the current cmp basin over-activated negatives:
    # mpos072 ep04 quick bit_F1=0.9631, Total FAR=9.67%.
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos072_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos075_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mneg008_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos072_mneg008_s7_ep10",
    # mpos065 finished FAR0 but reduced the gap and hurt bank_boundary+scratch:
    # full eval bit_F1=0.9993, gap=+0.193 vs champion +0.201.  mpos0675 sits
    # between that and over-positive mpos072, so keep compute on gap-preserving axes.
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos0675_s7_ep10",
    "fcm_margin_pair_T7_LS0295_g3_cmp07025_p02525_s7_pairbias_fork_scratch2_ep10_combo_lift",
    # New-SOTA pairbias basin, but cmp down + p up recovered too slowly:
    # ep03 quick bit_F1=0.9071/FAR=0.00%.
    "adapt_T7_LS02975_g3_cmp06975_p02525_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA pairbias basin, but cmp up + p down also lagged:
    # ep03 quick bit_F1=0.8752/FAR=0.00%.
    "adapt_T7_LS02975_g3_cmp07025_p02475_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA pairbias basin, cmp down alone leaked and lagged:
    # ep03 quick bit_F1=0.8738/FAR=4.00%.
    "adapt_T7_LS02975_g3_cmp06975_p025_pair_pbfork_scratchx2_s7_ep10",
    # LS lift back toward 0.30 with cmp/p lifted lost the new-SOTA gap:
    # ep03 quick bit_F1=0.9328/FAR=1.67%, ep06 quick bit_F1=0.9219/FAR=0.00%.
    "adapt_T7_LS029875_g3_cmp070_p025_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, cmp up + p down leaked OOD heavily:
    # ep05 quick bit_F1=0.9713, Total FAR=21.00%, OOD FAR=31.50%.
    "adapt_T7_LS02975_g3_cmp070_p0245_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, cmp down + p up stayed far below the recovery curve:
    # ep04-ep07 quick bit_F1=0.945-0.955 with FAR=0.00%.
    "adapt_T7_LS02975_g3_cmp0695_p025_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, cmp down + p down stayed stable but far below SOTA:
    # ep09 quick bit_F1=0.9816, Total FAR=0.00%.
    "adapt_T7_LS02975_g3_cmp0695_p02475_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, p lowered further slowed recall and kept Starburst leak:
    # ep05 quick bit_F1=0.9255, Total FAR=0.67%, Starburst FAR=4%.
    "adapt_T7_LS02975_g3_cmp06975_p0245_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, cmp and p lowered together did not recover and leaked OOD:
    # ep05 quick bit_F1=0.9013, Total FAR=6.33%, OOD FAR=9.50%.
    "adapt_T7_LS02975_g3_cmp0695_p0245_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, tiny cmp lift destabilized negative calibration:
    # ep03 quick bit_F1=0.9217, Total FAR=81.33%; ep04 Total FAR=20.33%.
    "adapt_T7_LS02975_g3_cmp069875_p02475_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, tiny p lift reached FAR0 but stayed below champion:
    # ep09 full eval bit_F1=0.9996, Total FAR=0.00%.
    "adapt_T7_LS02975_g3_cmp06975_p024875_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, tiny cmp+p lift followed the cmp-lift FAR explosion:
    # ep03 quick bit_F1=0.9217, Total FAR=81.33%; ep04 Total FAR=20.33%.
    "adapt_T7_LS02975_g3_cmp069875_p024875_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, lower cmp destabilized negatives after a brief FAR0 point:
    # ep05 quick bit_F1=0.9409/FAR=37.00%, ep06 quick FAR=67.67%.
    "adapt_T7_LS02975_g3_cmp069625_p02475_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, LS lift alone leaked OOD/Normal after ep03:
    # ep04 quick bit_F1=0.9557/FAR=16.67%, ep05 quick FAR=17.67%.
    "adapt_T7_LS029875_g3_cmp06975_p02475_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, p lowered 0.00125 stayed FAR0 but recovered too slowly:
    # ep05 quick bit_F1=0.9468, Total FAR=0.00%.
    "adapt_T7_LS02975_g3_cmp06975_p024625_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, cmp+p lowered gave unstable FAR and filled disk on save:
    # ep06 quick bit_F1=0.9981/FAR=1.00%, ep08 quick FAR=7.00%.
    "adapt_T7_LS02975_g3_cmp069625_p024625_pair_pbfork_scratchx2_s7_ep10",
    # New-SOTA basin, LS up + cmp down leaked OOD immediately:
    # ep02 quick bit_F1=0.7759, Total FAR=44.67%, OOD FAR=67.00%.
    "adapt_T7_LS029875_g3_cmp0695_p02475_pair_pbfork_scratchx2_s7_ep10",
    # Stale scalar nudges around the old fcm_margin/pairbias ridge are
    # dominated by the 0.2975/0.6975/0.2475 champion and nearby failures.
    "adapt_T7_LS02975_g3_cmp070125_p025_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp070_p025125_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp070125_p025125_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp069875_p025_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp070_p024875_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS029875_g3_cmp06975_p025_pair_pbfork_scratchx2_s7_ep10",
    # fork-only pos weight did not lift the bottleneck without hurting scratch:
    # ep10 quick bit_F1=0.9863, Total FAR=0.33%, sc_F1=0.953.
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_pwforkx1p01_s7_ep10",
    # scratch_rot-only pos weight destroyed negative separation:
    # ep02 quick bit_F1=0.7748, Total FAR=82.67%, OOD FAR=99.50%.
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_pwscratch_rotx1p01_s7_ep10",
    # tiny fork+scratch_rot pos weights still over-fired negatives:
    # ep02 quick bit_F1=0.8331, Total FAR=80.33%, OOD FAR=93.00%.
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_pwforkx1p005_scratch_rotx1p005_s7_ep10",
    # Direct fork+scratch_rot pair exposure also over-fired negatives:
    # ep02 quick bit_F1=0.8162, Total FAR=86.67%, OOD FAR=93.50%.
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratch_rotx2_s7_ep10",
    # Current-top3 KD with weak KD did not beat champion and leaked Starburst:
    # best quick ep08 bit_F1=0.9909, Total FAR=0.33%, OOD FAR=0.50%.
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_kd_teacher_probs_current_top3_iter116J_orig814_260529_kda080_kdt200_kdskipcm_s7_ep10",
    # Current-top3 KD with stronger KD under-recovered recall and leaked OOD:
    # ep03 quick bit_F1=0.8971, Total FAR=11.67%, OOD FAR=17.00%.
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_kd_teacher_probs_current_top3_iter116J_orig814_260529_kda030_kdt200_kdskipcm_s7_ep10",
}


FACTORIAL_PRUNED_TAGS = {
    # 260531 controlled factorial evidence on frozen_iter116J_orig814_v15direct_n2000:
    # g=2 with 3x3 grid over-fragmented FCMPM geometry; eval bit_F1=0.5393,
    # Total FAR=47.41%, Invalid FAR=100%, Starburst FAR=85.8%.
    "fcmpm_ablate_T7_LS02975_g2_grid3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # g=2 with 6x6 grid was FAR-safe but under-recalled combo POS; eval
    # bit_F1=0.8892, sc_F1=0.7693, sr_F1=0.8410, global_gap=-0.186.
    "fcmpm_ablate_T7_LS02975_g2_grid6_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # g=2 with 8x8 grid still under-recalled combo POS and leaked OOD sr tail;
    # eval bit_F1=0.9176, sr_F1=0.7797, Total FAR=0.32%, global_gap=-0.127.
    "fcmpm_ablate_T7_LS02975_g2_grid8_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
}


def is_live_pruned_tag(tag: str) -> bool:
    if tag in FACTORIAL_PRUNED_TAGS:
        return True
    # 260531 user request: re-open controlled FCMPM group/grid/p factorial
    # despite earlier quick-prune evidence, because g=2/3/4 and grid density
    # must be measured as an explicit ablation axis.
    if tag.startswith("fcmpm_ablate_") or tag.startswith("fcmpm_pggrid_"):
        return False
    if tag in LIVE_PRUNED_TAGS:
        return True
    # 260601 target-label matrix on frozen_iter116J_orig814_eval_n20000:
    # weak100/strong100/neg020 collapsed the weak combo POS
    # (bank_boundary+scratch/sc=0.470), eval bit_F1=0.9798, FAR=2.91.
    # Strong soft-negative labels are therefore outside the useful band.
    # neg=0.028 was worse than both neg=0.02 and neg=0.03 through ep04:
    # quick FAR 15.67/17.33/13.67/19.67 with Invalid 82-100% and OOD opening
    # at ep04, so stop spending full-eval time on this interpolation point.
    if tag == "targetlabel_refine_weak100_strong100_neg0028_T7_LS02950_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000":
        return True
    # neg=0.032 crossed the stable band in the opposite direction: ep01-ep03
    # quick FAR 66/96.67/98.33 with Normal/OOD broadly positive, so it is not
    # a useful neighbor of the neg=0.03 best.
    if tag == "targetlabel_refine_weak100_strong100_neg0032_T7_LS02950_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000":
        return True
    # p=0.60/neg=0.03 suppressed FAR but under-recalled POS; ep08 quick was
    # only bit_F1=0.9795 with FAR=0.33, so skip expensive full eval and move
    # to lower-p neighbors.
    if tag == "targetlabel_refine_weak100_strong100_neg003_T7_LS02950_g3_grid9_cmp10000_p06000_mpos065_s7_ep10_tr200_ev02000":
        return True
    # p=0.60/neg=0.02 also failed the best basin: quick ep01/02 leaked heavily
    # (FAR 95/80%), and ep03 only recovered to bit_F1=0.9570 with FAR=1.33.
    if tag == "targetlabel_refine_weak100_strong100_neg002_T7_LS02950_g3_grid9_cmp10000_p06000_mpos065_s7_ep10_tr200_ev02000":
        return True
    # p=0.55/neg=0.02 did not enter the best basin: quick ep03 reached
    # FAR=0 but only bit_F1=0.9825, then ep04 rose to 0.9892 while FAR
    # reopened to 1.67%; ep05/06 stayed below target.
    if tag == "targetlabel_refine_weak100_strong100_neg002_T7_LS02950_g3_grid9_cmp10000_p05500_mpos065_s7_ep10_tr200_ev02000":
        return True
    # p=0.45/neg=0.02 did not suppress the tail: quick ep04 reached bit_F1
    # 0.9899 but FAR stayed 11.33%, mainly NI leakage.
    if tag == "targetlabel_refine_weak100_strong100_neg002_T7_LS02950_g3_grid9_cmp10000_p04500_mpos065_s7_ep10_tr200_ev02000":
        return True
    # p=0.45/neg=0.03 recovered only briefly (ep04 0.9812/FAR4.67) and then
    # fell back to 0.9594/FAR6.33 by ep06, below the p=0.50 basin.
    if tag == "targetlabel_refine_weak100_strong100_neg003_T7_LS02950_g3_grid9_cmp10000_p04500_mpos065_s7_ep10_tr200_ev02000":
        return True
    # p=0.40/neg=0.02 stayed under-recalled: quick ep03/04 were
    # bit_F1=0.9817/0.9724 with FAR=0.67/0.33, below the p=0.50 basin.
    if tag == "targetlabel_refine_weak100_strong100_neg002_T7_LS02950_g3_grid9_cmp10000_p04000_mpos065_s7_ep10_tr200_ev02000":
        return True
    # p=0.40/neg=0.03 is FAR-safe but under-recovers POS: quick ep06/08 were
    # 0.9859/0.9831 with FAR=0.00, below the target p=0.50 basin.
    if tag == "targetlabel_refine_weak100_strong100_neg003_T7_LS02950_g3_grid9_cmp10000_p04000_mpos065_s7_ep10_tr200_ev02000":
        return True
    # p=0.35/neg=0.02 briefly reached ep03 0.9891/FAR0 but then became
    # unstable (ep06 0.8564/FAR85, ep07 0.9844/FAR20), matching the known
    # p=0.35 instability.
    if tag == "targetlabel_refine_weak100_strong100_neg002_T7_LS02950_g3_grid9_cmp10000_p03500_mpos065_s7_ep10_tr200_ev02000":
        return True
    # p=0.35/neg=0.03 was unstable: ep03 briefly reached 0.9752/FAR0 but ep04
    # reopened OOD/Normal tails to FAR=82.67, so it is not a stable basin.
    if tag == "targetlabel_refine_weak100_strong100_neg003_T7_LS02950_g3_grid9_cmp10000_p03500_mpos065_s7_ep10_tr200_ev02000":
        return True
    # p=0.30/neg=0.02 raised F1 late but did not suppress NI: quick ep04 was
    # bit_F1=0.9943 with FAR=16%, worse than p=0.30/neg=0.03 and p=0.50.
    if tag == "targetlabel_refine_weak100_strong100_neg002_T7_LS02950_g3_grid9_cmp10000_p03000_mpos065_s7_ep10_tr200_ev02000":
        return True
    # strong=0.95/neg=0.01 at p=0.50 under-recovers POS early and remains
    # unstable: quick ep04 reached bit_F1=0.9924 but FAR=1.67, then ep05
    # dropped to 0.9482/FAR2.0.
    if tag == "targetlabel_refine_weak100_strong095_neg001_T7_LS02950_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000":
        return True
    # strong=0.95/neg=0.02 at p=0.50 was worse: quick ep01/02 leaked heavily
    # (FAR 91.67/100), and ep03 still had only bit_F1=0.9878 with FAR=10.
    if tag == "targetlabel_refine_weak100_strong095_neg002_T7_LS02950_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000":
        return True
    # strong=0.95/neg=0.03 also missed the p=0.50 best basin: quick ep03 was
    # bit_F1=0.9878 with FAR=2.33, so softening the strong POS is not helping.
    if tag == "targetlabel_refine_weak100_strong095_neg003_T7_LS02950_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000":
        return True
    # strong=0.90/neg=0.02 collapsed the negative side after disk cleanup:
    # quick ep10 was only bit_F1=0.8992 with FAR=92.67% (NI=94/OOD=92).
    # It is far outside the target-label basin, so skip full eval/pcls.
    if tag == "targetlabel_refine_weak100_strong090_neg002_T7_LS02950_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000":
        return True
    if tag.startswith("targetlabel_") and "_neg020_" in tag:
        return True
    # Adaptive broad g=2/g=4 LS=0.30 grids repeatedly lagged or leaked on this
    # frozen old-eval dataset; keep compute on the proven g=3/fcm_margin basin.
    if re.fullmatch(
        r"histgrid_all_T7_LS030_g(2|4)_cmp(030|050|060|070|080|100)_p(020|025|035)_(no)?pair_s(1|7|42|77)_ep10",
        tag,
    ):
        return True
    # Broad LS sweeps away from the LS=0.295/0.30 g=3 basin have repeatedly
    # leaked FAR or collapsed recall; keep fallback compute on the proven basin.
    if re.fullmatch(
        r"histgrid_all_T7_LS(020|025|035|040|045|050)_g(2|3|4)_cmp(030|050|060|070|080|100)_p(020|025|035)_(no)?pair_s(1|7|42|77)_ep10",
        tag,
    ):
        return True
    if re.fullmatch(
        r"fcmpm_pggrid_T7_LS02975_g2_grid(3|6|8|9)_cmp06975_p(02000|02250|02750|03000)_pair_pbfork_scratchx2_mpos065_s7_ep10",
        tag,
    ):
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid3_cmp06975_p02000_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid3_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid6_cmp06975_p02000_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid6_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if re.fullmatch(
        r"fcmpm_pggrid_T7_LS02975_g3_grid(8|9)_cmp06975_p02000_pair_pbfork_scratchx2_mpos065_s7_ep10",
        tag,
    ):
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid8_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid9_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid3_cmp06975_p02750_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid6_cmp06975_p02750_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid8_cmp06975_p02750_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid9_cmp06975_p02750_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if re.fullmatch(
        r"fcmpm_pggrid_T7_LS02975_g3_grid(3|6|8|9)_cmp06975_p03000_pair_pbfork_scratchx2_mpos065_s7_ep10",
        tag,
    ):
        return True
    if re.fullmatch(
        r"fcmpm_pggrid_T7_LS02975_g4_grid(3|6|8|9)_cmp06975_p(02000|02250|02750|03000)_pair_pbfork_scratchx2_mpos065_s7_ep10",
        tag,
    ):
        return True
    return False


TRANSFER_PRUNE_DATASETS = {
    "frozen_iter116J_orig814_v15direct_n2000",
    "frozen_original",
    "sota_gapstress_seed31_260531",
    "sota_gapstress_seed97_260531",
}


TRANSFER_PRUNED_TAGS = {
    # v15direct_n2000 high-p matrix: p=0.30/cmp=0.50/g3/grid8 failed to
    # recover POS recall and repeatedly leaked tails (ep01 FAR 58.33%,
    # ep03 FAR 74.67%, ep06 FAR 54.67%; best quick bit_F1 only 0.7941).
    "fcmpm_hpmat_T7_LS02975_g3_grid8_cmp05000_p03000_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 area-prop first slice leaked immediately
    # (ep01 FAR 70.67%, ep02 FAR 51.67%).
    "fcmpm_hpmat_areaprop_T7_LS02975_g2_grid8_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 explicit A/B+raw target: g2/grid3/ab070_050 recovered
    # low FAR at ep01 but immediately leaked OOD by ep02
    # (bit_F1 0.7181, FAR 56.00%, OOD 82.50%).
    "fcmpm_hpmat_ab070_050_ol00_raw_T7_LS02975_g2_grid3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 explicit A/B+raw target on champion geometry:
    # g3/grid8/ab070_050/other0 improved recall by ep03 but leaked OOD
    # (bit_F1 0.9158, FAR 7.00%, OOD 10.50%).
    "fcmpm_hpmat_ab070_050_ol00_raw_T7_LS02975_g3_grid8_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 explicit A/B+raw target with soft off-class labels:
    # g3/grid8/ab070_050/other0.10 kept Normal lower at ep01 but still failed
    # recall and leaked by ep02 (bit_F1 0.7446, FAR 10.00%, OOD 13.50%).
    "fcmpm_hpmat_ab070_050_ol010_raw_T7_LS02975_g3_grid8_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 reverse asymmetric A/B on champion geometry:
    # g3/grid8/ab050_070/other0 held FAR=0 through ep02, but when POS recall
    # started to recover at ep03 it leaked badly (bit_F1 0.8615, FAR 27.67%,
    # OOD 39.50%).
    "fcmpm_hpmat_ab050_070_ol00_raw_T7_LS02975_g3_grid8_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 reverse asymmetric A/B with soft off-class labels:
    # g3/grid8/ab050_070/other0.10 leaked OOD immediately at ep01
    # (bit_F1 0.7767, FAR 69.33%, OOD 99.50%).
    "fcmpm_hpmat_ab050_070_ol010_raw_T7_LS02975_g3_grid8_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 symmetric explicit raw target:
    # g3/grid8/ab070_070/other0 raised POS by ep03 but leaked OOD massively
    # (ep02 FAR 78.67% OOD 99.50%; ep03 bit_F1 0.9109 FAR 67.33% OOD 99.50%).
    "fcmpm_hpmat_ab070_070_ol00_raw_T7_LS02975_g3_grid8_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 symmetric explicit raw target with soft off-class labels:
    # g3/grid8/ab070_070/other0.10 leaked immediately at ep01
    # (bit_F1 0.7367, FAR 73.00%, OOD 98.50%).
    "fcmpm_hpmat_ab070_070_ol010_raw_T7_LS02975_g3_grid8_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 explicit A/B through normal BCE+LS path:
    # g3/grid8/ab070_050/other0 stayed low-recall and began leaking NI by ep03
    # (bit_F1 0.7710, FAR 5.33%, NI 16.00%).
    "fcmpm_hpmat_ab070_050_ol00_T7_LS02975_g3_grid8_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: FCMPM p=0.0500 recovered FAR by ep4
    # but bit_F1 stayed 0.8553, far below the current p=0.2475 basin.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p00500_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.1000 had ep02 FAR=92.33%.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p01000_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.1500 stayed near bit_F1=0.80
    # with 32-59% FAR through ep03.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p01500_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.1750 collapsed to all-zero by ep02.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p01750_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2000 was FAR-safe but only
    # bit_F1=0.8421 by ep04; POS combo pressure is still too weak.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02000_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2250 reached only bit_F1=0.8993
    # and later leaked OOD FAR=39.00% by ep04.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2350 was FAR-safe but only
    # bit_F1=0.9154 by ep04, still below the current basin.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02350_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2400 briefly leaked FAR=88.00%
    # and only recovered to bit_F1=0.8693/FAR=0 by ep04.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02400_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2425 reached bit_F1=0.9651
    # but still had FAR=3.00% and later OOD FAR=45.00%.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02425_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2450 pushed POS but tail leaked,
    # ep05 bit_F1=0.9677 with FAR=41.33% and Normal FAR=78.00%.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02450_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2500 was FAR-safe at ep04 but
    # only bit_F1=0.9391, below the p=0.2475 basin trajectory.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02500_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2525 was FAR-safe but only
    # bit_F1=0.8738 by ep03; POS combo pressure did not recover.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02525_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2550 leaked FAR=72.00% at ep02
    # and only recovered to bit_F1=0.9385/FAR=0 by ep04.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02550_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2600 alternated tail leak and
    # weak POS, ending ep04 bit_F1=0.8872/FAR=0.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02600_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2650 was FAR-safe at ep04 but
    # only bit_F1=0.9337, still under the p=0.2475 basin.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02650_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.2750 over-activated negatives
    # at ep02 (FAR=81.00%) and then fell to bit_F1=0.8668 by ep03.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p02750_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.3000 hit FAR=100.00% at ep01
    # and only recovered to bit_F1=0.8611/FAR=1.00% by ep02.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p03000_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.3500 leaked OOD FAR=79.00%
    # at ep02 and only recovered to bit_F1=0.8951 by ep03.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p03500_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # v15direct_n2000 quick evidence: p=0.4000 fixed current condition
    # recovered FAR but only bit_F1=0.8852 by ep02.
    "fcmpm_paxis_T7_LS02975_g3_grid8_cmp06975_p04000_pair_pbfork_scratchx2_mpos065_s7_ep10",
    # Old-eval evidence: over-aggressive margin/gap knobs leaked or collapsed.
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos072_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos075_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mneg008_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos072_mneg008_s7_ep10",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_bceT095_ep10_invalid_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_bceT105_ep10_gap",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_bceT110_ep10_gap",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss125_ep10_invalid_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_dpr002_ep10_tail_guard",
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_masked2_w050_ep10_tail_guard",
    # Transfer sidecar: bank_boundary+scratch direct pair-bias recovered too
    # slowly on old-eval; do not replay it on v15/gapstress before stronger
    # grid/mpos/gclip basins finish.
    "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_bank_scratch2_ep10_combo_gap",
    # iter116J seed repeats preserve the negative gap problem rather than
    # solving the weak combo POS / Normal-scratch tail split.
    "iter116J_exact_T7_LS030_g3_cmp05_p025_s7_ep10",
    "iter116J_exact_T7_LS030_g3_cmp05_p025_s42_ep10",
    "iter116J_exact_T7_LS030_g3_cmp05_p025_s77_ep10",
    # Old-eval scalar nudges around the champion basin either leaked OOD/Normal
    # or recovered too slowly; do not spend transfer compute on them before the
    # stronger pairloss/gclip/deterministic basins are checked.
    "adapt_T7_LS02975_g3_cmp06975_p02525_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp07025_p02475_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p025_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS029875_g3_cmp070_p025_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp070_p0245_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp0695_p025_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp0695_p02475_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p0245_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp0695_p0245_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp069875_p02475_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p024875_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp069875_p024875_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp069625_p02475_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS029875_g3_cmp06975_p02475_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p024625_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp069625_p024625_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS029875_g3_cmp0695_p02475_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp070125_p025_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp070_p025125_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp070125_p025125_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp069875_p025_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS02975_g3_cmp070_p024875_pair_pbfork_scratchx2_s7_ep10",
    "adapt_T7_LS029875_g3_cmp06975_p025_pair_pbfork_scratchx2_s7_ep10",
    # v15direct_n2000 evidence: grid3/grid6 lowered combo min_pos and raised
    # OOD sr/sc tails versus the mpos065 champion; do not repeat exact grids on
    # transfer datasets before cmp-down variants prove they repair the tail.
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid3_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid6_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid9_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06875_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
    "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    "fcmpm_ablate_T7_LS02975_g2_grid3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    "fcmpm_ablate_T7_LS02975_g2_grid6_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    "fcmpm_ablate_T7_LS02975_g2_grid9_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    "fcmpm_ablate_T7_LS02975_g4_grid3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    "fcmpm_ablate_T7_LS02975_g4_grid6_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
    "fcmpm_ablate_T7_LS02975_g4_grid9_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
}


def is_transfer_pruned_tag(tag: str) -> bool:
    if tag in FACTORIAL_PRUNED_TAGS:
        return True
    # 260531 user request: re-open controlled FCMPM group/grid/p factorial
    # on transfer datasets too, instead of inheriting stale prune decisions.
    if tag.startswith("fcmpm_ablate_") or tag.startswith("fcmpm_pggrid_"):
        return False
    if tag in TRANSFER_PRUNED_TAGS:
        return True
    # Broad historical grids are fallback coverage, not current SOTA basin.
    # On transfer datasets keep compute on the proven LS≈0.2975/g3/cmp≈0.70
    # pair-bias basin unless adaptive evidence opens a new branch.
    if re.fullmatch(
        r"histgrid_all_T7_LS030_g3_cmp070_p050_(no)?pair_s(1|7|42|77)_ep10",
        tag,
    ):
        return True
    if re.fullmatch(
        r"histgrid_(cap200|oldest200|newest200)_T7_LS[0-9]+_g[0-9]+_cmp[0-9]+_p[0-9]+_(no)?pair_s(1|7|42|77)_ep10",
        tag,
    ):
        return True
    if re.fullmatch(
        r"histgrid_all_T7_LS030_g(3|4)_cmp(050|060)_p(020|025|035)_(no)?pair_s(1|7|42|77)_ep10",
        tag,
    ):
        return True
    if re.fullmatch(
        r"histgrid_all_T7_LS(020|025|035|040|045|050)_g(2|3|4)_cmp(030|050|060|070|080|100)_p(020|025|035)_(no)?pair_s(1|7|42|77)_ep10",
        tag,
    ):
        return True
    if re.fullmatch(
        r"fcmpm_pggrid_T7_LS02975_g2_grid(3|6|8|9)_cmp06975_p(02000|02250|02750|03000)_pair_pbfork_scratchx2_mpos065_s7_ep10",
        tag,
    ):
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid3_cmp06975_p02000_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid3_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid6_cmp06975_p02000_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid6_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if re.fullmatch(
        r"fcmpm_pggrid_T7_LS02975_g3_grid(8|9)_cmp06975_p02000_pair_pbfork_scratchx2_mpos065_s7_ep10",
        tag,
    ):
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid8_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid9_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid3_cmp06975_p02750_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid6_cmp06975_p02750_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid8_cmp06975_p02750_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if tag == "fcmpm_pggrid_T7_LS02975_g3_grid9_cmp06975_p02750_pair_pbfork_scratchx2_mpos065_s7_ep10":
        return True
    if re.fullmatch(
        r"fcmpm_pggrid_T7_LS02975_g3_grid(3|6|8|9)_cmp06975_p03000_pair_pbfork_scratchx2_mpos065_s7_ep10",
        tag,
    ):
        return True
    if re.fullmatch(
        r"fcmpm_pggrid_T7_LS02975_g4_grid(3|6|8|9)_cmp06975_p(02000|02250|02750|03000)_pair_pbfork_scratchx2_mpos065_s7_ep10",
        tag,
    ):
        return True
    return False


def is_live_pruned_for_dataset(ds_name: str, tag: str) -> bool:
    """Apply live-prune evidence only to the dataset where it was observed."""
    if ds_name in {"frozen_iter116J_orig814_old_eval", "frozen_iter116J_orig814_eval_n20000"}:
        return is_live_pruned_tag(tag)
    if ds_name in TRANSFER_PRUNE_DATASETS:
        return is_transfer_pruned_tag(tag)
    return False


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train: str
    eval: str


@dataclass(frozen=True)
class Recipe:
    tag: str
    variant: str
    ls: str
    groups: str
    cmp_ls: str
    cutmix_p: str
    seed: str
    extra: tuple[str, ...] = field(default_factory=tuple)
    epochs: str = "10"
    batch: str = "2"
    accum: str = "8"


DATASETS = {
    "frozen_iter116J_orig814_v15direct_n2000": DatasetSpec(
        name="frozen_iter116J_orig814_v15direct_n2000",
        train="E:/data/images/classification_chips_iter116J_orig814_260529",
        # Actual eval root recorded by the historical iter116J artifact.
        eval="E:/data/images/chip_multilabel_v15direct_n2000",
    ),
    "frozen_iter116J_orig814_eval_n20000": DatasetSpec(
        name="frozen_iter116J_orig814_eval_n20000",
        train="E:/data/images/classification_chips_iter116J_orig814_260529",
        eval="E:/data/images/eval_n20000",
    ),
    "frozen_iter116J_orig814_old_eval": DatasetSpec(
        name="frozen_iter116J_orig814_old_eval",
        train="E:/data/images/classification_chips_iter116J_orig814_260529",
        eval="E:/data/images/chip_multilabel_v15direct_iter116J_old_n2000_ood200_inv500_260529",
    ),
    "frozen_original": DatasetSpec(
        name="frozen_original",
        train="E:/data/images/classification_chips",
        # Historical frozen SOTA/eval replay root from iter116J exact-repro logs.
        # chip_multilabel_v15direct is the newer expanded 24-class set and is
        # not comparable to the frozen SOTA records.
        eval="E:/data/images/chip_multilabel_v15direct_n2000",
    ),
    "frozen_original_200_snapshot": DatasetSpec(
        name="frozen_original_200_snapshot",
        train="E:/data/images/classification_chips_200_snapshot_260529",
        eval="E:/data/images/chip_multilabel_v15direct_n2000",
    ),
    "frozen_original_2015_candidate": DatasetSpec(
        name="frozen_original_2015_candidate",
        train="E:/data/images/classification_chips_historical_2015_candidate_260529",
        eval="E:/data/images/chip_multilabel_v15direct_n2000",
    ),
    "sota_clean_260528": DatasetSpec(
        name="sota_clean_260528",
        train="E:/data/images/sota_clean_260528/classification_chips",
        eval="E:/data/images/sota_clean_260528/eval_set",
    ),
    "sota_gapstress_seed31_260531": DatasetSpec(
        name="sota_gapstress_seed31_260531",
        train="E:/data/images/sota_gapstress_seed31_260531/classification_chips",
        eval="E:/data/images/sota_gapstress_seed31_260531/eval_set",
    ),
    "sota_gapstress_seed97_260531": DatasetSpec(
        name="sota_gapstress_seed97_260531",
        train="E:/data/images/sota_gapstress_seed97_260531/classification_chips",
        eval="E:/data/images/sota_gapstress_seed97_260531/eval_set",
    ),
}


def recipes() -> list[Recipe]:
    return [
        # Past single-model SOTA regions and their strongest recorded neighbors.
        Recipe(
            "iter116J_exact_T7_LS030_g3_cmp05_p025_s1_ep10_saveevery_retry2",
            "T7",
            "0.30",
            "3",
            "0.5",
            "0.25",
            "1",
            ("--sweep-save-every-epoch",),
        ),
        Recipe(
            "iter116J_may22_T7_LS030_g3_cmp05_p025_s1_ep10_weightsdir_nograd",
            "T7",
            "0.30",
            "3",
            "0.5",
            "0.25",
            "1",
            (
                "--sweep-no-grad-checkpointing",
                "--sweep-weights",
                "weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth",
                "--sweep-save-every-epoch",
            ),
        ),
        Recipe(
            "iter116J_T7_g3_cmp055_p020_s1_ep10_I10only_partner_push",
            "T7",
            "0.30",
            "3",
            "0.55",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp060_p020_s1_ep10_I10only_partner_push",
            "T7",
            "0.30",
            "3",
            "0.60",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp0595_p020_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.595",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp0605_p020_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.605",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp061_p020_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.61",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp0615_p020_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.615",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp062_p020_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.62",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp060_p018_s1_ep10_I10only_partner_push",
            "T7",
            "0.30",
            "3",
            "0.60",
            "0.18",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp065_p018_s1_ep10_I10only_partner_push",
            "T7",
            "0.30",
            "3",
            "0.65",
            "0.18",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp0625_p020_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.625",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_nopair_T7_LS030_g3_cmp07_p025_s1_ep10",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_nopair_T7_LS032_g3_cmp07_p025_s1_ep10_normal_tail",
            "T7",
            "0.32",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_nopair_T7_LS035_g3_cmp07_p025_s1_ep10_normal_tail",
            "T7",
            "0.35",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_nopair_T7_LS030_g3_cmp07_p020_s1_ep10_normal_tail",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.20",
            "1",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_nopair_T7_LS030_g3_cmp065_p025_s1_ep10_normal_tail",
            "T7",
            "0.30",
            "3",
            "0.65",
            "0.25",
            "1",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS030_g3_cmp07_p025_s1_ep10_normal_tail",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS030_g3_cmp0725_p025_s1_ep10_combo_push",
            "T7",
            "0.30",
            "3",
            "0.725",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS030_g3_cmp075_p025_s1_ep10_combo_push",
            "T7",
            "0.30",
            "3",
            "0.75",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS030_g3_cmp07_p0275_s1_ep10_combo_push",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.275",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS030_g3_cmp07_p0225_s1_ep10_normal_tail",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.225",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS030_g3_cmp0675_p025_s1_ep10_normal_tail",
            "T7",
            "0.30",
            "3",
            "0.675",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS028_g3_cmp07_p025_s1_ep10_combo_pos_push",
            "T7",
            "0.28",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS032_g3_cmp07_p025_s1_ep10_normal_tail",
            "T7",
            "0.32",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS029_g3_cmp07_p025_s1_ep10_mid_balance",
            "T7",
            "0.29",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s1_ep10_mid_balance",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS02925_g3_cmp07_p025_s1_ep10_mid_balance",
            "T7",
            "0.2925",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s1_ep10_mid_balance",
            "T7",
            "0.2975",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p0255_s1_ep10_combo_pos_push",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.255",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p026_s1_ep10_combo_pos_push",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.26",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp0705_p025_s1_ep10_combo_pos_push",
            "T7",
            "0.295",
            "3",
            "0.705",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_ep10_seedcheck",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s42_ep10_seedcheck",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.25",
            "42",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s13_ep10_seedcheck",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.25",
            "13",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s99_ep10_seedcheck",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.25",
            "99",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp0685_p025_s7_ep10_seed7_neg_tail_guard",
            "T7",
            "0.295",
            "3",
            "0.685",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp0695_p025_s7_ep10_seed7_neg_tail_guard",
            "T7",
            "0.295",
            "3",
            "0.695",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp0705_p02475_s7_ep10_seed7_combo_balance",
            "T7",
            "0.295",
            "3",
            "0.705",
            "0.2475",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS02925_g3_cmp07_p025_s7_ep10_seed7_ls_combo_balance",
            "T7",
            "0.2925",
            "3",
            "0.7",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_ep10_seed7_ls_tail_balance",
            "T7",
            "0.2975",
            "3",
            "0.7",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p02475_s7_ep10_seed7_p_tail_balance",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.2475",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p02525_s7_ep10_seed7_p_combo_balance",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.2525",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p0245_s7_ep10_seed7_p_tail_guard",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.245",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p0255_s7_ep10_seed7_p_combo_guard",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.255",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s77_ep10_seedcheck",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.25",
            "77",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p0245_s1_ep10_neg_tail_guard",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.245",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p02475_s1_ep10_neg_tail_guard",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.2475",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p02525_s1_ep10_combo_pos_push",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.2525",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp0685_p025_s1_ep10_neg_tail_guard",
            "T7",
            "0.295",
            "3",
            "0.685",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp071_p02475_s1_ep10_combo_pos_push",
            "T7",
            "0.295",
            "3",
            "0.71",
            "0.2475",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p024_s1_ep10_neg_tail_guard",
            "T7",
            "0.295",
            "3",
            "0.7",
            "0.24",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp0695_p025_s1_ep10_neg_tail_guard",
            "T7",
            "0.295",
            "3",
            "0.695",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp069_p025_s1_ep10_neg_tail_guard",
            "T7",
            "0.295",
            "3",
            "0.69",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS030_g4_cmp07_p025_s1_ep10_normal_tail",
            "T7",
            "0.30",
            "4",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS030_g3_cmp07_p025_s7_ep10_seedcheck",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS030_g3_cmp07_p025_s42_ep10_seedcheck",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.25",
            "42",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_pair_T7_LS030_g3_cmp07_p025_s77_ep10_seedcheck",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.25",
            "77",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_g4_cls05_ls30_nopair_s42_ep10",
            "T7",
            "0.30",
            "4",
            "0.5",
            "0.25",
            "42",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_g4_cls05_ls30_nopair_s42_ep10_retry_old_eval",
            "T7",
            "0.30",
            "4",
            "0.5",
            "0.25",
            "42",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_g4_cls05_ls30_pair_s42_ep10_retry_old_eval",
            "T7",
            "0.30",
            "4",
            "0.5",
            "0.25",
            "42",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe("iter26B_T7_LS050_g3_cmp05_p025_s1_ep8", "T7", "0.50", "3", "0.5", "0.25", "1", (), "8", "4", "4"),
        Recipe(
            "iter116J_T7_g3_cmp06_p025_s1_ep10_I10only_combo_recall",
            "T7",
            "0.30",
            "3",
            "0.6",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp07_p025_s1_ep10_I10only_combo_recall",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g4_cmp06_p025_s1_ep10_I10only_combo_ood_balance",
            "T7",
            "0.30",
            "4",
            "0.6",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp06_p020_s1_ep10_I10only_less_cutmix",
            "T7",
            "0.30",
            "3",
            "0.6",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp065_p020_s1_ep10_I10only_combo_pos_push",
            "T7",
            "0.30",
            "3",
            "0.65",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp064_p020_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.64",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp0645_p020_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.645",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp063_p020_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.63",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp065_p0195_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.65",
            "0.195",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp065_p019_s1_ep10_I10only_far0_edge",
            "T7",
            "0.30",
            "3",
            "0.65",
            "0.19",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp065_p018_s1_ep10_I10only_combo_pos_push",
            "T7",
            "0.30",
            "3",
            "0.65",
            "0.18",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp06_p022_s1_ep10_I10only_combo_pos_push",
            "T7",
            "0.30",
            "3",
            "0.6",
            "0.22",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_LS028_g3_cmp06_p020_s1_ep10_I10only_scratch_combo_balance",
            "T7",
            "0.28",
            "3",
            "0.6",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_LS032_g3_cmp06_p020_s1_ep10_I10only_scratch_combo_balance",
            "T7",
            "0.32",
            "3",
            "0.6",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_LS028_g3_cmp065_p020_s1_ep10_I10only_scratch_combo_balance",
            "T7",
            "0.28",
            "3",
            "0.65",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_LS032_g3_cmp06_p022_s1_ep10_I10only_scratch_combo_balance",
            "T7",
            "0.32",
            "3",
            "0.6",
            "0.22",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g4_cmp06_p018_s1_ep10_I10only_g4_tail_balance",
            "T7",
            "0.30",
            "4",
            "0.6",
            "0.18",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp06_p025_s1_ep10_I10only_combo_recall_retry",
            "T7",
            "0.30",
            "3",
            "0.6",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp07_p025_s1_ep10_I10only_combo_recall_retry",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp06_p020_nopair_s1_ep10_I10only_tail_check",
            "T7",
            "0.30",
            "3",
            "0.6",
            "0.20",
            "1",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g4_cmp05_p020_s1_ep10_I10only_g4_tail_cmp05",
            "T7",
            "0.30",
            "4",
            "0.5",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g4_cmp06_p020_s1_ep10_I10only_g4_tail_cmp06",
            "T7",
            "0.30",
            "4",
            "0.6",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g3_cmp07_p020_s1_ep10_I10only_gap_balance",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g4_cmp07_p025_s1_ep10_I10only_gap_balance",
            "T7",
            "0.30",
            "4",
            "0.7",
            "0.25",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "iter116J_T7_g4_cmp07_p020_s1_ep10_I10only_gap_balance",
            "T7",
            "0.30",
            "4",
            "0.7",
            "0.20",
            "1",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_g3_cls06_ls30_pair_s42_ep8_I10only",
            "T7",
            "0.30",
            "3",
            "0.6",
            "0.25",
            "42",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
            "8",
        ),
        Recipe(
            "fcm_margin_g3_cls07_ls30_nopair_s42_ep8_I10only",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.25",
            "42",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10"),
            "8",
        ),
        Recipe(
            "fcm_margin_g3_cls07_ls30_pair_s42_ep8_I10only",
            "T7",
            "0.30",
            "3",
            "0.7",
            "0.25",
            "42",
            ("--sweep-eval-variants", "I10"),
            "8",
        ),
        Recipe(
            "fcm_margin_g4_cls05_ls30_nopair_s42_ep8_I10only",
            "T7",
            "0.30",
            "4",
            "0.5",
            "0.25",
            "42",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10"),
            "8",
        ),
        Recipe("fcm_margin_g3_cls07_ls30_nopair_s42_ep10", "T7", "0.30", "3", "0.7", "0.25", "42", ("--cutmix-pair", "none")),
        Recipe("fcm_margin_g3_cls07_ls30_pair_s42_ep10", "T7", "0.30", "3", "0.7", "0.25", "42"),
        Recipe(
            "fcm_margin_g3_cls065_ls30_nopair_s42_ep10_I10only",
            "T7",
            "0.30",
            "3",
            "0.65",
            "0.25",
            "42",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_g3_cls065_ls30_pair_s42_ep10_I10only",
            "T7",
            "0.30",
            "3",
            "0.65",
            "0.25",
            "42",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe(
            "fcm_margin_g4_cls06_ls30_nopair_s42_ep10_I10only",
            "T7",
            "0.30",
            "4",
            "0.6",
            "0.25",
            "42",
            ("--cutmix-pair", "none", "--sweep-eval-variants", "I10", "--sweep-save-every-epoch"),
        ),
        Recipe("fcm_margin_T7_LS030_g3_cmp07_p025_s1_ep10", "T7", "0.30", "3", "0.7", "0.25", "1"),
        Recipe("fcm_margin_nopair_T7_LS030_g3_cmp07_p025_s1_ep10", "T7", "0.30", "3", "0.7", "0.25", "1", ("--cutmix-pair", "none")),
        Recipe("fcm_margin_g4_cls05_ls30_nopair_s42_ep10", "T7", "0.30", "4", "0.5", "0.25", "42", ("--cutmix-pair", "none")),
        Recipe("fcm_margin_g4_cls05_ls30_pair_s42_ep10", "T7", "0.30", "4", "0.5", "0.25", "42"),
        Recipe("fcm_margin_g2_cls05_ls30_nopair_s42_ep10", "T7", "0.30", "2", "0.5", "0.25", "42", ("--cutmix-pair", "none")),
        Recipe("fcm_margin_g2_cls05_ls30_pair_s42_ep10", "T7", "0.30", "2", "0.5", "0.25", "42"),
        Recipe("fcm_margin_g3_cls05_ls40_nopair_s42_ep10", "T7", "0.40", "3", "0.5", "0.25", "42", ("--cutmix-pair", "none")),
        Recipe("fcm_margin_g3_cls05_ls40_pair_s42_ep10", "T7", "0.40", "3", "0.5", "0.25", "42"),
        Recipe("iter26B_T7_LS050_g3_cmp05_p025_s1_ep8", "T7", "0.50", "3", "0.5", "0.25", "1", (), "8", "4", "4"),
        Recipe("iter26D_T7_LS040_g4_cmp05_p025_s1_ep8", "T7", "0.40", "4", "0.5", "0.25", "1", (), "8", "4", "4"),
        Recipe("iter26H_T7_LS067_g3_cmp05_p025_white_s1_ep8", "T7", "0.67", "3", "0.5", "0.25", "1", ("--cutmix-pair-fill", "white"), "8", "4", "4"),
        Recipe("iter21F_T7_LS067_g3_cmp05_p025_s1_ep8", "T7", "0.67", "3", "0.5", "0.25", "1", (), "8", "4", "4"),
        Recipe("iter21H_T7_LS075_g4_cmp05_p025_s1_ep8", "T7", "0.75", "4", "0.5", "0.25", "1", (), "8", "4", "4"),
        Recipe("iter21E_T7_LS100_g2_cmp10_p025_s1_ep8", "T7", "1.00", "2", "1.0", "0.25", "1", (), "8", "4", "4"),
        Recipe("iter25_T7_LS020_g2_cmp05_p025_s1_ep8", "T7", "0.20", "2", "0.5", "0.25", "1", (), "8", "4", "4"),
        Recipe("iter25_T7_LS020_g2_cmp05_p025_s7_ep8", "T7", "0.20", "2", "0.5", "0.25", "7", (), "8", "4", "4"),
        Recipe("iter25_T7_LS020_g2_cmp05_p025_s42_ep8", "T7", "0.20", "2", "0.5", "0.25", "42", (), "8", "4", "4"),
        Recipe("iter25_T7_LS030_g2_cmp05_p025_s1_ep8", "T7", "0.30", "2", "0.5", "0.25", "1", (), "8", "4", "4"),
        Recipe("iter25_T7_LS030_g2_cmp05_p025_s7_ep8", "T7", "0.30", "2", "0.5", "0.25", "7", (), "8", "4", "4"),
        Recipe("iter25_T7_LS030_g2_cmp05_p025_s42_ep8", "T7", "0.30", "2", "0.5", "0.25", "42", (), "8", "4", "4"),
        Recipe("iter116J_exact_T7_LS030_g3_cmp05_p025_s7_ep10", "T7", "0.30", "3", "0.5", "0.25", "7"),
        Recipe("iter116J_exact_T7_LS030_g3_cmp05_p025_s42_ep10", "T7", "0.30", "3", "0.5", "0.25", "42"),
        Recipe("iter116J_exact_T7_LS030_g3_cmp05_p025_s77_ep10", "T7", "0.30", "3", "0.5", "0.25", "77"),
    ]


def prioritized_plan(ds: DatasetSpec, plan: list[Recipe]) -> list[Recipe]:
    """Move proven gap-first basins to the front on transfer datasets."""
    priority_by_dataset = {
        "frozen_iter116J_orig814_v15direct_n2000": [
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid3_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06875_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06875_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid9_s7_ep10",
            "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_lift",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss075_ep10_tail_gap",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_gclip05_ep10_stability",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_bank_scratch2_ep10_combo_gap",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_gap",
            "fcm_margin_pair_T7_LS030_g3_cmp07_p025_s1_ep10_normal_tail",
        ],
        "frozen_iter116J_orig814_eval_n20000": [
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
        ],
        "frozen_original": [
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid3_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06875_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid9_s7_ep10",
            "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_lift",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss075_ep10_tail_gap",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_gclip05_ep10_stability",
        ],
        "sota_gapstress_seed31_260531": [
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid3_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06875_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid9_s7_ep10",
            "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_lift",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss075_ep10_tail_gap",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_gclip05_ep10_stability",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_bank_scratch2_ep10_combo_gap",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_gap",
        ],
        "sota_gapstress_seed97_260531": [
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid3_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06875_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid6_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid9_s7_ep10",
            "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_lift",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss075_ep10_tail_gap",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_gclip05_ep10_stability",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_bank_scratch2_ep10_combo_gap",
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_gap",
        ],
    }
    priority = priority_by_dataset.get(ds.name)
    if not priority:
        return plan
    by_tag = {r.tag: r for r in plan}

    def champion_grid_recipe(grid_dim: str) -> Recipe:
        tag = f"adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid{grid_dim}_s7_ep10"
        return Recipe(
            tag,
            "T7",
            "0.29750",
            "3",
            "0.6975",
            "0.2475",
            "7",
            (
                "--sweep-eval-variants",
                "I10",
                "--sweep-save-every-epoch",
                "--cutmix-pair-bias",
                "fork,scratch:2",
                "--cutmix-grid-dim",
                grid_dim,
            ),
        )

    def champion_mpos065_recipe(tag: str, cmp_ls: str = "0.6975", cutmix_p: str = "0.2475", grid_dim: str | None = None) -> Recipe:
        extra = [
            "--sweep-eval-variants",
            "I10",
            "--sweep-save-every-epoch",
            "--cutmix-pair-bias",
            "fork,scratch:2",
            "--cutmix-mask-pos-target",
            "0.65",
        ]
        if grid_dim:
            extra.extend(["--cutmix-grid-dim", grid_dim])
        return Recipe(tag, "T7", "0.29750", "3", cmp_ls, cutmix_p, "7", tuple(extra))

    def fcmpm_grid_group_ablation_recipe(groups: str, grid_dim: str) -> Recipe:
        tag = (
            f"fcmpm_ablate_T7_LS02975_g{groups}_grid{grid_dim}_cmp06975_p02475_"
            "pair_pbfork_scratchx2_mpos065_s7_ep10"
        )
        return Recipe(
            tag,
            "T7",
            "0.29750",
            groups,
            "0.6975",
            "0.2475",
            "7",
            (
                "--sweep-eval-variants",
                "I10",
                "--sweep-save-every-epoch",
                "--cutmix-pair-bias",
                "fork,scratch:2",
                "--cutmix-mask-pos-target",
                "0.65",
                "--cutmix-grid-dim",
                grid_dim,
            ),
        )

    def fcmpm_p_grid_group_ablation_recipe(groups: str, grid_dim: str, cutmix_p: str) -> Recipe:
        p_tag = cutmix_p.replace(".", "")
        tag = (
            f"fcmpm_pggrid_T7_LS02975_g{groups}_grid{grid_dim}_cmp06975_p{p_tag}_"
            "pair_pbfork_scratchx2_mpos065_s7_ep10"
        )
        return Recipe(
            tag,
            "T7",
            "0.29750",
            groups,
            "0.6975",
            cutmix_p,
            "7",
            (
                "--sweep-eval-variants",
                "I10",
                "--sweep-save-every-epoch",
                "--cutmix-pair-bias",
                "fork,scratch:2",
                "--cutmix-mask-pos-target",
                "0.65",
                "--cutmix-grid-dim",
                grid_dim,
            ),
        )

    def fcmpm_cmp_grid_group_recipe(groups: str, grid_dim: str, cmp_ls: str, cutmix_p: str = "0.3000") -> Recipe:
        cmp_tag = cmp_ls.replace(".", "")
        p_tag = cutmix_p.replace(".", "")
        tag = (
            f"fcmpm_cmpgrid_T7_LS02975_g{groups}_grid{grid_dim}_cmp{cmp_tag}_p{p_tag}_"
            "pair_pbfork_scratchx2_mpos065_s7_ep10"
        )
        return Recipe(
            tag,
            "T7",
            "0.29750",
            groups,
            cmp_ls,
            cutmix_p,
            "7",
            (
                "--sweep-eval-variants",
                "I10",
                "--sweep-save-every-epoch",
                "--cutmix-pair-bias",
                "fork,scratch:2",
                "--cutmix-mask-pos-target",
                "0.65",
                "--cutmix-grid-dim",
                grid_dim,
            ),
        )

    def fcmpm_high_p_matrix_recipe(
        cutmix_p: str,
        cmp_ls: str,
        groups: str,
        grid_dim: str,
        mpos: str = "0.65",
        label_area_prop: bool = False,
        ab_labels: str = "",
        other_label: str = "",
        raw_target: bool = False,
        ls: str = "0.29750",
        variant: str = "T7",
    ) -> Recipe:
        p_tag = cutmix_p.replace(".", "")
        cmp_tag = cmp_ls.replace(".", "")
        mpos_tag = mpos.replace(".", "")
        ls_tag = ls.replace(".", "")
        mode_tag = "_areaprop" if label_area_prop else ""
        if ab_labels:
            mode_tag += "_ab" + ab_labels.replace(",", "_").replace(".", "")
        if other_label:
            mode_tag += "_ol" + other_label.replace(".", "")
        if raw_target:
            mode_tag += "_raw"
        tag = (
            f"fcmpm_hpmat{mode_tag}_{variant}_LS{ls_tag}_g{groups}_grid{grid_dim}_cmp{cmp_tag}_p{p_tag}_"
            f"pair_pbfork_scratchx2_mpos{mpos_tag}_s7_ep10"
        )
        extra = [
            "--sweep-eval-variants",
            "I10",
            "--sweep-save-every-epoch",
            "--cutmix-pair-bias",
            "fork,scratch:2",
            "--cutmix-mask-pos-target",
            mpos,
            "--cutmix-grid-dim",
            grid_dim,
        ]
        if label_area_prop:
            extra.append("--cutmix-label-area-prop")
        if ab_labels:
            extra.extend(["--cutmix-ab-labels", ab_labels])
        if other_label:
            extra.extend(["--cutmix-other-label", other_label])
        if raw_target:
            extra.append("--cutmix-mix-raw-target")
        return Recipe(
            tag,
            variant,
            ls,
            groups,
            cmp_ls,
            cutmix_p,
            "7",
            tuple(extra),
        )

    explicit = {
        "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_s7_ep10": Recipe(
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_s7_ep10",
            "T7",
            "0.29750",
            "3",
            "0.6975",
            "0.2475",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch", "--cutmix-pair-bias", "fork,scratch:2"),
        ),
        "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10": Recipe(
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            "T7",
            "0.29750",
            "3",
            "0.6975",
            "0.2475",
            "7",
            (
                "--sweep-eval-variants",
                "I10",
                "--sweep-save-every-epoch",
                "--cutmix-pair-bias",
                "fork,scratch:2",
                "--cutmix-mask-pos-target",
                "0.65",
            ),
        ),
        "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid3_s7_ep10": champion_grid_recipe("3"),
        "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid6_s7_ep10": champion_grid_recipe("6"),
        "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_grid9_s7_ep10": champion_grid_recipe("9"),
        "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10": champion_mpos065_recipe(
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            grid_dim="6",
        ),
        "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10": champion_mpos065_recipe(
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            cmp_ls="0.6925",
            grid_dim="6",
        ),
        "adapt_T7_LS02975_g3_cmp06875_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10": champion_mpos065_recipe(
            "adapt_T7_LS02975_g3_cmp06875_p02475_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            cmp_ls="0.6875",
            grid_dim="6",
        ),
        "adapt_T7_LS02975_g3_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10": champion_mpos065_recipe(
            "adapt_T7_LS02975_g3_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_grid6_s7_ep10",
            cutmix_p="0.2250",
            grid_dim="6",
        ),
        "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10": champion_mpos065_recipe(
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
            grid_dim="9",
        ),
        "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10": champion_mpos065_recipe(
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_grid9_s7_ep10",
            cmp_ls="0.6925",
            grid_dim="9",
        ),
        "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10": champion_mpos065_recipe(
            "adapt_T7_LS02975_g3_cmp06925_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            cmp_ls="0.6925",
        ),
        "adapt_T7_LS02975_g3_cmp06875_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10": champion_mpos065_recipe(
            "adapt_T7_LS02975_g3_cmp06875_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
            cmp_ls="0.6875",
        ),
        "adapt_T7_LS02975_g3_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10": champion_mpos065_recipe(
            "adapt_T7_LS02975_g3_cmp06975_p02250_pair_pbfork_scratchx2_mpos065_s7_ep10",
            cutmix_p="0.2250",
        ),
        "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_lift": Recipe(
            "fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_lift",
            "T7",
            "0.2975",
            "3",
            "0.70",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch", "--cutmix-pair-bias", "fork,scratch:2"),
        ),
        "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss075_ep10_tail_gap": Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss075_ep10_tail_gap",
            "T7",
            "0.295",
            "3",
            "0.70",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch", "--cutmix-pair-loss-w", "0.75"),
        ),
        "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_gclip05_ep10_stability": Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_gclip05_ep10_stability",
            "T7",
            "0.295",
            "3",
            "0.70",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch", "--grad-clip", "0.5"),
        ),
        "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_bank_scratch2_ep10_combo_gap": Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_bank_scratch2_ep10_combo_gap",
            "T7",
            "0.295",
            "3",
            "0.70",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch", "--cutmix-pair-bias", "bank_boundary,scratch:2"),
        ),
        "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_gap": Recipe(
            "fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_gap",
            "T7",
            "0.295",
            "3",
            "0.70",
            "0.25",
            "7",
            ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch", "--cutmix-pair-bias", "fork,scratch:2"),
        ),
    }
    if ds.name in {
        "frozen_iter116J_orig814_v15direct_n2000",
        "frozen_iter116J_orig814_eval_n20000",
        "frozen_original",
        "sota_gapstress_seed31_260531",
        "sota_gapstress_seed97_260531",
    }:
        # Controlled FCMPM grid/group matrix requested on 260531.  The spatial
        # grid must be group-aligned: for n_groups=g, use (g*k)x(g*k), not a
        # global grid list.  That gives g=2 -> 2/4/6/8, g=3 -> 3/6/9/12,
        # g=4 -> 4/8/12/16 under the same LS/cmp/p/seed/mpos condition.
        group_grid_dims = {
            "2": ("2", "4", "6", "8"),
            "3": ("3", "6", "9", "12"),
            "4": ("4", "8", "12", "16"),
        }
        group_primary_grid = {
            "2": "8",
            "3": "9",
            "4": "8",
        }
        grid_group_tags: list[str] = []
        for groups in ("2", "3", "4"):
            for grid_dim in group_grid_dims[groups]:
                recipe = fcmpm_grid_group_ablation_recipe(groups, grid_dim)
                explicit[recipe.tag] = recipe
                grid_group_tags.append(recipe.tag)
        # Follow-up controlled matrix: FCMPM generation probability together
        # with group count and cells-per-group resolution.  p=0.2475 is covered
        # by the grid_group_tags above, so this adds the neighboring p values.
        p_grid_group_tags: list[str] = []
        for cutmix_p in ("0.2000", "0.3000", "0.4000", "0.5000", "0.6000"):
            for groups in ("2", "3", "4"):
                for grid_dim in group_grid_dims[groups]:
                    recipe = fcmpm_p_grid_group_ablation_recipe(groups, grid_dim, cutmix_p)
                    explicit[recipe.tag] = recipe
                    p_grid_group_tags.append(recipe.tag)
        # Exact cmp crossing over aligned group grids.  This isolates label
        # strength from p/grid/group effects; cmp=0.6975 is the local basin,
        # but user-facing analysis needs the interpretable 0.5/0.6/0.7/1.0 axis.
        cmp_grid_group_tags: list[str] = []
        for cmp_ls in ("0.5000", "0.6000", "0.7000", "1.0000"):
            for groups in ("2", "3", "4"):
                for grid_dim in group_grid_dims[groups]:
                    recipe = fcmpm_cmp_grid_group_recipe(groups, grid_dim, cmp_ls, "0.3000")
                    explicit[recipe.tag] = recipe
                    cmp_grid_group_tags.append(recipe.tag)
        # One-axis FCMPM probability sweep on the aligned g=3 basin
        # (grid=9x9, g=3, cmp=0.6975, mpos065).  This separates
        # "how often to synthesize" from grid/group/cmp effects.
        p_axis_tags: list[str] = []
        p_axis_values = (
            "0.0500", "0.1000", "0.1500", "0.1750", "0.2000", "0.2250",
            "0.2350", "0.2400", "0.2425", "0.2450", "0.2500", "0.2525",
            "0.2550", "0.2600", "0.2650", "0.2750", "0.3000", "0.3500",
            "0.4000", "0.5000", "0.6000",
        )
        for cutmix_p in p_axis_values:
            p_tag = cutmix_p.replace(".", "")
            tag = (
                f"fcmpm_paxis_T7_LS02975_g3_grid9_cmp06975_p{p_tag}_"
                "pair_pbfork_scratchx2_mpos065_s7_ep10"
            )
            recipe = champion_mpos065_recipe(tag, cutmix_p=cutmix_p, grid_dim="9")
            explicit[recipe.tag] = recipe
            p_axis_tags.append(recipe.tag)
        # Explicit A/B labels under the normal mix-loss path.  Raw-target A/B
        # labels repeatedly raised OOD tails; this keeps the A/B label
        # asymmetry while allowing the established BCE+LS path to calibrate it
        # on the aligned g=3/grid9 geometry.
        smooth_ab_tags: list[str] = []
        smooth_ab_conditions = (
            ("0.70,0.50", "0.0"),
            ("0.50,0.70", "0.0"),
            ("0.70,0.70", "0.0"),
            ("0.65,0.65", "0.0"),
            ("0.65,0.55", "0.0"),
            ("0.55,0.65", "0.0"),
            ("0.70,0.70", "0.05"),
        )
        for ab_labels, other_label in smooth_ab_conditions:
            recipe = fcmpm_high_p_matrix_recipe(
                "0.2475",
                "0.6975",
                "3",
                "9",
                "0.65",
                ab_labels=ab_labels,
                other_label=other_label,
                raw_target=False,
            )
            explicit[recipe.tag] = recipe
            smooth_ab_tags.append(recipe.tag)
        # Controlled label matrix requested on 260531:
        # g=2/3/4 x grid resolution x FCMPM probability x explicit A/B combo
        # target x off-class negative target.  This separates "how often to
        # generate" from "how strong each combo bit should be" and from whether
        # the two non-combo bits should remain hard zero or soft negative.
        label_matrix_tags: list[str] = []
        label_matrix_main_p = ("0.2475", "0.3000")
        label_matrix_groups = ("3", "2", "4")
        label_matrix_ab = ("0.70,0.50", "0.50,0.70", "0.70,0.70")
        label_matrix_other = ("0.0", "0.10")
        for cutmix_p in label_matrix_main_p:
            for groups in label_matrix_groups:
                for grid_dim in group_grid_dims[groups]:
                    for ab_labels in label_matrix_ab:
                        for other_label in label_matrix_other:
                            recipe = fcmpm_high_p_matrix_recipe(
                                cutmix_p,
                                "0.6975",
                                groups,
                                grid_dim,
                                "0.65",
                                ab_labels=ab_labels,
                                other_label=other_label,
                                raw_target=True,
                            )
                            explicit[recipe.tag] = recipe
                            label_matrix_tags.append(recipe.tag)
        # Narrow negative-label interpolation around the current best geometry.
        for cutmix_p in label_matrix_main_p:
            for groups in label_matrix_groups:
                for ab_labels in label_matrix_ab:
                    recipe = fcmpm_high_p_matrix_recipe(
                        cutmix_p,
                        "0.6975",
                        groups,
                        group_primary_grid[groups],
                        "0.65",
                        ab_labels=ab_labels,
                        other_label="0.05",
                        raw_target=True,
                    )
                    explicit[recipe.tag] = recipe
                    label_matrix_tags.append(recipe.tag)
        # High-p saturation check with explicit labels.  Keep this to each
        # group's primary aligned grid so p=0.4/0.5/0.6 does not consume the full
        # geometry budget before it proves the POS/NEG gap can survive.
        for cutmix_p in ("0.4000", "0.5000", "0.6000"):
            for groups in label_matrix_groups:
                for ab_labels in ("0.70,0.50", "0.50,0.70"):
                    for other_label in ("0.0", "0.10"):
                        recipe = fcmpm_high_p_matrix_recipe(
                            cutmix_p,
                            "0.6975",
                            groups,
                            group_primary_grid[groups],
                            "0.65",
                            ab_labels=ab_labels,
                            other_label=other_label,
                            raw_target=True,
                        )
                        explicit[recipe.tag] = recipe
                        label_matrix_tags.append(recipe.tag)
        # Group-aware label matrix.  The default FCMPM label is a presence label:
        # A and B bits both receive cmp even when A occupies only 1/g of the image.
        # Area-prop tests whether labels should instead follow visible support:
        # A=cmp/g, B=cmp*(g-1)/g.
        group_label_tags: list[str] = []
        group_label_values = ("0.2475", "0.3000")
        group_label_conditions = (
            ("0.6975", "2", "8", "0.65"),
            ("1.0000", "2", "8", "0.65"),
            ("0.6975", "3", "9", "0.65"),
            ("1.0000", "3", "9", "0.65"),
            ("0.6975", "4", "8", "0.65"),
            ("1.0000", "4", "8", "0.65"),
            ("1.0000", "3", "6", "0.65"),
            ("1.0000", "3", "9", "0.65"),
        )
        for cutmix_p in group_label_values:
            for cmp_ls, groups, grid_dim, mpos in group_label_conditions:
                recipe = fcmpm_high_p_matrix_recipe(
                    cutmix_p,
                    cmp_ls,
                    groups,
                    grid_dim,
                    mpos,
                    label_area_prop=True,
                )
                explicit[recipe.tag] = recipe
                group_label_tags.append(recipe.tag)
        # High-p must not be judged from one fixed condition only.  For
        # p>=0.3, test whether lowering cmp or changing grid/group geometry
        # recovers POS/NEG gap instead of over-activating Normal/OOD tails.
        high_p_matrix_tags: list[str] = []
        high_p_values = ("0.3000", "0.4000", "0.5000", "0.6000")
        high_p_conditions = (
            ("0.5000", "3", "9", "0.65"),
            ("0.6000", "3", "9", "0.65"),
            ("0.8000", "3", "9", "0.65"),
            ("0.6000", "3", "6", "0.65"),
            ("0.6000", "3", "12", "0.65"),
            ("0.6000", "4", "8", "0.65"),
            ("0.6000", "4", "12", "0.65"),
            ("0.6000", "2", "8", "0.65"),
            ("0.6000", "2", "4", "0.65"),
            ("0.6000", "3", "9", "0.60"),
        )
        for cutmix_p in high_p_values:
            for cmp_ls, groups, grid_dim, mpos in high_p_conditions:
                recipe = fcmpm_high_p_matrix_recipe(cutmix_p, cmp_ls, groups, grid_dim, mpos)
                explicit[recipe.tag] = recipe
                high_p_matrix_tags.append(recipe.tag)
        # Overnight compensation sweep (260531 user).  In FCMPM complement,
        # mix = B base + A cells, so g=3 exposes A on 1/3 of the image and B
        # on 2/3.  Keep B at a hard positive target and lower A from 1.0 to
        # 0.8 to test whether the smaller visible support needs full target
        # pressure while the dominant support should be softened.  Pair this
        # with cmp=1.0/high-p and lower LS neighbors because aggregate evidence
        # showed larger cmp and p around 0.5 can improve POS min if NEG tails
        # stay controlled.
        target_comp_tags: list[str] = []
        target_comp_ab = ("1.00,1.00", "0.95,1.00", "0.90,1.00", "0.85,1.00", "0.80,1.00")
        target_comp_ls = ("0.28000", "0.29000", "0.29500", "0.29750")
        target_comp_p = ("0.3000", "0.4000", "0.5000", "0.6000")
        target_comp_geometries = (("3", "9"), ("3", "12"), ("4", "8"), ("2", "4"))
        for ls in target_comp_ls:
            for cutmix_p in target_comp_p:
                for groups, grid_dim in target_comp_geometries:
                    for ab_labels in target_comp_ab:
                        for raw_target in (False, True):
                            recipe = fcmpm_high_p_matrix_recipe(
                                cutmix_p,
                                "1.0000",
                                groups,
                                grid_dim,
                                "0.65",
                                ab_labels=ab_labels,
                                other_label="0.0",
                                raw_target=raw_target,
                                ls=ls,
                            )
                            explicit[recipe.tag] = recipe
                            target_comp_tags.append(recipe.tag)
        # ASL check: if positive pressure rather than label value is the right
        # lever, T10 should raise weak combo POS without forcing raw hard
        # targets as aggressively as T7 raw mix.
        for cutmix_p in ("0.4000", "0.5000"):
            for ab_labels in ("0.90,1.00", "0.80,1.00", "1.00,1.00"):
                recipe = fcmpm_high_p_matrix_recipe(
                    cutmix_p,
                    "1.0000",
                    "3",
                    "9",
                    "0.65",
                    ab_labels=ab_labels,
                    other_label="0.0",
                    raw_target=False,
                    ls="0.29500",
                    variant="T10",
                )
                explicit[recipe.tag] = recipe
                target_comp_tags.append(recipe.tag)
        # Focused A/B follow-up from the sample-count evidence: keep B at 1.0,
        # lower A to 0.9/0.8, and test both lower p and the coarser g3/grid6
        # geometry before the broad matrix consumes the queue.
        for cutmix_p, groups, grid_dim in (("0.2000", "3", "9"), ("0.5000", "3", "6")):
            for ab_labels in ("0.90,1.00", "0.80,1.00"):
                recipe = fcmpm_high_p_matrix_recipe(
                    cutmix_p,
                    "1.0000",
                    groups,
                    grid_dim,
                    "0.65",
                    ab_labels=ab_labels,
                    other_label="0.0",
                    raw_target=False,
                    ls="0.29500",
                )
                explicit[recipe.tag] = recipe
                target_comp_tags.append(recipe.tag)
        # Train/eval sample-size matrix requested on 260531.  Train each
        # train_n condition once, then replay the selected model at larger eval
        # caps.  eval_n=20000 means "use up to 20000/class"; on smaller roots it
        # becomes all available samples in that class.
        sample_size_tags: list[str] = []

        def add_sample_size_recipe(source: Recipe, tag_prefix: str, train_n: str, eval_n: str = "200") -> None:
            tag = f"{tag_prefix}_tr{int(train_n):03d}_ev{int(eval_n):05d}"
            extra_eval_caps = "2000,20000" if ds.name == "frozen_iter116J_orig814_eval_n20000" else "2000"
            extra = tuple(source.extra) + (
                "--max-per-class-defect",
                train_n,
                "--max-per-class-defect-select",
                "random",
                "--sweep-eval-n-per-class",
                eval_n,
                "--sweep-train-eval-n-per-class",
                train_n,
                "--sweep-train-diag-cap",
                train_n,
                "--sweep-eval-diag-cap",
                eval_n,
                "--sweep-extra-eval-caps",
                extra_eval_caps,
            )
            explicit[tag] = Recipe(
                tag,
                source.variant,
                source.ls,
                source.groups,
                source.cmp_ls,
                source.cutmix_p,
                source.seed,
                extra,
                source.epochs,
                source.batch,
                source.accum,
            )
            sample_size_tags.append(tag)

        # Exact target-label matrix requested on 260601.  In the implemented
        # FCMPM mix, B is the base image and A occupies the complement cells;
        # for g=3 this makes A the weaker/harder positive and B the stronger
        # positive.  Keep weak POS hard at 1.0, soften only the strong POS, and
        # test off-class negative targets independently.
        target_label_matrix_tags: list[str] = []

        def label_pct_tag(value: str) -> str:
            x = float(value)
            pct = x * 100
            if abs(pct - round(pct)) < 1e-9:
                return f"{int(round(pct)):03d}"
            return f"{int(round(x * 1000)):04d}"

        def add_target_label_matrix_recipe(cutmix_p: str, strong_pos: str, neg_target: str) -> None:
            source = fcmpm_high_p_matrix_recipe(
                cutmix_p,
                "1.0000",
                "3",
                "9",
                "0.65",
                ab_labels=f"1.00,{strong_pos}",
                other_label=neg_target,
                raw_target=False,
                ls="0.29500",
            )
            tag = (
                "targetlabel_weak100_"
                f"strong{label_pct_tag(strong_pos)}_neg{label_pct_tag(neg_target)}_"
                f"T7_LS02950_g3_grid9_cmp10000_p{label_pct_tag(cutmix_p)}00_"
                "mpos065_s7_ep10_tr200_ev02000"
            )
            extra = tuple(source.extra) + (
                "--max-per-class-defect",
                "200",
                "--max-per-class-defect-select",
                "random",
                "--sweep-eval-n-per-class",
                "2000",
                "--sweep-train-eval-n-per-class",
                "200",
                "--sweep-train-diag-cap",
                "200",
                "--sweep-eval-diag-cap",
                "2000",
            )
            explicit[tag] = Recipe(
                tag,
                source.variant,
                source.ls,
                source.groups,
                source.cmp_ls,
                source.cutmix_p,
                source.seed,
                extra,
                source.epochs,
                source.batch,
                source.accum,
            )
            target_label_matrix_tags.append(tag)

        for cutmix_p in ("0.5000", "0.4000", "0.3000", "0.6000"):
            for strong_pos in ("1.00", "0.90", "0.80", "0.70"):
                for neg_target in ("0.0", "0.10"):
                    add_target_label_matrix_recipe(cutmix_p, strong_pos, neg_target)

        # One-axis ablation suite (260603):
        # Keep all non-tested factors fixed, then change exactly one axis.
        # Korean report name: "단일 변수 분리 평가".
        # Baseline geometry:
        #   T7 / LS=0.295 / g=3 / grid=9x9 / cmp=1.0 / p=0.5
        #   mpos=0.65 / seed=7 / train=200/class / eval=2000/class.
        # First requested axis: A/B positive target, with B fixed at 1.0 and
        # A swept 1.0 -> 0.9 -> 0.8 -> 0.7.
        one_axis_ablation_tags: list[str] = []

        def add_one_axis_recipe(
            axis: str,
            value_tag: str,
            *,
            ab_labels: str = "1.00,1.00",
            other_label: str = "0.0",
            cutmix_p: str = "0.5000",
            cmp_ls: str = "1.0000",
            groups: str = "3",
            grid_dim: str = "9",
            ls: str = "0.29500",
            variant: str = "T7",
            seed: str = "7",
        ) -> None:
            source = fcmpm_high_p_matrix_recipe(
                cutmix_p,
                cmp_ls,
                groups,
                grid_dim,
                "0.65",
                ab_labels=ab_labels,
                other_label=other_label,
                raw_target=False,
                ls=ls,
                variant=variant,
            )
            tag = (
                f"oneaxis_{axis}_{value_tag}_"
                f"{variant}_LS{ls.replace('.', '')}_g{groups}_grid{grid_dim}_"
                f"cmp{cmp_ls.replace('.', '')}_p{cutmix_p.replace('.', '')}_"
                f"mpos065_s{seed}_ep10_tr200_ev02000"
            )
            extra = tuple(source.extra) + (
                "--max-per-class-defect",
                "200",
                "--max-per-class-defect-select",
                "random",
                "--sweep-eval-n-per-class",
                "2000",
                "--sweep-train-eval-n-per-class",
                "200",
                "--sweep-train-diag-cap",
                "200",
                "--sweep-eval-diag-cap",
                "2000",
            )
            explicit[tag] = Recipe(
                tag,
                source.variant,
                source.ls,
                source.groups,
                source.cmp_ls,
                source.cutmix_p,
                seed,
                extra,
                source.epochs,
                source.batch,
                source.accum,
            )
            one_axis_ablation_tags.append(tag)

        # Baseline A=1.00/B=1.00 already exists as
        # targetlabel_weak100_strong100_neg000_...; reuse that row instead of
        # spending another train on the same parameter key.
        for a_target in ("0.90", "0.80", "0.70"):
            add_one_axis_recipe(
                "abpos_Avar_B100",
                f"A{label_pct_tag(a_target)}_B100",
                ab_labels=f"{a_target},1.00",
            )

        # Additional controlled one-axis splits to identify which factor changes
        # POS min, NEG max, bit_F1, and FAR.  These run after the requested A/B
        # target axis and keep the same baseline unless the axis itself changes.
        # Baselines p=0.5 / neg=0.0 / grid=9 / g=3 are reused from the existing
        # targetlabel_weak100_strong100_neg000 row.
        for neg_target in ("0.015", "0.02", "0.025", "0.03", "0.05", "0.10"):
            add_one_axis_recipe(
                "neg_target",
                f"neg{label_pct_tag(neg_target)}",
                other_label=neg_target,
            )
        for cutmix_p in ("0.2000", "0.3000", "0.4000", "0.5500", "0.5750", "0.6000", "0.6250", "0.6500", "0.7000", "0.8000", "0.9000", "1.0000"):
            add_one_axis_recipe(
                "cutmix_p",
                f"p{label_pct_tag(cutmix_p)}",
                cutmix_p=cutmix_p,
            )
        # ASL/T4/T6 controls were completed on frozen_original and collapsed or
        # leaked (T10/T4 F1~0, T6 FAR-heavy).  Keep the evidence in reports but
        # do not spend transfer repeats on these loss variants.
        # Replicate only top/probable candidates.  One-axis sweeps are for
        # direction; paper evidence needs seed-repeat stability on the best
        # known rows.
        for seed in ("13", "42", "99"):
            add_one_axis_recipe(
                "seed_repeat_baseline",
                f"s{seed}",
                seed=seed,
            )
            for neg_target in ("0.02", "0.05"):
                add_one_axis_recipe(
                    "seed_repeat_neg",
                    f"neg{label_pct_tag(neg_target)}_s{seed}",
                    other_label=neg_target,
                    seed=seed,
                )
            for cutmix_p in ("0.5500", "0.5750", "0.6000", "0.6500", "0.7000", "0.8000"):
                add_one_axis_recipe(
                    "seed_repeat_p",
                    f"p{label_pct_tag(cutmix_p)}_s{seed}",
                    cutmix_p=cutmix_p,
                    seed=seed,
                )
            for grid_dim in ("3", "6", "9", "12"):
                add_one_axis_recipe(
                    "seed_repeat_grid_g3",
                    f"grid{grid_dim}_s{seed}",
                    grid_dim=grid_dim,
                    seed=seed,
                )
        # cmp axis already has extensive historical evidence; do not spend this
        # one-axis queue on repeated cmp=0.5/0.7/0.8/1.0 runs.
        for grid_dim in ("3", "6", "9", "12"):
            add_one_axis_recipe(
                "grid_g3",
                f"grid{grid_dim}",
                grid_dim=grid_dim,
            )
        for groups, grid_dim in (("2", "6"), ("4", "12")):
            add_one_axis_recipe(
                "group_aligned_grid",
                f"g{groups}_grid{grid_dim}",
                groups=groups,
                grid_dim=grid_dim,
            )

        # 2-factor / 3-factor follow-up queue (260603):
        # Keep this after the one-axis suite.  It is intentionally conservative:
        # combine only values near the current known-good baseline, then prune
        # weak regions from the leaderboard instead of exhaustively crossing all
        # axes.  This implements the user's requested sequence:
        #   one-axis -> promising 2-factor combinations -> compact 3-factor neighborhood.
        two_factor_ablation_tags: list[str] = []
        three_factor_ablation_tags: list[str] = []

        def add_factor_recipe(
            phase_tags: list[str],
            axis: str,
            value_tag: str,
            *,
            ab_labels: str = "1.00,1.00",
            other_label: str = "0.0",
            cutmix_p: str = "0.5000",
            groups: str = "3",
            grid_dim: str = "9",
            variant: str = "T7",
            seed: str = "7",
        ) -> None:
            before = len(one_axis_ablation_tags)
            add_one_axis_recipe(
                axis,
                value_tag,
                ab_labels=ab_labels,
                other_label=other_label,
                cutmix_p=cutmix_p,
                groups=groups,
                grid_dim=grid_dim,
                variant=variant,
                seed=seed,
            )
            tag = one_axis_ablation_tags.pop()
            explicit[tag] = explicit[tag]
            phase_tags.append(tag)
            assert len(one_axis_ablation_tags) == before

        for a_target, neg_target in (
            ("0.90", "0.02"),
            ("0.90", "0.05"),
        ):
            add_factor_recipe(
                two_factor_ablation_tags,
                "twofactor_abpos_neg",
                f"A{label_pct_tag(a_target)}_B100_neg{label_pct_tag(neg_target)}",
                ab_labels=f"{a_target},1.00",
                other_label=neg_target,
            )
        for a_target, cutmix_p in (
            ("0.90", "0.5500"),
            ("0.90", "0.5750"),
            ("0.90", "0.6000"),
        ):
            add_factor_recipe(
                two_factor_ablation_tags,
                "twofactor_abpos_p",
                f"A{label_pct_tag(a_target)}_B100_p{label_pct_tag(cutmix_p)}",
                ab_labels=f"{a_target},1.00",
                cutmix_p=cutmix_p,
            )
        for neg_target, cutmix_p in (
            ("0.015", "0.5750"),
            ("0.02", "0.5500"),
            ("0.02", "0.5750"),
            ("0.02", "0.6000"),
            ("0.05", "0.5750"),
        ):
            add_factor_recipe(
                two_factor_ablation_tags,
                "twofactor_neg_p",
                f"neg{label_pct_tag(neg_target)}_p{label_pct_tag(cutmix_p)}",
                other_label=neg_target,
                cutmix_p=cutmix_p,
            )
        for grid_dim, cutmix_p in (
            ("6", "0.5750"),
            ("12", "0.5750"),
        ):
            add_factor_recipe(
                two_factor_ablation_tags,
                "twofactor_grid_p",
                f"grid{grid_dim}_p{label_pct_tag(cutmix_p)}",
                grid_dim=grid_dim,
                cutmix_p=cutmix_p,
            )
        for a_target, neg_target, cutmix_p in (
            ("0.90", "0.015", "0.5750"),
            ("0.90", "0.02", "0.5500"),
            ("0.90", "0.02", "0.5750"),
            ("0.90", "0.02", "0.6000"),
            ("0.90", "0.05", "0.5750"),
        ):
            add_factor_recipe(
                three_factor_ablation_tags,
                "threefactor_abpos_neg_p",
                f"A{label_pct_tag(a_target)}_B100_neg{label_pct_tag(neg_target)}_p{label_pct_tag(cutmix_p)}",
                ab_labels=f"{a_target},1.00",
                other_label=neg_target,
                cutmix_p=cutmix_p,
            )
        for a_target, neg_target, grid_dim in (
            ("0.90", "0.02", "6"),
            ("0.90", "0.02", "12"),
        ):
            add_factor_recipe(
                three_factor_ablation_tags,
                "threefactor_abpos_neg_grid",
                f"A{label_pct_tag(a_target)}_B100_neg{label_pct_tag(neg_target)}_grid{grid_dim}",
                ab_labels=f"{a_target},1.00",
                other_label=neg_target,
                grid_dim=grid_dim,
            )
        # Follow-up around the target-label rows (260601):
        # neg=0.03 is current best on eval_n20000, neg=0.04 starts leaking,
        # neg=0.05 collapses, and neg=0.07 showed early FAR explosion.  Spend
        # the remaining matrix near the useful band instead of repeating bad
        # high-neg settings across every p/strong pair.
        target_label_refine_tags: list[str] = []

        def add_target_label_refine_recipe(cutmix_p: str, strong_pos: str, neg_target: str) -> None:
            source = fcmpm_high_p_matrix_recipe(
                cutmix_p,
                "1.0000",
                "3",
                "9",
                "0.65",
                ab_labels=f"1.00,{strong_pos}",
                other_label=neg_target,
                raw_target=False,
                ls="0.29500",
            )
            tag = (
                "targetlabel_refine_weak100_"
                f"strong{label_pct_tag(strong_pos)}_neg{label_pct_tag(neg_target)}_"
                f"T7_LS02950_g3_grid9_cmp10000_p{label_pct_tag(cutmix_p)}00_"
                "mpos065_s7_ep10_tr200_ev02000"
            )
            extra = tuple(source.extra) + (
                "--max-per-class-defect",
                "200",
                "--max-per-class-defect-select",
                "random",
                "--sweep-eval-n-per-class",
                "2000",
                "--sweep-train-eval-n-per-class",
                "200",
                "--sweep-train-diag-cap",
                "200",
                "--sweep-eval-diag-cap",
                "2000",
            )
            explicit[tag] = Recipe(
                tag,
                source.variant,
                source.ls,
                source.groups,
                source.cmp_ls,
                source.cutmix_p,
                source.seed,
                extra,
                source.epochs,
                source.batch,
                source.accum,
            )
            target_label_refine_tags.append(tag)

        # Prioritize the observed best band first.  neg=0.025 collapsed gap
        # (0.039) and neg=0.035 stayed FAR-heavy through ep05, so the p-axis
        # around neg=0.03 is higher-value than continuing fractional neg probes.
        target_label_refine_seen: set[tuple[str, str, str]] = set()

        def add_target_label_refine_once(cutmix_p: str, strong_pos: str, neg_target: str) -> None:
            key = (cutmix_p, strong_pos, neg_target)
            if key in target_label_refine_seen:
                return
            target_label_refine_seen.add(key)
            add_target_label_refine_recipe(cutmix_p, strong_pos, neg_target)

        add_target_label_refine_once("0.5500", "1.00", "0.03")
        # Fine interpolation around the current p=0.50 best row.  Completed
        # rows show neg=0.02 has better gap but slightly lower F1, while
        # neg=0.03 gives the lowest FAR; probe only the narrow band between
        # them and just above the best before broader p/strong sweeps.
        for neg_target in ("0.028", "0.032"):
            add_target_label_refine_once("0.5000", "1.00", neg_target)
        for cutmix_p in ("0.6000", "0.4500", "0.4000", "0.3500", "0.3000"):
            add_target_label_refine_once(cutmix_p, "1.00", "0.03")
        for cutmix_p in ("0.5500", "0.6000", "0.4500", "0.4000", "0.3500", "0.3000"):
            add_target_label_refine_once(cutmix_p, "1.00", "0.02")
        for cutmix_p in ("0.5000", "0.5500", "0.6000", "0.4500", "0.4000", "0.3500", "0.3000"):
            for strong_pos in ("1.00", "0.95", "0.90"):
                for neg_target in ("0.01", "0.02", "0.03", "0.04"):
                    add_target_label_refine_once(cutmix_p, strong_pos, neg_target)
        # Alternate family: strong=0.80/neg=0.00 lowered Invalid/sr in the
        # completed p=0.50 matrix.  Test whether slightly higher synthesis
        # probability recovers combo POS without re-opening the Invalid tail.
        add_target_label_refine_once("0.5500", "0.80", "0.00")

        def recipe_with_extra(source: Recipe, extra_suffix: tuple[str, ...]) -> Recipe:
            return Recipe(
                source.tag,
                source.variant,
                source.ls,
                source.groups,
                source.cmp_ls,
                source.cutmix_p,
                source.seed,
                tuple(source.extra) + extra_suffix,
                source.epochs,
                source.batch,
                source.accum,
            )

        sample_size_sources = (
            (
                "samplecap_T7_LS02975_g3_grid9_cmp06975_p02475_mpos065_s7_ep10",
                champion_mpos065_recipe(
                    "sample_src_T7_LS02975_g3_grid9_cmp06975_p02475_mpos065_s7_ep10",
                    cmp_ls="0.6975",
                    cutmix_p="0.2475",
                    grid_dim="9",
                ),
            ),
            (
                "samplecap_T7_LS02950_g3_grid9_cmp10000_p05000_ab090_100_mpos065_s7_ep10",
                fcmpm_high_p_matrix_recipe(
                    "0.5000",
                    "1.0000",
                    "3",
                    "9",
                    "0.65",
                    ab_labels="0.90,1.00",
                    other_label="0.0",
                    raw_target=False,
                    ls="0.29500",
                ),
            ),
        )
        for train_n in ("50", "100", "200"):
            for tag_prefix, source in sample_size_sources:
                add_sample_size_recipe(source, tag_prefix, train_n)
        # The sample-count matrix shows this high-pos source improves sharply
        # from train=50 -> 100 -> 200.  Extend only that source to larger train
        # caps; the cmp=0.6975/p=0.2475 source is weak at train=200 and should
        # not consume the next queue slots.
        highpos_sample_prefix, highpos_sample_source = sample_size_sources[1]
        extra_train_caps = ("300",) if ds.name == "frozen_iter116J_orig814_v15direct_n2000" else ("300", "400")
        for train_n in extra_train_caps:
            add_sample_size_recipe(highpos_sample_source, highpos_sample_prefix, train_n)
        # Train=200 is the current stable sample-count basin.  The 20k replay
        # bottleneck is NEG tail (Invalid/sr) while POS bank_boundary+scratch/sc
        # must stay above threshold.  Old-eval evidence showed grad-clip 0.5 can
        # lower Invalid/sr, so check tail-lowering variants directly in the
        # train=200 sample-count protocol before broad full-data sweeps.
        highpos_source = fcmpm_high_p_matrix_recipe(
            "0.5000",
            "1.0000",
            "3",
            "9",
            "0.65",
            ab_labels="0.90,1.00",
            other_label="0.0",
            raw_target=False,
            ls="0.29500",
        )
        highpos_p045_source = fcmpm_high_p_matrix_recipe(
            "0.4500",
            "1.0000",
            "3",
            "9",
            "0.65",
            ab_labels="0.90,1.00",
            other_label="0.0",
            raw_target=False,
            ls="0.29500",
        )
        highpos_cmp095_p045_source = fcmpm_high_p_matrix_recipe(
            "0.4500",
            "0.9500",
            "3",
            "9",
            "0.65",
            ab_labels="0.90,1.00",
            other_label="0.0",
            raw_target=False,
            ls="0.29500",
        )
        highpos_cmp095_p050_source = fcmpm_high_p_matrix_recipe(
            "0.5000",
            "0.9500",
            "3",
            "9",
            "0.65",
            ab_labels="0.90,1.00",
            other_label="0.0",
            raw_target=False,
            ls="0.29500",
        )
        highpos_ab080100_source = fcmpm_high_p_matrix_recipe(
            "0.5000",
            "1.0000",
            "3",
            "9",
            "0.65",
            ab_labels="0.80,1.00",
            other_label="0.0",
            raw_target=False,
            ls="0.29500",
        )
        highpos_ab090095_source = fcmpm_high_p_matrix_recipe(
            "0.5000",
            "1.0000",
            "3",
            "9",
            "0.65",
            ab_labels="0.90,0.95",
            other_label="0.0",
            raw_target=False,
            ls="0.29500",
        )
        sample_tail_train200_tags: list[str] = []
        sample_tail_train200_sources = (
            (
                "sampletail_T7_LS02950_g3_grid9_cmp10000_p05000_ab090_100_gclip05_mpos065_s7_ep10",
                recipe_with_extra(highpos_source, ("--grad-clip", "0.5")),
            ),
            (
                "sampletail_T7_LS02950_g3_grid9_cmp10000_p05000_ab090_100_gclip03_mpos065_s7_ep10",
                recipe_with_extra(highpos_source, ("--grad-clip", "0.3")),
            ),
            (
                "sampletail_T7_LS02950_g3_grid9_cmp10000_p05000_ab090_100_gclip075_mpos065_s7_ep10",
                recipe_with_extra(highpos_source, ("--grad-clip", "0.75")),
            ),
            (
                "sampletail_T7_LS02950_g3_grid9_cmp10000_p04500_ab090_100_gclip05_mpos065_s7_ep10",
                recipe_with_extra(highpos_p045_source, ("--grad-clip", "0.5")),
            ),
            (
                "sampletail_T7_LS02950_g3_grid9_cmp10000_p04500_ab090_100_mpos065_s7_ep10",
                highpos_p045_source,
            ),
            (
                "sampletail_T7_LS02950_g3_grid9_cmp09500_p04500_ab090_100_mpos065_s7_ep10",
                highpos_cmp095_p045_source,
            ),
            (
                "sampletail_T7_LS02950_g3_grid9_cmp09500_p05000_ab090_100_mpos065_s7_ep10",
                highpos_cmp095_p050_source,
            ),
            (
                "sampletail_T7_LS02950_g3_grid9_cmp10000_p05000_ab080_100_mpos065_s7_ep10",
                highpos_ab080100_source,
            ),
            (
                "sampletail_T7_LS02950_g3_grid9_cmp10000_p05000_ab090_095_gclip05_mpos065_s7_ep10",
                recipe_with_extra(highpos_ab090095_source, ("--grad-clip", "0.5")),
            ),
        )
        for tag_prefix, source in sample_tail_train200_sources:
            add_sample_size_recipe(source, tag_prefix, "200")
            sample_tail_train200_tags.append(sample_size_tags[-1])
        # Tail guard around the current best sample-size recipe (260531):
        # train=200, LS=0.295, cmp=1.0, p=0.5, A/B=0.90/1.00 raised POS min
        # strongly; remaining eval_n20000 bottleneck is Invalid/sr max_prob.
        # Put narrowly targeted neighbors before the broad grid so the next
        # sweep tests whether sr negative tail can be suppressed without losing
        # bank_boundary+scratch sc POS.
        target_tail_guard_specs = [
            ("0.30000", "0.4500", "0.9500", "0.90,1.00"),
            ("0.30000", "0.4500", "0.9500", "0.90,0.95"),
            ("0.30000", "0.4500", "0.9500", "0.85,0.95"),
            ("0.30000", "0.4500", "1.0000", "0.90,1.00"),
            ("0.30000", "0.4500", "1.0000", "0.90,0.95"),
            ("0.30500", "0.4500", "0.9500", "0.90,1.00"),
        ]
        for ls in ("0.30000", "0.30500"):
            for cutmix_p in ("0.4500", "0.5000"):
                for cmp_ls in ("0.9500", "1.0000"):
                    for ab_labels in ("0.90,1.00", "0.90,0.95", "0.85,0.95"):
                        spec = (ls, cutmix_p, cmp_ls, ab_labels)
                        if spec not in target_tail_guard_specs:
                            target_tail_guard_specs.append(spec)
        target_tail_guard_tags: list[str] = []
        for ls, cutmix_p, cmp_ls, ab_labels in target_tail_guard_specs:
            recipe = fcmpm_high_p_matrix_recipe(
                cutmix_p,
                cmp_ls,
                "3",
                "9",
                "0.65",
                ab_labels=ab_labels,
                other_label="0.0",
                raw_target=False,
                ls=ls,
            )
            explicit[recipe.tag] = recipe
            target_tail_guard_tags.append(recipe.tag)
        champion_tags = [
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_s7_ep10",
            "adapt_T7_LS02975_g3_cmp06975_p02475_pair_pbfork_scratchx2_mpos065_s7_ep10",
        ]
        if os.environ.get("CHIP_SWEEP_QUEUE_MODE") == "target_label_matrix":
            # 260601 live evidence:
            #   strong=0.80, neg=0.00 -> POS min already down to 0.666.
            #   strong=0.80, neg=0.10 -> full eval OOD FAR jumped to 17.77%.
            # Probe the small-neg interpolation band before spending time on
            # stronger down-weighting such as strong=0.70.
            target_label_core_tags = [
                tag
                for tag in target_label_matrix_tags
                if "_p05000_" in tag and "_strong070_" not in tag
            ]
            target_label_rest_tags = [tag for tag in target_label_matrix_tags if tag not in set(target_label_core_tags)]
            priority = (
                target_label_core_tags
                + target_label_refine_tags
                + target_label_rest_tags
                + sample_tail_train200_tags
                + target_tail_guard_tags
            )
        elif os.environ.get("CHIP_SWEEP_QUEUE_MODE") == "one_axis_ablation":
            priority = one_axis_ablation_tags + two_factor_ablation_tags + three_factor_ablation_tags
        else:
            priority = (
                target_label_matrix_tags
                + target_label_refine_tags
                + champion_tags
                + sample_size_tags
                + target_tail_guard_tags
                + grid_group_tags
                + p_grid_group_tags
                + cmp_grid_group_tags
                + target_comp_tags
                + high_p_matrix_tags
                + p_axis_tags
                + smooth_ab_tags
                + label_matrix_tags
                + group_label_tags
                + [tag for tag in priority if tag not in set(champion_tags)]
            )
    front = [by_tag.get(tag) or explicit[tag] for tag in priority if tag in by_tag or tag in explicit]
    if os.environ.get("CHIP_SWEEP_QUEUE_MODE") == "target_label_matrix":
        return front
    if os.environ.get("CHIP_SWEEP_QUEUE_MODE") == "one_axis_ablation":
        return front
    front_tags = {r.tag for r in front}
    return front + [r for r in plan if r.tag not in front_tags]


def cmd_base(weights: str | Path = WEIGHTS) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "chip_multilabel._train_chip_variant",
        "--num-workers",
        "0",
        "--lr",
        "1e-4",
        "--no-normal",
        "--grad-checkpointing",
        "--val-criterion",
        "margin_max",
        "--backbone-timm",
        BACKBONE,
        "--img-size",
        "384",
        "--backbone-timm-weights",
        str(weights),
    ]


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
        return int(proc.returncode)


def is_external_interrupt_rc(rc: int) -> bool:
    return rc in {-1, 4294967295}


def is_external_interrupt_status(status: str) -> bool:
    return status in {"train_fail_-1", "train_fail_4294967295", "train_interrupted_-1", "train_interrupted_4294967295"}


def newest_run_dir(out_dir: Path) -> Path | None:
    runs = sorted(
        [
            p
            for p in out_dir.iterdir()
            if p.is_dir()
            and ((p / "best_model.pth").exists() or (p / "final_epoch_model.pth").exists())
        ] if out_dir.exists() else [],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def parse_quick_eval_epochs(log_path: Path) -> list[tuple[int, float, float]]:
    if not log_path.exists():
        return []
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    out: list[tuple[int, float, float]] = []
    for m in re.finditer(
        r"\[ep\s+([0-9]+)/eval @ [^\]]+\]\s+bit_F1=([0-9.]+)\s+FAR=([0-9.]+)%",
        txt,
    ):
        ep, bit_f1, total_far = m.groups()
        out.append((int(ep), float(bit_f1), float(total_far)))
    return out


def checkpoint_candidates(out_dir: Path) -> list[tuple[str, Path]]:
    run_dir = newest_run_dir(out_dir)
    if run_dir is None:
        return []
    out: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    def add_candidate(name: str, ckpt: Path) -> None:
        resolved = ckpt.resolve() if ckpt.exists() else ckpt
        if ckpt.exists() and resolved not in seen:
            seen.add(resolved)
            out.append((name, ckpt))

    # Always include the trainer-selected best checkpoint. Quick eval is useful,
    # but its small validation split can rank late epochs above the saved best.
    add_candidate("best", run_dir / "best_model.pth")
    quick = parse_quick_eval_epochs(out_dir / "train.log")
    has_quick = bool(quick)
    quick = sorted(quick, key=lambda x: _far_first_score(x[1], x[2]), reverse=True)
    for ep, _bit_f1, _far in quick[:2]:
        if not ((_bit_f1 >= 0.985 and _far <= 5.0) or (_bit_f1 >= 0.970 and _far <= 0.5)):
            continue
        add_candidate(f"ep{ep:02d}_quick", run_dir / f"epoch_{ep:02d}_model.pth")
    if has_quick and out:
        return out
    for name, ckpt in [
        # Fixed epoch probes: historical frozen replay points at mid-training
        # checkpoints, and these are predeclared rather than chosen by eval FAR.
        ("ep05", run_dir / "epoch_05_model.pth"),
        ("ep06", run_dir / "epoch_06_model.pth"),
        ("ep08", run_dir / "epoch_08_model.pth"),
        ("final", run_dir / "final_epoch_model.pth"),
    ]:
        add_candidate(name, ckpt)
    return out


def has_promising_quick_checkpoint(out_dir: Path) -> bool:
    for _ep, bit_f1, total_far in parse_quick_eval_epochs(out_dir / "train.log"):
        if bit_f1 >= 0.970 and total_far <= 5.0:
            return True
        if bit_f1 >= 0.950 and total_far <= 0.5:
            return True
    return False


def parse_eval(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"BEST cell:\s+(\S+)\s+eval_bit_F1=([0-9.]+)\s+eval_FAR Total=([0-9.]+)% NI=([0-9.]+)% OOD=([0-9.]+)%",
        txt,
    )
    if not m:
        return {}
    cell, bit_f1, total_far, ni_far, ood_far = m.groups()
    out = {
        "cell": cell,
        "bit_F1": bit_f1,
        "NI_FAR": ni_far,
        "OOD_FAR": ood_far,
        "Total_FAR": total_far,
    }
    bits = re.search(
        r"bit_F1_by_bit:\s+bb=([0-9.]+)\s+fk=([0-9.]+)\s+sc=([0-9.]+)\s+sr=([0-9.]+)",
        txt,
    )
    if bits:
        out.update(dict(zip(["bb_F1", "fk_F1", "sc_F1", "sr_F1"], bits.groups())))
    return out


def parse_diag(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"OVERALL\s+pos_prob=([0-9.na-]+)\s+neg_prob=([0-9.na-]+)", txt)
    if not m:
        return {}
    return {"pos_prob": m.group(1), "neg_prob": m.group(2)}


def parse_pcls_rows(log_path: Path) -> list[dict[str, str]]:
    if not log_path.exists():
        return []
    rows = []
    rx = re.compile(
        r"PCLS\s+(.+?)\s+GT\[([01]+)\]\s+n=\s*([0-9]+)\s+"
        r"bank_boundary=([0-9.]+)\s+fork=([0-9.]+)\s+scratch=([0-9.]+)\s+scratch_rot=([0-9.]+)\s+"
        r"(bit_F1|FAR)=([0-9.]+)"
    )
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = rx.search(line)
        if not m:
            continue
        rows.append(
            {
                "class": m.group(1).strip(),
                "GT": m.group(2),
                "n": m.group(3),
                "bank_boundary_prob": m.group(4),
                "fork_prob": m.group(5),
                "scratch_prob": m.group(6),
                "scratch_rot_prob": m.group(7),
                "metric": m.group(8),
                "metric_value": m.group(9),
            }
        )
        tail = line[m.end():]
        for key in ["pos_min_p10", "pos_min_p50", "neg_max_p90", "neg_max_p95"]:
            qm = re.search(rf"{key}=([0-9.]+)", tail)
            if qm:
                rows[-1][key] = qm.group(1)
    return rows


def write_pcls_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "class",
        "GT",
        "n",
        "bank_boundary_prob",
        "fork_prob",
        "scratch_prob",
        "scratch_rot_prob",
        "metric",
        "metric_value",
        "pos_min_p10",
        "pos_min_p50",
        "neg_max_p90",
        "neg_max_p95",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _short_bit(name: str) -> str:
    return {
        "bank_boundary_prob": "bb",
        "fork_prob": "fk",
        "scratch_prob": "sc",
        "scratch_rot_prob": "sr",
    }[name]


def _fmt_pct(metric: str, value: str) -> str:
    v = float(value)
    if metric == "FAR":
        return f"FAR = {v * 100:5.1f}%"
    return f"bit_F1 = {v:.3f}"


def _active_min(row: dict[str, str]) -> tuple[str, float]:
    bits = [
        ("bank_boundary_prob", float(row["bank_boundary_prob"])),
        ("fork_prob", float(row["fork_prob"])),
        ("scratch_prob", float(row["scratch_prob"])),
        ("scratch_rot_prob", float(row["scratch_rot_prob"])),
    ]
    gt = row["GT"]
    active = [bits[i] for i, ch in enumerate(gt) if ch == "1"]
    return min(active, key=lambda kv: kv[1])


def _pos_or_max(row: dict[str, str]) -> str:
    bits = [
        ("bank_boundary_prob", float(row["bank_boundary_prob"])),
        ("fork_prob", float(row["fork_prob"])),
        ("scratch_prob", float(row["scratch_prob"])),
        ("scratch_rot_prob", float(row["scratch_rot_prob"])),
    ]
    gt = row["GT"]
    if "1" in gt:
        key, val = _active_min(row)
        return f"min_pos = {val:.3f} {_short_bit(key)}"
    key, val = max(bits, key=lambda kv: kv[1])
    return f"max_prob = {val:.3f} {_short_bit(key)}"


def _train_min_pos(row: dict[str, str]) -> str:
    key, val = _active_min(row)
    return f"{val:.3f} {_short_bit(key)}"


def _max_bit(row: dict[str, str]) -> tuple[str, float]:
    bits = [
        ("bank_boundary_prob", float(row["bank_boundary_prob"])),
        ("fork_prob", float(row["fork_prob"])),
        ("scratch_prob", float(row["scratch_prob"])),
        ("scratch_rot_prob", float(row["scratch_rot_prob"])),
    ]
    return max(bits, key=lambda kv: kv[1])


def pcls_gap_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    pos_rows = [r for r in _sort_pcls(rows) if "1" in r["GT"]]
    neg_rows = [r for r in _sort_pcls(rows) if "1" not in r["GT"]]
    if not pos_rows or not neg_rows:
        return ""

    pos_items = []
    for r in pos_rows:
        key, val = _active_min(r)
        pos_items.append((val, r["class"], r["GT"], _short_bit(key), r["metric"], r["metric_value"]))
    neg_items = []
    for r in neg_rows:
        key, val = _max_bit(r)
        neg_items.append((val, r["class"], r["GT"], _short_bit(key), r["metric"], r["metric_value"]))

    worst_pos = min(pos_items, key=lambda x: x[0])
    worst_neg = max(neg_items, key=lambda x: x[0])
    gap = worst_pos[0] - worst_neg[0]
    summary = (
        f"worst_pos_min={worst_pos[0]:.3f} {worst_pos[3]} @ {worst_pos[1]} "
        f"(margin_to_0.5={worst_pos[0] - 0.5:+.3f})\n"
        f"worst_neg_max={worst_neg[0]:.3f} {worst_neg[3]} @ {worst_neg[1]} "
        f"(margin_to_0.5={0.5 - worst_neg[0]:+.3f})\n"
        f"global_gap_pos_min_minus_neg_max={gap:+.3f}"
    )

    pos_table_rows = []
    for val, cls, gt, bit, metric, metric_value in sorted(pos_items, key=lambda x: x[0])[:8]:
        row = next(r for r in pos_rows if r["class"] == cls)
        pos_table_rows.append([
            cls,
            gt,
            f"{val:.3f}",
            bit,
            f"{val - 0.5:+.3f}",
            row.get("pos_min_p10", ""),
            row.get("pos_min_p50", ""),
            _fmt_pct(metric, metric_value),
        ])
    neg_table_rows = []
    for val, cls, gt, bit, metric, metric_value in sorted(neg_items, key=lambda x: x[0], reverse=True):
        row = next(r for r in neg_rows if r["class"] == cls)
        neg_table_rows.append([
            cls,
            gt,
            f"{val:.3f}",
            bit,
            f"{0.5 - val:+.3f}",
            row.get("neg_max_p90", ""),
            row.get("neg_max_p95", ""),
            _fmt_pct(metric, metric_value),
        ])

    pos_table = _aligned_table(
        ["POS class", "GT", "min_pos", "bit", "min_pos-0.5", "p10", "p50", "metric"],
        pos_table_rows,
    )
    neg_table = _aligned_table(
        ["NEG class", "GT", "max_prob", "bit", "0.5-max_prob", "p90", "p95", "metric"],
        neg_table_rows,
    )
    return (
        "## POS min / NEG max gap analysis\n\n"
        f"{summary}\n\n"
        "Worst POS min_pos rows:\n\n"
        "```\n"
        f"{pos_table}\n"
        "```\n\n"
        "NEG max_prob rows:\n\n"
        "```\n"
        f"{neg_table}\n"
        "```\n"
    )


def pcls_gap_fields(rows: list[dict[str, str]]) -> dict[str, str]:
    if not rows:
        return {}
    pos_rows = [r for r in _sort_pcls(rows) if "1" in r["GT"]]
    neg_rows = [r for r in _sort_pcls(rows) if "1" not in r["GT"]]
    if not pos_rows or not neg_rows:
        return {}
    pos_items = []
    for r in pos_rows:
        key, val = _active_min(r)
        pos_items.append((val, r["class"], _short_bit(key)))
    neg_items = []
    for r in neg_rows:
        key, val = _max_bit(r)
        neg_items.append((val, r["class"], _short_bit(key)))
    worst_pos = min(pos_items, key=lambda x: x[0])
    worst_neg = max(neg_items, key=lambda x: x[0])
    return {
        "eval_global_gap": f"{worst_pos[0] - worst_neg[0]:.3f}",
        "eval_worst_pos_class": worst_pos[1],
        "eval_worst_pos_bit": worst_pos[2],
        "eval_worst_pos_min_prob": f"{worst_pos[0]:.3f}",
        "eval_worst_neg_class": worst_neg[1],
        "eval_worst_neg_bit": worst_neg[2],
        "eval_worst_neg_max_prob": f"{worst_neg[0]:.3f}",
    }


def _sort_pcls(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: PCLS_RANK.get(r["class"], 999))


def _aligned_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_pcls_report(
    rows: list[dict[str, str]],
    md_path: Path,
    title: str,
    diag_summary: dict[str, str],
    mode: str,
    root: str,
    metrics: dict[str, str],
) -> None:
    if not rows:
        return
    rows = _sort_pcls(rows)
    pos_rows = [r for r in rows if "1" in r["GT"]]
    neg_rows = [r for r in rows if "1" not in r["GT"]]
    if mode == "train":
        display_rows = [r for r in pos_rows if r["GT"].count("1") == 1]
    else:
        display_rows = pos_rows + ([{
            "class": "=== NEG ===",
            "GT": "",
            "bank_boundary_prob": "",
            "fork_prob": "",
            "scratch_prob": "",
            "scratch_rot_prob": "",
            "metric": "",
            "metric_value": "",
        }] if neg_rows else []) + neg_rows
    if mode == "eval":
        display_rows = [{
            "class": "=== POS ===",
            "GT": "",
            "bank_boundary_prob": "",
            "fork_prob": "",
            "scratch_prob": "",
            "scratch_rot_prob": "",
            "metric": "",
            "metric_value": "",
        }] + display_rows

    table_rows: list[list[str]] = []
    for r in display_rows:
        if r["class"].startswith("==="):
            table_rows.append([r["class"], "", "", "", "", "", "", ""])
            continue
        if mode == "train":
            metric_text = f"{float(r['metric_value']):.3f}"
            last_text = _train_min_pos(r)
        else:
            metric_text = _fmt_pct(r["metric"], r["metric_value"])
            last_text = _pos_or_max(r)
        table_rows.append([
            r["class"],
            r["GT"],
            f"{float(r['bank_boundary_prob']):.3f}",
            f"{float(r['fork_prob']):.3f}",
            f"{float(r['scratch_prob']):.3f}",
            f"{float(r['scratch_rot_prob']):.3f}",
            metric_text,
            last_text,
        ])

    if mode == "train":
        headers = ["class (POS)", "GT", "bb", "fk", "sc", "sr", "bit_F1", "min_pos_prob"]
    else:
        headers = ["class (eval)", "GT", "bb", "fk", "sc", "sr", "metric", "min_pos / max_prob"]
    table = _aligned_table(
        headers,
        table_rows,
    )
    pos_prob = diag_summary.get("pos_prob", "")
    neg_prob = diag_summary.get("neg_prob", "")
    metric_line = (
        f"bit_F1={metrics.get('bit_F1', '')} "
        f"NI_FAR={metrics.get('NI_FAR', '')}% "
        f"OOD_FAR={metrics.get('OOD_FAR', '')}% "
        f"Total_FAR={metrics.get('Total_FAR', '')}% "
        f"cell={metrics.get('cell', '')}"
    ).strip()
    text = (
        f"# {title}\n\n"
        f"root={root}\n\n"
        f"{metric_line}\n\n"
        f"OVERALL pos_prob={pos_prob} neg_prob={neg_prob}\n\n"
        "```\n"
        f"{table}\n"
        "```\n"
    )
    if mode == "eval":
        gap_text = pcls_gap_analysis(rows)
        if gap_text:
            text += f"\n{gap_text}"
    md_path.write_text(text, encoding="utf-8")


def recipe_display(row: dict[str, str]) -> str:
    tag = row.get("tag", "")
    parts = []
    if row.get("variant"):
        parts.append(row["variant"])
    if row.get("LS"):
        parts.append(f"LS={row['LS']}")
    if row.get("n_groups"):
        parts.append(f"g={row['n_groups']}")
    if row.get("cmp_ls"):
        parts.append(f"cmp={row['cmp_ls']}")
    if row.get("seed"):
        parts.append(f"s={row['seed']}")
    params = " ".join(parts)
    return f"{tag} ({params})" if params else tag


def write_performance_report(
    path: Path,
    row: dict[str, str],
    train_report: Path,
    eval_report: Path,
) -> None:
    display = recipe_display(row)
    train_table = ""
    eval_table = ""
    if train_report.exists():
        train_parts = train_report.read_text(encoding="utf-8", errors="replace").split("```")
        if len(train_parts) >= 3:
            train_table = train_parts[1].strip()
    if eval_report.exists():
        eval_parts = eval_report.read_text(encoding="utf-8", errors="replace").split("```")
        if len(eval_parts) >= 3:
            eval_table = eval_parts[1].strip()
    eval_gap = ""
    if eval_report.exists():
        eval_text = eval_report.read_text(encoding="utf-8", errors="replace")
        marker = "## POS min / NEG max gap analysis"
        if marker in eval_text:
            eval_gap = eval_text[eval_text.index(marker):].strip()

    train_metrics = (
        f"train bit_F1={row.get('train_bit_F1', '')} "
        f"NI_FAR={row.get('train_NI_FAR', '')}% "
        f"OOD_FAR={row.get('train_OOD_FAR', '')}% "
        f"Total_FAR={row.get('train_Total_FAR', '')}% "
        f"pos_prob={row.get('train_pos_prob', '')} "
        f"neg_prob={row.get('train_neg_prob', '')}"
    )
    eval_metrics = (
        f"eval bit_F1={row.get('eval_bit_F1', '')} "
        f"NI_FAR={row.get('eval_NI_FAR', '')}% "
        f"OOD_FAR={row.get('eval_OOD_FAR', '')}% "
        f"Total_FAR={row.get('eval_Total_FAR', '')}% "
        f"pos_prob={row.get('eval_pos_prob', '')} "
        f"neg_prob={row.get('eval_neg_prob', '')}"
    )
    text = (
        f"# {display} performance report\n\n"
        f"dataset={row.get('dataset', '')}\n\n"
        f"train_root={row.get('train_root', '')}\n\n"
        f"eval_root={row.get('eval_root', '')}\n\n"
        f"recipe={row.get('variant', '')} LS={row.get('LS', '')} "
        f"g={row.get('n_groups', '')} cmp={row.get('cmp_ls', '')} "
        f"p={row.get('cutmix_p', '')} seed={row.get('seed', '')} "
        f"epochs={row.get('epochs', '')} ckpt={row.get('ckpt', '')} "
        f"extra={row.get('extra', '')}\n\n"
        f"sample_caps=train_cap_per_class={row.get('train_cap_per_class', '')} "
        f"train_cap_select={row.get('train_cap_select', '')} "
        f"train_eval_n_per_class={row.get('train_eval_n_per_class', '')} "
        f"eval_n_per_class={row.get('eval_n_per_class', '')} "
        f"train_diag_cap={row.get('train_diag_cap', '')} "
        f"eval_diag_cap={row.get('eval_diag_cap', '')} "
        f"sample_seed={row.get('sample_seed', '')}\n\n"
        "metric_note=run_stage1 train/eval metrics use the stratified eval split "
        "from each requested n_per_class; pcls probability tables use the full "
        "train_diag_cap/eval_diag_cap sample.\n\n"
        f"{train_metrics}\n\n"
        f"{display} -- TRAIN (4 single class):\n\n"
        "```\n"
        f"{train_table}\n"
        "```\n\n"
        f"{eval_metrics}\n\n"
        f"{display} -- EVAL per-class 4-bit prob "
        "(POS = single+combo, NEG = Normal/Invalid/OOD):\n\n"
        "```\n"
        f"{eval_table}\n"
        "```\n\n"
        f"{eval_gap}\n"
    )
    path.write_text(text, encoding="utf-8")


def append_leaderboard(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "dataset",
        "train_root",
        "eval_root",
        "tag",
        "variant",
        "LS",
        "n_groups",
        "cmp_ls",
        "cutmix_p",
        "seed",
        "extra",
        "train_cap_per_class",
        "train_cap_select",
        "eval_n_per_class",
        "train_eval_n_per_class",
        "train_diag_cap",
        "eval_diag_cap",
        "sample_seed",
        "epochs",
        "batch",
        "accum",
        "status",
        "ckpt",
        "train_cell",
        "train_bit_F1",
        "train_NI_FAR",
        "train_OOD_FAR",
        "train_Total_FAR",
        "train_pos_prob",
        "train_neg_prob",
        "eval_cell",
        "eval_bit_F1",
        "eval_NI_FAR",
        "eval_OOD_FAR",
        "eval_Total_FAR",
        "eval_bb_F1",
        "eval_fk_F1",
        "eval_sc_F1",
        "eval_sr_F1",
        "eval_pos_prob",
        "eval_neg_prob",
        "eval_global_gap",
        "eval_worst_pos_class",
        "eval_worst_pos_bit",
        "eval_worst_pos_min_prob",
        "eval_worst_neg_class",
        "eval_worst_neg_bit",
        "eval_worst_neg_max_prob",
        "train_pcls_report",
        "eval_pcls_report",
        "performance_report",
        "model",
        "out_dir",
    ]
    exists = path.exists()
    if exists:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            existing_rows = list(csv.DictReader(f))
            old_fields = list(existing_rows[0].keys()) if existing_rows else []
        if old_fields and old_fields != fields:
            with path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for old_row in existing_rows:
                    w.writerow({k: old_row.get(k, "") for k in fields})
            print(f"[sweep] migrated leaderboard header in-place -> {path}", flush=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def read_leaderboard(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _far_first_score(bit_f1: float, total_far: float) -> float:
    far0_bonus = 0.01 if bit_f1 >= 0.99 and total_far <= 0.0001 else 0.0
    return bit_f1 + far0_bonus - 0.02 * total_far


def _pcls_global_gap(path: str) -> float | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    txt = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"global_gap_pos_min_minus_neg_max=([+-]?[0-9.]+)", txt)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _row_gap(row: dict[str, str]) -> float:
    gap = _pcls_global_gap(row.get("eval_pcls_report", ""))
    if gap is not None:
        return gap
    return _f(row, "eval_pos_prob") - _f(row, "eval_neg_prob")


def _gap_first_score(row: dict[str, str]) -> float:
    bit_f1 = _f(row, "eval_bit_F1")
    total_far = _f(row, "eval_Total_FAR", 100.0)
    gap = _row_gap(row)
    far0_bonus = 0.02 if bit_f1 >= 0.99 and total_far <= 0.0001 else 0.0
    return bit_f1 + far0_bonus + 0.25 * gap - 0.02 * total_far


def _sweep_token(v: float, ndigits: int = 2) -> str:
    iv = int(round(v * (10 ** ndigits)))
    return f"{iv:0{ndigits + 1}d}"


def _param_token(v: str | float, min_decimals: int = 2, max_decimals: int = 5) -> str:
    txt = f"{float(v):.{max_decimals}f}".rstrip("0").rstrip(".")
    if "." not in txt:
        whole, frac = txt, ""
    else:
        whole, frac = txt.split(".", 1)
    return f"{whole}{frac.ljust(min_decimals, '0')}"


def _fmt_param(v: float, min_decimals: int = 2, max_decimals: int = 5) -> str:
    txt = f"{v:.{max_decimals}f}".rstrip("0").rstrip(".")
    if "." not in txt:
        return f"{txt}.{'0' * min_decimals}"
    whole, frac = txt.split(".", 1)
    return f"{whole}.{frac.ljust(min_decimals, '0')}"


COMPLEMENT_NOOP_FLAGS_WITH_VALUE = {
    "--cutmix-pair-fill",
    "--cutmix-grid-prob",
    "--cutmix-grid-k",
    "--cutmix-n-patches",
    "--cutmix-total-ratio",
    "--cutmix-discount",
    "--cutmix-alpha",
    # This sweep owns complement mode. Mode overrides make tags hard to
    # compare and were pruned from the canonical old-eval evidence.
    "--cutmix-mode",
}
COMPLEMENT_NOOP_FLAGS = {
    "--cutmix-rect",
}


def _sanitize_complement_extra(extra: tuple[str, ...] | str) -> tuple[str, ...]:
    parts = tuple(extra.split()) if isinstance(extra, str) else tuple(extra)
    out: list[str] = []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok in COMPLEMENT_NOOP_FLAGS_WITH_VALUE:
            i += 2
            continue
        if tok in COMPLEMENT_NOOP_FLAGS:
            i += 1
            continue
        out.append(tok)
        i += 1
    return tuple(out)


def _pair_key(extra: tuple[str, ...] | str) -> str:
    parts = list(_sanitize_complement_extra(extra))
    for i, tok in enumerate(parts[:-1]):
        if tok == "--cutmix-pair":
            if parts[i + 1] == "none":
                return "nopair"
            if parts[i + 1] == "masked2":
                return "pair2"
            return "pair"
    return "pair"


def _extra_value(extra: tuple[str, ...] | str, flag: str) -> str:
    parts = extra.split() if isinstance(extra, str) else list(extra)
    for i, tok in enumerate(parts[:-1]):
        if tok == flag:
            return parts[i + 1]
    return ""


def _invalid_group_grid(groups: str, extra: tuple[str, ...] | str) -> str:
    grid_dim = _extra_value(extra, "--cutmix-grid-dim") or "8"
    try:
        g = int(float(groups))
        grid = int(float(grid_dim))
    except ValueError:
        return f"g={groups}, grid={grid_dim}"
    if g <= 0 or grid <= 0 or grid % g != 0:
        return f"g={g}, grid={grid_dim}"
    return ""


def _replace_extra_value(extra: tuple[str, ...] | str, flag: str, value: str) -> tuple[str, ...]:
    parts = extra.split() if isinstance(extra, str) else list(extra)
    for i, tok in enumerate(parts[:-1]):
        if tok == flag:
            parts[i + 1] = value
            return tuple(parts)
    return tuple(parts + [flag, value])


def _target_key(extra: tuple[str, ...] | str) -> str:
    raw_parts = extra.split() if isinstance(extra, str) else list(extra)
    bce_temp = _extra_value(raw_parts, "--bce-temperature")
    mask_pos_target = _extra_value(raw_parts, "--cutmix-mask-pos-target")
    mask_neg_target = _extra_value(raw_parts, "--cutmix-mask-neg-target")
    drop_path_rate = _extra_value(raw_parts, "--drop-path-rate")
    kd_teacher = _extra_value(raw_parts, "--kd-teacher-probs")
    kd_alpha = _extra_value(raw_parts, "--kd-alpha")
    kd_temp = _extra_value(raw_parts, "--kd-temperature")
    kd_skip = "kdskipcm" if "--kd-skip-on-cutmix" in raw_parts else ""
    extra = _sanitize_complement_extra(extra)
    pos = _extra_value(extra, "--pos-target")
    neg = _extra_value(extra, "--neg-target")
    pos_asy = _extra_value(extra, "--pos-targets-asy")
    neg_asy = _extra_value(extra, "--neg-targets-asy")
    ab = _extra_value(extra, "--cutmix-ab-labels")
    other = _extra_value(extra, "--cutmix-other-label")
    grid_dim = _extra_value(extra, "--cutmix-grid-dim")
    best_from_ep = _extra_value(extra, "--best-from-epoch")
    warmup_ep = _extra_value(extra, "--warmup-epochs")
    eta_min = _extra_value(extra, "--lr-eta-min")
    ema_decay = _extra_value(extra, "--ema-decay")
    grad_clip = _extra_value(extra, "--grad-clip")
    pair_loss_w = _extra_value(extra, "--cutmix-pair-loss-w")
    pair_bias = _extra_value(extra, "--cutmix-pair-bias")
    pos_weight = _extra_value(extra, "--pos-weight")
    area_prop = "area" if "--cutmix-label-area-prop" in (extra if isinstance(extra, str) else " ".join(extra)) else ""
    raw = "rawmix" if "--cutmix-mix-raw-target" in (extra if isinstance(extra, str) else " ".join(extra)) else ""
    det = "det" if "--deterministic" in (extra if isinstance(extra, str) else " ".join(extra)) else ""
    vals = []
    if pos or neg:
        vals.append(f"pt{_param_token(pos or 0)}nt{_param_token(neg or 0)}")
    if pos_asy or neg_asy:
        vals.append(f"ptasy{pos_asy.replace(',', '_')}ntasy{neg_asy.replace(',', '_')}")
    if ab:
        vals.append(f"ab{ab.replace(',', '_').replace('.', '')}")
    if other:
        vals.append(f"ol{_param_token(other)}")
    if grid_dim:
        vals.append(f"grid{grid_dim}")
    if best_from_ep:
        vals.append(f"bfep{best_from_ep}")
    if warmup_ep:
        vals.append(f"warm{warmup_ep}")
    if eta_min:
        vals.append(f"eta{eta_min.replace('.', 'p').replace('-', 'm')}")
    if ema_decay:
        vals.append(f"ema{_param_token(ema_decay)}")
    if grad_clip:
        vals.append(f"gclip{_param_token(grad_clip)}")
    if pair_loss_w:
        vals.append(f"plw{_param_token(pair_loss_w)}")
    if pair_bias:
        vals.append(f"pb{pair_bias.replace(',', '_').replace(':', 'x').replace('.', 'p')}")
    if pos_weight:
        vals.append(f"pw{pos_weight.replace(',', '_').replace(':', 'x').replace('.', 'p')}")
    if bce_temp:
        vals.append(f"bceT{_param_token(bce_temp)}")
    if mask_pos_target:
        vals.append(f"mpos{_param_token(mask_pos_target)}")
    if mask_neg_target:
        vals.append(f"mneg{_param_token(mask_neg_target)}")
    if drop_path_rate:
        vals.append(f"dpr{_param_token(drop_path_rate)}")
    if kd_teacher:
        vals.append(f"kd{Path(kd_teacher).stem}")
    if kd_alpha:
        vals.append(f"kda{_param_token(kd_alpha)}")
    if kd_temp:
        vals.append(f"kdt{_param_token(kd_temp)}")
    if kd_skip:
        vals.append(kd_skip)
    if area_prop:
        vals.append(area_prop)
    if raw:
        vals.append(raw)
    if det:
        vals.append(det)
    return "_".join(vals) if vals else "base"


def _recipe_param_key(recipe: Recipe) -> tuple[str, float, str, float, float, str, str, str]:
    return (
        recipe.variant,
        round(float(recipe.ls), 5),
        recipe.groups,
        round(float(recipe.cmp_ls), 5),
        round(float(recipe.cutmix_p), 5),
        recipe.seed,
        _pair_key(recipe.extra),
        _target_key(recipe.extra),
    )


def _row_param_key(row: dict[str, str]) -> tuple[str, float, str, float, float, str, str, str] | None:
    try:
        return (
            row.get("variant", ""),
            round(float(row.get("LS", "")), 5),
            row.get("n_groups", ""),
            round(float(row.get("cmp_ls", "")), 5),
            round(float(row.get("cutmix_p", "")), 5),
            row.get("seed", ""),
            _pair_key(row.get("extra", "")),
            _target_key(row.get("extra", "")),
        )
    except ValueError:
        return None


def _tag_for_recipe(r: Recipe) -> str:
    pair = _pair_key(r.extra)
    target = _target_key(r.extra)
    target_suffix = "" if target == "base" else f"_{target}"
    return (
        f"adapt_{r.variant}_LS{_param_token(r.ls)}_g{r.groups}_"
        f"cmp{_param_token(r.cmp_ls)}_p{_param_token(r.cutmix_p)}_"
        f"{pair}{target_suffix}_s{r.seed}_ep{r.epochs}"
    )


def existing_tags(ds: DatasetSpec) -> set[str]:
    lead = REPO / "outputs" / ds.name / "_leaderboard.csv"
    tags = {r.get("tag", "") for r in read_leaderboard(lead)}
    out_root = REPO / "outputs" / ds.name
    if out_root.exists():
        tags.update(
            p.name
            for p in out_root.iterdir()
            if p.is_dir() and (p / "performance_report.md").is_file()
        )
    return {t for t in tags if t}


def adaptive_recipes(ds: DatasetSpec, limit: int) -> list[Recipe]:
    """Generate a compact next batch around the best completed rows.

    This is deliberately conservative: discard collapse regions and expand only
    around rows with good bit_F1/FAR balance.  Seed repeats are only scheduled
    for already-promising hparams.
    """
    lead = REPO / "outputs" / ds.name / "_leaderboard.csv"
    rows = [
        r
        for r in read_leaderboard(lead)
        if r.get("status") == "done"
        and r.get("variant") not in {"checkpoint_replay", "vote_majority_bits"}
        and r.get("LS")
        and r.get("n_groups")
        and r.get("cmp_ls")
    ]
    if not rows:
        return []
    scored = sorted(
        rows,
        key=lambda r: (_gap_first_score(r), _row_gap(r), _f(r, "eval_bit_F1")),
        reverse=True,
    )
    keep = []
    keep_keys: set[tuple[str, str, str, str, str, str]] = set()
    for r in scored:
        bit_f1 = _f(r, "eval_bit_F1")
        far = _f(r, "eval_Total_FAR", 100.0)
        gap = _row_gap(r)
        key = (
            r.get("variant", ""),
            r.get("LS", ""),
            r.get("n_groups", ""),
            r.get("cmp_ls", ""),
            r.get("cutmix_p", ""),
            r.get("seed", ""),
        )
        if key in keep_keys:
            continue
        if bit_f1 >= 0.985 and far <= 5.0 and gap >= 0.15:
            keep.append(r)
            keep_keys.add(key)
        elif bit_f1 >= 0.970 and far <= 0.5 and gap >= 0.15:
            keep.append(r)
            keep_keys.add(key)
        if len(keep) >= 5:
            break
    if not keep:
        return historical_grid_recipes(ds, limit)

    seen = existing_tags(ds)
    if ds.name == "frozen_iter116J_orig814_old_eval":
        seen.update(LIVE_PRUNED_TAGS)
    seen_param_keys = {k for k in (_row_param_key(r) for r in read_leaderboard(lead)) if k is not None}
    out: list[Recipe] = []
    for r in keep:
        variant = r.get("variant", "T7") or "T7"
        if variant == "checkpoint_replay" or not r.get("LS") or not r.get("n_groups") or not r.get("cmp_ls"):
            continue
        base_ls = _f(r, "LS", 0.30)
        base_g = int(round(_f(r, "n_groups", 3)))
        base_cmp = _f(r, "cmp_ls", 0.5)
        base_p = _f(r, "cutmix_p", 0.25)
        base_epochs = r.get("epochs") or "10"
        base_batch = r.get("batch") or "2"
        base_accum = r.get("accum") or "8"
        base_extra = _sanitize_complement_extra(tuple((r.get("extra") or "").split()))

        raw_candidates: list[Recipe] = []
        for cmp_ls, p in {
            (base_cmp + 0.0025, base_p - 0.0025),
            (base_cmp - 0.0025, base_p + 0.0025),
        }:
            if 0.685 <= cmp_ls <= 0.715 and 0.24 <= p <= 0.26:
                raw_candidates.append(Recipe("", variant, _fmt_param(base_ls, 3), str(base_g), _fmt_param(cmp_ls, 4), _fmt_param(p, 4), r.get("seed") or "1", base_extra, base_epochs, base_batch, base_accum))

        pair_bias = _extra_value(base_extra, "--cutmix-pair-bias")
        if (
            pair_bias == "fork,scratch:2"
            and 0.296 <= base_ls <= 0.299
            and base_g == 3
            and 0.695 <= base_cmp <= 0.705
            and 0.247 <= base_p <= 0.253
            and (r.get("seed") or "1") == "7"
            and _f(r, "eval_bit_F1") >= 0.995
            and _f(r, "eval_Total_FAR", 100.0) <= 0.5
        ):
            # New SOTA basin still has Normal/Starburst scratch-tail pressure in
            # pcls, so split cmp/p/LS nudges instead of only trading cmp against p.
            for ls, cmp_ls, p, extra in [
                (base_ls, base_cmp - 0.0025, base_p, base_extra),
                (base_ls, base_cmp, base_p - 0.0025, base_extra),
                (base_ls, base_cmp - 0.0025, base_p - 0.0025, base_extra),
                (base_ls, base_cmp + 0.00125, base_p, base_extra),
                (base_ls, base_cmp, base_p + 0.00125, base_extra),
                (base_ls, base_cmp + 0.00125, base_p + 0.00125, base_extra),
                (base_ls, base_cmp - 0.00125, base_p, base_extra),
                (base_ls, base_cmp, base_p - 0.00125, base_extra),
                (base_ls, base_cmp - 0.00125, base_p - 0.00125, base_extra),
                (base_ls + 0.00125, base_cmp, base_p, base_extra),
                (base_ls + 0.00125, base_cmp - 0.0025, base_p, base_extra),
                (base_ls, base_cmp, base_p, base_extra + ("--bce-temperature", "1.03")),
            ]:
                if 0.292 <= ls <= 0.300 and 0.685 <= cmp_ls <= 0.715 and 0.24 <= p <= 0.26:
                    raw_candidates.append(
                        Recipe(
                            "",
                            variant,
                            _fmt_param(ls, 5),
                            str(base_g),
                            _fmt_param(cmp_ls, 4),
                            _fmt_param(p, 4),
                            r.get("seed") or "1",
                            extra,
                            base_epochs,
                            base_batch,
                            base_accum,
                        )
                    )
        if (
            _f(r, "eval_bit_F1") >= 0.997
            and 0.0 < _f(r, "eval_Total_FAR", 100.0) <= 1.0
            and _row_gap(r) >= 0.15
        ):
            # High-gap but nonzero-FAR rows are useful: keep the good positive
            # separation and lower the tail with smaller cmp/p or slightly more LS.
            for ls, cmp_ls, p in [
                (base_ls, base_cmp - 0.0025, base_p),
                (base_ls, base_cmp, base_p - 0.0025),
                (base_ls, base_cmp - 0.0025, base_p - 0.0025),
                (base_ls + 0.00125, base_cmp, base_p),
                (base_ls + 0.00125, base_cmp - 0.0025, base_p),
            ]:
                if 0.292 <= ls <= 0.301 and 0.685 <= cmp_ls <= 0.715 and 0.24 <= p <= 0.26:
                    raw_candidates.append(
                        Recipe(
                            "",
                            variant,
                            _fmt_param(ls, 5),
                            str(base_g),
                            _fmt_param(cmp_ls, 4),
                            _fmt_param(p, 4),
                            r.get("seed") or "1",
                            base_extra,
                            base_epochs,
                            base_batch,
                            base_accum,
                        )
                    )
        if (
            _target_key(base_extra) == "base"
            and 0.292 <= base_ls <= 0.300
            and base_g == 3
            and 0.69 <= base_cmp <= 0.705
            and 0.245 <= base_p <= 0.255
            and (r.get("seed") or "1") == "7"
            and _f(r, "eval_bit_F1") >= 0.990
            and _f(r, "eval_Total_FAR", 100.0) <= 0.5
        ):
            pair_gap_extras = [
                ("--cutmix-label-area-prop",),
                ("--cutmix-grid-dim", "3"),
                ("--cutmix-grid-dim", "6"),
                ("--cutmix-grid-dim", "9"),
            ]
            for target_extra in pair_gap_extras:
                raw_candidates.append(
                    Recipe(
                        "",
                        variant,
                        _fmt_param(base_ls, 3),
                        str(base_g),
                        _fmt_param(base_cmp, 3),
                        _fmt_param(base_p, 3),
                        r.get("seed") or "1",
                        base_extra + target_extra,
                        base_epochs,
                        base_batch,
                        base_accum,
                    )
                )
        if _f(r, "eval_bit_F1") >= 0.995 and _f(r, "eval_Total_FAR", 100.0) <= 1.0:
            for seed in ["1", "7", "13"]:
                raw_candidates.append(Recipe("", variant, _fmt_param(base_ls, 3), str(base_g), _fmt_param(base_cmp, 3), _fmt_param(base_p, 3), seed, base_extra, base_epochs, base_batch, base_accum))

        for cand in raw_candidates:
            tag = _tag_for_recipe(cand)
            if is_live_pruned_for_dataset(ds.name, tag) or tag in seen:
                continue
            param_key = _recipe_param_key(cand)
            if param_key in seen_param_keys:
                continue
            seen_param_keys.add(param_key)
            seen.add(tag)
            out.append(Recipe(tag, cand.variant, cand.ls, cand.groups, cand.cmp_ls, cand.cutmix_p, cand.seed, cand.extra, cand.epochs, cand.batch, cand.accum))
            if len(out) >= limit:
                return out
    if out:
        return out
    return historical_grid_recipes(ds, limit)


def historical_grid_recipes(ds: DatasetSpec, limit: int) -> list[Recipe]:
    """Fallback queue when no single fresh run has passed the adaptive gate.

    The ordering starts from prior strong regions: fcm_margin, iter116J,
    iter26B/iter26D, then iter25-style lower LS/g=2.  Tags encode the real
    condition so reports do not need synthetic shorthand labels.
    """
    seen = existing_tags(ds)
    if ds.name == "frozen_iter116J_orig814_old_eval":
        seen.update(LIVE_PRUNED_TAGS)
    lead = REPO / "outputs" / ds.name / "_leaderboard.csv"
    seen_param_keys = {k for k in (_row_param_key(r) for r in read_leaderboard(lead)) if k is not None}
    out: list[Recipe] = []
    ls_values = ["0.30", "0.40", "0.20", "0.25", "0.35", "0.45", "0.50"]
    group_values = ["3", "4", "2"]
    # cmp=0.5 broad grid repeatedly failed on the canonical old-eval loop.
    # Keep compute on fcm_margin-like and neighboring complement strengths.
    cmp_values = ["0.7", "0.6", "0.8", "0.3", "1.0"]
    # p=0.10 was empirically too weak on the canonical old-eval loop; keep
    # fallback compute near prior useful cutmix rates instead.
    p_values = ["0.25", "0.20", "0.35"]
    seed_values = ["1", "42", "7", "77"]
    pair_options = [
        ("nopair", ("--cutmix-pair", "none")),
        ("pair", ()),
    ]
    if ds.name == "frozen_original":
        subset_options = [
            ("all", ()),
            ("cap200", ("--max-per-class-defect", "200")),
            ("oldest200", ("--max-per-class-defect", "200", "--max-per-class-defect-select", "oldest")),
            ("newest200", ("--max-per-class-defect", "200", "--max-per-class-defect-select", "newest")),
        ]
    else:
        subset_options = [("all", ())]

    def add_recipe(recipe: Recipe) -> bool:
        param_key = _recipe_param_key(recipe)
        if is_live_pruned_for_dataset(ds.name, recipe.tag) or recipe.tag in seen or param_key in seen_param_keys:
            return False
        seen_param_keys.add(param_key)
        seen.add(recipe.tag)
        out.append(recipe)
        return len(out) >= limit

    if ds.name == "frozen_iter116J_orig814_old_eval":
        i10_save = ("--sweep-eval-variants", "I10", "--sweep-save-every-epoch")
        focused: list[Recipe] = []
        for cmp_ls in ["0.615", "0.625", "0.635", "0.645", "0.650"]:
            focused.append(
                Recipe(
                    f"iter116J_T7_g3_cmp{_param_token(cmp_ls)}_p020_s1_ep10_I10only_partner_push",
                    "T7",
                    "0.30",
                    "3",
                    cmp_ls,
                    "0.20",
                    "1",
                    i10_save,
                )
            )
        for cutmix_p in ["0.19", "0.18", "0.21", "0.22"]:
            focused.append(
                Recipe(
                    f"iter116J_T7_g3_cmp060_p{_param_token(cutmix_p)}_s1_ep10_I10only_partner_push",
                    "T7",
                    "0.30",
                    "3",
                    "0.60",
                    cutmix_p,
                    "1",
                    i10_save,
                )
            )
        focused.extend(
            [
                Recipe(
                    "histgrid_all_T7_LS030_g3_cmp100_p020_pair_s1_ep10_saveevery_bfep4",
                    "T7",
                    "0.30",
                    "3",
                    "1.0",
                    "0.20",
                    "1",
                    i10_save + ("--best-from-epoch", "4"),
                ),
                Recipe("iter116J_T7_g4_cmp060_p020_s1_ep10_I10only_partner_push", "T7", "0.30", "4", "0.60", "0.20", "1", i10_save),
                Recipe("iter116J_T7_LS032_g3_cmp060_p020_s1_ep10_I10only_partner_push", "T7", "0.32", "3", "0.60", "0.20", "1", i10_save),
                Recipe("iter116J_T7_LS028_g3_cmp060_p020_s1_ep10_I10only_partner_push", "T7", "0.28", "3", "0.60", "0.20", "1", i10_save),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp0701_p0249_s7_ep10_tail_gap", "T7", "0.295", "3", "0.701", "0.249", "7", i10_save),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp0699_p0251_s7_ep10_combo_gap", "T7", "0.295", "3", "0.699", "0.251", "7", i10_save),
                Recipe("fcm_margin_pair_T7_LS02945_g3_cmp07_p025_s7_ep10_combo_gap", "T7", "0.2945", "3", "0.70", "0.25", "7", i10_save),
                Recipe("fcm_margin_pair_T7_LS02955_g3_cmp07_p025_s7_ep10_tail_gap", "T7", "0.2955", "3", "0.70", "0.25", "7", i10_save),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p0249_s7_ep10_tail_gap", "T7", "0.295", "3", "0.70", "0.249", "7", i10_save),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p0251_s7_ep10_combo_gap", "T7", "0.295", "3", "0.70", "0.251", "7", i10_save),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_det_ep10_stability", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--deterministic",)),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_ema095_ep10_stability", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--ema-decay", "0.95")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_warm1_eta1e6_ep10_stability", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--warmup-epochs", "1", "--lr-eta-min", "0.000001")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_gclip05_ep10_stability", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--grad-clip", "0.5")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss075_ep10_tail_gap", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-loss-w", "0.75")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_bceT105_ep10_gap", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--bce-temperature", "1.05")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_bceT110_ep10_gap", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--bce-temperature", "1.10")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s23_ep10_seedcheck", "T7", "0.295", "3", "0.70", "0.25", "23", i10_save),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_bceT095_ep10_invalid_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--bce-temperature", "0.95")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss125_ep10_invalid_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-loss-w", "1.25")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_dpr002_ep10_tail_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--drop-path-rate", "0.02")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_bceT095_ep10_combo_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:2", "--bce-temperature", "0.95")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_masked2_w050_ep10_tail_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair", "masked2", "--cutmix-pair-loss-w", "0.50")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss125_maskpos070_ep10_tail_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-loss-w", "1.25", "--cutmix-mask-pos-target", "0.70")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairloss125_maskpos070_apply_ep10_tail_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-loss-w", "1.25", "--cutmix-mask-pos-target", "0.70")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_rawmix_ab072072_pairloss125_ep10_combo_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-mix-raw-target", "--cutmix-ab-labels", "0.72,0.72", "--cutmix-pair-loss-w", "1.25")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_gap", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:2")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07025_p025_s7_pairbias_fork_scratch2_ep10_combo_lift", "T7", "0.295", "3", "0.7025", "0.25", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:2")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p02525_s7_pairbias_fork_scratch2_ep10_combo_lift", "T7", "0.295", "3", "0.70", "0.2525", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:2")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07025_p02525_s7_pairbias_fork_scratch2_ep10_combo_lift", "T7", "0.295", "3", "0.7025", "0.2525", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:2")),
                Recipe("fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_pairbias_fork_scratch2_ep10_combo_lift", "T7", "0.2975", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:2")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch3_ep10_combo_gap", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:3")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch1_ep10_far_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:1")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p0245_s7_pairbias_fork_scratch2_ep10_far_guard", "T7", "0.295", "3", "0.70", "0.245", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:2")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_bank_scratch2_ep10_combo_gap", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-bias", "bank_boundary,scratch:2")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_posw_fork103_scratch103_ep10_combo_gap", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--pos-weight", "fork:1.03,scratch:1.03")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_posw_fork105_scratch105_ep10_combo_gap", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--pos-weight", "fork:1.05,scratch:1.05")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_scratch_scratchrot2_ep10_sr_combo_gap", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-bias", "scratch,scratch_rot:2")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_bank_scratchrot2_ep10_sr_combo_gap", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-bias", "bank_boundary,scratch_rot:2")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_pairloss075_ep10_far_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:2", "--cutmix-pair-loss-w", "0.75")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp06975_p02525_s7_gclip05_ep10_combo_balance", "T7", "0.295", "3", "0.6975", "0.2525", "7", i10_save + ("--grad-clip", "0.5")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_pairbias_fork_scratch2_gclip05_ep10_far_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--cutmix-pair-bias", "fork,scratch:2", "--grad-clip", "0.5")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_posw_fork101_scratch101_gclip05_ep10_combo_guard", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--pos-weight", "fork:1.01,scratch:1.01", "--grad-clip", "0.5")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07025_p025_s7_gclip05_ep10_combo_balance", "T7", "0.295", "3", "0.7025", "0.25", "7", i10_save + ("--grad-clip", "0.5")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p02525_s7_gclip05_ep10_combo_balance", "T7", "0.295", "3", "0.70", "0.2525", "7", i10_save + ("--grad-clip", "0.5")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07025_p02475_s7_gclip05_ep10_tail_balance", "T7", "0.295", "3", "0.7025", "0.2475", "7", i10_save + ("--grad-clip", "0.5")),
                Recipe("fcm_margin_pair_T7_LS02975_g3_cmp07_p025_s7_gclip05_ep10_tail_balance", "T7", "0.2975", "3", "0.70", "0.25", "7", i10_save + ("--grad-clip", "0.5")),
                Recipe("fcm_margin_pair_T7_LS02925_g3_cmp07_p025_s7_gclip05_ep10_combo_balance", "T7", "0.2925", "3", "0.70", "0.25", "7", i10_save + ("--grad-clip", "0.5")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_gclip075_ep10_stability", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--grad-clip", "0.75")),
                Recipe("fcm_margin_pair_T7_LS0295_g3_cmp07_p025_s7_ema095_warm1_ep10_stability", "T7", "0.295", "3", "0.70", "0.25", "7", i10_save + ("--ema-decay", "0.95", "--warmup-epochs", "1", "--lr-eta-min", "0.000001")),
            ]
        )
        for recipe in focused:
            if add_recipe(recipe):
                return out

    for subset_name, subset_extra in subset_options:
        for ls in ls_values:
            for groups in group_values:
                for cmp_ls in cmp_values:
                    for cutmix_p in p_values:
                        for pair_name, pair_extra in pair_options:
                            for seed in seed_values:
                                extra = tuple(subset_extra) + tuple(pair_extra)
                                tag = (
                                    f"histgrid_{subset_name}_T7_LS{_sweep_token(float(ls))}_"
                                    f"g{groups}_cmp{_sweep_token(float(cmp_ls))}_"
                                    f"p{_sweep_token(float(cutmix_p))}_{pair_name}_s{seed}_ep10"
                                )
                                recipe = Recipe(tag, "T7", ls, groups, cmp_ls, cutmix_p, seed, extra)
                                param_key = _recipe_param_key(recipe)
                                if is_live_pruned_for_dataset(ds.name, tag) or tag in seen or param_key in seen_param_keys:
                                    continue
                                seen_param_keys.add(param_key)
                                seen.add(tag)
                                out.append(recipe)
                                if len(out) >= limit:
                                    return out
    return out


def eval_cmd(
    model: Path,
    root: str,
    out_root: Path,
    n_per_class: int,
    variants: str = "I10,I13",
    sample_seed: str = "42",
) -> list[str]:
    batch_size = os.environ.get("CHIP_EVAL_BATCH_SIZE", "64")
    return [
        sys.executable,
        "-u",
        "-m",
        "chip_multilabel.run_stage1",
        "--model",
        str(model),
        "--eval-set",
        root,
        "--out-root",
        str(out_root),
        "--variants",
        variants,
        "--n-per-class",
        str(n_per_class),
        "--batch-size",
        batch_size,
        "--num-workers",
        "0",
        "--strength-min",
        "0.0",
        "--strength-max",
        "1.0",
        "--seed",
        str(sample_seed),
        "--sample-seed",
        str(sample_seed),
    ]


def diag_cmd(model: Path, root: str, tag: str, cap: int, device: str, sample_seed: str = "42") -> list[str]:
    batch_size = os.environ.get("CHIP_PCLS_BATCH_SIZE", "48")
    return [
        sys.executable,
        "-u",
        "-m",
        "chip_multilabel._posneg_prob_diag",
        "--model",
        str(model),
        "--root",
        root,
        "--device",
        device,
        "--batch-size",
        batch_size,
        "--cap-per-class",
        str(cap),
        "--sample-seed",
        str(sample_seed),
        "--tag",
        tag,
    ]


def split_sweep_extra(
    extra: tuple[str, ...],
) -> tuple[bool, bool, str | Path, str, int | None, int | None, int | None, int | None, int | None, list[int], list[str]]:
    disable_grad_checkpointing = False
    save_every_epoch = False
    weights: str | Path = WEIGHTS
    eval_variants = "I10,I13"
    eval_n_per_class: int | None = None
    train_eval_n_per_class: int | None = None
    diag_cap: int | None = None
    train_diag_cap: int | None = None
    eval_diag_cap: int | None = None
    extra_eval_caps: list[int] = []
    train_extra: list[str] = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok == "--sweep-no-grad-checkpointing":
            disable_grad_checkpointing = True
            i += 1
        elif tok == "--sweep-save-every-epoch":
            save_every_epoch = True
            i += 1
        elif tok == "--sweep-weights":
            if i + 1 >= len(extra):
                raise ValueError("--sweep-weights requires a path")
            weights = extra[i + 1]
            i += 2
        elif tok == "--sweep-eval-variants":
            if i + 1 >= len(extra):
                raise ValueError("--sweep-eval-variants requires a value")
            eval_variants = extra[i + 1]
            i += 2
        elif tok == "--sweep-eval-n-per-class":
            if i + 1 >= len(extra):
                raise ValueError("--sweep-eval-n-per-class requires a value")
            eval_n_per_class = int(extra[i + 1])
            i += 2
        elif tok == "--sweep-train-eval-n-per-class":
            if i + 1 >= len(extra):
                raise ValueError("--sweep-train-eval-n-per-class requires a value")
            train_eval_n_per_class = int(extra[i + 1])
            i += 2
        elif tok == "--sweep-diag-cap":
            if i + 1 >= len(extra):
                raise ValueError("--sweep-diag-cap requires a value")
            diag_cap = int(extra[i + 1])
            i += 2
        elif tok == "--sweep-train-diag-cap":
            if i + 1 >= len(extra):
                raise ValueError("--sweep-train-diag-cap requires a value")
            train_diag_cap = int(extra[i + 1])
            i += 2
        elif tok == "--sweep-eval-diag-cap":
            if i + 1 >= len(extra):
                raise ValueError("--sweep-eval-diag-cap requires a value")
            eval_diag_cap = int(extra[i + 1])
            i += 2
        elif tok == "--sweep-extra-eval-caps":
            if i + 1 >= len(extra):
                raise ValueError("--sweep-extra-eval-caps requires a value")
            extra_eval_caps = [int(x.strip()) for x in extra[i + 1].split(",") if x.strip()]
            i += 2
        else:
            train_extra.append(tok)
            i += 1
    return (
        disable_grad_checkpointing,
        save_every_epoch,
        weights,
        eval_variants,
        eval_n_per_class,
        train_eval_n_per_class,
        diag_cap,
        train_diag_cap,
        eval_diag_cap,
        extra_eval_caps,
        list(_sanitize_complement_extra(tuple(train_extra))),
    )


def run_one(ds: DatasetSpec, recipe: Recipe, args: argparse.Namespace, env: dict[str, str]) -> None:
    if is_live_pruned_for_dataset(ds.name, recipe.tag):
        print(f"[sweep] SKIP live-pruned dataset={ds.name} tag={recipe.tag}", flush=True)
        return
    out_dir = REPO / "outputs" / ds.name / recipe.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    lead = REPO / "outputs" / ds.name / "_leaderboard.csv"
    leaderboard_rows = read_leaderboard(lead)
    done_tags = {r.get("tag", "") for r in leaderboard_rows if r.get("status") == "done"}
    if recipe.tag in done_tags:
        print(f"[sweep] SKIP done dataset={ds.name} tag={recipe.tag}", flush=True)
        return
    if any(
        r.get("tag", "") == recipe.tag and r.get("status") == "skipped_invalid_grid"
        for r in leaderboard_rows
    ):
        print(f"[sweep] SKIP invalid-grid recorded dataset={ds.name} tag={recipe.tag}", flush=True)
        return
    invalid_grid = _invalid_group_grid(recipe.groups, recipe.extra)
    if invalid_grid:
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": ds.name,
            "train_root": ds.train,
            "eval_root": ds.eval,
            "tag": recipe.tag,
            "variant": recipe.variant,
            "LS": recipe.ls,
            "n_groups": recipe.groups,
            "cmp_ls": recipe.cmp_ls,
            "cutmix_p": recipe.cutmix_p,
            "seed": recipe.seed,
            "extra": " ".join(recipe.extra),
            "epochs": recipe.epochs,
            "batch": recipe.batch,
            "accum": recipe.accum,
            "status": "skipped_invalid_grid",
            "out_dir": str(out_dir),
        }
        append_leaderboard(lead, row)
        print(f"[sweep] SKIP invalid grid geometry dataset={ds.name} tag={recipe.tag} {invalid_grid}", flush=True)
        return
    failed_statuses = [
        r.get("status", "")
        for r in leaderboard_rows
        if r.get("tag", "") == recipe.tag and r.get("status", "").startswith("train_fail")
    ]
    if failed_statuses:
        partial_ckpts = checkpoint_candidates(out_dir) if has_promising_quick_checkpoint(out_dir) else []
        hard_failures = [s for s in failed_statuses if not is_external_interrupt_status(s)]
        if hard_failures and not partial_ckpts:
            print(f"[sweep] SKIP failed-partial dataset={ds.name} tag={recipe.tag}", flush=True)
            return
        if partial_ckpts:
            print(f"[sweep] recover interrupted partial ckpts: {[str(p) for _, p in partial_ckpts]}", flush=True)
        else:
            print(f"[sweep] SKIP interrupted weak-partial dataset={ds.name} tag={recipe.tag}", flush=True)
            return
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": ds.name,
        "train_root": ds.train,
        "eval_root": ds.eval,
        "tag": recipe.tag,
        "variant": recipe.variant,
        "LS": recipe.ls,
        "n_groups": recipe.groups,
        "cmp_ls": recipe.cmp_ls,
        "cutmix_p": recipe.cutmix_p,
        "seed": recipe.seed,
        "extra": " ".join(recipe.extra),
        "epochs": recipe.epochs,
        "batch": recipe.batch,
        "accum": recipe.accum,
        "status": "start",
        "out_dir": str(out_dir),
    }
    print(f"[sweep] START dataset={ds.name} tag={recipe.tag}", flush=True)
    print(f"[sweep] train={ds.train}", flush=True)
    print(f"[sweep] eval ={ds.eval}", flush=True)
    (
        disable_grad_checkpointing,
        save_every_epoch,
        weights,
        eval_variants,
        recipe_eval_n_per_class,
        recipe_train_eval_n_per_class,
        recipe_diag_cap,
        recipe_train_diag_cap,
        recipe_eval_diag_cap,
        extra_eval_caps,
        train_extra,
    ) = split_sweep_extra(recipe.extra)
    eval_n_per_class = recipe_eval_n_per_class or args.eval_n_per_class
    train_eval_n_per_class = recipe_train_eval_n_per_class or args.train_eval_n_per_class
    train_diag_cap = recipe_train_diag_cap or recipe_diag_cap or args.diag_cap
    eval_diag_cap = recipe_eval_diag_cap or recipe_diag_cap or args.diag_cap
    row.update(
        {
            "train_cap_per_class": _extra_value(recipe.extra, "--max-per-class-defect"),
            "train_cap_select": _extra_value(recipe.extra, "--max-per-class-defect-select"),
            "eval_n_per_class": str(eval_n_per_class),
            "train_eval_n_per_class": str(train_eval_n_per_class),
            "train_diag_cap": str(train_diag_cap),
            "eval_diag_cap": str(eval_diag_cap),
            "sample_seed": recipe.seed,
        }
    )

    ckpts = checkpoint_candidates(out_dir)
    if not ckpts:
        has_pair = "--cutmix-pair" in train_extra
        cmd = cmd_base(weights) + [
            "--epochs",
            recipe.epochs,
            "--batch",
            recipe.batch,
            "--accum",
            recipe.accum,
            "--variant",
            recipe.variant,
            "--ls",
            recipe.ls,
            "--seed",
            recipe.seed,
            "--cutmix-mode",
            "complement",
        ]
        if disable_grad_checkpointing:
            cmd = [x for x in cmd if x != "--grad-checkpointing"]
        if save_every_epoch:
            cmd += ["--save-every-epoch"]
        if not has_pair:
            cmd += ["--cutmix-pair", "masked"]
        grid_dim_args = [] if "--cutmix-grid-dim" in train_extra else ["--cutmix-grid-dim", "8"]
        cmd += [
            "--cutmix-p",
            recipe.cutmix_p,
            *grid_dim_args,
            "--cutmix-n-groups",
            recipe.groups,
            "--cutmix-complete-label-scale",
            recipe.cmp_ls,
            *train_extra,
            "--data-root",
            ds.train,
            "--multi-val-set",
            ds.eval,
            "--multi-val-n-per-class",
            str(args.multi_val_n_per_class),
            "--out-root",
            str(out_dir),
            "--tag",
            recipe.tag,
        ]
        rc = run_logged(cmd, out_dir / "train.log", env)
        if rc != 0:
            ckpts = checkpoint_candidates(out_dir) if has_promising_quick_checkpoint(out_dir) else []
            if ckpts:
                row["status"] = f"train_interrupted_{rc}"
                print(f"[sweep] TRAIN INTERRUPTED rc={rc}; evaluating partial ckpts out={out_dir}", flush=True)
            else:
                row["status"] = f"train_interrupted_{rc}" if is_external_interrupt_rc(rc) else f"train_fail_{rc}"
                append_leaderboard(lead, row)
                print(f"[sweep] TRAIN FAIL rc={rc} out={out_dir}", flush=True)
                return
        ckpts = checkpoint_candidates(out_dir)
    else:
        print(f"[sweep] skip train, ckpts exist: {[str(p) for _, p in ckpts]}", flush=True)

    if not ckpts:
        row["status"] = "train_no_model"
        append_leaderboard(lead, row)
        print(f"[sweep] TRAIN FAIL no model out={out_dir}", flush=True)
        return

    selected_name = ""
    selected_model: Path | None = None
    selected_eval: dict[str, str] = {}
    for ckpt_name, ckpt_model in ckpts:
        eval_log_ckpt = out_dir / f"eval_{ckpt_name}.log"
        if not eval_log_ckpt.exists() or not parse_eval(eval_log_ckpt):
            rc = run_logged(
                eval_cmd(ckpt_model, ds.eval, out_dir / f"eval_{ckpt_name}", eval_n_per_class, eval_variants),
                eval_log_ckpt,
                env,
            )
            if rc != 0:
                print(f"[sweep] WARN eval {ckpt_name} failed rc={rc} out={out_dir}", flush=True)
                continue
        em_ckpt = parse_eval(eval_log_ckpt)
        if not em_ckpt:
            continue
        old_score = _far_first_score(_f(selected_eval, "bit_F1"), _f(selected_eval, "Total_FAR", 100.0))
        new_score = _far_first_score(_f(em_ckpt, "bit_F1"), _f(em_ckpt, "Total_FAR", 100.0))
        if selected_model is None or new_score > old_score:
            selected_name, selected_model, selected_eval = ckpt_name, ckpt_model, em_ckpt

    if selected_model is None:
        row["status"] = "eval_no_selected_model"
        append_leaderboard(lead, row)
        print(f"[sweep] EVAL FAIL no selected model out={out_dir}", flush=True)
        return

    model = selected_model
    row["ckpt"] = selected_name
    row["model"] = str(model)
    row.update(
        {
            "eval_cell": selected_eval.get("cell", ""),
            "eval_bit_F1": selected_eval.get("bit_F1", ""),
            "eval_NI_FAR": selected_eval.get("NI_FAR", ""),
            "eval_OOD_FAR": selected_eval.get("OOD_FAR", ""),
            "eval_Total_FAR": selected_eval.get("Total_FAR", ""),
            "eval_bb_F1": selected_eval.get("bb_F1", ""),
            "eval_fk_F1": selected_eval.get("fk_F1", ""),
            "eval_sc_F1": selected_eval.get("sc_F1", ""),
            "eval_sr_F1": selected_eval.get("sr_F1", ""),
        }
    )

    train_eval_log = out_dir / f"train_eval_{selected_name}.log"
    if not train_eval_log.exists() or not parse_eval(train_eval_log):
        rc = run_logged(
            eval_cmd(model, ds.train, out_dir / f"train_eval_{selected_name}", train_eval_n_per_class, eval_variants),
            train_eval_log,
            env,
        )
        if rc != 0:
            row["status"] = f"train_eval_fail_{rc}"
            append_leaderboard(lead, row)
            print(f"[sweep] TRAIN-EVAL FAIL rc={rc} out={out_dir}", flush=True)
            return
    tm = parse_eval(train_eval_log)
    row.update(
        {
            "train_cell": tm.get("cell", ""),
            "train_bit_F1": tm.get("bit_F1", ""),
            "train_NI_FAR": tm.get("NI_FAR", ""),
            "train_OOD_FAR": tm.get("OOD_FAR", ""),
            "train_Total_FAR": tm.get("Total_FAR", ""),
        }
    )

    train_diag_log = out_dir / f"train_posneg_pcls_{selected_name}.log"
    if not train_diag_log.exists() or not parse_diag(train_diag_log):
        rc = run_logged(
            diag_cmd(model, ds.train, f"{ds.name}:{recipe.tag}:train", train_diag_cap, args.diag_device, recipe.seed),
            train_diag_log,
            env,
        )
        if rc != 0:
            print(f"[sweep] WARN train diag failed rc={rc} out={out_dir}", flush=True)
    td = parse_diag(train_diag_log)
    row["train_pos_prob"] = td.get("pos_prob", "")
    row["train_neg_prob"] = td.get("neg_prob", "")
    train_rows = parse_pcls_rows(train_diag_log)
    if not train_rows:
        row["status"] = "train_diag_no_pcls"
        append_leaderboard(lead, row)
        print(f"[sweep] TRAIN DIAG FAIL no per-class rows out={out_dir}", flush=True)
        return
    write_pcls_csv(train_rows, out_dir / "train_pcls.csv")
    train_report = out_dir / "train_pcls_report.md"
    write_pcls_report(
        train_rows,
        train_report,
        f"{recipe_display(row)} -- TRAIN (4 single class)",
        td,
        "train",
        ds.train,
        tm,
    )
    row["train_pcls_report"] = str(train_report)

    eval_diag_log = out_dir / f"eval_posneg_pcls_{selected_name}.log"
    if not eval_diag_log.exists() or not parse_diag(eval_diag_log):
        rc = run_logged(
            diag_cmd(model, ds.eval, f"{ds.name}:{recipe.tag}:eval", eval_diag_cap, args.diag_device, "42"),
            eval_diag_log,
            env,
        )
        if rc != 0:
            print(f"[sweep] WARN eval diag failed rc={rc} out={out_dir}", flush=True)
    ed = parse_diag(eval_diag_log)
    row["eval_pos_prob"] = ed.get("pos_prob", "")
    row["eval_neg_prob"] = ed.get("neg_prob", "")
    eval_rows = parse_pcls_rows(eval_diag_log)
    if not eval_rows:
        row["status"] = "eval_diag_no_pcls"
        append_leaderboard(lead, row)
        print(f"[sweep] EVAL DIAG FAIL no per-class rows out={out_dir}", flush=True)
        return
    row.update(pcls_gap_fields(eval_rows))
    write_pcls_csv(eval_rows, out_dir / "eval_pcls.csv")
    eval_report = out_dir / "eval_pcls_report.md"
    write_pcls_report(
        eval_rows,
        eval_report,
        f"{recipe_display(row)} -- EVAL per-class 4-bit prob (POS = single+combo, NEG = Normal/Invalid/OOD)",
        ed,
        "eval",
        ds.eval,
        selected_eval,
    )
    row["eval_pcls_report"] = str(eval_report)
    perf_report = out_dir / "performance_report.md"
    write_performance_report(perf_report, row, train_report, eval_report)
    row["performance_report"] = str(perf_report)

    row["status"] = "done"
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    append_leaderboard(lead, row)
    for extra_cap in extra_eval_caps:
        if extra_cap == eval_n_per_class:
            continue
        extra_tag = re.sub(r"_ev\d{5}$", f"_ev{extra_cap:05d}", recipe.tag)
        if extra_tag == recipe.tag:
            extra_tag = f"{recipe.tag}_ev{extra_cap:05d}"
        if extra_tag in done_tags:
            print(f"[sweep] SKIP extra eval done dataset={ds.name} tag={extra_tag}", flush=True)
            continue
        extra_row = dict(row)
        extra_row.update(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "tag": extra_tag,
                "eval_n_per_class": str(extra_cap),
                "eval_diag_cap": str(extra_cap),
            }
        )
        extra_eval_log = out_dir / f"eval_{selected_name}_n{extra_cap}.log"
        if not extra_eval_log.exists() or not parse_eval(extra_eval_log):
            rc = run_logged(
                eval_cmd(model, ds.eval, out_dir / f"eval_{selected_name}_n{extra_cap}", extra_cap, eval_variants),
                extra_eval_log,
                env,
            )
            if rc != 0:
                extra_row["status"] = f"eval_extra_fail_{rc}"
                append_leaderboard(lead, extra_row)
                print(f"[sweep] EXTRA EVAL FAIL rc={rc} tag={extra_tag} out={out_dir}", flush=True)
                continue
        extra_eval = parse_eval(extra_eval_log)
        if not extra_eval:
            extra_row["status"] = "eval_extra_no_metrics"
            append_leaderboard(lead, extra_row)
            print(f"[sweep] EXTRA EVAL no metrics tag={extra_tag} out={out_dir}", flush=True)
            continue
        extra_row.update(
            {
                "eval_cell": extra_eval.get("cell", ""),
                "eval_bit_F1": extra_eval.get("bit_F1", ""),
                "eval_NI_FAR": extra_eval.get("NI_FAR", ""),
                "eval_OOD_FAR": extra_eval.get("OOD_FAR", ""),
                "eval_Total_FAR": extra_eval.get("Total_FAR", ""),
                "eval_bb_F1": extra_eval.get("bb_F1", ""),
                "eval_fk_F1": extra_eval.get("fk_F1", ""),
                "eval_sc_F1": extra_eval.get("sc_F1", ""),
                "eval_sr_F1": extra_eval.get("sr_F1", ""),
            }
        )
        extra_eval_diag_log = out_dir / f"eval_posneg_pcls_{selected_name}_n{extra_cap}.log"
        if not extra_eval_diag_log.exists() or not parse_diag(extra_eval_diag_log):
            rc = run_logged(
                diag_cmd(model, ds.eval, f"{ds.name}:{extra_tag}:eval", extra_cap, args.diag_device, "42"),
                extra_eval_diag_log,
                env,
            )
            if rc != 0:
                print(f"[sweep] WARN extra eval diag failed rc={rc} tag={extra_tag} out={out_dir}", flush=True)
        extra_ed = parse_diag(extra_eval_diag_log)
        extra_row["eval_pos_prob"] = extra_ed.get("pos_prob", "")
        extra_row["eval_neg_prob"] = extra_ed.get("neg_prob", "")
        extra_eval_rows = parse_pcls_rows(extra_eval_diag_log)
        if not extra_eval_rows:
            extra_row["status"] = "eval_extra_diag_no_pcls"
            append_leaderboard(lead, extra_row)
            print(f"[sweep] EXTRA EVAL DIAG FAIL no per-class rows tag={extra_tag} out={out_dir}", flush=True)
            continue
        extra_row.update(pcls_gap_fields(extra_eval_rows))
        extra_eval_csv = out_dir / f"eval_pcls_n{extra_cap}.csv"
        write_pcls_csv(extra_eval_rows, extra_eval_csv)
        extra_eval_report = out_dir / f"eval_pcls_report_n{extra_cap}.md"
        write_pcls_report(
            extra_eval_rows,
            extra_eval_report,
            f"{recipe_display(extra_row)} -- EVAL per-class 4-bit prob (POS = single+combo, NEG = Normal/Invalid/OOD)",
            extra_ed,
            "eval",
            ds.eval,
            extra_eval,
        )
        extra_row["eval_pcls_report"] = str(extra_eval_report)
        extra_perf_report = out_dir / f"performance_report_n{extra_cap}.md"
        write_performance_report(extra_perf_report, extra_row, train_report, extra_eval_report)
        extra_row["performance_report"] = str(extra_perf_report)
        extra_row["status"] = "done"
        append_leaderboard(lead, extra_row)
        print(
            "[sweep] DONE extra-eval "
            f"dataset={ds.name} tag={extra_tag} "
            f"eval_bit_F1={extra_row.get('eval_bit_F1', '')} "
            f"eval_Total_FAR={extra_row.get('eval_Total_FAR', '')}% "
            f"eval_pos={extra_row.get('eval_pos_prob', '')} "
            f"eval_neg={extra_row.get('eval_neg_prob', '')}",
            flush=True,
        )
    print(
        "[sweep] DONE "
        f"dataset={ds.name} tag={recipe.tag} "
        f"eval_bit_F1={row.get('eval_bit_F1', '')} "
        f"eval_Total_FAR={row.get('eval_Total_FAR', '')}% "
        f"eval_pos={row.get('eval_pos_prob', '')} "
        f"eval_neg={row.get('eval_neg_prob', '')}",
        flush=True,
    )
    print(f"[OUT] {out_dir}", flush=True)


def selected_datasets(names: str) -> Iterable[DatasetSpec]:
    for name in [x.strip() for x in names.split(",") if x.strip()]:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset '{name}' (allowed: {','.join(DATASETS)})")
        yield DATASETS[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="frozen_original")
    ap.add_argument("--diag-device", default="cuda")
    ap.add_argument("--diag-cap", type=int, default=2000)
    ap.add_argument("--eval-n-per-class", type=int, default=2000)
    ap.add_argument("--train-eval-n-per-class", type=int, default=2000)
    ap.add_argument("--multi-val-n-per-class", type=int, default=50)
    ap.add_argument("--max-recipes", type=int, default=0, help="0 means no limit")
    ap.add_argument("--forever", action="store_true", help="continue adaptive batches until killed")
    ap.add_argument("--adaptive-batch-size", type=int, default=12)
    ap.add_argument("--idle-seconds", type=int, default=60)
    args = ap.parse_args()

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "TORCH_NUM_THREADS": "2",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )

    plan = recipes()
    if args.max_recipes and args.max_recipes > 0:
        plan = plan[: args.max_recipes]

    datasets = list(selected_datasets(args.datasets))
    for ds in datasets:
        for recipe in prioritized_plan(ds, plan):
            run_one(ds, recipe, args, env)

    if not args.forever:
        print("[sweep] ALL REQUESTED DATASETS/RECIPES DONE", flush=True)
        return

    if os.environ.get("CHIP_SWEEP_QUEUE_MODE") == "target_label_matrix":
        print("[sweep] TARGET LABEL MATRIX QUEUE DONE; sleeping without generic adaptive recipes", flush=True)
        while True:
            time.sleep(args.idle_seconds)

    print("[sweep] FIXED QUEUE DONE; entering adaptive forever loop", flush=True)
    while True:
        did_work = False
        for ds in datasets:
            nxt = adaptive_recipes(ds, args.adaptive_batch_size)
            if not nxt:
                continue
            did_work = True
            print(f"[sweep] ADAPTIVE batch dataset={ds.name} n={len(nxt)}", flush=True)
            for recipe in nxt:
                run_one(ds, recipe, args, env)
        if not did_work:
            print(f"[sweep] ADAPTIVE no new candidates; sleep {args.idle_seconds}s", flush=True)
            time.sleep(args.idle_seconds)


if __name__ == "__main__":
    main()
