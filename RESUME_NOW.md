# RESUME after reboot — 2026-05-14 (UPDATED)

## 한 줄 상태

W1 g=2 LS×n sweep 진행 중 (BG `b7pz9kg30`, 34 cells D:-trained skip, 44 cells E:-data training).
D: chip data 외부 삭제 사고 → E:\data\images 로 데이터 전환됨.

## D: 데이터 삭제 사고 (260514 05:03 사이 발생)

**증상**: W1_n2_ls6 학습 중 `D:\project\data\wm-811k\classification_chips\bank_boundary\KMA981...png` not found.
→ 전수 확인: `D:/project/data/wm-811k/classification_chips/`, `chip_multilabel_v15direct/` 디렉토리 자체가 없음.

**원인**: 불명. 본 세션의 어떤 스크립트도 D: 데이터 건드린 적 없음 (확인 완료):
- `_gen_E_chips.sh` — `--out E:/...` 명시
- `_gen_E_ood_chips.py` — READ-only on D: `unknown/`, WRITE E:
- `_synth_chips_only.py` `--clean-first` flag 옵션 (passed: NO)
- `gen_eval_set.py` `--clear` flag 옵션 (passed: NO)

추정: Windows 외부 (디스크 정리, 스케줄링, 사용자 GUI 작업) 또는 disk 에러.

## 대응 — E: 데이터로 전환

W1 script 패치 (`_run_W1_g2_matrix.sh`):
- `export WM811K_ROOT="E:/data/images"` (이전 D:)
- `EVAL_SET="E:/data/images/chip_multilabel_v15direct"` (이전 D:)

trainer 의 `DEFAULT_CLASSIFICATION_CHIPS = DATA_ROOT / "classification_chips"` 가 자동으로 E: 경로 사용.

## E: chip 데이터 (260514 00:13 생성, 정상)

| 위치 | class | per-class |
|---|---|---|
| `E:/data/images/classification_chips/` | 5 (bb/fork/scratch/scratch_rot/invalid_main) | 200 |
| `E:/data/images/chip_multilabel_v15direct/` 4 single | bb/fork/scratch/scratch_rot | 200 |
| 동일 6 2-combo | bb+fork, bb+scratch, bb+scratch_rot, fork+scratch, fork+scratch_rot, scratch+scratch_rot | 200 |
| Normal/Invalid | Normal/Invalid | 200/50 |
| 4 OOD | CenterDonut/CrossScratch/DiagonalSmear/Starburst | 200 |

총 16 valid class (D: 의 4 triple-combo + 4 OOD-overlay 는 미생성, 필요시 추가).

## W1 sweep 진행 상태 (260514 05:28 기준)

| Phase | 진행 | 비고 |
|---|---|---|
| LS=1.0 (n=1..8) | ✅ 8/8 (D:-trained) | best_model.pth 보존, skip |
| LS=0.9 (n=1..8) | ✅ 8/8 (D:-trained) | skip |
| LS=0.8 (n=1..8) | ✅ 8/8 (D:-trained) | skip |
| LS=0.7 (n=1..8) | ✅ 8/8 (D:-trained) | skip |
| LS=0.6 n=1, n=2 | ✅ 2/8 (D:-trained) | skip |
| LS=0.6 n=3..8, LS=0.5/0.4/0.3 n=7/0.2/0.1/0.0 | 44 cells TBD | E:-trained 진행 |

총 34 cells skip + 44 cells E:-trained ≈ 5 hr 남음.

## 즉시 재시작 명령 (현재 BG 중단된 경우만)

```bash
cd D:/project/known-cnn
bash _run_W1_g2_matrix.sh
```

기존 best_model.pth 있는 cells 는 skip 됨.

## 분석 도구

- `_w2_aggregate.py` (현재) — W2 cells 만. W1 위해 글롭 한 줄 수정 필요:
  ```python
  for pq in sorted(glob.glob("outputs/W1_n*/T*/eval_v15direct/stage1_*/preds_chip.parquet")):
  ```
- W1 끝나면 78 cells × 4 inference variant (I3/I7/I10/I13) bit_F1+Total_FAR 표 생성.

## Absolute rules (compaction loss 방지)

1. **train: 4 single only** (`--no-normal`)
2. **eval composition**: 4 single + 6 combo (positive) + Normal + Invalid + OOD (negative)
3. **bit_F1** = positive macro-F1 (single+combo, 9~10 cells), **not** 11-class macro_f1
4. **Total FAR** = (Normal_fp + Invalid_fp + OOD_fp) / N_total_negative
5. **BCE pos/neg target 독립** (`pos_target` + `neg_target` ≠ 1 가능)
6. **threshold 는 eval label 없이** — calibration set 사용
7. **TTA / rotation aug 금지**
8. **W2 pt/nt 독립** + **per-bit asymmetric** + **bce_temperature** 지원 — `chip_multilabel/losses.py`
9. **데이터 위치 이제 E:** — D: 외부 삭제됨, 복구 시 user 명시 후

## 핵심 anchors (skip 됨)

| cell | dir | 비고 |
|---|---|---|
| g2_n1_LS30 | `outputs/iter124_a_g2_n1` | D:-trained anchor |
| g2_n2_LS30 | `outputs/iter124_b_g2_n2` | |
| g2_n3_LS30 | `outputs/iter124_c_g2_n3` | |
| g2_n4_LS30 | `outputs/iter124_d_g2_n4` | |
| g2_n5_LS30 | `outputs/iter125_d_g2_n5` | |
| g2_n6_LS30 | `outputs/iter125_f_g2_n6` | |
| g2_n8_LS30 | `outputs/iter126_e_g2_n8` | ★ paper SOTA 0.9906 |

W1 dispatcher 의 cell naming = `W1_n{N}_ls{LS×10}`, anchors 와 충돌 없음.

## Files modified this session

- `_run_W1_g2_matrix.sh` — env var + WEIGHTS path + DATA root (D: → E:)
- `_gen_E_chips.sh` — E: chip generation orchestrator (train+eval)
- `_gen_E_ood_chips.py` — wafer canvas → 200x200 OOD chip extractor (4 class × 200)
- `_w2_aggregate.py` — W2 sweep aggregator (W1 도 1줄 수정으로 재사용)
- `outputs/_W2_aggregate.csv` — 39-cell W2 result (보존)
- (E:/data/images/* — 새 chip 데이터)

## 검증 명령

```bash
cd D:/project/known-cnn
# W1 BG live tail
tail -5 outputs/_W1_g2_matrix_summary.log

# Done cells count
ls outputs/W1_n*_ls*/T*/best_model.pth 2>/dev/null | wc -l

# E: data sanity
ls E:/data/images/classification_chips/  # 5 dirs
ls E:/data/images/chip_multilabel_v15direct/  # 16+ dirs
```
