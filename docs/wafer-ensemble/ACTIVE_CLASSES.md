# Active class policy — 20 active + 14 archive

V3 의 saturated 분류 결과 (대부분 class 100% F1) 를 바탕으로 33-class →
**20 active + 14 archive** 결정. 데이터 자체는 보존 (copy 만), 학습 시
active list 만 사용.

## 결정 근거

1. **V3 의 obj-confusion weak point 모두 보존**: Edge-Bottom × 5, Edge-Top × 5
   (intra-distribution per-class breakdown 의 0.99 미만 유일한 두 분포)
2. **reference saturated subgroup**: Donut × 5, Edge-Ring × 4 (모두 100% F1)
3. **사용자 명시 보존**: Thick-Edge_invalid_main

총 20 class. 33 - 20 = 13 → 14 class archive (Edge-Ring_invalid_main 도 archive).

## Active 20 (`experiments/active_classes_20.yaml`)

| Subgroup | Classes | n |
|---|---|---|
| Donut × 5 obj | Donut_bank_boundary, Donut_invalid_main, Donut_particle_blast, Donut_scratch, Donut_scratch_21deg | 5 |
| **Edge-Bottom × 5 obj** ★ | Edge-Bottom_bank_boundary, Edge-Bottom_invalid_main, Edge-Bottom_particle_blast, Edge-Bottom_scratch, Edge-Bottom_scratch_21deg | 5 |
| Edge-Ring × 4 obj | Edge-Ring_bank_boundary, Edge-Ring_particle_blast, Edge-Ring_scratch, Edge-Ring_scratch_21deg | 4 |
| **Edge-Top × 5 obj** ★ | Edge-Top_bank_boundary, Edge-Top_invalid_main, Edge-Top_particle_blast, Edge-Top_scratch, Edge-Top_scratch_21deg | 5 |
| 특수 | Thick-Edge_invalid_main | 1 |
| **Total** | | **20** |

★ = V3 의 weak point (intra-distribution obj 식별 6 chip 안에서)

## Archive 14 (`experiments/archive_classes_14.yaml`)

| Subgroup | Classes | n | 이유 |
|---|---|---|---|
| Center × 5 obj | Center_bank_boundary, Center_invalid_main, Center_particle_blast, Center_scratch, Center_scratch_21deg | 5 | V3 100% saturated |
| Full × 5 obj | Full_bank_boundary, Full_invalid_main, Full_particle_blast, Full_scratch, Full_scratch_21deg | 5 | V3 100% saturated |
| Edge-Ring_invalid_main | Edge-Ring_invalid_main | 1 | saturated, 다른 4 obj 만 active |
| Normal_bank_boundary | Normal_bank_boundary | 1 | obj 있고 saturated, 추후 별도 학습 |
| Wafer-canvas (obj 없음) | Starburst, CommaCluster | 2 | 새 8 wafer-canvas class 와 같이 활용 가능 |
| **Total** | | **14** | |

## 데이터 위치

| Active 20 | Archive 14 |
|---|---|
| `D:/project/data/wm-811k/unknown/<class>/*.png` | `D:/project/data/wm-811k/unknown_archive/<class>/*.png` |
| `D:/project/data/positions/unknown/<class>/*.json` | `D:/project/data/positions/unknown_archive/<class>/*.json` |
| `D:/project/data/wm-811k/obj_id_maps/<basename>.npy` | (영향 X — flat basename lookup) |

archive = **copy** 이고 원본 unknown/ 는 그대로 둠. 추후 ablation, generalization
학습에 재활성화 가능 (active YAML 변경만으로).

## Future state

`experiments/configs/chipgrid_class30_target.yaml` = 8 wafer-canvas 새 class
(DiagonalSmear, CrossScratch, CrescentArc, SpiralTrail, ParallelScratches,
EdgeSmudge, BlobChain, BrokenRing) 합성 후 사용 target list. 합성 전엔 strict
check 로 fail 하는 게 정상.

## 사용

```bash
# Active 20 로 학습
python cnn_eval_chipgrid.py --variant V3 --no-r-channel \
    --active-classes-yaml experiments/active_classes_20.yaml

python cnn_train_objonly.py --epochs 30 \
    --active-classes-yaml experiments/active_classes_20.yaml --train-val-only

python cnn_train_compound.py --epochs 30 --g-channel-mode onehot \
    --active-classes-yaml experiments/active_classes_20.yaml
```

옵션 `--allow-missing-active-classes` 없으면 strict — YAML 안의 class 가 data
dir 에 없으면 fail.

## 정책 cross-link

- 정책 memory → `~/.claude/projects/D--project-known-cnn/memory/feedback_active_class_policy.md`
- V3 결정 근거 → `~/.claude/projects/D--project-known-cnn/memory/project_v3_chipgrid_best.md`
- intra-distribution data → `INTRA_DIST.md`
- 글로벌 룰 (학습 결과/데이터 폴더 절대 삭제 금지) → `~/.claude/CLAUDE.md`

## 절대 금지

- 데이터 폴더 (`unknown/<class>`) 무단 삭제 금지. archive 는 copy 만.
- `EXCLUDE_CLASSES` 같은 hardcoded list 에 새 class 추가 금지 (active YAML 사용).
- archive_14 의 6 wafer-canvas class (Starburst, CommaCluster) 데이터 삭제 금지
  — 8 새 wafer-canvas class 와 같이 활용 예정.
