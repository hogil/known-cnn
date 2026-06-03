# FCMPM Multi-Agent Experiment Loop

updated: 2026-06-04

이 문서는 FCMPM 논문 기초실험을 멈추지 않고 누적하기 위한 운영 구조다. 목적은 단순히 한 번의 SOTA row를 찾는 것이 아니라, 단일 변수 분리 평가와 조합 실험을 계속 돌려서 어떤 인자가 성능과 probability separation에 영향을 주는지 기록하는 것이다.

## 1. Master Agent

사용 skill:

```text
C:/Users/hgcho/.codex/skills/chip-multilabel-experiment-master/SKILL.md
```

역할:

- `recipe_sweep --forever`가 살아있는지 확인한다.
- resource / analysis / experiment-design agent를 나눠서 운용한다.
- 보고서와 leaderboard가 계속 갱신되는지 확인한다.
- 약한 조건은 prune하고, 좋은 조건은 seed/dataset repeat 및 2-factor/3-factor로 확장한다.

현재 기본 실행:

```powershell
python -u -m chip_multilabel.recipe_sweep --datasets frozen_original,sota_gapstress_seed31_260531,sota_gapstress_seed97_260531,frozen_original_200_snapshot,frozen_original_2015_candidate --diag-device cuda --forever
```

## 2. Resource Manager Agent

사용 skill:

```text
C:/Users/hgcho/.codex/skills/chip-multilabel-resource-manager/SKILL.md
```

역할:

- GPU / process / disk 상태 확인.
- `D:/project/known-cnn/outputs`의 `.pth` 폭증 방지.
- 성능 증거 파일은 보존하고, 낮은 성능 또는 redundant checkpoint만 삭제.

현재 운영 명령:

```powershell
python chip_multilabel\cleanup_checkpoints.py --min-f1 0.990 --max-far 5.0 --execute
```

삭제 금지:

- `train.log`, `eval_best.log`
- `_leaderboard.csv`
- `train_pcls.csv`, `eval_pcls.csv`
- `train_pcls_report.md`, `eval_pcls_report.md`
- manager report markdown

## 3. Performance + Literature Analyst Agent

사용 skill:

```text
C:/Users/hgcho/.codex/skills/chip-multilabel-performance-literature-analyst/SKILL.md
```

역할:

- bit_F1뿐 아니라 POS min / NEG max / global gap을 같이 본다.
- single, 2-combo, OOD의 4-bit probability pattern을 비교한다.
- NB reject가 필요한 case를 `max_prob`가 아니라 pattern likelihood 관점으로 설명한다.
- 외부 문헌은 primary source 중심으로 정리한다.

현재 참고할 1차 문헌 축:

- mixup: Zhang et al., ICLR 2018.
- CutMix: Yun et al., ICCV 2019.
- Mahalanobis/GDA OOD score: Lee et al., NeurIPS 2018.
- Selective classification / reject option: Geifman and El-Yaniv, NeurIPS 2017; SelectiveNet, ICML 2019.

## 4. Experiment Designer Agent

사용 skill:

```text
C:/Users/hgcho/.codex/skills/chip-multilabel-experiment-designer/SKILL.md
```

역할:

- 단일 변수 분리 평가를 유지한다.
- 고정 조건에서 하나의 변수만 바꿔 영향도를 본다.
- 충분한 one-axis evidence가 모이면 2-factor interaction으로 확장한다.
- 2-factor 상위권이 안정되면 3-factor neighborhood로 확장한다.
- collapse 조건은 재실험하지 않고 skip/prune한다.

현재 baseline:

```text
T7, LS=0.295, g=3, grid=9x9, cmp=1.0, cutmix_p=0.5
A/B target=1.00/1.00, neg target=0.0, mpos=0.65, seed=7
train=200/class, eval=2000/class
```

현재 관찰:

- `cutmix_p=0.575`가 현재 one-axis 최고권이다.
- `cutmix_p=0.70/0.80`은 F1은 높지만 tail leak이 커질 수 있어 repeat/분산 확인이 필요하다.
- `neg target=0.0015~0.005`는 tail 제어 후보로 남긴다.
- A target을 `1.0 -> 0.9/0.8/0.7`로 낮추는 방향은 현재까지 대체로 손해다.
- ASL/T10/T4/T6는 현재 frozen 기준 collapse/weak 결과라 prune 대상이다. 단, 다른 dataset에서 반례가 나오면 재검토한다.

## 5. Periodic Ops Loop

현재 운영 루프:

```text
_experiment_ops_loop_260603.ps1
```

30분마다 수행:

```powershell
python chip_multilabel\one_axis_ablation_report.py --leaderboards outputs\*\_leaderboard.csv --out docs\chip-multilabel\manager_report\ONE_AXIS_ABLATION_STATUS_260603.md
python chip_multilabel\live_experiment_audit.py --leaderboards outputs\*\_leaderboard.csv --out docs\chip-multilabel\manager_report\LIVE_EXPERIMENT_AUDIT_260603.md
python chip_multilabel\cleanup_checkpoints.py --min-f1 0.990 --max-far 5.0 --execute
```

## 6. Reporting Rule

관리자용 matrix:

- FAR 컬럼은 혼동을 줄이기 위해 기본 표에서는 제외한다.
- 대신 bit_F1, pos_prob, neg_prob, global_gap, worst POS min, worst NEG max를 표시한다.

기술 분석용 상세 보고:

- FAR 포함.
- train/eval root 포함.
- class별 4-bit probability table 포함.
- single / 2combo / OOD probability pattern bar chart 또는 compact table 포함.

## 7. Stop Condition

사용자가 명시적으로 “그만”, “멈춰”, “중단”이라고 말하기 전까지 loop를 멈추지 않는다.

일시적으로 weak run을 prune하거나 queue를 바꾸는 것은 중단이 아니라 실험 효율화다.
