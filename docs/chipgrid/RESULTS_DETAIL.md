# chipgrid 학습 결과 상세 (자동 생성)

_생성 시각: `python _chipgrid_summary.py`_

총 25 개 run 분석


### v0_260503_130538_0.37_0.35

**hparams**:
- `variant=V0` | `n_per_class=30` | `obj_norm=5.0` | `target_id=3` | `seed=42` | `epochs=20` | `batch=32` | `ema_decay=0.95`
- in_ch=`1`, n_classes=`34`, params=`1,154,786`

**데이터**:
- 총 sample (capped): **1010**, 80/10/10 split
- per-class count: 대부분 30, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=43.56%  f1=**37.00%**
- VAL   acc=39.60%  f1=**35.03%**
- best epoch = 20, total epochs run = 20

**TEST per-class (weak)**:
  - weak class (29/34, F1 < 0.95):
    - `Center_particle_blast`: F1=0.000  FP=0  FN=4  Sup=4
    - `Center_scratch`: F1=0.000  FP=0  FN=6  Sup=6
    - `Donut_scratch`: F1=0.000  FP=0  FN=4  Sup=4
    - `Donut_scratch_21deg`: F1=0.000  FP=0  FN=1  Sup=1
    - `Edge-Bottom_particle_blast`: F1=0.000  FP=0  FN=0  Sup=0
    - `Edge-Bottom_scratch`: F1=0.000  FP=0  FN=2  Sup=2
    - `Edge-Bottom_scratch_21deg`: F1=0.000  FP=0  FN=3  Sup=3
    - `Edge-Ring_bank_boundary`: F1=0.000  FP=0  FN=0  Sup=0
    - `Edge-Ring_particle_blast`: F1=0.000  FP=0  FN=3  Sup=3
    - `Edge-Top_particle_blast`: F1=0.000  FP=0  FN=3  Sup=3
    - `Edge-Top_scratch_21deg`: F1=0.000  FP=0  FN=4  Sup=4
    - `Full_bank_boundary`: F1=0.000  FP=0  FN=2  Sup=2
    - `Full_scratch`: F1=0.000  FP=0  FN=4  Sup=4
    - `CommaCluster`: F1=0.143  FP=10  FN=2  Sup=3
    - `Full_particle_blast`: F1=0.250  FP=4  FN=2  Sup=3
    - `Full_scratch_21deg`: F1=0.286  FP=4  FN=1  Sup=2
    - `Donut_particle_blast`: F1=0.333  FP=4  FN=0  Sup=1
    - `Edge-Top_bank_boundary`: F1=0.333  FP=1  FN=3  Sup=4
    - `Donut_bank_boundary`: F1=0.364  FP=7  FN=0  Sup=2
    - `Center_bank_boundary`: F1=0.400  FP=1  FN=2  Sup=3
    - `Center_scratch_21deg`: F1=0.400  FP=2  FN=1  Sup=2
    - `Edge-Bottom_bank_boundary`: F1=0.500  FP=0  FN=2  Sup=3
    - `Normal_bank_boundary`: F1=0.500  FP=2  FN=0  Sup=1
    - `Starburst`: F1=0.500  FP=0  FN=2  Sup=3
    - `Edge-Ring_scratch`: F1=0.667  FP=0  FN=1  Sup=2
    - `Edge-Top_invalid_main`: F1=0.667  FP=1  FN=1  Sup=3
    - `Edge-Top_scratch`: F1=0.667  FP=1  FN=1  Sup=3
    - `Edge-Bottom_invalid_main`: F1=0.714  FP=2  FN=2  Sup=7
    - `Edge-Ring_scratch_21deg`: F1=0.857  FP=0  FN=1  Sup=4

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.29%     0.18%
   4     2.98%     4.57%
   5    11.51%     9.44%
   6    13.18%    14.92%
   7    15.50%    17.47%
   9    17.33%    18.30%
  14    20.32%    24.59%
  20    35.03%    37.00%
```

### v0_full_260503_163546_0.40_0.47

**hparams**:
- `variant=V0` | `n_per_class=220` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`1`, n_classes=`34`, params=`1,154,786`

**데이터**:
- 총 sample (capped): **7100**, 80/10/10 split
- per-class count: 대부분 220, 예외: {'Center_invalid_main': 200, 'CommaCluster': 200, 'Donut_invalid_main': 200, 'Edge-Bottom_invalid_main': 200, 'Edge-Ring_invalid_main': 200, 'Edge-Top_invalid_main': 200, 'Full_invalid_main': 200, 'Normal_bank_boundary': 20, 'Starburst': 200, 'Thick-Edge_invalid_main': 200}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=43.66%  f1=**40.39%**
- VAL   acc=48.31%  f1=**46.98%**
- best epoch = 5, total epochs run = 12
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (28/34, F1 < 0.95):
    - `Center_bank_boundary`: F1=0.000  FP=0  FN=21  Sup=21
    - `Donut_particle_blast`: F1=0.000  FP=0  FN=30  Sup=30
    - `Donut_scratch`: F1=0.000  FP=0  FN=26  Sup=26
    - `Edge-Bottom_particle_blast`: F1=0.000  FP=0  FN=19  Sup=19
    - `Edge-Bottom_scratch_21deg`: F1=0.000  FP=0  FN=22  Sup=22
    - `Edge-Ring_particle_blast`: F1=0.000  FP=0  FN=14  Sup=14
    - `Full_particle_blast`: F1=0.000  FP=0  FN=23  Sup=23
    - `Normal_bank_boundary`: F1=0.000  FP=0  FN=1  Sup=1
    - `Center_scratch`: F1=0.069  FP=4  FN=23  Sup=24
    - `Edge-Top_scratch`: F1=0.114  FP=10  FN=21  Sup=23
    - `Edge-Top_bank_boundary`: F1=0.188  FP=5  FN=21  Sup=24
    - `Edge-Top_scratch_21deg`: F1=0.200  FP=5  FN=27  Sup=31
    - `Edge-Ring_scratch`: F1=0.245  FP=24  FN=13  Sup=19
    - `Edge-Top_particle_blast`: F1=0.273  FP=38  FN=10  Sup=19
    - `Donut_bank_boundary`: F1=0.312  FP=71  FN=4  Sup=21
    - `Edge-Bottom_scratch`: F1=0.319  FP=41  FN=6  Sup=17
    - `Edge-Ring_scratch_21deg`: F1=0.324  FP=6  FN=19  Sup=25
    - `Edge-Bottom_bank_boundary`: F1=0.340  FP=22  FN=13  Sup=22
    - `Edge-Ring_bank_boundary`: F1=0.361  FP=27  FN=12  Sup=23
    - `Donut_scratch_21deg`: F1=0.391  FP=9  FN=19  Sup=28
    - `Full_scratch`: F1=0.400  FP=13  FN=14  Sup=23
    - `Center_scratch_21deg`: F1=0.449  FP=14  FN=13  Sup=24
    - `Full_bank_boundary`: F1=0.451  FP=35  FN=4  Sup=20
    - `Center_particle_blast`: F1=0.452  FP=43  FN=8  Sup=29
    - `CommaCluster`: F1=0.473  FP=27  FN=2  Sup=15
    - `Full_scratch_21deg`: F1=0.609  FP=1  FN=8  Sup=15
    - `Starburst`: F1=0.828  FP=0  FN=5  Sup=17
    - `Edge-Bottom_invalid_main`: F1=0.938  FP=0  FN=2  Sup=17

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.13%     0.19%
   3     0.15%     0.12%
   4    11.68%    11.39%
   5    46.98%    40.39%
```

### v0_n100_260503_130816_0.44_0.44

**hparams**:
- `variant=V0` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`1`, n_classes=`34`, params=`1,154,786`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=44.58%  f1=**43.85%**
- VAL   acc=47.59%  f1=**43.59%**
- best epoch = 13, total epochs run = 20
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (26/34, F1 < 0.95):
    - `Donut_bank_boundary`: F1=0.000  FP=0  FN=11  Sup=11
    - `Edge-Bottom_particle_blast`: F1=0.000  FP=0  FN=7  Sup=7
    - `Edge-Top_bank_boundary`: F1=0.000  FP=0  FN=9  Sup=9
    - `Edge-Top_scratch_21deg`: F1=0.000  FP=0  FN=5  Sup=5
    - `Full_particle_blast`: F1=0.000  FP=0  FN=8  Sup=8
    - `Center_scratch_21deg`: F1=0.125  FP=3  FN=11  Sup=12
    - `Edge-Top_scratch`: F1=0.143  FP=1  FN=11  Sup=12
    - `Center_scratch`: F1=0.154  FP=0  FN=11  Sup=12
    - `Center_particle_blast`: F1=0.214  FP=10  FN=12  Sup=15
    - `Edge-Bottom_bank_boundary`: F1=0.222  FP=10  FN=4  Sup=6
    - `Full_bank_boundary`: F1=0.286  FP=4  FN=11  Sup=14
    - `Full_scratch`: F1=0.286  FP=18  FN=7  Sup=12
    - `Center_bank_boundary`: F1=0.294  FP=20  FN=4  Sup=9
    - `Edge-Top_particle_blast`: F1=0.294  FP=23  FN=1  Sup=6
    - `CommaCluster`: F1=0.300  FP=9  FN=5  Sup=8
    - `Donut_particle_blast`: F1=0.304  FP=27  FN=5  Sup=12
    - `Donut_scratch_21deg`: F1=0.308  FP=6  FN=12  Sup=16
    - `Donut_scratch`: F1=0.320  FP=7  FN=10  Sup=14
    - `Edge-Ring_bank_boundary`: F1=0.333  FP=1  FN=7  Sup=9
    - `Edge-Bottom_scratch`: F1=0.375  FP=5  FN=5  Sup=8
    - `Edge-Ring_scratch`: F1=0.400  FP=11  FN=4  Sup=9
    - `Edge-Ring_scratch_21deg`: F1=0.400  FP=5  FN=4  Sup=7
    - `Full_scratch_21deg`: F1=0.462  FP=9  FN=5  Sup=11
    - `Starburst`: F1=0.533  FP=2  FN=5  Sup=9
    - `Edge-Bottom_scratch_21deg`: F1=0.556  FP=3  FN=5  Sup=10
    - `Edge-Ring_particle_blast`: F1=0.600  FP=3  FN=5  Sup=11

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.16%     0.24%
   3     0.17%     0.12%
   4    13.96%    13.03%
   8    18.49%    17.47%
  11    30.39%    29.00%
  13    43.59%    43.85%
```

### v1_full_260503_163549_0.97_0.98

**hparams**:
- `variant=V1` | `n_per_class=220` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`2`, n_classes=`34`, params=`1,155,362`

**데이터**:
- 총 sample (capped): **7100**, 80/10/10 split
- per-class count: 대부분 220, 예외: {'Center_invalid_main': 200, 'CommaCluster': 200, 'Donut_invalid_main': 200, 'Edge-Bottom_invalid_main': 200, 'Edge-Ring_invalid_main': 200, 'Edge-Top_invalid_main': 200, 'Full_invalid_main': 200, 'Normal_bank_boundary': 20, 'Starburst': 200, 'Thick-Edge_invalid_main': 200}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=97.32%  f1=**97.35%**
- VAL   acc=98.17%  f1=**98.05%**
- best epoch = 11, total epochs run = 18
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (6/34, F1 < 0.95):
    - `Edge-Bottom_scratch`: F1=0.750  FP=3  FN=5  Sup=17
    - `Edge-Top_scratch`: F1=0.875  FP=4  FN=2  Sup=23
    - `Edge-Bottom_particle_blast`: F1=0.878  FP=4  FN=1  Sup=19
    - `Edge-Top_particle_blast`: F1=0.900  FP=3  FN=1  Sup=19
    - `Edge-Bottom_scratch_21deg`: F1=0.905  FP=1  FN=3  Sup=22
    - `Edge-Top_scratch_21deg`: F1=0.918  FP=2  FN=3  Sup=31

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.24%     0.21%
   4    95.51%    93.11%
   6    96.27%    96.69%
   7    97.60%    97.53%
  11    98.05%    97.35%
```

### v1_n100_260503_131535_0.97_0.95

**hparams**:
- `variant=V1` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`2`, n_classes=`34`, params=`1,155,362`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=97.59%  f1=**97.26%**
- VAL   acc=95.18%  f1=**95.05%**
- best epoch = 10, total epochs run = 17
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (7/34, F1 < 0.95):
    - `Edge-Bottom_scratch`: F1=0.824  FP=2  FN=1  Sup=8
    - `Edge-Bottom_particle_blast`: F1=0.875  FP=2  FN=0  Sup=7
    - `Edge-Bottom_scratch_21deg`: F1=0.889  FP=0  FN=2  Sup=10
    - `Edge-Bottom_bank_boundary`: F1=0.909  FP=0  FN=1  Sup=6
    - `Edge-Top_particle_blast`: F1=0.909  FP=0  FN=1  Sup=6
    - `Edge-Top_scratch_21deg`: F1=0.909  FP=1  FN=0  Sup=5
    - `Edge-Top_scratch`: F1=0.917  FP=1  FN=1  Sup=12

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.17%     0.21%
   4    82.00%    84.97%
   5    91.54%    94.60%
   6    93.72%    96.48%
  10    95.05%    97.26%
```

### v1_norm10_260503_165054_0.97_0.96

**hparams**:
- `variant=V1` | `n_per_class=100` | `obj_norm=10.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`2`, n_classes=`34`, params=`1,155,362`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=97.89%  f1=**97.37%**
- VAL   acc=95.78%  f1=**95.70%**
- best epoch = 11, total epochs run = 18
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (8/34, F1 < 0.95):
    - `Edge-Top_particle_blast`: F1=0.667  FP=0  FN=3  Sup=6
    - `Edge-Top_scratch`: F1=0.889  FP=3  FN=0  Sup=12
    - `Edge-Bottom_bank_boundary`: F1=0.909  FP=0  FN=1  Sup=6
    - `Donut_particle_blast`: F1=0.923  FP=2  FN=0  Sup=12
    - `CommaCluster`: F1=0.933  FP=0  FN=1  Sup=8
    - `Edge-Bottom_particle_blast`: F1=0.933  FP=1  FN=0  Sup=7
    - `Edge-Bottom_scratch`: F1=0.941  FP=1  FN=0  Sup=8
    - `Edge-Bottom_scratch_21deg`: F1=0.947  FP=0  FN=1  Sup=10

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.16%     0.24%
   2     0.17%     0.21%
   4    89.31%    90.15%
   5    91.61%    93.42%
   6    93.24%    95.89%
   8    93.58%    96.19%
  10    94.64%    96.23%
  11    95.70%    97.37%
```

### v1_norm1_260503_165049_0.99_0.96

**hparams**:
- `variant=V1` | `n_per_class=100` | `obj_norm=1.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`2`, n_classes=`34`, params=`1,155,362`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=99.40%  f1=**99.36%**
- VAL   acc=96.39%  f1=**96.06%**
- best epoch = 11, total epochs run = 18
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (1/34, F1 < 0.95):
    - `Edge-Top_particle_blast`: F1=0.909  FP=0  FN=1  Sup=6

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.14%     0.16%
   4    87.27%    88.87%
   5    90.73%    92.52%
   6    93.66%    97.38%
  10    94.39%    98.18%
  11    96.06%    99.36%
```

### v2_bank_boundary_260503_165821_0.65_0.65

**hparams**:
- `variant=V2` | `n_per_class=100` | `obj_norm=5.0` | `target_id=1` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`2`, n_classes=`34`, params=`1,155,362`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=63.55%  f1=**64.66%**
- VAL   acc=65.66%  f1=**64.53%**
- best epoch = 24, total epochs run = 30

**TEST per-class (weak)**:
  - weak class (23/34, F1 < 0.95):
    - `Center_scratch`: F1=0.111  FP=5  FN=11  Sup=12
    - `Edge-Bottom_particle_blast`: F1=0.182  FP=3  FN=6  Sup=7
    - `Full_particle_blast`: F1=0.261  FP=12  FN=5  Sup=8
    - `Donut_scratch_21deg`: F1=0.308  FP=6  FN=12  Sup=16
    - `Donut_particle_blast`: F1=0.320  FP=9  FN=8  Sup=12
    - `Edge-Ring_scratch`: F1=0.375  FP=4  FN=6  Sup=9
    - `Edge-Top_particle_blast`: F1=0.375  FP=7  FN=3  Sup=6
    - `Edge-Top_scratch_21deg`: F1=0.400  FP=3  FN=3  Sup=5
    - `Center_scratch_21deg`: F1=0.414  FP=11  FN=6  Sup=12
    - `Edge-Bottom_scratch`: F1=0.421  FP=7  FN=4  Sup=8
    - `Starburst`: F1=0.429  FP=2  FN=6  Sup=9
    - `Donut_scratch`: F1=0.432  FP=15  FN=6  Sup=14
    - `CommaCluster`: F1=0.444  FP=6  FN=4  Sup=8
    - `Full_scratch`: F1=0.444  FP=2  FN=8  Sup=12
    - `Edge-Bottom_scratch_21deg`: F1=0.500  FP=8  FN=4  Sup=10
    - `Center_particle_blast`: F1=0.552  FP=6  FN=7  Sup=15
    - `Full_scratch_21deg`: F1=0.571  FP=4  FN=5  Sup=11
    - `Edge-Ring_particle_blast`: F1=0.588  FP=1  FN=6  Sup=11
    - `Edge-Ring_scratch_21deg`: F1=0.600  FP=7  FN=1  Sup=7
    - `Edge-Top_scratch`: F1=0.600  FP=2  FN=6  Sup=12
    - `Edge-Bottom_bank_boundary`: F1=0.800  FP=0  FN=2  Sup=6
    - `Edge-Top_invalid_main`: F1=0.923  FP=0  FN=2  Sup=14
    - `Edge-Bottom_invalid_main`: F1=0.933  FP=1  FN=0  Sup=7

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.17%     0.12%
   4    18.93%    17.83%
   5    52.82%    51.97%
   8    58.16%    55.79%
   9    58.19%    58.46%
  13    61.73%    62.46%
  19    62.89%    65.66%
  24    64.53%    64.66%
```

### v2_invalid_main_260503_171834_0.54_0.51

**hparams**:
- `variant=V2` | `n_per_class=100` | `obj_norm=5.0` | `target_id=2` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`2`, n_classes=`34`, params=`1,155,362`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=53.31%  f1=**54.05%**
- VAL   acc=52.11%  f1=**51.14%**
- best epoch = 18, total epochs run = 25
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (27/34, F1 < 0.95):
    - `Full_particle_blast`: F1=0.200  FP=1  FN=7  Sup=8
    - `Center_scratch`: F1=0.222  FP=4  FN=10  Sup=12
    - `Edge-Top_scratch_21deg`: F1=0.222  FP=3  FN=4  Sup=5
    - `Donut_bank_boundary`: F1=0.229  FP=20  FN=7  Sup=11
    - `Edge-Ring_scratch`: F1=0.235  FP=6  FN=7  Sup=9
    - `Edge-Bottom_particle_blast`: F1=0.250  FP=7  FN=5  Sup=7
    - `Edge-Ring_bank_boundary`: F1=0.267  FP=4  FN=7  Sup=9
    - `Donut_particle_blast`: F1=0.273  FP=7  FN=9  Sup=12
    - `Edge-Bottom_bank_boundary`: F1=0.286  FP=6  FN=4  Sup=6
    - `Full_scratch`: F1=0.286  FP=18  FN=7  Sup=12
    - `Donut_scratch_21deg`: F1=0.333  FP=4  FN=12  Sup=16
    - `Edge-Top_bank_boundary`: F1=0.353  FP=5  FN=6  Sup=9
    - `Donut_scratch`: F1=0.364  FP=4  FN=10  Sup=14
    - `Edge-Top_particle_blast`: F1=0.364  FP=3  FN=4  Sup=6
    - `Center_bank_boundary`: F1=0.400  FP=11  FN=4  Sup=9
    - `Edge-Ring_scratch_21deg`: F1=0.400  FP=5  FN=4  Sup=7
    - `Full_bank_boundary`: F1=0.417  FP=5  FN=9  Sup=14
    - `Center_scratch_21deg`: F1=0.455  FP=5  FN=7  Sup=12
    - `Edge-Bottom_scratch`: F1=0.462  FP=2  FN=5  Sup=8
    - `Center_particle_blast`: F1=0.485  FP=10  FN=7  Sup=15
    - `Edge-Bottom_scratch_21deg`: F1=0.500  FP=5  FN=5  Sup=10
    - `Full_scratch_21deg`: F1=0.571  FP=4  FN=5  Sup=11
    - `Edge-Top_scratch`: F1=0.615  FP=6  FN=4  Sup=12
    - `CommaCluster`: F1=0.632  FP=5  FN=2  Sup=8
    - `Edge-Ring_particle_blast`: F1=0.750  FP=4  FN=2  Sup=11
    - `Starburst`: F1=0.875  FP=0  FN=2  Sup=9
    - `Edge-Bottom_invalid_main`: F1=0.933  FP=1  FN=0  Sup=7

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.17%     0.21%
   4    10.12%    10.99%
   5    43.64%    44.50%
   9    45.06%    42.55%
  10    46.39%    44.33%
  16    47.52%    50.62%
  17    48.25%    51.32%
  18    51.14%    54.05%
```

### v2_particle_n100_260503_132203_0.65_0.65

**hparams**:
- `variant=V2` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`2`, n_classes=`34`, params=`1,155,362`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=64.76%  f1=**64.79%**
- VAL   acc=66.27%  f1=**65.43%**
- best epoch = 13, total epochs run = 20
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (22/34, F1 < 0.95):
    - `Edge-Bottom_scratch`: F1=0.000  FP=0  FN=8  Sup=8
    - `Center_scratch_21deg`: F1=0.125  FP=3  FN=11  Sup=12
    - `Edge-Top_scratch_21deg`: F1=0.167  FP=6  FN=4  Sup=5
    - `Edge-Top_bank_boundary`: F1=0.222  FP=7  FN=7  Sup=9
    - `Edge-Bottom_scratch_21deg`: F1=0.286  FP=2  FN=8  Sup=10
    - `Edge-Top_scratch`: F1=0.286  FP=6  FN=9  Sup=12
    - `Donut_bank_boundary`: F1=0.316  FP=5  FN=8  Sup=11
    - `Edge-Bottom_bank_boundary`: F1=0.345  FP=18  FN=1  Sup=6
    - `Full_bank_boundary`: F1=0.348  FP=5  FN=10  Sup=14
    - `Center_bank_boundary`: F1=0.387  FP=16  FN=3  Sup=9
    - `Center_scratch`: F1=0.421  FP=3  FN=8  Sup=12
    - `Full_scratch`: F1=0.444  FP=9  FN=6  Sup=12
    - `Donut_scratch_21deg`: F1=0.455  FP=1  FN=11  Sup=16
    - `Edge-Ring_bank_boundary`: F1=0.500  FP=3  FN=5  Sup=9
    - `Edge-Ring_scratch_21deg`: F1=0.500  FP=2  FN=4  Sup=7
    - `Edge-Ring_scratch`: F1=0.526  FP=5  FN=4  Sup=9
    - `Donut_scratch`: F1=0.585  FP=15  FN=2  Sup=14
    - `Full_scratch_21deg`: F1=0.667  FP=5  FN=3  Sup=11
    - `Edge-Top_particle_blast`: F1=0.800  FP=0  FN=2  Sup=6
    - `Edge-Bottom_particle_blast`: F1=0.857  FP=1  FN=1  Sup=7
    - `CommaCluster`: F1=0.875  FP=1  FN=1  Sup=8
    - `Donut_particle_blast`: F1=0.917  FP=1  FN=1  Sup=12

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.16%     0.24%
   4    28.82%    25.79%
   6    57.42%    60.05%
  12    64.95%    66.24%
  13    65.43%    64.79%
```

### v2_scratch_21deg_260503_171840_0.66_0.66

**hparams**:
- `variant=V2` | `n_per_class=100` | `obj_norm=5.0` | `target_id=5` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`2`, n_classes=`34`, params=`1,155,362`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=64.16%  f1=**65.53%**
- VAL   acc=68.37%  f1=**65.87%**
- best epoch = 24, total epochs run = 30

**TEST per-class (weak)**:
  - weak class (23/34, F1 < 0.95):
    - `Donut_particle_blast`: F1=0.190  FP=7  FN=10  Sup=12
    - `Donut_bank_boundary`: F1=0.222  FP=13  FN=8  Sup=11
    - `Full_scratch`: F1=0.250  FP=9  FN=9  Sup=12
    - `Edge-Top_particle_blast`: F1=0.267  FP=7  FN=4  Sup=6
    - `Full_bank_boundary`: F1=0.273  FP=5  FN=11  Sup=14
    - `Edge-Ring_scratch`: F1=0.286  FP=9  FN=6  Sup=9
    - `Donut_scratch`: F1=0.296  FP=9  FN=10  Sup=14
    - `Edge-Ring_bank_boundary`: F1=0.353  FP=5  FN=6  Sup=9
    - `Full_particle_blast`: F1=0.364  FP=10  FN=4  Sup=8
    - `Center_bank_boundary`: F1=0.421  FP=6  FN=5  Sup=9
    - `Edge-Top_scratch`: F1=0.455  FP=5  FN=7  Sup=12
    - `Edge-Bottom_bank_boundary`: F1=0.462  FP=4  FN=3  Sup=6
    - `Edge-Bottom_particle_blast`: F1=0.471  FP=6  FN=3  Sup=7
    - `Edge-Top_bank_boundary`: F1=0.500  FP=3  FN=5  Sup=9
    - `Center_scratch`: F1=0.526  FP=2  FN=7  Sup=12
    - `Center_particle_blast`: F1=0.571  FP=5  FN=7  Sup=15
    - `Edge-Bottom_scratch`: F1=0.571  FP=2  FN=4  Sup=8
    - `Edge-Ring_particle_blast`: F1=0.600  FP=3  FN=5  Sup=11
    - `Starburst`: F1=0.600  FP=5  FN=3  Sup=9
    - `Edge-Top_scratch_21deg`: F1=0.833  FP=2  FN=0  Sup=5
    - `CommaCluster`: F1=0.875  FP=1  FN=1  Sup=8
    - `Center_invalid_main`: F1=0.947  FP=1  FN=0  Sup=9
    - `Edge-Bottom_scratch_21deg`: F1=0.947  FP=0  FN=1  Sup=10

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.17%     0.21%
   4    19.11%    20.27%
   5    25.43%    24.74%
   6    28.67%    29.48%
   7    29.66%    29.04%
  11    50.05%    52.24%
  12    60.61%    64.44%
  14    61.49%    63.03%
  15    63.01%    64.87%
  18    64.76%    64.75%
  24    65.87%    65.53%
```

### v2_scratch_260503_165826_0.54_0.55

**hparams**:
- `variant=V2` | `n_per_class=100` | `obj_norm=5.0` | `target_id=4` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`2`, n_classes=`34`, params=`1,155,362`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=57.53%  f1=**53.91%**
- VAL   acc=56.02%  f1=**55.27%**
- best epoch = 5, total epochs run = 12
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (21/34, F1 < 0.95):
    - `Center_particle_blast`: F1=0.000  FP=0  FN=15  Sup=15
    - `Donut_particle_blast`: F1=0.000  FP=0  FN=12  Sup=12
    - `Edge-Bottom_bank_boundary`: F1=0.000  FP=0  FN=6  Sup=6
    - `Edge-Bottom_scratch_21deg`: F1=0.000  FP=0  FN=10  Sup=10
    - `Edge-Ring_bank_boundary`: F1=0.000  FP=0  FN=9  Sup=9
    - `Edge-Ring_particle_blast`: F1=0.000  FP=0  FN=11  Sup=11
    - `Edge-Top_particle_blast`: F1=0.000  FP=0  FN=6  Sup=6
    - `CommaCluster`: F1=0.069  FP=48  FN=6  Sup=8
    - `Full_bank_boundary`: F1=0.133  FP=0  FN=13  Sup=14
    - `Edge-Bottom_particle_blast`: F1=0.143  FP=6  FN=6  Sup=7
    - `Edge-Top_bank_boundary`: F1=0.154  FP=3  FN=8  Sup=9
    - `Edge-Top_scratch_21deg`: F1=0.222  FP=3  FN=4  Sup=5
    - `Donut_bank_boundary`: F1=0.296  FP=12  FN=7  Sup=11
    - `Center_bank_boundary`: F1=0.316  FP=7  FN=6  Sup=9
    - `Edge-Ring_scratch_21deg`: F1=0.412  FP=20  FN=0  Sup=7
    - `Starburst`: F1=0.462  FP=1  FN=6  Sup=9
    - `Full_particle_blast`: F1=0.471  FP=18  FN=0  Sup=8
    - `Center_scratch_21deg`: F1=0.500  FP=3  FN=7  Sup=12
    - `Donut_scratch_21deg`: F1=0.638  FP=16  FN=1  Sup=16
    - `Full_scratch_21deg`: F1=0.706  FP=0  FN=5  Sup=11
    - `Edge-Bottom_scratch`: F1=0.933  FP=0  FN=1  Sup=8

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.17%     0.21%
   4    22.17%    20.86%
   5    55.27%    53.91%
```

### v3_full_260503_160436_0.99_0.99

**hparams**:
- `variant=V3` | `n_per_class=220` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **7100**, 80/10/10 split
- per-class count: 대부분 220, 예외: {'Center_invalid_main': 200, 'CommaCluster': 200, 'Donut_invalid_main': 200, 'Edge-Bottom_invalid_main': 200, 'Edge-Ring_invalid_main': 200, 'Edge-Top_invalid_main': 200, 'Full_invalid_main': 200, 'Normal_bank_boundary': 20, 'Starburst': 200, 'Thick-Edge_invalid_main': 200}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=98.59%  f1=**98.66%**
- VAL   acc=99.44%  f1=**99.45%**
- best epoch = 5, total epochs run = 12
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (3/34, F1 < 0.95):
    - `Edge-Bottom_particle_blast`: F1=0.923  FP=2  FN=1  Sup=19
    - `Edge-Top_scratch`: F1=0.939  FP=3  FN=0  Sup=23
    - `Edge-Top_particle_blast`: F1=0.947  FP=1  FN=1  Sup=19

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.18%     0.23%
   4    98.86%    96.18%
   5    99.45%    98.66%
```

### v3_full_noema_260503_163158_0.98_0.99

**hparams**:
- `variant=V3` | `n_per_class=220` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.0`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **7100**, 80/10/10 split
- per-class count: 대부분 220, 예외: {'Center_invalid_main': 200, 'CommaCluster': 200, 'Donut_invalid_main': 200, 'Edge-Bottom_invalid_main': 200, 'Edge-Ring_invalid_main': 200, 'Edge-Top_invalid_main': 200, 'Full_invalid_main': 200, 'Normal_bank_boundary': 20, 'Starburst': 200, 'Thick-Edge_invalid_main': 200}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=98.17%  f1=**98.11%**
- VAL   acc=99.30%  f1=**99.26%**
- best epoch = 6, total epochs run = 13
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (5/34, F1 < 0.95):
    - `Edge-Bottom_scratch`: F1=0.903  FP=0  FN=3  Sup=17
    - `Edge-Bottom_particle_blast`: F1=0.905  FP=4  FN=0  Sup=19
    - `Edge-Bottom_bank_boundary`: F1=0.927  FP=0  FN=3  Sup=22
    - `CommaCluster`: F1=0.938  FP=2  FN=0  Sup=15
    - `Edge-Top_scratch_21deg`: F1=0.949  FP=0  FN=3  Sup=31

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.18%     0.23%
   3    98.03%    97.79%
   6    99.26%    98.11%
```

### v3_full_noise10_260503_174451_0.98_0.99

**hparams**:
- `variant=V3` | `n_per_class=220` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.1` | `chip_noise_eval=True` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **7100**, 80/10/10 split
- per-class count: 대부분 220, 예외: {'Center_invalid_main': 200, 'CommaCluster': 200, 'Donut_invalid_main': 200, 'Edge-Bottom_invalid_main': 200, 'Edge-Ring_invalid_main': 200, 'Edge-Top_invalid_main': 200, 'Full_invalid_main': 200, 'Normal_bank_boundary': 20, 'Starburst': 200, 'Thick-Edge_invalid_main': 200}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=97.75%  f1=**97.85%**
- VAL   acc=99.15%  f1=**99.17%**
- best epoch = 8, total epochs run = 15
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (5/34, F1 < 0.95):
    - `Edge-Bottom_bank_boundary`: F1=0.842  FP=0  FN=6  Sup=22
    - `Edge-Bottom_scratch_21deg`: F1=0.857  FP=6  FN=1  Sup=22
    - `Edge-Top_bank_boundary`: F1=0.913  FP=1  FN=3  Sup=24
    - `Edge-Top_scratch`: F1=0.917  FP=3  FN=1  Sup=23
    - `Edge-Bottom_scratch`: F1=0.938  FP=0  FN=2  Sup=17

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.18%     0.23%
   4    98.59%    94.88%
   5    98.68%    95.48%
   8    99.17%    97.85%
```

### v3_noise05_260503_151438_0.99_0.97

**hparams**:
- `variant=V3` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.05` | `chip_noise_eval=True` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=99.10%  f1=**99.10%**
- VAL   acc=96.69%  f1=**96.67%**
- best epoch = 5, total epochs run = 12
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (2/34, F1 < 0.95):
    - `CommaCluster`: F1=0.857  FP=0  FN=2  Sup=8
    - `Donut_particle_blast`: F1=0.923  FP=2  FN=0  Sup=12

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.17%     0.12%
   4    96.16%    98.70%
   5    96.67%    99.10%
```

### v3_noise10_260503_152034_0.99_0.97

**hparams**:
- `variant=V3` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.1` | `chip_noise_eval=True` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=99.10%  f1=**99.19%**
- VAL   acc=97.29%  f1=**97.07%**
- best epoch = 8, total epochs run = 15
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (2/34, F1 < 0.95):
    - `Edge-Bottom_invalid_main`: F1=0.933  FP=1  FN=0  Sup=7
    - `Edge-Top_bank_boundary`: F1=0.947  FP=1  FN=0  Sup=9

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.19%     0.19%
   4    94.07%    96.62%
   5    96.51%    98.32%
   8    97.07%    99.19%
```

### v3_noise20_260503_152725_0.96_0.96

**hparams**:
- `variant=V3` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.2` | `chip_noise_eval=True` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=96.99%  f1=**96.36%**
- VAL   acc=96.08%  f1=**95.95%**
- best epoch = 18, total epochs run = 25
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (8/34, F1 < 0.95):
    - `Edge-Bottom_scratch_21deg`: F1=0.700  FP=3  FN=3  Sup=10
    - `Edge-Bottom_scratch`: F1=0.800  FP=1  FN=2  Sup=8
    - `Edge-Top_particle_blast`: F1=0.833  FP=1  FN=1  Sup=6
    - `Edge-Bottom_bank_boundary`: F1=0.857  FP=2  FN=0  Sup=6
    - `Edge-Top_bank_boundary`: F1=0.889  FP=1  FN=1  Sup=9
    - `Edge-Top_scratch_21deg`: F1=0.909  FP=1  FN=0  Sup=5
    - `Edge-Bottom_particle_blast`: F1=0.923  FP=0  FN=1  Sup=7
    - `CommaCluster`: F1=0.933  FP=0  FN=1  Sup=8

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.19%     0.16%
   4    95.42%    96.83%
  11    95.56%    98.25%
  12    95.56%    97.34%
  15    95.71%    97.96%
  16    95.84%    98.69%
  18    95.95%    96.36%
```

### v3_obj_only_5ch_260503_165252_0.99_0.99

**hparams**:
- `variant=V3` | `n_per_class=220` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`5`, n_classes=`34`, params=`1,157,090`

**데이터**:
- 총 sample (capped): **7100**, 80/10/10 split
- per-class count: 대부분 220, 예외: {'Center_invalid_main': 200, 'CommaCluster': 200, 'Donut_invalid_main': 200, 'Edge-Bottom_invalid_main': 200, 'Edge-Ring_invalid_main': 200, 'Edge-Top_invalid_main': 200, 'Full_invalid_main': 200, 'Normal_bank_boundary': 20, 'Starburst': 200, 'Thick-Edge_invalid_main': 200}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=98.73%  f1=**98.72%**
- VAL   acc=99.44%  f1=**99.46%**
- best epoch = 6, total epochs run = 13
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (3/34, F1 < 0.95):
    - `Edge-Bottom_scratch`: F1=0.903  FP=0  FN=3  Sup=17
    - `Edge-Bottom_particle_blast`: F1=0.905  FP=4  FN=0  Sup=19
    - `Edge-Top_particle_blast`: F1=0.947  FP=1  FN=1  Sup=19

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.22%     0.11%
   4    98.87%    95.82%
   5    99.28%    98.84%
   6    99.46%    98.72%
```

### v3_objonly_noise10_260503_171819_0.98_0.99

**hparams**:
- `variant=V3` | `n_per_class=220` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.1` | `chip_noise_eval=True` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`5`, n_classes=`34`, params=`1,157,090`

**데이터**:
- 총 sample (capped): **7100**, 80/10/10 split
- per-class count: 대부분 220, 예외: {'Center_invalid_main': 200, 'CommaCluster': 200, 'Donut_invalid_main': 200, 'Edge-Bottom_invalid_main': 200, 'Edge-Ring_invalid_main': 200, 'Edge-Top_invalid_main': 200, 'Full_invalid_main': 200, 'Normal_bank_boundary': 20, 'Starburst': 200, 'Thick-Edge_invalid_main': 200}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=98.31%  f1=**98.40%**
- VAL   acc=98.73%  f1=**98.70%**
- best epoch = 8, total epochs run = 15
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (4/34, F1 < 0.95):
    - `Edge-Bottom_bank_boundary`: F1=0.900  FP=0  FN=4  Sup=22
    - `Edge-Bottom_scratch_21deg`: F1=0.913  FP=3  FN=1  Sup=22
    - `Edge-Top_bank_boundary`: F1=0.917  FP=2  FN=2  Sup=24
    - `Edge-Top_scratch`: F1=0.936  FP=2  FN=1  Sup=23

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.22%     0.11%
   4    98.02%    94.90%
   6    98.26%    95.14%
   8    98.70%    98.40%
```

### v3_onehot_n100_260503_132834_0.99_0.97

**hparams**:
- `variant=V3` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `seed=42` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=98.80%  f1=**98.79%**
- VAL   acc=96.99%  f1=**96.89%**
- best epoch = 6, total epochs run = 13
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (3/34, F1 < 0.95):
    - `CommaCluster`: F1=0.857  FP=0  FN=2  Sup=8
    - `Donut_particle_blast`: F1=0.923  FP=2  FN=0  Sup=12
    - `Edge-Bottom_invalid_main`: F1=0.933  FP=1  FN=0  Sup=7

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.19%     0.16%
   4    96.72%    98.70%
   6    96.89%    98.79%
```

### v3_seed100_260503_173539_0.99_0.98

**hparams**:
- `variant=V3` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=100` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=99.40%  f1=**99.35%**
- VAL   acc=98.49%  f1=**98.42%**
- best epoch = 14, total epochs run = 21
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (3/34, F1 < 0.95):
    - `Edge-Top_bank_boundary`: F1=0.909  FP=0  FN=1  Sup=6
    - `Edge-Top_particle_blast`: F1=0.923  FP=1  FN=1  Sup=13
    - `Center_particle_blast`: F1=0.947  FP=1  FN=0  Sup=9

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.25%     0.10%
   4    96.46%    99.62%
   5    97.01%    99.35%
   7    97.18%    99.35%
   8    97.43%    99.29%
  12    97.71%    98.68%
  13    97.83%    99.02%
  14    98.42%    99.35%
```

### v3_seed1_260503_172707_0.98_0.99

**hparams**:
- `variant=V3` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=1` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=98.49%  f1=**98.38%**
- VAL   acc=99.10%  f1=**99.01%**
- best epoch = 5, total epochs run = 12
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (4/34, F1 < 0.95):
    - `Edge-Bottom_particle_blast`: F1=0.857  FP=0  FN=2  Sup=8
    - `CommaCluster`: F1=0.923  FP=1  FN=0  Sup=6
    - `Edge-Bottom_bank_boundary`: F1=0.941  FP=1  FN=0  Sup=8
    - `Edge-Bottom_scratch`: F1=0.947  FP=1  FN=0  Sup=9

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.10%     0.14%
   2     0.16%     0.17%
   4    98.60%    97.41%
   5    99.01%    98.38%
```

### v3_seed234_260503_173544_0.98_0.99

**hparams**:
- `variant=V3` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=234` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=98.19%  f1=**98.26%**
- VAL   acc=99.40%  f1=**99.36%**
- best epoch = 13, total epochs run = 20
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (4/34, F1 < 0.95):
    - `Edge-Top_bank_boundary`: F1=0.824  FP=2  FN=1  Sup=8
    - `Edge-Bottom_bank_boundary`: F1=0.870  FP=1  FN=2  Sup=12
    - `Edge-Top_scratch`: F1=0.889  FP=1  FN=1  Sup=9
    - `Edge-Bottom_scratch`: F1=0.941  FP=1  FN=0  Sup=8

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.25%     0.16%
   4    97.77%    97.73%
   6    98.20%    98.05%
   8    98.64%    98.13%
  10    98.96%    97.61%
  11    99.01%    97.96%
  13    99.36%    98.26%
```

### v3_seed7_260503_172711_0.99_0.98

**hparams**:
- `variant=V3` | `n_per_class=100` | `obj_norm=5.0` | `target_id=3` | `chip_noise=0.0` | `chip_noise_eval=False` | `seed=7` | `epochs=30` | `batch=32` | `ema_decay=0.95`
- in_ch=`6`, n_classes=`34`, params=`1,157,666`

**데이터**:
- 총 sample (capped): **3320**, 80/10/10 split
- per-class count: 대부분 100, 예외: {'Normal_bank_boundary': 20}
- obj_id 분포 (chip 단위, n_wafers=25 샘플): bank_boundary 12.5% / invalid_main 55.0% / particle_blast 10.9% / scratch 9.6% / scratch_21deg 12.0%

**BEST OVERALL**:
- TEST  acc=98.80%  f1=**98.68%**
- VAL   acc=98.80%  f1=**98.21%**
- best epoch = 21, total epochs run = 28
  - **early stopped** (patience=7)

**TEST per-class (weak)**:
  - weak class (4/34, F1 < 0.95):
    - `Edge-Bottom_scratch_21deg`: F1=0.875  FP=2  FN=0  Sup=7
    - `Edge-Bottom_particle_blast`: F1=0.923  FP=0  FN=1  Sup=7
    - `Edge-Bottom_invalid_main`: F1=0.933  FP=0  FN=1  Sup=8
    - `Edge-Top_scratch`: F1=0.941  FP=1  FN=0  Sup=8

**BEST UPDATES** (매 best 갱신):
```
  ep    val_f1   test_f1
   1     0.25%     0.16%
   4    97.49%    98.28%
   6    97.66%    99.00%
   8    97.67%    98.68%
  11    97.89%    99.00%
  15    97.99%    98.68%
  21    98.21%    98.68%
```