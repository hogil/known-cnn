param(
    [switch]$Smoke,
    [switch]$RunFullSweep
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# iter124 - 4-single-source FCM-PM distribution separation sweep.
#
# Absolute rule:
#   - train source remains 4 single defect classes only via classification_chips
#   - no Normal/Invalid/OOD/real combo/synthetic combo roots are used for training
#   - FCM-PM/CutMix pseudo-combo is generated on the fly from 4 single chips

Set-Location "D:\project\known-cnn"

$RunLog = "outputs\_iter124_fcmpm_distribution_sweep.log"
$V15 = "D:\project\data\wm-811k\chip_multilabel_v15direct"
$Backbone = "convnextv2_base.fcmae_ft_in22k_in1k_384"
$BackboneWeights = "mega_matrix\weights\$Backbone.pth"

if (-not $Smoke -and -not $RunFullSweep) {
    throw "Refusing to run full sweep by default. Use -Smoke for epoch=1 smoke, or -RunFullSweep only with explicit approval."
}

New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
"$(Get-Date) [iter124] start 4-single-source FCM-PM distribution sweep" | Set-Content -Path $RunLog -Encoding UTF8

function Train-Eval {
    param(
        [Parameter(Mandatory=$true)][string]$Tag,
        [Parameter(Mandatory=$true)][string[]]$TrainArgs,
        [string]$DataRoot = "",
        [int]$EvalNPerClass = 200
    )
    $OutRoot = "outputs\iter124$Tag"
    if (Test-Path $OutRoot) {
        "$(Get-Date) [iter124-$Tag] skip exists" | Add-Content -Path $RunLog -Encoding UTF8
        return
    }

    "$(Get-Date) [iter124-$Tag] $($TrainArgs -join ' ')" | Add-Content -Path $RunLog -Encoding UTF8
    $BaseArgs = @(
        "-u", "-m", "chip_multilabel._train_chip_variant",
        "--batch", "2", "--accum", "8", "--seed", "1",
        "--lr", "1e-4", "--no-normal", "--val-criterion", "margin_max",
        "--backbone-timm", $Backbone, "--img-size", "384",
        "--out-root", $OutRoot, "--tag", "iter124$Tag"
    )
    if (Test-Path $BackboneWeights) {
        $BaseArgs += @("--backbone-timm-weights", $BackboneWeights)
    }
    if ($DataRoot) {
        $BaseArgs += @("--data-root", $DataRoot)
    }
    & python @BaseArgs @TrainArgs *>> $RunLog
    if ($LASTEXITCODE -ne 0) {
        "$(Get-Date) [iter124-$Tag] TRAIN_FAIL rc=$LASTEXITCODE" | Add-Content -Path $RunLog -Encoding UTF8
        return
    }

    $Run = Get-ChildItem -Path $OutRoot -Directory -Filter "T*" | Select-Object -First 1
    if ($null -eq $Run) {
        "$(Get-Date) [iter124-$Tag] FAIL no run dir" | Add-Content -Path $RunLog -Encoding UTF8
        return
    }

    foreach ($Ck in @("best_model", "final_epoch_model")) {
        $Ckpt = Join-Path $Run.FullName "$Ck.pth"
        if (-not (Test-Path $Ckpt)) { continue }
        $OutEval = Join-Path $Run.FullName "eval_v15direct_n${EvalNPerClass}_$Ck"
        if (Test-Path $OutEval) { continue }
        & python -u -m chip_multilabel.run_stage1 --model $Ckpt `
            --eval-set $V15 --out-root $OutEval `
            --variants "I3,I7,I10,I13" --n-per-class $EvalNPerClass `
            --strength-min 0.0 --strength-max 1.0 --seed 42 `
            *>> $RunLog
        if ($LASTEXITCODE -ne 0) {
            "$(Get-Date) [iter124-$Tag] EVAL_FAIL $Ck rc=$LASTEXITCODE" | Add-Content -Path $RunLog -Encoding UTF8
        }
    }
    "$(Get-Date) [iter124-$Tag] DONE" | Add-Content -Path $RunLog -Encoding UTF8
}

if ($Smoke) {
    $SmokeData = "outputs\_smoke_iter124_4single_data"
    $SmokeOut = "outputs\iter124SMOKE_p025_g3"
    if (Test-Path $SmokeData) {
        Remove-Item -LiteralPath $SmokeData -Recurse -Force
    }
    if (Test-Path $SmokeOut) {
        Remove-Item -LiteralPath $SmokeOut -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $SmokeData | Out-Null
    foreach ($ClassName in @("bank_boundary", "fork", "scratch", "scratch_rot")) {
        $Dst = Join-Path $SmokeData $ClassName
        New-Item -ItemType Directory -Force -Path $Dst | Out-Null
        Get-ChildItem -Path "D:\project\data\wm-811k\classification_chips\$ClassName" -Filter "*.png" |
            Sort-Object Name |
            Select-Object -First 4 |
            Copy-Item -Destination $Dst
    }
    Train-Eval "SMOKE_p025_g3" @("--variant","T7","--ls","0.20","--epochs","1",
        "--batch","1","--accum","1","--freeze-backbone",
        "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
        "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5") $SmokeData 5
    "$(Get-Date) [iter124] SMOKE DONE" | Add-Content -Path $RunLog -Encoding UTF8
    exit 0
}

# A. FCM-PM signal strength. Baseline neighborhood, one axis at a time.
Train-Eval "A_p015" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.15","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

Train-Eval "B_p025_base" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

Train-Eval "C_p035" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.35","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

Train-Eval "D_p040" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.40","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

Train-Eval "E_g2" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","2","--cutmix-complete-label-scale","0.5")

Train-Eval "F_g4" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","4","--cutmix-complete-label-scale","0.5")

Train-Eval "G_cls07" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.7")

Train-Eval "H_cls10" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","1.0")

Train-Eval "I_ab1008" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5",
    "--cutmix-ab-labels","1.0,0.8")

Train-Eval "J_ab1010" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5",
    "--cutmix-ab-labels","1.0,1.0")

# B. Weak-pair focus: fork+scratch lower-tail boost.
Train-Eval "K_bias_fs2" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5",
    "--cutmix-pair-bias","fork,scratch:2")

Train-Eval "L_bias_fs3" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5",
    "--cutmix-pair-bias","fork,scratch:3")

# C. Spatial coherence alternatives.
Train-Eval "M_bisect_h" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","bisect_h","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-complete-label-scale","0.5")

Train-Eval "N_bisect_v" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","bisect_v","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-complete-label-scale","0.5")

Train-Eval "O_bisect_rand" @("--variant","T7","--ls","0.20","--epochs","10",
    "--cutmix-mode","bisect_rand","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-complete-label-scale","0.5")

# D. Calibration/regularization checks.
Train-Eval "P_ls10" @("--variant","T7","--ls","0.10","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

Train-Eval "Q_ls30" @("--variant","T7","--ls","0.30","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

Train-Eval "R_dp003" @("--variant","T7","--ls","0.20","--epochs","10","--drop-path-rate","0.03",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

Train-Eval "S_dp005" @("--variant","T7","--ls","0.20","--epochs","10","--drop-path-rate","0.05",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

Train-Eval "T_pos125" @("--variant","T7","--ls","0.20","--epochs","10",
    "--pos-weight","fork:1.25,scratch:1.25",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

Train-Eval "U_pos150" @("--variant","T7","--ls","0.20","--epochs","10",
    "--pos-weight","fork:1.5,scratch:1.5",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

Train-Eval "V_t9_focal" @("--variant","T9","--ls","0.00","--epochs","10",
    "--cutmix-mode","complement","--cutmix-pair","masked","--cutmix-pair-fill","corner",
    "--cutmix-p","0.25","--cutmix-n-groups","3","--cutmix-complete-label-scale","0.5")

# Refresh absolute-rule table and val2-margin audit for completed iter124 cells.
& python -X utf8 _reeval_absolute_rule.py *>> $RunLog

$Models = "outputs\_iter124_val2_margin_models.txt"
"" | Set-Content -Path $Models -Encoding UTF8
Get-ChildItem -Path "outputs" -Directory -Filter "iter124*" | ForEach-Object {
    $Out = $_
    Get-ChildItem -Path $Out.FullName -Directory -Filter "T*" | ForEach-Object {
        $Run = $_
        $Tag = $Out.Name
        $Best = Join-Path $Run.FullName "best_model.pth"
        $Final = Join-Path $Run.FullName "final_epoch_model.pth"
        if (Test-Path $Best) { "${Tag}_best=${Best}" | Add-Content -Path $Models -Encoding UTF8 }
        if (Test-Path $Final) { "${Tag}_final=${Final}" | Add-Content -Path $Models -Encoding UTF8 }
    }
}

$ModelLines = @(Get-Content -Path $Models | Where-Object { $_.Trim().Length -gt 0 })
if ($ModelLines.Count -gt 0) {
    & python -X utf8 _run_iter123_val2_margin_audit.py `
        --models-file $Models `
        --eval-set $V15 `
        --n-per-class 200 `
        --strength-min 0.0 --strength-max 1.0 `
        --seed 42 --val-ratio 0.2 `
        --q-low 0.05 --q-high 0.95 `
        --alphas "0.50,0.65,0.80" `
        --out-prefix "outputs\_iter124_val2_margin" `
        *>> $RunLog
}

"$(Get-Date) [iter124] DONE" | Add-Content -Path $RunLog -Encoding UTF8
