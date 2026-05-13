param(
    [switch]$Smoke,
    [switch]$RunFullAudit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# iter123 - val 2-combo bit-margin audit (no training)
# Source rule: audited checkpoints must be trained from 4 single defect classes only.

Set-Location "D:\project\known-cnn"

$RunLog = "outputs\_iter123_val2_margin_audit.log"
$Models = "outputs\_iter123_val2_margin_models.txt"
$V15 = "D:\project\data\wm-811k\chip_multilabel_v15direct"

if (-not $Smoke -and -not $RunFullAudit) {
    throw "Refusing to run full audit by default. Use -Smoke for n_per_class=5 smoke, or -RunFullAudit only with explicit approval."
}

New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
"$(Get-Date) [iter123] start val2 margin audit" | Set-Content -Path $RunLog -Encoding UTF8
"" | Set-Content -Path $Models -Encoding UTF8

function Add-Model {
    param(
        [Parameter(Mandatory=$true)][string]$Tag,
        [Parameter(Mandatory=$true)][string]$Ckpt
    )
    if (Test-Path $Ckpt) {
        "${Tag}=${Ckpt}" | Add-Content -Path $Models -Encoding UTF8
        "$(Get-Date) [iter123] add $Tag" | Add-Content -Path $RunLog -Encoding UTF8
    } else {
        "$(Get-Date) [iter123] skip missing ${Tag}: $Ckpt" | Add-Content -Path $RunLog -Encoding UTF8
    }
}

Add-Model "iter112_best_ep06" "outputs\iter112_ep20\T7_iter112_ep20_260512_214618\best_model.pth"
if ($RunFullAudit) {
    # Known reference points. Extend this list as new cells finish.
    Add-Model "iter112_final_ep20" "outputs\iter112_ep20\T7_iter112_ep20_260512_214618\final_epoch_model.pth"
    Add-Model "iter120A_baseline" "outputs\iter120A_baseline\T7_iter120A_baseline_260513_032245\best_model.pth"
    Add-Model "iter120B_dp005" "outputs\iter120B_dp005\T7_iter120B_dp005_260513_033255\best_model.pth"
    Add-Model "iter120C_dp010" "outputs\iter120C_dp010\T7_iter120C_dp010_260513_034134\best_model.pth"
    Add-Model "iter120D_p015" "outputs\iter120D_p015\T7_iter120D_p015_260513_034929\best_model.pth"
    Add-Model "iter121A_p040" "outputs\iter121A_p040\T7_iter121A_p040_260513_055923\best_model.pth"
    Add-Model "iter121B_p060" "outputs\iter121B_p060\T7_iter121B_p060_260513_061051\best_model.pth"
    Add-Model "iter121C_cls10" "outputs\iter121C_cls10\T7_iter121C_cls10_260513_062131\best_model.pth"
    Add-Model "iter121D_ab1008" "outputs\iter121D_ab1008\T7_iter121D_ab1008_260513_063315\best_model.pth"
    Add-Model "iter121E_ab1010" "outputs\iter121E_ab1010\T7_iter121E_ab1010_260513_064451\best_model.pth"
    Add-Model "iter121F_ep15" "outputs\iter121F_ep15\T7_iter121F_ep15_260513_075003\best_model.pth"
    Add-Model "iter121G_ep20" "outputs\iter121G_ep20\T7_iter121G_ep20_260513_080328\best_model.pth"
}

$ModelLines = @(Get-Content -Path $Models | Where-Object { $_.Trim().Length -gt 0 })
if ($ModelLines.Count -eq 0) {
    "$(Get-Date) [iter123] FAIL no checkpoints found" | Add-Content -Path $RunLog -Encoding UTF8
    throw "No checkpoints found for iter123 audit"
}

$NPerClass = if ($Smoke) { 5 } else { 200 }
$OutPrefix = if ($Smoke) { "outputs\_iter123_smoke_val2_margin" } else { "outputs\_iter123_val2_margin" }

& python -X utf8 _run_iter123_val2_margin_audit.py `
    --models-file $Models `
    --eval-set $V15 `
    --n-per-class $NPerClass `
    --strength-min 0.0 --strength-max 1.0 `
    --seed 42 --val-ratio 0.2 `
    --q-low 0.05 --q-high 0.95 `
    --alphas "0.50,0.65,0.80" `
    --out-prefix $OutPrefix `
    *>> $RunLog

if ($LASTEXITCODE -ne 0) {
    throw "iter123 audit failed with exit code $LASTEXITCODE"
}

"$(Get-Date) [iter123] DONE" | Add-Content -Path $RunLog -Encoding UTF8
