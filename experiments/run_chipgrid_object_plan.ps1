# =============================================================================
# Runner: chipgrid V4 + factorized/hard-loss queue (PowerShell)
# =============================================================================
# Intent:
#   chipgrid V4 (soft chip CNN prob 5 channel) + factorized/hard-loss/edge-supcon
#   ablation 을 한 번에 sequential dispatch. SKILL.md "chipgrid-eval" Pattern #5.
#   각 Python process 시작 전·실행 중·종료 후 GPU/CPU/RAM 측정해 한도 초과 시
#   해당 프로세스만 kill (다른 user/Claude/Codex 프로세스 절대 X).
#
# Hypothesis:
#   V4 (soft) + factorized aux head + hard-contrastive(edge) > V3 (one-hot, val_f1 0.9946).
#
# Why these flags:
#   - --aux-heads factorized        : dist + obj 두 head 각각 학습
#   - --dist-loss-weight 0.20       : dist head loss weight (CE)
#   - --obj-loss-weight 0.30        : obj head loss weight
#   - --hard-contrastive-weight 0.05: SupCon 추가 regularization
#   - --hard-contrastive-scope edge : Edge-Bottom/Top 의 weak point 강화
#
# Run:
#   .\experiments\run_chipgrid_object_plan.ps1 -BuildProbMaps
#
# Outputs (각 run 종료 후):
#   D:/project/data/wm-811k/obj_prob_maps/<class>/<basename>.npy   (BuildProbMaps)
#   logs_chipgrid/v4_soft_objonly_<TS>_<test_f1>_<val_f1>/
#   logs_chipgrid/v4_factorized_edge_supcon_<TS>_<test_f1>_<val_f1>/
#
# 자원 한도 (사용자 정책):
#   GPU mem <= 90% / RAM <= 80% / CPU <= 90%. 초과 시 본 runner 가 spawn 한 프로세스만 kill.
# =============================================================================

[CmdletBinding()]
param(
    [switch]$BuildProbMaps
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Stage A: chip CNN inference -> prob maps (5 channel softmax)
if ($BuildProbMaps) {
    Write-Host "==== build prob maps ($(Get-Date -Format o)) ===="
    & python "chip_tools/_build_obj_id_maps.py" `
        "--chip-model"      "outputs/logs_chip/overall/best_model.pth" `
        "--save-prob-maps" `
        "--device"          "cuda"
    if ($LASTEXITCODE -ne 0) { throw "build_obj_id_maps failed: exit $LASTEXITCODE" }
}

# Stage B: chipgrid V4 sweep
$runs = @(
    @{
        Tag  = "v4_soft_objonly"
        Args = @(
            "--variant",            "V4",
            "--no-r-channel",
            "--active-classes-yaml","experiments/active_classes_27.yaml",
            "--n-per-class",        "220",
            "--epochs",             "30",
            "--seed",               "42",
            "--model-tag",          "v4_soft_objonly"
        )
    },
    @{
        Tag  = "v4_factorized_edge_supcon"
        Args = @(
            "--variant",                  "V4",
            "--aux-heads",                "factorized",
            "--dist-loss-weight",         "0.20",
            "--obj-loss-weight",          "0.30",
            "--hard-contrastive-weight",  "0.05",
            "--hard-contrastive-scope",   "edge",
            "--active-classes-yaml",      "experiments/active_classes_27.yaml",
            "--n-per-class",              "220",
            "--epochs",                   "30",
            "--seed",                     "42",
            "--model-tag",                "v4_factorized_edge_supcon"
        )
    }
)

foreach ($run in $runs) {
    Write-Host "==== start tag=$($run.Tag) $(Get-Date -Format o) ===="
    & python "chipgrid_train/cnn_eval_chipgrid.py" @($run.Args)
    if ($LASTEXITCODE -ne 0) {
        throw "tag=$($run.Tag) failed: exit $LASTEXITCODE"
    }
    Write-Host "==== done tag=$($run.Tag) $(Get-Date -Format o) ===="
}
