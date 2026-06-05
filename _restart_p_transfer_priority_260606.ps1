$ErrorActionPreference = "Stop"

$repo = "D:\project\known-cnn"
$parentPid = 12676
$childPid = 42448
$log = Join-Path $repo "_restart_p_transfer_priority_260606.log"

function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $log -Value "[$ts] $msg"
}

Log "watch start parent=$parentPid child=$childPid"

while (Get-Process -Id $childPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

Log "child finished; stopping old recipe_sweep parent if still alive"
$parent = Get-CimInstance Win32_Process -Filter "ProcessId=$parentPid" -ErrorAction SilentlyContinue
if ($parent -and $parent.CommandLine -match "chip_multilabel\.recipe_sweep") {
    Stop-Process -Id $parentPid -Force
    Log "stopped old parent=$parentPid"
} else {
    Log "old parent not found or no longer recipe_sweep"
}

Start-Sleep -Seconds 3

$out = Join-Path $repo "_recipe_sweep_ptransfer_priority_260606.out.log"
$err = Join-Path $repo "_recipe_sweep_ptransfer_priority_260606.err.log"
$cmd = @"
cd /d/project/known-cnn
`$env:CHIP_SWEEP_QUEUE_MODE='one_axis_ablation'
python -u -m chip_multilabel.recipe_sweep --datasets frozen_original,sota_gapstress_seed31_260531,sota_gapstress_seed97_260531,frozen_original_200_snapshot,frozen_original_2015_candidate --diag-device cuda --forever > '$out' 2> '$err'
"@

Start-Process -FilePath "powershell" -WindowStyle Hidden -WorkingDirectory $repo -ArgumentList @("-NoProfile", "-Command", $cmd)
Log "started p-transfer-priority recipe_sweep"
