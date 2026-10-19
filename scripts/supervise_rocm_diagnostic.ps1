param(
    [ValidateSet('fingerprint', 'graph')][string]$Encoder = 'fingerprint',
    [ValidateSet('diagnostic', 'overfit', 'smoke', 'evaluate', 'midterm')][string]$Job = 'diagnostic',
    [ValidateRange(1, 5000)][int]$Steps = 8,
    [string]$RunName = 'math_fp_8',
    [string]$Resume = '',
    [string]$ReuseCompletedRawFrom = '',
    [ValidateRange(10, 3600)][int]$TimeoutSeconds = 45
)
$ErrorActionPreference = 'Stop'
$wslUserHome = (& wsl.exe -d Ubuntu-24.04 -- printenv HOME).Trim()
$projectLinux = "$wslUserHome/projects/phospholipid-structure-pretrain"
$projectShare = '\\wsl.localhost\Ubuntu-24.04' + $projectLinux.Replace('/', '\')
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must contain only letters, numbers, underscore or hyphen' }
$runRelative = "outputs/pretraining_v03/rocm_diagnosis/$RunName"
if ($Job -eq 'overfit') { $runRelative = "outputs/pretraining_v03/rocm_overfit/$RunName" }
if ($Job -eq 'evaluate') { $runRelative = "outputs/midterm_v03/evaluation/$RunName" }
if ($Job -eq 'midterm') { $runRelative = "outputs/midterm_v03/training/$RunName" }
if ($Job -in @('evaluate', 'midterm') -and -not $Resume) { throw 'A checkpoint is required' }
if ($Job -eq 'smoke') {
    $runRelative = "outputs/pretraining_v03/rocm_smoke/$RunName"
    $gatePath = Join-Path $projectShare "outputs/pretraining_v03/rocm_overfit/${Encoder}_math/overfit_report.json"
    if (-not (Test-Path -LiteralPath $gatePath)) { throw "Missing real overfit report: $gatePath" }
    $gate = Get-Content -LiteralPath $gatePath -Raw | ConvertFrom-Json
    if ($gate.status -ne 'passed') { throw 'Real overfit must pass before smoke' }
}
if ($Job -eq 'diagnostic' -and $Steps -gt 50) { throw 'Diagnostic steps must be at most 50' }
$runShare = Join-Path $projectShare $runRelative
if (Test-Path -LiteralPath $runShare) { throw "Use a new RunName: $RunName" }
New-Item -ItemType Directory -Path $runShare | Out-Null
$dumpDirs = @('WATCHDOG', 'AMD_WATCHDOG', 'AMD_REPORT_UM') | ForEach-Object { "C:\Windows\LiveKernelReports\$_" }
function Get-DumpSnapshot {
    $snapshot = @{}
    foreach ($directory in $dumpDirs) {
        Get-ChildItem -LiteralPath $directory -File -ErrorAction SilentlyContinue | ForEach-Object {
            $snapshot[$_.FullName] = $_.LastWriteTimeUtc.Ticks
        }
    }
    return $snapshot
}
$beforeDumps = Get-DumpSnapshot
$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = 'wsl.exe'
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$payload = @('-d', 'Ubuntu-24.04', '--', "$projectLinux/scripts/run_rocm.sh", '-u')
if ($Job -eq 'diagnostic') {
    $payload += @('-m', 'scripts.diagnose_rocm_v03', '--encoder', $Encoder, '--steps', "$Steps")
} elseif ($Job -eq 'overfit') {
    $payload += @('scripts/overfit_pretraining_v03.py', '--encoder', $Encoder,
        '--targets', '32', '--max-steps', "$Steps", '--batch-size', '8', '--synchronize')
} elseif ($Job -eq 'evaluate') {
    $payload += @('-m', 'scripts.evaluate_checkpoint_v03', '--checkpoint', $Resume)
    if ($ReuseCompletedRawFrom) { $payload += @('--reuse-completed-raw-from', $ReuseCompletedRawFrom) }
} else {
    $configName = if ($Encoder -eq 'fingerprint') { 'fp' } else { 'gine' }
    $payload += @('train_structure_real.py', '--device', 'rocm', '--steps', "$Steps",
        '--config', "configs/pretraining_v03_rocm_${configName}.yaml")
    if ($Job -eq 'midterm') {
        $payload += @('--sample-rows-from', "outputs/pretraining_v03/rocm_smoke/${Encoder}_resume/sample_rows",
            '--train-only', '--retain-periodic-checkpoints')
    }
}
$payload += @('--attention-backend', 'math', '--output', $runRelative)
if ($Resume -and $Job -ne 'evaluate') {
    if ($Job -notin @('smoke', 'midterm')) { throw 'Resume requires smoke or midterm' }
    $payload += @('--resume', $Resume)
}
$payload | ForEach-Object { $startInfo.ArgumentList.Add($_) }
$child = [System.Diagnostics.Process]::new()
$child.StartInfo = $startInfo
$null = $child.Start()
$stdout = $child.StandardOutput.ReadToEndAsync()
$stderr = $child.StandardError.ReadToEndAsync()
$watch = [System.Diagnostics.Stopwatch]::StartNew()
$reason = $null
$newDumps = @()
while (-not $child.HasExited) {
    Start-Sleep -Milliseconds 500
    $currentDumps = Get-DumpSnapshot
    $newDumps = @($currentDumps.Keys | Where-Object {
        -not $beforeDumps.ContainsKey($_) -or $beforeDumps[$_] -ne $currentDumps[$_]
    })
    if ($newDumps.Count -gt 0) { $reason = 'new_windows_gpu_fault_dump'; break }
    if ($watch.Elapsed.TotalSeconds -gt $TimeoutSeconds) { $reason = 'total_timeout'; break }
    $eventsPath = Join-Path $runShare 'events.jsonl'
    if (Test-Path -LiteralPath $eventsPath) {
        $age = ((Get-Date).ToUniversalTime() - (Get-Item -LiteralPath $eventsPath).LastWriteTimeUtc).TotalSeconds
        if ($age -gt 20) { $reason = 'no_progress_for_20_seconds'; break }
    }
}
if ($reason) {
    $pidPath = Join-Path $runShare 'diagnostic.pid'
    if (Test-Path -LiteralPath $pidPath) {
        $trainingPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
        if ($trainingPid -match '^\d+$') {
            $commandLine = & wsl.exe -d Ubuntu-24.04 -- cat "/proc/$trainingPid/cmdline" 2>$null
            if ($commandLine -match '(diagnose_rocm_v03|overfit_pretraining_v03|train_structure_real|evaluate_checkpoint_v03)' -and $commandLine.Contains($runRelative)) {
                & wsl.exe -d Ubuntu-24.04 -- kill -TERM $trainingPid 2>$null
            }
        }
    }
    if (-not $child.HasExited) { $child.Kill() }
}
$child.WaitForExit()
$standardOutput = $stdout.GetAwaiter().GetResult()
$standardError = $stderr.GetAwaiter().GetResult()
Set-Content -LiteralPath (Join-Path $runShare 'stdout.log') -Value $standardOutput -Encoding utf8
Set-Content -LiteralPath (Join-Path $runShare 'stderr.log') -Value $standardError -Encoding utf8
$report = @{
    child_exit_code = $child.ExitCode
    elapsed_seconds = $watch.Elapsed.TotalSeconds
    stop_reason = $reason
    new_windows_gpu_dumps = $newDumps
    encoder = $Encoder
    job = $Job
    steps = $Steps
    attention_backend = 'math'
}
$report | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runShare 'host_monitor.json') -Encoding utf8
($standardOutput -split '\r?\n' | Select-Object -Last 30) -join "`n"
$standardError
$report | ConvertTo-Json -Depth 4
if ($reason -or $child.ExitCode -ne 0) { exit 1 }
