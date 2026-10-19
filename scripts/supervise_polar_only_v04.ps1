param(
    [ValidateSet('fingerprint', 'graph')][string]$Encoder = 'fingerprint',
    [ValidateSet('overfit', 'smoke')][string]$Kind = 'overfit',
    [ValidateSet('train', 'evaluate')][string]$Action = 'train',
    [ValidateRange(1, 5000)][int]$Steps = 200,
    [Parameter(Mandatory)][string]$RunName,
    [string]$Checkpoint = '',
    [ValidateRange(30, 3600)][int]$TimeoutSeconds = 1200
)
$ErrorActionPreference = 'Stop'
$wslUserHome = (& wsl.exe -d Ubuntu-24.04 -- printenv HOME).Trim()
$projectLinux = "$wslUserHome/projects/phospholipid-structure-pretrain"
$projectShare = '\\wsl.localhost\Ubuntu-24.04' + $projectLinux.Replace('/', '\')
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid RunName' }
$rootShare = Join-Path $projectShare 'outputs/pretraining_v04'
$stopPath = Join-Path $rootShare 'GPU_STOP.json'
if (Test-Path -LiteralPath $stopPath) { throw 'A GPU stop is latched; no automatic retry' }
$runRelative = "outputs/pretraining_v04/runs/$RunName"
$runShare = Join-Path $projectShare $runRelative
if (Test-Path -LiteralPath $runShare) { throw 'Use a fresh RunName' }
if ($Kind -eq 'overfit' -and $Steps -gt 500) { throw 'Overfit cap is500 steps' }
if ($Action -eq 'evaluate' -and -not $Checkpoint) { throw 'Evaluation needs a checkpoint' }
New-Item -ItemType Directory -Path $runShare | Out-Null
$dumpDirs = @('WATCHDOG', 'AMD_WATCHDOG', 'AMD_REPORT_UM') | ForEach-Object {
    "C:\Windows\LiveKernelReports\$_"
}
function Get-DumpSnapshot {
    $snapshot = @{}
    foreach ($directory in $dumpDirs) {
        Get-ChildItem -LiteralPath $directory -File -ErrorAction SilentlyContinue | ForEach-Object {
            $snapshot[$_.FullName] = $_.LastWriteTimeUtc.Ticks
        }
    }
    return $snapshot
}
$epochPath = Join-Path $rootShare 'gpu_monitor_epoch_private.json'
if (-not (Test-Path -LiteralPath $epochPath)) {
    $epochUtc = (Get-Date).ToUniversalTime()
    $statePath = Join-Path $rootShare 'RUN_STATE.json'
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ($state.gpu_start_utc) { $epochUtc = [DateTimeOffset]::Parse($state.gpu_start_utc).UtcDateTime }
    }
    [System.IO.File]::WriteAllText($epochPath,
        (@{utc=$epochUtc.ToString('o')} | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
}
$epoch = Get-Content -LiteralPath $epochPath -Raw | ConvertFrom-Json
$startedUtc = [DateTimeOffset]::Parse($epoch.utc).UtcDateTime
$beforeDumps = Get-DumpSnapshot
$earlierFaults = @($beforeDumps.Keys | Where-Object { $beforeDumps[$_] -ge $startedUtc.Ticks })
if ($earlierFaults.Count -gt 0) {
    @{reason='gpu_fault_since_experiment_start';count=$earlierFaults.Count} |
        ConvertTo-Json | Set-Content -LiteralPath $stopPath -Encoding utf8
    throw 'GPU fault occurred since experiment start; refusing further GPU load'
}
$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = 'wsl.exe'
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$configName = if ($Encoder -eq 'fingerprint') { 'fp' } else { 'gine' }
$payload = @('-d', 'Ubuntu-24.04', '--', "$projectLinux/scripts/run_rocm.sh", '-u')
if ($Action -eq 'train') {
    $payload += @('train_polar_only_v04.py', '--steps', "$Steps")
    if ($Checkpoint) { $payload += @('--resume', $Checkpoint) }
} else {
    $payload += @('-m', 'structure_pretrain.polar_only_v04.evaluate', '--checkpoint', $Checkpoint)
}
$payload += @('--config', "configs/polar_only_v04_${configName}.yaml", '--kind', $Kind,
    '--output', $runRelative, '--attention-backend', 'math')
$payload | ForEach-Object { $startInfo.ArgumentList.Add($_) }
$child = [System.Diagnostics.Process]::new()
$child.StartInfo = $startInfo
$null = $child.Start()
$stdout = $child.StandardOutput.ReadToEndAsync()
$stderr = $child.StandardError.ReadToEndAsync()
$watch = [System.Diagnostics.Stopwatch]::StartNew()
$reason = $null
$newDumps = @()
$displayResets = @()
$lastEventPoll = 0.0
while (-not $child.HasExited) {
    Start-Sleep -Milliseconds 500
    $currentDumps = Get-DumpSnapshot
    $newDumps = @($currentDumps.Keys | Where-Object {
        -not $beforeDumps.ContainsKey($_) -or $beforeDumps[$_] -ne $currentDumps[$_]
    })
    if ($newDumps.Count -gt 0) { $reason = 'new_windows_gpu_fault_dump'; break }
    if ($watch.Elapsed.TotalSeconds - $lastEventPoll -gt 5) {
        $lastEventPoll = $watch.Elapsed.TotalSeconds
        $displayResets = @(Get-WinEvent -FilterHashtable @{
            LogName='System'; Id=4101; StartTime=$startedUtc
        } -MaxEvents 5 -ErrorAction SilentlyContinue)
        if ($displayResets.Count -gt 0) { $reason = 'windows_display_driver_reset'; break }
    }
    if (Test-Path -LiteralPath $stopPath) { $reason = 'external_stop_latch'; break }
    if ($watch.Elapsed.TotalSeconds -gt $TimeoutSeconds) { $reason = 'total_timeout'; break }
    $eventsPath = Join-Path $runShare 'events.jsonl'
    $age = $watch.Elapsed.TotalSeconds
    if (Test-Path -LiteralPath $eventsPath) {
        $age = ((Get-Date).ToUniversalTime() - (Get-Item -LiteralPath $eventsPath).LastWriteTimeUtc).TotalSeconds
    }
    if ($age -gt 20) { $reason = 'no_progress_for_20_seconds'; break }
}
if (-not $reason) {
    # Include delayed dumps between jobs and during process teardown.
    $finalDumps = Get-DumpSnapshot
    $newDumps = @($finalDumps.Keys | Where-Object { $finalDumps[$_] -ge $startedUtc.Ticks })
    $displayResets = @(Get-WinEvent -FilterHashtable @{
        LogName='System'; Id=4101; StartTime=$startedUtc
    } -MaxEvents 50 -ErrorAction SilentlyContinue)
    if ($newDumps.Count -gt 0) { $reason = 'new_windows_gpu_fault_dump' }
    if ($displayResets.Count -gt 0) { $reason = 'windows_display_driver_reset' }
}
if ($reason) {
    if (-not (Test-Path -LiteralPath $stopPath)) {
        @{ reason=$reason; run=$RunName; utc=(Get-Date).ToUniversalTime().ToString('o') } |
            ConvertTo-Json | Set-Content -LiteralPath $stopPath -Encoding utf8
    }
    $pidPath = Join-Path $runShare 'diagnostic.pid'
    if (Test-Path -LiteralPath $pidPath) {
        $trainingPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
        if ($trainingPid -match '^\d+$') {
            $commandLine = & wsl.exe -d Ubuntu-24.04 -- cat "/proc/$trainingPid/cmdline" 2>$null
            if ($commandLine -match '(train_polar_only_v04|polar_only_v04.evaluate)' -and $commandLine.Contains($runRelative)) {
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
    child_exit_code=$child.ExitCode; elapsed_seconds=$watch.Elapsed.TotalSeconds
    stop_reason=$reason; new_windows_gpu_dump_count=$newDumps.Count
    display_reset_count=$displayResets.Count; encoder=$Encoder; kind=$Kind
    action=$Action; training_steps_argument=$(if ($Action -eq 'train') { $Steps } else { $null })
    attention_backend='math'; precision='fp32'; experiment_monitor_start_utc=$startedUtc.ToString('o')
}
$evalPath = Join-Path $runShare 'evaluation_metrics.json'
if (Test-Path -LiteralPath $evalPath) {
    $report['evaluated_checkpoint_step'] = (Get-Content -LiteralPath $evalPath -Raw | ConvertFrom-Json).checkpoint_step
}
$report | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runShare 'host_monitor.json') -Encoding utf8
($standardOutput -split '\r?\n' | Select-Object -Last 8) -join "`n"
$standardError
$report | ConvertTo-Json
if ($reason -or $child.ExitCode -ne 0) { exit 1 }
