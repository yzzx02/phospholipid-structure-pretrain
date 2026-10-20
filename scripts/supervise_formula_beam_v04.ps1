param(
    [Parameter(Mandatory)][ValidateSet('fingerprint', 'graph')][string]$Encoder,
    [Parameter(Mandatory)][ValidateSet('raw', 'elements', 'elements_grammar')][string]$Mode,
    [ValidateSet(5, 20, 50, 128)][int]$BeamSize = 5,
    [ValidateRange(0, 511)][int]$Start = 0,
    [ValidateRange(1, 512)][int]$Count = 512,
    [Parameter(Mandatory)][string]$RunName,
    [ValidateRange(30, 1200)][int]$TimeoutSeconds = 1200
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$' -or $Start + $Count -gt 512) {
    throw 'Invalid fresh run name or cohort bounds'
}
$projectLinux = '/home/administrator/projects/phospholipid-structure-pretrain'
$projectShare = '\\wsl.localhost\Ubuntu-24.04' + $projectLinux.Replace('/', '\')
$experiment = 'outputs/formula_constrained_v04_20261020'
$rootShare = Join-Path $projectShare $experiment
$runRelative = "$experiment/$RunName"
$runShare = Join-Path $projectShare $runRelative
$stopPath = Join-Path $rootShare 'GPU_STOP.json'
$oldStopPath = Join-Path $projectShare 'outputs/pretraining_v04/GPU_STOP.json'
if ((Test-Path -LiteralPath $stopPath) -or (Test-Path -LiteralPath $oldStopPath)) {
    throw 'GPU stop latched; automatic retry is prohibited'
}
if (Test-Path -LiteralPath $runShare) { throw 'Fresh RunName required' }
$gate = Get-Content -LiteralPath (Join-Path $rootShare 'cpu_gate.json') -Raw | ConvertFrom-Json
if (-not $gate.passed) { throw 'CPU checks and both checkpoint audits must pass first' }

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
function Get-DisplayResets([datetime]$Since) {
    return @(Get-WinEvent -FilterHashtable @{
        LogName='System'; Id=4101; StartTime=$Since.ToLocalTime()
    } -MaxEvents 100 -ErrorAction SilentlyContinue | Where-Object {
        $_.TimeCreated.ToUniversalTime() -ge $Since
    })
}
function Write-JsonFile($Path, $Value) {
    [System.IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 12),
        [System.Text.UTF8Encoding]::new($false))
}
function Set-StopLatch([string]$Reason) {
    if (-not (Test-Path -LiteralPath $stopPath)) {
        Write-JsonFile $stopPath @{
            reason=$Reason; run=$RunName; utc=(Get-Date).ToUniversalTime().ToString('o')
        }
    }
}

# This inference experiment has its own persistent epoch. Never reset it between jobs.
# Earlier host incidents remain recorded, but this is a newly authorized GPU task.
$epochPath = Join-Path $rootShare 'gpu_monitor_epoch_private.json'
if (-not (Test-Path -LiteralPath $epochPath)) {
    Write-JsonFile $epochPath @{
        utc=(Get-Date).ToUniversalTime().ToString('o')
        existing_dumps=(Get-DumpSnapshot)
        driver=@(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
    }
}
$epoch = Get-Content -LiteralPath $epochPath -Raw | ConvertFrom-Json -DateKind String
$startedUtc = [DateTimeOffset]::Parse($epoch.utc).UtcDateTime
$beforeDumps = Get-DumpSnapshot
$earlierFaults = @($beforeDumps.Keys | Where-Object { $beforeDumps[$_] -ge $startedUtc.Ticks })
$earlierResets = @(Get-DisplayResets $startedUtc)
if ($earlierFaults.Count -or $earlierResets.Count) {
    Set-StopLatch 'gpu_fault_since_formula_experiment_start'
    throw 'A GPU fault occurred since this experiment began; refusing further GPU load'
}

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = 'wsl.exe'
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$payload = @('-d', 'Ubuntu-24.04', '--', "$projectLinux/scripts/run_rocm.sh", '-u',
    '-m', 'structure_pretrain.polar_only_v04.formula_beam_eval',
    '--encoder', $Encoder, '--mode', $Mode, '--beam-size', "$BeamSize",
    '--start', "$Start", '--count', "$Count", '--device', 'cuda', '--output', $runRelative)
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
    if ($newDumps.Count) { $reason = 'new_windows_gpu_fault_dump'; break }
    if ($watch.Elapsed.TotalSeconds - $lastEventPoll -ge 5) {
        $lastEventPoll = $watch.Elapsed.TotalSeconds
        $displayResets = @(Get-DisplayResets $startedUtc)
        if ($displayResets.Count) { $reason = 'windows_display_driver_reset'; break }
    }
    if ((Test-Path -LiteralPath $stopPath) -or (Test-Path -LiteralPath $oldStopPath)) {
        $reason = 'external_stop_latch'; break
    }
    if ($watch.Elapsed.TotalSeconds -gt $TimeoutSeconds) { $reason = 'total_timeout'; break }
    $eventsPath = Join-Path $runShare 'events.jsonl'
    $age = $watch.Elapsed.TotalSeconds
    if (Test-Path -LiteralPath $eventsPath) {
        $age = ((Get-Date).ToUniversalTime() -
            (Get-Item -LiteralPath $eventsPath).LastWriteTimeUtc).TotalSeconds
    }
    if ($age -gt 20) { $reason = 'no_progress_for_20_seconds'; break }
}
# Catch late reports, including faults that occur in the gap between jobs.
$finalDumps = Get-DumpSnapshot
$newDumps = @($finalDumps.Keys | Where-Object { $finalDumps[$_] -ge $startedUtc.Ticks })
$displayResets = @(Get-DisplayResets $startedUtc)
if ($newDumps.Count) { $reason = 'new_windows_gpu_fault_dump' }
if ($displayResets.Count) { $reason = 'windows_display_driver_reset' }
if ($reason) {
    Set-StopLatch $reason
    $pidPath = Join-Path $runShare 'diagnostic.pid'
    if (Test-Path -LiteralPath $pidPath) {
        $evaluationPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
        if ($evaluationPid -match '^\d+$') {
            $commandLine = & wsl.exe -d Ubuntu-24.04 -- cat "/proc/$evaluationPid/cmdline" 2>$null
            if ($commandLine -match 'structure_pretrain.polar_only_v04.formula_beam_eval' -and
                $commandLine.Contains($runRelative)) {
                & wsl.exe -d Ubuntu-24.04 -- kill -TERM $evaluationPid 2>$null
                Start-Sleep -Milliseconds 500
                $remaining = & wsl.exe -d Ubuntu-24.04 -- cat "/proc/$evaluationPid/cmdline" 2>$null
                if ($remaining -and $remaining.Contains($runRelative)) {
                    & wsl.exe -d Ubuntu-24.04 -- kill -KILL $evaluationPid 2>$null
                }
            }
        }
    }
    if (-not $child.HasExited) { $child.Kill() }
}
$child.WaitForExit()
$standardOutput = $stdout.GetAwaiter().GetResult()
$standardError = $stderr.GetAwaiter().GetResult()
if (-not (Test-Path -LiteralPath $runShare)) {
    New-Item -ItemType Directory -Path $runShare | Out-Null
}
[System.IO.File]::WriteAllText((Join-Path $runShare 'stdout.log'), $standardOutput)
[System.IO.File]::WriteAllText((Join-Path $runShare 'stderr.log'), $standardError)
$report = @{
    child_exit_code=$child.ExitCode; elapsed_seconds=$watch.Elapsed.TotalSeconds
    stop_reason=$reason; new_windows_gpu_dump_count=$newDumps.Count
    display_reset_count=$displayResets.Count; encoder=$Encoder; mode=$Mode
    beam_size=$BeamSize; start=$Start; count=$Count; training_started=$false
    attention_backend='math'; precision='fp32'; per_job_timeout_seconds=$TimeoutSeconds
    no_progress_timeout_seconds=20; experiment_monitor_start_utc=$startedUtc.ToString('o')
}
Write-JsonFile (Join-Path $runShare 'host_monitor.json') $report
($standardOutput -split '\r?\n' | Select-Object -Last 4) -join "`n"
if ($standardError) { $standardError }
$report | ConvertTo-Json
if ($reason -or $child.ExitCode -ne 0) { exit 1 }
