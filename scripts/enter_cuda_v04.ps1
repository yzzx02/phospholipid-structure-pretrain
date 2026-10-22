# Dot-source this file from any directory to select the G-drive workspace environment.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$workspaceRoot = Split-Path $projectRoot -Parent
$environmentPath = Join-Path $workspaceRoot 'environments\polar-v04-cu128'
$pythonPath = Join-Path $environmentPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Missing environment: $pythonPath" }
$env:VIRTUAL_ENV = $environmentPath
$env:Path = (Join-Path $environmentPath 'Scripts') + [IO.Path]::PathSeparator + $env:Path
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$locations = @{
    TEMP='cache\tmp'; TMP='cache\tmp'; PIP_CACHE_DIR='cache\pip'
    TORCH_HOME='cache\torch'; HF_HOME='cache\huggingface'; MPLCONFIGDIR='cache\matplotlib'
}
foreach ($entry in $locations.GetEnumerator()) {
    $folder = Join-Path $workspaceRoot $entry.Value
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
    [Environment]::SetEnvironmentVariable($entry.Key, $folder, 'Process')
}
Set-Location -LiteralPath $projectRoot
Write-Host "Python: $pythonPath"
Write-Host "Project, data, outputs and caches: $workspaceRoot"
