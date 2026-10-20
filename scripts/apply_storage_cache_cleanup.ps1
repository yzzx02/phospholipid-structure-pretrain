param(
    [string]$Manifest = 'outputs/polar_similarity_eval_v04_20261020/storage/cleanup_dry_run_private.json',
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'
$taskWorkspace = [IO.Path]::GetFullPath('G:\yzx\linzhiyuce').TrimEnd('\')
$taskPlan = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if (-not $taskPlan.dry_run -or $taskPlan.workspace -ne $taskWorkspace) { throw 'Invalid dry-run manifest' }
$taskProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    ([string]::IsNullOrWhiteSpace($_.ExecutablePath) -or
     $_.ExecutablePath -like "$taskWorkspace\*" -or $_.CommandLine -like "*$taskWorkspace*")
})
if ($taskProcesses.Count -gt 0) { throw 'Workspace Python processes still active; cleanup stopped' }
$taskFullWheel = $taskPlan.full_torch_wheel_preserved
if ((Get-FileHash -LiteralPath $taskFullWheel.path -Algorithm SHA256).Hash.ToLower() -ne $taskFullWheel.sha256) {
    throw 'Protected full Torch wheel hash mismatch'
}
function Assert-TaskPath([string]$Path) {
    $taskAbsolute = [IO.Path]::GetFullPath($Path)
    if (-not $taskAbsolute.StartsWith($taskWorkspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes workspace: $taskAbsolute"
    }
    if ($taskAbsolute -like "$taskWorkspace\restored_v04\*" -or
        $taskAbsolute -like "$taskWorkspace\phospholipid_rtx4070_transfer_20261020\*") {
        throw 'Frozen restore and transfer paths cannot be cleaned'
    }
    $taskAncestor = Get-Item -LiteralPath $taskAbsolute -Force
    while ($taskAncestor.FullName -ne $taskWorkspace) {
        if ($taskAncestor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse point in cleanup path' }
        $taskAncestor = if ($taskAncestor -is [IO.DirectoryInfo]) { $taskAncestor.Parent } else { $taskAncestor.Directory }
        if ($null -eq $taskAncestor) { throw 'Path validation failed' }
    }
    return $taskAbsolute
}
# Validate every target and every snapshot before deleting any file.
foreach ($taskTarget in $taskPlan.targets) {
    $taskPath = Assert-TaskPath $taskTarget.path
    $taskItem = Get-Item -LiteralPath $taskPath -Force
    if ($taskTarget.kind -eq 'cache_directory') {
        if (-not $taskItem.PSIsContainer -or $taskItem.Name -notin @('__pycache__', '.pytest_cache', '.ruff_cache')) {
            throw 'Only named Python/test caches may be recursively removed'
        }
        $taskActualFiles = @(Get-ChildItem -LiteralPath $taskPath -Recurse -Force -File)
        if ($taskActualFiles.Count -ne @($taskTarget.files).Count) { throw 'Cache changed since dry run' }
        if (@(Get-ChildItem -LiteralPath $taskPath -Recurse -Force | Where-Object {
            $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Nested reparse point' }
    } elseif ($taskTarget.kind -eq 'abandoned_partial_wheel') {
        if ($taskItem.PSIsContainer -or $taskPath -notlike "$taskWorkspace\cache\tmp\pip-unpack-*\*.whl") {
            throw 'Invalid partial download path'
        }
    } else { throw 'Unsupported cleanup kind' }
    foreach ($taskFile in $taskTarget.files) {
        $taskFilePath = Assert-TaskPath $taskFile.path
        if ($taskFilePath -ne $taskPath -and -not $taskFilePath.StartsWith($taskPath + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'File outside target'
        }
        $taskCurrent = Get-Item -LiteralPath $taskFilePath -Force
        if ($taskCurrent.Length -ne $taskFile.bytes -or
            (Get-FileHash -LiteralPath $taskFilePath -Algorithm SHA256).Hash.ToLower() -ne $taskFile.sha256) {
            throw 'Cleanup file changed since dry run'
        }
        $taskHandle = [IO.File]::Open($taskFilePath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
        $taskHandle.Dispose()
    }
}
if ($ValidateOnly) {
    Write-Output "Dry-run validated: $($taskPlan.target_files) files; no files deleted."
    return
}
$taskBefore = (Get-PSDrive -Name G).Free
foreach ($taskTarget in $taskPlan.targets) {
    $taskChecked = Assert-TaskPath $taskTarget.path
    if ($taskTarget.kind -eq 'cache_directory') { Remove-Item -LiteralPath $taskChecked -Recurse -Force }
    else { Remove-Item -LiteralPath $taskChecked -Force }
}
$taskAfter = (Get-PSDrive -Name G).Free
$taskReceipt = [ordered]@{
    utc = [DateTimeOffset]::UtcNow.ToString('o')
    dry_run_manifest_sha256 = (Get-FileHash -LiteralPath $Manifest -Algorithm SHA256).Hash.ToLower()
    deleted_files = $taskPlan.target_files
    deleted_logical_bytes = $taskPlan.logical_bytes
    free_bytes_before = $taskBefore
    free_bytes_after = $taskAfter
    observed_drive_free_bytes_increase = $taskAfter - $taskBefore
    no_active_workspace_python_processes = $true
    retained_full_torch_wheel_sha256 = $taskFullWheel.sha256
}
$taskReceipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath 'outputs/polar_similarity_eval_v04_20261020/storage/cleanup_receipt.json' -Encoding utf8
$taskReceipt | ConvertTo-Json -Depth 8
