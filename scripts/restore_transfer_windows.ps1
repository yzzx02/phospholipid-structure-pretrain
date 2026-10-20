param(
    [Parameter(Mandatory)][string]$PackageDirectory,
    [Parameter(Mandatory)][string]$DestinationParent
)
$ErrorActionPreference = 'Stop'
$packageRoot = (Resolve-Path -LiteralPath $PackageDirectory).ProviderPath
$receipt = Get-Content -LiteralPath (Join-Path $packageRoot 'PACKAGE_RECEIPT.json') -Raw |
    ConvertFrom-Json
$destination = [IO.Path]::GetFullPath($DestinationParent)
$projectPath = Join-Path $destination 'phospholipid-structure-pretrain'
if (Test-Path -LiteralPath $projectPath) { throw 'Use a destination with no existing project' }
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$zipPath = Join-Path $destination 'phospholipid_v04_transfer.zip'
if (Test-Path -LiteralPath $zipPath) { throw 'Merged ZIP already exists; no overwrite' }
$outputStream = [IO.File]::Open($zipPath, [IO.FileMode]::CreateNew)
try {
    foreach ($part in $receipt.parts) {
        if ([IO.Path]::GetFileName($part.name) -ne $part.name) { throw 'Unsafe part name' }
        $path = Join-Path $packageRoot $part.name
        $file = Get-Item -LiteralPath $path
        if ($file.Length -ne $part.bytes) { throw "Size mismatch: $($part.name)" }
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() -ne $part.sha256) {
            throw "SHA256 mismatch: $($part.name)"
        }
        $inputStream = [IO.File]::OpenRead($path)
        try { $inputStream.CopyTo($outputStream, 4MB) } finally { $inputStream.Dispose() }
        Write-Host "Verified and merged $($part.name)"
    }
} finally { $outputStream.Dispose() }
if ((Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLower() -ne $receipt.zip_sha256) {
    throw 'Merged ZIP checksum mismatch'
}
& tar.exe -xf $zipPath -C $destination
if ($LASTEXITCODE -ne 0) { throw 'ZIP extraction failed; no training started' }
$manifestPath = Join-Path $projectPath 'PROJECT_TRANSFER_MANIFEST.json'
if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLower() -ne
    $receipt.manifest_sha256) { throw 'Manifest checksum mismatch' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$verified = 0
foreach ($file in $manifest.files) {
    $target = [IO.Path]::GetFullPath((Join-Path $projectPath $file.path))
    if (-not $target.StartsWith($projectPath + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe manifest path' }
    if ((Get-Item -LiteralPath $target).Length -ne $file.bytes -or
        (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower() -ne $file.sha256) {
        throw "Extracted file mismatch: $($file.path)"
    }
    $verified++
    if ($verified % 100 -eq 0) { Write-Host "Verified $verified / $($manifest.files.Count) files" }
}
Write-Host "Transfer verified: $verified files; source commit $($manifest.source_commit)"
Write-Host "Project: $projectPath"
Write-Host 'Existing CUDA environment preserved. No model or GPU workload was launched.'
