$ErrorActionPreference = 'Stop'
$reviewDirectory = (Resolve-Path (Join-Path $PSScriptRoot '..\outputs\headgroup_template_proposal_20261022')).Path
$pythonPath = (Resolve-Path (Join-Path $PSScriptRoot '..\..\environments\polar-v04-cu128\Scripts\python.exe')).Path
$reviewUrl = 'http://127.0.0.1:8881/index.html'
try {
    $response = Invoke-WebRequest -Uri $reviewUrl -TimeoutSec 2 -UseBasicParsing
    if ($response.Content -notmatch 'headgroup-template-proposal-') {
        throw 'Port 8881 belongs to another page. Close that server or open index.html directly.'
    }
} catch {
    if ($_.Exception.Message -like '*another page*') { throw }
    Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'http.server', '8881', '--bind', '127.0.0.1', '--directory', $reviewDirectory) -WindowStyle Hidden
    $ready = $false
    for ($attempt=0; $attempt -lt 10; $attempt++) {
        Start-Sleep -Milliseconds 200
        try { $null = Invoke-WebRequest -Uri $reviewUrl -TimeoutSec 1 -UseBasicParsing; $ready=$true; break } catch { }
    }
    if (-not $ready) { throw 'Local review server did not start. Open index.html directly.' }
}
Start-Process $reviewUrl
