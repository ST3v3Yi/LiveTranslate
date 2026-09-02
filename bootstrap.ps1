$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Uv = Join-Path $Root "tools\uv.exe"
$Ready = Join-Path $Root ".venv\.livetranslate-ready"
$env:UV_LINK_MODE = "copy"

# Keep every download inside the app folder instead of the system drive:
# managed Python would default to %APPDATA%\uv, the wheel cache (several GB
# with CUDA torch) to %LOCALAPPDATA%\uv, temp files to %TEMP% on C:.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "tools\python"
$env:UV_CACHE_DIR = Join-Path $Root ".uv-cache"
$env:TMP = Join-Path $Root ".tmp"
$env:TEMP = $env:TMP
New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null

# A failed setup must never leave the environment looking complete.
Remove-Item -LiteralPath $Ready -Force -ErrorAction SilentlyContinue

function Enable-SystemProxy {
    # uv (Python download) and pip honor *_PROXY env vars but not the Windows
    # registry system proxy; bridge it here. An already-set env proxy wins.
    if ($env:HTTPS_PROXY -or $env:HTTP_PROXY) {
        Write-Host "Using proxy from environment" -ForegroundColor Gray
        return
    }
    try {
        $reg = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        $s = Get-ItemProperty -Path $reg -ErrorAction Stop
        if ($s.ProxyEnable -ne 1 -or -not $s.ProxyServer) { return }
        $server = [string]$s.ProxyServer
        $http = $null; $https = $null
        if ($server -like "*=*") {
            foreach ($part in ($server -split ';')) {
                $kv = $part -split '=', 2
                if ($kv.Count -eq 2 -and $kv[0] -eq 'http')  { $http  = $kv[1] }
                if ($kv.Count -eq 2 -and $kv[0] -eq 'https') { $https = $kv[1] }
            }
        } else {
            $http = $server; $https = $server
        }
        if (-not $http)  { $http  = $https }
        if (-not $https) { $https = $http }
        if (-not $http) { return }
        if ($http  -notmatch '^\w+://') { $http  = "http://$http" }
        if ($https -notmatch '^\w+://') { $https = "http://$https" }
        $env:HTTP_PROXY  = $http
        $env:HTTPS_PROXY = $https
        $env:ALL_PROXY   = $https
        Write-Host "Detected Windows system proxy: $https (applied to uv/pip)" -ForegroundColor Green
    } catch {}
}
Enable-SystemProxy

Write-Host "Creating virtual environment with Python 3.12..." -ForegroundColor Cyan
& $Uv venv --python 3.12 --managed-python --allow-existing .venv
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to create venv" -ForegroundColor Red; exit 1 }
$Py = ".venv\Scripts\python.exe"

# Blackwell (sm_120+) needs cu128; older NVIDIA uses cu126; no GPU falls back to CPU
$Index = "https://download.pytorch.org/whl/cpu"
try {
    $cc = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $cc) {
        $cap = [double]($cc.Trim() -split "`n")[0]
        if ($cap -ge 12.0) { $Index = "https://download.pytorch.org/whl/cu128" }
        else { $Index = "https://download.pytorch.org/whl/cu126" }
        Write-Host "NVIDIA GPU detected (compute $cap), using $Index" -ForegroundColor Green
    }
} catch {}
if ($Index -like "*cpu*") { Write-Host "No NVIDIA GPU detected, installing CPU-only PyTorch" -ForegroundColor Yellow }

Write-Host "Installing PyTorch (this may take a while)..." -ForegroundColor Cyan
& $Uv pip install --python $Py torch torchaudio --index-url $Index
if ($LASTEXITCODE -ne 0) { Write-Host "PyTorch install failed" -ForegroundColor Red; exit 1 }

Write-Host "Installing dependencies..." -ForegroundColor Cyan
& $Uv pip install --python $Py -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "Dependency install failed" -ForegroundColor Red; exit 1 }

& $Uv pip install --python $Py "yasbd-lib>=0.15,<1.0"
if ($LASTEXITCODE -ne 0) { Write-Host "yasbd-lib install failed" -ForegroundColor Red; exit 1 }

& $Uv pip check --python $Py
if ($LASTEXITCODE -ne 0) { Write-Host "Installed dependencies are inconsistent" -ForegroundColor Red; exit 1 }

Set-Content -LiteralPath $Ready -Value (Get-Date -Format o) -Encoding ascii

# The wheel cache and temp files are only needed during setup; drop them so
# the app folder does not keep a second copy of every downloaded wheel.
Remove-Item -Recurse -Force $env:UV_CACHE_DIR, $env:TMP -ErrorAction SilentlyContinue

Write-Host "Setup complete." -ForegroundColor Green
