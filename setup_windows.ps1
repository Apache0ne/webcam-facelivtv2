param(
    [string]$Python = "py",
    [string]$EnvironmentPath = ".venv",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$SkipTorchInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$venvRoot = if ([System.IO.Path]::IsPathRooted($EnvironmentPath)) {
    $EnvironmentPath
} else {
    Join-Path $projectRoot $EnvironmentPath
}
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$runtimeDllDirectory = Join-Path $env:WINDIR "System32"
$requiredRuntimeDlls = @("msvcp140.dll", "vcruntime140.dll")
$missingRuntimeDlls = @($requiredRuntimeDlls | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $runtimeDllDirectory $_) -PathType Leaf)
})
if ($missingRuntimeDlls.Count -gt 0) {
    throw "The Microsoft Visual C++ Redistributable is missing ($($missingRuntimeDlls -join ', ')). Install the x64 package from https://aka.ms/vs/17/release/vc_redist.x64.exe, then rerun setup."
}

function Resolve-Executable([string]$Name) {
    if (Test-Path -LiteralPath $Name -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Name).Path
    }
    $command = Get-Command -Name $Name -CommandType Application -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "Could not find '$Name'. Install Python 3.12 or pass its full path with -Python."
    }
    return $command.Source
}

function Invoke-Python([string[]]$Arguments) {
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Arguments -join ' ') (exit $LASTEXITCODE)"
    }
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    if ($SkipTorchInstall) {
        throw "-SkipTorchInstall requires an existing virtual environment at '$venvRoot'."
    }
    $launcher = Resolve-Executable $Python
    $launcherName = [System.IO.Path]::GetFileNameWithoutExtension($launcher)
    if ($launcherName -ieq "py") {
        & $launcher -3.12 -m venv $venvRoot
    } else {
        $baseVersion = & $launcher --version
        if ($LASTEXITCODE -ne 0 -or $baseVersion -notmatch "Python 3\.12\.") {
            throw "This setup is pinned to Python 3.12. Pass a Python 3.12 executable with -Python."
        }
        & $launcher -m venv $venvRoot
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the project virtual environment." }
}

$envVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $envVersion.Trim() -ne "3.12") {
    throw "The existing .venv must use Python 3.12 (found $($envVersion.Trim())). Remove or rename .venv, then rerun setup."
}

Push-Location $projectRoot
try {
    Invoke-Python -Arguments @("-m", "pip", "install", "--upgrade", "pip")
    if (-not $SkipTorchInstall) {
        Invoke-Python -Arguments @("-m", "pip", "install", "torch==2.11.0", "--index-url", $TorchIndexUrl)
        Invoke-Python -Arguments @("-m", "pip", "install", "-r", "requirements-build.txt")
    } else {
        Invoke-Python -Arguments @("-m", "pip", "install", "numpy==2.5.3", "triton-windows==3.6.0.post26")
    }
    Invoke-Python -Arguments @("-m", "pip", "install", "-r", "requirements-webcam.txt")

    $probe = "import torch, triton, onnxruntime as ort; assert torch.cuda.is_available(), 'PyTorch cannot see an NVIDIA CUDA GPU'; assert torch.version.cuda == '12.8', f'Expected PyTorch CUDA 12.8, found {torch.version.cuda}'; assert triton.__version__ == '3.6.0', f'Expected Triton 3.6.0, found {triton.__version__}'; assert ort.__version__ == '1.26.0', f'Expected ONNX Runtime 1.26.0, found {ort.__version__}'; assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'onnxruntime-gpu CUDA provider is unavailable'; print('Python:', __import__('sys').version.split()[0]); print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda); print('Triton:', triton.__version__); print('ONNX Runtime:', ort.__version__, '| CUDA provider available')"
    Invoke-Python -Arguments @("-c", $probe)

    Write-Host ""
    Write-Host "Environment ready: $venvPython"
    Write-Host "Next: review THIRD_PARTY.md, then run .\download_scrfd.ps1"
    Write-Host "Then: .\build_windows.ps1"
    Write-Host "Finally: .\run_webcam.ps1"
} finally {
    Pop-Location
}
