param(
  [string]$Python = "py",
  [string]$EnvironmentPath = ".venv",
  [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
  [switch]$SkipTorchInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$venvRoot = if ([System.IO.Path]::IsPathRooted($EnvironmentPath)) { $EnvironmentPath } else { Join-Path $projectRoot $EnvironmentPath }
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

function Resolve-Executable([string]$Name) {
  if (Test-Path -LiteralPath $Name -PathType Leaf) { return (Resolve-Path -LiteralPath $Name).Path }
  $command = Get-Command -Name $Name -CommandType Application -ErrorAction SilentlyContinue
  if (-not $command) { throw "Could not find '$Name'. Install Python 3.12 or pass its full path with -Python." }
  return $command.Source
}
function Invoke-Python([string[]]$Arguments) {
  & $venvPython @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($Arguments -join ' ') (exit $LASTEXITCODE)" }
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
  if ($SkipTorchInstall) { throw "-SkipTorchInstall requires an existing Python environment at '$venvRoot'." }
  $launcher = Resolve-Executable $Python
  if ([System.IO.Path]::GetFileNameWithoutExtension($launcher) -ieq "py") { & $launcher -3.12 -m venv $venvRoot }
  else {
    $baseVersion = & $launcher --version
    if ($LASTEXITCODE -ne 0 -or $baseVersion -notmatch "Python 3\.12\.") { throw "The tested Triton build uses Python 3.12; pass that executable with -Python." }
    & $launcher -m venv $venvRoot
  }
  if ($LASTEXITCODE -ne 0) { throw "Could not create the project virtual environment." }
}

$envVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $envVersion.Trim() -ne "3.12") { throw "The developer environment must use Python 3.12 (found $($envVersion.Trim()))." }
Push-Location $projectRoot
try {
  Invoke-Python @("-m", "pip", "install", "--upgrade", "pip")
  if (-not $SkipTorchInstall) { Invoke-Python @("-m", "pip", "install", "torch==2.11.0", "--index-url", $TorchIndexUrl) }
  Invoke-Python @("-m", "pip", "install", "numpy==2.5.3", "triton-windows==3.6.0.post26")
  $probe = "import sys, numpy, torch, triton; assert sys.version_info[:2] == (3,12); assert torch.cuda.is_available(), 'PyTorch cannot see an NVIDIA CUDA GPU'; print('Python:', sys.version.split()[0]); print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda); print('Triton:', triton.__version__)"
  Invoke-Python @("-c", $probe)
  Write-Host "Developer environment ready: $venvPython"
  Write-Host "This environment is used for checkpoint conversion and Triton cubin generation only."
  Write-Host "Build a native runtime with .\build_windows.ps1 -CudaMajor 13 (or 12 if that toolkit is installed)."
} finally { Pop-Location }
