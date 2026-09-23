param(
  [string]$Model = "",
  [string]$Python = "",
  [string]$CudaRoot = ""
)
$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

function Resolve-Executable([string]$Name) {
  if (Test-Path -LiteralPath $Name -PathType Leaf) {
    return (Resolve-Path -LiteralPath $Name).Path
  }
  $command = Get-Command -Name $Name -CommandType Application -ErrorAction SilentlyContinue
  if (-not $command) { throw "Could not find '$Name' on PATH." }
  return $command.Source
}

function Invoke-Checked([string]$Executable, [string[]]$Arguments, [string]$Stage) {
  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Stage failed (exit $LASTEXITCODE)." }
}

if ([string]::IsNullOrWhiteSpace($Python)) {
  $localPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
  $Python = if (Test-Path -LiteralPath $localPython -PathType Leaf) { $localPython } else { "python" }
}
$pythonExe = Resolve-Executable $Python
$cmakeExe = Resolve-Executable "cmake"
$cmakeOutput = & $cmakeExe --version
$cmakeMatch = [regex]::Match([string]$cmakeOutput[0], "cmake version\s+([0-9.]+)")
if ($LASTEXITCODE -ne 0 -or -not $cmakeMatch.Success) {
  throw "Could not read the installed CMake version. Install CMake 3.24 or newer."
}
if ([version]$cmakeMatch.Groups[1].Value -lt [version]"3.24") {
  throw "CMake 3.24 or newer is required (found $($cmakeMatch.Groups[1].Value))."
}

$vswherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$msvcInstall = $null
if (Test-Path -LiteralPath $vswherePath -PathType Leaf) {
  $msvcInstall = & $vswherePath -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if ([string]::IsNullOrWhiteSpace([string]$msvcInstall) -and -not (Get-Command -Name "cl.exe" -CommandType Application -ErrorAction SilentlyContinue)) {
  throw "MSVC C++ Build Tools were not found. Install Visual Studio Build Tools with the Desktop development with C++ workload and a Windows SDK."
}

$cudaRoot = if (-not [string]::IsNullOrWhiteSpace($CudaRoot)) { $CudaRoot } else { $env:CUDA_PATH }
$nvcc = Get-Command -Name "nvcc" -CommandType Application -ErrorAction SilentlyContinue
if ($nvcc) {
  $nvccPath = $nvcc.Source
  $discoveredCudaRoot = Split-Path (Split-Path $nvccPath -Parent) -Parent
  if (-not $cudaRoot -or -not (Test-Path (Join-Path $cudaRoot "include\cuda.h"))) {
    $cudaRoot = $discoveredCudaRoot
  }
} elseif ($cudaRoot -and (Test-Path (Join-Path $cudaRoot "bin\nvcc.exe"))) {
  $nvccPath = Join-Path $cudaRoot "bin\nvcc.exe"
} else {
  $cudaInstallRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
  $foundCuda = Get-ChildItem -LiteralPath $cudaInstallRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "^v?(\d+(\.\d+)+)$" -and (Test-Path (Join-Path $_.FullName "bin\nvcc.exe")) } |
    Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending |
    Select-Object -First 1
  if ($foundCuda) {
    $cudaRoot = $foundCuda.FullName
    $nvccPath = Join-Path $cudaRoot "bin\nvcc.exe"
  } else {
    throw "CUDA Toolkit was not found. Install it, or pass its install folder with -CudaRoot."
  }
}
if (-not $cudaRoot -or -not (Test-Path (Join-Path $cudaRoot "include\cuda.h"))) {
  throw "CUDA Toolkit headers were not found under '$cudaRoot'. Check CUDA_PATH or -CudaRoot."
}

$convertedModel = Join-Path $projectRoot "facelivtv2-l.fp16.flvt"
$modelPath = $null
if (-not [string]::IsNullOrWhiteSpace($Model)) {
  $modelPath = if ([System.IO.Path]::IsPathRooted($Model)) { $Model } else { Join-Path $projectRoot $Model }
  if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
    throw "FaceLiVT source checkpoint not found: $modelPath."
  }
} elseif (-not (Test-Path -LiteralPath $convertedModel -PathType Leaf)) {
  $defaultCheckpoint = Join-Path $projectRoot "facelivtv2-l.pt"
  if (Test-Path -LiteralPath $defaultCheckpoint -PathType Leaf) {
    $modelPath = $defaultCheckpoint
  } else {
    throw "No FaceLiVT weights found. Add facelivtv2-l.fp16.flvt or facelivtv2-l.pt to the project root, or pass -Model."
  }
}

$pythonProbe = "import sys, numpy, torch, triton; print('Python:', sys.version.split()[0]); assert sys.version_info[:2] == (3,12), 'Use Python 3.12 for this tested Windows build'; assert torch.cuda.is_available(), 'PyTorch cannot see an NVIDIA CUDA GPU'; print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch CUDA:', torch.version.cuda); print('Triton:', triton.__version__)"
Invoke-Checked -Executable $pythonExe -Arguments @("-c", $pythonProbe) -Stage "Python/CUDA preflight"

Push-Location $projectRoot
try {
  $kernelDir = Join-Path $projectRoot "generated\kernels"
  $buildDir = Join-Path $projectRoot "build"

  if ($modelPath) {
    Invoke-Checked -Executable $pythonExe -Arguments @("converter\convert_facelivtv2_l.py", "--checkpoint", $modelPath, "--output", $convertedModel) -Stage "Checkpoint conversion"
  } else {
    Write-Host "Using preconverted FaceLiVT weights: $convertedModel"
  }
  Invoke-Checked -Executable $pythonExe -Arguments @("tools\build_kernels.py", "--out", $kernelDir) -Stage "Triton kernel compilation"
  Invoke-Checked -Executable $cmakeExe -Arguments @("-S", $projectRoot, "-B", $buildDir, "-DCMAKE_BUILD_TYPE=Release", "-DCUDAToolkit_ROOT=$cudaRoot") -Stage "CMake configuration"
  Invoke-Checked -Executable $cmakeExe -Arguments @("--build", $buildDir, "--config", "Release", "--parallel") -Stage "C++ build"
  $cudaRootMetadata = Join-Path $buildDir "cuda_root.txt"
  [System.IO.File]::WriteAllText($cudaRootMetadata, $cudaRoot, [System.Text.UTF8Encoding]::new($false))

  foreach ($artifact in @(
    $convertedModel,
    (Join-Path $kernelDir "kernel_manifest.tsv")
  )) {
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
      throw "Build finished without expected artifact: $artifact"
    }
  }

  $runtimeDirectories = @(
    (Join-Path $buildDir "Release"),
    $buildDir
  )
  $runtimeDirectory = $runtimeDirectories | Where-Object {
    (Test-Path -LiteralPath (Join-Path $_ "facelivt_native.dll") -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $_ "facelivt_video.dll") -PathType Leaf)
  } | Select-Object -First 1
  if (-not $runtimeDirectory) {
    throw "CMake completed without both webcam DLLs in build\Release or build."
  }

  Write-Host ""
  Write-Host "Build complete for GPU: see the Triton preflight device above."
  Write-Host "Native DLLs: $runtimeDirectory"
  Write-Host "Run the CUDA-to-D3D11 display check described in WEBCAM.md before opening the camera."
  Write-Host "Then start attendance: .\run_webcam.ps1"
} finally {
  Pop-Location
}
