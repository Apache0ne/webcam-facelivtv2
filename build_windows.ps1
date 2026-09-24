param(
  [ValidateSet(12, 13)][int]$CudaMajor = 13,
  [string]$Python = "",
  [string]$CudaRoot = "",
  [string]$BuildDirectory = "",
  [string]$KernelDirectory = "",
  [string[]]$KernelArch = @(),
  [switch]$SkipKernelBuild,
  [switch]$AllowCpuBuild,
  [string]$Model = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$ortVersion = if ($CudaMajor -eq 13) { "1.27.0" } else { "1.26.0" }
$defaultArchitectures = if ($CudaMajor -eq 13) {
  @("75", "80", "86", "87", "89", "90", "100", "120")
} else {
  @("75", "80", "86", "87", "89", "90", "100", "120")
}
if ($KernelArch.Count -eq 0) { $KernelArch = $defaultArchitectures }
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) { $BuildDirectory = "build\cuda$CudaMajor" }
if ([string]::IsNullOrWhiteSpace($KernelDirectory)) { $KernelDirectory = "generated\cuda$CudaMajor\kernels" }

function Resolve-Executable([string]$Name) {
  if (Test-Path -LiteralPath $Name -PathType Leaf) { return (Resolve-Path -LiteralPath $Name).Path }
  $found = Get-Command -Name $Name -CommandType Application -ErrorAction SilentlyContinue
  if (-not $found) { throw "Could not find '$Name' on PATH." }
  return $found.Source
}
function Invoke-Checked([string]$Executable, [string[]]$Arguments, [string]$Stage) {
  & $Executable @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Stage failed (exit $LASTEXITCODE)." }
}
function Resolve-ProjectPath([string]$Path) {
  if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
  return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

if ([string]::IsNullOrWhiteSpace($Python)) {
  $localPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
  $Python = if (Test-Path -LiteralPath $localPython -PathType Leaf) { $localPython } else { "python" }
}
$pythonExe = Resolve-Executable $Python
$cmakeExe = Get-Command -Name "cmake" -CommandType Application -ErrorAction SilentlyContinue
if ($cmakeExe) { $cmakeExe = $cmakeExe.Source }
else {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  $vsInstall = if (Test-Path -LiteralPath $vswhere -PathType Leaf) { & $vswhere -latest -products '*' -property installationPath } else { $null }
  $vsCmake = if ($vsInstall) { Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" } else { "" }
  if (-not $vsCmake -or -not (Test-Path -LiteralPath $vsCmake -PathType Leaf)) { throw "CMake 3.24 or newer was not found. Install CMake or Visual Studio's CMake component." }
  $cmakeExe = $vsCmake
}
$cmakeVersionText = & $cmakeExe --version
$cmakeMatch = [regex]::Match([string]$cmakeVersionText[0], "cmake version\s+([0-9.]+)")
if ($LASTEXITCODE -ne 0 -or -not $cmakeMatch.Success -or [version]$cmakeMatch.Groups[1].Value -lt [version]"3.24") {
  throw "CMake 3.24 or newer is required."
}

$vswherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$msvcInstall = $null
if (Test-Path -LiteralPath $vswherePath -PathType Leaf) {
  $msvcInstall = & $vswherePath -latest -products '*' -version "[17.0,18.0)" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if ([string]::IsNullOrWhiteSpace([string]$msvcInstall)) {
  throw "Visual Studio 2022 C++ Build Tools and a Windows SDK are required for the supported CUDA toolkits. Install the Desktop development with C++ workload."
}

if ([string]::IsNullOrWhiteSpace($CudaRoot)) {
  $CudaRoot = if ($env:CUDA_PATH -and (Split-Path $env:CUDA_PATH -Leaf) -match "^v?$CudaMajor\.") { $env:CUDA_PATH } else { "" }
}
if (-not $CudaRoot) {
  $cudaInstallRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
  $foundCuda = Get-ChildItem -LiteralPath $cudaInstallRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "^v?$CudaMajor\." -and (Test-Path (Join-Path $_.FullName "bin\nvcc.exe")) } |
    Sort-Object { [version]($_.Name.TrimStart('v')) } -Descending | Select-Object -First 1
  if ($foundCuda) { $CudaRoot = $foundCuda.FullName }
}
$CudaRoot = Resolve-ProjectPath $CudaRoot
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf) -or -not (Test-Path -LiteralPath (Join-Path $CudaRoot "include\cuda.h") -PathType Leaf)) {
  throw "CUDA Toolkit $CudaMajor was not found at '$CudaRoot'. Install that toolkit or pass -CudaRoot."
}
$nvccVersion = & $nvcc --version
if ($LASTEXITCODE -ne 0 -or ($nvccVersion -join " ") -notmatch "release\s+$CudaMajor\.") {
  throw "The CUDA compiler under '$CudaRoot' does not match the requested CUDA $CudaMajor release lane."
}

if ($AllowCpuBuild) {
  $pythonProbe = "import sys, triton; print('Python:', sys.version.split()[0]); assert sys.version_info[:2] == (3,12), 'Use Python 3.12 for the tested Windows Triton build'; print('Triton:', triton.__version__); print('GPU: not used for cross-compiling cubins')"
} else {
  $pythonProbe = "import sys, numpy, torch, triton; print('Python:', sys.version.split()[0]); assert sys.version_info[:2] == (3,12), 'Use Python 3.12 for the tested Windows Triton build'; print('PyTorch CUDA:', torch.version.cuda); print('Triton:', triton.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cross-compiling without local GPU')"
}
Invoke-Checked -Executable $pythonExe -Arguments @("-c", $pythonProbe) -Stage "Developer Python/Triton preflight"

$kernelDir = Resolve-ProjectPath $KernelDirectory
$buildDir = Resolve-ProjectPath $BuildDirectory
$modelOut = Join-Path $projectRoot "facelivtv2-l.fp16.flvt"
if (-not (Test-Path -LiteralPath $modelOut -PathType Leaf)) {
  if (-not $Model) { throw "Converted FaceLiVT weights are missing. Provide the source checkpoint with -Model to convert it." }
  $sourceModel = Resolve-ProjectPath $Model
  if (-not (Test-Path -LiteralPath $sourceModel -PathType Leaf)) { throw "Checkpoint not found: $sourceModel" }
  Invoke-Checked -Executable $pythonExe -Arguments @("converter\convert_facelivtv2_l.py", "--checkpoint", $sourceModel, "--output", $modelOut) -Stage "Checkpoint conversion"
}

$previousTritonCache = $env:TRITON_CACHE_DIR
$scriptSetTritonCache = $false
Push-Location $projectRoot
try {
  if ([string]::IsNullOrWhiteSpace($env:TRITON_CACHE_DIR)) {
    $env:TRITON_CACHE_DIR = Join-Path $buildDir "triton_cache"
    New-Item -ItemType Directory -Path $env:TRITON_CACHE_DIR -Force | Out-Null
    $scriptSetTritonCache = $true
    Write-Host "Using project-local Triton compile cache: $env:TRITON_CACHE_DIR"
  }
  & (Join-Path $projectRoot "tools\fetch_onnxruntime.ps1") -Version $ortVersion
  if ($LASTEXITCODE -ne 0) { throw "Could not fetch the CUDA $CudaMajor ONNX Runtime C++ package." }

  if ($SkipKernelBuild) {
    $matrixPath = Join-Path $kernelDir "kernel_manifest.tsv"
    if (-not (Test-Path -LiteralPath $matrixPath -PathType Leaf)) { throw "-SkipKernelBuild requires an existing architecture matrix manifest at $matrixPath" }
    $rows = Get-Content -LiteralPath $matrixPath | Select-Object -Skip 1 | Where-Object { $_ }
    $available = @($rows | ForEach-Object { [int](($_ -split "`t",2)[0]) } | Sort-Object -Unique)
    foreach ($arch in $KernelArch) {
      $archNumber = [int]($arch -replace '^sm_?','')
      if ($archNumber -notin $available) { throw "Kernel matrix lacks sm_$archNumber; remove -SkipKernelBuild or add that architecture." }
    }
    foreach ($row in $rows) {
      $columns = $row -split "`t"
      if (-not (Test-Path -LiteralPath (Join-Path $kernelDir $columns[2]) -PathType Leaf)) { throw "Kernel matrix file is missing: $($columns[2])" }
    }
    Write-Host "Using existing Triton kernel matrix: $matrixPath"
  } else {
    $kernelArgs = @("tools\build_kernels.py", "--out", $kernelDir)
    foreach ($arch in $KernelArch) { $kernelArgs += @("--arch", $arch) }
    Invoke-Checked -Executable $pythonExe -Arguments $kernelArgs -Stage "Triton kernel compilation"
  }

  $architectureList = $KernelArch -join ";"
  $cmakeArgs = @("-S", $projectRoot, "-B", $buildDir, "-G", "Visual Studio 17 2022", "-A", "x64", "-DCMAKE_BUILD_TYPE=Release",
    "-DCUDAToolkit_ROOT=$CudaRoot", "-DCMAKE_CUDA_COMPILER=$nvcc",
    "-DCMAKE_CUDA_ARCHITECTURES=$architectureList",
    "-DFACELIVT_ONNXRUNTIME_VERSION=$ortVersion")
  Invoke-Checked -Executable $cmakeExe -Arguments $cmakeArgs -Stage "CMake configuration"
  Invoke-Checked -Executable $cmakeExe -Arguments @("--build", $buildDir, "--config", "Release", "--target", "facelivt_attendance", "facelivt_scrfd_smoke", "facelivt_embed", "--parallel") -Stage "Native C++/CUDA build"

  $releaseDir = Join-Path $buildDir "Release"
  foreach ($required in @("facelivt_attendance.exe", "facelivt_video.dll", "facelivt_scrfd_smoke.exe", "facelivt_embed.exe", "onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll")) {
    if (-not (Test-Path -LiteralPath (Join-Path $releaseDir $required) -PathType Leaf)) { throw "Build output is missing $required in $releaseDir" }
  }
  if (-not (Test-Path -LiteralPath (Join-Path $kernelDir "kernel_manifest.tsv") -PathType Leaf)) { throw "Kernel compiler did not produce $kernelDir\kernel_manifest.tsv" }
  Write-Host ""
  Write-Host "Native CUDA $CudaMajor build complete: $releaseDir"
  Write-Host "Build output: .\package_release.ps1 -CudaMajor $CudaMajor -BuildDirectory '$BuildDirectory' -KernelDirectory '$KernelDirectory' -CudaRoot '$CudaRoot'"
  Write-Host "The packaged app runs without Python; Python/Triton were used only to compile the architecture cubins."
  if ($AllowCpuBuild) { Write-Host "No PyTorch or NVIDIA GPU was used by this CPU cross-build." }
} finally {
  if ($scriptSetTritonCache) {
    if ($null -eq $previousTritonCache) { Remove-Item Env:TRITON_CACHE_DIR -ErrorAction SilentlyContinue }
    else { $env:TRITON_CACHE_DIR = $previousTritonCache }
  }
  Pop-Location
}
