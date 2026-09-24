param(
  [Parameter(Mandatory = $true)][ValidateSet(12, 13)][int]$CudaMajor,
  [string]$ReleaseVersion = "0.1.1",
  [string]$BuildDirectory = "",
  [string]$KernelDirectory = "",
  [string]$CudaRoot = "",
  [string]$CudnnWheel = "",
  [string]$OutputDirectory = "dist",
  [switch]$SkipRuntimeTest
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$ortVersion = if ($CudaMajor -eq 13) { "1.27.0" } else { "1.26.0" }
$cudnnVersion = "9.16.0.29"
$cudnnHash = if ($CudaMajor -eq 13) {
  "79DC1BFE8C1A780CF4EB7B334D14D7927576D6DD8823F8E2769911AF30FD4DA3"
} else {
  "142E2BD646A4573AB17D61A24C6359155CDFE1F34C67FC305B71222A7AE45B8E"
}
$cudaDlls = if ($CudaMajor -eq 13) {
  @("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll", "cufft64_12.dll", "curand64_10.dll", "nvrtc64_130_0.dll", "nvrtc-builtins64_133.dll", "nvJitLink_130_0.dll")
} else {
  @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll", "curand64_10.dll", "nvrtc64_120_0.dll", "nvrtc-builtins64_128.dll", "nvJitLink_120_0.dll")
}
$requiredArchitectures = if ($CudaMajor -eq 13) { @(75,80,86,87,89,90,100,120) } else { @(61,70,75,80,86,87,89,90,100,120) }
if (-not $BuildDirectory) { $BuildDirectory = "build\cuda$CudaMajor" }
if (-not $KernelDirectory) { $KernelDirectory = "generated\cuda$CudaMajor\kernels" }

function Resolve-PathFromRoot([string]$Path) {
  if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
  return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}
function Require-File([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required release input is missing: $Path" }
}

$buildRoot = Resolve-PathFromRoot $BuildDirectory
$kernelRoot = Resolve-PathFromRoot $KernelDirectory
$outputRoot = Resolve-PathFromRoot $OutputDirectory
$releaseRoot = Join-Path $outputRoot "facelivt-windows-cuda$CudaMajor-x64-v$ReleaseVersion"
$buildCache = Join-Path $buildRoot "CMakeCache.txt"
Require-File $buildCache
Require-File (Join-Path $buildRoot "Release\facelivt_attendance.exe")
Require-File (Join-Path $buildRoot "Release\facelivt_video.dll")
Require-File (Join-Path $buildRoot "Release\facelivt_embed.exe")
Require-File (Join-Path $buildRoot "Release\facelivt_scrfd_smoke.exe")
Require-File (Join-Path $kernelRoot "kernel_manifest.tsv")
Require-File (Join-Path $projectRoot "facelivtv2-l.fp16.flvt")
Require-File (Join-Path $projectRoot "models\scrfd_10g_bnkps.onnx")

$cache = Get-Content -LiteralPath $buildCache
$cudaCompilerLine = $cache | Where-Object { $_ -match '^CMAKE_CUDA_COMPILER:(FILEPATH|STRING)=' } | Select-Object -First 1
$configuredCudaRoot = ""
if ($cudaCompilerLine) {
  $cudaCompiler = $cudaCompilerLine -replace '^CMAKE_CUDA_COMPILER:(FILEPATH|STRING)=', ''
  $configuredCudaRoot = Split-Path (Split-Path $cudaCompiler -Parent) -Parent
}
if (-not $CudaRoot) {
  if ($configuredCudaRoot) { $CudaRoot = $configuredCudaRoot }
  elseif ($env:CUDA_PATH) { $CudaRoot = $env:CUDA_PATH }
  else { throw "Pass -CudaRoot for the toolkit used to build this package." }
} else { $CudaRoot = Resolve-PathFromRoot $CudaRoot }
if ($configuredCudaRoot -and [System.IO.Path]::GetFullPath($CudaRoot) -ne [System.IO.Path]::GetFullPath($configuredCudaRoot)) {
  throw "This build used CUDA from '$configuredCudaRoot', not the requested CUDA root '$CudaRoot'. Build the lane with the matching toolkit first."
}
$toolkitRootLine = $cache | Where-Object { $_ -match '^CUDAToolkit_ROOT:[^=]+=' } | Select-Object -First 1
if ($toolkitRootLine) {
  $cachedToolkitRoot = ($toolkitRootLine -split '=',2)[1]
  if ([System.IO.Path]::GetFullPath($CudaRoot).TrimEnd('\') -ne [System.IO.Path]::GetFullPath($cachedToolkitRoot).TrimEnd('\')) {
    throw "The CMake cache used '$cachedToolkitRoot', not the requested toolkit '$CudaRoot'."
  }
}
$nvccInfo = & (Join-Path $CudaRoot "bin\nvcc.exe") --version
if ($LASTEXITCODE -ne 0 -or ($nvccInfo -join " ") -notmatch "release\s+$CudaMajor\.") {
  throw "The build directory is not configured for CUDA $CudaMajor."
}
$archLine = $cache | Where-Object { $_ -match '^CMAKE_CUDA_ARCHITECTURES(:\w+)?=' } | Select-Object -First 1
if (-not $archLine) { throw "The CMake build did not record CMAKE_CUDA_ARCHITECTURES." }
$nativeArchitectures = (($archLine -split '=',2)[1] -split ';' | ForEach-Object { [int]($_ -replace '-real$','') })
if ($CudaMajor -eq 13 -and ($nativeArchitectures | Where-Object { $_ -lt 75 })) { throw "CUDA 13 release cannot include native detector code below SM 7.5." }
foreach ($arch in $requiredArchitectures) {
  if ($arch -notin $nativeArchitectures) { throw "Native detector CUDA objects are missing sm_$arch; rebuild with -KernelArch $($requiredArchitectures -join ',')." }
}

$manifest = Get-Content -LiteralPath (Join-Path $kernelRoot "kernel_manifest.tsv")
if ($manifest.Count -lt 2 -or $manifest[0] -ne "arch`talias`tcubin`tentry`twarps`tshared`tscratch_args") {
  throw "Release kernels must use the architecture matrix manifest generated by tools/build_kernels.py."
}
$builtArchitectures = @($manifest | Select-Object -Skip 1 | Where-Object { $_ } | ForEach-Object { [int](($_ -split "`t",2)[0]) } | Sort-Object -Unique)
foreach ($arch in $requiredArchitectures) {
  if ($arch -notin $builtArchitectures) { throw "Kernel matrix is missing sm_$arch. Rebuild with -KernelArch $($requiredArchitectures -join ',')." }
}

$ortRoot = Join-Path $projectRoot "build\deps\onnxruntime-gpu-windows-$ortVersion"
$ortBin = Join-Path $ortRoot "runtimes\win-x64\native"
foreach ($file in @("onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll")) { Require-File (Join-Path $ortBin $file) }
$cudaBin = Join-Path $CudaRoot "bin\x64"
if (-not (Test-Path -LiteralPath $cudaBin -PathType Container)) { $cudaBin = Join-Path $CudaRoot "bin" }
foreach ($file in $cudaDlls) { Require-File (Join-Path $cudaBin $file) }

if (-not $CudnnWheel) { $CudnnWheel = Join-Path $projectRoot "build\deps\nvidia_cudnn_cu$CudaMajor-$cudnnVersion-py3-none-win_amd64.whl" }
$CudnnWheel = Resolve-PathFromRoot $CudnnWheel
if (-not (Test-Path -LiteralPath $CudnnWheel -PathType Leaf)) {
  New-Item -ItemType Directory -Path (Split-Path $CudnnWheel -Parent) -Force | Out-Null
  $packageName = "nvidia-cudnn-cu$CudaMajor"
  $pypi = Invoke-RestMethod -Uri "https://pypi.org/pypi/$packageName/$cudnnVersion/json"
  $wheel = $pypi.urls | Where-Object { $_.filename -eq (Split-Path $CudnnWheel -Leaf) } | Select-Object -First 1
  if (-not $wheel -or $wheel.digests.sha256.ToUpperInvariant() -ne $cudnnHash) { throw "Official PyPI does not report the pinned Windows cuDNN $CudaMajor wheel/hash." }
  Write-Host "Downloading NVIDIA cuDNN $cudnnVersion for CUDA $CudaMajor..."
  Invoke-WebRequest -Uri $wheel.url -OutFile $CudnnWheel
}
$actualCudnnHash = (Get-FileHash -LiteralPath $CudnnWheel -Algorithm SHA256).Hash.ToUpperInvariant()
if ($actualCudnnHash -ne $cudnnHash) { throw "cuDNN wheel checksum mismatch. Expected $cudnnHash; received $actualCudnnHash." }

$distRoot = [System.IO.Path]::GetFullPath($outputRoot).TrimEnd('\')
$resolvedReleaseRoot = [System.IO.Path]::GetFullPath($releaseRoot)
if (-not $resolvedReleaseRoot.StartsWith($distRoot + '\',[System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to write package outside the output folder: $resolvedReleaseRoot"
}
if (Test-Path -LiteralPath $resolvedReleaseRoot) { Remove-Item -LiteralPath $resolvedReleaseRoot -Recurse -Force }
New-Item -ItemType Directory -Path $resolvedReleaseRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $resolvedReleaseRoot "generated\kernels") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $resolvedReleaseRoot "models") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $resolvedReleaseRoot "licenses") -Force | Out-Null

$releaseBin = Join-Path $buildRoot "Release"
Copy-Item -LiteralPath (Join-Path $releaseBin "facelivt_attendance.exe") -Destination $resolvedReleaseRoot
Copy-Item -LiteralPath (Join-Path $releaseBin "facelivt_embed.exe") -Destination $resolvedReleaseRoot
Copy-Item -LiteralPath (Join-Path $releaseBin "facelivt_scrfd_smoke.exe") -Destination $resolvedReleaseRoot
Copy-Item -LiteralPath (Join-Path $releaseBin "facelivt_video.dll") -Destination $resolvedReleaseRoot
Copy-Item -LiteralPath (Join-Path $ortBin "onnxruntime.dll") -Destination $resolvedReleaseRoot
Copy-Item -LiteralPath (Join-Path $ortBin "onnxruntime_providers_cuda.dll") -Destination $resolvedReleaseRoot
Copy-Item -LiteralPath (Join-Path $ortBin "onnxruntime_providers_shared.dll") -Destination $resolvedReleaseRoot
foreach ($file in $cudaDlls) { Copy-Item -LiteralPath (Join-Path $cudaBin $file) -Destination $resolvedReleaseRoot }
Copy-Item -LiteralPath (Join-Path $projectRoot "facelivtv2-l.fp16.flvt") -Destination $resolvedReleaseRoot
Copy-Item -LiteralPath (Join-Path $projectRoot "models\scrfd_10g_bnkps.onnx") -Destination (Join-Path $resolvedReleaseRoot "models")
Copy-Item -LiteralPath (Join-Path $projectRoot "run_webcam.ps1") -Destination $resolvedReleaseRoot

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
  throw "Visual Studio Installer locator not found; the Windows C++ runtime DLLs must be bundled for one-click deployment."
}
$vsInstall = (& $vswhere -latest -products "*" -property installationPath | Select-Object -First 1)
if (-not $vsInstall) { throw "No Visual Studio installation found to supply the Windows C++ runtime DLLs." }
$redistRoot = Join-Path $vsInstall "VC\Redist\MSVC"
$crtSource = $null
if (Test-Path -LiteralPath $redistRoot -PathType Container) {
  $redistVersions = Get-ChildItem -LiteralPath $redistRoot -Directory |
    Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } |
    Sort-Object { [version]$_.Name } -Descending
  foreach ($redistVersion in $redistVersions) {
    $x64Root = Join-Path $redistVersion.FullName "x64"
    if (-not (Test-Path -LiteralPath $x64Root -PathType Container)) { continue }
    $candidate = Get-ChildItem -LiteralPath $x64Root -Directory |
      Where-Object { $_.Name -match '^Microsoft\.VC[^.]+\.CRT$' } |
      Select-Object -First 1
    if ($candidate -and
        (Test-Path -LiteralPath (Join-Path $candidate.FullName "vcruntime140.dll")) -and
        (Test-Path -LiteralPath (Join-Path $candidate.FullName "msvcp140.dll"))) {
      $crtSource = $candidate.FullName
      break
    }
  }
}
if (-not $crtSource) { throw "Could not find the x64 Visual C++ app-local runtime DLL set under '$redistRoot'." }
$crtDlls = @(Get-ChildItem -LiteralPath $crtSource -Filter "*.dll" -File)
if (-not ($crtDlls.Name -contains "vcruntime140_1.dll")) { throw "The x64 Visual C++ runtime set is incomplete: vcruntime140_1.dll is missing." }
foreach ($crtDll in $crtDlls) { Copy-Item -LiteralPath $crtDll.FullName -Destination $resolvedReleaseRoot }

$manifestOut = Join-Path $resolvedReleaseRoot "generated\kernels\kernel_manifest.tsv"
$filteredRows = @($manifest[0])
foreach ($row in ($manifest | Select-Object -Skip 1)) {
  if (-not $row) { continue }
  $columns = $row -split "`t"
  $arch = [int]$columns[0]
  if ($arch -notin $requiredArchitectures) { continue }
  $cubinPath = Join-Path $kernelRoot $columns[2]
  Require-File $cubinPath
  $destination = Join-Path (Join-Path $resolvedReleaseRoot "generated\kernels") $columns[2]
  New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
  Copy-Item -LiteralPath $cubinPath -Destination $destination
  $filteredRows += $row
}
[System.IO.File]::WriteAllLines($manifestOut,$filteredRows,[System.Text.UTF8Encoding]::new($false))

Add-Type -AssemblyName System.IO.Compression.FileSystem
$wheelArchive = [System.IO.Compression.ZipFile]::OpenRead($CudnnWheel)
try {
  $dllEntries = @($wheelArchive.Entries | Where-Object { $_.FullName -match '^nvidia/cudnn/bin/[^/]+\.dll$' })
  if ($dllEntries.Count -lt 5) { throw "cuDNN wheel did not contain the expected split DLL set." }
  foreach ($entry in $dllEntries) {
    $destination = Join-Path $resolvedReleaseRoot $entry.Name
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$destination,$true)
  }
  $licenseEntry = $wheelArchive.Entries | Where-Object { $_.FullName -match '\.dist-info/licenses/License\.txt$' } | Select-Object -First 1
  if (-not $licenseEntry) { throw "cuDNN wheel is missing its license text." }
  $licenseFile = Join-Path $resolvedReleaseRoot "licenses\NVIDIA-cuDNN-License.txt"
  $stream = $licenseEntry.Open(); $fileStream = [System.IO.File]::Create($licenseFile)
  try { $stream.CopyTo($fileStream) } finally { $fileStream.Dispose(); $stream.Dispose() }
} finally { $wheelArchive.Dispose() }

Copy-Item -LiteralPath (Join-Path $projectRoot "THIRD_PARTY.md") -Destination (Join-Path $resolvedReleaseRoot "licenses\THIRD_PARTY.md")
Copy-Item -LiteralPath (Join-Path $projectRoot "third_party\FaceLiVT_LICENSE_BSD3.txt") -Destination (Join-Path $resolvedReleaseRoot "licenses\FaceLiVT_LICENSE_BSD3.txt")
$ortLicenses = Get-ChildItem -LiteralPath $ortRoot -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -in @("LICENSE", "LICENSE.txt", "ThirdPartyNotices.txt") }
foreach ($ortLicense in $ortLicenses) {
  $licenseName = "ONNX-Runtime-$($ortLicense.Name)"
  if (Test-Path -LiteralPath (Join-Path $resolvedReleaseRoot "licenses\$licenseName")) { continue }
  Copy-Item -LiteralPath $ortLicense.FullName -Destination (Join-Path $resolvedReleaseRoot "licenses\$licenseName")
}
foreach ($cudaNotice in @((Join-Path $CudaRoot "EULA.txt"),(Join-Path $CudaRoot "LICENSE"))) {
  if (Test-Path -LiteralPath $cudaNotice -PathType Leaf) {
    Copy-Item -LiteralPath $cudaNotice -Destination (Join-Path $resolvedReleaseRoot ("licenses\NVIDIA-CUDA-" + (Split-Path $cudaNotice -Leaf)))
  }
}

$readme = @"
FaceLiVTv2 native attendance runtime for Windows x64
CUDA lane: $CudaMajor.x | ONNX Runtime: $ortVersion | Release: $ReleaseVersion

Quick start:
1. Download the ZIP from Releases.
2. Unzip the complete archive.
3. Double-click facelivt_attendance.exe.

This ZIP bundles both the FaceLiVT and SCRFD-10G detector models, all
architecture-specific kernels, the CUDA/cuDNN/ONNX Runtime DLLs, and the x64
Visual C++ runtime DLLs. No separate model download, CUDA Toolkit, Python,
PyTorch, Triton, or Visual C++ Redistributable installation is needed.

Requires Windows 10/11 x64, a compatible NVIDIA GPU and display driver, and an
available webcam. The CUDA 13 build includes GPU code for SM 7.5, 8.0, 8.6,
8.7, 8.9, 9.0, 10.0, and 12.0: GeForce RTX 20/30/40/50 series, plus selected
workstation and data-center GPUs. SM 8.7 kernels are also present, but this
Windows x64 package is not a Jetson release. End-to-end webcam validation was
performed on an RTX 5060 Laptop GPU; other listed architectures are compiled
into the package but have not each been individually validated.

The camera preview uses the Windows D3D11/CUDA GPU-surface path. In the app,
use the Benchmark GPU button to measure uncapped inference on a replayed frame.
The live display rate is limited by the selected camera's frame rate.

facelivt_embed.exe and facelivt_scrfd_smoke.exe are native command-line
diagnostics for FaceLiVT embedding and SCRFD/alignment.

Read licenses/THIRD_PARTY.md before use. SCRFD pretrained weights are available
under non-commercial research terms. Do not copy personal attendance records
into a public repository or release archive.
"@
[System.IO.File]::WriteAllText((Join-Path $resolvedReleaseRoot "README.txt"),$readme,[System.Text.UTF8Encoding]::new($false))

if (-not $SkipRuntimeTest) {
  $smokeDir = Join-Path $env:TEMP ("facelivt-release-smoke-" + [Guid]::NewGuid().ToString("N"))
  New-Item -ItemType Directory -Path $smokeDir -Force | Out-Null
  $oldPath = $env:PATH
  try {
    [System.IO.File]::WriteAllBytes((Join-Path $smokeDir "face112.bgr"),[byte[]]::new(112*112*3))
    [System.IO.File]::WriteAllBytes((Join-Path $smokeDir "blank.bgra"),[byte[]]::new(640*480*4))
    $env:PATH = "$resolvedReleaseRoot;$env:WINDIR\System32;$env:WINDIR"
    Push-Location $resolvedReleaseRoot
    try {
      & (Join-Path $resolvedReleaseRoot "facelivt_embed.exe") (Join-Path $resolvedReleaseRoot "facelivtv2-l.fp16.flvt") (Join-Path $resolvedReleaseRoot "generated\kernels\kernel_manifest.tsv") (Join-Path $smokeDir "face112.bgr") (Join-Path $smokeDir "embedding.f32")
      if ($LASTEXITCODE -ne 0) { throw "Packaged FaceLiVT CUDA smoke test failed (exit $LASTEXITCODE)." }
      if ((Get-Item (Join-Path $smokeDir "embedding.f32")).Length -ne 512*4) { throw "Packaged FaceLiVT smoke test produced an invalid embedding size." }
      & (Join-Path $resolvedReleaseRoot "facelivt_scrfd_smoke.exe") (Join-Path $resolvedReleaseRoot "models\scrfd_10g_bnkps.onnx") (Join-Path $smokeDir "blank.bgra") 640 480
      if ($LASTEXITCODE -ne 0) { throw "Packaged SCRFD CUDA smoke test failed (exit $LASTEXITCODE)." }
    } finally { Pop-Location }
  } finally {
    $env:PATH = $oldPath
    $tempRoot = [System.IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
    $resolvedSmokeDir = [System.IO.Path]::GetFullPath($smokeDir)
    if (-not $resolvedSmokeDir.StartsWith($tempRoot,[System.StringComparison]::OrdinalIgnoreCase)) { throw "Refusing to remove smoke-test files outside TEMP: $resolvedSmokeDir" }
    if (Test-Path -LiteralPath $resolvedSmokeDir -PathType Container) { Remove-Item -LiteralPath $resolvedSmokeDir -Recurse -Force }
  }
  Write-Host "Runtime smoke tests passed with PATH restricted to the package and Windows."
}

$unexpectedFiles = @(Get-ChildItem -LiteralPath $resolvedReleaseRoot -Recurse -File -Force | Where-Object { $_.Extension -in @(".py", ".pt", ".whl", ".nupkg") })
$unexpectedDirectories = @(Get-ChildItem -LiteralPath $resolvedReleaseRoot -Recurse -Directory -Force | Where-Object { $_.Name -in @("data", ".venv", "__pycache__") })
if ($unexpectedFiles.Count -or $unexpectedDirectories.Count) { throw "Release stage contains developer or personal files that must not ship." }

$hashes = Get-ChildItem -LiteralPath $resolvedReleaseRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
  $relative = $_.FullName.Substring($resolvedReleaseRoot.Length + 1)
  "{0} *{1}" -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(),$relative
}
[System.IO.File]::WriteAllLines((Join-Path $resolvedReleaseRoot "SHA256SUMS.txt"),$hashes,[System.Text.UTF8Encoding]::new($false))

$zipPath = Join-Path $outputRoot "facelivt-windows-cuda$CudaMajor-x64-v$ReleaseVersion.zip"
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
[System.IO.Compression.ZipFile]::CreateFromDirectory($resolvedReleaseRoot,$zipPath,[System.IO.Compression.CompressionLevel]::Optimal,$false)
Write-Host "Created release package: $zipPath"
Write-Host "Package contents are Python-free; no personal data/ folder was copied."
