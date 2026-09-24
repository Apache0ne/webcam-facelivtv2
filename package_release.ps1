param(
  [Parameter(Mandatory = $true)][ValidateSet(12, 13)][int]$CudaMajor,
  [ValidatePattern('^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$')][string]$ReleaseVersion = "0.1.2",
  [ValidatePattern('^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$')][string]$RuntimeBundleVersion = "1.0.0",
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
$requiredArchitectures = @(75,80,86,87,89,90,100,120)
if (-not $BuildDirectory) { $BuildDirectory = "build\cuda$CudaMajor" }
if (-not $KernelDirectory) { $KernelDirectory = "generated\cuda$CudaMajor\kernels" }

function Resolve-PathFromRoot([string]$Path) {
  if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
  return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}
function Require-File([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required release input is missing: $Path" }
}
function Copy-IfPresent([string]$Source, [string]$Destination) {
  if (Test-Path -LiteralPath $Source -PathType Leaf) { Copy-Item -LiteralPath $Source -Destination $Destination }
}
function Write-Checksums([string]$Directory, [string]$ChecksumFileName) {
  $hashes = Get-ChildItem -LiteralPath $Directory -Recurse -File |
    Where-Object { $_.Name -ne $ChecksumFileName } |
    Sort-Object FullName | ForEach-Object {
      $relative = $_.FullName.Substring($Directory.Length + 1).Replace('\','/')
      "{0} *{1}" -f (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(),$relative
    }
  [System.IO.File]::WriteAllLines((Join-Path $Directory $ChecksumFileName),$hashes,[System.Text.UTF8Encoding]::new($false))
}
function New-Zip([string]$SourceDirectory, [string]$ZipPath) {
  if (Test-Path -LiteralPath $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
  [System.IO.Compression.ZipFile]::CreateFromDirectory($SourceDirectory,$ZipPath,[System.IO.Compression.CompressionLevel]::Optimal,$false)
}

$buildRoot = Resolve-PathFromRoot $BuildDirectory
$kernelRoot = Resolve-PathFromRoot $KernelDirectory
$outputRoot = Resolve-PathFromRoot $OutputDirectory
$stageRoot = Join-Path $outputRoot ".staging\facelivt-cuda$CudaMajor-v$ReleaseVersion"
$appStage = Join-Path $stageRoot "app"
$runtimeStage = Join-Path $stageRoot "runtime"
$appZip = Join-Path $outputRoot "facelivt-app-windows-cuda$CudaMajor-x64-v$ReleaseVersion.zip"
$runtimeZip = Join-Path $outputRoot "facelivt-runtime-windows-cuda$CudaMajor-x64-r$RuntimeBundleVersion.zip"

$buildCache = Join-Path $buildRoot "CMakeCache.txt"
Require-File $buildCache
$releaseBin = Join-Path $buildRoot "Release"
foreach ($file in @("facelivt_attendance.exe", "facelivt_video.dll", "facelivt_embed.exe", "facelivt_scrfd_smoke.exe")) {
  Require-File (Join-Path $releaseBin $file)
}
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
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$nvccInfo = & $nvcc --version
if ($LASTEXITCODE -ne 0 -or ($nvccInfo -join " ") -notmatch "release\s+$CudaMajor\.") {
  throw "The build directory is not configured for CUDA $CudaMajor."
}
$cudaReleaseMatch = [regex]::Match(($nvccInfo -join " "),'release\s+(\d+\.\d+)')
$cudaRelease = if ($cudaReleaseMatch.Success) { $cudaReleaseMatch.Groups[1].Value } else { "$CudaMajor.x" }
$archLine = $cache | Where-Object { $_ -match '^CMAKE_CUDA_ARCHITECTURES(:\w+)?=' } | Select-Object -First 1
if (-not $archLine) { throw "The CMake build did not record CMAKE_CUDA_ARCHITECTURES." }
$nativeArchitectures = (($archLine -split '=',2)[1] -split ';' | ForEach-Object { [int]($_ -replace '-real$','') })
if ($CudaMajor -eq 13 -and ($nativeArchitectures | Where-Object { $_ -lt 75 })) { throw "CUDA 13 release cannot include native detector code below SM 7.5." }
foreach ($arch in $requiredArchitectures) {
  if ($arch -notin $nativeArchitectures) { throw "Native detector CUDA objects are missing sm_$arch; rebuild with -KernelArch $($requiredArchitectures -join ',')." }
}

$manifest = Get-Content -LiteralPath (Join-Path $kernelRoot "kernel_manifest.tsv")
$expectedManifestHeader = "arch`talias`tcubin`tentry`twarps`tshared`tscratch_args"
if (@($manifest).Count -lt 2 -or -not [string]::Equals([string]$manifest[0],$expectedManifestHeader,[System.StringComparison]::Ordinal)) {
  throw "Release kernels must use the architecture matrix manifest generated by tools/build_kernels.py (found $(@($manifest).Count) rows; header='$($manifest[0])')."
}
$builtArchitectures = @($manifest | Select-Object -Skip 1 | Where-Object { $_ } | ForEach-Object { [int](($_ -split "`t",2)[0]) } | Sort-Object -Unique)
foreach ($arch in $requiredArchitectures) {
  if ($arch -notin $builtArchitectures) { throw "Kernel matrix is missing sm_$arch. Rebuild with -KernelArch $($requiredArchitectures -join ',')." }
}

$ortRoot = Join-Path $projectRoot "build\deps\onnxruntime-gpu-windows-$ortVersion"
$ortBin = Join-Path $ortRoot "runtimes\win-x64\native"
$ortDlls = @("onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll")
foreach ($file in $ortDlls) { Require-File (Join-Path $ortBin $file) }
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

$outputPrefix = [System.IO.Path]::GetFullPath($outputRoot).TrimEnd('\') + '\'
foreach ($path in @($stageRoot,$appZip,$runtimeZip)) {
  $resolved = [System.IO.Path]::GetFullPath($path)
  if (-not $resolved.StartsWith($outputPrefix,[System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write release output outside the output folder: $resolved"
  }
}
if (Test-Path -LiteralPath $stageRoot) { Remove-Item -LiteralPath $stageRoot -Recurse -Force }
New-Item -ItemType Directory -Path (Join-Path $appStage "generated\kernels") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $appStage "models") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $appStage "licenses") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $runtimeStage "licenses") -Force | Out-Null

# The compact application archive contains only FaceLiVT-owned files, models,
# architecture-specific kernels, notices, and the small app-local MSVC runtime.
foreach ($file in @("facelivt_attendance.exe", "facelivt_embed.exe", "facelivt_scrfd_smoke.exe", "facelivt_video.dll")) {
  Copy-Item -LiteralPath (Join-Path $releaseBin $file) -Destination $appStage
}
Copy-Item -LiteralPath (Join-Path $projectRoot "facelivtv2-l.fp16.flvt") -Destination $appStage
Copy-Item -LiteralPath (Join-Path $projectRoot "models\scrfd_10g_bnkps.onnx") -Destination (Join-Path $appStage "models")
Copy-Item -LiteralPath (Join-Path $projectRoot "run_webcam.ps1") -Destination $appStage
$filteredRows = @($manifest[0])
foreach ($row in ($manifest | Select-Object -Skip 1)) {
  if (-not $row) { continue }
  $columns = $row -split "`t"
  $arch = [int]$columns[0]
  if ($arch -notin $requiredArchitectures) { continue }
  $cubinPath = Join-Path $kernelRoot $columns[2]
  Require-File $cubinPath
  $destination = Join-Path (Join-Path $appStage "generated\kernels") $columns[2]
  New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
  Copy-Item -LiteralPath $cubinPath -Destination $destination
  $filteredRows += $row
}
[System.IO.File]::WriteAllLines((Join-Path $appStage "generated\kernels\kernel_manifest.tsv"),$filteredRows,[System.Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath (Join-Path $projectRoot "THIRD_PARTY.md") -Destination (Join-Path $appStage "licenses\THIRD_PARTY.md")
Copy-Item -LiteralPath (Join-Path $projectRoot "third_party\FaceLiVT_LICENSE_BSD3.txt") -Destination (Join-Path $appStage "licenses\FaceLiVT_LICENSE_BSD3.txt")

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
foreach ($crtDll in $crtDlls) { Copy-Item -LiteralPath $crtDll.FullName -Destination $appStage }

# The runtime archive contains only the large third-party GPU/runtime DLL set.
foreach ($file in $cudaDlls) { Copy-Item -LiteralPath (Join-Path $cudaBin $file) -Destination $runtimeStage }
foreach ($file in $ortDlls) { Copy-Item -LiteralPath (Join-Path $ortBin $file) -Destination $runtimeStage }

Add-Type -AssemblyName System.IO.Compression.FileSystem
$wheelArchive = [System.IO.Compression.ZipFile]::OpenRead($CudnnWheel)
try {
  $dllEntries = @($wheelArchive.Entries | Where-Object { $_.FullName -match '^nvidia/cudnn/bin/[^/]+\.dll$' })
  if ($dllEntries.Count -lt 5) { throw "cuDNN wheel did not contain the expected split DLL set." }
  foreach ($entry in $dllEntries) {
    $destination = Join-Path $runtimeStage $entry.Name
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$destination,$true)
  }
  $licenseEntry = $wheelArchive.Entries | Where-Object { $_.FullName -match '\.dist-info/licenses/License\.txt$' } | Select-Object -First 1
  if (-not $licenseEntry) { throw "cuDNN wheel is missing its license text." }
  $licenseFile = Join-Path $runtimeStage "licenses\NVIDIA-cuDNN-License.txt"
  $stream = $licenseEntry.Open(); $fileStream = [System.IO.File]::Create($licenseFile)
  try { $stream.CopyTo($fileStream) } finally { $fileStream.Dispose(); $stream.Dispose() }
} finally { $wheelArchive.Dispose() }

$ortLicenses = Get-ChildItem -LiteralPath $ortRoot -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -in @("LICENSE", "LICENSE.txt", "ThirdPartyNotices.txt") }
foreach ($ortLicense in $ortLicenses) {
  $licenseName = "ONNX-Runtime-$($ortLicense.Name)"
  if (Test-Path -LiteralPath (Join-Path $runtimeStage "licenses\$licenseName")) { continue }
  Copy-Item -LiteralPath $ortLicense.FullName -Destination (Join-Path $runtimeStage "licenses\$licenseName")
}
foreach ($cudaNotice in @((Join-Path $CudaRoot "EULA.txt"),(Join-Path $CudaRoot "LICENSE"))) {
  if (Test-Path -LiteralPath $cudaNotice -PathType Leaf) {
    Copy-Item -LiteralPath $cudaNotice -Destination (Join-Path $runtimeStage ("licenses\NVIDIA-CUDA-" + (Split-Path $cudaNotice -Leaf)))
  }
}
if (-not (Get-ChildItem -LiteralPath (Join-Path $runtimeStage "licenses") -File | Where-Object { $_.Name -like "NVIDIA-CUDA-*" })) {
  throw "The selected CUDA Toolkit did not provide its EULA/license text; refusing to package its DLLs without notices."
}

$architectureText = (($requiredArchitectures | ForEach-Object { "SM $_" }) -join ", ")
$appReadme = @"
FaceLiVTv2 native attendance app for Windows x64
App release: $ReleaseVersion | CUDA lane: $CudaMajor.x | Required runtime bundle: r$RuntimeBundleVersion | CUDA build toolkit: $cudaRelease | ONNX Runtime: $ortVersion

First install:
1. Download the matching facelivt-runtime-windows-cuda$CudaMajor-x64-r$RuntimeBundleVersion.zip and facelivt-app-windows-cuda$CudaMajor-x64-v$ReleaseVersion.zip.
2. Extract both ZIPs into the same new folder. If Windows asks, merge the files into that folder.
3. Double-click facelivt_attendance.exe.

For later app updates, extract only the newer app ZIP over this folder when its
README names the runtime bundle revision you already have. If a release names a
new runtime bundle revision, download that matching runtime ZIP too. Never mix
CUDA 12 and CUDA 13 bundles.

This app ZIP contains the FaceLiVT FP16 model, SCRFD-10G ONNX detector, native
programs, GPU kernels ($architectureText), and the small app-local Microsoft
C++ runtime. The matching runtime ZIP contains CUDA, cuDNN, and ONNX Runtime.
Neither ZIP needs Python, PyTorch, Triton, the CUDA Toolkit, WSL, or a separate
Visual C++ Redistributable to run. Windows x64, a compatible NVIDIA display
driver, and a webcam are required.

The app stores enrollment embeddings and attendance records in data\; keep that
folder private. The SCRFD pretrained model is distributed under non-commercial
research terms. Read licenses\THIRD_PARTY.md before use.

facelivt_embed.exe and facelivt_scrfd_smoke.exe are optional GPU diagnostics.
Run_webcam.ps1 starts the camera application from PowerShell.
"@
[System.IO.File]::WriteAllText((Join-Path $appStage "APP_README.txt"),$appReadme,[System.Text.UTF8Encoding]::new($false))

$runtimeReadme = @"
FaceLiVTv2 Windows x64 runtime DLL bundle
Runtime bundle: r$RuntimeBundleVersion | Built for app release: $ReleaseVersion | CUDA lane: $CudaMajor.x | CUDA Toolkit DLLs: $cudaRelease
ONNX Runtime: $ortVersion | cuDNN: $cudnnVersion

Extract this runtime ZIP and the matching facelivt-app-windows-cuda$CudaMajor-x64-v$ReleaseVersion.zip into the same folder. The app ZIP contains the
program, models, and GPU kernels. Keep these runtime DLLs beside the app files.
Do not mix the CUDA 12 runtime with the CUDA 13 app, or vice versa.

This ZIP contains the CUDA, cuDNN, and ONNX Runtime DLLs needed by the matching
FaceLiVT app. It does not contain a GPU driver. Install a compatible NVIDIA
display driver on the PC that will run the app. Read the license files in
licenses\ before redistributing this bundle.
"@
[System.IO.File]::WriteAllText((Join-Path $runtimeStage "RUNTIME_README.txt"),$runtimeReadme,[System.Text.UTF8Encoding]::new($false))

foreach ($file in $cudaDlls + $ortDlls) { Require-File (Join-Path $runtimeStage $file) }
if ((Get-ChildItem -LiteralPath $runtimeStage -Filter "cudnn*.dll" -File).Count -lt 5) { throw "The cuDNN runtime DLL set is incomplete." }
foreach ($file in @("facelivt_attendance.exe", "facelivt_video.dll", "facelivtv2-l.fp16.flvt", "models\scrfd_10g_bnkps.onnx", "generated\kernels\kernel_manifest.tsv")) {
  Require-File (Join-Path $appStage $file)
}
foreach ($file in $cudaDlls + $ortDlls) {
  if (Test-Path -LiteralPath (Join-Path $appStage $file)) { throw "Large runtime DLL '$file' was copied into the compact app bundle." }
}
$appPaths = @(Get-ChildItem -LiteralPath $appStage -Recurse -File | ForEach-Object { $_.FullName.Substring($appStage.Length + 1).Replace('\','/').ToLowerInvariant() })
$runtimePaths = @(Get-ChildItem -LiteralPath $runtimeStage -Recurse -File | ForEach-Object { $_.FullName.Substring($runtimeStage.Length + 1).Replace('\','/').ToLowerInvariant() })
$collidingPaths = @($appPaths | Where-Object { $_ -in $runtimePaths })
if ($collidingPaths.Count) { throw "App and runtime ZIPs would overwrite the same installed files: $($collidingPaths -join ', ')" }

if (-not $SkipRuntimeTest) {
  $smokeDir = Join-Path $env:TEMP ("facelivt-release-smoke-" + [Guid]::NewGuid().ToString("N"))
  $smokePackageRoot = Join-Path $stageRoot "smoke-package"
  New-Item -ItemType Directory -Path $smokeDir -Force | Out-Null
  New-Item -ItemType Directory -Path $smokePackageRoot -Force | Out-Null
  $oldPath = $env:PATH
  try {
    foreach ($sourceRoot in @($appStage,$runtimeStage)) {
      foreach ($sourceFile in (Get-ChildItem -LiteralPath $sourceRoot -Recurse -File)) {
        $relativePath = $sourceFile.FullName.Substring($sourceRoot.Length + 1)
        $linkedPath = Join-Path $smokePackageRoot $relativePath
        New-Item -ItemType Directory -Path (Split-Path $linkedPath -Parent) -Force | Out-Null
        New-Item -ItemType HardLink -Path $linkedPath -Target $sourceFile.FullName | Out-Null
      }
    }
    [System.IO.File]::WriteAllBytes((Join-Path $smokeDir "face112.bgr"),[byte[]]::new(112*112*3))
    [System.IO.File]::WriteAllBytes((Join-Path $smokeDir "blank.bgra"),[byte[]]::new(640*480*4))
    $env:PATH = "$smokePackageRoot;$env:WINDIR\System32;$env:WINDIR"
    Push-Location $smokePackageRoot
    try {
      & (Join-Path $smokePackageRoot "facelivt_embed.exe") (Join-Path $smokePackageRoot "facelivtv2-l.fp16.flvt") (Join-Path $smokePackageRoot "generated\kernels\kernel_manifest.tsv") (Join-Path $smokeDir "face112.bgr") (Join-Path $smokeDir "embedding.f32")
      if ($LASTEXITCODE -ne 0) { throw "Packaged FaceLiVT CUDA smoke test failed (exit $LASTEXITCODE)." }
      if ((Get-Item (Join-Path $smokeDir "embedding.f32")).Length -ne 512*4) { throw "Packaged FaceLiVT smoke test produced an invalid embedding size." }
      & (Join-Path $smokePackageRoot "facelivt_scrfd_smoke.exe") (Join-Path $smokePackageRoot "models\scrfd_10g_bnkps.onnx") (Join-Path $smokeDir "blank.bgra") 640 480
      if ($LASTEXITCODE -ne 0) { throw "Packaged SCRFD CUDA smoke test failed (exit $LASTEXITCODE)." }
    } finally { Pop-Location }
  } finally {
    $env:PATH = $oldPath
    $tempRoot = [System.IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
    $resolvedSmokeDir = [System.IO.Path]::GetFullPath($smokeDir)
    if (-not $resolvedSmokeDir.StartsWith($tempRoot,[System.StringComparison]::OrdinalIgnoreCase)) { throw "Refusing to remove smoke-test files outside TEMP: $resolvedSmokeDir" }
    if (Test-Path -LiteralPath $resolvedSmokeDir -PathType Container) { Remove-Item -LiteralPath $resolvedSmokeDir -Recurse -Force }
    $stagePrefix = [System.IO.Path]::GetFullPath($stageRoot).TrimEnd('\') + '\'
    $resolvedSmokePackageRoot = [System.IO.Path]::GetFullPath($smokePackageRoot)
    if (-not $resolvedSmokePackageRoot.StartsWith($stagePrefix,[System.StringComparison]::OrdinalIgnoreCase)) { throw "Refusing to remove smoke-test links outside the release stage: $resolvedSmokePackageRoot" }
    if (Test-Path -LiteralPath $resolvedSmokePackageRoot -PathType Container) { Remove-Item -LiteralPath $resolvedSmokePackageRoot -Recurse -Force }
  }
  Write-Host "GPU inference smoke tests passed with PATH restricted to the app/runtime ZIP contents and Windows."
} else {
  Write-Host "GPU inference smoke tests skipped. This is appropriate for CPU-only build runners; test the inference and webcam on an NVIDIA GPU before claiming GPU validation."
}

$stageDirectories = @($appStage,$runtimeStage)
$unexpectedFiles = @($stageDirectories | ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File -Force } | Where-Object { $_.Extension -in @(".py", ".pt", ".whl", ".nupkg") })
$unexpectedDirectories = @($stageDirectories | ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -Directory -Force } | Where-Object { $_.Name -in @("data", ".venv", "__pycache__") })
if ($unexpectedFiles.Count -or $unexpectedDirectories.Count) { throw "Release stage contains developer or personal files that must not ship." }

Write-Checksums $appStage "APP_SHA256SUMS.txt"
Write-Checksums $runtimeStage "RUNTIME_SHA256SUMS.txt"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
New-Zip $appStage $appZip
New-Zip $runtimeStage $runtimeZip
Write-Host "Created compact app package:   $appZip"
Write-Host "Created CUDA $CudaMajor runtime package: $runtimeZip"
Write-Host "For a first install, extract both matching ZIPs into the same folder."
