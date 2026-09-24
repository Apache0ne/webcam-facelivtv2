param(
  [Parameter(Mandatory = $true)][ValidateSet(12, 13)][int]$CudaMajor,
  [ValidatePattern('^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$')][string]$ReleaseVersion = "0.1.2",
  [ValidatePattern('^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$')][string]$RuntimeBundleVersion = "1.0.0",
  [string]$OutputDirectory = "dist"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $root $OutputDirectory }
$appZip = Join-Path $outputRoot "facelivt-app-windows-cuda$CudaMajor-x64-v$ReleaseVersion.zip"
$runtimeZip = Join-Path $outputRoot "facelivt-runtime-windows-cuda$CudaMajor-x64-r$RuntimeBundleVersion.zip"
$cudaDlls = if ($CudaMajor -eq 13) {
  @("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll", "cufft64_12.dll", "curand64_10.dll", "nvrtc64_130_0.dll", "nvrtc-builtins64_133.dll", "nvJitLink_130_0.dll")
} else {
  @("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll", "curand64_10.dll", "nvrtc64_120_0.dll", "nvrtc-builtins64_128.dll", "nvJitLink_120_0.dll")
}
$ortDlls = @("onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll")

Add-Type -AssemblyName System.IO.Compression.FileSystem
function Get-ZipEntries([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Release ZIP was not created: $Path" }
  return [System.IO.Compression.ZipFile]::OpenRead($Path)
}
function Test-ArchiveChecksums([System.IO.Compression.ZipArchive]$Archive, [string]$Label, [string]$ChecksumName) {
  $checksumEntry = $Archive.GetEntry($ChecksumName)
  if (-not $checksumEntry) { throw "$Label ZIP has no $ChecksumName." }
  $reader = [System.IO.StreamReader]::new($checksumEntry.Open())
  try { $lines = @($reader.ReadToEnd() -split "`r?`n" | Where-Object { $_ }) } finally { $reader.Dispose() }
  if (-not $lines.Count) { throw "$Label checksum file is empty." }
  $checked = 0
  foreach ($line in $lines) {
    if ($line -notmatch '^([0-9a-fA-F]{64}) \*(.+)$') { throw "$Label checksum row is malformed: $line" }
    $expectedHash = $Matches[1].ToLowerInvariant()
    $relativePath = $Matches[2].Replace('\','/')
    if ($relativePath.StartsWith('/') -or $relativePath -match '(^|/)\.\.(?:/|$)') { throw "$Label contains an unsafe checksum path: $relativePath" }
    $entry = $Archive.GetEntry($relativePath)
    if (-not $entry) { throw "$Label checksum references a missing file: $relativePath" }
    $stream = $entry.Open()
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
      $actualHash = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-','').ToLowerInvariant()
    } finally { $sha.Dispose(); $stream.Dispose() }
    if ($actualHash -ne $expectedHash) { throw "$Label file failed SHA-256 verification: $relativePath" }
    $checked++
  }
  $fileEntryCount = @($Archive.Entries | Where-Object { -not $_.FullName.EndsWith('/') }).Count
  if ($checked -ne ($fileEntryCount - 1)) { throw "$Label checksum count does not cover every file in the ZIP." }
  Write-Host "$Label ZIP: verified $checked file checksums."
}

$app = Get-ZipEntries $appZip
$runtime = Get-ZipEntries $runtimeZip
try {
  $appNames = @($app.Entries | ForEach-Object { $_.FullName })
  $runtimeNames = @($runtime.Entries | ForEach-Object { $_.FullName })
  foreach ($path in @(
    "facelivt_attendance.exe",
    "facelivt_embed.exe",
    "facelivt_scrfd_smoke.exe",
    "facelivt_video.dll",
    "facelivtv2-l.fp16.flvt",
    "vcruntime140.dll",
    "msvcp140.dll",
    "models/scrfd_10g_bnkps.onnx",
    "generated/kernels/kernel_manifest.tsv",
    "APP_README.txt",
    "APP_SHA256SUMS.txt"
  )) {
    if ($path -notin $appNames) { throw "App ZIP is missing required file: $path" }
  }
  foreach ($file in ($cudaDlls + $ortDlls)) {
    if ($file -notin $runtimeNames) { throw "Runtime ZIP is missing required DLL: $file" }
    if ($file -in $appNames) { throw "Large vendor DLL '$file' is duplicated in the app ZIP." }
  }
  $overlappingFiles = @($appNames | Where-Object { $_ -in $runtimeNames })
  if ($overlappingFiles.Count) { throw "App and runtime ZIPs contain duplicate paths that would overwrite on extraction: $($overlappingFiles -join ', ')" }
  foreach ($path in @("RUNTIME_README.txt", "RUNTIME_SHA256SUMS.txt")) {
    if ($path -notin $runtimeNames) { throw "Runtime ZIP is missing required file: $path" }
  }
  if (@($appNames | Where-Object { $_ -match '^cudnn[^/]*\.dll$' }).Count) { throw "The cuDNN DLLs must stay in the large runtime ZIP." }
  if (-not (@($runtimeNames | Where-Object { $_ -match '^cudnn[^/]*\.dll$' }).Count -ge 5)) {
    throw "Runtime ZIP is missing its cuDNN DLL set."
  }
  foreach ($pattern in @("facelivt_attendance.exe", "facelivtv2-l.fp16.flvt", "models/", "generated/kernels/")) {
    if (@($runtimeNames | Where-Object { $_ -like "$pattern*" }).Count) { throw "Runtime ZIP unexpectedly contains app files matching '$pattern'." }
  }
  if (@($appNames | Where-Object { $_ -match '\.(py|pt|whl|nupkg)$' }).Count) { throw "App ZIP contains a developer-only Python/checkpoint/package file." }
  Test-ArchiveChecksums $app "App" "APP_SHA256SUMS.txt"
  Test-ArchiveChecksums $runtime "Runtime" "RUNTIME_SHA256SUMS.txt"
} finally {
  $app.Dispose()
  $runtime.Dispose()
}

Write-Host "CUDA $CudaMajor split release structure is valid: app files and large runtime DLLs are separated."
