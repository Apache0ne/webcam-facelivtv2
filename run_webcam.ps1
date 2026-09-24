param(
  [string]$PackageRoot = $PSScriptRoot,
  [ValidateRange(0, 32)][int]$Camera = 0,
  [switch]$Benchmark
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$app = Join-Path $root "facelivt_attendance.exe"
$required = @(
  "facelivtv2-l.fp16.flvt",
  "generated\kernels\kernel_manifest.tsv",
  "models\scrfd_10g_bnkps.onnx",
  "facelivt_video.dll",
  "onnxruntime.dll",
  "onnxruntime_providers_shared.dll",
  "onnxruntime_providers_cuda.dll"
)
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $root $_) -PathType Leaf) })
if ($missing.Count) { throw "The native release package is incomplete: $($missing -join ', ')" }
if (-not (Test-Path -LiteralPath $app -PathType Leaf)) { throw "Native application not found: $app" }

$cudaMajor = if (Test-Path -LiteralPath (Join-Path $root "cudart64_13.dll")) { 13 } elseif (Test-Path -LiteralPath (Join-Path $root "cudart64_12.dll")) { 12 } else { 0 }
if (-not $cudaMajor) { throw "CUDA runtime DLLs are missing. Extract the matching app and runtime ZIPs into this folder." }
$expected = if ($cudaMajor -eq 13) { "cudart64_13.dll" } else { "cudart64_12.dll" }
if (-not (Test-Path -LiteralPath (Join-Path $root $expected) -PathType Leaf)) { throw "CUDA $cudaMajor runtime DLL is missing." }

Write-Host "Starting the native CUDA $cudaMajor attendance app. Python and the CUDA Toolkit are not used at runtime."
Write-Host "Press Esc or close the app window to stop it."
$arguments = @("--root", $root, "--camera", [string]$Camera)
if ($Benchmark) { $arguments += "--benchmark" }
Push-Location $root
try {
  & $app @arguments
  if ($LASTEXITCODE -ne 0) { throw "Attendance app exited with code $LASTEXITCODE." }
} finally { Pop-Location }
