param(
  [string]$Version = "1.27.0",
  [string]$Destination = "build\deps"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($Destination)) {
  $Destination = Join-Path $projectRoot $Destination
}

$knownHashes = @{
  "1.26.0" = "2820B8CCA24B7C2817F28AA6EA8774A5C519295CAA7447B86D190C7079AABF0F"
  "1.27.0" = "20DA5E7A22409E77BB8F7796920FE0931DA16741A61037F0138EC812072E71F8"
}
if (-not $knownHashes.ContainsKey($Version)) {
  throw "No verified ONNX Runtime GPU package hash is recorded for version $Version."
}

$package = "microsoft.ml.onnxruntime.gpu.windows"
$archive = Join-Path $Destination "onnxruntime-gpu-windows-$Version.nupkg"
$unpacked = Join-Path $Destination "onnxruntime-gpu-windows-$Version"
$uri = "https://api.nuget.org/v3-flatcontainer/$package/$Version/$package.$Version.nupkg"
New-Item -ItemType Directory -Path $Destination -Force | Out-Null

if (Test-Path -LiteralPath $archive -PathType Leaf) {
  $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
  if ($actual -ne $knownHashes[$Version]) {
    throw "Existing ONNX Runtime package has the wrong SHA-256: $archive"
  }
} else {
  Write-Host "Downloading the pinned ONNX Runtime CUDA C++ package ($Version)..."
  Invoke-WebRequest -Uri $uri -OutFile $archive
  $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
  if ($actual -ne $knownHashes[$Version]) {
    Remove-Item -LiteralPath $archive -Force
    throw "Downloaded ONNX Runtime package failed SHA-256 verification."
  }
}

$header = Join-Path $unpacked "buildTransitive\native\include\onnxruntime_cxx_api.h"
$library = Join-Path $unpacked "runtimes\win-x64\native\onnxruntime.lib"
$cudaProvider = Join-Path $unpacked "runtimes\win-x64\native\onnxruntime_providers_cuda.dll"
if (-not (Test-Path -LiteralPath $header) -or -not (Test-Path -LiteralPath $library) -or
    -not (Test-Path -LiteralPath $cudaProvider)) {
  if (Test-Path -LiteralPath $unpacked) {
    throw "The existing ONNX Runtime extraction is incomplete: $unpacked. Remove or rename it, then rerun this script."
  }
  Expand-Archive -LiteralPath $archive -DestinationPath $unpacked
}

foreach ($required in @($header,$library,$cudaProvider)) {
  if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
    throw "ONNX Runtime package is missing a required native file: $required"
  }
}
Write-Host "ONNX Runtime CUDA C++ SDK ready: $unpacked"
