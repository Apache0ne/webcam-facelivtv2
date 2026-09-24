param(
  [Parameter(Mandatory = $true)][ValidateSet(12, 13)][int]$CudaMajor
)

$ErrorActionPreference = "Stop"
$version = if ($CudaMajor -eq 12) { "12.8.1" } else { "13.3.0" }
$toolkitVersion = if ($CudaMajor -eq 12) { "12.8" } else { "13.3" }
$installerUrl = if ($CudaMajor -eq 12) {
  "https://developer.download.nvidia.com/compute/cuda/12.8.1/network_installers/cuda_12.8.1_windows_network.exe"
} else {
  "https://developer.download.nvidia.com/compute/cuda/13.3.0/network_installers/cuda_13.3.0_windows_network.exe"
}
$components = @(
  "nvcc_$toolkitVersion",
  "cudart_$toolkitVersion",
  "cublas_$toolkitVersion",
  "cufft_$toolkitVersion",
  "curand_$toolkitVersion",
  "nvrtc_$toolkitVersion",
  "nvjitlink_$toolkitVersion",
  "visual_studio_integration_$toolkitVersion"
)
$cudaRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v$toolkitVersion"
$installer = Join-Path $env:RUNNER_TEMP "cuda-$version-windows-network.exe"

Write-Host "Downloading NVIDIA's CUDA $version network installer."
Invoke-WebRequest -Uri $installerUrl -OutFile $installer
$signature = Get-AuthenticodeSignature -LiteralPath $installer
if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notmatch "NVIDIA") {
  throw "The CUDA installer did not have a valid NVIDIA Authenticode signature."
}
Write-Host "Installing nvcc, required CUDA libraries, and Visual Studio build customizations; no GPU driver is requested."
$arguments = @("-n", "-s") + $components
$process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru -NoNewWindow
if ($process.ExitCode -ne 0) { throw "CUDA $version installer failed (exit $($process.ExitCode))." }
if (-not (Test-Path -LiteralPath (Join-Path $cudaRoot "bin\nvcc.exe") -PathType Leaf)) {
  throw "CUDA installer completed but nvcc was not found under '$cudaRoot'."
}

$nvccOutput = & (Join-Path $cudaRoot "bin\nvcc.exe") --version
if ($LASTEXITCODE -ne 0 -or ($nvccOutput -join " ") -notmatch "release\s+$CudaMajor\.") {
  throw "Installed toolkit did not report CUDA $CudaMajor as expected."
}
"FACELIVT_CUDA_ROOT=$cudaRoot" | Add-Content -LiteralPath $env:GITHUB_ENV -Encoding utf8
"CUDA_PATH=$cudaRoot" | Add-Content -LiteralPath $env:GITHUB_ENV -Encoding utf8
Write-Host "Installed CUDA $version build toolkit at $cudaRoot"
Remove-Item -LiteralPath $installer -Force
