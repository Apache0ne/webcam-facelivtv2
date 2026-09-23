param(
    [string]$Python = "",
    [int]$Camera = 0
)
$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Python)) {
    $localPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $localPython -PathType Leaf) { $localPython } else { "python" }
}
if (Test-Path -LiteralPath $Python -PathType Leaf) {
    $pythonExe = (Resolve-Path -LiteralPath $Python).Path
} else {
    $pythonCommand = Get-Command -Name $Python -CommandType Application -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw "Python executable not found: $Python. Run .\setup_windows.ps1 or pass -Python." }
    $pythonExe = $pythonCommand.Source
}

$requiredFiles = @(
    "facelivtv2-l.fp16.flvt",
    "generated\kernels\kernel_manifest.tsv",
    "models\scrfd_10g_bnkps.onnx"
)
$missingFiles = @($requiredFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $projectRoot $_) -PathType Leaf) })
if ($missingFiles.Count -gt 0) {
    throw "Runtime files are missing: $($missingFiles -join ', '). Download the SCRFD model if needed, then run .\build_windows.ps1."
}
$runtimeDirectories = @((Join-Path $projectRoot "build\Release"), (Join-Path $projectRoot "build"))
$runtimeDirectory = $runtimeDirectories | Where-Object {
    (Test-Path -LiteralPath (Join-Path $_ "facelivt_native.dll") -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $_ "facelivt_video.dll") -PathType Leaf)
} | Select-Object -First 1
if (-not $runtimeDirectory) {
    throw "Webcam DLLs are missing. Run .\build_windows.ps1 before starting the app."
}

$runtimeProbe = "import sys, cv2, numpy, torch, onnxruntime as ort; assert torch.cuda.is_available(), 'PyTorch cannot see the NVIDIA GPU'; assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'onnxruntime-gpu CUDA provider is unavailable'; print('Starting with', torch.cuda.get_device_name(0), '| Python', sys.version.split()[0], '| PyTorch CUDA', torch.version.cuda, '| ONNX Runtime CUDA available')"
& $pythonExe -c $runtimeProbe
if ($LASTEXITCODE -ne 0) { throw "Python/CUDA runtime preflight failed. Run .\setup_windows.ps1 or pass the correct Python with -Python." }

Push-Location $projectRoot
try {
    & $pythonExe webcam_app.py --camera $Camera
    if ($LASTEXITCODE -ne 0) { throw "Webcam app exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
