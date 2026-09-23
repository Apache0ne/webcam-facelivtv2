$ErrorActionPreference = "Stop"
$modelDirectory = Join-Path $PSScriptRoot "models"
$modelPath = Join-Path $modelDirectory "scrfd_10g_bnkps.onnx"
$downloadUrl = "https://huggingface.co/LPDoctor/insightface/resolve/main/scrfd_10g_bnkps.onnx"
$expectedSha256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"

Write-Host "SCRFD pretrained weights have non-commercial research terms. Read THIRD_PARTY.md before continuing."
if (Test-Path -LiteralPath $modelPath -PathType Leaf) {
    $existingHash = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($existingHash -eq $expectedSha256) {
        Write-Host "SCRFD model already present and checksum verified: $modelPath"
        exit 0
    }
    throw "A different file already exists at '$modelPath'. The script left it untouched; move it aside only if you want to replace it."
}

New-Item -ItemType Directory -Path $modelDirectory -Force | Out-Null
$temporaryPath = "$modelPath.$([Guid]::NewGuid().ToString('N')).download"
try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $temporaryPath
    $downloadHash = (Get-FileHash -LiteralPath $temporaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($downloadHash -ne $expectedSha256) {
        throw "SCRFD download checksum mismatch. Expected $expectedSha256; received $downloadHash."
    }
    Move-Item -LiteralPath $temporaryPath -Destination $modelPath
    Write-Host "SCRFD model downloaded and verified: $modelPath"
} catch {
    if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }
    throw
}
