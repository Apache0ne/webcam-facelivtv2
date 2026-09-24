# Native runtime checks

The built `facelivt_embed.exe` runs one 112x112 BGR image through FaceLiVT and writes a normalized 512-float embedding. `facelivt_scrfd_smoke.exe` runs a packed BGRA frame through SCRFD and five-point GPU alignment. Both are native diagnostics and require a compatible CUDA driver plus the model and kernel files.

The scripts in this folder that use Python are developer-side model conversion/reference comparison helpers. They are not part of the release runtime. A live camera/display check still needs to be done interactively with `run_webcam.ps1`.
