"""NumPy/ctypes bridge; recognition runs entirely inside the native CUDA DLL."""
import ctypes as ct
import os
from pathlib import Path
import threading

import numpy as np


class NativeRuntime:
    def __init__(self, root, max_batch=8):
        root = Path(root)
        self.owner = threading.get_ident()
        self.handle = None
        self.dll_dirs = []
        cuda_root = os.environ.get("CUDA_PATH")
        local_cuda_root = root / "build/cuda_root.txt"
        if not cuda_root and local_cuda_root.is_file():
            cuda_root = local_cuda_root.read_text(encoding="utf-8-sig").strip()
        if os.name == "nt" and cuda_root:
            for suffix in ("bin", "bin/x64"):
                directory = Path(cuda_root) / suffix
                if directory.is_dir():
                    self.dll_dirs.append(os.add_dll_directory(str(directory)))
        dll_candidates = (root / "build/Release/facelivt_native.dll",
                          root / "build/facelivt_native.dll")
        dll = next((path for path in dll_candidates if path.is_file()), None)
        if dll is None:
            raise FileNotFoundError(f"Build the facelivt_native target first: {dll_candidates[0]}")
        self.lib = ct.CDLL(str(dll))
        self.lib.flvt_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_size_t]
        self.lib.flvt_create.restype = ct.c_void_p
        self.lib.flvt_infer.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int,
                                       ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
        self.lib.flvt_infer.restype = ct.c_int
        self.lib.flvt_infer_device.argtypes = self.lib.flvt_infer.argtypes
        self.lib.flvt_infer_device.restype = ct.c_int
        self.lib.flvt_destroy.argtypes = [ct.c_void_p]
        self.lib.flvt_destroy.restype = None
        self.error = ct.create_string_buffer(2048)
        self.max_batch = max_batch
        self.handle = self.lib.flvt_create(
            str(root / "facelivtv2-l.fp16.flvt").encode("utf-8"),
            str(root / "generated/kernels/kernel_manifest.tsv").encode("utf-8"),
            max_batch, self.error, len(self.error))
        if not self.handle:
            raise RuntimeError(self.error.value.decode("utf-8", errors="replace"))

    def infer(self, crops):
        if not self.handle or threading.get_ident() != self.owner:
            raise RuntimeError("Use an open native runtime on its creating thread")
        if getattr(crops, "is_cuda", False):
            return self._infer_device(crops)
        batch = np.asarray(crops)
        if batch.dtype != np.uint8 or batch.ndim != 4 or batch.shape[1:] != (112, 112, 3):
            raise ValueError("Input must be uint8 Bx112x112x3 BGR")
        if not 1 <= len(batch) <= self.max_batch:
            raise ValueError("Batch exceeds the configured runtime size")
        batch = np.ascontiguousarray(batch)
        out = np.empty((len(batch), 512), dtype=np.float32)
        status = self.lib.flvt_infer(self.handle, batch.ctypes.data, batch.nbytes,
                                    len(batch), out.ctypes.data, out.size,
                                    self.error, len(self.error))
        if status:
            raise RuntimeError(self.error.value.decode("utf-8", errors="replace"))
        if not np.isfinite(out).all() or not np.allclose(np.linalg.norm(out, axis=1), 1, atol=2e-3):
            raise RuntimeError("Runtime returned invalid or unnormalized embeddings")
        return out

    def _infer_device(self, crops):
        import torch
        if crops.device.index != 0 or crops.dtype != torch.uint8 or crops.ndim != 4 or tuple(crops.shape[1:]) != (112,112,3):
            raise ValueError("Device input must be uint8 Bx112x112x3 BGR on CUDA GPU 0")
        if not 1 <= len(crops) <= self.max_batch:
            raise ValueError("Batch exceeds the configured runtime size")
        crops = crops.contiguous()
        out = torch.empty((len(crops),512), dtype=torch.float32, device=crops.device)
        # The native runtime owns its own stream. Complete the crop producer
        # before handing its device pointer to that stream.
        torch.cuda.current_stream(crops.device).synchronize()
        status = self.lib.flvt_infer_device(self.handle, crops.data_ptr(), crops.numel(),
                                           len(crops), out.data_ptr(), out.numel(),
                                           self.error, len(self.error))
        if status:
            raise RuntimeError(self.error.value.decode("utf-8", errors="replace"))
        result = out.cpu().numpy()  # Only 512 floats/person for attendance decisions.
        if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result,axis=1),1,atol=2e-3):
            raise RuntimeError("Runtime returned invalid or unnormalized embeddings")
        return result

    def close(self):
        if self.handle:
            if threading.get_ident() != self.owner:
                raise RuntimeError("Close the runtime on its creating thread")
            self.lib.flvt_destroy(self.handle)
            self.handle = None
        for directory in self.dll_dirs:
            directory.close()
        self.dll_dirs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
