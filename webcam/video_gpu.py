"""Native Media Foundation/D3D11/CUDA camera and preview, without CPU frames."""
import ctypes as ct
import os
from pathlib import Path
import threading
import time


class Overlay(ct.Structure):
    _fields_ = [("box",ct.c_float*4),("landmarks",ct.c_float*10),
                ("red",ct.c_float),("green",ct.c_float),("blue",ct.c_float),("label",ct.c_char*128)]


class OverlayState:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = []
        self.updated = 0.0

    def update(self, rows):
        with self.lock:
            self.rows, self.updated = rows, time.perf_counter()

    def read(self):
        with self.lock:
            return list(self.rows) if time.perf_counter()-self.updated < 0.25 else []


class GPUVideoDevice:
    def __init__(self, root, index, window, overlays=None, preview_only=False):
        import torch
        self.torch = torch
        self.owner = threading.get_ident()
        self.handle = None
        self.dll_dirs = []
        self.overlays = overlays or OverlayState()
        root = Path(root)
        cuda_root = os.environ.get("CUDA_PATH")
        local_cuda_root = root/"build/cuda_root.txt"
        if not cuda_root and local_cuda_root.is_file():
            cuda_root = local_cuda_root.read_text(encoding="utf-8-sig").strip()
        if os.name == "nt" and cuda_root:
            for suffix in ("bin", "bin/x64"):
                directory = Path(cuda_root)/suffix
                if directory.is_dir():
                    self.dll_dirs.append(os.add_dll_directory(str(directory)))
        dll_candidates = (root/"build/Release/facelivt_video.dll",
                          root/"build/facelivt_video.dll")
        dll = next((path for path in dll_candidates if path.is_file()), None)
        if dll is None:
            raise RuntimeError(f"Build facelivt_video before starting GPU video: {dll_candidates[0]}")
        self.lib = ct.CDLL(str(dll))
        self.lib.flvt_video_create.argtypes = [ct.c_int,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_size_t]
        self.lib.flvt_video_create.restype = ct.c_void_p
        for name in ("flvt_video_read","flvt_video_test_frame"):
            getattr(self.lib,name).argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t]
            getattr(self.lib,name).restype = ct.c_int
        self.lib.flvt_video_present.argtypes = [ct.c_void_p,ct.POINTER(Overlay),ct.c_int,ct.c_void_p,ct.c_size_t]
        self.lib.flvt_video_present.restype = ct.c_int
        self.lib.flvt_video_destroy.argtypes = [ct.c_void_p]
        self.lib.flvt_video_destroy.restype = None
        self.error = ct.create_string_buffer(2048)
        self.handle = self.lib.flvt_video_create(index,window,int(preview_only),self.error,len(self.error))
        if not self.handle:
            raise RuntimeError(self.error.value.decode("utf-8",errors="replace"))

    def _check(self,status):
        if status:
            raise RuntimeError(self.error.value.decode("utf-8",errors="replace"))

    def read(self):
        frame = self.torch.empty((480,640,4),dtype=self.torch.uint8,device="cuda:0")
        self._check(self.lib.flvt_video_read(self.handle,frame.data_ptr(),frame.numel(),self.error,len(self.error)))
        self.present()
        return True,frame

    def present(self):
        rows = self.overlays.read()[:8]
        native = (Overlay*len(rows))()
        for value,row in zip(native,rows):
            value.box[:] = row["box"]
            value.landmarks[:] = row["landmarks"].reshape(-1)
            value.red,value.green,value.blue = row["color"]
            encoded = row["label"].encode("utf-8")[:127]
            value.label = encoded.decode("utf-8",errors="ignore").encode("utf-8")
        self._check(self.lib.flvt_video_present(self.handle,native,len(rows),self.error,len(self.error)))

    def test_frame(self,frame):
        self.torch.cuda.current_stream().synchronize()
        self._check(self.lib.flvt_video_test_frame(self.handle,frame.data_ptr(),frame.numel(),self.error,len(self.error)))
        self.present()

    def release(self):
        if self.handle:
            if threading.get_ident() != self.owner:
                raise RuntimeError("Close GPU video on its owning capture thread")
            self.lib.flvt_video_destroy(self.handle)
            self.handle = None
        for directory in self.dll_dirs:
            directory.close()
        self.dll_dirs.clear()
