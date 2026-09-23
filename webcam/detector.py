"""SCRFD-10G decoding and five-landmark similarity alignment.

Preprocessing and anchor conventions follow the upstream InsightFace SCRFD
implementation (see THIRD_PARTY.md). Network execution, letterboxing, and head
decoding use CUDA. Only compact candidates are returned to the host for NMS.
"""
from dataclasses import dataclass
import cv2
import numpy as np

TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


@dataclass
class Face:
    box: np.ndarray
    landmarks: np.ndarray
    confidence: float


def similarity_transform(source, target=TEMPLATE):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        raise ValueError("Expected five finite landmarks")
    src_mean, dst_mean = source.mean(0), target.mean(0)
    src, dst = source - src_mean, target - dst_mean
    variance = np.sum(src * src) / len(source)
    if variance < 1e-8:
        raise ValueError("Degenerate landmarks")
    u, singular, vt = np.linalg.svd(dst.T @ src / len(source))
    sign = np.array([1.0, 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0])
    rotation = (u * sign) @ vt
    scale = float(singular @ sign) / variance
    linear = scale * rotation
    return np.column_stack((linear, dst_mean - linear @ src_mean)).astype(np.float32)


def align_face(frame, face):
    matrix = similarity_transform(face.landmarks)
    return cv2.warpAffine(frame, matrix, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


def align_faces_cuda(frame, faces):
    """Keep full frames and aligned crops in CUDA memory (input is BGR/BGRA)."""
    import torch
    if not frame.is_cuda:
        raise ValueError("GPU alignment requires a CUDA video frame")
    height, width = frame.shape[:2]
    matrices = []
    for face in faces:
        transform = similarity_transform(face.landmarks)
        matrices.append(cv2.invertAffineTransform(transform))
    inverse = torch.as_tensor(np.stack(matrices), dtype=torch.float32, device=frame.device)
    y, x = torch.meshgrid(torch.arange(112,device=frame.device), torch.arange(112,device=frame.device), indexing="ij")
    coordinates = torch.stack((x,y,torch.ones_like(x)),dim=-1).float()
    source = torch.einsum("bij,hwj->bhwi",inverse,coordinates)
    # OpenCV INTER_LINEAR uses 1/32-pixel interpolation coordinates.
    source = (source*32).round()/32
    scale = torch.tensor([width,height],device=frame.device)
    grid = 2*(source+0.5)/scale-1
    image = frame[...,:3].permute(2,0,1).unsqueeze(0).float().expand(len(faces),-1,-1,-1)
    crop = torch.nn.functional.grid_sample(image,grid,mode="bilinear",padding_mode="zeros",align_corners=False)
    return crop.round().clamp_(0,255).to(torch.uint8).permute(0,2,3,1).contiguous()


def nms(boxes, scores, threshold=0.4):
    order = np.argsort(-scores, kind="stable")
    keep = []
    area = np.prod(np.maximum(0, boxes[:, 2:4] - boxes[:, :2] + 1), axis=1)
    while len(order):
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        lo = np.maximum(boxes[i, :2], boxes[rest, :2])
        hi = np.minimum(boxes[i, 2:4], boxes[rest, 2:4])
        intersection = np.prod(np.maximum(0, hi - lo + 1), axis=1)
        overlap = intersection / np.maximum(area[i] + area[rest] - intersection, 1e-8)
        order = rest[overlap <= threshold]
    return keep


class SCRFD:
    def __init__(self, model, size=640, threshold=0.65, profile_prefix=None):
        # Importing PyTorch loads its existing CUDA 12/cuDNN 9 DLLs on Windows.
        # Neither a second CUDA installation nor a CPU detector is needed.
        import torch
        import onnxruntime as ort
        if not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("SCRFD requires CUDA and onnxruntime-gpu; CPU fallback is disabled")
        if size <= 0 or size % 32:
            raise ValueError("SCRFD input size must be a positive multiple of 32")
        self.torch = torch
        self.size, self.threshold = size, threshold
        self.stream = torch.cuda.Stream(device=0)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_free_dimension_override_by_name("?", size)
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if profile_prefix is not None:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=[
            ("CUDAExecutionProvider", {"device_id": 0,
                                       "user_compute_stream": str(self.stream.cuda_stream),
                                       "enable_cuda_graph": "1",
                                       "use_tf32": "1",
                                       "cudnn_conv_algo_search": "HEURISTIC",
                                       "gpu_mem_limit": str(1024 * 1024 * 1024)})])
        self.session.disable_fallback()
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("SCRFD CUDA provider failed to initialize; refusing CPU fallback")
        self.input_name = self.session.get_inputs()[0].name
        self.binding = self.session.io_binding()
        self.outputs, self.anchors = {}, {}
        with torch.cuda.stream(self.stream):
            self.input = torch.empty((1, 3, size, size), dtype=torch.float32, device="cuda:0")
            self.binding.bind_input(self.input_name, "cuda", 0, np.float32,
                                    tuple(self.input.shape), self.input.data_ptr())
            for output in self.session.get_outputs():
                shape = tuple(output.shape)
                if not all(isinstance(dim, int) for dim in shape):
                    raise RuntimeError(f"SCRFD output is not fixed-size: {output.name} {shape}")
                tensor = torch.empty(shape, dtype=torch.float32, device="cuda:0")
                self.outputs[shape] = tensor
                self.binding.bind_output(output.name, "cuda", 0, np.float32, shape, tensor.data_ptr())
            for stride in (8, 16, 32):
                grid = np.stack(np.mgrid[:size // stride, :size // stride][::-1], axis=-1)
                centers = np.repeat(grid.reshape(-1, 2) * stride, 2, axis=0).astype(np.float32)
                self.anchors[stride] = torch.as_tensor(centers, device="cuda:0")
        # Capture the fixed-shape network graph before opening the camera.
        self.detect(np.zeros((480, 640, 3), np.uint8))

    def detect(self, frame):
        height, width = frame.shape[:2]
        ratio = min(self.size / width, self.size / height)
        nw, nh = max(1, int(width * ratio)), max(1, int(height * ratio))
        torch = self.torch
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            # Upload camera bytes once, rather than a 640x640 float input blob.
            if isinstance(frame, torch.Tensor):
                if not frame.is_cuda or frame.device.index != 0 or frame.dtype != torch.uint8:
                    raise ValueError("GPU camera frames must be uint8 on CUDA GPU 0")
                uploaded = frame
            else:
                uploaded = torch.as_tensor(np.ascontiguousarray(frame), device="cuda:0")
            rgb = uploaded[..., :3].permute(2, 0, 1).flip(0).unsqueeze(0).float()
            resized = torch.nn.functional.interpolate(rgb, (nh, nw), mode="bilinear", align_corners=False)
            # Match the previous uint8 resize contract before normalization.
            resized.round_().sub_(127.5).mul_(1 / 128.0)
            self.input.fill_(-127.5 / 128.0)
            self.input[:, :, :nh, :nw].copy_(resized)
            self.session.run_with_iobinding(self.binding)
            candidates = []
            for stride, centers in self.anchors.items():
                count = len(centers)
                scores = self.outputs[(count, 1)]
                distances = self.outputs[(count, 4)] * stride
                points = centers[:, None, :] + self.outputs[(count, 10)].reshape(-1, 5, 2) * stride
                boxes = torch.cat((centers - distances[:, :2], centers + distances[:, 2:]), dim=1)
                candidates.append(torch.cat((boxes, scores, points.flatten(1)), dim=1))
            candidates = torch.cat(candidates)
            selected = candidates[candidates[:, 4] >= self.threshold]
            packed = selected.cpu().numpy()
        if not len(packed):
            return []
        boxes, scores, landmarks = packed[:, :4], packed[:, 4], packed[:, 5:].reshape(-1, 5, 2)
        scale = np.array([nw / width, nh / height], dtype=np.float32)
        boxes /= np.tile(scale, 2)
        landmarks /= scale
        keep = nms(boxes, scores)
        faces = [Face(boxes[i], landmarks[i], float(scores[i])) for i in keep
                 if np.isfinite(boxes[i]).all() and np.isfinite(landmarks[i]).all()]
        return sorted(faces, key=lambda face: float(np.prod(face.box[2:] - face.box[:2])), reverse=True)
