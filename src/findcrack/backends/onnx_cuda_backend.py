import numpy as np
from findcrack.utils import sigmoid_np
import cv2

_DLL_DIRS_REGISTERED = False


def _register_cuda_dll_dirs():
    """Register CUDA runtime DLL dirs."""
    global _DLL_DIRS_REGISTERED
    if _DLL_DIRS_REGISTERED:
        return
    _DLL_DIRS_REGISTERED = True

    import os
    import sys
    from pathlib import Path

    candidate_dirs = []

    # 1. NVIDIA pip wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12)
    for site_pkg in sys.path:
        sp = Path(site_pkg)
        if not sp.is_dir():
            continue
        candidate_dirs.extend(sp.glob("nvidia" + os.sep + "*" + os.sep + "bin"))
        candidate_dirs.extend(sp.glob("nvidia" + os.sep + "*" + os.sep + "lib"))

    # 2. System CUDA toolkit

    for d in candidate_dirs:
        if not d.is_dir():
            continue
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        except OSError:
            pass


def _require_ort():
    try:
        import onnxruntime as ort

        # Load CUDA/cuDNN libraries from NVIDIA Python packages.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(
                directory=None,
                cuda=True,
                cudnn=True,
            )

        return ort
    except ImportError as error:
        raise ImportError(
            "onnxruntime is required for ONNX inference. "
            "Install it with: uv add onnxruntime-gpu"
        ) from error


class ONNXCUDABackend:
    """ONNX Runtime CUDA inference backend using IOBinding for zero-copy execution.

    Model weights stay resident on GPU. Only input/output data crosses the PCIe bus.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        ort = _require_ort()
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in the installed onnxruntime. "
                "Install onnxruntime-gpu to use the CUDA backend: "
                "pip install onnxruntime-gpu>=1.16.0"
            )

        providers = ["CUDAExecutionProvider"]
        if "CUDNNExecutionProvider" in available:
            providers.append("CUDNNExecutionProvider")
        providers.append("CPUExecutionProvider")

        _register_cuda_dll_dirs()

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except RuntimeError as e:
            if "CUDA Provider not available" in str(e) or "cublasLt64" in str(e):
                raise RuntimeError(
                    "onnxruntime-gpu is installed but its CUDA runtime dependencies are missing or mismatched.\n"
                    f"Detail: {e}\n"
                    "This usually means the CUDA runtime DLLs (cuBLAS/cuDNN) do not "
                    "match the onnxruntime-gpu build.\n"
                    "Options:\n"
                    "  1. Install onnxruntime-gpu>=1.20 together with torch built for "
                    "the same CUDA major version (e.g. cu124 wheels)\n"
                    "  2. Install the NVIDIA runtime pip wheels: nvidia-cublas-cu12, nvidia-cudnn-cu12\n"
                    "  3. Use CPU backend instead: device='cpu'"
                ) from e
            raise

        # ORT silently drops a broken CUDA EP to CPU; verify it actually loaded.
        active = set(self.session.get_providers())
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                "CUDAExecutionProvider failed to load (missing/mismatched CUDA "
                "runtime DLLs such as cuBLAS or cuDNN). Install a CUDA toolkit "
                "matching the onnxruntime-gpu build, or use device='cpu'."
            )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self._detect_yolo_seg()
        self._detect_expected_size()
        self._detect_baked_normalization()

        self._io_binding = self.session.io_binding()
        self._input_ortvalue = None
        self._output_ortvalues = {}

    def _detect_yolo_seg(self):
        self.is_yolo_seg = False
        if len(self.output_names) >= 2:
            out0_shape = self.session.get_outputs()[0].shape
            out1_shape = self.session.get_outputs()[1].shape
            if (
                len(out0_shape) == 3
                and out0_shape[2] == 38
                and len(out1_shape) == 4
                and out1_shape[1] == 32
            ):
                self.is_yolo_seg = True

    def _detect_expected_size(self):
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:
            h_dim, w_dim = 2, 3
            if isinstance(input_shape[3], int) and input_shape[3] in (1, 3):
                h_dim, w_dim = 1, 2
            h_expected = input_shape[h_dim]
            w_expected = input_shape[w_dim]
            self.expected_size = (
                h_expected if isinstance(h_expected, int) else None,
                w_expected if isinstance(w_expected, int) else None,
            )
        else:
            self.expected_size = (None, None)

    def _detect_baked_normalization(self):
        input_type = str(self.session.get_inputs()[0].type)
        if "uint8" in input_type:
            self.baked_normalization = True
            self.input_dtype = np.uint8
            return

        try:
            import onnx

            model = onnx.load(self.model_path)
            graph = model.graph
            input_name = graph.input[0].name
            initializer_names = {init.name for init in graph.initializer}

            for node in list(graph.node)[:30]:
                if input_name in node.input:
                    if (
                        node.op_type in ("Sub", "Div", "Mul", "Add")
                        and len(node.input) >= 2
                        and node.input[1] in initializer_names
                    ):
                        self.baked_normalization = True
                        self.input_dtype = np.float32
                        return
        except Exception:
            pass

        self.baked_normalization = False
        self.input_dtype = np.float32

    def _postprocess_yolo_seg(self, output0, output1, target_shape):
        h_img, w_img = target_shape
        combined_mask = np.zeros((h_img, w_img), dtype=np.float32)

        boxes = output0[0, :, 0:4]
        scores = output0[0, :, 4]
        coeffs = output0[0, :, 6:38]
        protos = output1[0]

        keep = scores > 0.25
        if not np.any(keep):
            return np.expand_dims(
                np.expand_dims(np.full((h_img, w_img), -10.0, dtype=np.float32), axis=0),
                axis=0,
            )

        valid_boxes = boxes[keep]
        valid_coeffs = coeffs[keep]
        N = valid_boxes.shape[0]
        _, H_proto, W_proto = protos.shape

        masks_in = np.matmul(valid_coeffs, protos.reshape(32, -1)).reshape(N, H_proto, W_proto)

        for i in range(N):
            box = valid_boxes[i]
            mask = sigmoid_np(masks_in[i])
            mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

            in_h = self.expected_size[0] or 512.0
            in_w = self.expected_size[1] or 512.0

            x1, y1, x2, y2 = box
            x1 = int(round(x1 * w_img / in_w))
            y1 = int(round(y1 * h_img / in_h))
            x2 = int(round(x2 * w_img / in_w))
            y2 = int(round(y2 * h_img / in_h))

            x1 = max(0, min(x1, w_img))
            y1 = max(0, min(y1, h_img))
            x2 = max(0, min(x2, w_img))
            y2 = max(0, min(y2, h_img))

            crop_mask = np.zeros_like(mask_resized)
            crop_mask[y1:y2, x1:x2] = 1.0
            mask_resized = mask_resized * crop_mask

            combined_mask = np.maximum(combined_mask, mask_resized)

        p = np.clip(combined_mask, 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p))
        return np.expand_dims(np.expand_dims(logits, axis=0), axis=0)


    def __call__(self, x) -> np.ndarray:
        ort = _require_ort()
        x_np = np.asarray(x, dtype=self.input_dtype)
        orig_size = (x_np.shape[2], x_np.shape[3])

        if self.expected_size[0] and self.expected_size[1]:
            target_size = self.expected_size
            if orig_size != target_size:
                x_hwc = np.transpose(x_np[0], (1, 2, 0))
                x_resized = cv2.resize(
                    x_hwc, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
                )
                if len(x_resized.shape) == 2:
                    x_resized = np.expand_dims(x_resized, axis=-1)
                x_np = np.expand_dims(np.transpose(x_resized, (2, 0, 1)), axis=0)

        input_ortvalue = ort.OrtValue.ortvalue_from_numpy(x_np, device_type="cuda")

        io_binding = self._io_binding
        io_binding.bind_ortvalue_input(self.input_name, input_ortvalue)

        for out_name in self.output_names:
            io_binding.bind_output(out_name, device_type="cuda")

        self.session.run_with_iobinding(io_binding)

        if self.is_yolo_seg:
            output0 = io_binding.get_outputs()[0].numpy()
            output1 = io_binding.get_outputs()[1].numpy()
            return self._postprocess_yolo_seg(output0, output1, orig_size)
        else:
            out_np = io_binding.get_outputs()[0].numpy()

            if out_np.shape[2:] != orig_size:
                out_hwc = np.transpose(out_np[0], (1, 2, 0))
                out_resized = cv2.resize(
                    out_hwc, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_LINEAR
                )
                if len(out_resized.shape) == 2:
                    out_resized = np.expand_dims(out_resized, axis=-1)
                out_np = np.expand_dims(np.transpose(out_resized, (2, 0, 1)), axis=0)

            return out_np

    @property
    def device(self) -> str:
        return "cuda"

    def is_gpu(self) -> bool:
        return True

    @property
    def backend_type(self) -> str:
        return "onnx_cuda"
