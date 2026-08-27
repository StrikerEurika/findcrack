import numpy as np
from findcrack.utils import sigmoid_np
import cv2


def _require_ort():
    try:
        import onnxruntime as ort
        return ort
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for ONNX inference. "
            "Install it with: uv add onnxruntime or uv add onnxruntime-gpu"
        ) from e


class ONNXCPUBackend:
    """ONNX Runtime CPU inference backend."""

    def __init__(self, model_path: str):
        ort = _require_ort()
        self.model_path = model_path
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self._detect_yolo_seg()
        self._detect_expected_size()
        self._detect_baked_normalization()

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
            from onnx import numpy_helper

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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        orig_size = (x.shape[2], x.shape[3])

        if self.expected_size[0] and self.expected_size[1]:
            target_size = self.expected_size
            if orig_size != target_size:
                x_hwc = np.transpose(x[0], (1, 2, 0))
                x_resized = cv2.resize(
                    x_hwc, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
                )
                if len(x_resized.shape) == 2:
                    x_resized = np.expand_dims(x_resized, axis=-1)
                x = np.expand_dims(np.transpose(x_resized, (2, 0, 1)), axis=0)

        if self.is_yolo_seg:
            outputs = self.session.run(self.output_names, {self.input_name: x.astype(self.input_dtype)})
            return self._postprocess_yolo_seg(outputs[0], outputs[1], orig_size)
        else:
            outputs = self.session.run([self.output_names[0]], {self.input_name: x.astype(self.input_dtype)})
            out_np = outputs[0]

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
        return "cpu"

    def is_gpu(self) -> bool:
        return False

    @property
    def backend_type(self) -> str:
        return "onnx_cpu"
