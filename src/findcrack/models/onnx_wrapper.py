import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class ONNXModelWrapper(nn.Module):
        """
        Wraps an ONNX Runtime InferenceSession inside a PyTorch nn.Module.
        This allows running inference on ONNX models using the exact same code
        and APIs as PyTorch models, supporting both CPU/GPU tensor operations
        and test-time augmentation (TTA) pipelines.
        """
        def __init__(self, model_path: str, device: str = "cpu"):
            super().__init__()
            import onnxruntime as ort
            
            # Select execution providers based on the target device
            if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 1. Convert PyTorch tensor to NumPy array (typically expected as float32)
            x_np = x.detach().cpu().numpy().astype(np.float32)
            
            # 2. Run inference using ONNX Runtime
            outputs = self.session.run([self.output_name], {self.input_name: x_np})
            
            # 3. Convert prediction back to PyTorch tensor and move to the original device
            out_tensor = torch.from_numpy(outputs[0]).to(x.device)
            return out_tensor
else:
    class ONNXModelWrapper:
        """
        Wraps an ONNX Runtime InferenceSession for pure NumPy inference.
        """
        def __init__(self, model_path: str, device: str = "cpu"):
            import onnxruntime as ort
            
            # Select execution providers based on the target device
            if device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
                
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

        def __call__(self, x: np.ndarray) -> np.ndarray:
            # Convert prediction back to NumPy array
            outputs = self.session.run([self.output_name], {self.input_name: x.astype(np.float32)})
            return outputs[0]

