import torch


def _normalize_device(device: str) -> str:
    d = str(device).lower().strip()
    if d in ("gpu", "cuda"):
        return "cuda"
    if d == "cpu":
        return "cpu"
    if d in ("mps", "apple"):
        return "mps"
    return d


class PyTorchBackend:
    """Native PyTorch inference backend. Optimal for GPU execution."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model.to(device).eval()
        self._device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)

    @property
    def device(self) -> str:
        return self._device

    def is_gpu(self) -> bool:
        return _normalize_device(self._device) == "cuda"

    @property
    def backend_type(self) -> str:
        return "pytorch"

    @property
    def baked_normalization(self) -> bool:
        return False
