from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class Backend(ABC):
    """Abstract interface for inference backends."""

    @abstractmethod
    def __call__(self, x: Union[Any, "torch.Tensor"]) -> Union[Any, "torch.Tensor"]:
        """Run inference, preserving input device type where possible."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        pass

    @abstractmethod
    def is_gpu(self) -> bool:
        pass

    @property
    def backend_type(self) -> str:
        return "unknown"

    @property
    def baked_normalization(self) -> bool:
        return False
