from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Abstract interface for inference backends."""

    @abstractmethod
    def __call__(self, x: Any) -> Any:
        """Run inference."""
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
