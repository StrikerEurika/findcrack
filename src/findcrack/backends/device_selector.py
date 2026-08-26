from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DeviceInfo:
    """Information about the resource device."""
    name: str
    display_name: str
    available: bool
    index: int = 0
    
    def __str__(self) -> str:
        ststus = "Available" if self.available else "Unavailable"
        return f"[{ststus} {self.display_name}]"
    
    class DeviceSelectionError(Exeption):
        """Raised when a requested device is not available."""
        pass
    
    def list_available_devices() -> List[DeviceInfo]:
        """Query and return all available devices."""

        devices = []

        # CPU is always available
        devices.append(DeviceInfo(
            name="cpu",
            display_name="CPU",
            available=True,
            index=0
        ))
        
        # Check CUDA availability
        cuda_available = False
        cuda_name = "CUDA (GPU)"

        try:
            import 