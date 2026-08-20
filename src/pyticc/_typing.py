from typing import Protocol


class JaxDevice(Protocol):
    """Structural type shared by JAX CPU and accelerator devices."""

    @property
    def platform(self) -> str:
        """Return the device platform name."""
        ...

    @property
    def id(self) -> int:
        """Return the platform-local device index."""
        ...
