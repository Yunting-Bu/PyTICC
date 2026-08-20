import re
from dataclasses import dataclass
from typing import Protocol

import jax
from loguru import logger

_DEVICE_PATTERN = re.compile(r"(?:auto|cpu(?::\d+)?|gpu(?::\d+)?)")


class JaxDevice(Protocol):
    """Structural type required from a concrete JAX execution device."""

    @property
    def platform(self) -> str:
        """Return the execution platform name."""
        ...

    @property
    def id(self) -> int:
        """Return the device index within its platform."""
        ...


@dataclass(frozen=True, slots=True)
class ResolvedDevice:
    """One concrete JAX device selected for radial propagation."""

    requested: str
    device: JaxDevice

    @property
    def label(self) -> str:
        """Return the normalized platform and device index."""
        return f"{self.device.platform}:{self.device.id}"


def normalize_device_spec(value: object) -> str:
    """Normalize and validate an auto, CPU, or GPU propagation-device request."""
    if not isinstance(value, str):
        message = f"Propagation device must be 'auto', 'cpu', 'gpu', or '<platform>:<index>', but got {value!r}"
        logger.error(message)
        raise ValueError(message)

    normalized = value.strip().lower()
    if _DEVICE_PATTERN.fullmatch(normalized) is None:
        message = f"Propagation device must be 'auto', 'cpu', 'gpu', or '<platform>:<index>', but got {value!r}"
        logger.error(message)
        raise ValueError(message)
    return normalized


def _platform_devices(platform: str) -> tuple[JaxDevice, ...]:
    """Return initialized devices for one optional JAX platform."""
    try:
        return tuple(jax.devices(platform))
    except RuntimeError:
        return ()


def resolve_device(value: str) -> ResolvedDevice:
    """Resolve a propagation-device request without silently changing an explicit platform."""
    requested = normalize_device_spec(value)
    if requested == "auto":
        devices = _platform_devices("gpu") or _platform_devices("cpu")
        if not devices:
            message = "No JAX CPU or GPU device is available for propagation"
            logger.error(message)
            raise RuntimeError(message)
        return ResolvedDevice(requested=requested, device=devices[0])

    platform, separator, index_text = requested.partition(":")
    index = int(index_text) if separator else 0
    devices = _platform_devices(platform)
    if index >= len(devices):
        available = ", ".join(f"{device.platform}:{device.id}" for device in devices) or "none"
        message = f"Propagation requested {requested!r}, but available {platform} devices are: {available}"
        logger.error(message)
        raise RuntimeError(message)
    return ResolvedDevice(requested=requested, device=devices[index])
