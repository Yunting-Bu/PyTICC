from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger

from pyticc.propagation.device import normalize_device_spec

PropagationMode = Literal["inelastic", "capture"]


def _validate_print_verbose(value: object) -> None:
    if not isinstance(value, bool):
        message = f"Propagation print_verbose must be a boolean, but got {value!r}"
        logger.error(message)
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class Propagation:
    """Radial propagation grid and runtime settings.

    Members:
        boundaries: tuple[float, ...] - increasing radial interval boundaries
        half_steps: tuple[float, ...] - nominal LDMD half-step in each interval
        mode: PropagationMode - inner-boundary condition
        memory_mb: float - target transient-memory limit in MiB
        device: str - auto, CPU, or GPU propagation-device request
        print_verbose: bool - whether to emit INFO-level propagation progress
    """

    boundaries: tuple[float, ...]
    half_steps: tuple[float, ...]
    mode: PropagationMode = "inelastic"
    memory_mb: float = 512.0
    device: str = "auto"
    print_verbose: bool = False

    def __post_init__(self) -> None:
        boundaries = tuple(float(value) for value in self.boundaries)
        half_steps = tuple(float(value) for value in self.half_steps)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "half_steps", half_steps)
        object.__setattr__(self, "device", normalize_device_spec(self.device))

        if len(boundaries) != len(half_steps) + 1 or not half_steps:
            message = f"Propagation boundaries and half_steps must have sizes N+1 and N, but got {len(boundaries)} and {len(half_steps)}"
            logger.error(message)
            raise ValueError(message)
        if not np.all(np.isfinite(boundaries)) or boundaries[0] <= 0.0 or not np.all(np.diff(boundaries) > 0.0):
            message = "Propagation boundaries must be finite, positive, and strictly increasing"
            logger.error(message)
            raise ValueError(message)
        if not np.all(np.isfinite(half_steps)) or not np.all(np.asarray(half_steps) > 0.0):
            message = "Propagation half_steps must be finite and positive"
            logger.error(message)
            raise ValueError(message)
        if self.mode not in ("inelastic", "capture"):
            message = f"Propagation mode must be 'inelastic' or 'capture', but got {self.mode!r}"
            logger.error(message)
            raise ValueError(message)
        if not np.isfinite(self.memory_mb) or self.memory_mb <= 0.0:
            message = f"Propagation memory_mb must be positive, but got {self.memory_mb}"
            logger.error(message)
            raise ValueError(message)
        _validate_print_verbose(self.print_verbose)

    @property
    def Rmatch(self) -> float:
        """Return the asymptotic matching radius."""
        return self.boundaries[-1]
