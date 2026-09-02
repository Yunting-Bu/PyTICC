from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger

from pyticc.propagation.device import normalize_device_spec

PropagationMode = Literal["inelastic", "capture"]


# ----------------------------------------------------------------------------------------
def _validate_print_verbose(value: object) -> None:
    """Require the propagation-progress switch to be boolean."""
    if not isinstance(value, bool):
        message = f"Propagation print_verbose must be a boolean, but got {value!r}"
        logger.error(message)
        raise ValueError(message)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Propagation:
    """Radial propagation runtime settings.

    Members:
        mode: PropagationMode - inner-boundary condition
        memory_mb: float - target transient-memory limit in MiB
        device: str - auto, CPU, or GPU propagation-device request
        print_verbose: bool - whether to emit INFO-level propagation progress
    """

    mode: PropagationMode = "inelastic"
    memory_mb: float = 512.0
    device: str = "auto"
    print_verbose: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", normalize_device_spec(self.device))

        if self.mode not in ("inelastic", "capture"):
            message = f"Propagation mode must be 'inelastic' or 'capture', but got {self.mode!r}"
            logger.error(message)
            raise ValueError(message)
        if not np.isfinite(self.memory_mb) or self.memory_mb <= 0.0:
            message = f"Propagation memory_mb must be positive, but got {self.memory_mb}"
            logger.error(message)
            raise ValueError(message)
        _validate_print_verbose(self.print_verbose)


# ----------------------------------------------------------------------------------------
