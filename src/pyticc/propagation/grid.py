from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RadialSector:
    """
    One Manolopoulos log-derivative propagation sector.

    Members:
        radial_start: float - left endpoint in atomic units
        radial_end: float - right endpoint in atomic units
    """

    radial_start: float
    radial_end: float

    @property
    def radial_mid(self) -> float:
        return 0.5 * (self.radial_start + self.radial_end)

    @property
    def radial_half_step(self) -> float:
        return 0.5 * (self.radial_end - self.radial_start)


# ----------------------------------------------------------------------------------------
def build_radial_sectors(radial_boundaries: ArrayLike, radial_half_steps: ArrayLike) -> tuple[RadialSector, ...]:
    """
    Build propagation sectors over any number of radial intervals.

    Each user interval is divided into sectors of nominal width ``2 * half_step``.
    The final sector is shortened when needed to end exactly at its breakpoint.

    Inputs:
        radial_boundaries: ArrayLike - increasing interval boundaries in atomic units
        radial_half_steps: ArrayLike - nominal a-to-c half-step for each interval

    Returns:
        sectors: tuple[RadialSector, ...] - contiguous propagation sectors
    """
    boundaries = np.asarray(radial_boundaries, dtype=np.float64)
    half_step_values = np.asarray(radial_half_steps, dtype=np.float64)

    if boundaries.ndim != 1 or half_step_values.ndim != 1 or boundaries.size != half_step_values.size + 1:
        message = (
            "radial_boundaries and radial_half_steps must be one-dimensional with sizes N+1 and N, "
            f"but got {boundaries.shape} and {half_step_values.shape}"
        )
        logger.error(message)
        raise ValueError(message)
    if half_step_values.size == 0:
        message = "At least one radial interval is required"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.diff(boundaries) > 0.0):
        message = "radial_boundaries must be strictly increasing"
        logger.error(message)
        raise ValueError(message)
    if not np.all(half_step_values > 0.0):
        message = "radial_half_steps must be positive"
        logger.error(message)
        raise ValueError(message)

    sectors: list[RadialSector] = []
    for interval_start, interval_end, radial_half_step in zip(boundaries[:-1], boundaries[1:], half_step_values, strict=True):
        interval_start = float(interval_start)
        interval_end = float(interval_end)
        sector_width = 2.0 * float(radial_half_step)
        sector_count = max(1, int(np.ceil((interval_end - interval_start) / sector_width - 1.0e-12)))
        for index in range(sector_count):
            radial_start = interval_start + index * sector_width
            radial_end = interval_end if index == sector_count - 1 else interval_start + (index + 1) * sector_width
            sectors.append(RadialSector(radial_start=radial_start, radial_end=radial_end))

    return tuple(sectors)


# ----------------------------------------------------------------------------------------
