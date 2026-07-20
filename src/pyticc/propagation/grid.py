from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import ceil

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

_MAX_SECTORS_PER_WINDOW = 64
_MEBIBYTE = 1024**2


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
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
        """Return the sector midpoint."""
        return 0.5 * (self.radial_start + self.radial_end)

    @property
    def radial_half_step(self) -> float:
        """Return the distance from either endpoint to the midpoint."""
        return 0.5 * (self.radial_end - self.radial_start)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_radial_sectors(radial_boundaries: ArrayLike, radial_half_steps: ArrayLike) -> tuple[RadialSector, ...]:
    """
    Build propagation sectors over any number of radial intervals.

    Each user interval is divided into sectors of nominal width ``2 * half_step``.
    The final sector is shortened when needed to end exactly at its breakpoint.

    Inputs:
        radial_boundaries: ArrayLike - increasing interval boundaries with shape
            (n_interval + 1,) in atomic units
        radial_half_steps: ArrayLike - nominal a-to-c half-step for each interval,
            shape (n_interval,)

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


# ----------------------------------------------------------------------------------------
def iter_radial_windows(
    sectors: Sequence[RadialSector],
    *,
    n_grid: int,
    n_channel: int,
    n_energy: int,
    memory_limit_mb: float,
    state_matrix_elements: int | None = None,
) -> Iterator[tuple[tuple[RadialSector, ...], NDArray[np.float64]]]:
    """
    Yield memory-bounded radial windows and their unique propagation points.

    For ``n_sector`` consecutive sectors, each radial-point array contains the
    ordered start, midpoint, and endpoint values with shape
    ``(2 * n_sector + 1,)``. Neighboring windows share their boundary point.

    sector 0: [r0 —— mid0 —— r1]
    sector 1:                 [r1 —— mid1 —— r2]

    points = [r0, mid0, r1, mid1, r2]
        index: 0    1    2    3    4      length = 2*2+1 = 5

    Inputs:
        sectors: Sequence[RadialSector] - contiguous propagation sectors
        n_grid: int - number of internal PES grid points per R
        n_channel: int - largest channel count propagated at once
        n_energy: int - number of total energies propagated together
        memory_limit_mb: float - target transient-memory limit in MiB
        state_matrix_elements: int | None - sum of n_channel_block**2 for all
            simultaneously resident propagation states

    Yields:
        window: tuple[RadialSector, ...] - consecutive sectors in one window
        radial_points: NDArray[np.float64] - start, midpoint, and endpoint grid,
            shape (2 * n_window_sector + 1,)
    """
    n_sector = len(sectors)
    if n_sector < 1 or n_grid < 0 or n_channel < 1 or n_energy < 1:
        message = (
            "Window dimensions must satisfy n_sector >= 1, n_grid >= 0, "
            f"n_channel >= 1, and n_energy >= 1, but got {(n_sector, n_grid, n_channel, n_energy)}"
        )
        logger.error(message)
        raise ValueError(message)

    largest_matrix = n_channel**2
    resident_elements = largest_matrix if state_matrix_elements is None else state_matrix_elements
    budget = int(memory_limit_mb * _MEBIBYTE)
    resident_bytes = 32 * n_energy * resident_elements
    initial_point_bytes = 8 * (n_grid + 3 * largest_matrix)
    sector_bytes = 16 * n_grid + 128 * largest_matrix
    available = max(0, budget - resident_bytes - initial_point_bytes)
    estimated = max(1, available // max(1, sector_bytes))
    window_size = min(n_sector, _MAX_SECTORS_PER_WINDOW, estimated)
    logger.debug(f"Using {ceil(n_sector / window_size)} radial windows of at most {window_size} sectors")

    for window_start in range(0, n_sector, window_size):
        window = tuple(sectors[window_start : window_start + window_size])
        radial_points = np.empty(2 * len(window) + 1, dtype=np.float64)
        radial_points[0] = window[0].radial_start
        for index, sector in enumerate(window):
            radial_points[2 * index + 1] = sector.radial_mid
            radial_points[2 * index + 2] = sector.radial_end
        yield window, radial_points


# ----------------------------------------------------------------------------------------
