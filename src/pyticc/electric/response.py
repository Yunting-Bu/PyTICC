import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline

ELECTRIC_RESPONSE_COLUMNS = ("r", "mu_z", "alpha_xx", "alpha_zz", "beta_zzz", "beta_xxz")


# ----------------------------------------------------------------------------------------
def _readonly_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        message = f"{name} must be one-dimensional, but got shape {array.shape}"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(array)):
        message = f"{name} contains non-finite values"
        logger.error(message)
        raise ValueError(message)
    array.setflags(write=False)
    return array


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ElectricResponseValues:
    """
    Electric-response components evaluated at one or more bond lengths.

    Members:
        mu_z: NDArray[np.float64] - body-fixed dipole component along the
            molecular axis, shape (...)
        alpha_xx: NDArray[np.float64] - perpendicular dipole polarizability,
            shape (...)
        alpha_zz: NDArray[np.float64] - parallel dipole polarizability,
            shape (...)
        beta_zzz: NDArray[np.float64] - parallel first hyperpolarizability,
            shape (...)
        beta_xxz: NDArray[np.float64] - mixed first hyperpolarizability,
            shape (...)
    """

    mu_z: NDArray[np.float64]
    alpha_xx: NDArray[np.float64]
    alpha_zz: NDArray[np.float64]
    beta_zzz: NDArray[np.float64]
    beta_xxz: NDArray[np.float64]

    def __post_init__(self) -> None:
        arrays = tuple(np.array(getattr(self, name), dtype=np.float64, copy=True) for name in ELECTRIC_RESPONSE_COLUMNS[1:])
        if len({array.shape for array in arrays}) != 1:
            message = f"Electric-response components must have one common shape, but got {[array.shape for array in arrays]}"
            logger.error(message)
            raise ValueError(message)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            message = "Electric-response values contain non-finite entries"
            logger.error(message)
            raise ValueError(message)
        for name, array in zip(ELECTRIC_RESPONSE_COLUMNS[1:], arrays, strict=True):
            array.setflags(write=False)
            object.__setattr__(self, name, array)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ElectricResponseTable:
    """
    Tabulated diatomic electric-response properties in atomic units.

    Rows are sorted by bond length during initialization. Every response column
    is reordered with the same permutation, so tensor components remain paired
    with their original bond length.

    Members:
        r: NDArray[np.float64] - bond lengths in bohr, shape (n_r,)
        mu_z: NDArray[np.float64] - body-fixed dipole component, shape (n_r,)
        alpha_xx: NDArray[np.float64] - perpendicular polarizability,
            shape (n_r,)
        alpha_zz: NDArray[np.float64] - parallel polarizability, shape (n_r,)
        beta_zzz: NDArray[np.float64] - parallel first hyperpolarizability,
            shape (n_r,)
        beta_xxz: NDArray[np.float64] - mixed first hyperpolarizability,
            shape (n_r,)
    """

    r: NDArray[np.float64]
    mu_z: NDArray[np.float64]
    alpha_xx: NDArray[np.float64]
    alpha_zz: NDArray[np.float64]
    beta_zzz: NDArray[np.float64]
    beta_xxz: NDArray[np.float64]

    def __post_init__(self) -> None:
        arrays = {name: _readonly_vector(getattr(self, name), name) for name in ELECTRIC_RESPONSE_COLUMNS}
        sizes = {array.size for array in arrays.values()}
        if len(sizes) != 1:
            message = f"Electric-response columns must have one common length, but got {[array.size for array in arrays.values()]}"
            logger.error(message)
            raise ValueError(message)
        if arrays["r"].size < 2:
            message = "Electric-response CSV must contain at least two rows"
            logger.error(message)
            raise ValueError(message)

        order = np.argsort(arrays["r"], kind="stable")
        arrays = {name: array[order] for name, array in arrays.items()}
        if np.any(np.diff(arrays["r"]) == 0.0):
            message = "Electric-response bond lengths must not contain duplicates"
            logger.error(message)
            raise ValueError(message)
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def evaluate(self, r: ArrayLike) -> ElectricResponseValues:
        r"""
        Interpolate all electric-response components with natural cubic splines.

        Formula:
            For each response component f(r), construct a piecewise cubic
            polynomial S(r) satisfying

            S(r_i) = f(r_i),
            S'(r_i^-) = S'(r_i^+),
            S''(r_i^-) = S''(r_i^+),

            with the natural boundary conditions

            S''(r_0) = S''(r_{n_r-1}) = 0.

            Extrapolation outside [r_0, r_{n_r-1}] is not allowed.

        Inputs:
            r: ArrayLike - bond length or bond-length array in bohr, shape (...)

        Returns:
            response: ElectricResponseValues - interpolated response components
                in atomic units, each with shape (...)
        """
        points = np.asarray(r, dtype=np.float64)
        if not np.all(np.isfinite(points)):
            message = "Interpolation coordinates contain non-finite values"
            logger.error(message)
            raise ValueError(message)
        if np.any(points < self.r[0]) or np.any(points > self.r[-1]):
            message = f"Interpolation coordinates must lie within [{self.r[0]}, {self.r[-1]}] bohr"
            logger.error(message)
            raise ValueError(message)

        values = np.column_stack((self.mu_z, self.alpha_xx, self.alpha_zz, self.beta_zzz, self.beta_xxz))
        interpolated = CubicSpline(self.r, values, axis=0, bc_type="natural", extrapolate=False)(points)
        return ElectricResponseValues(
            mu_z=interpolated[..., 0],
            alpha_xx=interpolated[..., 1],
            alpha_zz=interpolated[..., 2],
            beta_zzz=interpolated[..., 3],
            beta_xxz=interpolated[..., 4],
        )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def load_electric_response_csv(path: str | Path) -> ElectricResponseTable:
    """
    Load a fixed-schema electric-response CSV file.

    The required header is
    ``r,mu_z,alpha_xx,alpha_zz,beta_zzz,beta_xxz``. All quantities use atomic
    units, and r is measured in bohr. Input rows may be unordered; the returned
    table is sorted by r.

    Inputs:
        path: str | Path - CSV file path

    Returns:
        table: ElectricResponseTable - validated and bond-length-sorted response
            table
    """
    csv_path = Path(path).expanduser()
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(name.strip() for name in reader.fieldnames or ())
        if columns != ELECTRIC_RESPONSE_COLUMNS:
            message = f"Electric-response CSV header must be {','.join(ELECTRIC_RESPONSE_COLUMNS)}, but got {','.join(columns)}"
            logger.error(message)
            raise ValueError(message)

        data: dict[str, list[float]] = {name: [] for name in ELECTRIC_RESPONSE_COLUMNS}
        for line_number, row in enumerate(reader, start=2):
            try:
                for name in ELECTRIC_RESPONSE_COLUMNS:
                    data[name].append(float(row[name]))
            except (TypeError, ValueError) as error:
                message = f"Invalid numeric value in {csv_path} at line {line_number}"
                logger.error(message)
                raise ValueError(message) from error

    return ElectricResponseTable(
        r=np.asarray(data["r"], dtype=np.float64),
        mu_z=np.asarray(data["mu_z"], dtype=np.float64),
        alpha_xx=np.asarray(data["alpha_xx"], dtype=np.float64),
        alpha_zz=np.asarray(data["alpha_zz"], dtype=np.float64),
        beta_zzz=np.asarray(data["beta_zzz"], dtype=np.float64),
        beta_xxz=np.asarray(data["beta_xxz"], dtype=np.float64),
    )


# ----------------------------------------------------------------------------------------
