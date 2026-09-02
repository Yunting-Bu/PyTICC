from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.linalg import eigh

from pyticc.basis.dvr import SineDVR, build_SineDVR, phase_fix
from pyticc.basis.podvr import VibPODVR, build_VibPODVR
from pyticc.matrix.triatom import TriatomPES, get_hmat_triatom_unsym, prepare_triatom_hamiltonian
from pyticc.system import MolInnerState, MonomerType


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TriatomBlock:
    """
    Contracted eigenstates for one canonical triatomic (j, K) block.

    Members:
        j: int - triatomic rotational angular momentum
        K: int - canonical helicity, either 0 or 1
        qn: NDArray[np.int64] - primitive quantum numbers (j1, omega, v1, v2),
            shape (n_primitive, 4)
        coefficients: NDArray[np.float64] - contraction coefficients indexed as
            [chi, local_t], shape (n_primitive, n_state)
        t_indices: NDArray[np.int64] - global t index of every coefficient column,
            shape (n_state,)
    """

    j: int
    K: int
    qn: NDArray[np.int64]
    coefficients: NDArray[np.float64]
    t_indices: NDArray[np.int64]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TriatomBasis:
    """
    Contracted asymptotic eigenstates of a triatomic monomer.

    ``t`` is zero-based in Python. Non-finite entries in ``Eint`` represent states
    absent from the retained basis. Positive K blocks reuse the K=1 eigenvectors.

    Members:
        Eint: NDArray[np.float64] - internal energies indexed as Eint[j, t], shape
            (jmax + 1, tmax + 1)
        jmax: int - maximum triatomic rotational angular momentum
        tmax: int - maximum contracted-state index
        tmin: int - minimum contracted-state index
        exchange_parity: int - A2B exchange parity, or 0 for an ABC monomer
        parity_block_sign: int - epsilon*(-1)^J used to construct the K=0 basis
        K0_available: NDArray[np.bool_] | None - availability of each (j, t, K=0)
            state, shape (jmax + 1, tmax + 1)
        positive_K_available: NDArray[np.bool_] | None - availability of each
            positive-K state, shape (jmax + 1, tmax + 1)
        positive_K_blocks: dict[int, TriatomBlock] - canonical K=1 blocks indexed by j
        K0_blocks: dict[int, TriatomBlock] - parity-adapted K=0 blocks indexed by j
        cos_theta: NDArray[np.float64] | None - bending quadrature coordinates in
            cos(theta), shape (n_theta,)
        theta_weights: NDArray[np.float64] | None - bending quadrature weights,
            shape (n_theta,)
    """

    type = MonomerType.TRIATOM
    Eint: NDArray[np.float64]
    jmax: int
    tmax: int
    tmin: int = 0
    exchange_parity: int = 0
    parity_block_sign: int = 1
    K0_available: NDArray[np.bool_] | None = None
    positive_K_available: NDArray[np.bool_] | None = None
    positive_K_blocks: dict[int, TriatomBlock] = field(default_factory=dict)
    K0_blocks: dict[int, TriatomBlock] = field(default_factory=dict)
    radial_1: VibPODVR | None = None
    radial_2: VibPODVR | None = None
    cos_theta: NDArray[np.float64] | None = None
    theta_weights: NDArray[np.float64] | None = None
    energy_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.jmax < 0 or self.tmax < 0:
            message = f"jmax and tmax must be non-negative, but got jmax={self.jmax}, tmax={self.tmax}"
            logger.error(message)
            raise ValueError(message)
        if not 0 <= self.tmin <= self.tmax:
            message = f"tmin must satisfy 0 <= tmin <= tmax, but got tmin={self.tmin}, tmax={self.tmax}"
            logger.error(message)
            raise ValueError(message)
        if self.Eint.ndim != 2:
            message = f"Eint must be a two-dimensional array indexed as Eint[j, t], but got ndim={self.Eint.ndim}"
            logger.error(message)
            raise ValueError(message)
        if self.Eint.shape[0] <= self.jmax or self.Eint.shape[1] <= self.tmax:
            message = f"Eint shape {self.Eint.shape} does not cover jmax={self.jmax} and tmax={self.tmax}"
            logger.error(message)
            raise ValueError(message)
        if self.exchange_parity not in (-1, 0, 1):
            message = f"exchange_parity must be -1, 0, or 1, but got {self.exchange_parity}"
            logger.error(message)
            raise ValueError(message)
        if self.parity_block_sign not in (-1, 1):
            message = f"parity_block_sign must be -1 or 1, but got {self.parity_block_sign}"
            logger.error(message)
            raise ValueError(message)

        finite = np.isfinite(self.Eint)
        if self.K0_available is None:
            K0_available = finite.copy()
            if self.parity_block_sign == -1:
                K0_available[0] = False
            object.__setattr__(self, "K0_available", K0_available)
        if self.positive_K_available is None:
            object.__setattr__(self, "positive_K_available", finite.copy())

    def mis_iter(self, E_cut: float) -> Iterator[MolInnerState]:
        """Iterate over contracted ``(j, t)`` states below an energy cutoff."""
        for j in range(self.jmax + 1):
            for t in range(self.tmin, self.tmax + 1):
                energy = float(self.Eint[j, t])
                if np.isfinite(energy) and energy <= E_cut:
                    yield MolInnerState(j=j, t=t, Eint=energy)

    def energy(self, mis: MolInnerState, K: int) -> float:
        """Return the asymptotic energy of a contracted triatomic state."""
        if mis.t is None:
            message = "Triatomic inner state requires t"
            logger.error(message)
            raise ValueError(message)
        return float(self.Eint[mis.j, mis.t])

    def allows_K(self, mis: MolInnerState, K: int) -> bool:
        """Return whether a contracted state exists in the requested helicity block."""
        if mis.t is None:
            return False
        availability = self.K0_available if K == 0 else self.positive_K_available
        return bool(availability is not None and availability[mis.j, mis.t])


# ----------------------------------------------------------------------------------------
def get_triatom_expansion(
    basis: TriatomBasis,
    j: int,
    K: int,
    t: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Expand one contracted triatomic state in the unsymmetrized primitive basis.

    The returned primitive quantum numbers are ``(j1, omega, v1, v2)``. Both
    signs of omega and both radial permutations are expanded explicitly.

    Inputs:
        basis: TriatomBasis - contracted triatomic monomer basis
        j: int - triatomic rotational angular momentum
        K: int - system helicity
        t: int - zero-based contracted-state index

    Returns:
        qn: NDArray[np.int64] - unsymmetrized primitive quantum numbers, shape
            (n_primitive, 4)
        coefficients: NDArray[np.float64] - expansion coefficients, shape
            (n_primitive,)
    """
    blocks = basis.K0_blocks if K == 0 else basis.positive_K_blocks
    try:
        block = blocks[j]
    except KeyError as error:
        message = f"Triatomic basis has no block for j={j}, K={K}"
        logger.error(message)
        raise ValueError(message) from error

    columns = np.flatnonzero(block.t_indices == t)
    if columns.size != 1:
        message = f"Triatomic state (j={j}, t={t}, K={K}) is not present in its contraction block"
        logger.error(message)
        raise ValueError(message)

    j1max = int(np.max(block.qn[:, 0]))
    vmax_1 = int(np.max(block.qn[:, 2]))
    vmax_2 = int(np.max(block.qn[:, 3]))
    vmax = max(vmax_1, vmax_2)
    qn = _unsym_qn(j, j1max, vmax, vmax)
    transform = _symmetry_transform(qn, block.qn, K, basis.parity_block_sign, basis.exchange_parity)
    coefficients = transform @ block.coefficients[:, int(columns[0])]
    return qn, np.asarray(coefficients, dtype=np.float64)


# ----------------------------------------------------------------------------------------
def _unsym_qn(j2: int, j1max: int, vmax_1: int, vmax_2: int) -> NDArray[np.int64]:
    """Enumerate unsymmetrized (j1, omega, v1, v2) states, shape (n_primitive, 4)."""
    states = [
        (j1, omega, v1, v2)
        for j1 in range(j1max + 1)
        for omega in range(-min(j1, j2), min(j1, j2) + 1)
        for v1 in range(vmax_1 + 1)
        for v2 in range(vmax_2 + 1)
    ]
    return np.asarray(states, dtype=np.int64)


# ----------------------------------------------------------------------------------------
def _adapted_qn(
    j2: int,
    K: int,
    j1max: int,
    vmax_1: int,
    vmax_2: int,
    parity_block_sign: int,
    exchange_parity: int,
) -> NDArray[np.int64]:
    """Enumerate symmetry-retained primitive states with shape (n_adapted, 4)."""
    states: list[tuple[int, int, int, int]] = []
    for j1 in range(j1max + 1):
        omega_max = min(j1, j2)
        omega_min = 0 if parity_block_sign == 1 else 1
        omega_values = range(omega_min, omega_max + 1) if K == 0 else range(-omega_max, omega_max + 1)
        for omega in omega_values:
            for v1 in range(vmax_1 + 1):
                v2_max = vmax_2 if exchange_parity == 0 else min(v1, vmax_2)
                for v2 in range(v2_max + 1):
                    radial_parity = exchange_parity * (1 if omega % 2 == 0 else -1)
                    if exchange_parity != 0 and v1 == v2 and radial_parity != 1:
                        continue
                    states.append((j1, omega, v1, v2))
    return np.asarray(states, dtype=np.int64).reshape(-1, 4)


# ----------------------------------------------------------------------------------------
def _symmetry_transform(
    unsym_qn: NDArray[np.int64],
    adapted_qn: NDArray[np.int64],
    K: int,
    parity_block_sign: int,
    exchange_parity: int,
) -> NDArray[np.float64]:
    """
    Transform symmetry-adapted primitive states into the unsymmetrized basis.

    Inputs:
        unsym_qn: NDArray[np.int64] - unsymmetrized states, shape (n_unsym, 4)
        adapted_qn: NDArray[np.int64] - adapted states, shape (n_adapted, 4)

    Returns:
        transform: NDArray[np.float64] - basis transformation, shape
            (n_unsym, n_adapted)
    """
    lookup = {tuple(int(value) for value in state): index for index, state in enumerate(unsym_qn)}
    transform = np.zeros((unsym_qn.shape[0], adapted_qn.shape[0]), dtype=np.float64)

    for column, (j1, omega, v1, v2) in enumerate(adapted_qn):
        if K > 0 or omega == 0:
            angular_terms = [(int(omega), 1.0)]
        else:
            angular_terms = [(int(omega), 1.0 / np.sqrt(2.0)), (-int(omega), parity_block_sign / np.sqrt(2.0))]

        radial_parity = exchange_parity * (1 if omega % 2 == 0 else -1)
        if exchange_parity == 0 or v1 == v2:
            radial_terms = [(int(v1), int(v2), 1.0)]
        else:
            radial_terms = [(int(v1), int(v2), 1.0 / np.sqrt(2.0)), (int(v2), int(v1), radial_parity / np.sqrt(2.0))]

        for omega_value, angular_factor in angular_terms:
            for v1_value, v2_value, radial_factor in radial_terms:
                row = lookup[(int(j1), omega_value, v1_value, v2_value)]
                transform[row, column] += angular_factor * radial_factor

    return transform


# ----------------------------------------------------------------------------------------
def _diagonalize(H: NDArray[np.float64], n_state: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Diagonalize a symmetric Hamiltonian and retain its lowest states.

    Inputs:
        H: NDArray[np.float64] - Hamiltonian, shape (n_basis, n_basis)
        n_state: int - requested number of eigenstates

    Returns:
        energies: NDArray[np.float64] - retained eigenvalues, shape (n_keep,)
        coefficients: NDArray[np.float64] - phase-fixed eigenvectors, shape
            (n_basis, n_keep)
    """
    n_keep = min(n_state, H.shape[0])
    energies, coefficients = eigh(H, subset_by_index=(0, n_keep - 1))
    return np.asarray(energies), phase_fix(coefficients)


# ----------------------------------------------------------------------------------------
def _match_K0_states(
    K0_energies: NDArray[np.float64],
    positive_energies: NDArray[np.float64],
    tolerance: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Pair K=0 states with positive-K labels only when their energies agree.

    Inputs:
        K0_energies: NDArray[np.float64] - K=0 eigenvalues, shape (n_K0,)
        positive_energies: NDArray[np.float64] - canonical K=1 eigenvalues, shape
            (n_positive,)
        tolerance: float - largest accepted absolute energy difference

    Returns:
        K0_rows: NDArray[np.int64] - matched K=0 positions, shape (n_match,)
        positive_columns: NDArray[np.int64] - corresponding positive-K positions,
            shape (n_match,)
    """
    difference = np.abs(K0_energies[:, None] - positive_energies[None, :])
    candidates = np.argwhere(difference <= tolerance)
    candidates = candidates[np.argsort(difference[candidates[:, 0], candidates[:, 1]])]
    matched_rows: list[int] = []
    matched_columns: list[int] = []

    for row, column in candidates:
        if int(row) in matched_rows or int(column) in matched_columns:
            continue
        matched_rows.append(int(row))
        matched_columns.append(int(column))

    order = np.argsort(matched_columns)
    return np.asarray(matched_rows, dtype=np.int64)[order], np.asarray(matched_columns, dtype=np.int64)[order]


# ----------------------------------------------------------------------------------------
def build_TriatomBasis(
    potential: TriatomPES,
    dvr_1: SineDVR,
    dvr_2: SineDVR,
    n_podvr: tuple[int, int],
    vmax: tuple[int, int],
    masses: tuple[float, float, float],
    equilibrium: tuple[float, float, float],
    n_theta: int,
    j1max: int,
    j2max: int,
    tmax: int,
    parity_block_sign: int,
    exchange_parity: int = 0,
    energy_zero: float | None = None,
    matching_tolerance: float = 1.0e-8,
    Kmax: int | None = None,
) -> TriatomBasis:
    r"""
    Solve the contracted rovibrational eigenstates of a triatomic monomer.

    Formula:
        |j2 t K> = sum_chi T_chi,t^(j2 K) |chi j2 K>

    The K=0 Hamiltonian is diagonalized in the requested parity block. For j2>0
    and ``Kmax != 0``, K=1 defines the global t labels and all positive K blocks
    reuse its eigenvectors. With ``Kmax=0``, each K=0 block instead retains and
    labels its own lowest ``tmax+1`` states, matching a J=0 ABC+D calculation.

    Inputs:
        potential: TriatomPES - vectorized monomer potential mapping coordinates
            with shape (3, n_point) to values with shape (n_point,)
        dvr_1: SineDVR - first one-dimensional reference calculation
        dvr_2: SineDVR - second one-dimensional reference calculation
        n_podvr: tuple[int, int] - retained PODVR sizes for r1 and r2
        vmax: tuple[int, int] - maximum reference vibrational quantum numbers
        masses: tuple[float, float, float] - masses of atoms A, B, and C in atomic units
        equilibrium: tuple[float, float, float] - equilibrium (r1, r2, theta1)
        n_theta: int - number of bending Gauss-Legendre grids
        j1max: int - maximum bending angular momentum
        j2max: int - maximum triatomic rotational angular momentum
        tmax: int - maximum retained contracted-state index
        parity_block_sign: int - epsilon*(-1)^J for the K=0 basis
        exchange_parity: int - A2B exchange parity, or 0 for ABC
        energy_zero: float | None - energy subtracted from every level; lowest level when None
        matching_tolerance: float - maximum K=0/K=1 matching error in atomic units
        Kmax: int | None - retained helicity maximum; 0 constructs a K=0-only
            basis, while None or a positive value preserves positive-K blocks

    Returns:
        basis: TriatomBasis - contracted basis with Eint and K-availability arrays
            of shape (j2max + 1, tmax + 1)
    """
    if parity_block_sign not in (-1, 1):
        message = f"parity_block_sign must be -1 or 1, but got {parity_block_sign}"
        logger.error(message)
        raise ValueError(message)
    if exchange_parity not in (-1, 0, 1):
        message = f"exchange_parity must be -1, 0, or 1, but got {exchange_parity}"
        logger.error(message)
        raise ValueError(message)
    if Kmax is not None and Kmax < 0:
        message = f"Kmax must be non-negative or None, but got {Kmax}"
        logger.error(message)
        raise ValueError(message)
    if exchange_parity != 0 and (n_podvr[0] != n_podvr[1] or vmax[0] != vmax[1]):
        message = "A2B exchange symmetry requires identical radial basis sizes"
        logger.error(message)
        raise ValueError(message)

    radial_1 = build_VibPODVR(dvr_1, n_podvr[0], vmax[0])
    radial_2 = build_VibPODVR(dvr_2, n_podvr[1], vmax[1])
    data = prepare_triatom_hamiltonian(potential, radial_1, radial_2, masses, equilibrium, n_theta, j1max)
    n_state = tmax + 1
    Eint = np.full((j2max + 1, n_state), np.inf, dtype=np.float64)
    K0_available = np.zeros(Eint.shape, dtype=np.bool_)
    positive_K_available = np.zeros(Eint.shape, dtype=np.bool_)
    positive_K_blocks: dict[int, TriatomBlock] = {}
    K0_blocks: dict[int, TriatomBlock] = {}
    ground_energy: float | None = None

    for j2 in range(j2max + 1):
        unsym_qn = _unsym_qn(j2, j1max, vmax[0], vmax[1])
        H_unsym = get_hmat_triatom_unsym(data, j2, unsym_qn)

        if j2 == 0:
            ground_qn = _adapted_qn(j2, 0, j1max, vmax[0], vmax[1], 1, exchange_parity)
            ground_transform = _symmetry_transform(unsym_qn, ground_qn, 0, 1, exchange_parity)
            ground_values, _ = _diagonalize(ground_transform.T @ H_unsym @ ground_transform, 1)
            ground_energy = float(ground_values[0])

        positive_energies: NDArray[np.float64] | None = None
        if j2 > 0 and Kmax != 0:
            positive_qn = _adapted_qn(j2, 1, j1max, vmax[0], vmax[1], parity_block_sign, exchange_parity)
            positive_transform = _symmetry_transform(unsym_qn, positive_qn, 1, parity_block_sign, exchange_parity)
            positive_energies, positive_coefficients = _diagonalize(positive_transform.T @ H_unsym @ positive_transform, n_state)
            positive_K_blocks[j2] = TriatomBlock(
                j=j2,
                K=1,
                qn=positive_qn,
                coefficients=positive_coefficients,
                t_indices=np.arange(positive_energies.size, dtype=np.int64),
            )
            Eint[j2, : positive_energies.size] = positive_energies
            positive_K_available[j2, : positive_energies.size] = True

        K0_qn = _adapted_qn(j2, 0, j1max, vmax[0], vmax[1], parity_block_sign, exchange_parity)
        if K0_qn.size:
            K0_transform = _symmetry_transform(unsym_qn, K0_qn, 0, parity_block_sign, exchange_parity)
            K0_energies, K0_coefficients = _diagonalize(K0_transform.T @ H_unsym @ K0_transform, n_state)

            if positive_energies is None:
                t_indices = np.arange(K0_energies.size, dtype=np.int64)
                Eint[j2, : K0_energies.size] = K0_energies
            else:
                rows, columns = _match_K0_states(K0_energies, positive_energies, matching_tolerance)
                t_indices = columns
                K0_coefficients = K0_coefficients[:, rows]

            K0_available[j2, t_indices] = True
            K0_blocks[j2] = TriatomBlock(j=j2, K=0, qn=K0_qn, coefficients=K0_coefficients, t_indices=t_indices)

        logger.info(f"Solved triatomic eigenstates for j={j2}")

    finite = np.isfinite(Eint)
    shift = ground_energy if energy_zero is None else float(energy_zero)
    if shift is None:
        message = "Could not determine the triatomic reference energy"
        logger.error(message)
        raise ValueError(message)
    Eint[finite] -= shift
    return TriatomBasis(
        Eint=Eint,
        jmax=j2max,
        tmax=tmax,
        exchange_parity=exchange_parity,
        parity_block_sign=parity_block_sign,
        K0_available=K0_available,
        positive_K_available=positive_K_available,
        positive_K_blocks=positive_K_blocks,
        K0_blocks=K0_blocks,
        radial_1=radial_1,
        radial_2=radial_2,
        cos_theta=data.cos_theta,
        theta_weights=data.theta_weights,
        energy_shift=shift,
    )


# ----------------------------------------------------------------------------------------
def prepare_Triatom(
    potential: TriatomPES | None,
    *,
    r: tuple[tuple[float, float], tuple[float, float]],
    n_dvr: tuple[int, int],
    n_podvr: tuple[int, int],
    vmax: tuple[int, int],
    masses: tuple[float, float, float],
    equilibrium: tuple[float, float, float],
    n_theta: int,
    j1max: int,
    j2max: int,
    tmax: int,
    parity_block_sign: int,
    exchange_parity: int = 0,
    energy_zero: float | None = None,
    matching_tolerance: float = 1.0e-8,
    Kmax: int | None = None,
) -> TriatomBasis:
    r"""
    Prepare a contracted triatomic monomer basis from one physical monomer PES.

    The two one-dimensional sine-DVR reference potentials are slices of the
    supplied three-dimensional monomer potential. Users therefore specify the
    physical PES and equilibrium geometry once instead of constructing two
    coordinate-fixing callbacks.

    Formula:
        V_1(r_1) = V_ABC(r_1, r_2,eq, theta_1,eq)
        V_2(r_2) = V_ABC(r_1,eq, r_2, theta_1,eq)

    Here ``V_ABC`` and ``V_1``, ``V_2`` are in atomic units; ``r_1`` and
    ``r_2`` are Radau lengths in bohr, and ``theta_1`` is in radians. The
    resulting one-dimensional Hamiltonians are
    ``h_i = -(2 m_i)^-1 d^2/dr_i^2 + V_i`` with ``m_1=m_A`` and ``m_2=m_C``.

    Inputs:
        potential: TriatomPES | None - vectorized monomer PES mapping coordinates
            with shape (3, n_point) to energies with shape (n_point,); None raises
            an error
        r: tuple[tuple[float,float],tuple[float,float]] - sine-DVR boundaries for
            r1 and r2 in bohr
        n_dvr: tuple[int,int] - primitive sine-DVR sizes for r1 and r2
        n_podvr: tuple[int,int] - retained PODVR sizes for r1 and r2
        vmax: tuple[int,int] - maximum reference vibrational quantum numbers
        masses: tuple[float,float,float] - masses of atoms A, B, and C in atomic units
        equilibrium: tuple[float,float,float] - equilibrium (r1, r2, theta1) in
            bohr, bohr, and radians
        n_theta: int - number of bending Gauss-Legendre grids
        j1max: int - maximum bending angular momentum
        j2max: int - maximum triatomic rotational angular momentum
        tmax: int - maximum retained contracted-state index
        parity_block_sign: int - epsilon*(-1)^J for the K=0 basis
        exchange_parity: int - A2B exchange parity, or 0 for ABC
        energy_zero: float | None - energy subtracted from every level; lowest
            level when None
        matching_tolerance: float - maximum K=0/K=1 matching error in atomic units
        Kmax: int | None - retained helicity maximum; use 0 for a J=0 or
            explicitly K=0-only contracted basis

    Returns:
        basis: TriatomBasis - contracted basis with Eint and K-availability arrays
            of shape (j2max + 1, tmax + 1)
    """
    if potential is None:
        message = "Triatomic monomer preparation requires a monomer potential"
        logger.error(message)
        raise ValueError(message)

    equilibrium_array = np.asarray(equilibrium, dtype=np.float64)
    if equilibrium_array.shape != (3,) or not np.all(np.isfinite(equilibrium_array)):
        message = f"equilibrium must contain three finite values, but got {equilibrium!r}"
        logger.error(message)
        raise ValueError(message)

    def reference_potential(radial_index: int) -> TriatomPES:
        def evaluate(radial: NDArray[np.float64]) -> NDArray[np.float64]:
            radial_array = np.asarray(radial, dtype=np.float64)
            coordinates = np.broadcast_to(equilibrium_array[:, None], (3, radial_array.size)).copy()
            coordinates[radial_index] = radial_array
            return np.asarray(potential(coordinates), dtype=np.float64)

        return evaluate

    dvr_1 = build_SineDVR(r[0][0], r[0][1], n_dvr[0], masses[0], reference_potential(0))
    dvr_2 = build_SineDVR(r[1][0], r[1][1], n_dvr[1], masses[2], reference_potential(1))
    return build_TriatomBasis(
        potential=potential,
        dvr_1=dvr_1,
        dvr_2=dvr_2,
        n_podvr=n_podvr,
        vmax=vmax,
        masses=masses,
        equilibrium=equilibrium,
        n_theta=n_theta,
        j1max=j1max,
        j2max=j2max,
        tmax=tmax,
        parity_block_sign=parity_block_sign,
        exchange_parity=exchange_parity,
        energy_zero=energy_zero,
        matching_tolerance=matching_tolerance,
        Kmax=Kmax,
    )


# ----------------------------------------------------------------------------------------
