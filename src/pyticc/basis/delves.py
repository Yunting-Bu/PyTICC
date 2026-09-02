from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, overload

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.basis.angle import gauss_legendre_dvr, norm_YjK


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DelvesBasis:
    """
    Automatically resolved primitive Delves basis specification.

    Members:
        mass: tuple[float, float, float] - atomic masses (A,B,C) in atomic units
        Jtot: int - conserved total angular momentum
        system_parity: int - conserved total parity, -1 or 1
        exchange_parity: int - BC exchange parity, -1, 0, or 1
        jmax: int - largest primitive diatomic rotational angular momentum
        K_cut: int - largest retained body-fixed helicity
        E_max: float - maximum internal energy in any ABC channel, in Hartree
        rho_min: float - inner hyperradial hard wall in bohr
        scaled_r_max: float - largest mass-scaled diatomic bond coordinate in bohr
        n_sine: int - number of primitive sine FBR functions
        n_vib_quad: int - number of midpoint quadrature points in scaled-r or theta
        n_gamma_quad: int - number of Gauss--Legendre points in cos(gamma)
        angular_qns: tuple[tuple[int, int, int], ...] - primitive (arrangement,j,K)
            labels before adding the sine index; arrangements are one-based
        qns: tuple[tuple[int,int,int,int],...] - preselected ABC channels
            ``(arrangement,v,j,K)``; empty only for low-level primitive calculations
        energies: NDArray[np.float64] - fixed channel thresholds in Hartree, shape
            ``(n_channel,)``
        s_coefficients: NDArray[np.float64] - fixed-Jacobi sine-FBR coefficients,
            shape ``(n_sine,n_channel)``
        energy_zero: float - native total-PES energy subtracted when preparing the
            asymptotic diatoms, in Hartree
    """

    mass: tuple[float, float, float]
    Jtot: int
    system_parity: int
    exchange_parity: int
    jmax: int
    K_cut: int
    E_max: float
    rho_min: float
    scaled_r_max: float
    n_sine: int
    n_vib_quad: int
    n_gamma_quad: int
    angular_qns: tuple[tuple[int, int, int], ...]
    qns: tuple[tuple[int, int, int, int], ...] = ()
    energies: NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    s_coefficients: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    energy_zero: float = 0.0

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies, dtype=np.float64)
        coefficients = np.asarray(self.s_coefficients, dtype=np.float64)
        if energies.shape != (len(self.qns),):
            message = f"Delves channel energies must have shape {(len(self.qns),)}, but got {energies.shape}"
            logger.error(message)
            raise ValueError(message)
        expected_coefficients = (self.n_sine, len(self.qns)) if self.qns else (0, 0)
        if coefficients.shape != expected_coefficients:
            message = f"Delves channel s_coefficients must have shape {expected_coefficients}, but got {coefficients.shape}"
            logger.error(message)
            raise ValueError(message)
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(coefficients)) or not np.isfinite(self.energy_zero):
            message = "Delves channel energies, coefficients, and energy_zero must be finite"
            logger.error(message)
            raise ValueError(message)
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "s_coefficients", coefficients)

    @property
    def n_primitive(self) -> int:
        """Return the uncontracted primitive dimension including the sine index."""
        return self.n_sine * len(self.angular_qns)

    @property
    def n_channel(self) -> int:
        """Return the number of preselected ABC arrangement channels."""
        return len(self.qns)

    @property
    def E_int(self) -> NDArray[np.float64]:
        """Return fixed asymptotic channel thresholds in Hartree."""
        return self.energies


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def midpoint_quad(lower: float, upper: float, n_points: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Construct a uniform midpoint quadrature rule.

    Formula:
        For q = 1,...,M on lower < x < upper,

        x_q = lower + (q-1/2) Delta x,
        w_q = Delta x,
        Delta x = (upper-lower)/M.

    Inputs:
        lower: float - lower integration boundary
        upper: float - upper integration boundary
        n_points: int - number M of midpoint nodes

    Returns:
        grids: NDArray[np.float64] - midpoint nodes, shape (n_points,)
        weights: NDArray[np.float64] - uniform weights, shape (n_points,)
    """
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        message = f"Midpoint quadrature boundaries must be finite and increasing, but got lower={lower}, upper={upper}"
        logger.error(message)
        raise ValueError(message)
    if n_points < 1:
        message = f"n_points must be positive, but got {n_points}"
        logger.error(message)
        raise ValueError(message)

    spacing = (upper - lower) / n_points
    grids = lower + (np.arange(n_points, dtype=np.float64) + 0.5) * spacing
    weights = np.full(n_points, spacing, dtype=np.float64)
    return grids, weights


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def sine_basis(
    lower: float,
    upper: float,
    n_sine: int,
    grids: ArrayLike,
    *,
    derivative: int = 0,
) -> NDArray[np.float64]:
    r"""
    Evaluate normalized particle-in-a-box sine functions or first derivatives.

    Formula:
        For n = 1,...,N and L = upper-lower,

        u_n(x) = sqrt(2/L) sin[n pi (x-lower)/L],

        d u_n(x)/dx = sqrt(2/L) (n pi/L)
                         cos[n pi (x-lower)/L].

        The continuous normalization is

        integral_lower^upper u_m(x) u_n(x) dx = delta_mn.

    Inputs:
        lower: float - left box boundary
        upper: float - right box boundary
        n_sine: int - number N of sine functions
        grids: ArrayLike - evaluation coordinates, shape (n_grid,)
        derivative: int - 0 for values or 1 for first derivatives

    Returns:
        values: NDArray[np.float64] - basis values indexed as [grid,n], shape
            (n_grid,n_sine)
    """
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        message = f"Sine basis boundaries must be finite and increasing, but got lower={lower}, upper={upper}"
        logger.error(message)
        raise ValueError(message)
    if n_sine < 1:
        message = f"n_sine must be positive, but got {n_sine}"
        logger.error(message)
        raise ValueError(message)
    if derivative not in (0, 1):
        message = f"derivative must be 0 or 1, but got {derivative}"
        logger.error(message)
        raise ValueError(message)

    coordinates = np.asarray(grids, dtype=np.float64)
    if coordinates.ndim != 1 or not np.all(np.isfinite(coordinates)):
        message = f"grids must be a finite one-dimensional array, but got shape={coordinates.shape}"
        logger.error(message)
        raise ValueError(message)

    length = upper - lower
    modes = np.arange(1, n_sine + 1, dtype=np.float64)
    arguments = np.pi * (coordinates[:, None] - lower) * modes[None, :] / length
    normalization = np.sqrt(2.0 / length)
    if derivative == 0:
        return normalization * np.sin(arguments)
    return normalization * (np.pi * modes[None, :] / length) * np.cos(arguments)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@overload
def theta_max(rho: float, scaled_r_max: float) -> float: ...


# ----------------------------------------------------------------------------------------
@overload
def theta_max(rho: NDArray[Any], scaled_r_max: float) -> NDArray[np.float64]: ...


# ----------------------------------------------------------------------------------------
def theta_max(rho: float | ArrayLike, scaled_r_max: float) -> float | NDArray[np.float64]:
    r"""
    Return the arrangement-local Delves hyperangle boundary.

    Formula:
        theta_max(rho) = asin[min(1, scaled_r_max/rho)],

        so every retained point satisfies

        scaled_r = rho sin(theta) <= scaled_r_max.

    Inputs:
        rho: float | ArrayLike - positive hyperradius in bohr, scalar or shape (...)
        scaled_r_max: float - positive mass-scaled bond-coordinate limit in bohr

    Returns:
        theta_limit: float | NDArray[np.float64] - hyperangle boundary in radians,
            with the same scalar/array form as rho
    """
    radii = np.asarray(rho, dtype=np.float64)
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        message = "rho must contain finite positive hyperradii"
        logger.error(message)
        raise ValueError(message)
    if not np.isfinite(scaled_r_max) or scaled_r_max <= 0.0:
        message = f"scaled_r_max must be finite and positive, but got {scaled_r_max}"
        logger.error(message)
        raise ValueError(message)

    result = np.arcsin(np.minimum(1.0, scaled_r_max / radii))
    return float(result) if result.ndim == 0 else result


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def delves_theta_basis(
    basis: DelvesBasis,
    rho: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Construct the hyperangle quadrature and sine primitive at one hyperradius.

    Formula:
        theta_max(rho) = asin[min(1,scaled_r_max/rho)],

        theta_q = (q-1/2) theta_max/M_theta,
        w_q = theta_max/M_theta,

        u_n(theta_q;rho) = sqrt(2/theta_max)
                           sin(n pi theta_q/theta_max),

        for q=1,...,M_theta and n=1,...,n_sine. The sine functions obey

        integral_0^theta_max u_m(theta;rho)u_n(theta;rho)dtheta = delta_mn.

    Inputs:
        basis: DelvesBasis - resolved Delves sizes and scaled-r boundary
        rho: float - positive hyperradius in bohr

    Returns:
        theta: NDArray[np.float64] - midpoint hyperangle nodes, shape
            ``(n_vib_quad,)``
        weights: NDArray[np.float64] - hyperangle quadrature weights, shape
            ``(n_vib_quad,)``
        values: NDArray[np.float64] - normalized sine values indexed as
            ``[theta,n]``, shape ``(n_vib_quad,n_sine)``
    """
    limit = float(theta_max(rho, basis.scaled_r_max))
    theta, weights = midpoint_quad(0.0, limit, basis.n_vib_quad)
    return theta, weights, sine_basis(0.0, limit, basis.n_sine, theta)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def delves_angular_basis(
    basis: DelvesBasis,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Construct the Jacobi-angle quadrature and normalized angular primitives.

    Formula:
        x_p = cos(gamma_p), p=1,...,M_gamma, is an M_gamma-point
        Gauss--Legendre rule on [-1,1]. The angular primitive is

        P_tilde_j^K(x) = sqrt[(2j+1)/2 * (j-K)!/(j+K)!] P_j^K(x),

        for 0 <= K <= min(j,K_cut), including the Condon--Shortley phase. Its
        normalization is

        integral_-1^1 P_tilde_j^K(x) P_tilde_j'^K(x) dx = delta_jj'.

    Inputs:
        basis: DelvesBasis - resolved angular and quadrature limits

    Returns:
        cos_gamma: NDArray[np.float64] - Gauss--Legendre nodes, shape
            ``(n_gamma_quad,)``
        weights: NDArray[np.float64] - Gauss--Legendre weights, shape
            ``(n_gamma_quad,)``
        values: NDArray[np.float64] - normalized angular values indexed as
            ``[cos_gamma,j,K]``, shape ``(n_gamma_quad,jmax+1,K_cut+1)``;
            entries with K>j are zero
    """
    cos_gamma, weights = gauss_legendre_dvr(-1.0, 1.0, basis.n_gamma_quad)
    values = np.zeros((basis.n_gamma_quad, basis.jmax + 1, basis.K_cut + 1), dtype=np.float64)
    for K in range(basis.K_cut + 1):
        for j in range(K, basis.jmax + 1):
            values[:, j, K] = norm_YjK(j, K, cos_gamma)
    return cos_gamma, weights, values


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_delves_qns(
    mass: Sequence[float],
    Jtot: int,
    system_parity: int,
    exchange_parity: int,
    jmax: int,
    K_cut: int,
) -> tuple[tuple[int, int, int], ...]:
    r"""
    Enumerate arrangement, diatomic rotation, and nonnegative helicity labels.

    Formula:
        K_min = 0 when system_parity = (-1)^Jtot, and K_min = 1 otherwise,

        K_min <= K <= min(j,Jtot,K_cut).

        For exchange_parity = +/-1, arrangements 2 and 3 are exchange-related and only
        arrangements a = 1,2 are explicit. Arrangement 1 contains even j for
        exchange_parity = +1 and odd j for exchange_parity = -1. Otherwise all three arrangements and
        every j = 0,...,jmax are explicit.

    Inputs:
        mass: Sequence[float] - three positive atomic masses (A,B,C) in any
            common unit
        Jtot: int - conserved total angular momentum
        system_parity: int - conserved total parity, -1 or 1
        exchange_parity: int - BC exchange parity, -1, 0, or 1
        jmax: int - largest diatomic rotational angular momentum
        K_cut: int - largest retained helicity

    Returns:
        qns: tuple[tuple[int, int, int], ...] - one-based arrangement and (j,K)
            labels ordered as arrangement, j, K
    """
    masses = _validate_delves_inputs(mass, Jtot, system_parity, exchange_parity, jmax, K_cut)
    effective_exchange_parity = exchange_parity
    if not np.isclose(masses[1], masses[2], rtol=1.0e-12, atol=0.0):
        effective_exchange_parity = 0

    K_min = 0 if system_parity == (-1) ** Jtot else 1
    n_arrangement = 3 - abs(effective_exchange_parity)
    qns: list[tuple[int, int, int]] = []
    for arrangement in range(1, n_arrangement + 1):
        if arrangement == 1:
            j_min = (1 - effective_exchange_parity) // 2
            j_step = 1 + abs(effective_exchange_parity)
        else:
            j_min = 0
            j_step = 1
        for j in range(j_min, jmax + 1, j_step):
            for K in range(K_min, min(j, Jtot, K_cut) + 1):
                qns.append((arrangement, j, K))
    return tuple(qns)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def sine_kinetic(n_sine: int, length: float, mass_factor: float, *, theta: bool) -> NDArray[np.float64]:
    r"""
    Return diagonal sine-FBR kinetic energies in ``s`` or Delves ``theta``.

    Formula:
        For ``n=1,...,n_sine``,

        T_n^(s) = (n pi/L)^2/(2 mu),

        T_n^(theta) = [(n pi/L)^2-1/4]/(2 mu rho^2),

        where ``mass_factor=mu`` for ``s`` and ``mass_factor=mu*rho^2`` for
        ``theta``.

    Inputs:
        n_sine: int - number of sine functions
        length: float - sine-box length in bohr or radians
        mass_factor: float - positive mass factor in atomic units
        theta: bool - whether to include the Delves ``-1/4`` term

    Returns:
        energies: NDArray[np.float64] - diagonal kinetic energies, shape
            ``(n_sine,)``
    """
    modes = np.arange(1, n_sine + 1, dtype=np.float64)
    eigenvalues = (np.pi * modes / length) ** 2
    if theta:
        eigenvalues -= 0.25
    return eigenvalues / (2.0 * mass_factor)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def sine_reference_hamiltonian(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    kinetic: NDArray[np.float64],
    effective_potential: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""
    Construct one sine-FBR reference Hamiltonian by midpoint quadrature.

    Formula:
        H_nm = delta_nm T_n
             + sum_q w_q u_n(x_q) V_eff(x_q) u_m(x_q).

    Inputs:
        values: NDArray[np.float64] - sine values indexed as ``[grid,n]``
        weights: NDArray[np.float64] - quadrature weights, shape ``(n_grid,)``
        kinetic: NDArray[np.float64] - diagonal kinetic energies, shape ``(n_sine,)``
        effective_potential: NDArray[np.float64] - sampled effective potential,
            shape ``(n_grid,)``

    Returns:
        hamiltonian: NDArray[np.float64] - symmetric FBR Hamiltonian, shape
            ``(n_sine,n_sine)``
    """
    hamiltonian = values.T @ (weights[:, None] * effective_potential[:, None] * values)
    hamiltonian += np.diag(kinetic)
    return 0.5 * (hamiltonian + hamiltonian.T)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def clean_sine_phases(vectors: NDArray[np.float64], primitive_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply ABC's positive-inner-turning-point phase convention."""
    result = vectors.copy()
    wavefunctions = primitive_values @ result
    for state in range(result.shape[1]):
        maximum = float(np.max(np.abs(wavefunctions[:, state])))
        significant = np.flatnonzero(np.abs(wavefunctions[:, state]) > 0.1 * maximum)
        if significant.size and wavefunctions[significant[0], state] < 0.0:
            result[:, state] *= -1.0
    return result


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_delves_channels(
    diatom_basis: object,
    Jtot: int,
    system_parity: int,
    exchange_parity: int,
    K_cut: int,
) -> DelvesBasis:
    """
    Build and order ABC channels ``(arrangement,v,j,K)`` from prepared diatoms.

    Inputs:
        diatom_basis: DelvesDiatomBasis - three arrangement-asymptotic diatomic bases
        Jtot: int - conserved total angular momentum
        system_parity: int - conserved total parity, -1 or 1
        exchange_parity: int - BC exchange parity, -1, 0, or 1
        K_cut: int - largest retained nonnegative helicity

    Returns:
        basis: DelvesBasis - primitive numerical support plus the fixed channel
            labels, thresholds, and sine-FBR coefficients required by ABC
    """
    from pyticc.basis.monomer.delves import DelvesDiatomBasis

    if not isinstance(diatom_basis, DelvesDiatomBasis):
        message = "build_delves_channels requires a DelvesDiatomBasis"
        logger.error(message)
        raise TypeError(message)
    angular_qns = build_delves_qns(
        diatom_basis.mass,
        Jtot,
        system_parity,
        exchange_parity,
        diatom_basis.jmax,
        K_cut,
    )
    angular_set = set(angular_qns)
    records: list[tuple[tuple[int, int, int, int], float, NDArray[np.float64]]] = []
    for state_index, (arrangement, v, j) in enumerate(diatom_basis.qns):
        for K in range(min(j, Jtot, K_cut) + 1):
            if (arrangement, j, K) not in angular_set:
                continue
            records.append(
                (
                    (arrangement, v, j, K),
                    float(diatom_basis.energies[state_index]),
                    diatom_basis.coefficients[:, state_index].copy(),
                )
            )
    if not records:
        message = "The requested Jtot, parities, and K_cut produce no Delves channels"
        logger.error(message)
        raise ValueError(message)
    records.sort(key=lambda item: item[0])
    effective_exchange_parity = exchange_parity if np.isclose(diatom_basis.mass[1], diatom_basis.mass[2], rtol=1.0e-12, atol=0.0) else 0
    basis = DelvesBasis(
        mass=diatom_basis.mass,
        Jtot=Jtot,
        system_parity=system_parity,
        exchange_parity=effective_exchange_parity,
        jmax=diatom_basis.jmax,
        K_cut=min(K_cut, Jtot, diatom_basis.jmax),
        E_max=diatom_basis.E_max,
        rho_min=diatom_basis.rho_min,
        scaled_r_max=diatom_basis.scaled_r_max,
        n_sine=diatom_basis.n_sine,
        n_vib_quad=diatom_basis.n_vib_quad,
        n_gamma_quad=diatom_basis.n_gamma_quad,
        angular_qns=angular_qns,
        qns=tuple(record[0] for record in records),
        energies=np.asarray([record[1] for record in records], dtype=np.float64),
        s_coefficients=np.column_stack([record[2] for record in records]),
        energy_zero=diatom_basis.energy_zero,
    )
    logger.info(
        "Reactive channels prepared:\n"
        + "\n".join(
            f"{index:5d}  a={arrangement} v={v} j={j} K={K}  E={energy:.12e} Hartree"
            for index, ((arrangement, v, j, K), energy) in enumerate(zip(basis.qns, basis.energies, strict=True), start=1)
        )
    )
    return basis


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _validate_delves_inputs(
    mass: Sequence[float],
    Jtot: int,
    system_parity: int,
    exchange_parity: int,
    jmax: int,
    K_cut: int,
) -> tuple[float, float, float]:
    """Validate shared Delves mass and angular-momentum inputs."""
    masses = tuple(float(value) for value in mass)
    if len(masses) != 3 or not np.all(np.isfinite(masses)) or not np.all(np.asarray(masses) > 0.0):
        message = f"mass must contain three finite positive atomic masses, but got {mass!r}"
        logger.error(message)
        raise ValueError(message)
    if Jtot < 0 or jmax < 0 or K_cut < 0:
        message = f"Jtot, jmax, and K_cut must be non-negative, but got {(Jtot, jmax, K_cut)}"
        logger.error(message)
        raise ValueError(message)
    if system_parity not in (-1, 1):
        message = f"system_parity must be -1 or 1, but got {system_parity}"
        logger.error(message)
        raise ValueError(message)
    if exchange_parity not in (-1, 0, 1):
        message = f"exchange_parity must be -1, 0, or 1, but got {exchange_parity}"
        logger.error(message)
        raise ValueError(message)
    return masses


# ----------------------------------------------------------------------------------------
