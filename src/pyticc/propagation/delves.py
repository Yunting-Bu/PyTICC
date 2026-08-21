from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc._typing import JaxDevice
from pyticc.energy import EnergyInput, get_Etot
from pyticc.matrix.delves import mass_scale
from pyticc.propagation.config import Propagation
from pyticc.propagation.device import resolve_device
from pyticc.propagation.grid import build_radial_sectors
from pyticc.propagation.logd import LogDInput, initialize_logD_capture, propagate_logD


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DelvesPropagationResult:
    """
    Final piecewise-fixed-grid Delves propagation state.

    Members:
        Y_final: LogDInput - final LogD matrices indexed as
            ``[energy,surface,surface]``
        rho_final: float - final hyperradius in bohr
        surface_rho: float - midpoint hyperradius of the final surface basis in
            bohr; generally smaller than ``rho_final``
        surface_energies: NDArray[np.float64] - final adiabatic surface energies
            in Hartree, shape ``(n_surface,)``
        surface_coefficients: NDArray[np.float64] - final primitive-to-surface
            coefficients, shape ``(basis.n_primitive,n_surface)``
        radial_points: NDArray[np.float64] - fixed sector endpoints generated
            from ``Propagation.boundaries`` and ``Propagation.half_steps``,
            shape ``(n_sector+1,)``
    """

    Y_final: LogDInput
    rho_final: float
    surface_rho: float
    surface_energies: NDArray[np.float64]
    surface_coefficients: NDArray[np.float64]
    radial_points: NDArray[np.float64]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def propagate_delves(
    hamiltonian: object,
    Etot: EnergyInput,
    config: Propagation,
) -> DelvesPropagationResult:
    r"""
    Propagate reactive Delves surface channels on the configured radial grid.

    The existing Manolopoulos LogD implementation is reused unchanged. Surface
    Hamiltonians are placed at sector midpoints, as in ABC. For an inelastic
    calculation the first sector applies the exact hard wall
    ``F(rho_min)=0``; capture mode retains its incoming-wave initialization.

    ``config.boundaries`` and ``config.half_steps`` have exactly the same
    meaning as in the fixed-channel propagators. Every user interval is divided
    into sectors of width ``2 * half_step``; only the last sector is shortened
    when needed to end exactly at the interval boundary.

    Formula:
        On a sector ``[rho_L,rho_R]`` of width ``Delta rho``, the surface point
        is its midpoint

        rho_c=(rho_L+rho_R)/2.

        Canonical orthogonalization there gives energies
        epsilon_q and primitive coefficients C_q. The radial equation matrix in
        this orthonormal surface basis is diagonal:

        W_q(E) = 2 mu [diag(epsilon_q)-E I],

        where mu=sqrt(m_A m_B m_C/(m_A+m_B+m_C)). Between two surface bases,

        T_qr = C_q^T P(rho_q,rho_r) C_r,

        and the LogD matrix is transformed following ABC ``logder``:

        Y_r = T_qr^T Y_q T_qr.

        For the first inelastic sector, the exact hard-wall LogD is diagonal:

        Y_i(rho_R)=p_i coth(p_i Delta rho),  p_i^2>0,
        Y_i(rho_R)=q_i cot(q_i Delta rho),   p_i^2=-q_i^2<0.

        For every later sector s, transform the previous midpoint LogD to the
        new midpoint surface basis and propagate through the configured width:

        Y_s^L = T_(s-1,s)^T Y_(s-1)^R T_(s-1,s),

        Y_s^R = L_(Delta rho_s)[Y_s^L;epsilon_s].

        Here ``L_(Delta rho_s)`` is one call to the existing LDMD propagator
        with half-step ``Delta rho_s/2`` and constant diagonal surface energy
        ``epsilon_s``. No accept/reject controller changes the configured grid.

    Inputs:
        hamiltonian: DelvesHamiltonian - resolved primitive basis, total PES,
            surface solver, and sector-overlap provider
        Etot: EnergyInput - total scattering energies in Hartree
        config: Propagation - piecewise hyperradial boundaries and half-steps,
            boundary mode, device, and logging settings

    Returns:
        result: DelvesPropagationResult - final LogD, surface data, and fixed
            sector endpoints required for asymptotic matching
    """
    from pyticc.scattering.reactive.delves import DelvesHamiltonian, DelvesSurface

    if not isinstance(hamiltonian, DelvesHamiltonian):
        message = "Delves propagation requires a DelvesHamiltonian"
        logger.error(message)
        raise TypeError(message)
    basis = hamiltonian.basis
    energies = get_Etot(Etot)
    if not np.isclose(config.boundaries[0], basis.rho_min, rtol=0.0, atol=1.0e-12):
        message = f"Delves propagation must start at the basis hard wall rho_min={basis.rho_min}, but got {config.boundaries[0]}"
        logger.error(message)
        raise ValueError(message)
    selected_device = resolve_device(config.device)
    reduced_mass, _ = mass_scale(basis.mass)
    sectors = build_radial_sectors(config.boundaries, config.half_steps)
    n_sector = len(sectors)
    surface_current: DelvesSurface | None = None
    Y_current: jax.Array | None = None
    propagation_start = perf_counter()
    progress_interval = max(1, n_sector // 10)
    logger.info(f"Propagation device: {selected_device.label}, x64={jax.config.read('jax_enable_x64')}")
    if config.print_verbose:
        logger.info(f"Delves propagation started: sectors={n_sector}, primitive={basis.n_primitive}, energies={energies.size}")

    for sector_index, sector in enumerate(sectors, start=1):
        surface_next = hamiltonian.surface(sector.radial_mid)
        width = sector.radial_end - sector.radial_start
        if Y_current is None:
            if config.mode == "capture":
                W_initial = _surface_W(surface_next.energies, reduced_mass, energies)
                Y_initial = initialize_logD_capture(jax.device_put(W_initial, selected_device.device))
                Y_current = _propagate_constant_sector(
                    Y_initial,
                    energies,
                    reduced_mass,
                    surface_next.energies,
                    width,
                    selected_device.device,
                )
            else:
                Y_current = _hard_wall_logD(energies, reduced_mass, surface_next.energies, width, selected_device.device)
        else:
            assert surface_current is not None
            transform = hamiltonian.transform(surface_current, surface_next)
            Y_current = _propagate_constant_sector(
                _transform_logD(Y_current, transform, selected_device.device),
                energies,
                reduced_mass,
                surface_next.energies,
                width,
                selected_device.device,
            )
        surface_current = surface_next
        if config.print_verbose and (sector_index == n_sector or sector_index % progress_interval == 0):
            Y_current.block_until_ready()
            logger.info(
                f"Delves propagation: {sector_index}/{n_sector} sectors, rho={sector.radial_end:.6f} bohr, "
                f"surfaces={surface_current.energies.size}, wall={perf_counter() - propagation_start:.3f} s"
            )

    if Y_current is None or surface_current is None:
        message = "Delves propagation produced no sectors"
        logger.error(message)
        raise RuntimeError(message)
    radial_points = np.asarray([sectors[0].radial_start, *(sector.radial_end for sector in sectors)], dtype=np.float64)
    return DelvesPropagationResult(
        Y_final=Y_current,
        rho_final=sectors[-1].radial_end,
        surface_rho=surface_current.rho,
        surface_energies=surface_current.energies,
        surface_coefficients=surface_current.coefficients,
        radial_points=radial_points,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _surface_W(surface_energies: NDArray[np.float64], reduced_mass: float, energies: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return energy-dependent diagonal radial matrices for initialization."""
    diagonal = 2.0 * reduced_mass * (surface_energies[None, :] - energies[:, None])
    return diagonal[:, :, None] * np.eye(surface_energies.size, dtype=np.float64)[None, :, :]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _hard_wall_logD(
    energies: NDArray[np.float64],
    reduced_mass: float,
    surface_energies: NDArray[np.float64],
    width: float,
    device: JaxDevice,
) -> jax.Array:
    r"""
    Return the exact constant-sector LogD for ``F(rho_left)=0``.

    Formula:
        For ``p_i^2=2 mu [U_i-E]`` over a sector of width ``Delta rho``,

        Y_i(rho_right)=p_i coth(p_i Delta rho),  p_i^2>0,

        Y_i(rho_right)=q_i cot(q_i Delta rho),   p_i^2=-q_i^2<0,

        with the threshold limit ``Y_i=1/Delta rho``. The result is diagonal in
        the constant midpoint surface basis and is batched over total energies.

    Inputs:
        energies: NDArray[np.float64] - total energies in Hartree, shape
            ``(n_energy,)``
        reduced_mass: float - Delves hyperradial mass in electron masses
        surface_energies: NDArray[np.float64] - midpoint surface energies in
            Hartree, shape ``(n_surface,)``
        width: float - complete first-sector width in bohr
        device: JaxDevice - selected execution device

    Returns:
        Ymat: jax.Array - real diagonal hard-wall LogD, shape
            ``(n_energy,n_surface,n_surface)``
    """
    radial_values = 2.0 * reduced_mass * (surface_energies[None, :] - energies[:, None])
    magnitude = np.sqrt(np.abs(radial_values))
    argument = magnitude * width
    small = argument < 1.0e-6
    safe_argument = np.where(small, 1.0, argument)
    positive = magnitude / np.tanh(safe_argument)
    negative = magnitude / np.tan(safe_argument)
    series = 1.0 / width + radial_values * width / 3.0 - radial_values**2 * width**3 / 45.0
    diagonal = np.where(small, series, np.where(radial_values >= 0.0, positive, negative))
    Ymat = diagonal[:, :, None] * np.eye(surface_energies.size, dtype=np.float64)[None, :, :]
    return jax.device_put(Ymat, device)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _transform_logD(Ymat: jax.Array, transform: NDArray[np.float64], device: JaxDevice) -> jax.Array:
    """Apply ABC's congruence transformation to an energy batch of LogD matrices."""
    transform_device = jax.device_put(transform, device)
    return jnp.einsum("pi,epq,qj->eij", transform_device, Ymat, transform_device, optimize=True)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _propagate_constant_sector(
    Ymat: jax.Array,
    energies: NDArray[np.float64],
    reduced_mass: float,
    surface_energies: NDArray[np.float64],
    width: float,
    device: JaxDevice,
) -> jax.Array:
    """Reuse LDMD for one constant diagonal surface-energy sector."""
    W_base = np.diag(2.0 * reduced_mass * surface_energies)[None, :, :]
    return propagate_logD(
        Ymat,
        energies,
        reduced_mass,
        np.asarray([0.5 * width]),
        W_base,
        W_base,
        W_base,
        device=device,
    )
