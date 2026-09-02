from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import cast

import jax
import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.atom_diatom as vmat
from pyticc._typing import JaxDevice
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF
from pyticc.basis.monomer import AtomSpec, DiatomBasis, DiatomElectricBasis
from pyticc.matrix.interaction import VBasisDevice, contract, contract_device, device_basis
from pyticc.pes.adiabatic import PESWrapper, RadialInput, get_Vgrid_atom_diatom, get_Vgrid_atom_diatom_electric_sf
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.potential import PotentialGrid, _potential_radial_grid, _require_type
from pyticc.system import ScatteringType, ScattSystem

_SCATTERING_TYPE = ScatteringType.ATOM_DIATOM
_ELECTRIC_SCATTERING_TYPE = ScatteringType.ATOM_DIATOM_ELECTRIC


# ----------------------------------------------------------------------------------------
def prepare_potential(
    system: ScattSystem,
    boundaries: Sequence[float],
    half_steps: Sequence[float],
    *,
    n_theta: int = 16,
    processes: int = 1,
) -> PotentialGrid:
    """Evaluate an adiabatic atom-diatom PES on the complete propagation grid.

    Inputs:
        system: ScattSystem - prepared atom-diatom scattering system
        boundaries: Sequence[float] - radial interval boundaries in bohr
        half_steps: Sequence[float] - propagation half-step in each interval
        n_theta: int - retained Jacobi-angle quadrature points
        processes: int - worker processes used for the radial PES batch

    Returns:
        potential_grid: PotentialGrid - raw interaction values and quadrature data
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, DiatomBasis):
        message = "Atom-diatom potential preparation requires AtomSpec and DiatomBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper) or not isinstance(system.basis, ChannelBasis):
        message = "Atom-diatom potential preparation requires a scalar PES and prepared channels"
        logger.error(message)
        raise TypeError(message)

    exchange_parity = system.basis.channel_spec.exchange_parity_Y
    exchange_parities = (exchange_parity,) if isinstance(exchange_parity, int) else exchange_parity
    cos_theta, theta_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta, symmetry=all(parity != 0 for parity in exchange_parities))
    theta = np.arccos(cos_theta)
    boundaries_value, half_steps_value, sectors, radial_points = _potential_radial_grid(boundaries, half_steps)
    radial = system.monomer_Y.rovib.grids
    values = get_Vgrid_atom_diatom(system.potential, radial_points, radial, theta, processes=processes)
    return PotentialGrid(
        boundaries=boundaries_value,
        half_steps=half_steps_value,
        sectors=sectors,
        radial_points=radial_points,
        scattering_type=_SCATTERING_TYPE,
        coordinates=(("r", radial), ("cos_theta", cos_theta)),
        weights=(("theta", theta_weights),),
        values=values,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_hamiltonian(
    system: ScattSystem,
    *,
    n_theta: int = 16,
    potential_grid: PotentialGrid | None = None,
) -> ScattHamiltonian:
    """Build an adiabatic atom-diatom scattering Hamiltonian.

    Inputs:
        system: ScattSystem - atom-diatom system with a scalar PES
        n_theta: int - retained Jacobi-angle quadrature points
        potential_grid: PotentialGrid | None - optional precomputed raw PES grid

    Returns:
        hamiltonian: ScattHamiltonian - projected body-fixed Hamiltonian
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, DiatomBasis):
        message = "Atom-diatom Hamiltonian requires AtomSpec as monomer_X and DiatomBasis as monomer_Y"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper):
        message = "Adiabatic atom-diatom Hamiltonian requires a PESWrapper"
        logger.error(message)
        raise TypeError(message)
    if system.reduced_mass is None:
        message = "Adiabatic atom-diatom Hamiltonian requires a collision reduced mass"
        logger.error(message)
        raise ValueError(message)
    if not isinstance(system.basis, ChannelBasis):
        message = "Atom-diatom Hamiltonian requires a field-free channel basis prepared by build_ScattSystem"
        logger.error(message)
        raise TypeError(message)

    diatom = system.monomer_Y
    rovib = diatom.rovib
    pes = system.potential
    basis = system.basis
    exchange_parity = basis.channel_spec.exchange_parity_Y
    exchange_parities = (exchange_parity,) if isinstance(exchange_parity, int) else exchange_parity
    if potential_grid is None:
        cos_theta, theta_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta, symmetry=all(parity != 0 for parity in exchange_parities))
    else:
        _require_type(potential_grid, _SCATTERING_TYPE)
        cos_theta = potential_grid.coordinate("cos_theta")
        theta_weights = potential_grid.weight("theta")
    theta = np.arccos(cos_theta)
    V_basis = vmat.prepare(basis, rovib, cos_theta, theta_weights)
    device_bases: dict[tuple[str, int], VBasisDevice] = {}

    def Vgrid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the atom-diatom PES grid."""
        if potential_grid is not None:
            return cast(NDArray[np.float64], potential_grid.take(radial_points))
        return get_Vgrid_atom_diatom(pes, radial_points, rovib.grids, theta)

    def Vmat(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Contract the PES grid into the channel basis."""
        return contract(V_basis, Vgrid(radial_points))

    def V_blocks(
        radial_points: NDArray[np.float64],
        channel_blocks: tuple[tuple[int, ...], ...],
    ) -> tuple[NDArray[np.float64], ...]:
        """Contract one shared PES grid into several channel blocks."""
        potential_grid = Vgrid(radial_points)
        return tuple(contract(V_basis, potential_grid, indices) for indices in channel_blocks)

    def V_blocks_device(radial_points: NDArray[np.float64], channel_blocks: tuple[tuple[int, ...], ...], device: JaxDevice) -> tuple[jax.Array, ...]:
        """Contract channel blocks from host or device-resident PES values."""
        key = (device.platform, device.id)
        if key not in device_bases:
            device_bases[key] = device_basis(V_basis, device)
        values = (
            Vgrid(radial_points)
            if potential_grid is None
            else cast(NDArray[np.float64] | jax.Array, potential_grid.take_device(radial_points, device))
        )
        return tuple(contract_device(V_basis, device_bases[key], values, device, indices) for indices in channel_blocks)

    return ScattHamiltonian(
        basis=basis,
        reduced_mass=system.reduced_mass,
        interaction=Vmat,
        approx=system.approx,
        K_delta=system.K_delta,
        block_interaction=V_blocks,
        device_block_interaction=V_blocks_device,
        potential_grid_size=prod(V_basis.grid_shape),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare_potential_electric_sf(
    system: ScattSystem,
    boundaries: Sequence[float],
    half_steps: Sequence[float],
    *,
    n_theta_r: int = 16,
    n_theta_R: int = 16,
    n_delta: int = 16,
    delta_symmetry: bool = True,
    processes: int = 1,
) -> PotentialGrid:
    """Evaluate an electric-field atom-diatom PES on the complete SF grid.

    Inputs:
        system: ScattSystem - prepared fixed-M electric-field scattering system
        boundaries: Sequence[float] - radial interval boundaries in bohr
        half_steps: Sequence[float] - propagation half-step in each interval
        n_theta_r: int - quadrature order for the molecular polar angle
        n_theta_R: int - quadrature order for the intermolecular polar angle
        n_delta: int - retained relative-azimuth quadrature points
        delta_symmetry: bool - whether to use the paired half-interval rule
        processes: int - worker processes used for the radial PES batch

    Returns:
        potential_grid: PotentialGrid - raw interaction values and SF quadrature data
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, DiatomElectricBasis):
        message = "Electric potential preparation requires AtomSpec and DiatomElectricBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper):
        message = "Electric potential preparation requires a scalar PES"
        logger.error(message)
        raise TypeError(message)

    cos_theta_r, theta_r_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta_r)
    cos_theta_R, theta_R_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta_R)
    delta, delta_weights = gauss_legendre_dvr(0.0, 2.0 * np.pi, n_delta, symmetry=delta_symmetry)
    gamma = _electric_gamma(cos_theta_r, cos_theta_R, delta)
    boundaries_value, half_steps_value, sectors, radial_points = _potential_radial_grid(boundaries, half_steps)
    radial = system.monomer_Y.grids
    values = get_Vgrid_atom_diatom_electric_sf(system.potential, radial_points, radial, gamma, processes=processes)
    return PotentialGrid(
        boundaries=boundaries_value,
        half_steps=half_steps_value,
        sectors=sectors,
        radial_points=radial_points,
        scattering_type=_ELECTRIC_SCATTERING_TYPE,
        coordinates=(("r", radial), ("cos_theta_r", cos_theta_r), ("cos_theta_R", cos_theta_R), ("delta", delta), ("gamma", gamma)),
        weights=(("theta_r", theta_r_weights), ("theta_R", theta_R_weights), ("delta", delta_weights)),
        values=values,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_hamiltonian_electric_sf(
    system: ScattSystem,
    *,
    n_theta_r: int = 16,
    n_theta_R: int = 16,
    n_delta: int = 16,
    delta_symmetry: bool = True,
    potential_grid: PotentialGrid | None = None,
) -> ScattHamiltonian:
    r"""
    Build an electric-field atom-diatom Hamiltonian in the space-fixed representation.

    Formula:
        Channels eta = (alpha,m,l,m_l) are generated from

        m = M-m_l,    l = 0,...,lmax,    m_l = -l,...,l,

        and retained when epsilon_{alpha m} <= E_cut. The three angular
        integrations use Gauss-Legendre rules in

        x_r = cos(theta_r) in [-1,1],
        x_R = cos(theta_R) in [-1,1],
        delta = phi_r-phi_R in [0,2 pi].

        The interaction callback evaluates Delta V(R,r,gamma) and contracts it
        into V^M(R). The resulting Hamiltonian is

        H(R) = diag(E_int) + U_SF/(2 mu_R R^2) + V^M(R).

        When delta_symmetry is True, n_delta nodes are retained from the lower
        half of a 2*n_delta rule and paired weights are doubled. This is exact
        for a scalar PES depending on delta only through cos(gamma).

    Inputs:
        system: ScattSystem - atom-diatom system containing AtomSpec,
            DiatomElectricBasis, M, interaction PES, and collision reduced mass
        n_theta_r: int - Gauss-Legendre order for cos(theta_r)
        n_theta_R: int - Gauss-Legendre order for cos(theta_R)
        n_delta: int - retained relative-azimuth quadrature points
        delta_symmetry: bool - whether to use the paired half-interval rule
        potential_grid: PotentialGrid | None - optional precomputed raw PES grid

    Returns:
        hamiltonian: ScattHamiltonian - electric-field fixed-M SF Hamiltonian
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, DiatomElectricBasis):
        message = "Electric atom-diatom Hamiltonian requires AtomSpec as monomer_X and DiatomElectricBasis as monomer_Y"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper):
        message = "Electric atom-diatom Hamiltonian requires a PESWrapper"
        logger.error(message)
        raise TypeError(message)
    if system.M is None:
        message = "Electric atom-diatom Hamiltonian requires M"
        logger.error(message)
        raise ValueError(message)
    if system.reduced_mass is None:
        message = "Electric atom-diatom Hamiltonian requires a collision reduced mass"
        logger.error(message)
        raise ValueError(message)
    if not isinstance(system.basis, ChannelBasisElectricSF):
        message = "Electric atom-diatom Hamiltonian requires SF channels prepared by build_ScattSystem"
        logger.error(message)
        raise TypeError(message)

    monomer_basis = system.monomer_Y
    pes = system.potential
    basis = system.basis
    if potential_grid is None:
        cos_theta_r, theta_r_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta_r)
        cos_theta_R, theta_R_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta_R)
        delta, delta_weights = gauss_legendre_dvr(0.0, 2.0 * np.pi, n_delta, symmetry=delta_symmetry)
    else:
        _require_type(potential_grid, _ELECTRIC_SCATTERING_TYPE)
        cos_theta_r = potential_grid.coordinate("cos_theta_r")
        cos_theta_R = potential_grid.coordinate("cos_theta_R")
        delta = potential_grid.coordinate("delta")
        theta_r_weights = potential_grid.weight("theta_r")
        theta_R_weights = potential_grid.weight("theta_R")
        delta_weights = potential_grid.weight("delta")
    V_basis = vmat.build_AtomDiatomVBasisElectricSF(
        basis,
        monomer_basis,
        cos_theta_r,
        theta_r_weights,
        cos_theta_R,
        theta_R_weights,
        delta,
        delta_weights,
    )
    electric_device_bases: dict[tuple[str, int], vmat.AtomDiatomVBasisElectricSFDevice] = {}

    def Vgrid(radial_points: RadialInput) -> NDArray[np.float64]:
        """Evaluate the atom-diatom PES on the prepared SF geometry grid."""
        if potential_grid is not None:
            return cast(NDArray[np.float64], potential_grid.take(radial_points))
        return get_Vgrid_atom_diatom_electric_sf(pes, radial_points, V_basis.r, V_basis.gamma)

    def Vmat(radial_points: RadialInput) -> NDArray[np.float64]:
        """Contract the SF PES grid into the fixed-M channel basis."""
        return vmat.contract_electric_sf(V_basis, Vgrid(radial_points))

    def V_blocks_device(radial_points: NDArray[np.float64], channel_blocks: tuple[tuple[int, ...], ...], device: JaxDevice) -> tuple[jax.Array, ...]:
        """Contract electric SF blocks from host or device-resident PES values."""
        key = (device.platform, device.id)
        if key not in electric_device_bases:
            electric_device_bases[key] = vmat.device_basis_electric_sf(V_basis, device)
        values = (
            Vgrid(radial_points)
            if potential_grid is None
            else cast(NDArray[np.float64] | jax.Array, potential_grid.take_device(radial_points, device))
        )
        return tuple(vmat.contract_electric_sf_device(V_basis, electric_device_bases[key], values, device, indices) for indices in channel_blocks)

    return ScattHamiltonian(
        basis=basis,
        reduced_mass=system.reduced_mass,
        interaction=Vmat,
        device_block_interaction=V_blocks_device,
        potential_grid_size=prod(V_basis.grid_shape),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _electric_gamma(cos_theta_r: NDArray[np.float64], cos_theta_R: NDArray[np.float64], delta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the Jacobi angle on the electric-field SF quadrature grid."""
    sin_theta_r = np.sqrt(np.clip(1.0 - cos_theta_r**2, 0.0, None))
    sin_theta_R = np.sqrt(np.clip(1.0 - cos_theta_R**2, 0.0, None))
    cos_gamma = (
        cos_theta_r[:, None, None] * cos_theta_R[None, :, None]
        + sin_theta_r[:, None, None] * sin_theta_R[None, :, None] * np.cos(delta)[None, None, :]
    )
    return np.arccos(np.clip(cos_gamma, -1.0, 1.0))


# ----------------------------------------------------------------------------------------
