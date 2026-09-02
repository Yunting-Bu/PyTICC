from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import cast

import jax
import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.atom_triatom as vmat
from pyticc._typing import JaxDevice
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import ChannelBasis
from pyticc.basis.monomer import AtomSpec
from pyticc.basis.triatom import TriatomBasis
from pyticc.matrix.interaction import VBasisDevice, contract, contract_device, device_basis
from pyticc.pes.adiabatic import PESWrapper, get_Vgrid_atom_triatom
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.potential import PotentialGrid, _potential_radial_grid, _require_type
from pyticc.system import ScatteringType, ScattSystem

_SCATTERING_TYPE = ScatteringType.ATOM_TRIATOM


# ----------------------------------------------------------------------------------------
def prepare_potential(
    system: ScattSystem,
    boundaries: Sequence[float],
    half_steps: Sequence[float],
    *,
    n_theta_1: int | None = None,
    n_theta_2: int = 16,
    n_phi: int = 16,
    processes: int = 1,
) -> PotentialGrid:
    """Evaluate an atom-triatom PES on the complete propagation grid.

    Inputs:
        system: ScattSystem - prepared atom-triatom scattering system
        boundaries: Sequence[float] - radial interval boundaries in bohr
        half_steps: Sequence[float] - propagation half-step in each interval
        n_theta_1: int | None - optional internal-bend quadrature order
        n_theta_2: int - external polar-angle quadrature order
        n_phi: int - dihedral-angle quadrature order
        processes: int - worker processes used for the radial PES batch

    Returns:
        potential_grid: PotentialGrid - raw interaction values and quadrature data
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, TriatomBasis):
        message = "Atom-triatom potential preparation requires AtomSpec and TriatomBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper):
        message = "Atom-triatom potential preparation requires a scalar PES"
        logger.error(message)
        raise TypeError(message)

    triatom = system.monomer_Y
    if n_theta_1 is None:
        if triatom.cos_theta is None or triatom.theta_weights is None:
            message = "TriatomBasis has no stored bending quadrature; provide n_theta_1"
            logger.error(message)
            raise ValueError(message)
        cos_theta_1 = triatom.cos_theta
        theta_weights_1 = triatom.theta_weights
    else:
        cos_theta_1, theta_weights_1 = gauss_legendre_dvr(-1.0, 1.0, n_theta_1)
    cos_theta_2, theta_weights_2 = gauss_legendre_dvr(-1.0, 1.0, n_theta_2)
    phi, phi_weights = gauss_legendre_dvr(0.0, np.pi, n_phi)
    radial_1 = triatom.radial_1
    radial_2 = triatom.radial_2
    if radial_1 is None or radial_2 is None:
        message = "Atom-triatom scattering requires radial PODVR data in TriatomBasis"
        logger.error(message)
        raise ValueError(message)

    boundaries_value, half_steps_value, sectors, radial_points = _potential_radial_grid(boundaries, half_steps)
    values = get_Vgrid_atom_triatom(
        system.potential,
        radial_points,
        radial_1.grids,
        radial_2.grids,
        np.arccos(cos_theta_1),
        np.arccos(cos_theta_2),
        phi,
        processes=processes,
    )
    return PotentialGrid(
        boundaries=boundaries_value,
        half_steps=half_steps_value,
        sectors=sectors,
        radial_points=radial_points,
        scattering_type=_SCATTERING_TYPE,
        coordinates=(("r_1", radial_1.grids), ("r_2", radial_2.grids), ("cos_theta_1", cos_theta_1), ("cos_theta_2", cos_theta_2), ("phi", phi)),
        weights=(("theta_1", theta_weights_1), ("theta_2", theta_weights_2), ("phi", phi_weights)),
        values=values,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_hamiltonian(
    system: ScattSystem,
    *,
    n_theta_1: int | None = None,
    n_theta_2: int = 16,
    n_phi: int = 16,
    potential_grid: PotentialGrid | None = None,
) -> ScattHamiltonian:
    """Build an atom-triatom scattering Hamiltonian.

    Inputs:
        system: ScattSystem - prepared atom-triatom scattering system
        n_theta_1: int | None - optional internal-bend quadrature order
        n_theta_2: int - external polar-angle quadrature order
        n_phi: int - dihedral-angle quadrature order
        potential_grid: PotentialGrid | None - optional precomputed raw PES grid

    Returns:
        hamiltonian: ScattHamiltonian - projected channel Hamiltonian
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, TriatomBasis):
        message = "Atom-triatom Hamiltonian requires AtomSpec as monomer_X and TriatomBasis as monomer_Y"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper):
        message = "Atom-triatom Hamiltonian requires a PESWrapper"
        logger.error(message)
        raise TypeError(message)
    if system.reduced_mass is None:
        message = "Atom-triatom Hamiltonian requires a collision reduced mass"
        logger.error(message)
        raise ValueError(message)
    if not isinstance(system.basis, ChannelBasis):
        message = "Atom-triatom Hamiltonian requires channels prepared by build_ScattSystem"
        logger.error(message)
        raise TypeError(message)

    triatom = system.monomer_Y
    pes = system.potential
    basis = system.basis

    if potential_grid is not None:
        _require_type(potential_grid, _SCATTERING_TYPE)
        cos_theta_1 = potential_grid.coordinate("cos_theta_1")
        theta_weights_1 = potential_grid.weight("theta_1")
        cos_theta_2 = potential_grid.coordinate("cos_theta_2")
        theta_weights_2 = potential_grid.weight("theta_2")
        phi = potential_grid.coordinate("phi")
        phi_weights = potential_grid.weight("phi")
    else:
        if n_theta_1 is None:
            if triatom.cos_theta is None or triatom.theta_weights is None:
                message = "TriatomBasis has no stored bending quadrature; provide n_theta_1"
                logger.error(message)
                raise ValueError(message)
            cos_theta_1 = triatom.cos_theta
            theta_weights_1 = triatom.theta_weights
        else:
            cos_theta_1, theta_weights_1 = gauss_legendre_dvr(-1.0, 1.0, n_theta_1)
        cos_theta_2, theta_weights_2 = gauss_legendre_dvr(-1.0, 1.0, n_theta_2)
        phi, phi_weights = gauss_legendre_dvr(0.0, np.pi, n_phi)
    theta_1 = np.arccos(cos_theta_1)
    theta_2 = np.arccos(cos_theta_2)
    V_basis = vmat.prepare(
        basis,
        triatom,
        cos_theta_1,
        theta_weights_1,
        cos_theta_2,
        theta_weights_2,
        phi,
        phi_weights,
    )
    device_bases: dict[tuple[str, int], VBasisDevice] = {}

    radial_1 = triatom.radial_1
    radial_2 = triatom.radial_2
    if radial_1 is None or radial_2 is None:
        message = "Atom-triatom scattering requires radial PODVR data in TriatomBasis"
        logger.error(message)
        raise ValueError(message)

    def Vgrid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the atom-triatom PES grid."""
        if potential_grid is not None:
            return cast(NDArray[np.float64], potential_grid.take(radial_points))
        return get_Vgrid_atom_triatom(pes, radial_points, radial_1.grids, radial_2.grids, theta_1, theta_2, phi)

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
