from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import cast

import jax
import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.diatom_diatom as vmat
from pyticc._typing import JaxDevice
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import ChannelBasis
from pyticc.basis.monomer import DiatomBasis
from pyticc.matrix.interaction import VBasisDevice, contract, contract_device, device_basis
from pyticc.pes.adiabatic import PESWrapper, get_Vgrid_diatom_diatom
from pyticc.pes.molecule_exchange import validate_exchange_potential, validate_exchange_quadrature
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.potential import PotentialGrid, _potential_radial_grid, _require_type
from pyticc.system import ScatteringType, ScattSystem

_SCATTERING_TYPE = ScatteringType.DIATOM_DIATOM


# ----------------------------------------------------------------------------------------
def prepare_potential(
    system: ScattSystem,
    boundaries: Sequence[float],
    half_steps: Sequence[float],
    *,
    n_theta_X: int = 15,
    n_theta_Y: int = 15,
    n_phi: int = 12,
    processes: int = 1,
) -> PotentialGrid:
    """Evaluate a diatom-diatom PES on the complete propagation grid.

    Inputs:
        system: ScattSystem - prepared diatom-diatom scattering system
        boundaries: Sequence[float] - radial interval boundaries in bohr
        half_steps: Sequence[float] - propagation half-step in each interval
        n_theta_X: int - first polar-angle quadrature order
        n_theta_Y: int - second polar-angle quadrature order
        n_phi: int - dihedral-angle quadrature order
        processes: int - worker processes used for the radial PES batch

    Returns:
        potential_grid: PotentialGrid - raw interaction values and quadrature data
    """
    if not isinstance(system.monomer_X, DiatomBasis) or not isinstance(system.monomer_Y, DiatomBasis):
        message = "Diatom-diatom potential preparation requires two DiatomBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper):
        message = "Diatom-diatom potential preparation requires a scalar PES"
        logger.error(message)
        raise TypeError(message)

    cos_theta_X, theta_weights_X = gauss_legendre_dvr(-1.0, 1.0, n_theta_X)
    cos_theta_Y, theta_weights_Y = gauss_legendre_dvr(-1.0, 1.0, n_theta_Y)
    phi, phi_weights = gauss_legendre_dvr(0.0, np.pi, n_phi)
    boundaries_value, half_steps_value, sectors, radial_points = _potential_radial_grid(boundaries, half_steps)
    radial_X = system.monomer_X.rovib.grids
    radial_Y = system.monomer_Y.rovib.grids
    if system.molecule_exchange:
        validate_exchange_quadrature(radial_X, radial_Y, cos_theta_X, cos_theta_Y, theta_weights_X, theta_weights_Y)
    values = get_Vgrid_diatom_diatom(
        system.potential,
        radial_points,
        radial_X,
        radial_Y,
        np.arccos(cos_theta_X),
        np.arccos(cos_theta_Y),
        phi,
        processes=processes,
    )
    if system.molecule_exchange:
        validate_exchange_potential(values)
    return PotentialGrid(
        boundaries=boundaries_value,
        half_steps=half_steps_value,
        sectors=sectors,
        radial_points=radial_points,
        scattering_type=_SCATTERING_TYPE,
        coordinates=(("r_X", radial_X), ("r_Y", radial_Y), ("cos_theta_X", cos_theta_X), ("cos_theta_Y", cos_theta_Y), ("phi", phi)),
        weights=(("theta_X", theta_weights_X), ("theta_Y", theta_weights_Y), ("phi", phi_weights)),
        values=values,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_hamiltonian(
    system: ScattSystem,
    *,
    n_theta_X: int = 15,
    n_theta_Y: int = 15,
    n_phi: int = 12,
    potential_grid: PotentialGrid | None = None,
) -> ScattHamiltonian:
    """Build a diatom-diatom scattering Hamiltonian.

    Inputs:
        system: ScattSystem - prepared diatom-diatom scattering system
        n_theta_X: int - first polar-angle quadrature order
        n_theta_Y: int - second polar-angle quadrature order
        n_phi: int - dihedral-angle quadrature order
        potential_grid: PotentialGrid | None - optional precomputed raw PES grid

    Returns:
        hamiltonian: ScattHamiltonian - projected channel Hamiltonian
    """
    if not isinstance(system.monomer_X, DiatomBasis) or not isinstance(system.monomer_Y, DiatomBasis):
        message = "Diatom-diatom Hamiltonian requires two DiatomBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper):
        message = "Diatom-diatom Hamiltonian requires a PESWrapper"
        logger.error(message)
        raise TypeError(message)
    if system.reduced_mass is None:
        message = "Diatom-diatom Hamiltonian requires a collision reduced mass"
        logger.error(message)
        raise ValueError(message)
    if not isinstance(system.basis, ChannelBasis):
        message = "Diatom-diatom Hamiltonian requires channels prepared by build_ScattSystem"
        logger.error(message)
        raise TypeError(message)

    pes = system.potential
    rovib_X = system.monomer_X.rovib
    rovib_Y = system.monomer_Y.rovib
    basis = system.basis
    if potential_grid is None:
        cos_theta_X, theta_weights_X = gauss_legendre_dvr(-1.0, 1.0, n_theta_X)
        cos_theta_Y, theta_weights_Y = gauss_legendre_dvr(-1.0, 1.0, n_theta_Y)
        phi, phi_weights = gauss_legendre_dvr(0.0, np.pi, n_phi)
    else:
        _require_type(potential_grid, _SCATTERING_TYPE)
        cos_theta_X = potential_grid.coordinate("cos_theta_X")
        cos_theta_Y = potential_grid.coordinate("cos_theta_Y")
        phi = potential_grid.coordinate("phi")
        theta_weights_X = potential_grid.weight("theta_X")
        theta_weights_Y = potential_grid.weight("theta_Y")
        phi_weights = potential_grid.weight("phi")
    theta_X = np.arccos(cos_theta_X)
    theta_Y = np.arccos(cos_theta_Y)
    if basis.molecule_exchange != system.molecule_exchange:
        raise ValueError("System and channel basis have different molecule_exchange settings")
    if system.molecule_exchange:
        validate_exchange_quadrature(rovib_X.grids, rovib_Y.grids, cos_theta_X, cos_theta_Y, theta_weights_X, theta_weights_Y)
        if potential_grid is not None:
            for name in ("r_X", "r_Y"):
                radial = potential_grid.coordinate(name)
                if radial.shape != rovib_X.grids.shape or not np.allclose(radial, rovib_X.grids, rtol=0.0, atol=1.0e-14):
                    raise ValueError("Molecule-exchange cached radial grids must match the shared monomer basis")
            validate_exchange_potential(cast(NDArray[np.float64], potential_grid.values))
    V_basis = vmat.prepare(
        basis,
        rovib_X,
        rovib_Y,
        cos_theta_X,
        theta_weights_X,
        cos_theta_Y,
        theta_weights_Y,
        phi,
        phi_weights,
    )
    device_bases: dict[tuple[str, int], VBasisDevice] = {}

    def Vgrid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the diatom-diatom PES grid."""
        if potential_grid is not None:
            return cast(NDArray[np.float64], potential_grid.take(radial_points))
        values = get_Vgrid_diatom_diatom(pes, radial_points, rovib_X.grids, rovib_Y.grids, theta_X, theta_Y, phi)
        if system.molecule_exchange:
            validate_exchange_potential(values)
        return values

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
