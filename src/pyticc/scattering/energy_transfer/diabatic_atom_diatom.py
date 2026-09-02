from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import jax
import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.diabatic_atom_diatom as vmat
from pyticc._typing import JaxDevice
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import ChannelBasis
from pyticc.basis.monomer import AtomSpec, DiabaticDiatomBasis
from pyticc.pes.diabatic import DiabaticPESWrapper, get_diabatic_potential_grid_atom_diatom
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.potential import PotentialGrid, _potential_radial_grid, _require_type
from pyticc.system import Approx, ScatteringType, ScattSystem

_SCATTERING_TYPE = ScatteringType.ATOM_DIATOM_DIABATIC


# ----------------------------------------------------------------------------------------
def prepare_potential(
    system: ScattSystem,
    boundaries: Sequence[float],
    half_steps: Sequence[float],
    *,
    n_theta: int = 16,
    processes: int = 1,
) -> PotentialGrid:
    """Evaluate a diabatic atom-diatom PES on the complete propagation grid.

    Inputs:
        system: ScattSystem - prepared diabatic atom-diatom scattering system
        boundaries: Sequence[float] - radial interval boundaries in bohr
        half_steps: Sequence[float] - propagation half-step in each interval
        n_theta: int - retained Jacobi-angle quadrature points
        processes: int - worker processes used for the radial PES batch

    Returns:
        potential_grid: PotentialGrid - state-resolved raw PES values and grids
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, DiabaticDiatomBasis):
        message = "Diabatic potential preparation requires AtomSpec and DiabaticDiatomBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, DiabaticPESWrapper) or not isinstance(system.basis, ChannelBasis):
        message = "Diabatic potential preparation requires a diabatic PES and prepared channels"
        logger.error(message)
        raise TypeError(message)

    diatom = system.monomer_Y
    exchange_parity = system.basis.channel_spec.exchange_parity_Y
    exchange_parities = (exchange_parity,) if isinstance(exchange_parity, int) else exchange_parity
    cos_theta, theta_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta, symmetry=all(parity != 0 for parity in exchange_parities))
    theta = np.arccos(cos_theta)
    diagonal_grids = tuple(state.contracted.grids for state in diatom.states)
    coupling_grid = diatom.states[0].primitive.grids
    radial_grids = (*diagonal_grids, coupling_grid)
    radial_sizes = tuple(grid.size for grid in radial_grids)
    combined_grid = np.concatenate(radial_grids)
    boundaries_value, half_steps_value, sectors, radial_points = _potential_radial_grid(boundaries, half_steps)
    combined = get_diabatic_potential_grid_atom_diatom(
        system.potential,
        radial_points,
        combined_grid,
        theta,
        processes=processes,
    )
    sampled = np.split(combined, np.cumsum(radial_sizes[:-1]), axis=1)
    values = vmat.DiabaticVGridBF(
        diagonal=tuple(sampled[state][..., state, state] for state in range(diatom.n_state)),
        coupling=sampled[-1],
    )
    coordinates = tuple((f"diagonal_r_{state}", grid) for state, grid in enumerate(diagonal_grids))
    return PotentialGrid(
        boundaries=boundaries_value,
        half_steps=half_steps_value,
        sectors=sectors,
        radial_points=radial_points,
        scattering_type=_SCATTERING_TYPE,
        coordinates=(*coordinates, ("coupling_r", coupling_grid), ("cos_theta", cos_theta)),
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
    """Build a diabatic atom-diatom scattering Hamiltonian.

    Inputs:
        system: ScattSystem - prepared diabatic atom-diatom scattering system
        n_theta: int - retained Jacobi-angle quadrature points
        potential_grid: PotentialGrid | None - optional precomputed raw PES grid

    Returns:
        hamiltonian: ScattHamiltonian - exact diabatic channel Hamiltonian
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, DiabaticDiatomBasis):
        message = "Diabatic atom-diatom Hamiltonian requires AtomSpec and DiabaticDiatomBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, DiabaticPESWrapper):
        message = "Diabatic atom-diatom Hamiltonian requires a DiabaticPESWrapper"
        logger.error(message)
        raise TypeError(message)
    if system.approx is not Approx.EXACT:
        message = "Diabatic atom-diatom Hamiltonian currently requires approx='exact'"
        logger.error(message)
        raise ValueError(message)
    if system.reduced_mass is None:
        message = "Diabatic atom-diatom Hamiltonian requires a collision reduced mass"
        logger.error(message)
        raise ValueError(message)
    if not isinstance(system.basis, ChannelBasis):
        message = "Diabatic atom-diatom Hamiltonian requires channels prepared by build_ScattSystem"
        logger.error(message)
        raise TypeError(message)

    diatom = system.monomer_Y
    pes = system.potential
    if pes.n_state != diatom.n_state:
        message = f"PES has {pes.n_state} electronic states, but the diatomic basis has {diatom.n_state}"
        logger.error(message)
        raise ValueError(message)
    basis = system.basis
    exchange_parity = basis.channel_spec.exchange_parity_Y
    exchange_parities = (exchange_parity,) if isinstance(exchange_parity, int) else exchange_parity
    angular_symmetry = all(parity != 0 for parity in exchange_parities)
    if potential_grid is None:
        cos_theta, theta_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta, symmetry=angular_symmetry)
    else:
        _require_type(potential_grid, _SCATTERING_TYPE)
        cos_theta = potential_grid.coordinate("cos_theta")
        theta_weights = potential_grid.weight("theta")
    V_basis = vmat.prepare(basis, diatom, cos_theta, theta_weights)
    device_bases: dict[tuple[str, int], vmat.DiabaticVBasisDevice] = {}

    def Vgrid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> vmat.DiabaticVGridBF:
        """Return sampled diabatic values on all required radial grids."""
        if potential_grid is not None:
            return cast(vmat.DiabaticVGridBF, potential_grid.take(radial_points))
        return vmat.sample(pes, radial_points, V_basis)

    def Vmat(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Sample and contract the diabatic potential."""
        return vmat.contract(V_basis, Vgrid(radial_points))

    def V_blocks_device(radial_points: NDArray[np.float64], channel_blocks: tuple[tuple[int, ...], ...], device: JaxDevice) -> tuple[jax.Array, ...]:
        """Contract requested blocks from host or device-resident diabatic values."""
        key = (device.platform, device.id)
        if key not in device_bases:
            device_bases[key] = vmat.device_basis(V_basis, device)
        values = Vgrid(radial_points) if potential_grid is None else cast(vmat.DiabaticVGridBF, potential_grid.take_device(radial_points, device))
        return tuple(vmat.contract_device(V_basis, device_bases[key], values, device, indices) for indices in channel_blocks)

    potential_grid_size = V_basis.theta.size * (sum(grid.size for grid in V_basis.diagonal_grids) + V_basis.coupling_grid.size) * diatom.n_state**2
    return ScattHamiltonian(
        basis=basis,
        reduced_mass=system.reduced_mass,
        interaction=Vmat,
        device_block_interaction=V_blocks_device,
        potential_grid_size=potential_grid_size,
    )


# ----------------------------------------------------------------------------------------
