from collections.abc import Sequence
from math import prod
from typing import cast

import jax
import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.fs_atom_diatom as vmat
from pyticc._typing import JaxDevice
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.monomer import AtomSpec
from pyticc.fine_structure.channel import FSChannelBasis, FSMonomerBasis
from pyticc.pes.lambda_pes import LambdaPES, RadialInput, get_lambda_grid_atom_diatom
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.potential import PotentialGrid, _potential_radial_grid, _require_type
from pyticc.system import Approx, ScatteringType, ScattSystem

_SCATTERING_TYPE = ScatteringType.ATOM_DIATOM_FINE_STRUCTURE


# ----------------------------------------------------------------------------------------
def prepare_potential(
    system: ScattSystem,
    boundaries: Sequence[float],
    half_steps: Sequence[float],
    *,
    n_theta: int = 24,
    processes: int = 1,
) -> PotentialGrid:
    """Evaluate a signed-Lambda PES on the complete propagation grid.

    Inputs:
        system: ScattSystem - prepared fine-structure atom-diatom system
        boundaries: Sequence[float] - radial interval boundaries in bohr
        half_steps: Sequence[float] - propagation half-step in each interval
        n_theta: int - full Gauss-Legendre angular order
        processes: int - worker processes used for the radial PES batch

    Returns:
        potential_grid: PotentialGrid - raw Lambda components and quadrature data
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, FSMonomerBasis):
        message = "Fine-structure potential preparation requires AtomSpec and FSMonomerBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, LambdaPES):
        message = "Fine-structure potential preparation requires a LambdaPES"
        logger.error(message)
        raise TypeError(message)

    cos_theta, weights = gauss_legendre_dvr(-1.0, 1.0, n_theta)
    boundaries_value, half_steps_value, sectors, radial_points = _potential_radial_grid(boundaries, half_steps)
    radial = system.monomer_Y.vib.grids
    values = get_lambda_grid_atom_diatom(system.potential, radial_points, radial, np.arccos(cos_theta), processes=processes)
    return PotentialGrid(
        boundaries=boundaries_value,
        half_steps=half_steps_value,
        sectors=sectors,
        radial_points=radial_points,
        scattering_type=_SCATTERING_TYPE,
        coordinates=(("r", radial), ("cos_theta", cos_theta)),
        weights=(("theta", weights),),
        values=values,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_hamiltonian(
    system: ScattSystem,
    *,
    n_theta: int = 24,
    potential_grid: PotentialGrid | None = None,
) -> ScattHamiltonian:
    """
    Build an exact BF atom-diatom Hamiltonian in the FS basis.

    Inputs:
        system: ScattSystem - atom plus fine-structure diatom system containing
            prepared channels, LambdaPES, and collision reduced mass
        n_theta: int - full Gauss-Legendre angular order
        potential_grid: PotentialGrid | None - optional precomputed raw PES grid

    Returns:
        hamiltonian: ScattHamiltonian - exact coupled-channel Hamiltonian
    """
    if not isinstance(system.monomer_X, AtomSpec) or not isinstance(system.monomer_Y, FSMonomerBasis):
        message = "Fine-structure atom-diatom Hamiltonian requires AtomSpec and FSMonomerBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, LambdaPES):
        message = "Fine-structure atom-diatom Hamiltonian requires a LambdaPES"
        logger.error(message)
        raise TypeError(message)
    if system.reduced_mass is None:
        message = "Fine-structure atom-diatom Hamiltonian requires a collision reduced mass"
        logger.error(message)
        raise ValueError(message)
    if system.approx is not Approx.EXACT:
        message = "Fine-structure atom-diatom Hamiltonian currently requires approx='exact'"
        logger.error(message)
        raise ValueError(message)
    if not isinstance(system.basis, FSChannelBasis):
        message = "Fine-structure atom-diatom Hamiltonian requires channels prepared by build_ScattSystem"
        logger.error(message)
        raise TypeError(message)

    basis = system.basis
    potential = system.potential
    if potential_grid is None:
        cos_theta, weights = gauss_legendre_dvr(-1.0, 1.0, n_theta)
    else:
        _require_type(potential_grid, _SCATTERING_TYPE)
        cos_theta = potential_grid.coordinate("cos_theta")
        weights = potential_grid.weight("theta")
    theta = np.arccos(cos_theta)
    V_basis = vmat.prepare(basis, theta, weights)
    device_bases: dict[tuple[str, int], vmat.FSVBasisDevice] = {}

    def Vgrid(radial_points: RadialInput) -> NDArray[np.float64]:
        """Return signed-Lambda values on the prepared internal grid."""
        if potential_grid is not None:
            return cast(NDArray[np.float64], potential_grid.take(radial_points))
        return get_lambda_grid_atom_diatom(potential, radial_points, basis.monomer.vib.grids, theta)

    def Vmat(radial_points: RadialInput) -> NDArray[np.float64]:
        """Contract signed-Lambda components into the FS channel basis."""
        return vmat.contract(V_basis, Vgrid(radial_points))

    def V_blocks_device(radial_points: np.ndarray, channel_blocks: tuple[tuple[int, ...], ...], device: JaxDevice) -> tuple[jax.Array, ...]:
        """Contract FS blocks from host or device-resident Lambda PES values."""
        key = (device.platform, device.id)
        if key not in device_bases:
            device_bases[key] = vmat.device_basis(V_basis, device)
        values = (
            Vgrid(radial_points)
            if potential_grid is None
            else cast(NDArray[np.float64] | jax.Array, potential_grid.take_device(radial_points, device))
        )
        return tuple(vmat.contract_device(V_basis, device_bases[key], values, device, indices) for indices in channel_blocks)

    return ScattHamiltonian(
        basis=basis,
        reduced_mass=system.reduced_mass,
        interaction=Vmat,
        approx=Approx.EXACT,
        device_block_interaction=V_blocks_device,
        potential_grid_size=prod(V_basis.grid_shape),
    )


# ----------------------------------------------------------------------------------------
