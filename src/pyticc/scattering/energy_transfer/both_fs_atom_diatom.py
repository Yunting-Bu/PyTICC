from collections.abc import Sequence
from math import prod
from typing import cast

import jax
import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.fs_both_atom_diatom as vmat
from pyticc._typing import JaxDevice
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.fine_structure.atom import FSAtomBasis
from pyticc.fine_structure.atom_diatom import FSAtomDiatomBasis
from pyticc.fine_structure.channel import FSMonomerBasis
from pyticc.pes.spin_resolved_atom_diatom import (
    RadialInput,
    SpinResolvedAtomDiatomPES,
    get_spin_resolved_grid_atom_diatom,
)
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.potential import PotentialGrid, _potential_radial_grid, _require_type
from pyticc.system import ScatteringType, ScattSystem

_SCATTERING_TYPE = ScatteringType.ATOM_DIATOM_BOTH_FS


# ----------------------------------------------------------------------------------------
def _spin_coordinates(potential: SpinResolvedAtomDiatomPES) -> tuple[tuple[str, NDArray[np.float64]], ...]:
    """Record ordered electronic-axis labels alongside the geometric grid."""
    return (
        ("two_total_spins", np.asarray(potential.two_total_spins, dtype=np.float64)),
        ("orbital_two_lambda_atom", np.asarray([s.two_lambda_atom for s in potential.orbital_states], dtype=np.float64)),
        ("orbital_two_lambda_Y", np.asarray([s.two_lambda_Y for s in potential.orbital_states], dtype=np.float64)),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare_potential(
    system: ScattSystem,
    boundaries: Sequence[float],
    half_steps: Sequence[float],
    *,
    n_theta: int = 24,
    processes: int = 1,
) -> PotentialGrid:
    """Evaluate a total-spin-resolved A+BC PES on the complete propagation grid.

    Inputs:
        system: ScattSystem - prepared fine-structure atom plus fine-structure
            diatom system
        boundaries: Sequence[float] - radial interval boundaries in bohr
        half_steps: Sequence[float] - propagation half-step in each interval
        n_theta: int - full Gauss-Legendre angular order
        processes: int - worker processes used for the radial PES batch

    Returns:
        potential_grid: PotentialGrid - raw spin-resolved orbital values with
            coordinates ordered as ``(r,cos_theta)``
    """
    if not isinstance(system.monomer_X, FSAtomBasis) or not isinstance(system.monomer_Y, FSMonomerBasis):
        message = "Both-fine-structure potential preparation requires FSAtomBasis and FSMonomerBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, SpinResolvedAtomDiatomPES):
        message = "Both-fine-structure potential preparation requires a SpinResolvedAtomDiatomPES"
        logger.error(message)
        raise TypeError(message)

    cos_theta, weights = gauss_legendre_dvr(-1.0, 1.0, n_theta)
    boundaries_value, half_steps_value, sectors, radial_points = _potential_radial_grid(boundaries, half_steps)
    radial = system.monomer_Y.vib.grids
    values = get_spin_resolved_grid_atom_diatom(system.potential, radial_points, radial, np.arccos(cos_theta), processes=processes)
    return PotentialGrid(
        boundaries=boundaries_value,
        half_steps=half_steps_value,
        sectors=sectors,
        radial_points=radial_points,
        scattering_type=_SCATTERING_TYPE,
        coordinates=(("r", radial), ("cos_theta", cos_theta)) + _spin_coordinates(system.potential),
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
    Build a BF Hamiltonian for a fine-structure atom plus fine-structure diatom.

    Inputs:
        system: ScattSystem - prepared channels, FS atom and FS diatom monomers,
            spin-resolved PES, and collision reduced mass
        n_theta: int - full Gauss-Legendre angular order
        potential_grid: PotentialGrid | None - optional precomputed raw PES grid

    Returns:
        hamiltonian: ScattHamiltonian - fine-structure coupled-channel Hamiltonian
            for exact CC, CS, or NNCC
    """
    if not isinstance(system.monomer_X, FSAtomBasis) or not isinstance(system.monomer_Y, FSMonomerBasis):
        message = "Both-fine-structure Hamiltonian requires FSAtomBasis and FSMonomerBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, SpinResolvedAtomDiatomPES):
        message = "Both-fine-structure Hamiltonian requires a SpinResolvedAtomDiatomPES"
        logger.error(message)
        raise TypeError(message)
    if system.reduced_mass is None:
        message = "Both-fine-structure Hamiltonian requires a collision reduced mass"
        logger.error(message)
        raise ValueError(message)
    if not isinstance(system.basis, FSAtomDiatomBasis):
        message = "Both-fine-structure Hamiltonian requires channels prepared by build_ScattSystem"
        logger.error(message)
        raise TypeError(message)

    basis = system.basis
    potential = system.potential
    if basis.atom is not system.monomer_X or basis.monomer_Y is not system.monomer_Y:
        raise ValueError("System and channel basis must use the same monomer objects")
    if potential_grid is None:
        cos_theta, weights = gauss_legendre_dvr(-1.0, 1.0, n_theta)
    else:
        _require_type(potential_grid, _SCATTERING_TYPE)
        cos_theta = potential_grid.coordinate("cos_theta")
        weights = potential_grid.weight("theta")
    theta = np.arccos(cos_theta)
    V_basis = vmat.prepare(basis, potential.two_total_spins, potential.orbital_states, theta, weights)
    device_bases: dict[tuple[str, int], vmat.SpinResolvedFSBothAtomDiatomVBasisDevice] = {}

    def Vgrid(radial_points: RadialInput) -> NDArray[np.float64] | NDArray[np.complex128]:
        """Return spin-resolved orbital values on the prepared internal grid."""
        if potential_grid is not None:
            return cast(NDArray[np.float64] | NDArray[np.complex128], potential_grid.take(radial_points))
        return get_spin_resolved_grid_atom_diatom(potential, radial_points, basis.monomer_Y.vib.grids, theta)

    def Vmat(radial_points: RadialInput) -> NDArray[np.float64] | NDArray[np.complex128]:
        """Contract spin-resolved orbital components into the channel basis."""
        return vmat.contract(V_basis, Vgrid(radial_points))

    def V_blocks(
        radial_points: NDArray[np.float64], channel_blocks: tuple[tuple[int, ...], ...]
    ) -> tuple[NDArray[np.float64] | NDArray[np.complex128], ...]:
        """Contract only the requested channel windows from one shared PES batch."""
        values = Vgrid(radial_points)
        return tuple(vmat.contract(V_basis, values, indices) for indices in channel_blocks)

    def V_blocks_device(radial_points: np.ndarray, channel_blocks: tuple[tuple[int, ...], ...], device: JaxDevice) -> tuple[jax.Array, ...]:
        """Contract FS blocks from host or device-resident spin-resolved PES values."""
        key = (device.platform, device.id)
        if key not in device_bases:
            device_bases[key] = vmat.device_basis(V_basis, device)
        values = (
            Vgrid(radial_points)
            if potential_grid is None
            else cast(NDArray[np.float64] | NDArray[np.complex128] | jax.Array, potential_grid.take_device(radial_points, device))
        )
        return tuple(vmat.contract_device(V_basis, device_bases[key], values, device, indices) for indices in channel_blocks)

    return ScattHamiltonian(
        basis=basis,
        reduced_mass=system.reduced_mass,
        interaction=Vmat,
        approx=system.approx,
        K_delta=system.K_delta,
        block_interaction=V_blocks,
        device_block_interaction=V_blocks_device,
        potential_grid_size=prod(V_basis.grid_shape) * prod(V_basis.electronic_shape),
    )


# ----------------------------------------------------------------------------------------
