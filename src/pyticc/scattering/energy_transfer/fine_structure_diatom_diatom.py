from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import cast

import jax
import numpy as np
from loguru import logger
from numpy.typing import NDArray

import pyticc.matrix.interaction.fs_diatom_diatom as scalar_vmat
import pyticc.matrix.interaction.fs_diatom_diatom_spin as spin_vmat
from pyticc._typing import JaxDevice
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.fine_structure.channel import FSMonomerBasis
from pyticc.fine_structure.diatom_diatom import FSDiatomDiatomBasis
from pyticc.pes.adiabatic import PESWrapper, get_Vgrid_diatom_diatom
from pyticc.pes.spin_resolved_diatom_diatom import SpinResolvedDiatomDiatomPES, get_spin_resolved_grid_diatom_diatom
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.potential import PotentialGrid, _potential_radial_grid, _require_type
from pyticc.system import Approx, ScatteringType, ScattSystem

_SCATTERING_TYPE = ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE


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
    """Evaluate a scalar or total-spin-resolved AB+CD PES on the propagation grid.

    Inputs:
        system: ScattSystem - prepared two-fine-structure-diatom system
        boundaries: Sequence[float] - radial interval boundaries in bohr
        half_steps: Sequence[float] - propagation half-step in each interval
        n_theta_X: int - monomer-X polar quadrature order
        n_theta_Y: int - monomer-Y polar quadrature order
        n_phi: int - torsional quadrature order on ``[0,pi]``
        processes: int - worker processes used for the radial PES batch

    Returns:
        potential_grid: PotentialGrid - raw interaction values with
            coordinates ordered as ``(r_X,r_Y,theta_X,theta_Y,phi)``
    """
    if not isinstance(system.monomer_X, FSMonomerBasis) or not isinstance(system.monomer_Y, FSMonomerBasis):
        message = "Fine-structure diatom-diatom potential preparation requires two FSMonomerBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper | SpinResolvedDiatomDiatomPES):
        message = "Fine-structure diatom-diatom preparation requires a scalar or spin-resolved PES"
        logger.error(message)
        raise TypeError(message)

    cos_theta_X, theta_weights_X = gauss_legendre_dvr(-1.0, 1.0, n_theta_X)
    cos_theta_Y, theta_weights_Y = gauss_legendre_dvr(-1.0, 1.0, n_theta_Y)
    phi, phi_weights = gauss_legendre_dvr(0.0, np.pi, n_phi)
    boundaries_value, half_steps_value, sectors, radial_points = _potential_radial_grid(boundaries, half_steps)
    radial_X = system.monomer_X.vib.grids
    radial_Y = system.monomer_Y.vib.grids
    if isinstance(system.potential, PESWrapper):
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
    else:
        values = get_spin_resolved_grid_diatom_diatom(
            system.potential,
            radial_points,
            radial_X,
            radial_Y,
            np.arccos(cos_theta_X),
            np.arccos(cos_theta_Y),
            phi,
            processes=processes,
        )
    return PotentialGrid(
        boundaries=boundaries_value,
        half_steps=half_steps_value,
        sectors=sectors,
        radial_points=radial_points,
        scattering_type=_SCATTERING_TYPE,
        coordinates=(
            ("r_X", radial_X),
            ("r_Y", radial_Y),
            ("cos_theta_X", cos_theta_X),
            ("cos_theta_Y", cos_theta_Y),
            ("phi", phi),
        ),
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
    """Build an exact scalar or total-spin-resolved two-FS-diatom Hamiltonian.

    Inputs:
        system: ScattSystem - prepared channels, two FS monomers, interaction PES,
            and collision reduced mass
        n_theta_X: int - monomer-X polar quadrature order when no grid is given
        n_theta_Y: int - monomer-Y polar quadrature order when no grid is given
        n_phi: int - torsional quadrature order when no grid is given
        potential_grid: PotentialGrid | None - optional precomputed PES grid

    Returns:
        hamiltonian: ScattHamiltonian - exact BF coupled-channel Hamiltonian
    """
    if not isinstance(system.monomer_X, FSMonomerBasis) or not isinstance(system.monomer_Y, FSMonomerBasis):
        message = "Fine-structure diatom-diatom Hamiltonian requires two FSMonomerBasis monomers"
        logger.error(message)
        raise TypeError(message)
    if not isinstance(system.potential, PESWrapper | SpinResolvedDiatomDiatomPES):
        message = "Fine-structure diatom-diatom Hamiltonian requires a scalar or spin-resolved PES"
        logger.error(message)
        raise TypeError(message)
    if system.reduced_mass is None:
        message = "Fine-structure diatom-diatom Hamiltonian requires a collision reduced mass"
        logger.error(message)
        raise ValueError(message)
    if system.approx is not Approx.EXACT:
        message = "Fine-structure diatom-diatom Hamiltonian currently requires approx='exact'"
        logger.error(message)
        raise ValueError(message)
    if not isinstance(system.basis, FSDiatomDiatomBasis):
        message = "Fine-structure diatom-diatom Hamiltonian requires channels prepared by build_ScattSystem"
        logger.error(message)
        raise TypeError(message)

    basis = system.basis
    potential = system.potential
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
    dipole_matrix = spin_vmat.magnetic_dipole_matrix(
        basis,
        theta_X,
        theta_weights_X,
        theta_Y,
        theta_weights_Y,
        phi,
        phi_weights,
    )
    dipole_coefficient = system.magnetic_dipole_coefficient

    def add_magnetic_dipole(
        matrix: NDArray[np.float64] | NDArray[np.complex128],
        radial_points: float | Sequence[float] | NDArray[np.float64],
    ) -> NDArray[np.float64] | NDArray[np.complex128]:
        radial = np.asarray(radial_points, dtype=np.float64)
        if radial.ndim == 0:
            return matrix + dipole_coefficient * dipole_matrix / float(radial) ** 3
        return matrix + dipole_coefficient * dipole_matrix[None, :, :] / radial[:, None, None] ** 3

    if isinstance(potential, PESWrapper):
        scalar_basis = scalar_vmat.prepare(
            basis,
            theta_X,
            theta_weights_X,
            theta_Y,
            theta_weights_Y,
            phi,
            phi_weights,
        )
        scalar_device_bases: dict[tuple[str, int], scalar_vmat.FSDiatomDiatomVBasisDevice] = {}

        def scalar_grid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
            if potential_grid is not None:
                return cast(NDArray[np.float64], potential_grid.take(radial_points))
            return get_Vgrid_diatom_diatom(
                potential,
                radial_points,
                basis.monomer_X.vib.grids,
                basis.monomer_Y.vib.grids,
                theta_X,
                theta_Y,
                phi,
            )

        def scalar_matrix(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
            return add_magnetic_dipole(scalar_vmat.contract(scalar_basis, scalar_grid(radial_points)), radial_points)

        def scalar_blocks_device(
            radial_points: NDArray[np.float64],
            channel_blocks: tuple[tuple[int, ...], ...],
            device: JaxDevice,
        ) -> tuple[jax.Array, ...]:
            key = (device.platform, device.id)
            if key not in scalar_device_bases:
                scalar_device_bases[key] = scalar_vmat.device_basis(scalar_basis, device)
            values = (
                scalar_grid(radial_points)
                if potential_grid is None
                else cast(NDArray[np.float64] | jax.Array, potential_grid.take_device(radial_points, device))
            )
            radial_device = jax.device_put(radial_points, device)
            return tuple(
                scalar_vmat.contract_device(scalar_basis, scalar_device_bases[key], values, device, indices)
                + dipole_coefficient * jax.device_put(dipole_matrix[np.ix_(indices, indices)], device)[None, :, :] / radial_device[:, None, None] ** 3
                for indices in channel_blocks
            )

        return ScattHamiltonian(
            basis=basis,
            reduced_mass=system.reduced_mass,
            interaction=scalar_matrix,
            approx=Approx.EXACT,
            device_block_interaction=scalar_blocks_device,
            potential_grid_size=prod(scalar_basis.grid_shape),
        )

    spin_basis = spin_vmat.prepare(
        basis,
        potential.two_total_spins,
        potential.orbital_states,
        theta_X,
        theta_weights_X,
        theta_Y,
        theta_weights_Y,
        phi,
        phi_weights,
    )
    spin_device_bases: dict[tuple[str, int], spin_vmat.SpinResolvedFSDiatomDiatomVBasisDevice] = {}

    def spin_grid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64] | NDArray[np.complex128]:
        if potential_grid is not None:
            return cast(NDArray[np.float64] | NDArray[np.complex128], potential_grid.take(radial_points))
        return get_spin_resolved_grid_diatom_diatom(
            potential,
            radial_points,
            basis.monomer_X.vib.grids,
            basis.monomer_Y.vib.grids,
            theta_X,
            theta_Y,
            phi,
        )

    def spin_matrix(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64] | NDArray[np.complex128]:
        return add_magnetic_dipole(spin_vmat.contract(spin_basis, spin_grid(radial_points)), radial_points)

    def spin_blocks_device(
        radial_points: NDArray[np.float64],
        channel_blocks: tuple[tuple[int, ...], ...],
        device: JaxDevice,
    ) -> tuple[jax.Array, ...]:
        key = (device.platform, device.id)
        if key not in spin_device_bases:
            spin_device_bases[key] = spin_vmat.device_basis(spin_basis, device)
        values = (
            spin_grid(radial_points)
            if potential_grid is None
            else cast(NDArray[np.float64] | NDArray[np.complex128] | jax.Array, potential_grid.take_device(radial_points, device))
        )
        radial_device = jax.device_put(radial_points, device)
        return tuple(
            spin_vmat.contract_device(spin_basis, spin_device_bases[key], values, device, indices)
            + dipole_coefficient * jax.device_put(dipole_matrix[np.ix_(indices, indices)], device)[None, :, :] / radial_device[:, None, None] ** 3
            for indices in channel_blocks
        )

    return ScattHamiltonian(
        basis=basis,
        reduced_mass=system.reduced_mass,
        interaction=spin_matrix,
        approx=Approx.EXACT,
        device_block_interaction=spin_blocks_device,
        potential_grid_size=prod(spin_basis.grid_shape) * prod(spin_basis.electronic_shape),
    )


# ----------------------------------------------------------------------------------------
