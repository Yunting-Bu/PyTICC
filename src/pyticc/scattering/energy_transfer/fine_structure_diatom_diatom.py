from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
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
from pyticc.pes.molecule_exchange import validate_exchange_potential, validate_exchange_quadrature
from pyticc.pes.spin_resolved_diatom_diatom import SpinResolvedDiatomDiatomPES, get_spin_resolved_grid_diatom_diatom
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.potential import PotentialGrid, _potential_radial_grid, _require_type
from pyticc.system import Approx, ScatteringType, ScattSystem

_SCATTERING_TYPE = ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE


# ----------------------------------------------------------------------------------------
def _spin_coordinates(potential: SpinResolvedDiatomDiatomPES) -> tuple[tuple[str, NDArray[np.float64]], ...]:
    """Record ordered electronic-axis labels alongside the geometric grid."""
    return (
        ("two_total_spins", np.asarray(potential.two_total_spins, dtype=np.float64)),
        ("orbital_two_lambda_X", np.asarray([s.two_lambda_X for s in potential.orbital_states], dtype=np.float64)),
        ("orbital_two_lambda_Y", np.asarray([s.two_lambda_Y for s in potential.orbital_states], dtype=np.float64)),
    )


# ----------------------------------------------------------------------------------------
def _project_exchange(
    basis: FSDiatomDiatomBasis,
    matrix: NDArray[np.float64] | NDArray[np.complex128] | jax.Array,
    channel_indices: Sequence[int] | None = None,
) -> NDArray[np.float64] | NDArray[np.complex128] | jax.Array:
    r"""Check exchange invariance in the labeled basis and project its matrix.

    Formula:
        In the finite exchange-closed source basis, E|c>=s_c|bar(c)> and
        E^2=I. Check (E.T M E)_ij=s_i s_j M_(bar(i),bar(j))=M_ij,
        then return M_eta=T_eta.T M T_eta using at most four indexed terms.
        T_eta is the dimensionless, real normalized expansion in basis.exchange.
        This tests the contracted operator, not raw signed-Lambda PES entries:
        electronic frame phases and spin rotations have already been included.
        It is a finite-basis check, not a proof of continuum PES symmetry.

    Inputs:
        basis: FSDiatomDiatomBasis - target exchange block and source metadata
        matrix: NDArray | jax.Array - (...,n_source,n_source), Hartree for
            orbital interactions or dimensionless for the spin-dipole operator
        channel_indices: Sequence[int] | None - target channel positions

    Returns:
        projected: NDArray | jax.Array - (...,n_selected,n_selected), retaining
            input units, dtype and host/device representation. Validation uses
            a host copy and tolerance atol=1e-12, rtol=1e-10; no averaging.
    """
    indices = np.arange(basis.n_channel) if channel_indices is None else np.asarray(tuple(channel_indices), dtype=np.int64)
    if len(set(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= basis.n_channel):
        raise ValueError("channel_indices must be unique complete-basis positions")
    if basis.exchange is None:
        return matrix[..., indices[:, None], indices]
    exchange = basis.exchange
    if matrix.shape[-2:] != (len(exchange.source_channels),) * 2:
        raise ValueError("Exchange projection requires a full labeled-source matrix")
    host = np.asarray(matrix)
    permuted = host[..., exchange.permutation[:, None], exchange.permutation] * exchange.phases[:, None] * exchange.phases
    if not np.all(np.isfinite(host)) or not np.allclose(host, permuted, rtol=1.0e-10, atol=1.0e-12):
        error = float(np.max(np.abs(host - permuted))) if host.size else 0.0
        raise ValueError(
            f"Interaction violates complete-molecule exchange symmetry in the retained FS basis (maximum difference {error:.6g}); "
            "check the PES convention and angular quadrature convergence"
        )
    positions, weights = exchange.source_indices[indices], exchange.coefficients[indices]
    result = matrix[..., positions[:, 0, None], positions[:, 0]] * weights[:, 0, None] * weights[:, 0]
    for a, b in ((0, 1), (1, 0), (1, 1)):
        result = result + matrix[..., positions[:, a, None], positions[:, b]] * weights[:, a, None] * weights[:, b]
    return result


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
    if system.molecule_exchange:
        validate_exchange_quadrature(radial_X, radial_Y, cos_theta_X, cos_theta_Y, theta_weights_X, theta_weights_Y)
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
    if system.molecule_exchange and isinstance(system.potential, PESWrapper):
        validate_exchange_potential(values)
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
        )
        + (_spin_coordinates(system.potential) if isinstance(system.potential, SpinResolvedDiatomDiatomPES) else ()),
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
    if basis.molecule_exchange != system.molecule_exchange:
        raise ValueError("System and channel basis have different molecule_exchange settings")
    if basis.monomer_X is not system.monomer_X or basis.monomer_Y is not system.monomer_Y:
        raise ValueError("System and FS channel basis must use the same monomer objects")
    source_basis = basis if basis.exchange is None else replace(basis, channels=basis.exchange.source_channels, exchange=None)
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
    if system.molecule_exchange:
        validate_exchange_quadrature(basis.monomer_X.vib.grids, basis.monomer_Y.vib.grids, cos_theta_X, cos_theta_Y, theta_weights_X, theta_weights_Y)
        if potential_grid is not None:
            for name in ("r_X", "r_Y"):
                radial = potential_grid.coordinate(name)
                if radial.shape != basis.monomer_X.vib.grids.shape or not np.allclose(radial, basis.monomer_X.vib.grids, rtol=0.0, atol=1.0e-14):
                    raise ValueError("Molecule-exchange cached radial grids must match the shared FS monomer basis")
            if isinstance(potential, PESWrapper):
                validate_exchange_potential(potential_grid.values)
            else:
                values = potential_grid.values
                for name, expected in _spin_coordinates(potential):
                    if not np.array_equal(potential_grid.coordinate(name), expected):
                        raise ValueError("Cached spin-resolved PES electronic-axis order does not match the interaction model")
                electronic_shape = (len(potential.two_total_spins), len(potential.orbital_states), len(potential.orbital_states))
                if values.ndim != 9 or values.shape[-3:] != electronic_shape:
                    raise ValueError("Cached spin-resolved PES has incompatible electronic dimensions")
                if not np.all(np.isfinite(values)) or not np.allclose(values, values.conj().swapaxes(-1, -2), rtol=0.0, atol=1.0e-12):
                    raise ValueError("Cached spin-resolved orbital PES must be finite and Hermitian")
    dipole_matrix = spin_vmat.magnetic_dipole_matrix(
        source_basis,
        theta_X,
        theta_weights_X,
        theta_Y,
        theta_weights_Y,
        phi,
        phi_weights,
    )
    dipole_matrix = _project_exchange(basis, dipole_matrix)
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
            source_basis,
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
            values = get_Vgrid_diatom_diatom(
                potential,
                radial_points,
                basis.monomer_X.vib.grids,
                basis.monomer_Y.vib.grids,
                theta_X,
                theta_Y,
                phi,
            )
            if basis.molecule_exchange:
                validate_exchange_potential(values)
            return values

        def scalar_matrix(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
            orbital = scalar_vmat.contract(scalar_basis, scalar_grid(radial_points))
            return add_magnetic_dipole(_project_exchange(basis, orbital), radial_points)

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
            if basis.exchange is not None:
                for indices in channel_blocks:
                    scalar_vmat._packed_positions(basis.n_channel, indices)
                orbital = scalar_vmat.contract_device(scalar_basis, scalar_device_bases[key], values, device)
                projected = _project_exchange(basis, orbital)
                projected = projected + dipole_coefficient * jax.device_put(dipole_matrix, device)[None, :, :] / radial_device[:, None, None] ** 3
                return tuple(
                    projected[:, np.asarray(indices, dtype=np.int64)[:, None], np.asarray(indices, dtype=np.int64)] for indices in channel_blocks
                )
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
        source_basis,
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
        orbital = spin_vmat.contract(spin_basis, spin_grid(radial_points))
        return add_magnetic_dipole(_project_exchange(basis, orbital), radial_points)

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
        if basis.exchange is not None:
            for indices in channel_blocks:
                scalar_vmat._packed_positions(basis.n_channel, indices)
            orbital = spin_vmat.contract_device(spin_basis, spin_device_bases[key], values, device)
            projected = _project_exchange(basis, orbital)
            projected = projected + dipole_coefficient * jax.device_put(dipole_matrix, device)[None, :, :] / radial_device[:, None, None] ** 3
            return tuple(
                projected[:, np.asarray(indices, dtype=np.int64)[:, None], np.asarray(indices, dtype=np.int64)] for indices in channel_blocks
            )
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
