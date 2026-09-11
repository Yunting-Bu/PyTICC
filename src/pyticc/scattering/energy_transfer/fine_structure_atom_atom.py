from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from pyticc._typing import JaxDevice
from pyticc.fine_structure.atom_atom import FSAtomAtomBasis
from pyticc.matrix.interaction.fs_atom_atom import magnetic_dipole_matrix, prepare_fs_atom_atom_vbasis
from pyticc.pes.spin_resolved_atom_atom import SpinResolvedAtomAtomPES
from pyticc.scattering.hamiltonian import ScattHamiltonian
from pyticc.scattering.potential import PotentialGrid, _potential_radial_grid, _require_type
from pyticc.system import ScatteringType, ScattSystem


def prepare_potential(system: ScattSystem, boundaries: Sequence[float], half_steps: Sequence[float], *, processes: int = 1) -> PotentialGrid:
    """Sample raw atomic diabatic potentials on the propagation grid.

    Inputs:
        system: ScattSystem - structured atomic pair and orbital PES
        boundaries, half_steps: sequences - radial grid parameters in bohr
        processes: int - currently only serial callback evaluation is supported
    Returns:
        grid: PotentialGrid - Hartree values ordered (R,S,a_bra,a_ket)
    """
    if processes != 1:
        raise ValueError("Atomic PES preparation currently requires processes=1; use interaction_many for batched evaluation")
    if not isinstance(system.potential, SpinResolvedAtomAtomPES):
        raise TypeError("Expected SpinResolvedAtomAtomPES")
    boundaries, half_steps, sectors, radial = _potential_radial_grid(boundaries, half_steps)
    pes = system.potential
    coordinates = (
        ("two_total_spins", np.asarray(pes.two_total_spins)),
        ("orbital_two_lambda_X", np.asarray([a.two_lambda_X for a in pes.orbital_states])),
        ("orbital_two_lambda_Y", np.asarray([a.two_lambda_Y for a in pes.orbital_states])),
    )
    return PotentialGrid(boundaries, half_steps, sectors, radial, ScatteringType.ATOM_ATOM_FINE_STRUCTURE, coordinates, (), pes.evaluate(radial))


def build_hamiltonian(system: ScattSystem, *, potential_grid: PotentialGrid | None = None) -> ScattHamiltonian:
    r"""Build the structured atomic-pair Hamiltonian for shared propagation.

    Formula:
        H(R)=diag(E_X+E_Y)+U/(2 mu R^2)+A-dagger Q W(R) A+C_dd D/R^3.
        Atomic thresholds are experimental values relative to monomer zeros;
        they are inserted by ScattHamiltonian, never inside the PES.

    Inputs:
        system: ScattSystem - atomic channels, diabatic PES and reduced mass
        potential_grid: PotentialGrid | None - optional cached raw values
    Returns:
        hamiltonian: ScattHamiltonian - Hartree Hamiltonian with shared Umat
    """
    if not isinstance(system.basis, FSAtomAtomBasis) or not isinstance(system.potential, SpinResolvedAtomAtomPES):
        raise TypeError("Expected prepared atomic channels and SpinResolvedAtomAtomPES")
    if system.reduced_mass is None:
        raise ValueError("Atomic scattering requires reduced_mass")
    if system.basis.monomer_X is not system.monomer_X or system.basis.monomer_Y is not system.monomer_Y:
        raise ValueError("System and atomic channel basis must use the same monomer objects")
    if potential_grid is not None:
        _require_type(potential_grid, ScatteringType.ATOM_ATOM_FINE_STRUCTURE)
        expected = (
            ("two_total_spins", system.potential.two_total_spins),
            ("orbital_two_lambda_X", [a.two_lambda_X for a in system.potential.orbital_states]),
            ("orbital_two_lambda_Y", [a.two_lambda_Y for a in system.potential.orbital_states]),
        )
        if any(not np.array_equal(potential_grid.coordinate(name), labels) for name, labels in expected):
            raise ValueError("Cached atomic PES has different electronic-axis ordering")
    vbasis = prepare_fs_atom_atom_vbasis(system.basis, system.potential)
    dipole = system.magnetic_dipole_coefficient * magnetic_dipole_matrix(system.basis, vbasis)

    def interaction(radial):
        values = system.potential.evaluate(radial) if potential_grid is None else potential_grid.take(radial)
        return vbasis.contract(values) + dipole / np.asarray(radial)[..., None, None] ** 3

    device_arrays = {}

    def blocks_device(radial: np.ndarray, blocks: tuple[tuple[int, ...], ...], device: JaxDevice) -> tuple[jax.Array, ...]:
        key = (device.platform, device.id)
        if key not in device_arrays:
            device_arrays[key] = tuple(jax.device_put(a, device) for a in (vbasis.amplitudes, vbasis.projectors, dipole, vbasis.two_K))
        A, Q, D, K = device_arrays[key]
        values = jax.device_put(system.potential.evaluate(radial), device) if potential_grid is None else potential_grid.take_device(radial, device)
        radial_device = jax.device_put(radial, device)
        result = []
        for block in blocks:
            indices = jax.device_put(np.asarray(block, dtype=np.int64), device)
            selected = A[indices]
            matrix = jnp.einsum("cau,suv,rsab,dbv->rcd", selected, Q, values, selected, optimize=True)
            mask = K[indices, None] == K[None, indices]
            result.append(matrix * mask + D[indices[:, None], indices[None, :]] / radial_device[:, None, None] ** 3)
        return tuple(result)

    return ScattHamiltonian(
        basis=system.basis,
        reduced_mass=system.reduced_mass,
        interaction=interaction,
        approx=system.approx,
        K_delta=system.K_delta,
        device_block_interaction=blocks_device,
        potential_grid_size=len(system.potential.two_total_spins) * len(system.potential.orbital_states) ** 2,
    )
