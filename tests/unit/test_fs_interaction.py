import jax
import numpy as np
import pytest

import pyticc as ticc
from pyticc.basis.podvr import VibPODVR
from pyticc.fine_structure import FSConstants, build_fs_channels, build_fs_monomer_basis
from pyticc.matrix.interaction import fs_atom_diatom as vmat
from pyticc.pes import LambdaPES
from pyticc.scattering import fine_structure_atom_diatom


def _contraction_devices():
    devices = [jax.devices("cpu")[0]]
    try:
        devices.extend(jax.devices("gpu")[:1])
    except RuntimeError:
        pass
    return tuple(devices)


def _pi_basis():
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    monomer = build_fs_monomer_basis(vib, (1, 3), 2, 1, FSConstants(A=0.01, B=0.001))
    return build_fs_channels(monomer, two_J=3, system_parity=1)


def test_constant_vsum_is_identity_and_vdif_is_hermitian() -> None:
    basis = _pi_basis()
    x, weights = np.polynomial.legendre.leggauss(24)
    V_basis = vmat.prepare(basis, np.arccos(x), weights)
    potential = np.empty((1, x.size, 2))
    potential[..., 0] = 2.5
    potential[..., 1] = 0.0
    sum_matrix = vmat.contract(V_basis, potential)

    np.testing.assert_allclose(sum_matrix, 2.5 * np.eye(basis.n_channel), atol=2.0e-13)
    potential[..., 0] = 0.0
    potential[..., 1] = 0.3 * (1.0 - x**2)
    matrix = vmat.contract(V_basis, potential)
    np.testing.assert_allclose(matrix, matrix.T, atol=2.0e-14)


@pytest.mark.parametrize("device", _contraction_devices(), ids=lambda device: device.platform)
def test_fs_device_contraction_matches_numpy_for_batch_and_channel_selection(device) -> None:
    basis = _pi_basis()
    x, weights = np.polynomial.legendre.leggauss(16)
    V_basis = vmat.prepare(basis, np.arccos(x), weights)
    rng = np.random.default_rng(42)
    potential = rng.normal(size=(3, 1, x.size, 2))
    selected = tuple(range(basis.n_channel - 1, -1, -2))

    expected_full = vmat.contract(V_basis, potential)
    expected = expected_full[:, selected, :][:, :, selected]
    result = vmat.contract_device(V_basis, vmat.device_basis(V_basis, device), potential, device, selected)
    resident_result = vmat.contract_device(V_basis, vmat.device_basis(V_basis, device), jax.device_put(potential, device), device, selected)

    assert result.devices() == {device}
    np.testing.assert_allclose(result, expected, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(resident_result, expected, rtol=2.0e-13, atol=2.0e-13)


def test_half_integer_fs_channels_propagate_and_match() -> None:
    basis = _pi_basis()
    potential = LambdaPES(lambda R, coordinates: np.zeros((coordinates.shape[1], 2)))
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        basis.monomer,
        scattering_type="A+BC_fine_structure",
        two_J=basis.two_J,
        system_parity=basis.system_parity,
        potential=potential,
        reduced_mass=2.0,
    )
    potential_grid = fine_structure_atom_diatom.prepare_potential(
        system,
        (4.0, 4.2),
        (0.1,),
        n_theta=16,
    )
    hamiltonian = fine_structure_atom_diatom.build_hamiltonian(system, potential_grid=potential_grid)
    energy = float(np.max(basis.E_int) + 0.1)

    assert hamiltonian.device_block_interaction is not None
    device = jax.devices("cpu")[0]
    radial_points = np.array([4.0, 4.1])
    indices = tuple(range(basis.n_channel))
    device_matrix = hamiltonian.device_block_interaction(radial_points, (indices,), device)[0]
    np.testing.assert_allclose(device_matrix, hamiltonian.V(radial_points), rtol=2.0e-13, atol=2.0e-13)

    result = ticc.solve(system, [energy], potential_grid, ticc.Propagation())

    assert isinstance(result, ticc.ScatteringResult)
    assert result.basis.n_channel == basis.n_channel
    np.testing.assert_allclose(result.Smat[0].conj().T @ result.Smat[0], np.eye(basis.n_channel), atol=2.0e-12)
