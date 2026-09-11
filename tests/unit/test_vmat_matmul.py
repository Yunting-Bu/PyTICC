"""Independent old-expression references for explicit GEMM contractions."""

import jax
import numpy as np
import pytest

from pyticc.matrix.interaction import fs_diatom_diatom_spin as spin
from pyticc.matrix.interaction.atom_diatom import _contract_electric_sf_device
from pyticc.matrix.interaction.common import _contract_block_device
from pyticc.matrix.interaction.diabatic_atom_diatom import _contract_weighted_basis_device
from pyticc.matrix.interaction.fs_both_atom_diatom import _contract_device as atom_spin_contract
from pyticc.matrix.interaction.fs_diatom_diatom_spin import _contract_device as pair_spin_contract
from pyticc.pes.spin_resolved_diatom_diatom import OrbitalState


@pytest.mark.parametrize("batch", (1, 4))
def test_weighted_gemm_matches_original_contractions(batch):
    rng = np.random.default_rng(24)
    left, right = rng.normal(size=(5, 17)), rng.normal(size=(3, 17))
    weights = rng.normal(size=(batch, 17))
    expected = np.einsum("ig,bg,jg->bij", left, weights, right, optimize=True)
    actual = _contract_weighted_basis_device(*map(jax.device_put, (left, weights, right)))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    imag = rng.normal(size=left.shape)
    expected = np.einsum("ig,bg,jg->bij", left, weights, left) + np.einsum("ig,bg,jg->bij", imag, weights, imag)
    for function in (_contract_electric_sf_device,):
        actual = function(*map(jax.device_put, (left, imag, weights)))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    actual = _contract_block_device(*map(jax.device_put, (left, imag, weights)), 0.3)
    np.testing.assert_allclose(actual, 0.3 * expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("function", (atom_spin_contract, pair_spin_contract))
@pytest.mark.parametrize("complex_values", (False, True))
def test_spin_gemm_preserves_complex_electronic_order(function, complex_values):
    rng = np.random.default_rng(25)
    potential = rng.normal(size=(4, 13, 12))
    kernel = rng.normal(size=(13, 12, 9)) + 1j * rng.normal(size=(13, 12, 9))
    if complex_values:
        potential = potential + 1j * rng.normal(size=potential.shape)
    expected = np.einsum("bge,gep->bp", potential, kernel, optimize=True)
    np.testing.assert_allclose(function(jax.device_put(potential), jax.device_put(kernel)), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("budget", (1, 1024**2))
@pytest.mark.parametrize("indices", (None, (2, 0)))
def test_resident_and_streamed_kernel_match_host(monkeypatch, budget, indices):
    rng = np.random.default_rng(26)
    rows, columns = np.tril_indices(3)
    kernel = rng.normal(size=(8, 3, 6)) + 1j * rng.normal(size=(8, 3, 6))
    basis = spin.SpinResolvedFSDiatomDiatomVBasis(3, (1, 1, 2, 2, 2), (0, 2, 4), (OrbitalState(0, 0),), rows, columns, kernel)
    monkeypatch.setattr(spin, "_DEVICE_KERNEL_TARGET_BYTES", budget)
    device = jax.devices("cpu")[0]
    cached = spin.device_basis(basis, device)
    assert (cached.resident_kernel is not None) == (kernel.nbytes <= budget)
    for _ in range(2):
        values = rng.normal(size=(2, 8, 3))
        expected = spin.contract(basis, values, indices)
        actual = spin.contract_device(basis, cached, values, device, indices)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
