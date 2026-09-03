from dataclasses import replace
from pathlib import Path
from typing import get_type_hints

import jax
import numpy as np
import pytest

import pyticc as ticc
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import Channel, ChannelBasis, ExchangeAdaptation, adapt_molecule_exchange
from pyticc.basis.rovib import RovibBasis
from pyticc.input.diatom_diatom import run as run_diatom_toml
from pyticc.match.asymptotic import get_Bmat_BF_to_SF
from pyticc.matrix.centrifugal import get_Umat_BF
from pyticc.matrix.interaction import contract, contract_device, device_basis
from pyticc.matrix.interaction.diatom_diatom import prepare
from pyticc.pes.molecule_exchange import validate_exchange_potential
from pyticc.propagation.logd import initialize_logD_capture, initialize_logD_inelastic, propagate_logD
from pyticc.scattering.energy_transfer.diatom_diatom import build_hamiltonian


def test_exchange_annotations_resolve_at_runtime() -> None:
    assert get_type_hints(ExchangeAdaptation)["source_channels"] == tuple[Channel, ...]
    assert get_type_hints(ChannelBasis)["exchange"] == ExchangeAdaptation | None
    assert get_type_hints(adapt_molecule_exchange) == {"basis": ChannelBasis, "eta": int, "return": ChannelBasis}


def _monomer(jmax: int = 2, nv: int = 2) -> ticc.DiatomBasis:
    rotation = np.arange(jmax + 1)
    energies = 0.001 * np.arange(nv)[:, None] + 0.0001 * (rotation * (rotation + 1))[None, :]
    rng = np.random.default_rng(349)
    wavefunctions = np.stack([np.linalg.qr(rng.normal(size=(nv, nv)))[0] for _ in rotation], axis=-1)
    return ticc.DiatomBasis(RovibBasis(np.linspace(1.5, 2.0, nv), energies, wavefunctions), energy_zero=0.0)


def _pes() -> ticc.PESWrapper:
    def interaction(R: float, q: np.ndarray) -> np.ndarray:
        x, y = np.cos(q[2]), np.cos(q[3])
        return (-1.0 + 0.03 * (q[0] + q[1]) + 0.07 * (q[0] - q[1]) * (x + y) + 0.1 * x * y + 0.04 * (x - y)) / R**6

    return ticc.PESWrapper(interaction=interaction)


def _system(monomer: ticc.DiatomBasis, eta: int = 0, J: int = 1, parity: int = 1, **kwargs: object) -> ticc.ScattSystem:
    return ticc.build_ScattSystem(
        monomer,
        monomer,
        scattering_type="AB+CD",
        Jtot=J,
        system_parity=parity,
        molecule_exchange=eta,
        potential=_pes(),
        reduced_mass=2.0,
        **kwargs,
    )


def _transform(basis: ChannelBasis) -> np.ndarray:
    assert basis.exchange is not None
    adaptation = basis.exchange
    result = np.zeros((len(adaptation.source_channels), basis.n_channel))
    for i, (positions, weights) in enumerate(zip(adaptation.source_indices, adaptation.coefficients, strict=True)):
        for position, weight in zip(positions, weights, strict=True):
            result[position, i] += weight
    return result


def _permutation(basis: ChannelBasis) -> np.ndarray:
    lookup = {(c.mis_X, c.mis_Y, c.j_couple, c.K): i for i, c in enumerate(basis)}
    result = np.zeros((basis.n_channel, basis.n_channel))
    for i, c in enumerate(basis):
        result[lookup[c.mis_Y, c.mis_X, c.j_couple, c.K], i] = basis.system_parity * (-1) ** c.j_couple
    return result


@pytest.mark.parametrize("J,parity", [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1), (3, 1), (3, -1)])
def test_exchange_basis_is_complete_orthonormal_and_matches_sf(J: int, parity: int) -> None:
    monomer = _monomer(nv=1)
    full = _system(monomer, J=J, parity=parity).basis
    adapted = [_system(monomer, eta, J, parity).basis for eta in (1, -1)]
    Tplus, Tminus = map(_transform, adapted)
    E = _permutation(full)
    np.testing.assert_allclose(E @ E, np.eye(full.n_channel), atol=1e-14)
    for eta, basis, T in zip((1, -1), adapted, (Tplus, Tminus), strict=True):
        np.testing.assert_allclose(T.T @ T, np.eye(basis.n_channel), atol=1e-14)
        np.testing.assert_allclose(E @ T, eta * T, atol=1e-14)
        np.testing.assert_allclose(get_Umat_BF(basis), T.T @ get_Umat_BF(full) @ T, atol=1e-13)
        for c in basis:
            assert (c.mis_X.v, c.mis_X.j) <= (c.mis_Y.v, c.mis_Y.j)
            if c.mis_X == c.mis_Y:
                assert eta * parity * (-1) ** c.j_couple == 1
    np.testing.assert_allclose(Tplus.T @ Tminus, 0.0, atol=1e-14)
    np.testing.assert_allclose(Tplus @ Tplus.T + Tminus @ Tminus.T, np.eye(full.n_channel), atol=1e-14)
    if not full.n_channel:
        return
    B, L = get_Bmat_BF_to_SF(full)
    lookup = {(c.mis_X, c.mis_Y, c.j_couple, int(round(L[i]))): i for i, c in enumerate(full)}
    E_sf = np.zeros_like(E)
    for i, c in enumerate(full):
        ell = int(round(L[i]))
        E_sf[lookup[c.mis_Y, c.mis_X, c.j_couple, ell], i] = (-1) ** (c.mis_X.j + c.mis_Y.j - c.j_couple + ell)
    np.testing.assert_allclose(E, B @ E_sf @ B.T, atol=2e-14)


@pytest.mark.parametrize("eta", [1, -1])
def test_adapted_kernel_equals_full_projection_on_host_and_device(eta: int) -> None:
    monomer = _monomer(jmax=1)
    full = _system(monomer).basis
    adapted = _system(monomer, eta).basis
    cos, weight = gauss_legendre_dvr(-1.0, 1.0, 5)
    phi, weight_phi = gauss_legendre_dvr(0.0, np.pi, 12)
    args = (monomer.rovib, monomer.rovib, cos, weight, cos, weight, phi, weight_phi)
    full_kernel, adapted_kernel = prepare(full, *args), prepare(adapted, *args)
    q = np.asarray(np.meshgrid(monomer.rovib.grids, monomer.rovib.grids, np.arccos(cos), np.arccos(cos), phi, indexing="ij"))
    values = np.stack([_pes().interaction(R, q.reshape(5, -1)).reshape(full_kernel.grid_shape) for R in (3.0, 4.0)])
    validate_exchange_potential(values)
    matrix = contract(full_kernel, values)
    E, T = _permutation(full), _transform(adapted)
    np.testing.assert_allclose(E @ matrix, matrix @ E, atol=1e-15)
    expected = T.T @ matrix @ T
    np.testing.assert_allclose(contract(adapted_kernel, values), expected, atol=1e-15)
    np.testing.assert_allclose(contract(adapted_kernel, np.ones(full_kernel.grid_shape)), np.eye(adapted.n_channel), atol=1e-13)
    other = _transform(_system(monomer, -eta).basis)
    np.testing.assert_allclose(T.T @ matrix @ other, 0.0, atol=1e-15)
    devices = [jax.devices("cpu")[0]]
    try:
        devices.extend(jax.devices("gpu")[:1])
    except RuntimeError:
        pass
    indices = tuple(range(adapted.n_channel - 1, -1, -2))
    for device in devices:
        actual = contract_device(adapted_kernel, device_basis(adapted_kernel, device), values, device, indices)
        np.testing.assert_allclose(np.asarray(actual), expected[:, indices, :][:, :, indices], atol=1e-15)


@pytest.mark.parametrize("eta", [1, -1])
@pytest.mark.parametrize("mode", ["inelastic", "capture"])
def test_exchange_solver_cache_and_boundary_modes(eta: int, mode: str) -> None:
    monomer = _monomer(jmax=1, nv=1)
    system = _system(monomer, eta)
    grid = ticc.prepare_potential(system, (3.0, 4.0), (0.025,), n_theta_X=5, n_theta_Y=5, n_phi=10)
    runtime = ticc.Propagation(mode=mode, device="cpu")
    result = ticc.solve(system, [0.01, 0.02], grid, runtime)
    assert result.molecule_exchange == eta
    assert f"molecule_exchange={eta:+d}" in ticc.report.channels(result.basis)
    assert f"molecule_exchange={eta:+d}" in ticc.report.smatrix(result)
    full = _system(monomer)
    Hfull = build_hamiltonian(full, potential_grid=grid)
    Hadapted = build_hamiltonian(system, potential_grid=grid)
    Hdirect = build_hamiltonian(system, n_theta_X=5, n_theta_Y=5, n_phi=10)
    T = _transform(system.basis)
    for R in (3.0, 3.5, 4.0):
        np.testing.assert_allclose(Hadapted.H(R), T.T @ Hfull.H(R) @ T, atol=1e-14)
        np.testing.assert_allclose(Hadapted.V(R), Hdirect.V(R), atol=1e-15)
    for S in result.Smat:
        if mode == "inelastic":
            np.testing.assert_allclose(S.conj().T @ S, np.eye(S.shape[0]), atol=1e-11)
        else:
            loss = 1.0 - np.sum(np.abs(S) ** 2, axis=0)
            assert np.min(loss) >= -1e-11
            assert np.max(loss) <= 1.0 + 1e-11
    opposite = _system(monomer, -eta)
    reused = ticc.solve(opposite, [0.01], grid, runtime)
    assert reused.molecule_exchange == -eta


def test_empty_forbidden_block_skips_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    monomer = _monomer(jmax=0, nv=1)
    system = _system(monomer, eta=1, J=1, parity=-1)
    assert system.n_channel == 0

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("An empty exchange block must not be propagated")

    monkeypatch.setattr("pyticc.scattering.solver.propagate", forbidden)
    grid = ticc.prepare_potential(system, (3.0, 3.2), (0.1,), n_theta_X=3, n_theta_Y=3, n_phi=4)
    result = ticc.solve(system, [0.01], grid, ticc.Propagation(mode="capture"))
    assert result.Smat[0].shape == (0, 0)
    assert result.Y_asymptotic.shape == (1, 0, 0)
    assert result.open_channel_indices[0].size == 0
    assert "molecule_exchange=+1" in ticc.report.smatrix(result)
    allowed = _system(monomer, eta=-1, J=1, parity=-1)
    assert allowed.n_channel == 1
    np.testing.assert_allclose(get_Bmat_BF_to_SF(allowed.basis)[1], [1.0])


def test_invalid_exchange_models_and_truncations_are_rejected() -> None:
    monomer = _monomer()
    for eta in (True, 1.0, 2):
        with pytest.raises(ValueError, match="molecule_exchange must"):
            _system(monomer, eta)
    for approx in (ticc.Approx.CS, ticc.Approx.NNCC):
        with pytest.raises(ValueError, match="exact CC"):
            _system(monomer, 1, approx=approx)
    with pytest.raises(ValueError, match="same diatomic monomer"):
        replace(_system(monomer, 1), monomer_Y=_monomer())
    with pytest.raises(ValueError, match="only AB\\+CD"):
        replace(_system(monomer, 1), scattering_type=ticc.ScatteringType.ATOM_DIATOM)
    with pytest.raises(ValueError, match="retained X/Y"):
        _system(monomer, 1, channel=ticc.ChannelSpec(vmin_X=1))
    with pytest.raises(ValueError, match="retained X/Y"):
        _system(monomer, 1, channel=ticc.ChannelSpec(exchange_parity_X=1))


def test_asymmetric_potential_and_quadrature_are_rejected() -> None:
    system = _system(_monomer(), 1)
    with pytest.raises(ValueError, match="polar quadratures"):
        ticc.prepare_potential(system, (3.0, 3.2), (0.1,), n_theta_X=3, n_theta_Y=4)
    bad = replace(system, potential=ticc.PESWrapper(interaction=lambda R, q: q[0] / R**6))
    with pytest.raises(ValueError, match="PES violates"):
        ticc.prepare_potential(bad, (3.0, 3.2), (0.1,), n_theta_X=3, n_theta_Y=3)
    with pytest.raises(ValueError, match="PES violates"):
        build_hamiltonian(bad, n_theta_X=3, n_theta_Y=3).V(3.0)
    grid = ticc.prepare_potential(system, (3.0, 3.2), (0.1,), n_theta_X=3, n_theta_Y=3)
    values = grid.values.copy()
    values[:, 0, 1] += 1.0
    with pytest.raises(ValueError, match="PES violates"):
        build_hamiltonian(system, potential_grid=replace(grid, values=values))


@pytest.mark.parametrize("mode", ["inelastic", "capture"])
def test_full_and_reduced_propagation_use_the_same_physical_boundary(mode: str) -> None:
    monomer = _monomer(jmax=1, nv=1)
    full_system = _system(monomer)
    H = build_hamiltonian(full_system, n_theta_X=5, n_theta_Y=5, n_phi=10)
    transforms = [_transform(_system(monomer, eta).basis) for eta in (1, -1)]
    radial = np.linspace(3.0, 3.4, 81)
    half_steps = np.diff(radial)[::2]
    W_base = np.stack([2.0 * H.reduced_mass * H.H(R) for R in radial])
    energies = np.array([0.01])
    shift = 2.0 * H.reduced_mass * energies[0]
    initialize = initialize_logD_capture if mode == "capture" else initialize_logD_inelastic
    Y_full_initial = np.zeros_like(W_base[0], dtype=np.complex128)
    Y_full_expected = np.zeros_like(Y_full_initial)
    for T in transforms:
        projected = T.T @ W_base @ T
        Y_initial = np.asarray(initialize(projected[0] - shift * np.eye(T.shape[1])))
        Y_full_initial += T @ Y_initial @ T.T
        propagated = propagate_logD(
            Y_initial[None],
            energies,
            H.reduced_mass,
            half_steps,
            projected[::2][:-1],
            projected[1::2],
            projected[2::2],
        )
        Y_full_expected += T @ np.asarray(propagated)[0] @ T.T
    actual = propagate_logD(
        Y_full_initial[None],
        energies,
        H.reduced_mass,
        half_steps,
        W_base[::2][:-1],
        W_base[1::2],
        W_base[2::2],
    )
    np.testing.assert_allclose(np.asarray(actual)[0], Y_full_expected, atol=2e-10, rtol=2e-10)


def test_zero_exchange_keeps_labeled_channels_and_legacy_matrices() -> None:
    monomer = _monomer(jmax=1, nv=1)
    explicit = _system(monomer, eta=0)
    legacy = ticc.build_ScattSystem(
        monomer,
        monomer,
        scattering_type="AB+CD",
        Jtot=1,
        system_parity=1,
        potential=_pes(),
        reduced_mass=2.0,
    )
    assert explicit.basis.exchange is None
    assert explicit.basis.channels == legacy.basis.channels
    assert any(c.mis_X.j > c.mis_Y.j for c in explicit.basis)
    options = dict(n_theta_X=4, n_theta_Y=4, n_phi=8)
    np.testing.assert_array_equal(build_hamiltonian(explicit, **options).V(3.0), build_hamiltonian(legacy, **options).V(3.0))


def test_exchange_metadata_and_cached_grids_cannot_be_silently_mismatched() -> None:
    system = _system(_monomer(jmax=1), 1)
    grid = ticc.prepare_potential(system, (3.0, 3.2), (0.1,), n_theta_X=3, n_theta_Y=3)
    with pytest.raises(ValueError, match="different molecule_exchange"):
        build_hamiltonian(replace(system, molecule_exchange=-1), potential_grid=grid)
    coordinates = tuple((name, values + 0.1 if name == "r_X" else values) for name, values in grid.coordinates)
    with pytest.raises(ValueError, match="cached radial grids"):
        build_hamiltonian(system, potential_grid=replace(grid, coordinates=coordinates))
    hamiltonian = build_hamiltonian(system, potential_grid=grid)
    with pytest.raises(ValueError, match="exact CC"):
        replace(hamiltonian, approx=ticc.Approx.NNCC)
    for invalid in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite real"):
            validate_exchange_potential(np.full((1, 1, 1, 1, 1), invalid))


def test_toml_does_not_silently_ignore_molecule_exchange() -> None:
    with pytest.raises(NotImplementedError, match="Python API"):
        run_diatom_toml({"molecule_exchange": 1}, Path.cwd(), _pes())
