from dataclasses import replace
from typing import get_type_hints

import jax
import numpy as np
import pytest

import pyticc as ticc
import pyticc.matrix.interaction.fs_diatom_diatom as scalar_vmat
import pyticc.matrix.interaction.fs_diatom_diatom_spin as spin_vmat
from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.podvr import VibPODVR
from pyticc.fine_structure import build_fs_monomer_basis
from pyticc.fine_structure.diatom_diatom import FSExchangeAdaptation, adapt_fs_molecule_exchange
from pyticc.match.asymptotic import get_Bmat_FS_DiatomDiatom_BF_to_SF
from pyticc.match.finalize import finalize_scattering
from pyticc.matrix.centrifugal import get_Umat_FS_DiatomDiatom_BF
from pyticc.pes import allowed_total_spins, orbital_product_states
from pyticc.propagation.logd import initialize_logD_capture, initialize_logD_inelastic, propagate_logD
from pyticc.scattering.energy_transfer.fine_structure_diatom_diatom import _project_exchange, build_hamiltonian


def _monomer(lam: int = 0, spin: int = 1, js: tuple[int, ...] = (1,), nv: int = 1) -> ticc.FSMonomerBasis:
    wavefunctions = np.linalg.qr(np.random.default_rng(622).normal(size=(nv, nv)))[0]
    vib = VibPODVR(np.linspace(1.7, 2.1, nv), 0.001 * np.arange(nv), wavefunctions)
    return build_fs_monomer_basis(vib, js, lam, spin, ticc.FSConstants(B=0.0001, A=-0.001))


def _scalar_pes() -> ticc.PESWrapper:
    def interaction(R: float, q: np.ndarray) -> np.ndarray:
        x, y = np.cos(q[2]), np.cos(q[3])
        return (-1 + 0.04 * (q[0] + q[1]) + 0.13 * (x - y) + 0.08 * x * y + 0.03 * (q[0] - q[1]) * (x + y)) / R**6

    return ticc.PESWrapper(interaction=interaction)


def _spin_pes(lam: int = 2, spin: int = 1, imaginary: bool = True) -> ticc.SpinResolvedDiatomDiatomPES:
    spins = allowed_total_spins(spin, spin)
    orbitals = orbital_product_states(lam, lam)
    surfaces = np.arange(1, len(spins) + 1)[:, None, None] * np.eye(len(orbitals))[None]
    if imaginary:
        a, b = np.array([1, 0, 0, 1]) / np.sqrt(2), np.array([0, 1, 1, 0]) / np.sqrt(2)
        surfaces = surfaces + 0.3j * (np.outer(a, b) - np.outer(b, a))[None]
    scalar = _scalar_pes()

    def interaction(R: float, q: np.ndarray) -> np.ndarray:
        return scalar.interaction(R, q)[:, None, None, None] * surfaces[None]

    return ticc.SpinResolvedDiatomDiatomPES(interaction, spins, orbitals)


def _system(monomer: ticc.FSMonomerBasis, eta: int = 0, *, pes=None, J: int = 1, parity: int = 1, **kwargs) -> ticc.ScattSystem:
    return ticc.build_ScattSystem(
        monomer,
        monomer,
        scattering_type="AB+CD_fine_structure",
        two_J=2 * J,
        system_parity=parity,
        molecule_exchange=eta,
        potential=_scalar_pes() if pes is None else pes,
        reduced_mass=2.0,
        **kwargs,
    )


def _transform(basis: ticc.FSDiatomDiatomBasis) -> np.ndarray:
    exchange = basis.exchange
    assert exchange is not None
    result = np.zeros((len(exchange.source_channels), basis.n_channel))
    for column, (indices, weights) in enumerate(zip(exchange.source_indices, exchange.coefficients, strict=True)):
        for index, weight in zip(indices, weights, strict=True):
            result[index, column] += weight
    return result


def _quadrature() -> tuple[np.ndarray, ...]:
    x, w = gauss_legendre_dvr(-1, 1, 5)
    phi, wp = gauss_legendre_dvr(0, np.pi, 8)
    return np.arccos(x), w, np.arccos(x), w, phi, wp


def test_exchange_types_resolve_without_conditional_imports() -> None:
    assert get_type_hints(ticc.FSDiatomDiatomBasis)["exchange"] == FSExchangeAdaptation | None
    assert get_type_hints(adapt_fs_molecule_exchange)["return"] is ticc.FSDiatomDiatomBasis


@pytest.mark.parametrize("J", [0, 1, 2])
def test_isotropic_sigma_N_zero_limit_is_analytic_total_spin_surface(J: int) -> None:
    monomer = _monomer()
    basis = _system(monomer, J=J).basis
    kernel = spin_vmat.prepare(basis, (0, 2), orbital_product_states(0, 0), *_quadrature())
    values = np.broadcast_to(np.array([1.0, 3.0])[:, None, None], (*kernel.grid_shape, 2, 1, 1))
    matrix = spin_vmat.contract(kernel, values)
    np.testing.assert_allclose(matrix.imag, 0, atol=2e-14)
    for i, c in enumerate(basis):
        if monomer.blocks[c.block_X].parity == monomer.blocks[c.block_Y].parity == 1:
            expected = np.zeros(basis.n_channel)
            expected[i] = 1 if c.two_j12 == 0 else 3
            np.testing.assert_allclose(matrix[i], expected, atol=2e-14)


@pytest.mark.parametrize("lam,spin,js", [(0, 1, (1, 3)), (2, 1, (1, 3)), (4, 1, (3,)), (0, 2, (0, 2))])
@pytest.mark.parametrize("J,parity", [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)])
def test_exchange_basis_is_complete_and_matches_independent_sf_phase(lam: int, spin: int, js: tuple[int, ...], J: int, parity: int) -> None:
    monomer = _monomer(lam, spin, js)
    full = _system(monomer, J=J, parity=parity).basis
    adapted = [_system(monomer, eta, J=J, parity=parity).basis for eta in (1, -1)]
    Tplus, Tminus = map(_transform, adapted)
    exchange = adapted[0].exchange
    E = np.zeros((full.n_channel, full.n_channel))
    E[exchange.permutation, np.arange(full.n_channel)] = exchange.phases
    np.testing.assert_allclose(E @ E, np.eye(full.n_channel), atol=1e-14)
    for eta, basis, T in zip((1, -1), adapted, (Tplus, Tminus), strict=True):
        np.testing.assert_allclose(T.T @ T, np.eye(basis.n_channel), atol=1e-14)
        np.testing.assert_allclose(E @ T, eta * T, atol=1e-14)
        np.testing.assert_allclose(get_Umat_FS_DiatomDiatom_BF(basis), T.T @ get_Umat_FS_DiatomDiatom_BF(full) @ T, atol=1e-13)
        np.testing.assert_allclose(np.diag(basis.E_int), T.T @ np.diag(full.E_int) @ T, atol=1e-14)
        assert all((c.block_X, c.tau_X) <= (c.block_Y, c.tau_Y) for c in basis)
    np.testing.assert_allclose(Tplus.T @ Tminus, 0, atol=1e-14)
    np.testing.assert_allclose(Tplus @ Tplus.T + Tminus @ Tminus.T, np.eye(full.n_channel), atol=1e-14)
    if not full.n_channel:
        return
    B, L = get_Bmat_FS_DiatomDiatom_BF_to_SF(full)
    lookup = {(c.block_X, c.tau_X, c.block_Y, c.tau_Y, c.two_j12, round(ell)): i for i, (c, ell) in enumerate(zip(full, L, strict=True))}
    E_sf = np.zeros_like(E)
    for i, (c, ell) in enumerate(zip(full, L, strict=True)):
        a, b = monomer.blocks[c.block_X], monomer.blocks[c.block_Y]
        assert ell == pytest.approx(round(ell), abs=1e-12)
        assert parity == a.parity * b.parity * (-1) ** round(ell)
        j = lookup[(c.block_Y, c.tau_Y, c.block_X, c.tau_X, c.two_j12, round(ell))]
        E_sf[j, i] = (-1) ** ((a.two_j + b.two_j - c.two_j12) // 2 + round(ell))
    np.testing.assert_allclose(B.T @ E @ B, E_sf, atol=1e-13)


@pytest.mark.parametrize("kind", ["scalar", "sigma_spin", "pi_complex", "triplet_spin", "delta_spin"])
def test_all_interactions_project_and_device_subsets_preserve_complex_order(kind: str) -> None:
    lam, spin, js = {
        "scalar": (0, 1, (1,)),
        "sigma_spin": (0, 1, (1,)),
        "pi_complex": (2, 1, (3,)),
        "triplet_spin": (0, 2, (0, 2)),
        "delta_spin": (4, 1, (3,)),
    }[kind]
    monomer = _monomer(lam, spin, js, nv=2 if kind == "scalar" else 1)
    pes = _scalar_pes() if kind == "scalar" else _spin_pes(lam, spin, imaginary=kind == "pi_complex")
    options = dict(n_theta_X=5, n_theta_Y=5, n_phi=8)
    coefficient = 0.0002
    full_system = _system(monomer, pes=pes, magnetic_dipole_coefficient=coefficient)
    full = build_hamiltonian(full_system, **options)
    radii = np.array([3.0, 4.0])
    matrix = full.V(radii)
    dipole = spin_vmat.magnetic_dipole_matrix(full.basis, *_quadrature())
    device = jax.devices("cpu")[0]
    Ts = []
    for eta in (1, -1):
        system = _system(monomer, eta, pes=pes, magnetic_dipole_coefficient=coefficient)
        hamiltonian = build_hamiltonian(system, **options)
        T = _transform(system.basis)
        Ts.append(T)
        expected = T.T @ matrix @ T
        actual = hamiltonian.V(radii)
        np.testing.assert_allclose(actual, expected, atol=2e-13)
        np.testing.assert_allclose(actual, actual.conj().swapaxes(-1, -2), atol=2e-13)
        np.testing.assert_allclose(spin_vmat.magnetic_dipole_matrix(system.basis, *_quadrature()), T.T @ dipole @ T, atol=2e-13)
        indices = tuple(range(system.n_channel - 1, -1, -2))
        actual_device = hamiltonian.device_block_interaction(radii, (indices,), device)[0]
        np.testing.assert_allclose(np.asarray(actual_device), actual[:, indices, :][:, :, indices], atol=2e-13)
        if kind == "pi_complex":
            assert np.max(abs(actual.imag)) > 1e-7
        if kind in ("scalar", "pi_complex"):
            q = _quadrature()
            if kind == "scalar":
                kernel = scalar_vmat.prepare(system.basis, *q)
                values = ticc.prepare_potential(system, (3.0, 3.2), (0.1,), **options).values
                direct = scalar_vmat.contract(kernel, values[0])
            else:
                kernel = spin_vmat.prepare(system.basis, pes.two_total_spins, pes.orbital_states, *q)
                values = ticc.prepare_potential(system, (3.0, 3.2), (0.1,), **options).values
                direct = spin_vmat.contract(kernel, values[0])
                selected = spin_vmat.contract(kernel, values[:2], indices)
                np.testing.assert_allclose(selected, spin_vmat.contract(kernel, values[:2])[:, indices, :][:, :, indices], atol=2e-13)
                selected_device = spin_vmat.contract_device(kernel, spin_vmat.device_basis(kernel, device), values[:2], device, indices)
                np.testing.assert_allclose(np.asarray(selected_device), selected, atol=2e-13)
            np.testing.assert_allclose(direct + coefficient * T.T @ dipole @ T / radii[0] ** 3, actual[0], atol=2e-13)
    np.testing.assert_allclose(Ts[0].T @ matrix @ Ts[1], 0, atol=2e-13)
    np.testing.assert_allclose(Ts[0].T @ dipole @ Ts[1], 0, atol=2e-13)


@pytest.mark.parametrize("mode", ["inelastic", "capture"])
@pytest.mark.parametrize("spin_resolved", [False, True])
def test_solver_projects_full_solution_with_matched_boundary_and_mixed_open_closed_channels(mode: str, spin_resolved: bool) -> None:
    monomer = _monomer()
    pes = _spin_pes(0, imaginary=False) if spin_resolved else _scalar_pes()
    options = dict(n_theta_X=5, n_theta_Y=5, n_phi=8)
    full_system = _system(monomer, pes=pes, magnetic_dipole_coefficient=0.0002)
    grid = ticc.prepare_potential(full_system, (3.0, 3.4), (0.005,), **options)
    energies = np.array([0.00015, 0.2])
    propagation = ticc.Propagation(mode=mode)
    full_H = build_hamiltonian(full_system, potential_grid=grid)
    W_base = 2 * full_H.reduced_mass * np.stack([full_H.H(float(R)) for R in grid.radial_points])
    systems = [_system(monomer, eta, pes=pes, magnetic_dipole_coefficient=0.0002) for eta in (1, -1)]
    transforms = [_transform(system.basis) for system in systems]
    initialize = initialize_logD_capture if mode == "capture" else initialize_logD_inelastic
    Y_initial = np.zeros((len(energies), full_system.n_channel, full_system.n_channel), dtype=np.complex128)
    # Diagonal WKB initialization is basis-dependent: impose precisely the
    # same physical inner boundary before comparing full and reduced solves.
    for T in transforms:
        projected = T.T @ W_base[0] @ T
        Y = np.asarray(initialize(projected[None] - 2 * full_H.reduced_mass * energies[:, None, None] * np.eye(T.shape[1])))
        Y_initial += T @ Y @ T.T
    full_Y = propagate_logD(Y_initial, energies, full_H.reduced_mass, np.diff(grid.radial_points)[::2], W_base[:-1:2], W_base[1::2], W_base[2::2])
    full_result = finalize_scattering(full_system.basis, np.asarray(full_Y), energies, full_H.reduced_mass, grid.radial_points[-1])
    for eta, system, T in zip((1, -1), systems, transforms, strict=True):
        result = ticc.solve(system, energies, grid, propagation)
        np.testing.assert_allclose(result.Y_propagated, T.T @ full_result.Y_propagated @ T, atol=2e-10, rtol=2e-10)
        T_sf = full_result.asymptotic_transform.T @ T @ result.asymptotic_transform
        for e, (full_open, adapted_open) in enumerate(zip(full_result.open_channel_indices, result.open_channel_indices, strict=True)):
            transform = T_sf[np.ix_(full_open, adapted_open)]
            np.testing.assert_allclose(result.Smat[e], transform.T @ full_result.Smat[e] @ transform, atol=2e-10)
            if mode == "inelastic":
                np.testing.assert_allclose(result.Smat[e].conj().T @ result.Smat[e], np.eye(len(adapted_open)), atol=2e-10)
            else:
                assert np.all(np.sum(abs(result.Smat[e]) ** 2, axis=0) <= 1 + 1e-10)
                if e == 1:
                    assert np.any(np.sum(abs(result.Smat[e]) ** 2, axis=0) < 0.9)
        assert result.molecule_exchange == eta
        assert isinstance(system.basis, ticc.FSDiatomDiatomBasis)
        assert f"molecule_exchange={eta:+d}" in ticc.report.channels(system.basis)
        assert f"molecule_exchange={eta:+d}" in ticc.report.smatrix(result)


def test_invalid_models_cutoffs_and_cached_data_are_rejected() -> None:
    monomer = _monomer()
    for eta in (True, 1.0, 2):
        with pytest.raises(ValueError, match="molecule_exchange must"):
            ticc.build_fs_diatom_diatom_channels(monomer, monomer, 2, 1, molecule_exchange=eta)
    with pytest.raises(ValueError, match="same FS monomer"):
        ticc.build_fs_diatom_diatom_channels(monomer, _monomer(), 2, 1, molecule_exchange=1)
    with pytest.raises(ValueError, match="same diatomic monomer"):
        replace(_system(monomer, 1), monomer_Y=_monomer())
    with pytest.raises(ValueError, match="exchange-closed energy cutoffs"):
        _system(monomer, 1, channel=ticc.ChannelSpec(E_X_cut=0.0001))
    system = _system(monomer, 1)
    with pytest.raises(ValueError, match="already molecule-exchange"):
        adapt_fs_molecule_exchange(system.basis, 1)
    with pytest.raises(ValueError, match="exact CC"):
        _system(monomer, 1, approx=ticc.Approx.CS)
    with pytest.raises(ValueError, match="polar quadratures"):
        build_hamiltonian(system, n_theta_X=3, n_theta_Y=4)
    grid = ticc.prepare_potential(system, (3, 3.2), (0.1,), n_theta_X=4, n_theta_Y=4, n_phi=6)
    with pytest.raises(ValueError, match="different molecule_exchange"):
        build_hamiltonian(replace(system, molecule_exchange=-1), potential_grid=grid)
    coordinates = tuple((name, values + 0.1 if name == "r_X" else values) for name, values in grid.coordinates)
    with pytest.raises(ValueError, match="cached radial grids"):
        build_hamiltonian(system, potential_grid=replace(grid, coordinates=coordinates))
    hamiltonian = build_hamiltonian(system, potential_grid=grid)
    with pytest.raises(ValueError, match="exact CC"):
        replace(hamiltonian, approx=ticc.Approx.NNCC)
    broken = replace(grid, values=grid.values + np.arange(grid.values.shape[-3])[None, None, None, :, None, None])
    with pytest.raises(ValueError, match="PES violates"):
        build_hamiltonian(system, potential_grid=broken)


def test_exchange_violating_spin_operator_is_not_silently_projected() -> None:
    monomer = _monomer()
    pes = _spin_pes(0, imaginary=False)
    bad = replace(pes, interaction=lambda R, q: pes.interaction(R, q) * np.cos(q[2])[:, None, None, None])
    system = _system(monomer, 1, pes=bad)
    options = dict(n_theta_X=5, n_theta_Y=5, n_phi=8)
    with pytest.raises(ValueError, match="exchange symmetry"):
        build_hamiltonian(system, **options).V(3.0)
    grid = ticc.prepare_potential(system, (3, 3.2), (0.1,), **options)
    hamiltonian = build_hamiltonian(system, potential_grid=grid)
    with pytest.raises(ValueError, match="exchange symmetry"):
        hamiltonian.V(3.0)
    with pytest.raises(ValueError, match="exchange symmetry"):
        hamiltonian.device_block_interaction(np.array([3.0]), (tuple(range(system.n_channel)),), jax.devices("cpu")[0])
    good_system = _system(monomer, 1, pes=pes)
    with pytest.raises(ValueError, match="finite and Hermitian"):
        build_hamiltonian(good_system, potential_grid=replace(grid, values=np.full(grid.values.shape, np.nan)))
    with pytest.raises(ValueError, match="electronic-axis order"):
        build_hamiltonian(replace(good_system, potential=replace(pes, two_total_spins=(2, 0))), potential_grid=grid)
    e = system.basis.exchange
    bad_matrix = np.diag(np.arange(len(e.source_channels), dtype=float))
    with pytest.raises(ValueError, match="exchange symmetry"):
        _project_exchange(system.basis, bad_matrix)


@pytest.mark.parametrize("mode", ["inelastic", "capture"])
def test_forbidden_exchange_block_returns_empty_result_without_propagating(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    system = _system(_monomer(0, 0, (0,)), -1, J=0)
    grid = ticc.prepare_potential(system, (3, 3.2), (0.1,), n_theta_X=3, n_theta_Y=3, n_phi=4)
    assert system.n_channel == 0

    def forbidden(*args, **kwargs):
        pytest.fail("Forbidden exchange block must not be propagated")

    monkeypatch.setattr("pyticc.scattering.solver.propagate", forbidden)
    result = ticc.solve(system, [0.001], grid, ticc.Propagation(mode=mode))
    assert result.Smat[0].shape == (0, 0)
    assert result.molecule_exchange == -1
    assert "No allowed channels" in ticc.report.smatrix(result)
