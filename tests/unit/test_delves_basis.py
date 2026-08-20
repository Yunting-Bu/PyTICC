import numpy as np
import pytest

from pyticc.basis.delves import (
    DelvesBasis,
    build_delves_qns,
    delves_angular_basis,
    delves_theta_basis,
    midpoint_quad,
    sine_basis,
    theta_max,
)
from pyticc.basis.monomer.delves import _resolve_delves_sizes


def sample_delves_basis() -> DelvesBasis:
    return DelvesBasis(
        mass=(1.0, 2.0, 3.0),
        Jtot=2,
        system_parity=1,
        exchange_parity=0,
        jmax=3,
        K_cut=2,
        E_max=1.0,
        rho_min=2.0,
        scaled_r_max=5.0,
        n_sine=4,
        n_vib_quad=12,
        n_gamma_quad=10,
        angular_qns=(),
    )


def test_midpoint_quad_returns_uniform_internal_nodes() -> None:
    grids, weights = midpoint_quad(0.0, 2.0, 4)

    np.testing.assert_allclose(grids, [0.25, 0.75, 1.25, 1.75])
    np.testing.assert_allclose(weights, 0.5)
    assert np.sum(weights) == pytest.approx(2.0)


def test_sine_basis_is_continuously_normalized() -> None:
    grids, weights = midpoint_quad(0.0, 3.0, 4000)
    values = sine_basis(0.0, 3.0, 5, grids)
    derivatives = sine_basis(0.0, 3.0, 5, grids, derivative=1)

    overlap = values.T @ (weights[:, None] * values)
    np.testing.assert_allclose(overlap, np.eye(5), atol=1.0e-12)
    assert values.shape == derivatives.shape == (4000, 5)


def test_theta_max_preserves_the_scaled_bond_limit() -> None:
    radii = np.array([2.0, 5.0, 10.0])
    limits = theta_max(radii, scaled_r_max=5.0)

    np.testing.assert_allclose(limits[:2], np.pi / 2.0)
    assert radii[2] * np.sin(limits[2]) == pytest.approx(5.0)


def test_delves_theta_basis_changes_with_rho_and_remains_orthonormal() -> None:
    basis = sample_delves_basis()
    theta_near, weights_near, values_near = delves_theta_basis(basis, rho=4.0)
    theta_far, weights_far, values_far = delves_theta_basis(basis, rho=10.0)

    assert theta_near[-1] < 0.5 * np.pi
    assert theta_far[-1] < np.arcsin(0.5)
    assert np.sum(weights_near) == pytest.approx(0.5 * np.pi)
    assert np.sum(weights_far) == pytest.approx(np.arcsin(0.5))
    np.testing.assert_allclose(values_near.T @ (weights_near[:, None] * values_near), np.eye(basis.n_sine), atol=1.0e-14)
    np.testing.assert_allclose(values_far.T @ (weights_far[:, None] * values_far), np.eye(basis.n_sine), atol=1.0e-14)


def test_delves_angular_basis_is_orthonormal_at_fixed_K() -> None:
    basis = sample_delves_basis()
    cos_gamma, weights, values = delves_angular_basis(basis)

    assert cos_gamma.shape == weights.shape == (basis.n_gamma_quad,)
    assert values.shape == (basis.n_gamma_quad, basis.jmax + 1, basis.K_cut + 1)
    for K in range(basis.K_cut + 1):
        block = values[:, K:, K]
        np.testing.assert_allclose(block.T @ (weights[:, None] * block), np.eye(basis.jmax - K + 1), atol=1.0e-14)
        np.testing.assert_allclose(values[:, :K, K], 0.0)


def test_build_delves_qns_applies_parity_and_exchange_rules() -> None:
    masses = (1.0, 1.0, 1.0)
    qns = build_delves_qns(masses, Jtot=2, system_parity=1, exchange_parity=1, jmax=3, K_cut=2)

    assert {arrangement for arrangement, _, _ in qns} == {1, 2}
    assert {j for arrangement, j, _ in qns if arrangement == 1} == {0, 2}
    assert {j for arrangement, j, _ in qns if arrangement == 2} == {0, 1, 2, 3}
    assert (1, 2, 2) in qns

    heteronuclear = build_delves_qns((1.0, 2.0, 3.0), Jtot=0, system_parity=1, exchange_parity=1, jmax=1, K_cut=0)
    assert {arrangement for arrangement, _, _ in heteronuclear} == {1, 2, 3}
    assert {j for arrangement, j, _ in heteronuclear if arrangement == 1} == {0, 1}


def test_resolve_delves_sizes_matches_the_abc_integer_rules() -> None:
    masses = (np.sqrt(3.0) / 2.0,) * 3
    sampled_potential = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0])

    def asymptotic_potential(arrangement: int, scaled_r: np.ndarray) -> np.ndarray:
        assert arrangement in (1, 2, 3)
        assert scaled_r.shape == sampled_potential.shape
        return sampled_potential

    _, rho_min, scaled_r_max, n_sine, n_vib_quad, n_gamma_quad = _resolve_delves_sizes(
        asymptotic_potential,
        masses,
        jmax=2,
        E_max=1.0,
        scaled_r_step=1.0,
        scaled_r_scan_max=10.0,
        tail_cut=2.0,
    )

    assert rho_min == pytest.approx(np.sqrt(2.0))
    assert scaled_r_max == pytest.approx(9.0)
    assert n_sine == 5
    assert n_vib_quad == 11
    assert n_gamma_quad == 9


def test_resolve_delves_sizes_rejects_an_unresolved_outer_tail() -> None:
    def dissociating_potential(arrangement: int, scaled_r: np.ndarray) -> np.ndarray:
        del arrangement
        return np.zeros_like(scaled_r)

    with pytest.raises(ValueError, match="outer forbidden region"):
        _resolve_delves_sizes(
            dissociating_potential,
            (1.0, 1.0, 1.0),
            jmax=0,
            E_max=1.0,
            scaled_r_step=1.0,
            scaled_r_scan_max=10.0,
            tail_cut=2.0,
        )
