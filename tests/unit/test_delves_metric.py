import numpy as np
import pytest

from pyticc.basis.delves import DelvesBasis, build_delves_qns, delves_theta_basis, sine_basis, theta_max
from pyticc.matrix.delves_metric import _cross_arrangement_overlap, get_sector_overlap_delves, get_sector_transform_delves
from pyticc.matrix.delves_surface import get_surface_matrices_delves, solve_surface_delves
from pyticc.pes.total import TotalPES


def make_basis(
    *,
    mass: tuple[float, float, float] = (2.0, 3.0, 5.0),
    exchange_parity: int = 0,
    jmax: int = 1,
    n_sine: int = 3,
    n_quad: int = 100,
) -> DelvesBasis:
    return DelvesBasis(
        mass=mass,
        Jtot=0,
        system_parity=1,
        exchange_parity=exchange_parity,
        jmax=jmax,
        K_cut=0,
        E_max=1.0,
        rho_min=2.0,
        scaled_r_max=5.0,
        n_sine=n_sine,
        n_vib_quad=n_quad,
        n_gamma_quad=n_quad,
        angular_qns=build_delves_qns(mass, 0, 1, exchange_parity, jmax, 0),
    )


def test_equal_radius_sector_overlap_recovers_the_fixed_rho_metric() -> None:
    basis = make_basis(n_quad=100)
    rho = 7.0

    overlap = get_sector_overlap_delves(basis, rho, rho)

    np.testing.assert_allclose(overlap, overlap.T, atol=8.0e-5)
    arrangements = tuple(dict.fromkeys(arrangement for arrangement, _, _ in basis.angular_qns))
    start = 0
    for arrangement in arrangements:
        size = sum(value == arrangement for value, _, _ in basis.angular_qns) * basis.n_sine
        np.testing.assert_allclose(overlap[start : start + size, start : start + size], np.eye(size), atol=1.0e-13)
        start += size


def test_same_arrangement_sector_overlap_uses_source_grid_and_target_boundary() -> None:
    basis = make_basis(jmax=0, n_quad=120)
    rho_a = 6.0
    rho_b = 9.0
    overlap = get_sector_overlap_delves(basis, rho_a, rho_b)
    block_size = basis.n_sine
    block = overlap[:block_size, :block_size]

    theta_a, weights_a, sine_a = delves_theta_basis(basis, rho_a)
    limit_b = float(theta_max(rho_b, basis.scaled_r_max))
    sine_b = sine_basis(0.0, limit_b, basis.n_sine, theta_a)
    expected = sine_a.T @ (weights_a[:, None] * (theta_a < limit_b)[:, None] * sine_b)

    np.testing.assert_allclose(block, expected, atol=1.0e-14)
    assert not np.allclose(block, np.eye(block_size), atol=1.0e-3)


def test_reverse_sector_overlap_converges_to_the_transpose() -> None:
    coarse = make_basis(n_quad=30)
    fine = make_basis(n_quad=120)

    coarse_error = np.linalg.norm(get_sector_overlap_delves(coarse, 6.5, 7.0) - get_sector_overlap_delves(coarse, 7.0, 6.5).T)
    overlap_ab = get_sector_overlap_delves(fine, 6.5, 7.0)
    overlap_ba = get_sector_overlap_delves(fine, 7.0, 6.5)
    fine_error = np.linalg.norm(overlap_ab - overlap_ba.T)

    assert fine_error < 0.35 * coarse_error
    assert fine_error < 4.0e-3


def test_exchange_image_is_added_to_the_sector_overlap_two_block() -> None:
    symmetric = make_basis(mass=(2.0, 2.0, 2.0), exchange_parity=1, jmax=2, n_quad=80)
    rho_a, rho_b = 6.5, 7.0

    full_overlap = get_sector_overlap_delves(symmetric, rho_a, rho_b)
    block_2_size = sum(arrangement == 2 for arrangement, _, _ in symmetric.angular_qns) * symmetric.n_sine
    block_2 = full_overlap[-block_2_size:, -block_2_size:]
    image_23 = _cross_arrangement_overlap(symmetric, rho_a, rho_b, 2, 3)

    np.testing.assert_allclose(block_2 - image_23, np.kron(np.eye(3), block_2[:3, :3] - image_23[:3, :3]), atol=1.0e-13)
    assert np.linalg.norm(image_23) > 1.0e-4


def test_sector_transform_has_old_by_new_surface_shape() -> None:
    basis = make_basis(jmax=0, n_sine=2, n_quad=60)
    rng = np.random.default_rng(3)
    coefficients_a = rng.normal(size=(basis.n_primitive, 4))
    coefficients_b = rng.normal(size=(basis.n_primitive, 3))
    primitive_overlap = get_sector_overlap_delves(basis, 6.5, 7.0)

    transform = get_sector_transform_delves(basis, 6.5, coefficients_a, 7.0, coefficients_b)

    assert transform.shape == (4, 3)
    np.testing.assert_allclose(transform, coefficients_a.T @ primitive_overlap @ coefficients_b, atol=1.0e-14)


def test_sector_transform_connects_actual_adjacent_surface_solutions() -> None:
    basis = make_basis(n_sine=2, n_quad=120)
    pes = TotalPES(lambda bonds: 0.01 * bonds[0] + 0.02 * bonds[1] + 0.03 * bonds[2])
    rho_a, rho_b = 6.8, 7.0
    H_a, S_a = get_surface_matrices_delves(basis, pes, rho_a)
    H_b, S_b = get_surface_matrices_delves(basis, pes, rho_b)
    _, coefficients_a, _ = solve_surface_delves(H_a, S_a)
    _, coefficients_b, _ = solve_surface_delves(H_b, S_b)

    identity_transform = get_sector_transform_delves(basis, rho_a, coefficients_a, rho_a, coefficients_a)
    transform_ab = get_sector_transform_delves(basis, rho_a, coefficients_a, rho_b, coefficients_b)
    transform_ba = get_sector_transform_delves(basis, rho_b, coefficients_b, rho_a, coefficients_a)

    np.testing.assert_allclose(identity_transform, np.eye(identity_transform.shape[0]), atol=4.0e-5)
    np.testing.assert_allclose(transform_ab, transform_ba.T, atol=9.0e-5)


def test_sector_metric_validates_inputs() -> None:
    basis = make_basis()
    with pytest.raises(ValueError, match="rho_a"):
        get_sector_overlap_delves(basis, 0.0, 7.0)
    with pytest.raises(ValueError, match="coefficients_a"):
        get_sector_transform_delves(basis, 6.5, np.ones((2, 2)), 7.0, np.ones((basis.n_primitive, 2)))
