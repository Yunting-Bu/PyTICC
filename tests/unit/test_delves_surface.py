import numpy as np
import pytest

from pyticc.basis.delves import DelvesBasis, build_delves_qns
from pyticc.matrix.delves.coupling import get_HSmat_delves
from pyticc.matrix.delves.hamiltonian import get_Hmat_delves
from pyticc.matrix.delves.surface import get_surface_matrices_delves, solve_surface_delves
from pyticc.pes.total import TotalPES


def make_basis(
    *,
    mass: tuple[float, float, float] = (2.0, 3.0, 5.0),
    Jtot: int = 0,
    system_parity: int = 1,
    exchange_parity: int = 0,
    jmax: int = 1,
    K_cut: int = 0,
    n_sine: int = 2,
    n_quad: int = 40,
) -> DelvesBasis:
    return DelvesBasis(
        mass=mass,
        Jtot=Jtot,
        system_parity=system_parity,
        exchange_parity=exchange_parity,
        jmax=jmax,
        K_cut=K_cut,
        E_max=1.0,
        rho_min=2.0,
        scaled_r_max=5.0,
        n_sine=n_sine,
        n_vib_quad=n_quad,
        n_gamma_quad=n_quad,
        angular_qns=build_delves_qns(mass, Jtot, system_parity, exchange_parity, jmax, K_cut),
    )


def test_global_surface_matrices_follow_basis_order_and_are_symmetric() -> None:
    basis = make_basis()
    pes = TotalPES(lambda bonds: 0.01 * bonds[0] + 0.02 * bonds[1] + 0.03 * bonds[2])
    rho = 7.0

    H, S = get_surface_matrices_delves(basis, pes, rho)
    n_arrangement = basis.n_primitive // 3

    assert H.shape == S.shape == (basis.n_primitive, basis.n_primitive)
    np.testing.assert_allclose(H, H.T, atol=1.0e-14)
    np.testing.assert_allclose(S, S.T, atol=1.0e-14)
    for arrangement in (1, 2, 3):
        block = slice((arrangement - 1) * n_arrangement, arrangement * n_arrangement)
        np.testing.assert_allclose(H[block, block], get_Hmat_delves(basis, pes, rho, arrangement), atol=1.0e-13)
        np.testing.assert_array_equal(S[block, block], np.eye(n_arrangement))

    H_12, S_12 = get_HSmat_delves(basis, pes, rho, 1, 2)
    H_21, S_21 = get_HSmat_delves(basis, pes, rho, 2, 1)
    np.testing.assert_allclose(H[:n_arrangement, n_arrangement : 2 * n_arrangement], 0.5 * (H_12 + H_21.T), atol=1.0e-13)
    np.testing.assert_allclose(S[:n_arrangement, n_arrangement : 2 * n_arrangement], 0.5 * (S_12 + S_21.T), atol=1.0e-13)


def test_exchange_image_is_added_to_the_explicit_arrangement_two_block() -> None:
    basis = make_basis(mass=(2.0, 2.0, 2.0), exchange_parity=1, jmax=2)
    zero = TotalPES(lambda bonds: np.zeros(bonds.shape[1]))
    rho = 7.0

    H, S = get_surface_matrices_delves(basis, zero, rho)
    n_arrangement_1 = sum(arrangement == 1 for arrangement, _, _ in basis.angular_qns) * basis.n_sine
    block_2 = slice(n_arrangement_1, basis.n_primitive)
    H_23, S_23 = get_HSmat_delves(basis, zero, rho, 2, 3)

    expected_H = get_Hmat_delves(basis, zero, rho, 2) + 0.5 * (H_23 + H_23.T)
    expected_S = np.eye(block_2.stop - block_2.start) + 0.5 * (S_23 + S_23.T)
    np.testing.assert_allclose(H[block_2, block_2], expected_H, atol=1.0e-13)
    np.testing.assert_allclose(S[block_2, block_2], expected_S, atol=1.0e-13)


def test_global_builder_reuses_each_arrangement_pes_grids() -> None:
    basis = make_basis(n_quad=20)
    calls: list[np.ndarray] = []

    def potential(bonds: np.ndarray) -> np.ndarray:
        calls.append(bonds.copy())
        return np.zeros(bonds.shape[1])

    get_surface_matrices_delves(basis, TotalPES(potential), 7.0)

    full_size = basis.n_vib_quad * basis.n_gamma_quad
    assert [call.shape for call in calls] == [(3, full_size), (3, basis.n_vib_quad)] * 3


def test_canonical_surface_solver_matches_a_direct_generalized_problem() -> None:
    overlap_vectors = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    overlap_values = np.array([0.25, 1.5])
    S = (overlap_vectors * overlap_values) @ overlap_vectors.T
    exact_energies = np.array([0.7, 1.9])
    coefficients = overlap_vectors / np.sqrt(overlap_values)[None, :]
    H = S @ coefficients @ np.diag(exact_energies) @ coefficients.T @ S

    energies, solved_coefficients, sigma = solve_surface_delves(H, S)

    np.testing.assert_allclose(energies, exact_energies, atol=1.0e-14)
    np.testing.assert_allclose(sigma, overlap_values, atol=1.0e-14)
    np.testing.assert_allclose(solved_coefficients.T @ S @ solved_coefficients, np.eye(2), atol=1.0e-14)
    np.testing.assert_allclose(H @ solved_coefficients, S @ solved_coefficients @ np.diag(energies), atol=1.0e-14)


def test_canonical_surface_solver_removes_near_linear_dependence() -> None:
    S = np.diag([1.0e-6, 0.2, 1.5])
    H = np.diag([10.0e-6, 0.4, 4.5])

    energies, coefficients, sigma = solve_surface_delves(H, S)

    np.testing.assert_allclose(energies, [2.0, 3.0], atol=1.0e-14)
    np.testing.assert_allclose(sigma, [1.0e-6, 0.2, 1.5], atol=1.0e-14)
    assert coefficients.shape == (3, 2)
    np.testing.assert_allclose(coefficients.T @ S @ coefficients, np.eye(2), atol=1.0e-14)


def test_surface_solver_and_builder_validate_inputs() -> None:
    basis = make_basis()
    zero = TotalPES(lambda bonds: np.zeros(bonds.shape[1]))
    with pytest.raises(ValueError, match="rho"):
        get_surface_matrices_delves(basis, zero, 0.0)
    with pytest.raises(ValueError, match="square"):
        solve_surface_delves(np.zeros((2, 3)), np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        solve_surface_delves(np.array([[0.0, 1.0], [0.0, 0.0]]), np.eye(2))
    with pytest.raises(ValueError, match="No overlap"):
        solve_surface_delves(np.eye(2), 1.0e-8 * np.eye(2))
