import numpy as np
import pytest

from pyticc.basis.delves import DelvesBasis, build_delves_qns, delves_angular_basis, delves_theta_basis
from pyticc.matrix.delves import get_Vgrid_delves
from pyticc.matrix.delves_hamiltonian import get_Hmat_delves, get_Hmat_delves_K
from pyticc.pes.total import TotalPES


def make_basis(*, Jtot: int = 0, jmax: int = 0, K_cut: int = 0, n_sine: int = 4) -> DelvesBasis:
    mass = (1.0, 2.0, 3.0)
    return DelvesBasis(
        mass=mass,
        Jtot=Jtot,
        system_parity=1,
        exchange_parity=0,
        jmax=jmax,
        K_cut=K_cut,
        E_max=1.0,
        rho_min=2.0,
        scaled_r_max=5.0,
        n_sine=n_sine,
        n_vib_quad=40,
        n_gamma_quad=20,
        angular_qns=build_delves_qns(mass, Jtot, 1, 0, jmax, K_cut),
    )


def test_delves_K_block_reduces_to_the_exact_theta_kinetic_energy() -> None:
    basis = make_basis(n_sine=4)
    rho = 4.0
    zero_pes = TotalPES(lambda bonds: np.zeros(bonds.shape[1]))

    H = get_Hmat_delves_K(basis, zero_pes, rho, arrangement=1, K=0)

    reduced_mass = np.sqrt(np.prod(basis.mass) / np.sum(basis.mass))
    modes = np.arange(1, basis.n_sine + 1)
    expected = ((2.0 * modes) ** 2 - 0.25) / (2.0 * reduced_mass * rho**2)
    np.testing.assert_allclose(H, np.diag(expected), atol=1.0e-13)


def test_constant_total_potential_shifts_the_primitive_block_by_identity() -> None:
    basis = make_basis(jmax=2, n_sine=3)
    rho = 8.0
    zero_pes = TotalPES(lambda bonds: np.zeros(bonds.shape[1]))
    constant_pes = TotalPES(lambda bonds: np.full(bonds.shape[1], 0.37))

    H_zero = get_Hmat_delves_K(basis, zero_pes, rho, arrangement=2, K=0)
    H_constant = get_Hmat_delves_K(basis, constant_pes, rho, arrangement=2, K=0)

    np.testing.assert_allclose(H_constant - H_zero, 0.37 * np.eye(H_zero.shape[0]), atol=1.0e-13)


def test_delves_K_block_matches_direct_product_quadrature() -> None:
    basis = make_basis(jmax=2, n_sine=3)
    rho = 7.0
    pes = TotalPES(lambda bonds: 0.01 * bonds[0] ** 2 + 0.02 * bonds[1] + 0.03 * bonds[2])
    theta, theta_weights, sine = delves_theta_basis(basis, rho)
    cos_gamma, gamma_weights, angular_all = delves_angular_basis(basis)
    potential = get_Vgrid_delves(pes, rho, 3, theta, cos_gamma, basis.mass)
    angular = angular_all[:, :, 0]
    n_sine = basis.n_sine
    expected_potential = np.zeros((3 * n_sine, 3 * n_sine))

    for ja in range(3):
        for jb in range(3):
            angular_integral = np.sum(gamma_weights[None, :] * potential * angular[:, ja][None, :] * angular[:, jb][None, :], axis=1)
            block = sine.T @ (theta_weights[:, None] * angular_integral[:, None] * sine)
            expected_potential[ja * n_sine : (ja + 1) * n_sine, jb * n_sine : (jb + 1) * n_sine] = block

    H = get_Hmat_delves_K(basis, pes, rho, arrangement=3, K=0)
    H_zero = get_Hmat_delves_K(basis, TotalPES(lambda bonds: np.zeros(bonds.shape[1])), rho, arrangement=3, K=0)

    np.testing.assert_allclose(H - H_zero, expected_potential, atol=1.0e-13)
    np.testing.assert_allclose(H, H.T, atol=1.0e-14)


def test_delves_K_block_uses_only_quantum_numbers_present_in_the_arrangement() -> None:
    mass = (1.0, 1.0, 1.0)
    qns = build_delves_qns(mass, Jtot=0, system_parity=1, exchange_parity=1, jmax=3, K_cut=0)
    basis = DelvesBasis(
        mass=mass,
        Jtot=0,
        system_parity=1,
        exchange_parity=1,
        jmax=3,
        K_cut=0,
        E_max=1.0,
        rho_min=2.0,
        scaled_r_max=5.0,
        n_sine=2,
        n_vib_quad=20,
        n_gamma_quad=12,
        angular_qns=qns,
    )
    zero_pes = TotalPES(lambda bonds: np.zeros(bonds.shape[1]))

    H = get_Hmat_delves_K(basis, zero_pes, rho=8.0, arrangement=1, K=0)

    assert H.shape == (4, 4)  # j=0,2 and two sine functions
    with pytest.raises(ValueError, match="no primitive"):
        get_Hmat_delves_K(basis, zero_pes, rho=8.0, arrangement=3, K=0)


def test_complete_arrangement_reuses_one_pes_grid_and_contains_every_K_block() -> None:
    basis = make_basis(Jtot=2, jmax=2, K_cut=2, n_sine=3)
    calls: list[np.ndarray] = []

    def potential(bonds: np.ndarray) -> np.ndarray:
        calls.append(bonds.copy())
        return 0.01 * bonds[0] + 0.02 * bonds[1] + 0.03 * bonds[2]

    pes = TotalPES(potential)
    H = get_Hmat_delves(basis, pes, rho=8.0, arrangement=1)
    qns = [(j, K) for arrangement, j, K in basis.angular_qns if arrangement == 1]

    assert H.shape == (len(qns) * basis.n_sine,) * 2
    assert len(calls) == 1
    for K in range(3):
        j_values = [j for j, value_K in qns if value_K == K]
        indices = np.concatenate([np.arange(qns.index((j, K)) * basis.n_sine, (qns.index((j, K)) + 1) * basis.n_sine) for j in j_values])
        expected = get_Hmat_delves_K(basis, pes, rho=8.0, arrangement=1, K=K)
        np.testing.assert_allclose(H[np.ix_(indices, indices)], expected, atol=1.0e-13)


def test_complete_arrangement_has_the_abc_parity_adapted_coriolis_couplings() -> None:
    basis = make_basis(Jtot=2, jmax=2, K_cut=2, n_sine=3)
    rho = 8.0
    H = get_Hmat_delves(basis, TotalPES(lambda bonds: np.zeros(bonds.shape[1])), rho, arrangement=1)
    qns = [(j, K) for arrangement, j, K in basis.angular_qns if arrangement == 1]
    theta, weights, sine = delves_theta_basis(basis, rho)
    reduced_mass = np.sqrt(np.prod(basis.mass) / np.sum(basis.mass))
    inverse_cosine = sine.T @ (weights[:, None] * sine / np.cos(theta[:, None]) ** 2)
    radial_factor = 1.0 / (2.0 * reduced_mass * rho**2)

    for j, K, coefficient in ((1, 0, -np.sqrt(24.0)), (2, 0, -np.sqrt(72.0)), (2, 1, -4.0)):
        lower = qns.index((j, K)) * basis.n_sine
        upper = qns.index((j, K + 1)) * basis.n_sine
        coupling = H[lower : lower + basis.n_sine, upper : upper + basis.n_sine]
        np.testing.assert_allclose(coupling, coefficient * radial_factor * inverse_cosine, atol=1.0e-13)

    j_1_K_0 = qns.index((1, 0)) * basis.n_sine
    j_2_K_1 = qns.index((2, 1)) * basis.n_sine
    np.testing.assert_allclose(H[j_1_K_0 : j_1_K_0 + basis.n_sine, j_2_K_1 : j_2_K_1 + basis.n_sine], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(H, H.T, atol=1.0e-14)
