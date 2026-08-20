import numpy as np
import pytest

from pyticc.basis.angle import norm_YjK
from pyticc.basis.delves import DelvesBasis, build_delves_qns, delves_angular_basis, delves_theta_basis, sine_basis, theta_max
from pyticc.matrix.delves import mass_scale, transform_delves_coordinates
from pyticc.matrix.delves_overlap import _cross_integrals, get_HSmat_delves, get_Smat_delves, parity_rotation
from pyticc.pes.total import TotalPES


def make_basis(
    *,
    mass: tuple[float, float, float] = (2.0, 3.0, 5.0),
    Jtot: int = 1,
    system_parity: int = -1,
    exchange_parity: int = 0,
    jmax: int = 2,
    K_cut: int = 1,
    n_quad: int = 80,
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
        n_sine=3,
        n_vib_quad=n_quad,
        n_gamma_quad=n_quad,
        angular_qns=build_delves_qns(mass, Jtot, system_parity, exchange_parity, jmax, K_cut),
    )


def test_parity_rotation_matches_the_J1_abc_convention() -> None:
    beta = np.array([-0.7, 0.0, 0.4])

    rotation = np.array([[parity_rotation(1, -1, K_a, K_b, beta) for K_b in range(2)] for K_a in range(2)])
    expected = np.array([[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]])

    np.testing.assert_allclose(rotation, expected, atol=1.0e-14)


@pytest.mark.parametrize("Jtot", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("system_parity", [-1, 1])
def test_parity_rotation_is_orthogonal_in_the_allowed_K_space(Jtot: int, system_parity: int) -> None:
    K_min = 0 if system_parity == (-1) ** Jtot else 1
    K_values = range(K_min, Jtot + 1)
    rotation = np.array([[parity_rotation(Jtot, system_parity, K_a, K_b, 0.73) for K_b in K_values] for K_a in K_values])

    np.testing.assert_allclose(rotation @ rotation.T, np.eye(len(rotation)), atol=2.0e-14)


def test_same_arrangement_primitive_overlap_is_identity() -> None:
    basis = make_basis(n_quad=20)
    n_arrangement = sum(arrangement == 2 for arrangement, _, _ in basis.angular_qns) * basis.n_sine

    overlap = get_Smat_delves(basis, rho=7.0, arrangement_a=2, arrangement_b=2)

    np.testing.assert_array_equal(overlap, np.eye(n_arrangement))


def test_directed_cross_arrangement_overlaps_converge_to_transposes() -> None:
    coarse = make_basis(n_quad=30)
    fine = make_basis(n_quad=100)

    coarse_error = np.linalg.norm(get_Smat_delves(coarse, 7.0, 1, 2) - get_Smat_delves(coarse, 7.0, 2, 1).T)
    overlap_12 = get_Smat_delves(fine, 7.0, 1, 2)
    overlap_21 = get_Smat_delves(fine, 7.0, 2, 1)
    fine_error = np.linalg.norm(overlap_12 - overlap_21.T)

    assert overlap_12.shape == (15, 15)
    assert np.linalg.norm(overlap_12) > 0.1
    assert fine_error < 0.2 * coarse_error
    assert fine_error < 2.0e-4


def test_exchange_symmetric_blocks_include_abc_normalization_and_phase() -> None:
    mass = (2.0, 2.0, 2.0)
    full = make_basis(mass=mass, Jtot=0, system_parity=1, exchange_parity=0, jmax=2, K_cut=0, n_quad=80)
    symmetric = make_basis(mass=mass, Jtot=0, system_parity=1, exchange_parity=1, jmax=2, K_cut=0, n_quad=80)
    overlap_full_12 = get_Smat_delves(full, 7.0, 1, 2)
    overlap_symmetric_12 = get_Smat_delves(symmetric, 7.0, 1, 2)
    overlap_full_23 = get_Smat_delves(full, 7.0, 2, 3)
    overlap_symmetric_23 = get_Smat_delves(symmetric, 7.0, 2, 3)

    n_sine = full.n_sine
    even_j_rows = np.r_[0:n_sine, 2 * n_sine : 3 * n_sine]
    np.testing.assert_allclose(overlap_symmetric_12, np.sqrt(2.0) * overlap_full_12[even_j_rows], atol=1.0e-13)
    for j_b in range(3):
        column = slice(j_b * n_sine, (j_b + 1) * n_sine)
        np.testing.assert_allclose(overlap_symmetric_23[:, column], (-1) ** j_b * overlap_full_23[:, column], atol=1.0e-13)


def test_delves_overlap_validates_missing_arrangements() -> None:
    basis = make_basis()
    with pytest.raises(ValueError, match="arrangement"):
        get_Smat_delves(basis, 7.0, 0, 2)


def test_constant_total_potential_shifts_cross_hamiltonian_by_overlap() -> None:
    basis = make_basis(n_quad=60)
    zero = TotalPES(lambda bonds: np.zeros(bonds.shape[1]))
    constant = TotalPES(lambda bonds: np.full(bonds.shape[1], 0.37))

    H_zero, S_ab = get_HSmat_delves(basis, zero, 7.0, 1, 2)
    H_constant, S_constant = get_HSmat_delves(basis, constant, 7.0, 1, 2)

    np.testing.assert_allclose(S_constant, S_ab, atol=0.0)
    np.testing.assert_allclose(H_constant - H_zero, 0.37 * S_ab, atol=4.0e-15)


def test_cross_hamiltonian_evaluates_total_and_reference_pes_grids_once() -> None:
    basis = make_basis(n_quad=20)
    calls: list[np.ndarray] = []

    def potential(bonds: np.ndarray) -> np.ndarray:
        calls.append(bonds.copy())
        return 0.01 * bonds[0] + 0.02 * bonds[1] + 0.03 * bonds[2]

    H_ab, S_ab = get_HSmat_delves(basis, TotalPES(potential), 7.0, 1, 2)

    assert H_ab.shape == S_ab.shape == (15, 15)
    assert np.all(np.isfinite(H_ab))
    assert [call.shape for call in calls] == [(3, basis.n_vib_quad * basis.n_gamma_quad), (3, basis.n_vib_quad)]


def test_cross_hamiltonian_retains_the_Kcut_plus_one_coriolis_function() -> None:
    basis = make_basis(jmax=1, K_cut=0, n_quad=80)
    rho = 7.0
    zero = TotalPES(lambda bonds: np.zeros(bonds.shape[1]))
    _, _, coriolis, _ = _cross_integrals(basis, rho, 1, 2, zero)
    qns = [(j, K) for arrangement, j, K in basis.angular_qns if arrangement == 1]
    row = slice(qns.index((1, 0)) * basis.n_sine, (qns.index((1, 0)) + 1) * basis.n_sine)
    column = slice(qns.index((0, 0)) * basis.n_sine, (qns.index((0, 0)) + 1) * basis.n_sine)

    theta_a, theta_weights, sine_a = delves_theta_basis(basis, rho)
    cos_gamma_a, gamma_weights, _ = delves_angular_basis(basis)
    theta_b, cos_gamma_b, beta_ab = transform_delves_coordinates(theta_a[:, None], cos_gamma_a[None, :], 1, 2, basis.mass)
    limit = float(theta_max(rho, basis.scaled_r_max))
    valid = theta_b < limit
    jacobian = np.divide(
        np.sin(2.0 * theta_a[:, None]),
        np.sin(2.0 * theta_b),
        out=np.zeros_like(theta_b),
        where=valid,
    )
    jacobian *= theta_weights[:, None] * gamma_weights[None, :]
    sine_b = sine_basis(0.0, limit, basis.n_sine, theta_b.ravel()).reshape(*theta_b.shape, basis.n_sine)
    reduced_mass, _ = mass_scale(basis.mass)
    coefficient = -np.sqrt(2.0 * basis.Jtot * (basis.Jtot + 1) * 2.0)
    kernel = jacobian * norm_YjK(1, 1, cos_gamma_a)[None, :] * norm_YjK(0, 0, cos_gamma_b)
    kernel *= parity_rotation(1, -1, 1, 0, beta_ab) / np.cos(theta_a[:, None]) ** 2
    expected = coefficient / (2.0 * reduced_mass * rho**2) * np.einsum("qn,qpm,qp->nm", sine_a, sine_b, kernel, optimize=True)

    np.testing.assert_allclose(coriolis[row, column], expected, atol=1.0e-14)
    assert np.linalg.norm(expected) > 1.0e-5


def test_cross_hamiltonian_rejects_one_arrangement_block() -> None:
    basis = make_basis()
    zero = TotalPES(lambda bonds: np.zeros(bonds.shape[1]))
    with pytest.raises(ValueError, match="distinct arrangements"):
        get_HSmat_delves(basis, zero, 7.0, 1, 1)
