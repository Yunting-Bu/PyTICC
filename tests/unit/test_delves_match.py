import numpy as np
import pytest

from pyticc.basis.delves import DelvesBasis
from pyticc.match.bessel import riccati_bessel_jy
from pyticc.match.delves import build_delves_asymptotic_basis, transform_logD_to_delves_channels
from pyticc.match.delves_bessel import _delves_reference_matrices, get_delves_frame_transform, get_delves_Smat, match_delves
from pyticc.matrix.delves import mass_scale
from pyticc.pes.total import TotalPES
from pyticc.propagation.delves import DelvesPropagationResult


def make_basis(*, E_max: float = 2.0, qns: tuple[tuple[int, int, int], ...] = ((1, 0, 0),)) -> DelvesBasis:
    return DelvesBasis(
        mass=(2.0, 3.0, 5.0),
        Jtot=0,
        system_parity=1,
        exchange_parity=0,
        jmax=max(j for _, j, _ in qns),
        K_cut=max(K for _, _, K in qns),
        E_max=E_max,
        rho_min=2.0,
        scaled_r_max=4.0,
        n_sine=3,
        n_vib_quad=30,
        n_gamma_quad=8,
        angular_qns=qns,
    )


def test_delves_asymptotic_basis_has_analytic_free_thresholds_and_order() -> None:
    basis = make_basis(qns=((1, 0, 0), (2, 0, 0)))
    channels = build_delves_asymptotic_basis(basis, TotalPES(lambda bonds: np.zeros(bonds.shape[1])), rho_match=20.0)
    reduced_mass, _ = mass_scale(basis.mass)
    expected = (np.pi * np.arange(1, basis.n_sine + 1) / basis.scaled_r_max) ** 2 / (2.0 * reduced_mass)

    assert channels.qns == ((1, 0, 0, 0), (1, 1, 0, 0), (1, 2, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 2, 0, 0))
    np.testing.assert_allclose(channels.energies, np.tile(expected, 2), atol=1.0e-14)
    assert channels.s_coefficients.shape == (3, 6)
    assert channels.theta_coefficients.shape == (6, 6)
    np.testing.assert_allclose(channels.s_coefficients[:, :3].T @ channels.s_coefficients[:, :3], np.eye(3), atol=1.0e-14)
    np.testing.assert_allclose(channels.s_coefficients[:, 3:].T @ channels.s_coefficients[:, 3:], np.eye(3), atol=1.0e-14)


def test_delves_asymptotic_basis_replicates_vibrational_states_over_K() -> None:
    basis = make_basis(E_max=0.8, qns=((1, 1, 0), (1, 1, 1)))
    channels = build_delves_asymptotic_basis(basis, TotalPES(lambda bonds: np.zeros(bonds.shape[1])), rho_match=20.0)

    assert channels.qns == ((1, 0, 1, 0), (1, 0, 1, 1))
    np.testing.assert_allclose(channels.energies[0], channels.energies[1], atol=0.0)
    np.testing.assert_allclose(channels.s_coefficients[:, 0], channels.s_coefficients[:, 1], atol=0.0)
    assert np.flatnonzero(channels.theta_coefficients[:, 0]).tolist() == [0, 1, 2]
    assert np.flatnonzero(channels.theta_coefficients[:, 1]).tolist() == [3, 4, 5]


def test_final_delves_channel_transform_is_a_congruence_at_equal_rho() -> None:
    basis = make_basis()
    channels = build_delves_asymptotic_basis(basis, TotalPES(lambda bonds: np.zeros(bonds.shape[1])), rho_match=20.0)
    Y = np.stack((np.diag([1.0, 2.0, 3.0]), np.diag([4.0, 5.0, 6.0])))

    transformed = transform_logD_to_delves_channels(
        basis,
        channels.rho_match,
        channels.theta_coefficients,
        Y,
        channels,
    )

    np.testing.assert_allclose(transformed, Y, atol=2.0e-14)


def test_delves_asymptotic_basis_validates_radius_and_cutoff() -> None:
    pes = TotalPES(lambda bonds: np.zeros(bonds.shape[1]))
    with pytest.raises(ValueError, match="rho_match"):
        build_delves_asymptotic_basis(make_basis(), pes, rho_match=0.0)
    with pytest.raises(ValueError, match="No Delves asymptotic channels"):
        build_delves_asymptotic_basis(make_basis(E_max=-1.0), pes, rho_match=20.0)


def test_delves_frame_transform_recovers_parity_allowed_integer_L() -> None:
    basis = DelvesBasis(
        mass=(2.0, 3.0, 5.0),
        Jtot=1,
        system_parity=1,
        exchange_parity=0,
        jmax=1,
        K_cut=1,
        E_max=1.0,
        rho_min=2.0,
        scaled_r_max=4.0,
        n_sine=1,
        n_vib_quad=20,
        n_gamma_quad=8,
        angular_qns=((1, 1, 0), (1, 1, 1)),
    )
    channels = build_delves_asymptotic_basis(basis, TotalPES(lambda bonds: np.zeros(bonds.shape[1])), rho_match=20.0)

    transform, orbital_L = get_delves_frame_transform(basis, channels)

    np.testing.assert_allclose(transform.T @ transform, np.eye(2), atol=1.0e-14)
    np.testing.assert_allclose(orbital_L, [0.0, 2.0], atol=0.0)
    assert transform[-1, 0] > 0.0
    assert transform[-1, 1] > 0.0


def test_delves_match_returns_unitary_open_channel_Smat() -> None:
    basis = make_basis(E_max=0.2)
    channels = build_delves_asymptotic_basis(basis, TotalPES(lambda bonds: np.zeros(bonds.shape[1])), rho_match=30.0)
    energy = channels.energies[0] + 0.05

    (Smat,) = get_delves_Smat(np.array([[[0.3]]]), [energy], basis, channels)

    assert Smat.shape == (1, 1)
    np.testing.assert_allclose(Smat.conj().T @ Smat, np.eye(1), atol=1.0e-13)


def test_delves_match_handles_closed_channels_and_empty_open_block() -> None:
    basis = make_basis(E_max=2.0)
    channels = build_delves_asymptotic_basis(basis, TotalPES(lambda bonds: np.zeros(bonds.shape[1])), rho_match=20.0)
    one_open = 0.5 * (channels.energies[0] + channels.energies[1])

    below, mixed = get_delves_Smat(
        np.stack((np.eye(3), 0.2 * np.eye(3))),
        [channels.energies[0] - 0.01, one_open],
        basis,
        channels,
    )

    assert below.shape == (0, 0)
    assert mixed.shape == (1, 1)
    np.testing.assert_allclose(mixed.conj().T @ mixed, np.eye(1), atol=1.0e-13)


def test_delves_projection_tends_to_fixed_R_bessel_matching_at_large_rho() -> None:
    basis = make_basis(E_max=0.2)
    channels = build_delves_asymptotic_basis(basis, TotalPES(lambda bonds: np.zeros(bonds.shape[1])), rho_match=1.0e5)
    energy = channels.energies[0] + 0.05
    regular, irregular, regular_prime, irregular_prime = _delves_reference_matrices(energy, basis, channels, np.array([0.0]))
    reduced_mass, _ = mass_scale(basis.mass)
    momentum = np.sqrt(2.0 * reduced_mass * (energy - channels.energies[0]))
    j_value, y_value, j_prime, y_prime = riccati_bessel_jy(0.0, momentum * channels.rho_match)

    expected = np.array([j_value / np.sqrt(momentum), y_value / np.sqrt(momentum), np.sqrt(momentum) * j_prime, np.sqrt(momentum) * y_prime])
    projected = np.array([regular[0, 0], irregular[0, 0], regular_prime[0, 0], irregular_prime[0, 0]])
    np.testing.assert_allclose(projected, expected, rtol=3.0e-5, atol=1.0e-8)


def test_match_delves_connects_propagation_result_to_channels_and_Smat() -> None:
    basis = make_basis(E_max=0.2)
    pes = TotalPES(lambda bonds: np.zeros(bonds.shape[1]))
    channels_at_boundary = build_delves_asymptotic_basis(basis, pes, rho_match=30.0)
    propagation = DelvesPropagationResult(
        Y_final=np.array([[[0.3]]]),
        rho_final=30.0,
        surface_rho=30.0,
        surface_energies=np.array([0.0]),
        surface_coefficients=channels_at_boundary.theta_coefficients,
        radial_points=np.array([2.0, 30.0]),
    )
    energy = channels_at_boundary.energies[0] + 0.05

    channels, (Smat,) = match_delves(propagation, [energy], basis, pes)

    assert channels.qns == channels_at_boundary.qns
    assert Smat.shape == (1, 1)
    np.testing.assert_allclose(Smat.conj().T @ Smat, np.eye(1), atol=1.0e-13)
