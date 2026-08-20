import numpy as np
import pytest

from pyticc.basis.monomer.delves import _resolve_delves_sizes
from pyticc.matrix.delves import asymptotic_potential, delves_bonds, get_Vgrid_delves, mass_scale, transform_delves_coordinates
from pyticc.pes.total import TotalPES


def test_mass_scale_matches_the_abc_ooh_values() -> None:
    reduced_mass, scale = mass_scale((15.99492, 15.99492, 1.007825))

    assert reduced_mass == pytest.approx(2.795329651, abs=5.0e-10)
    np.testing.assert_allclose(scale, [1.717087539, 1.717087539, 0.591208239], rtol=1.0e-9)


def test_delves_bonds_places_each_arrangement_diatom_in_the_right_slot() -> None:
    mass = (2.0, 3.0, 5.0)
    _, scale = mass_scale(mass)
    scaled_r = 4.0

    arrangement_1 = delves_bonds(20.0, scaled_r, 0.0, 1, mass)
    arrangement_2 = delves_bonds(20.0, scaled_r, 0.0, 2, mass)
    arrangement_3 = delves_bonds(20.0, scaled_r, 0.0, 3, mass)

    assert arrangement_1[1] == pytest.approx(scale[0] * scaled_r)  # BC
    assert arrangement_2[2] == pytest.approx(scale[1] * scaled_r)  # CA
    assert arrangement_3[0] == pytest.approx(scale[2] * scaled_r)  # AB


@pytest.mark.parametrize("arrangement", [1, 2, 3])
def test_delves_bonds_reconstructs_one_geometry_in_every_arrangement(arrangement: int) -> None:
    mass = np.array([2.0, 3.0, 5.0])
    position = np.array(
        [
            [1.2, -0.7, 0.4],
            [-0.3, 0.8, -1.1],
            [0.6, 1.5, 0.9],
        ]
    )
    expected = np.array(
        [
            np.linalg.norm(position[0] - position[1]),
            np.linalg.norm(position[1] - position[2]),
            np.linalg.norm(position[2] - position[0]),
        ]
    )
    _, scale = mass_scale(mass)
    ia = arrangement - 1
    ib = (ia + 1) % 3
    ic = 3 - ia - ib
    diatom_vector = position[ic] - position[ib]
    diatom_center = (mass[ib] * position[ib] + mass[ic] * position[ic]) / (mass[ib] + mass[ic])
    separation_vector = position[ia] - diatom_center
    physical_r = np.linalg.norm(diatom_vector)
    physical_R = np.linalg.norm(separation_vector)
    cos_gamma = np.dot(separation_vector, diatom_vector) / (physical_R * physical_r)

    bonds = delves_bonds(
        scale[ia] * physical_R,
        physical_r / scale[ia],
        cos_gamma,
        arrangement,
        mass,
    )

    np.testing.assert_allclose(bonds, expected, rtol=1.0e-14, atol=1.0e-14)


def test_delves_bonds_broadcasts_and_returns_ab_bc_ca_order() -> None:
    bonds = delves_bonds(
        np.array([[8.0], [10.0]]),
        np.array([1.0, 2.0, 3.0]),
        0.0,
        1,
        (1.0, 2.0, 3.0),
    )

    assert bonds.shape == (3, 2, 3)
    assert np.all(bonds >= 0.0)


@pytest.mark.parametrize("arrangement_a", [1, 2, 3])
@pytest.mark.parametrize("arrangement_b", [1, 2, 3])
def test_delves_coordinate_transform_preserves_geometry_and_is_reversible(arrangement_a: int, arrangement_b: int) -> None:
    mass = (2.0, 3.0, 5.0)
    theta_a = np.array([[0.2], [0.4], [0.6]])
    cos_gamma_a = np.array([[-0.7, 0.0, 0.8]])

    theta_b, cos_gamma_b, beta_ab = transform_delves_coordinates(theta_a, cos_gamma_a, arrangement_a, arrangement_b, mass)
    theta_back, cos_gamma_back, beta_ba = transform_delves_coordinates(theta_b, cos_gamma_b, arrangement_b, arrangement_a, mass)
    bonds_a = delves_bonds(np.cos(theta_a), np.sin(theta_a), cos_gamma_a, arrangement_a, mass)
    bonds_b = delves_bonds(np.cos(theta_b), np.sin(theta_b), cos_gamma_b, arrangement_b, mass)

    np.testing.assert_allclose(bonds_b, bonds_a, atol=1.0e-14)
    np.testing.assert_allclose(theta_back, np.broadcast_to(theta_a, theta_b.shape), atol=1.0e-14)
    np.testing.assert_allclose(cos_gamma_back, np.broadcast_to(cos_gamma_a, cos_gamma_b.shape), atol=1.0e-14)
    np.testing.assert_allclose(beta_ba, -beta_ab, atol=1.0e-14)


def test_delves_coordinate_transform_validates_inputs() -> None:
    with pytest.raises(ValueError, match="arrangements"):
        transform_delves_coordinates(0.2, 0.0, 0, 1, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="theta_a"):
        transform_delves_coordinates(0.5 * np.pi, 0.0, 1, 2, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="cos_gamma_a"):
        transform_delves_coordinates(0.2, 1.1, 1, 2, (1.0, 1.0, 1.0))


def test_asymptotic_potential_passes_physical_bonds_to_the_total_pes() -> None:
    mass = (2.0, 3.0, 5.0)
    received: list[np.ndarray] = []

    def total_pes(bonds: np.ndarray) -> np.ndarray:
        assert bonds.flags.f_contiguous
        received.append(bonds.copy())
        return bonds[0] + 2.0 * bonds[1] + 3.0 * bonds[2]

    potential = asymptotic_potential(TotalPES(total_pes), mass, scaled_R=12.0, cos_gamma=0.25)
    scaled_r = np.array([1.0, 2.0, 3.0])
    values = potential(2, scaled_r)
    expected_bonds = delves_bonds(12.0, scaled_r, 0.25, 2, mass)

    np.testing.assert_allclose(received[0], expected_bonds)
    np.testing.assert_allclose(values, expected_bonds[0] + 2.0 * expected_bonds[1] + 3.0 * expected_bonds[2])


@pytest.mark.parametrize("arrangement", [1, 2, 3])
def test_delves_potential_grid_uses_fixed_rho_coordinates(arrangement: int) -> None:
    mass = (2.0, 3.0, 5.0)
    theta = np.array([0.2, 0.4, 0.6])
    cos_gamma = np.array([-0.75, 0.0, 0.75])
    rho = 8.0
    received: list[np.ndarray] = []

    def potential(bonds: np.ndarray) -> np.ndarray:
        received.append(bonds.copy())
        return bonds[0] + 2.0 * bonds[1] + 3.0 * bonds[2]

    values = get_Vgrid_delves(TotalPES(potential), rho, arrangement, theta, cos_gamma, mass)
    expected_bonds = delves_bonds(
        rho * np.cos(theta[:, None]),
        rho * np.sin(theta[:, None]),
        cos_gamma[None, :],
        arrangement,
        mass,
    )

    assert values.shape == (theta.size, cos_gamma.size)
    np.testing.assert_allclose(received[0], expected_bonds.reshape(3, -1))
    np.testing.assert_allclose(values, expected_bonds[0] + 2.0 * expected_bonds[1] + 3.0 * expected_bonds[2])
    np.testing.assert_allclose(
        (rho * np.cos(theta)) ** 2 + (rho * np.sin(theta)) ** 2,
        rho**2,
        atol=1.0e-14,
    )


def test_total_pes_adapter_connects_to_the_delves_size_scan() -> None:
    mass = (np.sqrt(3.0) / 2.0,) * 3
    _, scale = mass_scale(mass)

    def total_pes(bonds: np.ndarray) -> np.ndarray:
        # At scaled_R=100 the shortest bond is the arrangement diatom.
        scaled_r = np.min(bonds, axis=0) / scale[0]
        return np.where((scaled_r >= 3.0) & (scaled_r <= 7.0), 0.0, 5.0)

    _, rho_min, scaled_r_max, n_sine, n_vib_quad, n_gamma_quad = _resolve_delves_sizes(
        asymptotic_potential(TotalPES(total_pes), mass),
        mass,
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


def test_delves_coordinate_and_total_pes_validation() -> None:
    with pytest.raises(ValueError, match="arrangement"):
        delves_bonds(1.0, 1.0, 0.0, 0, (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="cos_gamma"):
        delves_bonds(1.0, 1.0, 1.1, 1, (1.0, 1.0, 1.0))

    potential = asymptotic_potential(TotalPES(lambda bonds: np.zeros((1, bonds.shape[1]))), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="Total PES returned shape"):
        potential(1, np.array([1.0, 2.0]))
