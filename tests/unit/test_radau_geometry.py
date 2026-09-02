import numpy as np

from pyticc.pes.radau import atom_triatom_cartesian, radau_triatom_cartesian


def _canonical_point(cartesian: np.ndarray, masses: tuple[float, float, float]) -> np.ndarray:
    """Reconstruct the Radau canonical point from Cartesian A, B, and C positions."""
    mass_A, mass_B, mass_C = masses
    mass_ABC = mass_A + mass_B + mass_C
    center_AC = (mass_A * cartesian[0] + mass_C * cartesian[2]) / (mass_A + mass_C)
    scale = 1.0 - np.sqrt(mass_ABC / mass_B)
    return (cartesian[1] - scale * center_AC) / (1.0 - scale)


def test_radau_triatom_cartesian_uses_corrected_bisector_z_embedding() -> None:
    masses = (1.0, 16.0, 2.0)
    r_1 = np.array([1.7, 2.1])
    r_2 = np.array([2.3, 1.8])
    theta = np.array([1.2, 2.0])
    cartesian = radau_triatom_cartesian(np.stack((r_1, r_2, theta)), masses)

    center_of_mass = np.einsum("a,axg->xg", np.asarray(masses), cartesian) / sum(masses)
    canonical = _canonical_point(cartesian, masses)
    vector_1 = cartesian[0] - canonical
    vector_2 = cartesian[2] - canonical
    bisector_z = -(r_1[None, :] * vector_2 + r_2[None, :] * vector_1)
    bisector_z /= np.linalg.norm(bisector_z, axis=0)
    normal_y = np.cross(vector_1.T, vector_2.T).T
    normal_y /= np.linalg.norm(normal_y, axis=0)

    np.testing.assert_allclose(center_of_mass, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(np.linalg.norm(vector_1, axis=0), r_1, atol=1.0e-15)
    np.testing.assert_allclose(np.linalg.norm(vector_2, axis=0), r_2, atol=1.0e-15)
    np.testing.assert_allclose(np.sum(vector_1 * vector_2, axis=0) / (r_1 * r_2), np.cos(theta), atol=1.0e-15)
    np.testing.assert_allclose(bisector_z, np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]), atol=1.0e-15)
    np.testing.assert_allclose(normal_y, np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]), atol=1.0e-15)


def test_atom_triatom_cartesian_orients_bisector_and_places_collision_atom() -> None:
    masses = (1.0, 16.0, 1.0)
    theta_2 = np.array([0.3, 1.1])
    coordinates = np.stack(
        (
            np.array([1.9, 2.0]),
            np.array([2.1, 1.8]),
            np.array([1.4, 1.7]),
            theta_2,
            np.array([0.2, 1.3]),
        )
    )
    cartesian = atom_triatom_cartesian(6.5, coordinates, masses)

    canonical = _canonical_point(cartesian[:3], masses)
    vector_1 = cartesian[0] - canonical
    vector_2 = cartesian[2] - canonical
    bisector_z = -(coordinates[0][None, :] * vector_2 + coordinates[1][None, :] * vector_1)
    bisector_z /= np.linalg.norm(bisector_z, axis=0)
    expected_z = np.stack((np.sin(theta_2), np.zeros_like(theta_2), np.cos(theta_2)))

    np.testing.assert_allclose(bisector_z, expected_z, atol=1.0e-15)
    np.testing.assert_allclose(cartesian[3, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(cartesian[3, 1], 0.0, atol=0.0)
    np.testing.assert_allclose(cartesian[3, 2], 6.5, atol=0.0)
