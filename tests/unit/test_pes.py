import numpy as np
import pytest

from pyticc.pes import PESWrapper, get_Vgrid_atom_diatom, get_Vgrid_diatom_diatom


def ArHF_interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
    r, theta = coordinates
    return np.exp(-R) * (r + np.cos(theta))


def HF_monomer(r: np.ndarray) -> np.ndarray:
    return (r - 1.75) ** 2


def test_ArHF_python_pes_builds_atom_diatom_grid() -> None:
    pes = PESWrapper(interaction=ArHF_interaction, monomer_Y=HF_monomer)
    r = np.array([1.5, 2.0])
    theta = np.array([0.0, 0.5 * np.pi, np.pi])

    V = get_Vgrid_atom_diatom(pes, R=6.0, r=r, theta=theta)

    expected = np.exp(-6.0) * (r[:, None] + np.cos(theta)[None, :])
    np.testing.assert_allclose(V, expected)
    assert pes.monomer_Y is not None
    np.testing.assert_allclose(pes.monomer_Y(r), (r - 1.75) ** 2)


def test_diatom_diatom_grid_uses_interaction_coordinate_order() -> None:
    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        coefficients = np.arange(1.0, 6.0)[:, None]
        return R + np.sum(coefficients * coordinates, axis=0)

    pes = PESWrapper(interaction=interaction)
    axes = tuple(np.array([value]) for value in (1.0, 2.0, 0.3, 0.4, 0.5))

    V = get_Vgrid_diatom_diatom(pes, 7.0, *axes)

    assert V.shape == (1, 1, 1, 1, 1)
    assert V.item() == pytest.approx(7.0 + 1.0 + 4.0 + 0.9 + 1.6 + 2.5)


def test_pes_wrapper_rejects_wrong_interaction_output_shape() -> None:
    pes = PESWrapper(interaction=lambda R, coordinates: np.zeros((coordinates.shape[1], 1)))

    with pytest.raises(ValueError, match="Interaction PES returned shape"):
        get_Vgrid_atom_diatom(pes, 5.0, np.ones(2), np.ones(3))
