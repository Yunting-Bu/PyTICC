import numpy as np
import pytest

from pyticc.pes import DiabaticPESWrapper, get_diabatic_potential_grid_atom_diatom


def _monomer(r: np.ndarray) -> np.ndarray:
    return np.stack(((r - 1.5) ** 2, (r - 2.0) ** 2 + 0.25), axis=-1)


def _interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
    r, theta = coordinates
    values = np.zeros((r.size, 2, 2))
    values[:, 0, 0] = R + r + np.cos(theta)
    values[:, 1, 1] = 2.0 * R - r
    values[:, 0, 1] = values[:, 1, 0] = np.sin(theta) / R
    return values


def test_python_diabatic_pes_builds_atom_diatom_grid() -> None:
    pes = DiabaticPESWrapper(n_state=2, monomer=_monomer, interaction=_interaction)
    r = np.array([1.5, 2.0])
    theta = np.array([0.0, 0.5 * np.pi, np.pi])

    potential = get_diabatic_potential_grid_atom_diatom(pes, 6.0, r, theta)

    assert potential.shape == (2, 3, 2, 2)
    np.testing.assert_allclose(potential[..., 0, 0], 6.0 + r[:, None] + np.cos(theta)[None, :])
    np.testing.assert_allclose(potential[..., 1, 1], 12.0 - r[:, None] + np.zeros_like(theta)[None, :])
    np.testing.assert_allclose(potential[..., 0, 1], np.zeros_like(r)[:, None] + np.sin(theta)[None, :] / 6.0)
    np.testing.assert_allclose(potential[..., 1, 0], potential[..., 0, 1])


def test_python_diabatic_pes_evaluates_radial_batch() -> None:
    pes = DiabaticPESWrapper(n_state=2, monomer=_monomer, interaction=_interaction)
    radial_points = np.array([5.0, 6.0])

    potential = get_diabatic_potential_grid_atom_diatom(pes, radial_points, np.array([1.5]), np.array([0.0, np.pi]))

    expected = np.stack([get_diabatic_potential_grid_atom_diatom(pes, RR, np.array([1.5]), np.array([0.0, np.pi])) for RR in radial_points])
    assert potential.shape == (2, 1, 2, 2, 2)
    np.testing.assert_allclose(potential, expected)


def test_diabatic_pes_uses_specialized_batch_interface() -> None:
    calls: list[np.ndarray] = []

    def interaction_many(R: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        calls.append(R)
        return np.stack([_interaction(float(RR), coordinates) for RR in R])

    pes = DiabaticPESWrapper(n_state=2, monomer=_monomer, interaction=_interaction, interaction_many=interaction_many)

    potential = get_diabatic_potential_grid_atom_diatom(pes, np.array([5.0, 6.0]), np.array([1.5]), np.array([0.0]))

    assert len(calls) == 1
    assert potential.shape == (2, 1, 1, 2, 2)


def test_diabatic_pes_exposes_state_specific_monomer_callables() -> None:
    pes = DiabaticPESWrapper(n_state=2, monomer=_monomer, interaction=_interaction)
    r = np.array([1.5, 2.0])

    values = pes.monomer_values(r)

    np.testing.assert_allclose(values, _monomer(r))
    np.testing.assert_allclose(pes.monomer_state(0)(r), values[:, 0])
    np.testing.assert_allclose(pes.monomer_state(1)(r), values[:, 1])


def test_diabatic_pes_supports_empty_radial_batch() -> None:
    pes = DiabaticPESWrapper(n_state=2, monomer=_monomer, interaction=_interaction)

    potential = get_diabatic_potential_grid_atom_diatom(pes, np.array([]), np.ones(2), np.ones(3))

    assert potential.shape == (0, 2, 3, 2, 2)


@pytest.mark.parametrize(
    ("interaction", "message"),
    [
        (lambda R, coordinates: np.zeros((coordinates.shape[1], 2)), "returned shape"),
        (lambda R, coordinates: np.full((coordinates.shape[1], 2, 2), np.nan), "non-finite"),
        (
            lambda R, coordinates: np.tile(np.array([[0.0, 1.0], [0.0, 0.0]]), (coordinates.shape[1], 1, 1)),
            "symmetric matrix",
        ),
        (lambda R, coordinates: np.zeros((coordinates.shape[1], 2, 2), dtype=np.complex128), "real values"),
    ],
)
def test_diabatic_pes_rejects_invalid_interaction_values(interaction, message: str) -> None:
    pes = DiabaticPESWrapper(n_state=2, monomer=_monomer, interaction=interaction)

    with pytest.raises(ValueError, match=message):
        get_diabatic_potential_grid_atom_diatom(pes, 5.0, np.ones(2), np.ones(3))


def test_diabatic_pes_rejects_invalid_monomer_values() -> None:
    pes = DiabaticPESWrapper(
        n_state=2,
        monomer=lambda r: np.zeros((r.size, 1)),
        interaction=_interaction,
    )

    with pytest.raises(ValueError, match="returned shape"):
        pes.monomer_values(np.ones(2))


def test_diabatic_pes_validates_state_count_and_index() -> None:
    with pytest.raises(ValueError, match="n_state must be positive"):
        DiabaticPESWrapper(n_state=0, monomer=_monomer, interaction=_interaction)

    pes = DiabaticPESWrapper(n_state=2, monomer=_monomer, interaction=_interaction)
    with pytest.raises(ValueError, match="outside"):
        pes.monomer_state(2)
