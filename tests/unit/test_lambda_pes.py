import numpy as np
import pytest

from pyticc.pes import LambdaPES, PESWrapper, as_lambda_pes, get_lambda_grid_atom_diatom


def test_lambda_grid_preserves_component_axis_for_scalar_and_batch() -> None:
    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        r, theta = coordinates
        return np.stack((R + r + theta, R - r + theta), axis=-1)

    pes = LambdaPES(interaction)
    r = np.array([1.0, 2.0])
    theta = np.array([0.0, 0.5, 1.0])

    scalar = get_lambda_grid_atom_diatom(pes, 4.0, r, theta)
    batch = get_lambda_grid_atom_diatom(pes, np.array([4.0, 5.0]), r, theta)

    assert scalar.shape == (2, 3, 2)
    assert batch.shape == (2, 2, 3, 2)
    np.testing.assert_allclose(batch[0], scalar)


def test_scalar_adapter_is_exact_sigma_limit() -> None:
    scalar = PESWrapper(interaction=lambda R, coordinates: R + coordinates[0] * np.cos(coordinates[1]))
    potential = get_lambda_grid_atom_diatom(as_lambda_pes(scalar), 3.0, np.array([1.0, 2.0]), np.array([0.0, np.pi / 2]))

    np.testing.assert_allclose(potential[..., 0], [[4.0, 3.0], [5.0, 3.0]])
    np.testing.assert_array_equal(potential[..., 1], 0.0)


def test_lambda_pes_rejects_wrong_output_shape() -> None:
    pes = LambdaPES(lambda R, coordinates: np.zeros(coordinates.shape[1]))

    with pytest.raises(ValueError, match="expected"):
        get_lambda_grid_atom_diatom(pes, 3.0, np.array([2.0]), np.array([0.0]))
