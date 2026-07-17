import numpy as np
import pytest

from pyticc.basis.monomer import DiatomSpec, arrange_diatom_levels


def test_arrange_diatom_levels_maps_labels_to_vj_indices() -> None:
    levels = [
        (1, 2, 1.2),
        (0, 0, 0.0),
        (0, 2, 0.2),
    ]

    Eint = arrange_diatom_levels(levels, vmax=1, jmax=2)

    assert Eint.shape == (2, 3)
    assert Eint[0, 0] == pytest.approx(0.0)
    assert Eint[0, 2] == pytest.approx(0.2)
    assert Eint[1, 2] == pytest.approx(1.2)
    assert np.isinf(Eint[0, 1])
    assert np.isinf(Eint[1, 0])


def test_missing_diatom_levels_do_not_generate_inner_states() -> None:
    Eint = arrange_diatom_levels([(0, 0, 0.0), (0, 2, 0.2)], vmax=1, jmax=2)
    diatom = DiatomSpec(Eint=Eint, vmax=1, jmax=2, jpar=1)

    states = list(diatom.mis_iter(np.inf))

    assert [(state.v, state.j, state.Eint) for state in states] == [(0, 0, 0.0), (0, 2, 0.2)]


def test_arrange_diatom_levels_rejects_duplicate_labels() -> None:
    levels = [(0, 0, 0.0), (0, 0, 0.1)]

    with pytest.raises(ValueError, match="Duplicate diatomic level"):
        arrange_diatom_levels(levels, vmax=0, jmax=0)


def test_arrange_diatom_levels_rejects_labels_outside_range() -> None:
    with pytest.raises(ValueError, match="outside the requested range"):
        arrange_diatom_levels([(1, 0, 0.1)], vmax=0, jmax=0)


@pytest.mark.parametrize(
    ("Eint", "vmax", "jmax", "message"),
    [
        (np.zeros(3), 0, 2, "two-dimensional"),
        (np.zeros((1, 2)), 0, 2, "does not cover"),
    ],
)
def test_diatom_spec_validates_energy_array(Eint: np.ndarray, vmax: int, jmax: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DiatomSpec(Eint=Eint, vmax=vmax, jmax=jmax)
