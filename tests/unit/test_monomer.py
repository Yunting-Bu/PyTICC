import numpy as np
import pytest

from pyticc.basis.monomer import DiatomBasis, DiatomSpec
from pyticc.basis.podvr import RovibPODVR


def test_missing_diatom_levels_do_not_generate_inner_states() -> None:
    Eint = np.array([[0.0, np.inf, 0.2], [np.inf, np.inf, np.inf]])
    diatom = DiatomSpec(Eint=Eint, vmax=1, jmax=2, jpar=1)

    states = list(diatom.mis_iter(np.inf))

    assert [(state.v, state.j, state.Eint) for state in states] == [(0, 0, 0.0), (0, 2, 0.2)]


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


def test_diatom_basis_uses_explicit_energy_zero() -> None:
    rovib = RovibPODVR(
        grids=np.array([1.4, 1.6]),
        E_vj=np.array([[0.25, 0.27], [0.35, 0.37]]),
        WF_vj=np.ones((2, 2, 2)),
    )
    diatom = DiatomBasis(rovib=rovib, energy_zero=0.25, vmax=1, jmax=1)

    np.testing.assert_allclose(diatom.Eint, [[0.0, 0.02], [0.1, 0.12]])
    assert [(state.v, state.j) for state in diatom.mis_iter(np.inf)] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_diatom_basis_validates_rovib_shapes() -> None:
    rovib = RovibPODVR(
        grids=np.array([1.5]),
        E_vj=np.zeros((1, 1)),
        WF_vj=np.ones((2, 1, 1)),
    )

    with pytest.raises(ValueError, match="RovibPODVR shapes"):
        DiatomBasis(rovib=rovib, energy_zero=0.0, vmax=0, jmax=0)
