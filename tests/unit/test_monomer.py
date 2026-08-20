import numpy as np
import pytest

from pyticc.basis.monomer import DiatomBasis, DiatomSpec, prepare_Diatom
from pyticc.basis.rovib import RovibBasis


def test_missing_diatom_levels_do_not_generate_inner_states() -> None:
    Eint = np.array([[0.0, np.inf, 0.2], [np.inf, np.inf, np.inf]])
    diatom = DiatomSpec(Eint=Eint)

    states = list(diatom.mis_iter(np.inf))

    assert [(state.v, state.j, state.Eint) for state in states] == [(0, 0, 0.0), (0, 2, 0.2)]


@pytest.mark.parametrize(
    ("Eint", "message"),
    [
        (np.zeros(3), "two-dimensional"),
        (np.zeros((0, 2)), "at least one"),
    ],
)
def test_diatom_spec_validates_energy_array(Eint: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DiatomSpec(Eint=Eint)


def test_diatom_basis_uses_explicit_energy_zero() -> None:
    rovib = RovibBasis(
        grids=np.array([1.4, 1.6]),
        E_vj=np.array([[0.25, 0.27], [0.35, 0.37]]),
        WF_vj=np.ones((2, 2, 2)),
    )
    diatom = DiatomBasis(rovib=rovib, energy_zero=0.25)

    np.testing.assert_allclose(diatom.Eint, [[0.0, 0.02], [0.1, 0.12]])
    assert [(state.v, state.j) for state in diatom.mis_iter(np.inf)] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_diatom_basis_validates_rovib_shapes() -> None:
    with pytest.raises(ValueError, match="RovibBasis shapes"):
        RovibBasis(
            grids=np.array([1.5]),
            E_vj=np.zeros((1, 1)),
            WF_vj=np.ones((2, 1, 1)),
        )


def test_prepare_diatom_runs_dvr_and_podvr_in_one_call() -> None:
    monomer = prepare_Diatom(
        lambda r: 0.02 * (r - 2.0) ** 2,
        r=(1.0, 4.0),
        n_dvr=30,
        n_podvr=6,
        vmax=1,
        jmax=2,
        mass=1000.0,
    )

    assert monomer.rovib.grids.shape == (6,)
    assert monomer.rovib.E_vj.shape == (2, 3)
    assert monomer.rovib.WF_vj.shape == (6, 2, 3)


def test_prepare_diatom_reports_missing_monomer_potential() -> None:
    with pytest.raises(ValueError, match="requires a monomer potential"):
        prepare_Diatom(
            None,
            r=(1.0, 4.0),
            n_dvr=30,
            n_podvr=6,
            vmax=1,
            jmax=2,
            mass=1000.0,
        )
