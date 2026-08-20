import numpy as np
import pytest
from numpy.typing import NDArray

from pyticc.basis.channel import ChannelSpec, build_ChannelBasis
from pyticc.basis.dvr import SineDVR, build_SineDVR
from pyticc.basis.monomer import AtomSpec, build_DiabaticDiatomBasis, prepare_DiabaticDiatom
from pyticc.system import ScattSystem


def _potential(offset: float):
    def potential(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return 0.02 * (r - 2.0) ** 2 + offset

    return potential


def _dvrs() -> tuple[SineDVR, SineDVR]:
    return (
        build_SineDVR(1.0, 4.0, 30, 1000.0, _potential(0.0)),
        build_SineDVR(1.0, 4.0, 30, 1000.0, _potential(0.3)),
    )


def _diabatic_potential(r: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack((_potential(0.0)(r), _potential(0.3)(r)))


def test_prepare_diabatic_diatom_runs_dvr_and_podvr_in_one_call() -> None:
    basis = prepare_DiabaticDiatom(
        _diabatic_potential,
        n_state=2,
        r=(1.0, 4.0),
        n_dvr=30,
        n_podvr=(6, 5),
        vmax=(1, 0),
        jmax=(1, 2),
        mass=1000.0,
    )

    assert basis.n_state == 2
    assert basis.states[0].contracted.grids.shape == (6,)
    assert basis.states[1].contracted.grids.shape == (5,)
    assert basis.states[0].primitive.grids.shape == (30,)
    assert basis.states[1].primitive.grids.shape == (30,)


def test_diabatic_basis_uses_one_energy_zero_and_prepares_complete_state_tables() -> None:
    basis = build_DiabaticDiatomBasis(
        _dvrs(),
        n_podvr=(6, 5),
        vmax=(1, 0),
        jmax=(1, 2),
        mass=1000.0,
    )

    states = list(basis.mis_iter(np.inf))

    assert basis.n_state == 2
    assert basis.states[0].contracted.grids.size == 6
    assert basis.states[1].contracted.grids.size == 5
    assert basis.relative_energies(0)[0, 0] == pytest.approx(0.0, abs=1.0e-14)
    assert basis.relative_energies(1)[0, 0] == pytest.approx(0.3, abs=1.0e-13)
    assert [(state.electronic_state, state.v, state.j) for state in states] == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
    ]


def test_diabatic_basis_retains_orthonormal_primitive_dvr_rovibrational_states() -> None:
    dvrs = _dvrs()
    basis = build_DiabaticDiatomBasis(
        dvrs,
        n_podvr=(6, 5),
        vmax=(1, 0),
        jmax=(1, 2),
        mass=1000.0,
    )

    for state in basis.states:
        assert state.primitive.WF_vj.shape == (state.primitive.grids.size, *state.contracted.E_vj.shape)
        for j in range(state.contracted.E_vj.shape[1]):
            wavefunctions = state.primitive.WF_vj[:, :, j]
            np.testing.assert_allclose(wavefunctions.T @ wavefunctions, np.eye(wavefunctions.shape[1]), atol=1.0e-13)

    for dvr, state in zip(dvrs, basis.states, strict=True):
        np.testing.assert_allclose(state.primitive.WF_vj[:, :, 0], dvr.eigen_vec[:, : state.contracted.E_vj.shape[0]], atol=1.0e-12)


def test_diabatic_basis_accepts_an_explicit_energy_zero() -> None:
    reference = 0.125
    basis = build_DiabaticDiatomBasis(
        _dvrs(),
        n_podvr=5,
        vmax=0,
        jmax=0,
        mass=1000.0,
        energy_zero=reference,
    )

    assert basis.energy_zero == reference
    for electronic_state, state in enumerate(basis.states):
        np.testing.assert_allclose(basis.relative_energies(electronic_state), state.contracted.E_vj - reference)


def test_diabatic_basis_generates_energy_sorted_channels_with_state_labels() -> None:
    basis = build_DiabaticDiatomBasis(
        _dvrs(),
        n_podvr=5,
        vmax=0,
        jmax=(1, 0),
        mass=1000.0,
    )
    system = ScattSystem(AtomSpec(), basis, Jtot=0, system_parity=1)

    channels = build_ChannelBasis(system, ChannelSpec())

    assert np.all(np.diff(channels.E_int) >= 0.0)
    assert {(channel.mis_Y.electronic_state, channel.mis_Y.v, channel.mis_Y.j) for channel in channels} == {
        (0, 0, 0),
        (0, 0, 1),
        (1, 0, 0),
    }
    assert "Y(e=0, v=0, j=0)" in str(channels[0])


def test_diabatic_basis_requires_a_shared_primitive_dvr_grid() -> None:
    incompatible_dvrs = (
        build_SineDVR(1.0, 4.0, 20, 1000.0, _potential(0.0)),
        build_SineDVR(1.1, 4.0, 20, 1000.0, _potential(0.3)),
    )

    with pytest.raises(ValueError, match="same primitive DVR grid"):
        build_DiabaticDiatomBasis(incompatible_dvrs, n_podvr=5, vmax=0, jmax=0, mass=1000.0)


def test_diabatic_basis_validates_per_state_inputs() -> None:
    with pytest.raises(ValueError, match="one value per electronic state"):
        build_DiabaticDiatomBasis(_dvrs(), n_podvr=(5,), vmax=0, jmax=0, mass=1000.0)

    basis = build_DiabaticDiatomBasis(_dvrs(), n_podvr=5, vmax=0, jmax=0, mass=1000.0)
    with pytest.raises(ValueError, match="outside"):
        basis.state(-1)


def test_diabatic_basis_rejects_nonfinite_energy_zero() -> None:
    with pytest.raises(ValueError, match="energy_zero must be finite"):
        build_DiabaticDiatomBasis(_dvrs(), n_podvr=5, vmax=0, jmax=0, mass=1000.0, energy_zero=np.nan)
