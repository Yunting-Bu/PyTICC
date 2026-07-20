import numpy as np
from scipy.special import roots_legendre

import pyticc as ticc


def _rovib() -> ticc.RovibPODVR:
    return ticc.RovibPODVR(
        grids=np.array([1.5]),
        E_vj=np.zeros((1, 1)),
        WF_vj=np.ones((1, 1, 1)),
    )


def _rotational_rovib(jmax: int) -> ticc.RovibPODVR:
    return ticc.RovibPODVR(
        grids=np.array([1.5]),
        E_vj=np.zeros((1, jmax + 1)),
        WF_vj=np.ones((1, 1, jmax + 1)),
    )


def test_common_setup_tools_are_available_from_top_level() -> None:
    assert ticc.CM2AU * ticc.AU2CM == 1.0
    assert ticc.reduced_mass(2.0, 2.0) == 1.0
    assert callable(ticc.build_SineDVR)


def test_run_atom_diatom_returns_complete_scattering_result() -> None:
    rovib = _rovib()
    diatom = ticc.DiatomSpec(Eint=rovib.E_vj, vmax=0, jmax=0)
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))

    result = ticc.run_atom_diatom(
        diatom,
        rovib,
        pes,
        Jtot=0,
        system_parity=1,
        Etot=[0.1, 0.2],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.2],
        radial_half_steps=[0.1],
        n_theta=4,
    )

    assert isinstance(result, ticc.ScatteringResult)
    assert result.basis.n_channel == 1
    assert result.Y_BF.shape == (2, 1, 1)
    assert len(result.Smat) == 2
    assert result.open_channel_indices[0].tolist() == [0]
    np.testing.assert_allclose(np.abs(result.Smat[0]), 1.0, atol=1.0e-13)


def test_run_atom_diatom_uses_half_angle_rule_for_rotational_exchange_parity() -> None:
    rovib = _rovib()
    diatom = ticc.DiatomSpec(Eint=rovib.E_vj, vmax=0, jmax=0, jpar=1)
    sampled_angles: list[np.ndarray] = []

    def interaction(RR: float, coordinates: np.ndarray) -> np.ndarray:
        sampled_angles.append(np.unique(coordinates[1]))
        return np.zeros(coordinates.shape[1])

    ticc.run_atom_diatom(
        diatom,
        rovib,
        ticc.PESWrapper(interaction=interaction),
        Jtot=0,
        system_parity=1,
        Etot=[0.1],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.2],
        radial_half_steps=[0.1],
        n_theta=4,
    )

    full_cos_theta, _ = roots_legendre(8)
    assert sampled_angles
    np.testing.assert_allclose(sampled_angles[0], np.sort(np.arccos(full_cos_theta[:4])))


def test_run_diatom_diatom_returns_complete_scattering_result() -> None:
    rovib_X = _rovib()
    rovib_Y = _rovib()
    diatom_X = ticc.DiatomSpec(Eint=rovib_X.E_vj, vmax=0, jmax=0)
    diatom_Y = ticc.DiatomSpec(Eint=rovib_Y.E_vj, vmax=0, jmax=0)
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))

    result = ticc.run_diatom_diatom(
        diatom_X,
        rovib_X,
        diatom_Y,
        rovib_Y,
        pes,
        Jtot=0,
        system_parity=1,
        Etot=[0.1],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.2],
        radial_half_steps=[0.1],
        n_theta_X=3,
        n_theta_Y=3,
        n_phi=4,
    )

    assert isinstance(result, ticc.ScatteringResult)
    assert result.basis.n_channel == 1
    assert result.Y_SF.shape == (1, 1, 1)
    assert result.Smat[0].shape == (1, 1)
    np.testing.assert_allclose(np.abs(result.Smat[0]), 1.0, atol=1.0e-13)


def test_run_atom_diatom_cs_returns_independent_K_blocks() -> None:
    rovib = _rotational_rovib(2)
    diatom = ticc.DiatomSpec(Eint=rovib.E_vj, vmax=0, jmax=2)
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))

    result = ticc.run_atom_diatom(
        diatom,
        rovib,
        pes,
        Jtot=2,
        system_parity=1,
        Etot=[0.1],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.2],
        radial_half_steps=[0.1],
        n_theta=6,
        approx=ticc.Approx.CS,
    )

    assert isinstance(result, ticc.CoupledStatesResult)
    assert [block.block.K_values for block in result.blocks] == [(0,), (1,), (2,)]
    for block in result.blocks:
        Smat = block.Smat_asymptotic[0]
        np.testing.assert_allclose(Smat.conj().T @ Smat, np.eye(Smat.shape[0]), atol=1.0e-12)


def test_run_atom_diatom_nncc_reuses_one_batched_pes_grid() -> None:
    rovib = _rotational_rovib(4)
    diatom = ticc.DiatomSpec(Eint=rovib.E_vj, vmax=0, jmax=4)
    calls = 0

    def interaction_many(RR: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((RR.size, coordinates.shape[1]))

    pes = ticc.PESWrapper(
        interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]),
        interaction_many=interaction_many,
    )
    result = ticc.run_atom_diatom(
        diatom,
        rovib,
        pes,
        Jtot=4,
        system_parity=1,
        Etot=[0.1],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.2],
        radial_half_steps=[0.1],
        n_theta=8,
        approx=ticc.Approx.NNCC,
        K_delta=1,
    )

    assert isinstance(result, ticc.CoupledStatesResult)
    assert [block.block.K_values for block in result.blocks] == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert calls == 1
    owned_indices = sorted(index for block in result.blocks for index in block.block.owned_channel_indices)
    assert owned_indices == list(range(result.basis.n_channel))


def test_run_atom_diatom_nncc_shares_each_radial_window_across_blocks() -> None:
    rovib = _rotational_rovib(4)
    diatom = ticc.DiatomSpec(Eint=rovib.E_vj, vmax=0, jmax=4)
    evaluated_R: list[np.ndarray] = []

    def interaction_many(RR: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        evaluated_R.append(RR.copy())
        return np.zeros((RR.size, coordinates.shape[1]))

    pes = ticc.PESWrapper(
        interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]),
        interaction_many=interaction_many,
    )
    result = ticc.run_atom_diatom(
        diatom,
        rovib,
        pes,
        Jtot=4,
        system_parity=1,
        Etot=[0.1],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.4],
        radial_half_steps=[0.1],
        n_theta=8,
        approx=ticc.Approx.NNCC,
        K_delta=1,
        memory_limit_mb=1.0e-6,
    )

    assert isinstance(result, ticc.CoupledStatesResult)
    assert len(result.blocks) == 3
    assert len(evaluated_R) == 2
    np.testing.assert_allclose(evaluated_R[0], [3.0, 3.1, 3.2])
    np.testing.assert_allclose(evaluated_R[1], [3.3, 3.4])
