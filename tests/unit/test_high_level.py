import numpy as np
from scipy.special import roots_legendre

import pyticc as ticc
from pyticc.scattering import atom_diatom, diatom_diatom


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


def _diatom(rovib: ticc.RovibPODVR, *, jmax: int = 0, jpar: int = 0) -> ticc.DiatomBasis:
    return ticc.DiatomBasis(rovib=rovib, energy_zero=0.0, vmax=0, jmax=jmax, jpar=jpar)


def _solve_atom(
    diatom: ticc.DiatomBasis,
    pes: ticc.PESWrapper,
    *,
    Jtot: int = 0,
    energies: tuple[float, ...] = (0.1,),
    n_theta: int = 4,
    approx: ticc.Approx = ticc.Approx.EXACT,
    K_delta: int = 1,
    propagation: ticc.Propagation | None = None,
) -> ticc.ScatteringResult | ticc.CoupledStatesResult:
    system = ticc.ScattSystem(
        ticc.AtomSpec(),
        diatom,
        Jtot=Jtot,
        system_parity=1,
        approx=approx,
        K_delta=K_delta,
        potential=pes,
        reduced_mass=2.0,
    )
    hamiltonian = atom_diatom.build_hamiltonian(system, n_theta=n_theta)
    radial = ticc.Propagation((3.0, 3.2), (0.1,)) if propagation is None else propagation
    return ticc.solve(hamiltonian, energies, radial)


def test_common_setup_tools_are_available_from_top_level() -> None:
    assert ticc.CM2AU * ticc.AU2CM == 1.0
    assert ticc.reduced_mass(2.0, 2.0) == 1.0
    assert callable(ticc.build_SineDVR)
    assert "solve" in ticc.__all__
    assert "build_k_blocks" in ticc.__all__
    assert "report" in ticc.__all__
    assert "get_Wmat" not in ticc.__all__
    for legacy_name in ("run_atom_diatom", "run_diatom_diatom", "run_atom_triatom", "run_diabatic_atom_diatom"):
        assert not hasattr(ticc, legacy_name)


def test_solve_atom_diatom_returns_complete_scattering_result() -> None:
    rovib = _rovib()
    diatom = _diatom(rovib)
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))

    result = _solve_atom(diatom, pes, energies=(0.1, 0.2))

    assert isinstance(result, ticc.ScatteringResult)
    assert result.basis.n_channel == 1
    assert result.Y_BF.shape == (2, 1, 1)
    assert len(result.Smat) == 2
    assert result.open_channel_indices[0].tolist() == [0]
    assert result.timing is not None
    assert result.timing.wall_seconds >= 0.0
    assert result.timing.cpu_seconds >= 0.0
    np.testing.assert_allclose(np.abs(result.Smat[0]), 1.0, atol=1.0e-13)


def test_solve_atom_diatom_uses_half_angle_rule_for_rotational_exchange_parity() -> None:
    rovib = _rovib()
    diatom = _diatom(rovib, jpar=1)
    sampled_angles: list[np.ndarray] = []

    def interaction(RR: float, coordinates: np.ndarray) -> np.ndarray:
        sampled_angles.append(np.unique(coordinates[1]))
        return np.zeros(coordinates.shape[1])

    _solve_atom(diatom, ticc.PESWrapper(interaction=interaction))

    full_cos_theta, _ = roots_legendre(8)
    assert sampled_angles
    np.testing.assert_allclose(sampled_angles[0], np.sort(np.arccos(full_cos_theta[:4])))


def test_solve_diatom_diatom_returns_complete_scattering_result() -> None:
    rovib_X = _rovib()
    rovib_Y = _rovib()
    diatom_X = _diatom(rovib_X)
    diatom_Y = _diatom(rovib_Y)
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))

    system = ticc.ScattSystem(
        diatom_X,
        diatom_Y,
        Jtot=0,
        system_parity=1,
        potential=pes,
        reduced_mass=2.0,
    )
    hamiltonian = diatom_diatom.build_hamiltonian(
        system,
        n_theta_X=3,
        n_theta_Y=3,
        n_phi=4,
    )
    result = ticc.solve(hamiltonian, [0.1], ticc.Propagation((3.0, 3.2), (0.1,)))

    assert isinstance(result, ticc.ScatteringResult)
    assert result.basis.n_channel == 1
    assert result.Y_SF.shape == (1, 1, 1)
    assert result.Smat[0].shape == (1, 1)
    np.testing.assert_allclose(np.abs(result.Smat[0]), 1.0, atol=1.0e-13)


def test_solve_atom_diatom_cs_returns_independent_K_blocks() -> None:
    rovib = _rotational_rovib(2)
    diatom = _diatom(rovib, jmax=2)
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))

    result = _solve_atom(diatom, pes, Jtot=2, n_theta=6, approx=ticc.Approx.CS)

    assert isinstance(result, ticc.CoupledStatesResult)
    system = ticc.ScattSystem(
        ticc.AtomSpec(),
        diatom,
        Jtot=2,
        system_parity=1,
        approx=ticc.Approx.CS,
        potential=pes,
        reduced_mass=2.0,
    )
    hamiltonian = atom_diatom.build_hamiltonian(system, n_theta=6)
    assert [block.K_values for block in ticc.build_k_blocks(hamiltonian)] == [(0,), (1,), (2,)]
    assert [block.block.K_values for block in result.blocks] == [(0,), (1,), (2,)]
    for block in result.blocks:
        Smat = block.Smat_asymptotic[0]
        np.testing.assert_allclose(Smat.conj().T @ Smat, np.eye(Smat.shape[0]), atol=1.0e-12)


def test_solve_atom_diatom_nncc_reuses_one_batched_pes_grid() -> None:
    rovib = _rotational_rovib(4)
    diatom = _diatom(rovib, jmax=4)
    calls = 0

    def interaction_many(RR: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((RR.size, coordinates.shape[1]))

    pes = ticc.PESWrapper(
        interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]),
        interaction_many=interaction_many,
    )
    result = _solve_atom(diatom, pes, Jtot=4, n_theta=8, approx=ticc.Approx.NNCC)

    assert isinstance(result, ticc.CoupledStatesResult)
    assert [block.block.K_values for block in result.blocks] == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert calls == 1
    owned_indices = sorted(index for block in result.blocks for index in block.block.owned_channel_indices)
    assert owned_indices == list(range(result.basis.n_channel))


def test_solve_atom_diatom_nncc_shares_each_radial_window_across_blocks() -> None:
    rovib = _rotational_rovib(4)
    diatom = _diatom(rovib, jmax=4)
    evaluated_R: list[np.ndarray] = []

    def interaction_many(RR: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        evaluated_R.append(RR.copy())
        return np.zeros((RR.size, coordinates.shape[1]))

    pes = ticc.PESWrapper(
        interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]),
        interaction_many=interaction_many,
    )
    result = _solve_atom(
        diatom,
        pes,
        Jtot=4,
        n_theta=8,
        approx=ticc.Approx.NNCC,
        propagation=ticc.Propagation((3.0, 3.4), (0.1,), memory_mb=1.0e-6),
    )

    assert isinstance(result, ticc.CoupledStatesResult)
    assert len(result.blocks) == 3
    assert len(evaluated_R) == 2
    np.testing.assert_allclose(evaluated_R[0], [3.0, 3.1, 3.2])
    np.testing.assert_allclose(evaluated_R[1], [3.3, 3.4])
