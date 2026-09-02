from dataclasses import replace

import numpy as np
from loguru import logger
from scipy.special import roots_legendre

import pyticc as ticc
from pyticc.basis.rovib import RovibBasis
from pyticc.scattering import atom_diatom
from pyticc.scattering.solver import build_k_blocks


def _rovib() -> RovibBasis:
    return RovibBasis(
        grids=np.array([1.5]),
        E_vj=np.zeros((1, 1)),
        WF_vj=np.ones((1, 1, 1)),
    )


def _rotational_rovib(jmax: int) -> RovibBasis:
    return RovibBasis(
        grids=np.array([1.5]),
        E_vj=np.zeros((1, jmax + 1)),
        WF_vj=np.ones((1, 1, jmax + 1)),
    )


def _diatom(rovib: RovibBasis) -> ticc.DiatomBasis:
    return ticc.DiatomBasis(rovib=rovib, energy_zero=0.0)


def _solve_atom(
    diatom: ticc.DiatomBasis,
    pes: ticc.PESWrapper,
    *,
    Jtot: int = 0,
    energies: tuple[float, ...] = (0.1,),
    n_theta: int = 4,
    approx: ticc.Approx = ticc.Approx.EXACT,
    K_delta: int = 1,
    boundaries: tuple[float, ...] = (3.0, 3.2),
    half_steps: tuple[float, ...] = (0.1,),
    propagation: ticc.Propagation | None = None,
    channel: ticc.ChannelSpec | None = None,
) -> ticc.ScatteringResult | ticc.CoupledStatesResult:
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        diatom,
        scattering_type="A+BC",
        Jtot=Jtot,
        system_parity=1,
        approx=approx,
        K_delta=K_delta,
        channel=channel,
        potential=pes,
        reduced_mass=2.0,
    )
    potential_grid = ticc.prepare_potential(system, boundaries, half_steps, n_theta=n_theta)
    runtime = ticc.Propagation() if propagation is None else propagation
    return ticc.solve(system, energies, potential_grid, runtime)


def test_common_setup_tools_are_available_from_top_level() -> None:
    assert ticc.CM2AU * ticc.AU2CM == 1.0
    np.testing.assert_allclose(
        [
            ticc.HZ2AU * ticc.AU2HZ,
            ticc.KHZ2AU * ticc.AU2KHZ,
            ticc.MHZ2AU * ticc.AU2MHZ,
            ticc.GHZ2AU * ticc.AU2GHZ,
        ],
        1.0,
    )
    assert ticc.reduced_mass(2.0, 2.0) == 1.0
    assert callable(ticc.build_SineDVR)
    assert callable(ticc.prepare_Diatom)
    assert callable(ticc.prepare_DiabaticDiatom)
    assert callable(ticc.prepare_DiatomElectric)
    assert callable(ticc.prepare_Triatom)
    assert "solve" in ticc.__all__
    assert "ScatteringType" in ticc.__all__
    assert "build_k_blocks" not in ticc.__all__
    assert "report" in ticc.__all__
    assert "get_Wmat" not in ticc.__all__
    for legacy_name in ("run_atom_diatom", "run_diatom_diatom", "run_atom_triatom", "run_diabatic_atom_diatom"):
        assert not hasattr(ticc, legacy_name)


def test_build_scatt_system_requires_a_supported_explicit_type() -> None:
    with np.testing.assert_raises_regex(ValueError, "Unsupported scattering_type"):
        ticc.build_ScattSystem(
            ticc.AtomSpec(),
            _diatom(_rovib()),
            scattering_type="atom-diatom",
            Jtot=0,
            system_parity=1,
        )


def test_solve_atom_diatom_returns_complete_scattering_result() -> None:
    rovib = _rovib()
    diatom = _diatom(rovib)
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))

    result = _solve_atom(diatom, pes, energies=(0.1, 0.2))

    assert isinstance(result, ticc.ScatteringResult)
    assert result.basis.n_channel == 1
    assert result.Y_propagated.shape == (2, 1, 1)
    assert len(result.Smat) == 2
    assert result.open_channel_indices[0].tolist() == [0]
    assert result.timing is not None
    assert result.timing.wall_seconds >= 0.0
    assert result.timing.cpu_seconds >= 0.0
    np.testing.assert_allclose(np.abs(result.Smat[0]), 1.0, atol=1.0e-13)


def test_prepare_potential_logs_start_and_completion() -> None:
    diatom = _diatom(_rovib())
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        diatom,
        scattering_type="A+BC",
        Jtot=0,
        system_parity=1,
        potential=pes,
        reduced_mass=2.0,
    )
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        ticc.prepare_potential(system, (3.0, 3.2), (0.1,), n_theta=4, processes=1)
    finally:
        logger.remove(sink)

    assert any("Potential preparation started" in message and "radial_points=3" in message for message in messages)
    assert any("Potential preparation complete" in message and "wall=" in message for message in messages)


def test_solve_atom_diatom_uses_half_angle_rule_for_rotational_exchange_parity() -> None:
    rovib = _rovib()
    diatom = _diatom(rovib)
    sampled_angles: list[np.ndarray] = []

    def interaction(RR: float, coordinates: np.ndarray) -> np.ndarray:
        sampled_angles.append(np.unique(coordinates[1]))
        return np.zeros(coordinates.shape[1])

    _solve_atom(
        diatom,
        ticc.PESWrapper(interaction=interaction),
        channel=ticc.ChannelSpec(exchange_parity_Y=1),
    )

    full_cos_theta, _ = roots_legendre(8)
    assert sampled_angles
    np.testing.assert_allclose(sampled_angles[0], np.sort(np.arccos(full_cos_theta[:4])))


def test_solve_diatom_diatom_returns_complete_scattering_result() -> None:
    rovib_X = _rovib()
    rovib_Y = _rovib()
    diatom_X = _diatom(rovib_X)
    diatom_Y = _diatom(rovib_Y)
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))

    system = ticc.build_ScattSystem(
        diatom_X,
        diatom_Y,
        scattering_type="AB+CD",
        Jtot=0,
        system_parity=1,
        potential=pes,
        reduced_mass=2.0,
    )
    potential_grid = ticc.prepare_potential(
        system,
        (3.0, 3.2),
        (0.1,),
        n_theta_X=3,
        n_theta_Y=3,
        n_phi=4,
    )
    result = ticc.solve(system, [0.1], potential_grid, ticc.Propagation())

    assert isinstance(result, ticc.ScatteringResult)
    assert result.basis.n_channel == 1
    assert result.Y_asymptotic.shape == (1, 1, 1)
    assert result.Smat[0].shape == (1, 1)
    np.testing.assert_allclose(np.abs(result.Smat[0]), 1.0, atol=1.0e-13)


def test_solve_rejects_potential_grid_from_another_scattering_type() -> None:
    diatom = _diatom(_rovib())
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        diatom,
        scattering_type="A+BC",
        Jtot=0,
        system_parity=1,
        potential=pes,
        reduced_mass=2.0,
    )
    potential_grid = ticc.prepare_potential(system, (3.0, 3.2), (0.1,), n_theta=4)
    wrong_grid = replace(potential_grid, scattering_type=ticc.ScatteringType.DIATOM_DIATOM)

    with np.testing.assert_raises_regex(TypeError, "cannot use"):
        ticc.solve(system, [0.1], wrong_grid, ticc.Propagation())


def test_solve_atom_diatom_cs_returns_independent_K_blocks() -> None:
    rovib = _rotational_rovib(2)
    diatom = _diatom(rovib)
    pes = ticc.PESWrapper(interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]))

    result = _solve_atom(diatom, pes, Jtot=2, n_theta=6, approx=ticc.Approx.CS)

    assert isinstance(result, ticc.CoupledStatesResult)
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        diatom,
        scattering_type="A+BC",
        Jtot=2,
        system_parity=1,
        approx=ticc.Approx.CS,
        potential=pes,
        reduced_mass=2.0,
    )
    hamiltonian = atom_diatom.build_hamiltonian(system, n_theta=6)
    assert [block.K_values for block in build_k_blocks(hamiltonian)] == [(0,), (1,), (2,)]
    assert [block.block.K_values for block in result.blocks] == [(0,), (1,), (2,)]
    for block in result.blocks:
        Smat = block.Smat_asymptotic[0]
        np.testing.assert_allclose(Smat.conj().T @ Smat, np.eye(Smat.shape[0]), atol=1.0e-12)
        Smat_BF = block.Smat_BF[0]
        np.testing.assert_allclose(Smat_BF.conj().T @ Smat_BF, np.eye(Smat_BF.shape[0]), atol=1.0e-12)


def test_solve_atom_diatom_nncc_reuses_one_batched_pes_grid() -> None:
    rovib = _rotational_rovib(4)
    diatom = _diatom(rovib)
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
    diatom = _diatom(rovib)
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
        boundaries=(3.0, 3.4),
        propagation=ticc.Propagation(memory_mb=1.0e-6),
    )

    assert isinstance(result, ticc.CoupledStatesResult)
    assert len(result.blocks) == 3
    assert len(evaluated_R) == 1
    np.testing.assert_allclose(evaluated_R[0], [3.0, 3.1, 3.2, 3.3, 3.4])
