import numpy as np

import pyticc as ticc
from pyticc.scattering import diatom_diatom


def _rovib(grids: list[float], energies: list[float]) -> ticc.RovibPODVR:
    jmax = len(energies) - 1
    wavefunctions = np.empty((len(grids), 1, jmax + 1), dtype=np.float64)
    for j in range(jmax + 1):
        angle = 0.15 * (j + 1)
        wavefunctions[:, 0, j] = np.array([np.cos(angle), np.sin(angle)])
    return ticc.RovibPODVR(
        grids=np.asarray(grids, dtype=np.float64),
        E_vj=np.asarray([energies], dtype=np.float64),
        WF_vj=wavefunctions,
    )


def _model() -> tuple[ticc.DiatomBasis, ticc.DiatomBasis, ticc.PESWrapper]:
    rovib_X = _rovib([1.2, 1.6], [0.0, 0.01, 0.02])
    rovib_Y = _rovib([1.4, 1.8], [0.0, 0.015])
    diatom_X = ticc.DiatomBasis(rovib=rovib_X, energy_zero=0.0, vmax=0, jmax=2, jpar=1)
    diatom_Y = ticc.DiatomBasis(rovib=rovib_Y, energy_zero=0.0, vmax=0, jmax=1)

    def interaction(RR: float, coordinates: np.ndarray) -> np.ndarray:
        r_X, r_Y, theta_X, theta_Y, phi = coordinates
        radial = 1.0e-3 * max(4.5 - RR, 0.0) ** 2
        angular = 1.0 + 0.20 * np.cos(theta_X) * np.cos(theta_Y) + 0.15 * np.sin(theta_X) * np.sin(theta_Y) * np.cos(phi)
        stretch = 1.0 + 0.10 * (r_X - 1.4) - 0.08 * (r_Y - 1.6)
        return radial * angular * stretch

    return diatom_X, diatom_Y, ticc.PESWrapper(interaction=interaction)


def _run(
    Jtot: int,
    approx: ticc.Approx,
    K_cut: int | None = None,
    K_delta: int = 1,
) -> ticc.ScatteringResult | ticc.CoupledStatesResult:
    diatom_X, diatom_Y, pes = _model()
    system = ticc.ScattSystem(
        diatom_X,
        diatom_Y,
        Jtot=Jtot,
        system_parity=1,
        approx=approx,
        K_delta=K_delta,
        potential=pes,
        reduced_mass=2.0,
    )
    hamiltonian = diatom_diatom.build_hamiltonian(
        system,
        trunc=ticc.TruncSpec(K_cut=K_cut),
        n_theta_X=4,
        n_theta_Y=4,
        n_phi=5,
    )
    return ticc.solve(hamiltonian, [0.08], ticc.Propagation((3.0, 4.0, 5.0), (0.25, 0.25)))


def _assert_exact_matches_single_block(exact: ticc.ScatteringResult, approximate: ticc.CoupledStatesResult) -> None:
    assert exact.basis.channels == approximate.basis.channels
    assert len(approximate.blocks) == 1
    block = approximate.blocks[0]
    assert block.block.channel_indices == tuple(range(exact.basis.n_channel))
    np.testing.assert_allclose(block.Y_BF, exact.Y_BF, atol=1.0e-13)
    np.testing.assert_allclose(block.Bmat, exact.Bmat, atol=1.0e-13)
    np.testing.assert_allclose(block.L, exact.L, atol=1.0e-13)
    np.testing.assert_allclose(block.Y_asymptotic, exact.Y_SF, atol=1.0e-13)
    for Smat_block, Smat_exact in zip(block.Smat_asymptotic, exact.Smat, strict=True):
        np.testing.assert_allclose(Smat_block, Smat_exact, atol=1.0e-13)


def test_J_zero_exact_cs_and_nncc_are_identical() -> None:
    exact = _run(Jtot=0, approx=ticc.Approx.EXACT)
    cs = _run(Jtot=0, approx=ticc.Approx.CS)
    nncc = _run(Jtot=0, approx=ticc.Approx.NNCC)

    assert isinstance(exact, ticc.ScatteringResult)
    assert isinstance(cs, ticc.CoupledStatesResult)
    assert isinstance(nncc, ticc.CoupledStatesResult)
    _assert_exact_matches_single_block(exact, cs)
    _assert_exact_matches_single_block(exact, nncc)


def test_K_cut_zero_exact_and_cs_are_identical() -> None:
    exact = _run(Jtot=2, approx=ticc.Approx.EXACT, K_cut=0)
    cs = _run(Jtot=2, approx=ticc.Approx.CS, K_cut=0)

    assert isinstance(exact, ticc.ScatteringResult)
    assert isinstance(cs, ticc.CoupledStatesResult)
    _assert_exact_matches_single_block(exact, cs)


def test_nncc_covering_every_K_is_identical_to_exact() -> None:
    exact = _run(Jtot=2, approx=ticc.Approx.EXACT)
    nncc = _run(Jtot=2, approx=ticc.Approx.NNCC, K_delta=1)

    assert isinstance(exact, ticc.ScatteringResult)
    assert isinstance(nncc, ticc.CoupledStatesResult)
    assert {channel.K for channel in exact.basis} == {0, 1, 2}
    _assert_exact_matches_single_block(exact, nncc)
