import numpy as np

import pyticc as ticc
from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF, build_ChannelBasis
from pyticc.basis.monomer.diatom_electric import DiatomElectricBlock
from pyticc.basis.rovib import RovibBasis
from pyticc.scattering.atom_diatom import build_hamiltonian, build_hamiltonian_electric_sf


def _hamiltonian() -> ticc.ScattHamiltonian:
    rovib = RovibBasis(grids=np.array([1.5]), E_vj=np.array([[0.02]]), WF_vj=np.ones((1, 1, 1)))
    diatom = ticc.DiatomBasis(rovib=rovib, energy_zero=0.0)
    potential = ticc.PESWrapper(interaction=lambda R, coordinates: np.zeros(coordinates.shape[1]))
    system = ticc.ScattSystem(
        ticc.AtomSpec(),
        diatom,
        Jtot=0,
        system_parity=1,
        potential=potential,
        reduced_mass=2.0,
    )
    basis = build_ChannelBasis(system, ticc.ChannelSpec())
    return ticc.ScattHamiltonian(
        basis=basis,
        reduced_mass=2.0,
        interaction=lambda R: np.array([[0.01]]),
    )


def _electric_monomer() -> ticc.DiatomElectricBasis:
    block = DiatomElectricBlock(
        m=0,
        j_values=np.array([0], dtype=np.int64),
        energies=np.array([0.02]),
        coefficients=np.ones((1, 1, 1)),
    )
    return ticc.DiatomElectricBasis(
        grids=np.array([1.5]),
        blocks=(block,),
        energy_zero=0.01,
        electric_strength=1.0e-3,
        jmax=0,
        mass=1000.0,
    )


def _electric_system(pes: ticc.PESWrapper) -> ticc.ScattSystem:
    return ticc.build_ScattSystem(
        ticc.AtomSpec(),
        _electric_monomer(),
        M=0,
        lmax=1,
        potential=pes,
        reduced_mass=2.0,
    )


def test_scatt_hamiltonian_exposes_V_H_and_W() -> None:
    hamiltonian = _hamiltonian()

    np.testing.assert_allclose(hamiltonian.V(4.0), [[0.01]])
    np.testing.assert_allclose(hamiltonian.H(4.0), [[0.03]])
    np.testing.assert_allclose(hamiltonian.W(4.0, 0.025), [[0.02]])


def test_system_build_hamiltonian_solve_flow() -> None:
    rovib = RovibBasis(
        grids=np.array([1.5]),
        E_vj=np.zeros((1, 1)),
        WF_vj=np.ones((1, 1, 1)),
    )
    diatom = ticc.DiatomBasis(rovib=rovib, energy_zero=0.0)
    potential = ticc.PESWrapper(interaction=lambda R, coordinates: np.zeros(coordinates.shape[1]))
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        diatom,
        Jtot=0,
        system_parity=1,
        potential=potential,
        reduced_mass=2.0,
    )

    hamiltonian = build_hamiltonian(system, n_theta=4)
    result = ticc.solve(
        hamiltonian,
        [0.1],
        ticc.Propagation(boundaries=(3.0, 3.2), half_steps=(0.1,)),
    )

    assert isinstance(result, ticc.ScatteringResult)
    assert isinstance(hamiltonian.basis, ChannelBasis)
    assert hamiltonian.basis.Jtot == system.Jtot
    np.testing.assert_allclose(np.abs(result.Smat[0]), 1.0, atol=1.0e-13)


def test_build_hamiltonian_electric_sf_assembles_threshold_centrifugal_and_interaction_terms() -> None:
    pes = ticc.PESWrapper(interaction=lambda R, coordinates: np.full(coordinates.shape[1], 0.05))

    hamiltonian = build_hamiltonian_electric_sf(
        _electric_system(pes),
        n_theta_r=3,
        n_theta_R=3,
        n_delta=4,
    )

    assert isinstance(hamiltonian, ticc.ScattHamiltonian)
    assert isinstance(hamiltonian.basis, ChannelBasisElectricSF)
    assert hamiltonian.basis.M == 0
    assert [(channel.l, channel.m_l, channel.m) for channel in hamiltonian.basis] == [(0, 0, 0), (1, 0, 0)]
    np.testing.assert_allclose(hamiltonian.E_int, [0.01, 0.01])
    np.testing.assert_allclose(hamiltonian.U, np.diag([0.0, 2.0]))
    np.testing.assert_allclose(hamiltonian.V(2.0), 0.05 * np.eye(2), atol=1.0e-13)
    np.testing.assert_allclose(hamiltonian.H(2.0), np.diag([0.06, 0.185]), atol=1.0e-13)
    np.testing.assert_allclose(hamiltonian.W(2.0, 0.04), np.diag([0.08, 0.58]), atol=1.0e-13)


def test_hamiltonian_electric_sf_accepts_radial_batches_and_selected_channel_order() -> None:
    pes = ticc.PESWrapper(interaction=lambda R, coordinates: np.full(coordinates.shape[1], R))
    hamiltonian = build_hamiltonian_electric_sf(
        _electric_system(pes),
        n_theta_r=3,
        n_theta_R=3,
        n_delta=4,
        delta_symmetry=False,
    )

    matrices = hamiltonian.V(np.array([0.25, 0.5]))
    selected = hamiltonian.H(2.0, (1, 0))

    assert matrices.shape == (2, 2, 2)
    np.testing.assert_allclose(matrices[0], 0.25 * np.eye(2), atol=1.0e-13)
    np.testing.assert_allclose(matrices[1], 0.5 * np.eye(2), atol=1.0e-13)
    np.testing.assert_allclose(selected, np.diag([2.135, 2.01]), atol=1.0e-13)


def test_solve_electric_sf_propagates_and_matches_in_the_same_basis() -> None:
    pes = ticc.PESWrapper(interaction=lambda R, coordinates: np.zeros(coordinates.shape[1]))
    hamiltonian = build_hamiltonian_electric_sf(
        _electric_system(pes),
        n_theta_r=3,
        n_theta_R=3,
        n_delta=4,
    )

    result = ticc.solve(
        hamiltonian,
        [0.1],
        ticc.Propagation(boundaries=(3.0, 3.2), half_steps=(0.1,)),
    )

    assert isinstance(result, ticc.ScatteringResult)
    assert isinstance(result.basis, ChannelBasisElectricSF)
    np.testing.assert_allclose(result.asymptotic_transform, np.eye(2))
    np.testing.assert_allclose(result.L, [0.0, 1.0])
    np.testing.assert_allclose(result.Y_asymptotic, result.Y_propagated)
    np.testing.assert_allclose(result.Smat[0].conj().T @ result.Smat[0], np.eye(2), atol=1.0e-13)
