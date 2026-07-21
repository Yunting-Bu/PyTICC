import numpy as np
import pytest

import pyticc as ticc
from pyticc.basis.channel import ChannelBuilder
from pyticc.scattering.atom_diatom import build_hamiltonian


def _hamiltonian() -> ticc.ScattHamiltonian:
    rovib = ticc.RovibPODVR(grids=np.array([1.5]), E_vj=np.array([[0.02]]), WF_vj=np.ones((1, 1, 1)))
    diatom = ticc.DiatomBasis(rovib=rovib, energy_zero=0.0, vmax=0, jmax=0)
    potential = ticc.PESWrapper(interaction=lambda R, coordinates: np.zeros(coordinates.shape[1]))
    system = ticc.ScattSystem(
        ticc.AtomSpec(),
        diatom,
        Jtot=0,
        system_parity=1,
        potential=potential,
        reduced_mass=2.0,
    )
    basis = ChannelBuilder(system, ticc.TruncSpec()).build()
    return ticc.ScattHamiltonian(
        system=system,
        basis=basis,
        interaction=lambda R: np.array([[0.01]]),
        batched=False,
    )


def test_scatt_hamiltonian_exposes_V_H_and_W() -> None:
    hamiltonian = _hamiltonian()

    np.testing.assert_allclose(hamiltonian.V(4.0), [[0.01]])
    np.testing.assert_allclose(hamiltonian.H(4.0), [[0.03]])
    np.testing.assert_allclose(hamiltonian.W(4.0, 0.025), [[0.02]])


def test_scatt_hamiltonian_rejects_nonpositive_mass() -> None:
    hamiltonian = _hamiltonian()

    with pytest.raises(ValueError, match="Invalid reduced_mass"):
        ticc.ScattSystem(
            hamiltonian.system.monomer_X,
            hamiltonian.system.monomer_Y,
            Jtot=0,
            system_parity=1,
            potential=hamiltonian.system.potential,
            reduced_mass=0.0,
        )


def test_system_build_hamiltonian_solve_flow() -> None:
    rovib = ticc.RovibPODVR(
        grids=np.array([1.5]),
        E_vj=np.zeros((1, 1)),
        WF_vj=np.ones((1, 1, 1)),
    )
    diatom = ticc.DiatomBasis(rovib=rovib, energy_zero=0.0, vmax=0, jmax=0)
    potential = ticc.PESWrapper(interaction=lambda R, coordinates: np.zeros(coordinates.shape[1]))
    system = ticc.ScattSystem(
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
    assert hamiltonian.system is system
    np.testing.assert_allclose(np.abs(result.Smat[0]), 1.0, atol=1.0e-13)
