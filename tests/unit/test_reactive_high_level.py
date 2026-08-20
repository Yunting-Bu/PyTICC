import numpy as np

import pyticc as ticc
from pyticc.scattering import reactive_atom_diatom


def toy_total_pes() -> ticc.TotalPES:
    """Return a symmetric three-body PES with bound pair asymptotes."""

    def potential(bonds: np.ndarray) -> np.ndarray:
        displacement = (bonds - 2.0) / 0.5
        pair_values = np.exp(-2.0 * displacement) - 2.0 * np.exp(-displacement)
        return np.sum(pair_values, axis=0)

    return ticc.TotalPES(potential)


def test_reactive_system_hamiltonian_and_solve_follow_common_flow() -> None:
    pes = toy_total_pes()
    monomer = ticc.prepare_Delves(
        pes,
        (10.0, 10.0, 10.0),
    )
    system = ticc.build_ScattSystem(
        monomer,
        Jtot=0,
        system_parity=1,
        jmax=0,
        channel=ticc.ChannelSpec(exchange_parity_Y=1, E_Y_cut=-0.2, K_cut=0),
        total_potential=pes,
    )
    assert isinstance(system.basis, ticc.DelvesBasis)
    basis = system.basis
    hamiltonian = reactive_atom_diatom.build_hamiltonian(system)
    propagation = ticc.Propagation(
        (hamiltonian.basis.rho_min, 8.0),
        (1.0,),
        device="cpu",
    )

    result = ticc.solve(hamiltonian, [-0.1], propagation)

    assert isinstance(monomer, ticc.DelvesMonomer)
    assert not hasattr(monomer, "jmax")
    assert not hasattr(monomer, "E_max")
    assert not hasattr(monomer, "rho_min")
    assert isinstance(system, ticc.ScattSystem)
    assert not hasattr(ticc, "ReactiveScattSystem")
    assert not hasattr(reactive_atom_diatom, "build_channels")
    assert basis.qns == ((1, 0, 0, 0), (2, 0, 0, 0))
    assert basis.n_channel == 2
    assert isinstance(hamiltonian, ticc.DelvesHamiltonian)
    assert isinstance(result, ticc.ReactiveScatteringResult)
    assert result.basis.qns == ((1, 0, 0, 0), (2, 0, 0, 0))
    assert result.Y_propagated.shape[0] == 1
    assert result.Y_asymptotic.shape == (1, 2, 2)
    assert result.Smat[0].shape == (2, 2)
    assert result.radial_points[0] == hamiltonian.basis.rho_min
    assert result.rho_final == propagation.Rmatch
    assert result.timing is not None
    np.testing.assert_allclose(result.Smat[0].conj().T @ result.Smat[0], np.eye(2), atol=2.0e-13)


def test_reactive_preparation_validates_energy_zero_and_symmetry() -> None:
    pes = toy_total_pes()
    monomer = ticc.prepare_Delves(
        pes,
        (10.0, 10.0, 10.0),
    )

    try:
        ticc.build_ScattSystem(
            monomer,
            Jtot=0,
            system_parity=1,
            jmax=0,
            channel=ticc.ChannelSpec(exchange_parity_Y=2, E_Y_cut=-0.2, K_cut=0),
            total_potential=pes,
        )
    except ValueError as error:
        assert "exchange_parity_Y" in str(error)
    else:
        raise AssertionError("invalid exchange parity was accepted")

    try:
        ticc.prepare_Delves(
            pes,
            (10.0, 10.0, 10.0),
            energy_zero="lowest",
        )
    except ValueError as error:
        assert "energy_zero" in str(error)
    else:
        raise AssertionError("invalid reactive energy zero was accepted")


def test_minimum_energy_zero_is_equivalent_to_converted_native_energies() -> None:
    pes = toy_total_pes()
    minimum_monomer = ticc.prepare_Delves(
        pes,
        (10.0, 10.0, 10.0),
        energy_zero="minimum",
        scaled_r_step=0.1,
        scaled_r_scan_max=6.0,
    )
    minimum_system = ticc.build_ScattSystem(
        minimum_monomer,
        Jtot=0,
        system_parity=1,
        jmax=0,
        channel=ticc.ChannelSpec(exchange_parity_Y=1, E_Y_cut=0.8, K_cut=0),
        total_potential=pes,
    )
    minimum_hamiltonian = reactive_atom_diatom.build_hamiltonian(minimum_system)

    native_monomer = ticc.prepare_Delves(
        pes,
        minimum_monomer.mass,
        energy_zero="native",
    )
    native_system = ticc.build_ScattSystem(
        native_monomer,
        Jtot=0,
        system_parity=1,
        jmax=0,
        channel=ticc.ChannelSpec(exchange_parity_Y=1, E_Y_cut=0.8 + minimum_monomer.energy_zero, K_cut=0),
        total_potential=pes,
    )
    native_hamiltonian = reactive_atom_diatom.build_hamiltonian(native_system)

    assert minimum_hamiltonian.energy_zero < 0.0
    assert native_hamiltonian.energy_zero == 0.0
    assert minimum_hamiltonian.basis.rho_min == native_hamiltonian.basis.rho_min
    assert minimum_hamiltonian.basis.scaled_r_max == native_hamiltonian.basis.scaled_r_max
    assert minimum_hamiltonian.basis.n_sine == native_hamiltonian.basis.n_sine

    selected_energy = -0.1 - minimum_hamiltonian.energy_zero
    propagation = ticc.Propagation((minimum_hamiltonian.basis.rho_min, 8.0), (1.0,), device="cpu")
    minimum_result = ticc.solve(minimum_hamiltonian, [selected_energy], propagation)
    native_result = ticc.solve(native_hamiltonian, [-0.1], propagation)

    assert minimum_result.energy_zero == minimum_hamiltonian.energy_zero
    np.testing.assert_allclose(
        minimum_result.basis.energies + minimum_result.energy_zero,
        native_result.basis.energies,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(minimum_result.Smat[0], native_result.Smat[0], rtol=0.0, atol=3.0e-10)
