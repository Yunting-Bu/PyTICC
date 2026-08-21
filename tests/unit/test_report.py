import numpy as np

from pyticc import report
from pyticc.basis.channel import Channel, ChannelBasis, ChannelBasisElectricSF, ChannelElectricSF
from pyticc.basis.kblock import KBlock
from pyticc.basis.monomer import DiatomBasis
from pyticc.basis.podvr import VibPODVR
from pyticc.basis.rovib import RovibBasis
from pyticc.fine_structure import FSConstants, build_fs_channels, build_fs_monomer_basis
from pyticc.match.delves import DelvesAsymptoticBasis
from pyticc.result import ReactiveScatteringResult, ScatteringResult
from pyticc.system import MolInnerState


def _diabatic_basis() -> ChannelBasis:
    atom = MolInnerState(j=0)
    states = (
        MolInnerState(j=1, v=0, Eint=0.001, electronic_state=0),
        MolInnerState(j=3, v=0, Eint=0.002, electronic_state=0),
        MolInnerState(j=1, v=1, Eint=0.003, electronic_state=1),
    )
    return ChannelBasis(
        channels=tuple(
            Channel(
                mis_X=atom,
                mis_Y=state,
                j_couple=state.j,
                K=0,
                E_int=state.Eint,
            )
            for state in states
        ),
        Jtot=0,
        system_parity=1,
    )


def _result() -> ScatteringResult:
    basis = _diabatic_basis()
    energies = np.array([0.01, 0.02])
    matrices = np.zeros((2, 3, 3))
    first = np.arange(9, dtype=np.float64).reshape(3, 3).astype(np.complex128)
    second = first + 10.0 + 1.0j * np.arange(9, dtype=np.float64).reshape(3, 3)
    return ScatteringResult(
        basis=basis,
        Etot=energies,
        Y_propagated=matrices,
        asymptotic_transform=np.eye(3),
        L=np.array([0.0, 2.0, 1.0]),
        Smat=(first, second),
    )


def _electric_result() -> ScatteringResult:
    basis = ChannelBasisElectricSF(
        channels=(
            ChannelElectricSF(alpha=0, m=0, l=0, m_l=0, E_int=0.001),
            ChannelElectricSF(alpha=1, m=-1, l=1, m_l=1, E_int=0.002),
        ),
        M=0,
    )
    matrices = np.zeros((1, 2, 2))
    return ScatteringResult(
        basis=basis,
        Etot=np.array([0.01]),
        Y_propagated=matrices,
        asymptotic_transform=np.eye(2),
        L=np.array([0.0, 1.0]),
        Smat=(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128),),
    )


def _diatom_diatom_result() -> ScatteringResult:
    basis = ChannelBasis(
        channels=(
            Channel(
                mis_X=MolInnerState(v=0, j=0, Eint=0.001),
                mis_Y=MolInnerState(v=1, j=1, Eint=0.002),
                j_couple=1,
                K=0,
                E_int=0.003,
            ),
            Channel(
                mis_X=MolInnerState(v=1, j=2, Eint=0.003),
                mis_Y=MolInnerState(v=0, j=1, Eint=0.001),
                j_couple=2,
                K=0,
                E_int=0.004,
            ),
        ),
        Jtot=2,
        system_parity=1,
    )
    matrices = np.zeros((1, 2, 2))
    return ScatteringResult(
        basis=basis,
        Etot=np.array([0.01]),
        Y_propagated=matrices,
        asymptotic_transform=np.eye(2),
        L=np.array([1.0, 2.0]),
        Smat=(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128),),
    )


def _reactive_result() -> ReactiveScatteringResult:
    basis = DelvesAsymptoticBasis(
        qns=((1, 0, 0, 0), (2, 1, 1, 0)),
        energies=np.array([0.001, 0.002]),
        s_coefficients=np.zeros((1, 2)),
        rho_match=12.0,
        theta_coefficients=np.zeros((1, 2)),
        theta_energies=np.array([0.0011, 0.0021]),
    )
    matrices = np.zeros((1, 2, 2))
    return ReactiveScatteringResult(
        basis=basis,
        Etot=np.array([0.01]),
        Y_propagated=matrices,
        Y_asymptotic=matrices,
        Smat=(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128),),
        rho_final=12.0,
        surface_rho=11.5,
        radial_points=np.array([3.0, 12.0]),
    )


def test_rovib_levels_reports_only_cm_inverse() -> None:
    rovib = RovibBasis(
        grids=np.array([1.0]),
        E_vj=np.array([[0.0, 0.001]]),
        WF_vj=np.ones((1, 1, 2)),
    )
    basis = DiatomBasis(rovib=rovib, energy_zero=0.0)

    output = report.rovib_levels(basis)

    assert "E_int/cm-1" in output
    assert "219.474631" in output
    assert "a.u." not in output


def test_channels_omits_global_block_quantum_numbers_and_includes_electronic_state() -> None:
    output = report.channels(_diabatic_basis())

    header = output.splitlines()[0]
    assert "state" in header
    assert "v" in header
    assert "j" in header
    assert "K" in header
    assert "E_int/cm-1" in header
    assert "Jtot" not in output
    assert "parity" not in output


def test_open_closed_uses_cm_inverse() -> None:
    basis = _diabatic_basis()

    output = report.open_closed(basis, [0.01])

    assert "Etot/cm-1" in output
    assert f"{0.01 * 219474.6313705:.8f}" in output
    assert output.splitlines()[-1].split()[-2:] == ["3", "0"]


def test_electric_channels_reports_alpha_m_l_and_m_l() -> None:
    output = report.channels(_electric_result().basis)

    assert output.splitlines()[0].split() == ["n", "alpha", "m", "l", "m_l", "E_int/cm-1"]
    assert output.splitlines()[-1].split()[1:5] == ["1", "-1", "1", "1"]


def test_fine_structure_channels_report_block_quantum_numbers() -> None:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    monomer = build_fs_monomer_basis(vib, (1,), 2, 1, FSConstants(A=0.01, B=0.001))
    basis = build_fs_channels(monomer, two_J=1, system_parity=1)

    channel_output = report.channels(basis)
    count_output = report.open_closed(basis, [1.0])

    assert channel_output.splitlines()[0].split() == ["n", "v", "j", "tau", "epsilon", "K", "E_int/cm-1"]
    assert channel_output.splitlines()[-1].split()[1:6] == ["0", "0.5", "0", "1", "0.5"]
    assert count_output.splitlines()[-1].split()[-2:] == [str(basis.n_channel), "0"]


def test_fine_structure_levels_report_tau_parity_and_relative_energy() -> None:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    monomer = build_fs_monomer_basis(vib, (1,), 2, 1, FSConstants(A=0.01, B=0.001))

    output = report.fine_structure_levels(monomer)

    assert output.splitlines()[0].split() == ["v", "j", "tau", "epsilon", "E_int/cm-1"]
    assert "0.00000000" in output


def test_fine_structure_smatrix_reports_asymptotic_quantum_numbers() -> None:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    monomer = build_fs_monomer_basis(vib, (1,), 2, 1, FSConstants(A=0.01, B=0.001))
    basis = build_fs_channels(monomer, two_J=1, system_parity=1)
    matrix = np.arange(basis.n_channel**2, dtype=np.float64).reshape(basis.n_channel, basis.n_channel).astype(np.complex128)
    result = ScatteringResult(
        basis=basis,
        Etot=np.array([1.0]),
        Y_propagated=np.zeros((1, basis.n_channel, basis.n_channel)),
        asymptotic_transform=np.eye(basis.n_channel),
        L=np.ones(basis.n_channel),
        Smat=(matrix,),
    )

    output = report.smatrix(result)

    assert output.splitlines()[0].split()[1:11] == ["v", "j", "tau", "epsilon", "L", "v'", "j'", "tau'", "epsilon'", "L'"]
    assert f"{matrix[-1, 0].real:.16E}" in output


def test_delves_channels_and_open_closed_use_arrangement_quantum_numbers() -> None:
    result = _reactive_result()

    channel_output = report.channels(result.basis)
    count_output = report.open_closed(result.basis, result.Etot)

    assert channel_output.splitlines()[0].split() == ["n", "a", "v", "j", "K", "E_int/cm-1"]
    assert channel_output.splitlines()[-1].split()[1:5] == ["2", "1", "1", "0"]
    assert count_output.splitlines()[-1].split()[-2:] == ["2", "0"]


def test_k_blocks_reports_membership_without_writing_or_printing() -> None:
    block = KBlock(
        index=0,
        center_K=1,
        K_delta=1,
        K_values=(0, 1, 2),
        channel_indices=(0, 1, 2),
        owned_K_values=(1,),
        owned_channel_indices=(1,),
    )

    output = report.k_blocks([block])

    assert "center_K" in output
    assert "0,1,2" in output
    assert "owned_channels" in output


def test_smatrix_filters_energy_and_initial_and_final_vj_ranges() -> None:
    result = _result()

    output = report.smatrix(
        result,
        energy_indices=1,
        state=0,
        v=0,
        j=range(1, 2),
        state_prime=1,
        v_prime=range(1, 2),
        j_prime=[1],
    )

    lines = output.splitlines()
    assert "state'" in lines[0]
    assert "v'" in lines[0]
    assert "j'" in lines[0]
    assert len(lines) == 3
    assert f"{result.Etot[1] * 219474.6313705:.8f}" in lines[-1]
    assert "1.6000000000000000E+01" in lines[-1]
    assert "6.0000000000000000E+00" in lines[-1]


def test_smatrix_uses_outgoing_row_and_incoming_column() -> None:
    result = _result()

    output = report.smatrix(result, energy_indices=0, state=0, v=0, j=1, state_prime=0, v_prime=0, j_prime=3)

    assert "3.0000000000000000E+00" in output


def test_electric_smatrix_uses_sf_channel_quantum_numbers() -> None:
    output = report.smatrix(_electric_result())

    assert output.splitlines()[0].split()[1:9] == ["alpha", "m", "l", "m_l", "alpha'", "m'", "l'", "m_l'"]
    assert "3.0000000000000000E+00" in output


def test_electric_smatrix_filters_initial_and_final_channels() -> None:
    output = report.smatrix(
        _electric_result(),
        alpha=0,
        m=0,
        l=0,
        m_l=0,
        alpha_prime=1,
        m_prime=-1,
        l_prime=1,
        m_l_prime=1,
    )

    assert len(output.splitlines()) == 3
    assert "3.0000000000000000E+00" in output


def test_diatom_diatom_smatrix_filters_both_monomers() -> None:
    output = report.smatrix(
        _diatom_diatom_result(),
        v_X=0,
        j_X=0,
        v_Y=1,
        j_Y=1,
        j_couple=1,
        v_X_prime=1,
        j_X_prime=2,
        v_Y_prime=0,
        j_Y_prime=1,
        j_couple_prime=2,
    )

    header = output.splitlines()[0].split()
    assert header[1:7] == ["v_X", "j_X", "v_Y", "j_Y", "j_couple", "L"]
    assert len(output.splitlines()) == 3
    assert "3.0000000000000000E+00" in output


def test_diatom_diatom_smatrix_rejects_ambiguous_single_monomer_filter() -> None:
    with np.testing.assert_raises_regex(ValueError, "do not apply to diatom-diatom"):
        report.smatrix(_diatom_diatom_result(), v=0)


def test_delves_smatrix_filters_arrangement_and_vjK() -> None:
    output = report.smatrix(
        _reactive_result(),
        arrangement=1,
        v=0,
        j=0,
        K=0,
        arrangement_prime=2,
        v_prime=1,
        j_prime=1,
        K_prime=0,
    )

    assert output.splitlines()[0].split()[1:9] == ["a", "v", "j", "K", "a'", "v'", "j'", "K'"]
    assert len(output.splitlines()) == 3
    assert "3.0000000000000000E+00" in output


def test_smatrix_rejects_unmatched_quantum_numbers() -> None:
    result = _result()

    with np.testing.assert_raises_regex(ValueError, "initial-state selection"):
        report.smatrix(result, v=99)


def test_report_has_no_streaming_api() -> None:
    assert not hasattr(report, "channel_lines")
    assert not hasattr(report, "smatrix_lines")
