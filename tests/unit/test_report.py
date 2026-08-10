import numpy as np

from pyticc import report
from pyticc.basis.channel import Channel, ChannelBasis, ChannelBasisElectricSF, ChannelElectricSF
from pyticc.basis.kblock import KBlock
from pyticc.basis.monomer import DiatomBasis
from pyticc.basis.podvr import RovibPODVR
from pyticc.result import ScatteringResult
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
                Jtot=0,
                system_parity=1,
                E_int=state.Eint,
                index=index,
            )
            for index, state in enumerate(states)
        )
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
        open_closed=basis.open_closed(energies),
        Y_propagated=matrices,
        asymptotic_transform=np.eye(3),
        L=np.array([0.0, 2.0, 1.0]),
        Y_asymptotic=matrices,
        Smat=(first, second),
    )


def _electric_result() -> ScatteringResult:
    basis = ChannelBasisElectricSF(
        channels=(
            ChannelElectricSF(alpha=0, m=0, l=0, m_l=0, E_int=0.001, index=0),
            ChannelElectricSF(alpha=1, m=-1, l=1, m_l=1, E_int=0.002, index=1),
        ),
        M=0,
    )
    matrices = np.zeros((1, 2, 2))
    return ScatteringResult(
        basis=basis,
        Etot=np.array([0.01]),
        open_closed=basis.open_closed([0.01]),
        Y_propagated=matrices,
        asymptotic_transform=np.eye(2),
        L=np.array([0.0, 1.0]),
        Y_asymptotic=matrices,
        Smat=(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128),),
    )


def test_rovib_levels_reports_only_cm_inverse() -> None:
    rovib = RovibPODVR(
        grids=np.array([1.0]),
        E_vj=np.array([[0.0, 0.001]]),
        WF_vj=np.ones((1, 1, 2)),
    )
    basis = DiatomBasis(rovib=rovib, energy_zero=0.0, vmax=0, jmax=1)

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


def test_smatrix_rejects_unmatched_quantum_numbers() -> None:
    result = _result()

    with np.testing.assert_raises_regex(ValueError, "initial-state selection"):
        report.smatrix(result, v=99)


def test_report_has_no_streaming_api() -> None:
    assert not hasattr(report, "channel_lines")
    assert not hasattr(report, "smatrix_lines")
