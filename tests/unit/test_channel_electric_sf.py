import numpy as np

from pyticc.basis.channel import build_ChannelBasisElectricSF
from pyticc.basis.monomer import DiatomElectricBasis, DiatomElectricBlock
from pyticc.matrix.centrifugal import get_Umat_ElectricSF
from pyticc.system import ChannelSpec


def make_monomer_basis(m_values: tuple[int, ...] = (-2, -1, 0, 1, 2)) -> DiatomElectricBasis:
    jmax = 2
    blocks = []
    for m in m_values:
        j_values = np.arange(abs(m), jmax + 1, dtype=np.int64)
        energies = np.array([0.10 + 0.01 * abs(m), 0.20 + 0.01 * abs(m)])
        coefficients = np.zeros((1, j_values.size, energies.size))
        coefficients[0, 0, :] = 1.0
        blocks.append(
            DiatomElectricBlock(
                m=m,
                j_values=j_values,
                energies=energies,
                coefficients=coefficients,
            )
        )
    return DiatomElectricBasis(
        grids=np.array([1.75]),
        blocks=tuple(blocks),
        energy_zero=0.10,
        electric_strength=1.0e-3,
        jmax=jmax,
        mass=1000.0,
    )


def test_build_ChannelBasisElectricSF_determines_m_from_M_minus_m_l() -> None:
    basis = build_ChannelBasisElectricSF(make_monomer_basis(), M=1, lmax=2)

    assert basis.M == 1
    assert all(channel.m == basis.M - channel.m_l for channel in basis)
    assert all(abs(channel.m_l) <= channel.l for channel in basis)
    assert {(channel.l, channel.m_l, channel.m) for channel in basis} == {
        (0, 0, 1),
        (1, -1, 2),
        (1, 0, 1),
        (1, 1, 0),
        (2, -1, 2),
        (2, 0, 1),
        (2, 1, 0),
        (2, 2, -1),
    }


def test_build_ChannelBasisElectricSF_orders_thresholds() -> None:
    basis = build_ChannelBasisElectricSF(make_monomer_basis(), M=0, lmax=2)

    assert np.all(np.diff(basis.E_int) >= 0.0)


def test_build_ChannelBasisElectricSF_applies_relative_energy_cutoff() -> None:
    basis = build_ChannelBasisElectricSF(make_monomer_basis(), M=0, lmax=1, channel=ChannelSpec(E_Y_cut=0.015))

    assert basis.n_channel == 4
    assert {channel.alpha for channel in basis} == {0}
    assert np.all(basis.E_int <= 0.015)


def test_ChannelBasisElectricSF_classifies_open_and_closed_channels() -> None:
    basis = build_ChannelBasisElectricSF(make_monomer_basis(), M=0, lmax=0)

    result = basis.open_closed(np.array([0.0, 0.05, 0.15]))

    np.testing.assert_array_equal(result.n_open, np.array([0, 1, 2]))
    np.testing.assert_array_equal(result.n_closed, np.array([2, 1, 0]))


def test_get_Umat_ElectricSF_is_diagonal_and_preserves_selected_order() -> None:
    basis = build_ChannelBasisElectricSF(make_monomer_basis(), M=0, lmax=2, channel=ChannelSpec(E_Y_cut=0.015))
    indices = (2, 0, 1)

    Umat = get_Umat_ElectricSF(basis, indices)

    expected = np.diag([basis[index].l * (basis[index].l + 1) for index in indices])
    np.testing.assert_allclose(Umat, expected)
