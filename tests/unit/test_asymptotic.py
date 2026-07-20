from collections.abc import Callable

import numpy as np
import pytest

import pyticc as ticc


def _atom_diatom_basis() -> ticc.ChannelBasis:
    atom = ticc.AtomSpec()
    diatom = ticc.DiatomSpec(Eint=np.zeros((1, 2)), vmax=0, jmax=1)
    system = ticc.ScattSystem(atom, diatom, Jtot=2, system_parity=1)
    return ticc.ChannelBuilder(system, ticc.TruncSpec()).build()


def _diatom_diatom_basis() -> ticc.ChannelBasis:
    diatom_X = ticc.DiatomSpec(Eint=np.zeros((1, 1)), vmax=0, jmax=0)
    diatom_Y = ticc.DiatomSpec(Eint=np.zeros((1, 2)), vmax=0, jmax=1)
    system = ticc.ScattSystem(diatom_X, diatom_Y, Jtot=2, system_parity=1)
    return ticc.ChannelBuilder(system, ticc.TruncSpec()).build()


@pytest.mark.parametrize("basis_factory", [_atom_diatom_basis, _diatom_diatom_basis])
def test_get_Bmat_BF_to_SF_diagonalizes_each_internal_block(basis_factory: Callable[[], ticc.ChannelBasis]) -> None:
    basis = basis_factory()
    indices = tuple(index for index, channel in enumerate(basis) if channel.j_couple == 1)
    Umat = ticc.get_Umat_BF(basis, indices)

    Bmat, L = ticc.get_Bmat_BF_to_SF(basis, indices)

    np.testing.assert_allclose(Bmat.T @ Bmat, np.eye(len(indices)), atol=1.0e-14)
    np.testing.assert_allclose(Bmat.T @ Umat @ Bmat, np.diag(L * (L + 1.0)), atol=1.0e-13)
    np.testing.assert_allclose(L, [1.0, 3.0], atol=1.0e-13)
    assert np.all(Bmat[-1] >= 0.0)


def test_transform_logD_BF_to_SF_supports_complex_batches() -> None:
    angle = 0.37
    Bmat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    Ymat = np.array(
        [
            [[1.0, 0.2], [0.2, 2.0]],
            [[1.0j, 0.2 - 0.1j], [0.2 - 0.1j, 2.0j]],
        ],
        dtype=np.complex128,
    )

    transformed = ticc.transform_logD_BF_to_SF(Ymat, Bmat)

    expected = np.stack([Bmat.T @ matrix @ Bmat for matrix in Ymat])
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_allclose(transformed, np.swapaxes(transformed, -1, -2))


def test_get_Bmat_BF_to_SF_returns_noninteger_L_for_incomplete_helicity_basis() -> None:
    atom = ticc.AtomSpec()
    diatom = ticc.DiatomSpec(Eint=np.zeros((1, 4)), vmax=0, jmax=3)
    system = ticc.ScattSystem(atom, diatom, Jtot=3, system_parity=-1)
    basis = ticc.ChannelBuilder(system, ticc.TruncSpec(K_cut=1)).build()
    indices = tuple(index for index, channel in enumerate(basis) if channel.j_couple == 3)

    _, L = ticc.get_Bmat_BF_to_SF(basis, indices)

    np.testing.assert_allclose(L, [2.0, np.sqrt(40.25) - 0.5], rtol=1.0e-13, atol=1.0e-13)
    assert not float(L[1]).is_integer()


def test_get_Bmat_BF_to_SF_keeps_electronic_states_in_separate_blocks() -> None:
    channels = tuple(
        ticc.Channel(
            mis_X=ticc.MolInnerState(j=0),
            mis_Y=ticc.MolInnerState(v=0, j=2, electronic_state=electronic_state),
            j_couple=2,
            K=K,
            Jtot=2,
            system_parity=1,
            E_int=float(electronic_state),
            index=index,
        )
        for index, (electronic_state, K) in enumerate((state, K) for state in range(2) for K in range(3))
    )
    basis = ticc.ChannelBasis(channels)

    Bmat, L = ticc.get_Bmat_BF_to_SF(basis)

    np.testing.assert_allclose(Bmat[:3, 3:], 0.0)
    np.testing.assert_allclose(Bmat[3:, :3], 0.0)
    np.testing.assert_allclose(L[:3], L[3:])
