import numpy as np

from pyticc.basis.channel import Channel, ChannelBasis
from pyticc.matrix.centrifugal import get_Umat_BF
from pyticc.system import MolInnerState


def make_atom_diatom_basis(v_values: tuple[int, ...] = (0,)) -> ChannelBasis:
    channels = tuple(
        Channel(
            mis_X=MolInnerState(j=0),
            mis_Y=MolInnerState(v=v, j=2),
            j_couple=2,
            K=K,
            Jtot=2,
            system_parity=1,
            E_int=float(v),
            index=index,
        )
        for index, (v, K) in enumerate((v, K) for v in v_values for K in range(3))
    )
    return ChannelBasis(channels)


def test_get_Umat_BF_has_expected_diagonal_and_coriolis_couplings() -> None:
    basis = make_atom_diatom_basis()

    Umat = get_Umat_BF(basis)

    expected = np.array(
        [
            [12.0, -6.0 * np.sqrt(2.0), 0.0],
            [-6.0 * np.sqrt(2.0), 10.0, -4.0],
            [0.0, -4.0, 4.0],
        ]
    )
    np.testing.assert_allclose(Umat, expected)


def test_get_Umat_BF_can_remove_coriolis_couplings_for_CS() -> None:
    basis = make_atom_diatom_basis()

    Umat = get_Umat_BF(basis, coriolis=False)

    np.testing.assert_allclose(Umat, np.diag([12.0, 10.0, 4.0]))


def test_get_Umat_BF_does_not_couple_different_internal_states() -> None:
    basis = make_atom_diatom_basis(v_values=(0, 1))

    Umat = get_Umat_BF(basis)

    np.testing.assert_allclose(Umat[:3, 3:], 0.0)
    np.testing.assert_allclose(Umat[3:, :3], 0.0)


def test_get_Umat_BF_preserves_requested_channel_order() -> None:
    basis = make_atom_diatom_basis()
    indices = (2, 1, 0)

    full_Umat = get_Umat_BF(basis)
    block_Umat = get_Umat_BF(basis, indices)

    np.testing.assert_allclose(block_Umat, full_Umat[np.ix_(indices, indices)])
