import numpy as np

from pyticc.matrix.radial import get_Wmat


def test_get_Wmat_has_open_and_closed_channel_signs() -> None:
    E_int = np.array([0.1, 0.3])
    zeros = np.zeros((2, 2))

    Wmat = get_Wmat(R=5.0, Etot=0.2, reduced_mass=10.0, E_int=E_int, Umat=zeros, Vmat=zeros)

    np.testing.assert_allclose(Wmat, np.diag([-2.0, 2.0]))


def test_get_Wmat_combines_centrifugal_and_interaction_couplings() -> None:
    E_int = np.array([0.1, 0.2])
    Umat = np.array([[2.0, -0.5], [-0.5, 6.0]])
    Vmat = np.array([[0.01, 0.02], [0.02, 0.03]])

    Wmat = get_Wmat(R=4.0, Etot=0.15, reduced_mass=2.0, E_int=E_int, Umat=Umat, Vmat=Vmat)

    expected = Umat / 16.0 + 4.0 * Vmat
    expected[np.diag_indices(2)] += 4.0 * (E_int - 0.15)
    np.testing.assert_allclose(Wmat, expected)


def test_get_Wmat_does_not_modify_input_matrices() -> None:
    E_int = np.array([0.1, 0.2])
    Umat = np.array([[2.0, -0.5], [-0.5, 6.0]])
    Vmat = np.array([[0.01, 0.02], [0.02, 0.03]])
    Umat_before = Umat.copy()
    Vmat_before = Vmat.copy()

    get_Wmat(R=4.0, Etot=0.15, reduced_mass=2.0, E_int=E_int, Umat=Umat, Vmat=Vmat)

    np.testing.assert_array_equal(Umat, Umat_before)
    np.testing.assert_array_equal(Vmat, Vmat_before)
