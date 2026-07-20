import numpy as np
from scipy.special import roots_legendre

import pyticc as ticc
from pyticc.basis.angle import norm_reduced_wigner_d
from pyticc.basis.channel import ChannelBuilder, TruncSpec
from pyticc.basis.monomer import AtomSpec
from pyticc.basis.podvr import VibPODVR
from pyticc.basis.triatom import TriatomBasis, TriatomBlock
from pyticc.matrix.atom_triatom import prepare_Vmat_BF_atom_triatom
from pyticc.matrix.interaction import get_Vmat_BF
from pyticc.pes.wrapper import PESWrapper, get_Vgrid_atom_triatom
from pyticc.system import ScattSystem


def make_triatom() -> TriatomBasis:
    """Build a minimal normalized j=0,1 triatomic basis for interaction tests."""
    radial_1 = VibPODVR(grids=np.array([2.0]), energies=np.array([0.0]), wavefunctions=np.ones((1, 1)))
    radial_2 = VibPODVR(grids=np.array([2.2]), energies=np.array([0.0]), wavefunctions=np.ones((1, 1)))
    cos_theta, theta_weights = roots_legendre(6)
    qn = np.array([[0, 0, 0, 0]], dtype=np.int64)
    coefficients = np.ones((1, 1))
    t_indices = np.array([0], dtype=np.int64)
    return TriatomBasis(
        Eint=np.array([[0.0], [0.01]]),
        jmax=1,
        tmax=0,
        parity_block_sign=1,
        K0_blocks={
            0: TriatomBlock(j=0, K=0, qn=qn, coefficients=coefficients, t_indices=t_indices),
            1: TriatomBlock(j=1, K=0, qn=qn, coefficients=coefficients, t_indices=t_indices),
        },
        positive_K_blocks={1: TriatomBlock(j=1, K=1, qn=qn, coefficients=coefficients, t_indices=t_indices)},
        radial_1=radial_1,
        radial_2=radial_2,
        cos_theta=cos_theta,
        theta_weights=theta_weights,
    )


def test_reduced_wigner_d_matches_reference_normalization_and_phase() -> None:
    cos_theta, weights = roots_legendre(8)
    theta = np.arccos(cos_theta)
    d_0 = np.asarray(norm_reduced_wigner_d(0, 0, 0, theta))
    d_1 = np.asarray(norm_reduced_wigner_d(1, 0, 0, theta))

    np.testing.assert_allclose(np.sum(weights * d_0**2), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(np.sum(weights * d_1**2), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(np.sum(weights * d_0 * d_1), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(norm_reduced_wigner_d(1, 1, 0, np.pi / 2.0), -np.sqrt(3.0) / 2.0)


def test_constant_potential_is_identity_for_atom_triatom_basis() -> None:
    triatom = make_triatom()
    system = ScattSystem(AtomSpec(), triatom, Jtot=1, system_parity=-1)
    basis = ChannelBuilder(system, TruncSpec()).build()
    cos_theta_2, theta_weights_2 = roots_legendre(6)
    phi_grid, phi_weights = roots_legendre(8)
    phi = 0.5 * np.pi * (phi_grid + 1.0)
    phi_weights *= 0.5 * np.pi
    assert triatom.cos_theta is not None
    assert triatom.theta_weights is not None

    V_basis = prepare_Vmat_BF_atom_triatom(
        basis,
        triatom,
        triatom.cos_theta,
        triatom.theta_weights,
        cos_theta_2,
        theta_weights_2,
        phi,
        phi_weights,
    )
    Vmat = get_Vmat_BF(V_basis, np.full(V_basis.grid_shape, 1.75))

    assert V_basis.grid_shape == (1, 1, 6, 6, 8)
    np.testing.assert_allclose(Vmat, 1.75 * np.eye(basis.n_channel), atol=1.0e-13)


def test_atom_triatom_pes_grid_accepts_radial_batch() -> None:
    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        r_1, r_2, theta_1, theta_2, phi = coordinates
        return R + r_1 + 2.0 * r_2 + theta_1 + theta_2 + phi

    pes = PESWrapper(interaction=interaction)
    R = np.array([4.0, 5.0])
    grid = get_Vgrid_atom_triatom(
        pes,
        R,
        np.array([1.0, 2.0]),
        np.array([3.0]),
        np.array([0.1]),
        np.array([0.2, 0.3]),
        np.array([0.4]),
    )

    assert grid.shape == (2, 2, 1, 1, 2, 1)
    np.testing.assert_allclose(grid[1, 0, 0, 0, 1, 0], 5.0 + 1.0 + 6.0 + 0.1 + 0.3 + 0.4)


def test_run_atom_triatom_completes_minimal_exact_calculation() -> None:
    triatom = make_triatom()

    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        return np.zeros(coordinates.shape[1])

    result = ticc.run_atom_triatom(
        triatom,
        PESWrapper(interaction=interaction),
        Jtot=0,
        system_parity=1,
        Etot=np.array([0.02]),
        reduced_mass=1000.0,
        radial_boundaries=[4.0, 4.2],
        radial_half_steps=[0.05],
        trunc=TruncSpec(E_Y_cut=0.005),
        n_theta_2=4,
        n_phi=4,
    )

    assert isinstance(result, ticc.ScatteringResult)
    assert result.basis.n_channel == 1
    assert result.Y_BF.shape == (1, 1, 1)
    assert result.Smat[0].shape == (1, 1)
    assert np.all(np.isfinite(result.Smat))
