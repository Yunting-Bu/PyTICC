import numpy as np
import pytest
from scipy.special import roots_legendre

import pyticc.basis.monomer.diatom_electric as electric_module
from pyticc.basis.dvr import build_SineDVR
from pyticc.basis.monomer import build_DiatomElectricBasis, diatom_electric_amplitude, required_m_values
from pyticc.basis.podvr import build_RovibPODVR
from pyticc.electric import ElectricResponseTable


def _zero_response() -> ElectricResponseTable:
    zeros = np.zeros(2)
    return ElectricResponseTable(
        r=np.array([0.7, 3.3]),
        mu_z=zeros,
        alpha_xx=zeros,
        alpha_zz=zeros,
        beta_zzz=zeros,
        beta_xxz=zeros,
    )


def _nonzero_response() -> ElectricResponseTable:
    return ElectricResponseTable(
        r=np.array([0.7, 3.3]),
        mu_z=np.array([0.8, 1.0]),
        alpha_xx=np.array([4.0, 5.0]),
        alpha_zz=np.array([6.0, 7.0]),
        beta_zzz=np.array([2.0, 3.0]),
        beta_xxz=np.array([0.5, 0.8]),
    )


def _dvr():
    return build_SineDVR(0.8, 3.2, 40, 1000.0, lambda r: 0.03 * (r - 1.8) ** 2)


def test_required_m_values_are_derived_from_M_lmax_and_jmax() -> None:
    assert required_m_values(M=2, lmax=3, jmax=4) == (-1, 0, 1, 2, 3, 4)
    assert required_m_values(M=-3, lmax=1, jmax=5) == (-4, -3, -2)


def test_zero_field_basis_recovers_field_free_podvr_spectrum() -> None:
    dvr = _dvr()
    n_podvr = 8
    jmax = 3
    basis = build_DiatomElectricBasis(
        dvr,
        _zero_response(),
        electric_strength=0.0,
        n_podvr=n_podvr,
        jmax=jmax,
        M=0,
        lmax=0,
        n_alpha=n_podvr * (jmax + 1),
        mass=1000.0,
    )
    field_free = build_RovibPODVR(dvr, n_podvr=n_podvr, vmax=n_podvr - 1, jmax=jmax, mass=1000.0)

    np.testing.assert_allclose(basis.block(0).energies, np.sort(field_free.E_vj.ravel()), atol=2.0e-13)
    assert basis.energy_zero == basis.block(0).energies[0]
    assert not hasattr(basis, "M")
    assert not hasattr(basis, "lmax")


def test_positive_and_negative_m_blocks_have_equal_energies() -> None:
    basis = build_DiatomElectricBasis(
        _dvr(),
        _nonzero_response(),
        electric_strength=0.02,
        n_podvr=8,
        jmax=2,
        M=0,
        lmax=2,
        n_alpha=5,
        mass=1000.0,
    )

    np.testing.assert_allclose(basis.block(-1).energies, basis.block(1).energies, atol=2.0e-13)
    np.testing.assert_allclose(basis.block(-2).energies, basis.block(2).energies, atol=2.0e-13)


def test_positive_and_negative_m_share_one_diagonalization(monkeypatch: pytest.MonkeyPatch) -> None:
    original_solver = electric_module.solve_diatom_electric_block
    solved_m: list[int] = []

    def tracked_solver(*args, **kwargs):
        solved_m.append(kwargs["m"])
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(electric_module, "solve_diatom_electric_block", tracked_solver)
    build_DiatomElectricBasis(
        _dvr(),
        _nonzero_response(),
        electric_strength=0.02,
        n_podvr=8,
        jmax=2,
        M=0,
        lmax=2,
        n_alpha=5,
        mass=1000.0,
    )

    assert solved_m == [2, 1, 0]


def test_amplitude_is_orthonormal_on_weighted_angular_grid() -> None:
    basis = build_DiatomElectricBasis(
        _dvr(),
        _nonzero_response(),
        electric_strength=0.02,
        n_podvr=8,
        jmax=3,
        M=0,
        lmax=0,
        n_alpha=6,
        mass=1000.0,
    )
    nodes, weights = roots_legendre(20)

    amplitude = diatom_electric_amplitude(basis.block(0), nodes, weights)
    overlap = np.einsum("apq,bpq->ab", amplitude, amplitude, optimize=True)

    np.testing.assert_allclose(overlap, np.eye(6), atol=3.0e-13)
