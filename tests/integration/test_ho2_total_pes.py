from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import pyticc.pes.fortran.compiler as compiler_module
from pyticc.basis.delves import build_delves_basis, build_delves_qns, delves_angular_basis, delves_theta_basis
from pyticc.constants import AMU2AU, EV2AU
from pyticc.matrix.delves import asymptotic_potential, get_Vgrid_delves
from pyticc.matrix.delves_hamiltonian import get_Hmat_delves_K
from pyticc.pes import load_fortran_total_pes


@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_ho2_total_pes_drives_the_abc_basis_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pes_dir = Path(__file__).parents[2] / "example" / "HO2_diabatic" / "pes"
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    pes = load_fortran_total_pes(
        [pes_dir / "ho2-dpme.f", pes_dir / "long_range_H_O2.f"],
        pes_dir / "pyticc_total_wrapper.f90",
        workdir=pes_dir,
        lapack=True,
    )
    mass = tuple(np.asarray([15.99492, 15.99492, 1.007825]) * AMU2AU)

    basis = build_delves_basis(
        asymptotic_potential(pes, mass),
        mass,
        Jtot=0,
        system_parity=1,
        exchange_parity=0,
        jmax=30,
        K_cut=0,
        E_max=2.4 * EV2AU,
    )

    assert basis.rho_min == pytest.approx(3.04072, abs=1.0e-5)
    assert basis.scaled_r_max == pytest.approx(5.53)
    assert basis.n_sine == 105
    assert basis.n_vib_quad == 182
    assert basis.n_gamma_quad == 126

    theta, _, _ = delves_theta_basis(basis, rho=10.0)
    cos_gamma, _, _ = delves_angular_basis(basis)
    arrangement_1 = get_Vgrid_delves(pes, 10.0, 1, theta, cos_gamma, mass)
    arrangement_2 = get_Vgrid_delves(pes, 10.0, 2, theta, cos_gamma, mass)
    arrangement_3 = get_Vgrid_delves(pes, 10.0, 3, theta, cos_gamma, mass)

    assert arrangement_1.shape == arrangement_2.shape == arrangement_3.shape == (basis.n_vib_quad, basis.n_gamma_quad)
    assert np.all(np.isfinite([arrangement_1, arrangement_2, arrangement_3]))
    np.testing.assert_allclose(arrangement_1, arrangement_2[:, ::-1], rtol=1.0e-11, atol=1.0e-11)

    small_basis = replace(
        basis,
        jmax=2,
        n_sine=4,
        n_vib_quad=16,
        n_gamma_quad=12,
        angular_qns=build_delves_qns(mass, 0, 1, 0, 2, 0),
    )
    H_1 = get_Hmat_delves_K(small_basis, pes, 10.0, arrangement=1, K=0)
    H_2 = get_Hmat_delves_K(small_basis, pes, 10.0, arrangement=2, K=0)
    exchange_phase = np.repeat([1.0, -1.0, 1.0], small_basis.n_sine)

    np.testing.assert_allclose(H_1, exchange_phase[:, None] * H_2 * exchange_phase[None, :], rtol=1.0e-11, atol=1.0e-11)
    pes.close()
