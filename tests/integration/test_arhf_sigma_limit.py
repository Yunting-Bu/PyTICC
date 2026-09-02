from pathlib import Path

import numpy as np
import pytest

import pyticc as ticc
import pyticc.pes.fortran.compiler as compiler_module
from pyticc.basis.podvr import VibPODVR
from pyticc.fine_structure import FSConstants, build_fs_monomer_basis
from pyticc.propagation.grid import build_radial_sectors
from pyticc.scattering import atom_diatom


@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_ArHF_singlet_sigma_path_matches_scalar_ticc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).parents[2]
    pes_dir = root / "example" / "ArHF" / "pes"
    source = pes_dir / "interaction-PES.f"
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    scalar_pes = ticc.load_fortran_pes([source], pes_dir / "pyticc_wrapper.f90", workdir=pes_dir)
    lambda_pes = ticc.load_fortran_lambda_pes([source], pes_dir / "pyticc_fs_wrapper.f90", workdir=pes_dir)
    assert scalar_pes.monomer_Y is not None

    scalar_diatom = ticc.prepare_Diatom(
        scalar_pes.monomer_Y,
        r=(1.5, 4.5),
        n_dvr=30,
        n_podvr=3,
        vmax=1,
        jmax=0,
        mass=ticc.reduced_mass(ticc.element_mass_au("H"), ticc.element_mass_au("F")),
    )
    collision_mass = ticc.reduced_mass(ticc.element_mass_au("Ar"), ticc.element_mass_au("H") + ticc.element_mass_au("F"))
    monomer_X = ticc.AtomSpec()
    scalar_system = ticc.build_ScattSystem(
        monomer_X=monomer_X,
        monomer_Y=scalar_diatom,
        scattering_type="A+BC",
        Jtot=0,
        system_parity=1,
        reduced_mass=collision_mass,
        potential=scalar_pes,
    )
    vib = VibPODVR(
        scalar_diatom.rovib.grids,
        scalar_diatom.rovib.E_vj[:, 0],
        scalar_diatom.rovib.WF_vj[:, :, 0],
    )
    fs_monomer = build_fs_monomer_basis(vib, two_j_values=(0,), two_lambda_abs=0, two_S=0, constants=FSConstants())
    fs_system = ticc.build_ScattSystem(
        monomer_X,
        fs_monomer,
        scattering_type="A+BC_fine_structure",
        two_J=0,
        system_parity=1,
        reduced_mass=collision_mass,
        potential=lambda_pes,
    )
    scalar_hamiltonian = atom_diatom.build_hamiltonian(scalar_system, n_theta=12)
    fs_hamiltonian = ticc.build_fs_hamiltonian(fs_system, n_theta=12)

    radial_points = np.array([5.0, 6.0, 8.0])
    np.testing.assert_allclose(fs_hamiltonian.V(radial_points), scalar_hamiltonian.V(radial_points), atol=2.0e-15, rtol=2.0e-13)
    np.testing.assert_allclose(fs_hamiltonian.H(6.0), scalar_hamiltonian.H(6.0), atol=2.0e-15, rtol=2.0e-13)

    energy = float(max(scalar_hamiltonian.E_int) + 0.02)
    radial_sectors = build_radial_sectors((5.0, 5.2), (0.1,))
    propagation = ticc.Propagation()
    scalar_result = ticc.solve(scalar_hamiltonian, [energy], radial_sectors, propagation)
    fs_result = ticc.solve(fs_hamiltonian, [energy], radial_sectors, propagation)
    assert isinstance(scalar_result, ticc.ScatteringResult)
    assert isinstance(fs_result, ticc.ScatteringResult)
    np.testing.assert_allclose(fs_result.Y_propagated, scalar_result.Y_propagated, atol=2.0e-13, rtol=2.0e-12)
    np.testing.assert_allclose(fs_result.Smat[0], scalar_result.Smat[0], atol=2.0e-12, rtol=2.0e-11)

    scalar_pes.close()
    lambda_pes.close()
