from pathlib import Path

import pytest

import pyticc.pes.fortran.compiler as compiler_module
from pyticc.pes import load_fortran_diabatic_pes, load_fortran_pes


def test_load_fortran_pes_reports_missing_build_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "pes.f90"
    wrapper = tmp_path / "wrapper.f90"
    source.write_text("subroutine unused\nend subroutine unused\n")
    wrapper.write_text(
        """
subroutine pyticc_interaction_grid(RR, coordinates, V, n_coordinate, n_grid)
    integer, intent(in) :: n_coordinate, n_grid
    real(8), intent(in) :: RR, coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid)
end subroutine pyticc_interaction_grid
"""
    )
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(compiler_module, "_build_tools", lambda: (True, None, "/usr/bin/ninja", "/usr/bin/gfortran"))

    with pytest.raises(RuntimeError, match="Meson: missing"):
        load_fortran_pes([source], wrapper)


def test_load_fortran_diabatic_pes_requires_both_wrapper_routines(tmp_path: Path) -> None:
    source = tmp_path / "pes.f90"
    wrapper = tmp_path / "wrapper.f90"
    source.write_text("subroutine unused\nend subroutine unused\n")
    wrapper.write_text(
        """
subroutine pyticc_diabatic_interaction_grid(RR, coordinates, V, n_coordinate, n_grid, n_state)
    integer, intent(in) :: n_coordinate, n_grid, n_state
    real(8), intent(in) :: RR, coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid, n_state, n_state)
end subroutine pyticc_diabatic_interaction_grid
"""
    )

    with pytest.raises(ValueError, match="pyticc_diabatic_monomer_grid"):
        load_fortran_diabatic_pes([source], wrapper)


def test_load_fortran_diabatic_pes_validates_state_count_before_sources() -> None:
    with pytest.raises(ValueError, match="n_state must be positive"):
        load_fortran_diabatic_pes([], n_state=0)


def test_load_fortran_pes_requires_boolean_lapack_switch() -> None:
    with pytest.raises(ValueError, match="lapack must be a boolean"):
        load_fortran_pes([], Path("missing.f90"), lapack="yes")  # type: ignore[arg-type]


def test_fortran_source_digest_tracks_local_include_files(tmp_path: Path) -> None:
    include = tmp_path / "parameters.inc"
    source = tmp_path / "pes.f"
    wrapper = tmp_path / "wrapper.f90"
    include.write_text("real(8), parameter :: scale = 1.0d0\n")
    source.write_text("include 'parameters.inc'\n")
    wrapper.write_text("subroutine wrapper\nend subroutine wrapper\n")

    original = compiler_module._source_digest((source,), wrapper)
    include.write_text("real(8), parameter :: scale = 2.0d0\n")

    assert compiler_module._source_digest((source,), wrapper) != original


def test_fortran_source_digest_tracks_lapack_and_compiler(tmp_path: Path) -> None:
    source = tmp_path / "pes.f90"
    wrapper = tmp_path / "wrapper.f90"
    source.write_text("subroutine unused\nend subroutine unused\n")
    wrapper.write_text("subroutine wrapper\nend subroutine wrapper\n")

    default = compiler_module._source_digest((source,), wrapper)

    assert compiler_module._source_digest((source,), wrapper, lapack=True) != default
    assert compiler_module._source_digest((source,), wrapper, compiler="ifx") != default


def test_lapack_options_use_meson_for_gfortran_and_onemkl_for_intel() -> None:
    intel_flag = "/Qmkl:sequential" if compiler_module.os.name == "nt" else "-qmkl=sequential"

    assert compiler_module._lapack_options("/usr/bin/gfortran", False) == ((), ())
    assert compiler_module._lapack_options("/usr/bin/gfortran", True) == (("lapack",), ())
    assert compiler_module._lapack_options("/opt/intel/bin/ifx", True) == ((), (intel_flag,))


def test_lapack_fallback_prefers_mkl_from_python_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    library_name = "mkl_rt.lib" if compiler_module.os.name == "nt" else "libmkl_rt.so"
    (library_dir / library_name).touch()
    monkeypatch.setattr(compiler_module.sys, "base_prefix", str(tmp_path))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("MKLROOT", raising=False)

    link_args = compiler_module._lapack_fallback_link_args()

    assert link_args[0] == f"-L{library_dir}"
    assert link_args[-1] == "-lmkl_rt"


def test_lapack_fallback_uses_system_libraries_without_mkl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compiler_module.sys, "base_prefix", str(tmp_path))
    monkeypatch.setattr(compiler_module.sys, "prefix", str(tmp_path))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("MKLROOT", raising=False)

    assert compiler_module._lapack_fallback_link_args() == ("-llapack", "-lblas")
