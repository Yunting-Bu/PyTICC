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
