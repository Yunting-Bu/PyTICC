from pathlib import Path

import pytest

import pyticc.pes.fortran.compiler as compiler_module
from pyticc.pes import load_fortran_pes


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
