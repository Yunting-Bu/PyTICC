from pathlib import Path

import numpy as np
import pytest

import pyticc.pes.fortran as fortran_module
from pyticc.pes import get_Vgrid_atom_diatom, load_fortran_pes

ARHF_SOURCE = """
subroutine arhf_point(RR, r, theta_degree, V)
    implicit none
    real(8), intent(in) :: RR, r, theta_degree
    real(8), intent(out) :: V

    V = RR + 2.0d0 * r + theta_degree / 180.0d0
end subroutine arhf_point

subroutine hf_point(r, V)
    implicit none
    real(8), intent(in) :: r
    real(8), intent(out) :: V

    V = (r - 1.75d0)**2
end subroutine hf_point
"""

ARHF_WRAPPER = """
subroutine pyticc_interaction_grid(RR, coordinates, V, n_coordinate, n_grid)
    implicit none
    integer, intent(in) :: n_coordinate, n_grid
    real(8), intent(in) :: RR
    real(8), intent(in) :: coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid)
    integer :: i
    real(8), parameter :: pi = acos(-1.0d0)

    do i = 1, n_grid
        call arhf_point(RR, coordinates(1, i), coordinates(2, i) * 180.0d0 / pi, V(i))
    end do
end subroutine pyticc_interaction_grid

subroutine pyticc_monomer_y_grid(r, V, n_grid)
    implicit none
    integer, intent(in) :: n_grid
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid)
    integer :: i

    do i = 1, n_grid
        call hf_point(r(i), V(i))
    end do
end subroutine pyticc_monomer_y_grid
"""


@pytest.mark.skipif(not all(fortran_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_ArHF_fortran_wrapper_compiles_and_evaluates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "ArHF_PES.f90"
    wrapper = tmp_path / "pyticc_wrapper.f90"
    source.write_text(ARHF_SOURCE)
    wrapper.write_text(ARHF_WRAPPER)
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    pes = load_fortran_pes([source], wrapper)
    r = np.array([1.5, 2.0])
    theta = np.array([0.0, 0.5 * np.pi])

    V = get_Vgrid_atom_diatom(pes, R=6.0, r=r, theta=theta)

    expected = 6.0 + 2.0 * r[:, None] + theta[None, :] / np.pi
    np.testing.assert_allclose(V, expected)
    assert pes.monomer_Y is not None
    np.testing.assert_allclose(pes.monomer_Y(r), (r - 1.75) ** 2)

    config = tmp_path / "arhf.toml"
    config.write_text(
        "\n".join(
            (
                'sources = ["ArHF_PES.f90"]',
                'wrapper = "pyticc_wrapper.f90"',
                'workdir = "."',
            )
        )
    )
    monkeypatch.setattr(fortran_module, "_require_build_tools", lambda: pytest.fail("Cached PES requested the build toolchain"))
    cached_pes = load_fortran_pes(config)
    np.testing.assert_allclose(get_Vgrid_atom_diatom(cached_pes, 6.0, r, theta), expected)
