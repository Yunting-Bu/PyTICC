from pathlib import Path

import numpy as np
import pytest

import pyticc.pes.fortran.compiler as compiler_module
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

LAPACK_SOURCE = """
subroutine lowest_eigenvalue(RR, coupling, value)
    implicit none
    real(8), intent(in) :: RR, coupling
    real(8), intent(out) :: value
    real(8) :: matrix(2, 2), eigenvalues(2), work(6)
    integer :: info

    matrix(1, 1) = RR
    matrix(1, 2) = coupling
    matrix(2, 1) = coupling
    matrix(2, 2) = RR + 2.0d0
    call dsyev('N', 'U', 2, matrix, 2, eigenvalues, work, 6, info)
    if (info /= 0) stop "DSYEV failed"
    value = eigenvalues(1)
end subroutine lowest_eigenvalue
"""

LAPACK_WRAPPER = """
subroutine pyticc_interaction_grid(RR, coordinates, V, n_coordinate, n_grid)
    implicit none
    integer, intent(in) :: n_coordinate, n_grid
    real(8), intent(in) :: RR, coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid)
    integer :: i

    do i = 1, n_grid
        call lowest_eigenvalue(RR, coordinates(1, i), V(i))
    end do
end subroutine pyticc_interaction_grid
"""


@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
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
    monkeypatch.setattr(compiler_module, "_require_build_tools", lambda: pytest.fail("Cached PES requested the build toolchain"))
    cached_pes = load_fortran_pes(config)
    np.testing.assert_allclose(get_Vgrid_atom_diatom(cached_pes, 6.0, r, theta), expected)

    parallel_pes = load_fortran_pes(config, processes=2)
    radial_batch = np.array([6.0, 7.0])
    parallel_values = get_Vgrid_atom_diatom(parallel_pes, radial_batch, r, theta)
    parallel_expected = radial_batch[:, None, None] + 2.0 * r[None, :, None] + theta[None, None, :] / np.pi
    np.testing.assert_allclose(parallel_values, parallel_expected)
    np.testing.assert_allclose(get_Vgrid_atom_diatom(parallel_pes, np.array([8.0]), r, theta)[0], 8.0 + 2.0 * r[:, None] + theta[None, :] / np.pi)
    parallel_pes.close()


@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_fortran_pes_links_and_executes_lapack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "lapack_pes.f90"
    wrapper = tmp_path / "pyticc_wrapper.f90"
    config = tmp_path / "lapack.toml"
    source.write_text(LAPACK_SOURCE)
    wrapper.write_text(LAPACK_WRAPPER)
    config.write_text("\n".join(('sources = ["lapack_pes.f90"]', 'wrapper = "pyticc_wrapper.f90"', "lapack = true")))
    cache = tmp_path / "cache"
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(cache))

    pes = load_fortran_pes(config)
    coupling = np.array([0.0, 0.5])
    values = get_Vgrid_atom_diatom(pes, R=3.0, r=coupling, theta=np.array([0.0]))[:, 0]

    np.testing.assert_allclose(values, 4.0 - np.sqrt(1.0 + coupling**2))
    build_log = next(cache.glob("pyticc_pes_*/build.log")).read_text()
    assert "lapack=true" in build_log
    assert "dependencies=lapack" in build_log
