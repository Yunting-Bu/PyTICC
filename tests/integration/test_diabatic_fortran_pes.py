from pathlib import Path

import numpy as np
import pytest

import pyticc.pes.fortran.compiler as compiler_module
from pyticc.pes import get_DPEM_grid_atom_diatom, load_fortran_diabatic_pes

UNUSED_SOURCE = """
subroutine unused_source()
end subroutine unused_source
"""

DIABATIC_WRAPPER = """
subroutine pyticc_diabatic_interaction_grid(RR, coordinates, V, n_coordinate, n_grid, n_state)
    implicit none
    integer, intent(in) :: n_coordinate, n_grid, n_state
    real(8), intent(in) :: RR
    real(8), intent(in) :: coordinates(n_coordinate, n_grid)
    real(8), intent(out) :: V(n_grid, n_state, n_state)
    integer :: i

    V = 0.0d0
    do i = 1, n_grid
        V(i, 1, 1) = RR + coordinates(1, i) + cos(coordinates(2, i))
        V(i, 2, 2) = 2.0d0 * RR - coordinates(1, i)
        V(i, 1, 2) = sin(coordinates(2, i)) / RR
        V(i, 2, 1) = V(i, 1, 2)
    end do
end subroutine pyticc_diabatic_interaction_grid

subroutine pyticc_diabatic_monomer_grid(r, V, n_grid, n_state)
    implicit none
    integer, intent(in) :: n_grid, n_state
    real(8), intent(in) :: r(n_grid)
    real(8), intent(out) :: V(n_grid, n_state)

    V(:, 1) = (r - 1.5d0)**2
    V(:, 2) = (r - 2.0d0)**2 + 0.25d0
end subroutine pyticc_diabatic_monomer_grid
"""


@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_diabatic_fortran_wrapper_compiles_and_evaluates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "unused.f90"
    wrapper = tmp_path / "pyticc_diabatic_wrapper.f90"
    source.write_text(UNUSED_SOURCE)
    wrapper.write_text(DIABATIC_WRAPPER)
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    r = np.array([1.5, 2.0])
    theta = np.array([0.0, 0.5 * np.pi])

    pes = load_fortran_diabatic_pes([source], wrapper, n_state=2)
    dpem = get_DPEM_grid_atom_diatom(pes, 6.0, r, theta)

    assert dpem.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(dpem[..., 0, 0], 6.0 + r[:, None] + np.cos(theta)[None, :])
    np.testing.assert_allclose(dpem[..., 1, 1], 12.0 - r[:, None] + np.zeros_like(theta)[None, :])
    np.testing.assert_allclose(dpem[..., 0, 1], np.zeros_like(r)[:, None] + np.sin(theta)[None, :] / 6.0)
    np.testing.assert_allclose(pes.monomer_values(r), np.stack(((r - 1.5) ** 2, (r - 2.0) ** 2 + 0.25), axis=-1))

    parallel_pes = load_fortran_diabatic_pes([source], wrapper, n_state=2, processes=2)
    radial_points = np.array([5.0, 6.0])
    parallel_values = get_DPEM_grid_atom_diatom(parallel_pes, radial_points, r, theta)
    expected = np.stack([get_DPEM_grid_atom_diatom(pes, RR, r, theta) for RR in radial_points])
    np.testing.assert_allclose(parallel_values, expected)
    parallel_pes.close()
