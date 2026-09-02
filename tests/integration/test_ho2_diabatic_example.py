from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import pyticc.pes.fortran.compiler as compiler_module
from pyticc.pes import (
    DiabaticPESWrapper,
    get_diabatic_potential_grid_atom_diatom,
    load_fortran_diabatic_pes,
)

PES_DIR = Path(__file__).parents[2] / "example" / "HO2_diabatic" / "pes"
HAS_FORTRAN_TOOLCHAIN = all(compiler_module._build_tools())


@pytest.fixture(scope="module")
def ho2_pes() -> Iterator[DiabaticPESWrapper]:
    pes = load_fortran_diabatic_pes(
        [PES_DIR / "ho2-dpme.f", PES_DIR / "long_range_H_O2.f"],
        PES_DIR / "pyticc_wrapper.f90",
        workdir=PES_DIR,
        lapack=True,
    )
    yield pes
    pes.close()


@pytest.mark.skipif(not HAS_FORTRAN_TOOLCHAIN, reason="Fortran f2py/Meson toolchain is unavailable")
def test_ho2_adapter_returns_reference_monomers_and_respects_oxygen_exchange(ho2_pes: DiabaticPESWrapper) -> None:
    monomer = ho2_pes.monomer_values(np.array([2.0, 2.3]))
    np.testing.assert_allclose(
        monomer,
        np.array(
            [
                [4.59166182e-2, 8.36779692e-2],
                [7.13550696e-5, 3.55130975e-2],
            ]
        ),
        rtol=1.0e-8,
        atol=1.0e-11,
    )

    coordinates = np.asfortranarray([[2.3, 2.3], [np.pi / 3.0, 2.0 * np.pi / 3.0]])
    dpem = ho2_pes.interaction(6.0, coordinates)
    np.testing.assert_allclose(
        dpem[0],
        np.array(
            [
                [-2.48828309e-4, -1.57852332e-4],
                [-1.57852332e-4, -3.59143055e-5],
            ]
        ),
        rtol=1.0e-8,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(np.diagonal(dpem[1]), np.diagonal(dpem[0]), atol=1.0e-12)
    np.testing.assert_allclose(dpem[1, 0, 1], -dpem[0, 0, 1], atol=1.0e-12)
    np.testing.assert_allclose(dpem, np.swapaxes(dpem, -2, -1), atol=0.0)

    asymptotic = ho2_pes.interaction(7.0, coordinates)
    np.testing.assert_allclose(asymptotic[:, 0, 1], 0.0, atol=0.0)
    np.testing.assert_allclose(asymptotic[:, 1, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(asymptotic[:, 1, 1], 0.0, atol=0.0)


@pytest.mark.skipif(not HAS_FORTRAN_TOOLCHAIN, reason="Fortran f2py/Meson toolchain is unavailable")
def test_ho2_adapter_initializes_independently_in_worker_processes(ho2_pes: DiabaticPESWrapper) -> None:
    radial_points = np.array([5.5, 6.0])
    r_oo = np.array([2.3])
    theta = np.array([np.pi / 3.0])
    parallel_pes = load_fortran_diabatic_pes(
        [PES_DIR / "ho2-dpme.f", PES_DIR / "long_range_H_O2.f"],
        PES_DIR / "pyticc_wrapper.f90",
        workdir=PES_DIR,
        lapack=True,
    )
    try:
        actual = get_diabatic_potential_grid_atom_diatom(
            parallel_pes,
            radial_points,
            r_oo,
            theta,
            processes=2,
        )
        expected = np.stack([get_diabatic_potential_grid_atom_diatom(ho2_pes, R, r_oo, theta) for R in radial_points])
        np.testing.assert_allclose(actual, expected, atol=1.0e-13)
    finally:
        parallel_pes.close()
