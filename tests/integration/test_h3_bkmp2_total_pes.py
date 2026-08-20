import subprocess
from io import StringIO
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

import pyticc.pes.fortran.compiler as compiler_module
from pyticc.basis.channel import ChannelSpec
from pyticc.basis.delves import DelvesBasis, build_delves_channels
from pyticc.basis.monomer.delves import build_delves_diatom_basis, prepare_Delves
from pyticc.constants import AMU2AU
from pyticc.match.delves import build_delves_asymptotic_basis, transform_logD_to_delves_channels
from pyticc.match.delves_bessel import get_delves_Smat, match_delves
from pyticc.matrix.delves import asymptotic_potential, delves_bonds, get_Vgrid_delves, mass_scale
from pyticc.matrix.delves_metric import get_sector_transform_delves
from pyticc.matrix.delves_surface import get_surface_matrices_delves, solve_surface_delves
from pyticc.pes import load_fortran_total_pes
from pyticc.propagation import Propagation, propagate_delves
from pyticc.result import ReactiveScatteringResult
from pyticc.scattering.delves_hamiltonian import DelvesHamiltonian
from pyticc.scattering.reactive_atom_diatom import build_hamiltonian
from pyticc.scattering.solver import solve
from pyticc.system import build_ScattSystem

BKMP2_SOURCE = Path("/Users/byt/software/TICC/ABC_reac/bkmp2.f")
WRAPPER = Path(__file__).parents[2] / "example" / "H3_Delves" / "pyticc_total_wrapper.f90"
ABC_ENERGY_SHIFT_HARTREE = (4.47809 + 0.27018326) / 27.2114
ABC_EV_PER_HARTREE = 27.2114
ABC_HBAR_SQUARED = 0.014927625
ABC_SOURCE = Path("/Users/byt/software/TICC/ABC_reac/abc.f")
ABC_FUNCTIONS = Path("/Users/byt/software/TICC/ABC_reac/fun.f")
ABC_LINEAR_ALGEBRA = Path("/Users/byt/software/TICC/ABC_reac/lin.f")
ABC_ADAPTER = Path(__file__).parents[2] / "example" / "H3_Delves" / "abc_bkmp2_adapter.f90"
ABC_INPUT = Path(__file__).parents[2] / "example" / "H3_Delves" / "abc_J0.in"


def _compile_abc_probe(tmp_path: Path) -> Path:
    source = tmp_path / "abc_bkmp2_probe.f90"
    source.write_text(
        """
program abc_bkmp2_probe
    implicit none
    real(8) :: bonds(3), derivatives(3), raw_energy, shifted_energy(1,1)

    do
        read(*, *, end=100) bonds
        call bkmp2(bonds, raw_energy, derivatives, -1)
        call pot0(1, shifted_energy, bonds)
        write(*, '(2ES25.16E3)') raw_energy, shifted_energy(1,1)
    end do
100 continue
end program abc_bkmp2_probe
"""
    )
    executable = tmp_path / "abc_bkmp2_probe"
    subprocess.run(
        ["gfortran", "-std=legacy", "-O0", str(BKMP2_SOURCE), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def _abc_values(executable: Path, bonds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    completed = subprocess.run(
        [str(executable)],
        input="\n".join(" ".join(f"{value:.17e}" for value in point) for point in bonds.T) + "\n",
        check=True,
        capture_output=True,
        text=True,
    )
    values = np.loadtxt(StringIO(completed.stdout), ndmin=2)
    return values[:, 0], values[:, 1]


def _compile_abc_potenl_probe(tmp_path: Path) -> Path:
    source = tmp_path / "abc_potenl_probe.f90"
    source.write_text(
        """
program abc_potenl_probe
    implicit none
    real(8) :: mass(3), mtot, mred, scale(3), rmlmda
    real(8) :: ra, sa, cosa, value
    integer :: arrangement, i
    common /masses/ mass, mtot, mred
    common /scales/ scale, rmlmda

    mass = 1.007825d0
    mtot = sum(mass)
    mred = sqrt(product(mass) / mtot)
    do i = 1, 3
        scale(i) = sqrt((mass(i) / mred) * (1.d0 - mass(i) / mtot))
    end do
    rmlmda = 2.d0 * mred / 0.014927625d0
    do
        read(*, *, end=100) ra, sa, cosa, arrangement
        call potenl(ra, sa, cosa, value, arrangement)
        write(*, '(ES25.16E3)') value
    end do
100 continue
end program abc_potenl_probe

subroutine potenl(ra, sa, cosa, va, ia)
    implicit none
    real(8), intent(in) :: ra, sa, cosa
    real(8), intent(out) :: va
    integer, intent(in) :: ia
    real(8) :: mass(3), mtot, mred, scale(3), rmlmda
    real(8) :: r(3), rap, cap, sap, sbp, scp, vev
    integer :: ib, ic
    common /masses/ mass, mtot, mred
    common /scales/ scale, rmlmda

    ib = ia + 1
    if (ib > 3) ib = 1
    ic = 6 - ia - ib
    rap = ra / scale(ia)
    cap = 2.d0 * cosa * rap
    sap = scale(ia) * sa
    sbp = mass(ib) / (mass(ib) + mass(ic)) * sap
    scp = mass(ic) / (mass(ib) + mass(ic)) * sap
    r(ia) = sap
    r(ib) = sqrt(rap**2 - cap * sbp + sbp**2)
    r(ic) = sqrt(rap**2 + cap * scp + scp**2)
    call pot0(1, vev, r)
    vev = vev * 27.2114d0
    va = rmlmda * vev
end subroutine potenl
"""
    )
    executable = tmp_path / "abc_potenl_probe"
    subprocess.run(
        ["gfortran", "-std=legacy", "-O0", str(BKMP2_SOURCE), str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def _abc_potenl_values(executable: Path, rho: float, theta: np.ndarray, cos_gamma: np.ndarray, arrangement: int) -> np.ndarray:
    rows = [
        f"{rho * np.cos(theta_value):.17e} {rho * np.sin(theta_value):.17e} {cos_value:.17e} {arrangement}"
        for theta_value in theta
        for cos_value in cos_gamma
    ]
    completed = subprocess.run(
        [str(executable)],
        input="\n".join(rows) + "\n",
        check=True,
        capture_output=True,
        text=True,
    )
    return np.loadtxt(StringIO(completed.stdout))


def _compile_instrumented_abc(tmp_path: Path) -> Path:
    source_text = ABC_SOURCE.read_text()
    before_frames = """         call getrec (y,n,nrg)
         call frames (y,z,cro,jlev,klev,llev,+1)"""
    source_text = source_text.replace(
        before_frames,
        """         call getrec (y,n,nrg)
         write (97,*) n
         do jdump = 1,n
            write (97,*) (y(idump,jdump),idump=1,n)
         enddo
         call frames (y,z,cro,jlev,klev,llev,+1)""",
        1,
    )
    before_probabilities = """         endif
c
c        cumulative reaction probabilities"""
    source_text = source_text.replace(
        before_probabilities,
        """         endif
         write (98,*) n
         do jdump = 1,n
            write (98,*) (x(idump,jdump),y(idump,jdump),idump=1,n)
         enddo
c
c        cumulative reaction probabilities""",
        1,
    )
    assert "write (97,*) n" in source_text
    assert "write (98,*) n" in source_text
    instrumented_source = tmp_path / "abc_bkmp2_instrumented.f"
    instrumented_source.write_text(source_text)
    executable = tmp_path / "abc_bkmp2"
    subprocess.run(
        [
            "gfortran",
            "-std=legacy",
            "-O2",
            "-ffixed-line-length-none",
            str(instrumented_source),
            str(ABC_FUNCTIONS),
            str(ABC_LINEAR_ALGEBRA),
            str(BKMP2_SOURCE),
            str(ABC_ADAPTER),
            "-o",
            str(executable),
            "-framework",
            "Accelerate",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def _run_instrumented_abc(executable: Path, tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    subprocess.run(
        [str(executable)],
        input=ABC_INPUT.read_text(),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    logD_values = np.loadtxt(tmp_path / "fort.97", skiprows=1)
    scattering_values = np.loadtxt(tmp_path / "fort.98", skiprows=1)
    n_channel = logD_values.shape[0]
    scattering = scattering_values.reshape(n_channel, 2 * n_channel)
    return logD_values.reshape(n_channel, n_channel), scattering[:, 0::2] + 1.0j * scattering[:, 1::2]


def _h3_abc_basis(pes):
    abc_mass_to_electron_mass = ABC_EV_PER_HARTREE / ABC_HBAR_SQUARED
    mass = (1.007825 * abc_mass_to_electron_mass,) * 3
    energy_shift = ABC_ENERGY_SHIFT_HARTREE
    diatom_basis = build_delves_diatom_basis(
        asymptotic_potential(pes, mass),
        mass,
        jmax=0,
        E_max=0.6 / ABC_EV_PER_HARTREE - energy_shift,
    )
    basis = build_delves_channels(diatom_basis, Jtot=0, system_parity=1, exchange_parity=1, K_cut=0)
    return basis, 0.45 / ABC_EV_PER_HARTREE - energy_shift


def _pyticc_abc_reference_logD(pes, basis, total_energy: float) -> np.ndarray:
    reduced_mass, _ = mass_scale(basis.mass)
    sector_width = (12.0 - basis.rho_min) / 120
    previous_rho = None
    previous_coefficients = None
    logD = None

    for sector in range(120):
        rho = basis.rho_min + (sector + 0.5) * sector_width
        reference_channels = build_delves_asymptotic_basis(basis, pes, rho)
        reference_coefficients = reference_channels.theta_coefficients
        primitive_H, primitive_S = get_surface_matrices_delves(basis, pes, rho)
        contracted_H = reference_coefficients.T @ primitive_H @ reference_coefficients
        contracted_S = reference_coefficients.T @ primitive_S @ reference_coefficients
        surface_energies, contraction, _ = solve_surface_delves(contracted_H, contracted_S)
        surface_coefficients = reference_coefficients @ contraction

        radial_values = 2.0 * reduced_mass * (surface_energies - total_energy)
        momenta = np.sqrt(np.abs(radial_values))
        arguments = momenta * sector_width
        same_end = np.where(radial_values > 0.0, momenta / np.tanh(arguments), momenta / np.tan(arguments))
        cross_end = np.where(radial_values > 0.0, momenta / np.sinh(arguments), momenta / np.sin(arguments))

        if logD is None:
            logD = np.diag(same_end)
        else:
            assert previous_rho is not None and previous_coefficients is not None
            transform = get_sector_transform_delves(
                basis,
                previous_rho,
                previous_coefficients,
                rho,
                surface_coefficients,
            )
            logD = transform.T @ logD @ transform
            logD = np.diag(same_end) - cross_end[:, None] * np.linalg.solve(logD + np.diag(same_end), np.diag(cross_end))
            logD = 0.5 * (logD + logD.T)

        previous_rho = rho
        previous_coefficients = surface_coefficients

    assert logD is not None and previous_rho is not None and previous_coefficients is not None
    final_channels = build_delves_asymptotic_basis(basis, pes, rho_match=12.0)
    return transform_logD_to_delves_channels(
        basis,
        previous_rho,
        previous_coefficients,
        logD[None, :, :],
        final_channels,
    )[0]


@pytest.mark.skipif(not BKMP2_SOURCE.is_file(), reason="ABC_reac/bkmp2.f is unavailable")
@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_h3_bkmp2_matches_abc_and_pyticc_total_pes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    probe = _compile_abc_probe(tmp_path)
    pes = load_fortran_total_pes([BKMP2_SOURCE], WRAPPER)
    bonds = np.array(
        [
            [1.401, 1.401, 4.0, 1.757, 2.3, 3.2],
            [1.401, 4.0, 4.6, 1.757, 2.8, 3.7],
            [1.401, 4.2, 1.401, 1.757, 3.4, 4.1],
        ],
        dtype=np.float64,
    )

    abc_raw, abc_shifted = _abc_values(probe, bonds)
    pyticc_raw = pes(bonds)

    np.testing.assert_allclose(pyticc_raw, abc_raw, rtol=0.0, atol=5.0e-14)
    np.testing.assert_allclose(abc_shifted, abc_raw + ABC_ENERGY_SHIFT_HARTREE, rtol=0.0, atol=5.0e-14)
    pes.close()


@pytest.mark.skipif(not BKMP2_SOURCE.is_file(), reason="ABC_reac/bkmp2.f is unavailable")
@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_h3_bkmp2_is_permutation_invariant_in_pyticc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    pes = load_fortran_total_pes([BKMP2_SOURCE], WRAPPER)
    geometry = np.array([1.65, 2.25, 3.10], dtype=np.float64)
    permuted = np.column_stack([geometry[list(order)] for order in permutations(range(3))])

    values = pes(permuted)

    np.testing.assert_allclose(values, values[0], rtol=0.0, atol=2.0e-14)
    pes.close()


@pytest.mark.skipif(not BKMP2_SOURCE.is_file(), reason="ABC_reac/bkmp2.f is unavailable")
@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_h3_bkmp2_delves_grid_matches_abc_potenl_scaling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    pes = load_fortran_total_pes([BKMP2_SOURCE], WRAPPER)
    potenl_probe = _compile_abc_potenl_probe(tmp_path)
    mass_amu = np.array([1.007825, 1.007825, 1.007825])
    mass_au = tuple(mass_amu * AMU2AU)
    reduced_mass_amu, _ = mass_scale(mass_amu)
    rho = 7.5
    theta = np.array([0.10, 0.17, 0.24], dtype=np.float64)
    cos_gamma = np.array([-0.65, 0.0, 0.70], dtype=np.float64)

    for arrangement in (1, 2, 3):
        scaled_R = rho * np.cos(theta[:, None])
        scaled_r = rho * np.sin(theta[:, None])
        bonds = delves_bonds(scaled_R, scaled_r, cos_gamma[None, :], arrangement, mass_au).reshape(3, -1)
        pyticc_grid = get_Vgrid_delves(pes, rho, arrangement, theta, cos_gamma, mass_au)
        abc_potenl = _abc_potenl_values(potenl_probe, rho, theta, cos_gamma, arrangement)
        raw_direct = pes(bonds)

        np.testing.assert_allclose(pyticc_grid, raw_direct.reshape(theta.size, cos_gamma.size), rtol=0.0, atol=2.0e-14)
        np.testing.assert_allclose(
            abc_potenl,
            (2.0 * reduced_mass_amu / ABC_HBAR_SQUARED) * ABC_EV_PER_HARTREE * (pyticc_grid.ravel() + ABC_ENERGY_SHIFT_HARTREE),
            rtol=0.0,
            atol=2.0e-11,
        )
    pes.close()


@pytest.mark.skipif(
    not all(path.is_file() for path in (BKMP2_SOURCE, ABC_SOURCE, ABC_FUNCTIONS, ABC_LINEAR_ALGEBRA)),
    reason="ABC_reac sources are unavailable",
)
@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_h3_bkmp2_delves_match_reproduces_abc_complex_Smat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    abc_executable = _compile_instrumented_abc(tmp_path)
    abc_logD, abc_Smat = _run_instrumented_abc(abc_executable, tmp_path)
    pes = load_fortran_total_pes([BKMP2_SOURCE], WRAPPER)
    basis, total_energy = _h3_abc_basis(pes)
    channels = build_delves_asymptotic_basis(basis, pes, rho_match=12.0)

    (pyticc_Smat,) = get_delves_Smat(abc_logD[None, :, :], [total_energy], basis, channels)

    assert channels.qns == ((1, 0, 0, 0), (2, 0, 0, 0))
    np.testing.assert_allclose(pyticc_Smat, abc_Smat, rtol=2.0e-10, atol=2.0e-10)
    np.testing.assert_allclose(pyticc_Smat.conj().T @ pyticc_Smat, np.eye(2), atol=2.0e-14)
    assert abs(pyticc_Smat[0, 1]) ** 2 == pytest.approx(9.84805708e-6, rel=2.0e-8)
    pes.close()


@pytest.mark.skipif(
    not all(path.is_file() for path in (BKMP2_SOURCE, ABC_SOURCE, ABC_FUNCTIONS, ABC_LINEAR_ALGEBRA)),
    reason="ABC_reac sources are unavailable",
)
@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_h3_bkmp2_pyticc_reference_chain_reproduces_abc_logD_and_Smat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    abc_executable = _compile_instrumented_abc(tmp_path)
    abc_logD, abc_Smat = _run_instrumented_abc(abc_executable, tmp_path)
    pes = load_fortran_total_pes([BKMP2_SOURCE], WRAPPER)
    basis, total_energy = _h3_abc_basis(pes)
    pyticc_logD = _pyticc_abc_reference_logD(pes, basis, total_energy)
    channels = build_delves_asymptotic_basis(basis, pes, rho_match=12.0)

    (pyticc_Smat,) = get_delves_Smat(pyticc_logD[None, :, :], [total_energy], basis, channels)

    monomer = prepare_Delves(
        pes,
        basis.mass,
    )
    system = build_ScattSystem(
        monomer,
        Jtot=basis.Jtot,
        system_parity=basis.system_parity,
        jmax=basis.jmax,
        channel=ChannelSpec(exchange_parity_Y=basis.exchange_parity, E_Y_cut=basis.E_max, K_cut=basis.K_cut),
        total_potential=pes,
    )
    assert isinstance(system.basis, DelvesBasis)
    prepared_basis = system.basis
    hamiltonian = build_hamiltonian(system)
    sector_width = (12.0 - prepared_basis.rho_min) / 120
    result = solve(
        hamiltonian,
        [total_energy],
        Propagation((prepared_basis.rho_min, 12.0), (0.5 * sector_width,), device="cpu"),
    )

    np.testing.assert_allclose(pyticc_logD, abc_logD, rtol=2.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(pyticc_Smat, abc_Smat, rtol=2.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(np.abs(pyticc_Smat) ** 2, np.abs(abc_Smat) ** 2, rtol=2.0e-9, atol=2.0e-12)
    np.testing.assert_allclose(result.Y_asymptotic[0], abc_logD, rtol=2.0e-9, atol=2.0e-9)
    np.testing.assert_allclose(result.Smat[0], abc_Smat, rtol=2.0e-9, atol=2.0e-9)
    pes.close()


@pytest.mark.skipif(not BKMP2_SOURCE.is_file(), reason="ABC_reac/bkmp2.f is unavailable")
@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_h3_bkmp2_prepared_channel_propagation_produces_unitary_Smat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    pes = load_fortran_total_pes([BKMP2_SOURCE], WRAPPER)
    basis, total_energy = _h3_abc_basis(pes)
    abc_sector_width = (12.0 - basis.rho_min) / 120

    result = propagate_delves(
        DelvesHamiltonian(basis, pes),
        [total_energy],
        Propagation((basis.rho_min, 12.0), (0.5 * abc_sector_width,), device="cpu"),
    )
    channels, (Smat,) = match_delves(result, [total_energy], basis, pes)

    assert channels.qns == ((1, 0, 0, 0), (2, 0, 0, 0))
    assert result.surface_energies.size == basis.n_channel
    assert result.radial_points.size == 121
    np.testing.assert_allclose(Smat.conj().T @ Smat, np.eye(2), rtol=0.0, atol=2.0e-14)
    pes.close()


@pytest.mark.skipif(not BKMP2_SOURCE.is_file(), reason="ABC_reac/bkmp2.f is unavailable")
@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_h3_bkmp2_common_solve_flow_reproduces_direct_delves_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pyticc as ticc

    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    pes = load_fortran_total_pes([BKMP2_SOURCE], WRAPPER)
    basis, total_energy = _h3_abc_basis(pes)
    abc_sector_width = (12.0 - basis.rho_min) / 120
    hamiltonian = DelvesHamiltonian(basis, pes)
    propagation = Propagation((basis.rho_min, 12.0), (0.5 * abc_sector_width,), device="cpu")

    direct = propagate_delves(hamiltonian, [total_energy], propagation)
    _, direct_Smat = match_delves(direct, [total_energy], basis, pes)
    result = ticc.solve(hamiltonian, [total_energy], propagation)

    assert isinstance(result, ReactiveScatteringResult)
    np.testing.assert_allclose(result.Y_propagated, direct.Y_final, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.Smat[0], direct_Smat[0], rtol=2.0e-13, atol=2.0e-13)
    pes.close()
