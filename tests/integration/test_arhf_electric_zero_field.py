from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import pyticc as ticc
import pyticc.pes.fortran.compiler as compiler_module
from pyticc.basis.angle import clebsch_gordan
from pyticc.basis.channel import ChannelBasis, ChannelBasisElectricSF
from pyticc.basis.rovib import RovibBasis
from pyticc.scattering import atom_diatom


def _zero_field_J0_transform(
    basis_bf: ChannelBasis,
    basis_sf: ChannelBasisElectricSF,
    electric_basis: ticc.DiatomElectricBasis,
    rovib: RovibBasis,
) -> NDArray[np.float64]:
    r"""
    Transform the zero-field Electric-SF basis to the regular J=0 TICC basis.

    Formula:
        For J=M=0, l=j and m_l=-m. With the phase conventions used by the
        regular BF basis, the transformation is

        T_{(alpha,m,l,m_l),(v,j)}
          = (-1)^j s_{alpha m}^{vj}
            <j m, j -m | 0 0>
            delta_{l,j} delta_{m_l,-m},

        where s_{alpha m}^{vj} is the sign of the radial overlap between the
        zero-field dressed state and the regular PODVR rovibrational state.

    Inputs:
        basis_bf: ChannelBasis - regular J=0 positive-parity Ar-HF
            channel basis
        basis_sf: ChannelBasisElectricSF - zero-field fixed-M Electric-SF
            channel basis
        electric_basis: ticc.DiatomElectricBasis - zero-field dressed HF states
        rovib: RovibBasis - regular HF rovibrational states

    Returns:
        transform: NDArray[np.float64] - Electric-SF to regular J=0
            transformation, shape (n_sf,n_bf)
    """
    transform = np.zeros((basis_sf.n_channel, basis_bf.n_channel))
    for bf_index, bf_channel in enumerate(basis_bf):
        j = bf_channel.mis_Y.j
        for m in range(-j, j + 1):
            block = electric_basis.block(m)
            j_index = int(np.flatnonzero(block.j_values == j)[0])
            overlaps = rovib.WF_vj[:, 0, j] @ block.coefficients[:, j_index, :]
            alpha = int(np.argmax(np.abs(overlaps)))
            radial_phase = float(np.sign(overlaps[alpha]))
            sf_index = next(
                index for index, channel in enumerate(basis_sf) if channel.alpha == alpha and channel.m == m and channel.l == j and channel.m_l == -m
            )
            transform[sf_index, bf_index] = (-1.0) ** j * radial_phase * clebsch_gordan(j, m, j, -m, 0)
    return transform


@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_ArHF_zero_electric_field_recovers_regular_TICC_Hamiltonian() -> None:
    example_dir = Path(__file__).parents[2] / "example"
    pes_dir = example_dir / "ArHF/pes"
    response_file = example_dir / "ArHF_electric/pes/HF_ele.csv"
    pes = ticc.load_fortran_pes(
        [pes_dir / "interaction-PES.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
    )
    assert pes.monomer_Y is not None

    mass_Ar, mass_H, mass_F = ticc.element_masses_au("Ar", "H", "F")
    mass_HF = ticc.reduced_mass(mass_H, mass_F)
    mass_ArHF = ticc.reduced_mass(mass_Ar, mass_H + mass_F)
    diatom = ticc.prepare_Diatom(
        pes.monomer_Y,
        r=(1.5, 4.5),
        n_dvr=100,
        n_podvr=5,
        vmax=0,
        jmax=1,
        mass=mass_HF,
    )
    electric_basis = ticc.prepare_DiatomElectric(
        pes.monomer_Y,
        response_file,
        r=(1.5, 4.5),
        n_dvr=100,
        electric_strength=0.0,
        n_podvr=5,
        jmax=1,
        M=0,
        lmax=1,
        n_alpha=5,
        mass=mass_HF,
        energy_zero=diatom.energy_zero,
    )

    regular_system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        diatom,
        Jtot=0,
        system_parity=1,
        channel=ticc.ChannelSpec(E_Y_cut=2000.0 * ticc.CM2AU),
        potential=pes,
        reduced_mass=mass_ArHF,
    )
    regular = atom_diatom.build_hamiltonian(
        regular_system,
        n_theta=35,
    )
    electric_system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        electric_basis,
        M=0,
        lmax=1,
        channel=ticc.ChannelSpec(E_Y_cut=2000.0 * ticc.CM2AU),
        potential=pes,
        reduced_mass=mass_ArHF,
    )
    electric = atom_diatom.build_hamiltonian_electric_sf(
        electric_system,
        n_theta_r=24,
        n_theta_R=24,
        n_delta=48,
    )
    assert isinstance(regular.basis, ChannelBasis)
    assert isinstance(electric.basis, ChannelBasisElectricSF)
    transform = _zero_field_J0_transform(regular.basis, electric.basis, electric_basis, diatom.rovib)

    np.testing.assert_allclose(transform.T @ transform, np.eye(regular.basis.n_channel), atol=2.0e-15)
    np.testing.assert_allclose(transform.T @ np.diag(electric.E_int) @ transform, np.diag(regular.E_int), atol=2.0e-15)
    np.testing.assert_allclose(transform.T @ electric.U @ transform, regular.U, atol=2.0e-15)

    radial_points = np.array([5.0, 6.5, 8.0])
    projected_interaction = np.einsum("ia,rij,jb->rab", transform, electric.V(radial_points), transform, optimize=True)
    np.testing.assert_allclose(projected_interaction, regular.V(radial_points), rtol=0.0, atol=2.0e-8)

    energy = np.array([300.0 * ticc.CM2AU])
    propagation = ticc.Propagation(
        boundaries=(4.5, 6.5, 8.0, 12.0),
        half_steps=(0.05, 0.05, 0.05),
    )
    regular_result = ticc.solve(regular, energy, propagation)
    electric_result = ticc.solve(electric, energy, propagation)
    assert isinstance(regular_result, ticc.ScatteringResult)
    assert isinstance(electric_result, ticc.ScatteringResult)

    electric_open = electric_result.open_channel_indices[0]
    asymptotic_transform = transform @ regular_result.asymptotic_transform
    transform_open = asymptotic_transform[electric_open]
    projected_Smat = transform_open.T @ electric_result.Smat[0] @ transform_open
    np.testing.assert_allclose(projected_Smat, regular_result.Smat[0], rtol=0.0, atol=2.0e-5)
