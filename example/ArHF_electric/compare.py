from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import pyticc as ticc
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
    Transform the zero-field Electric-SF basis to the regular J=0 BF basis.

    Formula:
        For J=M=0,

        l=j,    m_l=-m,

        and the transformation consistent with the PyTICC BF phase convention
        is

        T_{(alpha,m,l,m_l),(v,j)}
          = (-1)^j s_{alpha m}^{vj}
            <j m,j -m|0 0>
            delta_{l,j} delta_{m_l,-m},

        where

        s_{alpha m}^{vj}
          = sign[<vj|phi_{alpha m}(E=0)>]

        removes the arbitrary real-eigenvector sign.

    Inputs:
        basis_bf: ChannelBasis - regular J=0 positive-parity BF channels
        basis_sf: ChannelBasisElectricSF - zero-field M=0 Electric-SF
            channels
        electric_basis: ticc.DiatomElectricBasis - zero-field dressed HF basis
        rovib: RovibBasis - regular zero-field HF rovibrational basis

    Returns:
        transform: NDArray[np.float64] - Electric-SF to regular BF transform,
            shape (n_sf,n_bf)
    """
    transform = np.zeros((basis_sf.n_channel, basis_bf.n_channel))
    for bf_index, bf_channel in enumerate(basis_bf):
        if bf_channel.mis_Y.v is None:
            raise ValueError("The regular diatomic channel does not have a vibrational quantum number")
        v = bf_channel.mis_Y.v
        j = bf_channel.mis_Y.j
        for m in range(-j, j + 1):
            block = electric_basis.block(m)
            j_index = int(np.flatnonzero(block.j_values == j)[0])
            overlaps = rovib.WF_vj[:, v, j] @ block.coefficients[:, j_index, :]
            alpha = int(np.argmax(np.abs(overlaps)))
            radial_phase = float(np.sign(overlaps[alpha]))
            sf_index = next(
                index for index, channel in enumerate(basis_sf) if channel.alpha == alpha and channel.m == m and channel.l == j and channel.m_l == -m
            )
            transform[sf_index, bf_index] = (-1.0) ** j * radial_phase * clebsch_gordan(j, m, j, -m, 0)
    return transform


def _max_abs(values: NDArray) -> float:
    """Return the largest absolute array element."""
    return float(np.max(np.abs(values)))


def _format_complex(value: complex) -> str:
    """Format one S-matrix element as a compact complex number."""
    return f"{value.real:+.8f}{value.imag:+.8f}i"


def main() -> None:
    r"""
    Compare only the field-free and zero-field Electric-TICC S matrices.

    Formula:
        If T maps the selected Electric-SF subspace to the regular J=0 BF basis
        and B is the regular BF-to-asymptotic transformation, define Q=T B.
        The open-channel comparison is

        S_projected = Q_open.T S_electric Q_open.
    """
    example_dir = Path(__file__).parent
    pes_dir = example_dir.parent / "ArHF/pes"
    response_file = example_dir / "pes/HF_ele.csv"
    pes = ticc.load_fortran_pes(
        [pes_dir / "interaction-PES.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
        processes=4,
    )

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

    regular = atom_diatom.build_hamiltonian(
        ticc.build_ScattSystem(
            ticc.AtomSpec(),
            diatom,
            Jtot=0,
            system_parity=1,
            channel=ticc.ChannelSpec(E_Y_cut=2000.0 * ticc.CM2AU),
            potential=pes,
            reduced_mass=mass_ArHF,
        ),
        n_theta=35,
    )
    electric = atom_diatom.build_hamiltonian_electric_sf(
        ticc.build_ScattSystem(
            ticc.AtomSpec(),
            electric_basis,
            M=0,
            lmax=1,
            channel=ticc.ChannelSpec(E_Y_cut=2000.0 * ticc.CM2AU),
            potential=pes,
            reduced_mass=mass_ArHF,
        ),
        n_theta_r=24,
        n_theta_R=24,
        n_delta=48,
    )
    if not isinstance(regular.basis, ChannelBasis) or not isinstance(electric.basis, ChannelBasisElectricSF):
        raise TypeError("Unexpected channel-basis representation")

    transform = _zero_field_J0_transform(regular.basis, electric.basis, electric_basis, diatom.rovib)
    propagation = ticc.Propagation(
        boundaries=(4.5, 6.5, 8.0, 12.0),
        half_steps=(0.05, 0.05, 0.05),
    )
    energy = np.array([300.0 * ticc.CM2AU])
    regular_result = ticc.solve(regular, energy, propagation)
    electric_result = ticc.solve(electric, energy, propagation)
    if not isinstance(regular_result, ticc.ScatteringResult) or not isinstance(electric_result, ticc.ScatteringResult):
        raise TypeError("The exact comparison requires ScatteringResult objects")

    electric_open = electric_result.open_channel_indices[0]
    transform_open = (transform @ regular_result.asymptotic_transform)[electric_open]
    projected_Smat = transform_open.T @ electric_result.Smat[0] @ transform_open
    Smat_error = _max_abs(projected_Smat - regular_result.Smat[0])
    tolerance = 2.0e-5
    regular_open = regular_result.open_channel_indices[0]
    labels = tuple(
        f"(v={regular.basis[int(index)].mis_Y.v},j={regular.basis[int(index)].mis_Y.j},l={int(round(regular_result.L[int(index)]))})"
        for index in regular_open
    )

    print("\nAr-HF S-matrix comparison")
    print("Field-free TICC: J=0, p=+1")
    print("Electric-TICC:   E=0, M=0, projected onto J=0, p=+1")
    print(f"Collision energy: {energy[0] * ticc.AU2CM:.8f} cm^-1")
    print()
    print(f"{'S(out <- in)':<35} {'field-free TICC':>26} {'E=0 Electric-TICC':>26} {'|Delta S|':>14}")
    print("-" * 106)
    for incoming, incoming_label in enumerate(labels):
        for outgoing, outgoing_label in enumerate(labels):
            regular_value = regular_result.Smat[0][outgoing, incoming]
            electric_value = projected_Smat[outgoing, incoming]
            difference = abs(electric_value - regular_value)
            transition = f"{outgoing_label} <- {incoming_label}"
            print(f"{transition:<35} {_format_complex(regular_value):>26} {_format_complex(electric_value):>26} {difference:14.6E}")

    status = "PASS" if Smat_error <= tolerance else "FAIL"
    print(f"\nMaximum |Delta S| : {Smat_error:.8E}")
    print(f"Tolerance         : {tolerance:.8E}")
    print(f"Result            : {status}")
    if status == "FAIL":
        raise RuntimeError("Zero-field Electric-TICC S matrix does not reproduce field-free J=0 TICC")


if __name__ == "__main__":
    main()
