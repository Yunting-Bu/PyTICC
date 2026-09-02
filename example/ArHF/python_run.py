from pathlib import Path

import numpy as np

import pyticc as ticc


def main() -> None:
    pes_dir = Path(__file__).with_name("pes")
    pes = ticc.load_fortran_pes(
        [pes_dir / "interaction-PES.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
    )

    monomer_HF = pes.monomer_Y

    mass_Ar, mass_H, mass_F = ticc.element_masses_au("Ar", "H", "F")
    mass_HF = mass_H + mass_F
    reduced_mass_HF = ticc.reduced_mass(mass_H, mass_F)
    reduced_mass_ArHF = ticc.reduced_mass(mass_Ar, mass_HF)

    diatom_HF = ticc.prepare_Diatom(
        monomer_HF,
        r=(1.5, 4.5),
        n_dvr=100,
        n_podvr=5,
        vmax=0,
        jmax=4,
        mass=reduced_mass_HF,
    )
    total_energies = np.array([100.0, 300.0, 500.0]) * ticc.CM2AU

    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        diatom_HF,
        scattering_type="A+BC",
        Jtot=0,
        system_parity=1,
        channel=ticc.ChannelSpec(E_Y_cut=2000.0 * ticc.CM2AU, K_cut=None),
        potential=pes,
        reduced_mass=reduced_mass_ArHF,
    )
    potential_grid = ticc.prepare_potential(
        system,
        (4.5, 6.5, 8.0, 12.0),
        (0.05, 0.08, 0.10),
        n_theta=35,
        processes=4,
    )
    result = ticc.solve(
        system,
        total_energies,
        potential_grid,
        ticc.Propagation(),
    )
    print("\nChannels:")
    print(ticc.report.channels(result.basis))
    print("\nOpen/closed channels: ")
    print(ticc.report.open_closed(result.basis, result.Etot))
    print("\nEvj: ")
    print(ticc.report.rovib_levels(diatom_HF))
    print("\nS-matrix: ")
    print(ticc.report.smatrix(result))


if __name__ == "__main__":
    main()
