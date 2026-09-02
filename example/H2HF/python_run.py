from pathlib import Path

import numpy as np

import pyticc as ticc


def main() -> None:
    pes_dir = Path(__file__).with_name("pes")
    pes = ticc.load_fortran_pes(
        [pes_dir / "pesh2hf.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
    )

    monomer_H2 = pes.monomer_X
    monomer_HF = pes.monomer_Y

    mass_H, mass_F = ticc.element_masses_au("H", "F")
    mass_H2 = 2.0 * mass_H
    mass_HF = mass_H + mass_F
    reduced_mass_H2 = ticc.reduced_mass(mass_H, mass_H)
    reduced_mass_HF = ticc.reduced_mass(mass_H, mass_F)
    reduced_mass_H2HF = ticc.reduced_mass(mass_H2, mass_HF)

    diatom_H2 = ticc.prepare_Diatom(
        monomer_H2,
        r=(0.4, 3.5),
        n_dvr=99,
        n_podvr=5,
        vmax=0,
        jmax=2,
        mass=reduced_mass_H2,
    )

    diatom_HF = ticc.prepare_Diatom(
        monomer_HF,
        r=(0.7, 4.7),
        n_dvr=99,
        n_podvr=3,
        vmax=0,
        jmax=2,
        mass=reduced_mass_HF,
    )
    total_energies = np.array([100.0, 300.0, 500.0]) * ticc.CM2AU

    system = ticc.build_ScattSystem(
        diatom_H2,
        diatom_HF,
        scattering_type="AB+CD",
        Jtot=0,
        system_parity=1,
        channel=ticc.ChannelSpec(
            exchange_parity_X=1,
            E_X_cut=1000.0 * ticc.CM2AU,
            E_Y_cut=1000.0 * ticc.CM2AU,
            K_cut=None,
        ),
        potential=pes,
        reduced_mass=reduced_mass_H2HF,
    )
    potential_grid = ticc.prepare_potential(
        system,
        (4.5, 6.5, 10.5, 20.5),
        (0.10, 0.20, 0.50),
        n_theta_X=10,
        n_theta_Y=10,
        n_phi=10,
        processes=4,
    )
    result = ticc.solve(
        system,
        total_energies,
        potential_grid,
        ticc.Propagation(),
    )
    print(ticc.report.open_closed(result.basis, result.Etot))


if __name__ == "__main__":
    main()
