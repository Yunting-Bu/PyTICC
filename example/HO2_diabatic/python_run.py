from pathlib import Path

import numpy as np

import pyticc as ticc


def main() -> None:
    pes_dir = Path(__file__).with_name("pes")
    pes = ticc.load_fortran_diabatic_pes(
        [pes_dir / "ho2-dpme.f", pes_dir / "long_range_H_O2.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
        lapack=True,
    )

    mass_O, mass_H = ticc.element_masses_au("O", "H")
    mass_O2 = 2.0 * mass_O
    reduced_mass_O2 = ticc.reduced_mass(mass_O, mass_O)
    reduced_mass_HO2 = ticc.reduced_mass(mass_H, mass_O2)

    diatom_O2 = ticc.prepare_DiabaticDiatom(
        pes.monomer_values,
        n_state=pes.n_state,
        r=(1.2, 5.0),
        n_dvr=135,
        n_podvr=80,
        vmax=(29, 26),
        jmax=(55, 56),
        mass=reduced_mass_O2,
    )
    total_energies = np.array([17.38631, 18.18561, 18.46362, 18.77639]) * ticc.CM2AU

    try:
        system = ticc.build_ScattSystem(
            ticc.AtomSpec(),
            diatom_O2,
            scattering_type="A+BC_diabatic",
            Jtot=0,
            system_parity=1,
            channel=ticc.ChannelSpec(
                exchange_parity_Y=(-1, 1),
                E_Y_cut=38000.0 * ticc.CM2AU,
            ),
            potential=pes,
            reduced_mass=reduced_mass_HO2,
        )
        potential_grid = ticc.prepare_potential(
            system,
            (0.8, 2.5, 6.0, 30.0),
            (0.002, 0.005, 0.3),
            n_theta=30,
            processes=4,
        )
        result = ticc.solve(
            system,
            total_energies,
            potential_grid,
            ticc.Propagation(
                memory_mb=4096.0,
                device="auto",
            ),
        )
        print(ticc.report.open_closed(result.basis, result.Etot))
        print(ticc.report.smatrix(result))
    finally:
        pes.close()


if __name__ == "__main__":
    main()
