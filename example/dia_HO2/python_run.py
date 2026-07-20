from pathlib import Path

import numpy as np

import pyticc as ticc


def main() -> None:
    pes_dir = Path(__file__).with_name("pes")
    pes = ticc.load_fortran_diabatic_pes(
        [pes_dir / "ho2-dpme.f", pes_dir / "long_range_H_O2.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
        processes=4,
    )

    mass_O, mass_H = ticc.element_masses_au("O", "H")
    mass_O2 = 2.0 * mass_O
    reduced_mass_O2 = ticc.reduced_mass(mass_O, mass_O)
    reduced_mass_HO2 = ticc.reduced_mass(mass_H, mass_O2)

    dvrs = tuple(
        ticc.build_SineDVR(
            a=1.2,
            b=5.0,
            n_dvr=135,
            mass=reduced_mass_O2,
            pot_func=pes.monomer_state(electronic_state),
        )
        for electronic_state in range(pes.n_state)
    )
    diatom_O2 = ticc.build_DiabaticDiatomBasis(
        dvrs,
        n_podvr=80,
        vmax=(29, 26),
        jmax=(55, 56),
        mass=reduced_mass_O2,
        jpar=(-1, 1),
    )
    total_energies = np.array([17.38631, 18.18561, 18.46362, 18.77639]) * ticc.CM2AU

    try:
        result = ticc.run_diabatic_atom_diatom(
            diatom_O2,
            pes,
            Jtot=0,
            system_parity=1,
            Etot=total_energies,
            reduced_mass=reduced_mass_HO2,
            radial_boundaries=[0.8, 2.5, 6.0, 30.0],
            radial_half_steps=[0.002, 0.005, 0.3],
            trunc=ticc.TruncSpec(E_Y_cut=38000.0 * ticc.CM2AU),
            n_theta=30,
            mode="inelastic",
            memory_limit_mb=4096.0,
        )
        result.print_summary()
    finally:
        pes.close()


if __name__ == "__main__":
    main()
