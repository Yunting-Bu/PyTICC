from pathlib import Path

import numpy as np

import pyticc as ticc


def main() -> None:
    pes_dir = Path(__file__).with_name("pes")
    pes = ticc.load_fortran_pes(
        [pes_dir / "pesh2hf.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
        processes=4,
    )

    monomer_H2 = pes.monomer_X
    monomer_HF = pes.monomer_Y
    if monomer_H2 is None or monomer_HF is None:
        raise RuntimeError("H2HF PES does not provide both monomer potentials")

    mass_H, mass_F = ticc.element_masses_au("H", "F")
    mass_H2 = 2.0 * mass_H
    mass_HF = mass_H + mass_F
    reduced_mass_H2 = ticc.reduced_mass(mass_H, mass_H)
    reduced_mass_HF = ticc.reduced_mass(mass_H, mass_F)
    reduced_mass_H2HF = ticc.reduced_mass(mass_H2, mass_HF)

    dvr_H2 = ticc.build_SineDVR(
        a=0.4,
        b=3.5,
        n_dvr=99,
        mass=reduced_mass_H2,
        pot_func=monomer_H2,
    )
    rovib_H2 = ticc.build_RovibPODVR(
        dvr=dvr_H2,
        n_podvr=5,
        vmax=0,
        jmax=2,
        mass=reduced_mass_H2,
    )

    dvr_HF = ticc.build_SineDVR(
        a=0.7,
        b=4.7,
        n_dvr=99,
        mass=reduced_mass_HF,
        pot_func=monomer_HF,
    )
    rovib_HF = ticc.build_RovibPODVR(
        dvr=dvr_HF,
        n_podvr=3,
        vmax=0,
        jmax=2,
        mass=reduced_mass_HF,
    )

    Eint_H2 = rovib_H2.E_vj - rovib_H2.E_vj[0, 0]
    Eint_HF = rovib_HF.E_vj - rovib_HF.E_vj[0, 0]
    diatom_H2 = ticc.DiatomSpec(Eint=Eint_H2, vmin=0, vmax=0, jmax=2, jpar=1)
    diatom_HF = ticc.DiatomSpec(Eint=Eint_HF, vmin=0, vmax=0, jmax=2, jpar=0)
    total_energies = np.array([100.0, 300.0, 500.0]) * ticc.CM2AU

    result = ticc.run_diatom_diatom(
        diatom_H2,
        rovib_H2,
        diatom_HF,
        rovib_HF,
        pes,
        Jtot=0,
        system_parity=1,
        Etot=total_energies,
        reduced_mass=reduced_mass_H2HF,
        radial_boundaries=[4.5, 6.5, 10.5, 20.5],
        radial_half_steps=[0.10, 0.20, 0.50],
        trunc=ticc.TruncSpec(
            E_X_cut=1000.0 * ticc.CM2AU,
            E_Y_cut=1000.0 * ticc.CM2AU,
            K_cut=None,
        ),
        n_theta_X=10,
        n_theta_Y=10,
        n_phi=10,
        mode="inelastic",
        approx=ticc.Approx.EXACT,
        memory_limit_mb=512.0,
    )
    result.print_summary()


if __name__ == "__main__":
    main()
