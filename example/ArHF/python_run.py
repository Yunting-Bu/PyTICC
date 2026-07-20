from pathlib import Path

import numpy as np

import pyticc as ticc


def main() -> None:
    pes_dir = Path(__file__).with_name("pes")
    pes = ticc.load_fortran_pes(
        [pes_dir / "interaction-PES.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
        processes=4,
    )

    monomer_HF = pes.monomer_Y
    if monomer_HF is None:
        raise RuntimeError("ArHF PES does not provide the HF monomer potential")

    mass_Ar, mass_H, mass_F = ticc.element_masses_au("Ar", "H", "F")
    mass_HF = mass_H + mass_F
    reduced_mass_HF = ticc.reduced_mass(mass_H, mass_F)
    reduced_mass_ArHF = ticc.reduced_mass(mass_Ar, mass_HF)

    dvr_HF = ticc.build_SineDVR(
        a=1.5,
        b=4.5,
        n_dvr=100,
        mass=reduced_mass_HF,
        pot_func=monomer_HF,
    )
    rovib_HF = ticc.build_RovibPODVR(
        dvr=dvr_HF,
        n_podvr=5,
        vmax=0,
        jmax=4,
        mass=reduced_mass_HF,
    )

    Eint_HF = rovib_HF.E_vj - rovib_HF.E_vj[0, 0]
    diatom_HF = ticc.DiatomSpec(Eint=Eint_HF, vmax=0, jmax=4)
    total_energies = np.array([100.0, 300.0, 500.0]) * ticc.CM2AU

    result = ticc.run_atom_diatom(
        diatom_HF,
        rovib_HF,
        pes,
        Jtot=0,
        system_parity=1,
        Etot=total_energies,
        reduced_mass=reduced_mass_ArHF,
        radial_boundaries=[4.5, 6.5, 8.0, 12.0],
        radial_half_steps=[0.05, 0.08, 0.10],
        trunc=ticc.TruncSpec(E_Y_cut=2000.0 * ticc.CM2AU, K_cut=None),
        n_theta=35,
        mode="inelastic",
        approx=ticc.Approx.EXACT,
        memory_limit_mb=512.0,
    )
    result.print_summary()


if __name__ == "__main__":
    main()
