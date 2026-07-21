from pathlib import Path

import numpy as np

import pyticc as ticc
from pyticc.scattering import diatom_diatom


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

    diatom_H2 = ticc.DiatomBasis(
        rovib=rovib_H2,
        energy_zero=float(rovib_H2.E_vj[0, 0]),
        vmin=0,
        vmax=0,
        jmax=2,
        jpar=1,
    )
    diatom_HF = ticc.DiatomBasis(
        rovib=rovib_HF,
        energy_zero=float(rovib_HF.E_vj[0, 0]),
        vmin=0,
        vmax=0,
        jmax=2,
        jpar=0,
    )
    total_energies = np.array([100.0, 300.0, 500.0]) * ticc.CM2AU

    system = ticc.ScattSystem(
        diatom_H2,
        diatom_HF,
        Jtot=0,
        system_parity=1,
        potential=pes,
        reduced_mass=reduced_mass_H2HF,
    )
    hamiltonian = diatom_diatom.build_hamiltonian(
        system,
        trunc=ticc.TruncSpec(
            E_X_cut=1000.0 * ticc.CM2AU,
            E_Y_cut=1000.0 * ticc.CM2AU,
            K_cut=None,
        ),
        n_theta_X=10,
        n_theta_Y=10,
        n_phi=10,
    )
    result = ticc.solve(
        hamiltonian,
        total_energies,
        ticc.Propagation(
            boundaries=(4.5, 6.5, 10.5, 20.5),
            half_steps=(0.10, 0.20, 0.50),
        ),
    )
    print(ticc.report.open_closed(result.basis, result.Etot))


if __name__ == "__main__":
    main()
