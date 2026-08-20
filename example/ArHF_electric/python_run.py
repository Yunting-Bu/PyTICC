from pathlib import Path

import numpy as np

import pyticc as ticc
from pyticc.scattering import atom_diatom

ELECTRIC_STRENGTH = 1.0e-3
M = 0
LMAX = 1


def main() -> None:
    """Run one exact Electric-SF Ar-HF scattering calculation."""
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
    electric_HF = ticc.prepare_DiatomElectric(
        pes.monomer_Y,
        response_file,
        r=(1.5, 4.5),
        n_dvr=100,
        electric_strength=ELECTRIC_STRENGTH,
        n_podvr=5,
        jmax=8,
        M=M,
        lmax=LMAX,
        n_alpha=3,
        mass=mass_HF,
    )

    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        electric_HF,
        M=M,
        lmax=LMAX,
        channel=ticc.ChannelSpec(E_Y_cut=2000.0 * ticc.CM2AU),
        potential=pes,
        reduced_mass=mass_ArHF,
    )
    hamiltonian = atom_diatom.build_hamiltonian_electric_sf(
        system,
        n_theta_r=16,
        n_theta_R=16,
        n_delta=16,
    )
    result = ticc.solve(
        hamiltonian,
        np.array([300.0]) * ticc.CM2AU,
        ticc.Propagation(
            boundaries=(4.5, 6.5, 8.0, 12.0),
            half_steps=(0.05, 0.08, 0.10),
        ),
    )

    print("\nElectric-SF channels:")
    print(ticc.report.channels(result.basis))
    print("\nOpen/closed channels:")
    print(ticc.report.open_closed(result.basis, result.Etot))
    print("\nS-matrix:")
    print(ticc.report.smatrix(result))


if __name__ == "__main__":
    main()
