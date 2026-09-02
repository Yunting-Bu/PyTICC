from pathlib import Path

import numpy as np

import pyticc as ticc

ELECTRIC_STRENGTH = 0.0
M = 0
LMAX = 5


def main() -> None:
    """Run one exact Electric-SF Ar-HF scattering calculation."""
    example_dir = Path(__file__).parent
    pes_dir = example_dir.parent / "ArHF/pes"
    response_file = example_dir / "pes/HF_ele.csv"
    pes = ticc.load_fortran_pes(
        [pes_dir / "interaction-PES.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
    )

    mass_Ar, mass_H, mass_F = ticc.element_masses_au("Ar", "H", "F")
    mass_HF = ticc.reduced_mass(mass_H, mass_F)
    mass_ArHF = ticc.reduced_mass(mass_Ar, mass_H + mass_F)
    electric_HF = ticc.prepare_DiatomElectric(
        pes.monomer_Y,
        response_file,
        r=(0.75, 6.55),
        n_dvr=50,
        electric_strength=ELECTRIC_STRENGTH,
        n_podvr=10,
        jmax=5,
        M=M,
        lmax=LMAX,
        n_alpha=2,
        mass=mass_HF,
    )

    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        electric_HF,
        scattering_type="A+BC_electric",
        M=M,
        lmax=LMAX,
        channel=ticc.ChannelSpec(E_Y_cut=2000000.0 * ticc.CM2AU),
        potential=pes,
        reduced_mass=mass_ArHF,
    )
    potential_grid = ticc.prepare_potential(
        system,
        (2.0, 20.0, 50.0, 100.0),
        (0.01, 0.1, 1.0),
        n_theta_r=20,
        n_theta_R=20,
        n_delta=40,
        processes=4,
    )
    result = ticc.solve(
        system,
        np.array([100.0]) * ticc.CM2AU,
        potential_grid,
        ticc.Propagation(memory_mb=1024.0),
    )

    print("\nElectric-SF channels:")
    print(ticc.report.channels(result.basis))
    print("\nOpen/closed channels:")
    print(ticc.report.open_closed(result.basis, result.Etot))
    print("\nS-matrix:")
    print(ticc.report.smatrix(result))


if __name__ == "__main__":
    main()
