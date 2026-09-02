from pathlib import Path

import numpy as np

import pyticc as ticc


def main() -> None:
    """Calculate one Ar + CH(A 2Delta) total-angular-momentum block."""
    root = Path(__file__).parent
    pes = ticc.load_fortran_lambda_pes(root / "pes" / "pes.toml")
    if pes.monomer_Y is None:
        raise RuntimeError("Ar--CH example PES does not provide the isolated CH potential")

    mass_ar, mass_c, mass_h = ticc.element_masses_au("Ar", "C", "H")
    try:
        monomer = ticc.prepare_fs_monomer(
            pes.monomer_Y,
            r=(1.6, 3.2),
            n_dvr=80,
            n_podvr=5,
            vmax=0,
            mass=ticc.reduced_mass(mass_c, mass_h),
            two_j_values=(3, 5, 7),
            two_lambda_abs=4,
            two_S=1,
            constants=root / "constant_2Delta_CH.csv",
        )
        print(ticc.report.fine_structure_levels(monomer))

        system = ticc.build_ScattSystem(
            ticc.AtomSpec(),
            monomer,
            scattering_type="A+BC_fine_structure",
            two_J=3,
            system_parity=1,
            potential=pes,
            reduced_mass=ticc.reduced_mass(mass_ar, mass_c + mass_h),
        )
        potential_grid = ticc.prepare_potential(
            system,
            (4.0, 7.0, 12.0, 40.0),
            (0.05, 0.10, 0.25),
            n_theta=20,
        )
        result = ticc.solve(
            system,
            np.arange(10.0, 30.1, 4.0) * ticc.CM2AU,
            potential_grid,
            ticc.Propagation(),
        )
        print(ticc.report.smatrix(result))
    finally:
        pes.close()


if __name__ == "__main__":
    main()
