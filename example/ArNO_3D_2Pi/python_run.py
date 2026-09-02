from pathlib import Path

import numpy as np

import pyticc as ticc


def main() -> None:
    """Calculate one Ar + NO(X 2Pi) block on the 3D Teplukhin-Kendrick surface."""
    root = Path(__file__).parent
    pes = ticc.load_fortran_lambda_pes(root / "pes" / "pes.toml")
    if pes.monomer_Y is None:
        raise RuntimeError("Ar--NO example PES does not provide the isolated NO potential")

    mass_ar, mass_n, mass_o = ticc.element_masses_au("Ar", "N", "O")
    try:
        monomer = ticc.prepare_fs_monomer(
            pes.monomer_Y,
            r=(1.7, 3.2),
            n_dvr=80,
            n_podvr=5,
            vmax=0,
            mass=ticc.reduced_mass(mass_n, mass_o),
            two_j_values=(1, 3, 5),
            two_lambda_abs=2,
            two_S=1,
            constants=root / "constant_2Pi_NO.csv",
        )
        print(ticc.report.fine_structure_levels(monomer))

        system = ticc.build_ScattSystem(
            ticc.AtomSpec(),
            monomer,
            scattering_type="A+BC_fine_structure",
            two_J=1,
            system_parity=1,
            potential=pes,
            reduced_mass=ticc.reduced_mass(mass_ar, mass_n + mass_o),
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
