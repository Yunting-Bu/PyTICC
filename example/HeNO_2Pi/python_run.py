from pathlib import Path

import numpy as np

import pyticc as ticc


def main() -> None:
    """Calculate one He + NO(X 2Pi) total-angular-momentum block."""
    root = Path(__file__).parent
    pes = ticc.load_fortran_lambda_pes(root / "pes" / "pes.toml")
    if pes.monomer_Y is None:
        raise RuntimeError("He--NO example PES does not provide the isolated NO potential")

    mass_he, mass_n, mass_o = ticc.element_masses_au("He", "N", "O")
    try:
        monomer = ticc.prepare_fs_monomer(
            pes.monomer_Y,
            r=(1.6, 3.2),
            n_dvr=80,
            n_podvr=5,
            vmax=0,
            mass=ticc.reduced_mass(mass_n, mass_o),
            two_j_values=(1, 3),
            two_lambda_abs=2,
            two_S=1,
            constants=root / "constant_2Pi_NO.csv",
        )
        print(ticc.report.fine_structure_levels(monomer))

        system = ticc.build_ScattSystem(
            ticc.AtomSpec(),
            monomer,
            two_J=1,
            system_parity=1,
            potential=pes,
            reduced_mass=ticc.reduced_mass(mass_he, mass_n + mass_o),
        )
        hamiltonian = ticc.build_fs_hamiltonian(
            system,
            n_theta=20,
        )
        result = ticc.solve(
            hamiltonian,
            np.arange(10.0, 20.1, 2.0) * ticc.CM2AU,
            ticc.Propagation((4.0, 7.0, 12.0, 40.0), (0.05, 0.10, 0.25)),
        )
        print(ticc.report.smatrix(result))
    finally:
        pes.close()


if __name__ == "__main__":
    main()
