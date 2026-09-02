from pathlib import Path

import numpy as np

import pyticc as ticc


def main() -> None:
    pes_dir = Path(__file__).with_name("pes")
    pes = ticc.load_fortran_pes(
        [pes_dir / "pes_interface.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
    )

    try:
        mass_Rb, mass_K = ticc.element_masses_au("Rb", "K")
        mass_KRb = mass_Rb + mass_K
        diatom_KRb = ticc.prepare_Diatom(
            pes.monomer_X,
            r=(4.8, 13.2),
            n_dvr=99,
            n_podvr=3,
            vmax=0,
            jmax=3,
            mass=ticc.reduced_mass(mass_Rb, mass_K),
        )
        system = ticc.build_ScattSystem(
            diatom_KRb,
            diatom_KRb,
            scattering_type="AB+CD",
            Jtot=2,
            system_parity=1,
            approx=ticc.Approx.NNCC,
            K_delta=1,
            channel=ticc.ChannelSpec(
                vmin_X=0,
                vmin_Y=0,
                exchange_parity_X=0,
                exchange_parity_Y=0,
                E_X_cut=100.0 * ticc.CM2AU,
                E_Y_cut=100.0 * ticc.CM2AU,
                K_cut=None,
            ),
            potential=pes,
            reduced_mass=ticc.reduced_mass(mass_KRb, mass_KRb),
        )
        potential_grid = ticc.prepare_potential(
            system,
            (20.0, 25.0, 60.0, 100.0),
            (0.05, 0.5, 1.0),
            n_theta_X=21,
            n_theta_Y=15,
            n_phi=21,
            processes=4,
        )
        result = ticc.solve(
            system,
            np.array([1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0]) * ticc.CM2AU,
            potential_grid,
            ticc.Propagation(
                mode="capture",
                memory_mb=512.0,
                device="auto",
                print_verbose=False,
            ),
        )
        print(ticc.report.open_closed(result.basis, result.Etot))
        if isinstance(result, ticc.CoupledStatesResult):
            print(ticc.report.k_blocks(result.blocks))
    finally:
        pes.close()


if __name__ == "__main__":
    main()
