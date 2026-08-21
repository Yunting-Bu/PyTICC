from pathlib import Path

import numpy as np

import pyticc as ticc
from pyticc.scattering import delves

EV2AU = 1.0 / 27.2114


def main() -> None:
    example_dir = Path(__file__).parent
    pes = ticc.load_fortran_total_pes(
        [example_dir / "bkmp2.f"],
        example_dir / "pyticc_total_wrapper.f90",
        workdir=example_dir,
    )

    try:
        mass_H = ticc.element_mass_au("H")
        mass = (mass_H, mass_H, mass_H)
        energy_cut = 0.6 * EV2AU
        total_energies = np.array([0.45 * EV2AU])

        monomer = ticc.prepare_Delves(
            pes,
            mass,
            energy_zero="minimum",
        )

        system = ticc.build_ScattSystem(
            monomer,
            Jtot=0,
            system_parity=1,
            jmax=0,
            channel=ticc.ChannelSpec(exchange_parity_Y=1, E_Y_cut=energy_cut, K_cut=0),
            total_potential=pes,
        )

        hamiltonian = delves.build_hamiltonian(system)
        basis = hamiltonian.basis

        print("Delves asymptotic diatoms:")
        print(f"native PES energy zero={monomer.energy_zero / EV2AU:.10f} eV")
        print(
            f"rho_min={basis.rho_min:.8f} bohr, "
            f"scaled_r_max={basis.scaled_r_max:.8f} bohr, "
            f"n_sine={basis.n_sine}, "
            f"n_vib_quad={basis.n_vib_quad}, "
            f"n_gamma_quad={basis.n_gamma_quad}, "
            f"n_primitive={basis.n_primitive}, "
            f"n_channel={basis.n_channel}"
        )

        print("\nPrepared channels:")
        for index, (qns, threshold) in enumerate(zip(basis.qns, basis.energies, strict=True)):
            arrangement, v, j, K = qns
            print(f"{index:3d}: a={arrangement}, v={v}, j={j}, K={K}, threshold={threshold / EV2AU: .10f} eV")

        result = ticc.solve(
            hamiltonian,
            total_energies,
            ticc.Propagation(
                boundaries=(hamiltonian.basis.rho_min, 12.0),
                half_steps=((12.0 - hamiltonian.basis.rho_min) / 240.0,),
                device="auto",
            ),
        )

        print(f"\npropagation sectors={result.radial_points.size - 1}")

        print("\nAsymptotic channels:")
        print(ticc.report.channels(result.basis))
        print("\nOpen/closed channels:")
        print(ticc.report.open_closed(result.basis, result.Etot))
        print("\nComplex S matrix:")
        print(ticc.report.smatrix(result, energy_indices=0))
        if result.Smat[0].shape == (2, 2):
            print(f"\n|S(2 <- 1)|^2 = {abs(result.Smat[0][1, 0]) ** 2:.12e}")
    finally:
        pes.close()


if __name__ == "__main__":
    main()
