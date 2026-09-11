import numpy as np
from model import h2_potential, interaction, oh_potential

import pyticc as ticc


def main() -> None:
    """Run H2 + OH(2Pi) by explicitly constructing every PyTICC object."""
    mass_H, mass_O = ticc.element_masses_au("H", "O")
    diatom_H2 = ticc.prepare_fs_monomer(
        h2_potential,
        r=(0.8, 3.2),
        n_dvr=24,
        n_podvr=2,
        vmax=0,
        mass=ticc.reduced_mass(mass_H, mass_H),
        two_j_values=(0, 2),
        two_lambda_abs=0,
        two_S=0,
        constants=ticc.FSConstants.from_unit("cm-1", B=60.8),
    )
    diatom_OH = ticc.prepare_fs_monomer(
        oh_potential,
        r=(1.2, 3.6),
        n_dvr=24,
        n_podvr=2,
        vmax=0,
        mass=ticc.reduced_mass(mass_O, mass_H),
        two_j_values=(1,),
        two_lambda_abs=2,
        two_S=1,
        constants=ticc.FSConstants.from_unit("cm-1", A=-139.2, B=18.9),
    )
    pes = ticc.SpinResolvedDiatomDiatomPES(
        interaction=interaction,
        two_total_spins=(1,),
        orbital_states=(ticc.OrbitalState(0, -2), ticc.OrbitalState(0, 2)),
        monomer_X=h2_potential,
        monomer_Y=oh_potential,
    )
    system = ticc.build_ScattSystem(
        diatom_H2,
        diatom_OH,
        two_J=1,
        system_parity=1,
        channel=ticc.ChannelSpec(E_X_cut=300.0 * ticc.CM2AU, E_Y_cut=300.0 * ticc.CM2AU, K_cut=None),
        potential=pes,
        reduced_mass=ticc.reduced_mass(2.0 * mass_H, mass_O + mass_H),
    )
    potential_grid = ticc.prepare_potential(
        system,
        (6.0, 7.0),
        (0.25,),
        n_theta_X=4,
        n_theta_Y=4,
        n_phi=6,
    )
    result = ticc.solve(
        system,
        np.array([30.0, 60.0]) * ticc.CM2AU,
        potential_grid,
        ticc.Propagation(memory_mb=256.0, device="cpu"),
    )
    print(ticc.report.smatrix(result))


if __name__ == "__main__":
    main()
