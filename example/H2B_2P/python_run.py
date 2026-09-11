import numpy as np
from model import h2_potential, interaction

import pyticc as ticc


def main() -> None:
    mass_H, mass_B = ticc.element_masses_au("H", "B")
    atom_B = ticc.build_fs_atom_basis(
        two_L=2,
        two_S=1,
        atom_parity=-1,
        two_j_levels=(1, 3),
        level_energies=(0.0, 15.3),
        unit="cm-1",
    )
    diatom_H2 = ticc.prepare_fs_monomer(
        h2_potential,
        r=(0.8, 3.2),
        n_dvr=30,
        n_podvr=2,
        vmax=0,
        mass=ticc.reduced_mass(mass_H, mass_H),
        two_j_values=(0, 2),
        two_lambda_abs=0,
        two_S=0,
        constants=ticc.FSConstants.from_unit("cm-1", B=60.8),
    )
    pes = ticc.SpinResolvedAtomDiatomPES(
        interaction=interaction,
        two_total_spins=(1,),
        orbital_states=ticc.atom_diatom_orbital_states(2, 0),
        monomer_Y=h2_potential,
    )
    system = ticc.build_ScattSystem(
        atom_B,
        diatom_H2,
        two_J=1,
        system_parity=1,
        channel=ticc.ChannelSpec(E_X_cut=100.0 * ticc.CM2AU, E_Y_cut=300.0 * ticc.CM2AU, K_cut=None),
        potential=pes,
        reduced_mass=ticc.reduced_mass(mass_B, 2.0 * mass_H),
    )
    potential_grid = ticc.prepare_potential(
        system,
        (5.0, 6.0),
        (0.25,),
        n_theta=8,
    )
    result = ticc.solve(
        system,
        np.array([20.0, 40.0]) * ticc.CM2AU,
        potential_grid,
        ticc.Propagation(memory_mb=256.0, device="cpu"),
    )
    print(ticc.report.smatrix(result))


if __name__ == "__main__":
    main()
