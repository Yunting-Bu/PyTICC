import numpy as np
from pes import pes

import pyticc as ticc
from pyticc.constants import AMU2AU


def main() -> None:
    # Ca(3P), [Ar]4s5p, L=1, S=1, J=0,1,2, parity=-1
    monomer_X = ticc.build_fs_atom_basis(
        two_L=2,
        two_S=2,
        atom_parity=-1,
        two_j_levels=(0, 2, 4),
        level_energies=(0.0, 7.076, 27.44),
        unit="cm-1",
    )
    monomer_Y = ticc.build_fs_atom_basis(
        two_L=0,
        two_S=0,
        atom_parity=1,
        two_j_levels=(0,),
        level_energies=(0.0,),
        unit="cm-1",
    )
    total_energies = np.array([113.04080295]) * ticc.CM2AU
    system = ticc.build_ScattSystem(
        monomer_X,
        monomer_Y,
        two_J=2,
        system_parity=-1,
        approx=ticc.Approx.EXACT,
        potential=pes,
        reduced_mass=3.638205 * AMU2AU,
    )
    potential_grid = ticc.prepare_potential(
        system,
        boundaries=(4.0, 25.0),
        half_steps=(0.025,),
    )
    result = ticc.solve(system, total_energies, potential_grid, ticc.Propagation(mode="inelastic"))
    print(ticc.report.channels(result.basis))
    print(ticc.report.smatrix(result))


if __name__ == "__main__":
    main()
