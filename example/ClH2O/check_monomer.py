from pathlib import Path

import numpy as np

import pyticc as ticc


def h2o_radau_potential(
    total_pes: ticc.TotalPES,
    masses: tuple[float, float, float],
    coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate the H2O monomer PES from Radau coordinates."""
    cartesian = ticc.radau_triatom_cartesian(coordinates, masses)
    bonds = np.stack(
        (
            np.linalg.norm(cartesian[0] - cartesian[1], axis=0),
            np.linalg.norm(cartesian[1] - cartesian[2], axis=0),
            np.linalg.norm(cartesian[2] - cartesian[0], axis=0),
        )
    )
    return total_pes(bonds)


def main() -> None:
    pes_dir = Path(__file__).with_name("pes")
    h2o_total = ticc.load_fortran_total_pes(pes_dir / "h2o_pes.toml")

    mass_h, mass_o, _ = ticc.element_masses_au("H", "O", "Cl")
    masses = (mass_h, mass_o, mass_h)
    equilibrium_values = (1.8, 1.8, float(np.deg2rad(105.0)))

    def monomer_potential(coordinates: np.ndarray) -> np.ndarray:
        return h2o_radau_potential(h2o_total, masses, coordinates)

    basis = ticc.prepare_Triatom(
        potential=monomer_potential,
        r=((1.2, 2.8), (1.2, 2.8)),
        n_dvr=(99, 99),
        n_podvr=(4, 4),
        vmax=(3, 3),
        masses=masses,
        equilibrium=equilibrium_values,
        n_theta=31,
        j1max=25,
        j2max=12,
        tmax=29,
        parity_block_sign=1,
        exchange_parity=-1,
        energy_zero=4609.232 * ticc.CM2AU,
    )
    radial_1 = basis.radial_1
    radial_2 = basis.radial_2
    if radial_1 is None or radial_2 is None:
        raise RuntimeError("Prepared triatomic basis has no radial PODVR data")
    print("PODVR r1 points:", np.round(radial_1.grids, 4))
    print("PODVR r2 points:", np.round(radial_2.grids, 4))
    print(f"ground energy shift: {basis.energy_shift * ticc.AU2CM:.4f} cm^-1")
    print("j2  t  Eint/cm^-1  (PyTICC, zero-based t)")
    for j2 in range(13):
        for t in range(31):
            e = basis.Eint[j2, t]
            if np.isfinite(e):
                print(f"{j2:3d} {t:3d} {e * ticc.AU2CM:12.5f}")


if __name__ == "__main__":
    main()
