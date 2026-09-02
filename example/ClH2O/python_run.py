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
    """Run a reduced end-to-end Cl + H2O scattering calculation."""
    pes_dir = Path(__file__).with_name("pes")
    pes = ticc.load_fortran_pes(pes_dir / "pes.toml")
    h2o_total = ticc.load_fortran_total_pes(pes_dir / "h2o_pes.toml")

    equilibrium = np.array([[1.8], [1.8], [np.deg2rad(105.0)]])
    orientations = np.array(
        [
            [equilibrium[0, 0], equilibrium[0, 0]],
            [equilibrium[1, 0], equilibrium[1, 0]],
            [equilibrium[2, 0], equilibrium[2, 0]],
            [0.0, 0.5 * np.pi],
            [0.0, 0.0],
        ]
    )
    values = pes.interaction(6.5, orientations) / ticc.CM2AU

    print("Cl + H2O corrected-Radau interaction energies at R=6.5 bohr:")
    print(f"  theta2=0 deg:  {values[0]: .8f} cm^-1")
    print(f"  theta2=90 deg: {values[1]: .8f} cm^-1")

    mass_h, mass_o, mass_cl = ticc.element_masses_au("H", "O", "Cl")
    masses = (mass_h, mass_o, mass_h)
    equilibrium_values = (
        float(equilibrium[0, 0]),
        float(equilibrium[1, 0]),
        float(equilibrium[2, 0]),
    )

    def monomer_potential(coordinates: np.ndarray) -> np.ndarray:
        return h2o_radau_potential(h2o_total, masses, coordinates)

    triatom = ticc.prepare_Triatom(
        potential=monomer_potential,
        r=((1.2, 2.8), (1.2, 2.8)),
        n_dvr=(24, 24),
        n_podvr=(2, 2),
        vmax=(0, 0),
        masses=masses,
        equilibrium=equilibrium_values,
        n_theta=8,
        j1max=2,
        j2max=1,
        tmax=2,
        parity_block_sign=1,
        exchange_parity=1,
    )
    reduced_mass = ticc.reduced_mass(mass_cl, 2.0 * mass_h + mass_o)
    system = ticc.build_ScattSystem(
        ticc.AtomSpec(),
        triatom,
        scattering_type="A+BCD",
        Jtot=0,
        system_parity=1,
        channel=ticc.ChannelSpec(E_Y_cut=500.0 * ticc.CM2AU),
        potential=pes,
        reduced_mass=reduced_mass,
    )
    potential_grid = ticc.prepare_potential(
        system,
        (5.0, 7.0, 10.0),
        (0.25, 0.5),
        n_theta_2=5,
        n_phi=6,
    )
    result = ticc.solve(
        system,
        np.array([300.0]) * ticc.CM2AU,
        potential_grid,
        ticc.Propagation(device="cpu"),
    )
    if not isinstance(result, ticc.ScatteringResult):
        raise RuntimeError("Exact Cl + H2O calculation did not return a scattering S matrix")
    smatrix = result.Smat[0]
    unitarity_error = np.linalg.norm(smatrix.conj().T @ smatrix - np.eye(smatrix.shape[0]))

    print("\nReduced scattering smoke test:")
    print(f"  channels: {result.basis.n_channel}")
    print(f"  open channels at 300 cm^-1: {smatrix.shape[0]}")
    print(f"  ||S^dagger S-I||: {unitarity_error:.3e}")


if __name__ == "__main__":
    main()
