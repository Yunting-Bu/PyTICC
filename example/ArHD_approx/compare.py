from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import pyticc as ticc
from pyticc.basis.channel import ChannelBasis

INITIAL_STATE = (0, 5)
FINAL_STATE = (0, 2)
JTOT = 5
SYSTEM_PARITY = -1


def state_to_state_probability(
    result: ticc.ScatteringResult | ticc.CoupledStatesResult,
    initial: tuple[int, int],
    final: tuple[int, int],
) -> NDArray[np.float64]:
    r"""
    Sum one inelastic state-to-state probability over angular channels.

    Formula:
        P_fi^{J epsilon}(E) = sum_a sum_b |S_ba^{J epsilon}(E)|^2.

        For exact CC, a and b span the initial and final SF orbital-angular-
        momentum channels l and l'. For CS and NNCC, they span the K and K'
        channels supplied by the K block that owns each incoming channel. The
        ownership rule prevents overlapping NNCC blocks from being counted twice.

    Inputs:
        result: ScatteringResult | CoupledStatesResult - one fixed-J,
            fixed-parity calculation containing n_energy energies
        initial: tuple[int,int] - initial HD state (v,j)
        final: tuple[int,int] - final HD state (v',j') distinct from initial

    Returns:
        probability: NDArray[np.float64] - state-to-state probabilities with
            shape (n_energy,)
    """
    basis = result.basis
    assert isinstance(basis, ChannelBasis)
    incoming = {index for index, channel in enumerate(basis) if (channel.mis_Y.v, channel.mis_Y.j) == initial}
    outgoing = {index for index, channel in enumerate(basis) if (channel.mis_Y.v, channel.mis_Y.j) == final}
    probability = np.zeros(result.Etot.size, dtype=np.float64)

    if isinstance(result, ticc.ScatteringResult):
        sources = ((incoming, result.open_channel_indices, result.Smat),)
    else:
        sources = tuple(
            (incoming.intersection(block.block.owned_channel_indices), block.open_channel_indices, block.Smat_BF) for block in result.blocks
        )

    for owned_incoming, open_batches, matrices in sources:
        for energy_index, (open_indices, Smat) in enumerate(zip(open_batches, matrices, strict=True)):
            positions = {int(index): position for position, index in enumerate(open_indices)}
            rows = [positions[index] for index in outgoing if index in positions]
            columns = [positions[index] for index in owned_incoming if index in positions]
            if rows and columns:
                probability[energy_index] += float(np.sum(np.abs(Smat[np.ix_(rows, columns)]) ** 2))
    return probability


def main() -> None:
    """Compare exact CC, NNCC, and CS state-to-state probabilities for Ar+HD."""
    directory = Path(__file__).resolve().parent
    pes_dir = directory / "pes"
    pes = ticc.load_fortran_pes(
        [pes_dir / "interaction-PES.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
    )

    try:
        mass_Ar, mass_H, mass_D = ticc.element_masses_au("Ar", "H", "D")
        mass_HD = mass_H + mass_D
        diatom = ticc.prepare_Diatom(
            pes.monomer_Y,
            r=(1.5, 4.5),
            n_dvr=100,
            n_podvr=5,
            vmax=0,
            jmax=30,
            mass=ticc.reduced_mass(mass_H, mass_D),
        )
        collision_mass = ticc.reduced_mass(mass_Ar, mass_HD)
        channel = ticc.ChannelSpec(E_Y_cut=25000.0 * ticc.CM2AU)
        methods = (
            ("Exact CC", ticc.Approx.EXACT),
            (r"NNCC ($\Delta=1$)", ticc.Approx.NNCC),
            ("CS", ticc.Approx.CS),
        )
        systems = {
            label: ticc.build_ScattSystem(
                ticc.AtomSpec(),
                diatom,
                scattering_type="A+BC",
                Jtot=JTOT,
                system_parity=SYSTEM_PARITY,
                channel=channel,
                approx=approx,
                K_delta=1,
                potential=pes,
                reduced_mass=collision_mass,
            )
            for label, approx in methods
        }

        initial_threshold = float(diatom.Eint[INITIAL_STATE])
        collision_energies_cm = np.geomspace(0.1, 100.0, 31)
        total_energies = initial_threshold + collision_energies_cm * ticc.CM2AU
        potential_grid = ticc.prepare_potential(
            next(iter(systems.values())),
            (3.0, 6.0, 10.0, 50.0),
            (0.01, 0.03, 0.05),
            n_theta=30,
            processes=4,
        )
        propagation = ticc.Propagation()

        probabilities: dict[str, NDArray[np.float64]] = {}
        for label, system in systems.items():
            result = ticc.solve(system, total_energies, potential_grid, propagation)
            probabilities[label] = state_to_state_probability(result, INITIAL_STATE, FINAL_STATE)

        figure, axes = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        for label, values in probabilities.items():
            axes.plot(collision_energies_cm, values, marker="o", label=label)
        axes.set_xscale("log")
        axes.set(xlabel=r"Collision energy / cm$^{-1}$", ylabel=r"$P_{(0,4)\leftarrow(0,5)}$")
        axes.set_title(r"Ar + HD: $J=5$, total parity $=-1$")
        axes.grid(alpha=0.25)
        axes.legend(frameon=False)
        figure.savefig(directory / "probability.png", dpi=300)
        plt.close(figure)
    finally:
        pes.close()


if __name__ == "__main__":
    main()
