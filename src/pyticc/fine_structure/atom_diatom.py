from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.channel import OpenClosedChannels
from pyticc.energy import EnergyInput, get_Etot
from pyticc.fine_structure.atom import FSAtomBasis
from pyticc.fine_structure.channel import FSMonomerBasis


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSAtomDiatomChannel:
    """
    One parity-adapted body-fixed channel for a fine-structure atom and a
    fine-structure diatom.

    Members:
        level_X: int - atomic fine-structure level index in ``FSAtomBasis``
        block_Y: int - fixed-(v_Y,j_Y,epsilon_Y) monomer-Y block index
        tau_Y: int - monomer-Y eigenlevel index within ``block_Y``
        two_j12: int - twice the coupled monomer angular momentum j_12
        two_K: int - twice the nonnegative BF projection K
        E_int: float - summed channel threshold relative to both monomer zeros,
            in Hartree
    """

    level_X: int
    block_Y: int
    tau_Y: int
    two_j12: int
    two_K: int
    E_int: float

    @property
    def K(self) -> float:
        """Return the physical BF projection for shared CS/NNCC blocking."""
        return self.two_K / 2.0


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSAtomDiatomBasis(Sequence[FSAtomDiatomChannel]):
    """
    Complete fine-structure-atom plus fine-structure-diatom channel basis at
    fixed total J and parity.

    Members:
        channels: tuple[FSAtomDiatomChannel,...] - energy-ordered BF channels
        atom: FSAtomBasis - atomic fine-structure monomer basis
        monomer_Y: FSMonomerBasis - fine-structure diatom eigenbasis
        two_J: int - twice the conserved total angular momentum J
        system_parity: int - conserved total spatial parity P, -1 or 1
    """

    channels: tuple[FSAtomDiatomChannel, ...]
    atom: FSAtomBasis
    monomer_Y: FSMonomerBasis
    two_J: int
    system_parity: int

    @property
    def n_channel(self) -> int:
        """Return the number of scattering channels."""
        return len(self.channels)

    @property
    def molecule_exchange(self) -> int:
        """Return zero because atom--diatom channels have no molecule exchange."""
        return 0

    @property
    def Jtot(self) -> float:
        """Return the physical total angular momentum J."""
        return self.two_J / 2.0

    @property
    def E_int(self) -> NDArray[np.float64]:
        """Return channel thresholds in Hartree, shape ``(n_channel,)``."""
        return np.asarray([channel.E_int for channel in self.channels], dtype=np.float64)

    def open_closed(self, total_energies: EnergyInput) -> OpenClosedChannels:
        """Classify channels as open or closed at every total energy."""
        energies = get_Etot(total_energies)
        return OpenClosedChannels(self.E_int[np.newaxis, :] < energies[:, np.newaxis])

    def __len__(self) -> int:
        return self.n_channel

    @overload
    def __getitem__(self, index: int) -> FSAtomDiatomChannel: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FSAtomDiatomChannel, ...]: ...

    def __getitem__(self, index: int | slice) -> FSAtomDiatomChannel | tuple[FSAtomDiatomChannel, ...]:
        return self.channels[index]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_fs_atom_diatom_channels(
    atom: FSAtomBasis,
    monomer_Y: FSMonomerBasis,
    two_J: int,
    system_parity: int,
    *,
    E_X_cut: float = np.inf,
    E_Y_cut: float = np.inf,
    two_K_cut: int | None = None,
) -> FSAtomDiatomBasis:
    r"""
    Build parity-adapted BF channels for a fine-structure atom and a
    fine-structure diatom.

    Formula:
        The atomic level j_X and the molecular eigenlevel j_Y are coupled to

        |j_X-j_Y| <= j_12 <= j_X+j_Y,

        and the end-over-end angular momentum is an integer, so ``2*j_12`` and
        ``2*J`` must have the same parity. Retained nonnegative helicities obey

        0 <= K <= min(J,j_12).

        At the K=0 parity boundary, which exists only for integer J and j_12,
        the channel survives exactly when

        P epsilon_X epsilon_Y (-1)^(J+j_12) = +1,

        where epsilon_X is the intrinsic atomic parity. The channel threshold is

        E_int = (E_X-E_X,zero) + (E_Y-E_Y,zero).

        Energies and cutoffs are in Hartree. Channels are returned in increasing
        ``E_int`` order.

    Inputs:
        atom: FSAtomBasis - atomic fine-structure monomer basis
        monomer_Y: FSMonomerBasis - diagonalized molecular basis
        two_J: int - twice total angular momentum J
        system_parity: int - total spatial parity P, -1 or 1
        E_X_cut: float - largest retained atomic relative level energy, Hartree
        E_Y_cut: float - largest retained monomer-Y relative threshold, Hartree
        two_K_cut: int | None - optional largest retained twice-helicity

    Returns:
        basis: FSAtomDiatomBasis - complete energy-ordered channel basis
    """
    if two_J < 0 or system_parity not in (-1, 1):
        message = "two_J must be nonnegative and system_parity must be -1 or 1"
        logger.error(message)
        raise ValueError(message)
    if two_K_cut is not None and two_K_cut < 0:
        message = f"two_K_cut must be nonnegative, but got {two_K_cut}"
        logger.error(message)
        raise ValueError(message)

    channels: list[FSAtomDiatomChannel] = []
    for level_X, two_j_X in enumerate(atom.two_j_levels):
        threshold_X = float(atom.level_energies[level_X] - atom.energy_zero)
        if threshold_X > E_X_cut:
            continue
        for block_Y_index, block_Y in enumerate(monomer_Y.blocks):
            for tau_Y, energy_Y in enumerate(block_Y.energies):
                threshold_Y = float(energy_Y - monomer_Y.energy_zero)
                if threshold_Y > E_Y_cut:
                    continue
                for two_j12 in range(abs(two_j_X - block_Y.two_j), two_j_X + block_Y.two_j + 1, 2):
                    if two_j12 % 2 != two_J % 2:
                        continue
                    two_K_max = min(two_J, two_j12)
                    if two_K_cut is not None:
                        two_K_max = min(two_K_max, two_K_cut)
                    for two_K in range(two_J % 2, two_K_max + 1, 2):
                        if two_K == 0:
                            exponent = (two_J + two_j12) // 2
                            parity_phase = system_parity * atom.atom_parity * block_Y.parity * (-1) ** exponent
                            if parity_phase != 1:
                                continue
                        channels.append(
                            FSAtomDiatomChannel(
                                level_X=level_X,
                                block_Y=block_Y_index,
                                tau_Y=tau_Y,
                                two_j12=two_j12,
                                two_K=two_K,
                                E_int=threshold_X + threshold_Y,
                            )
                        )

    channels.sort(
        key=lambda channel: (
            channel.E_int,
            channel.level_X,
            channel.block_Y,
            channel.tau_Y,
            channel.two_j12,
            channel.two_K,
        )
    )
    return FSAtomDiatomBasis(tuple(channels), atom, monomer_Y, two_J, system_parity)


# ----------------------------------------------------------------------------------------
