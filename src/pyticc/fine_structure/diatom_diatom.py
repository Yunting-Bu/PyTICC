from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.channel import OpenClosedChannels
from pyticc.energy import EnergyInput, get_Etot
from pyticc.fine_structure.channel import FSMonomerBasis


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSDiatomDiatomChannel:
    """
    One parity-adapted body-fixed channel for two fine-structure diatoms.

    Members:
        block_X: int - fixed-(v_X,j_X,epsilon_X) monomer-X block index
        tau_X: int - monomer-X eigenlevel index within ``block_X``
        block_Y: int - fixed-(v_Y,j_Y,epsilon_Y) monomer-Y block index
        tau_Y: int - monomer-Y eigenlevel index within ``block_Y``
        two_j12: int - twice the coupled monomer angular momentum j_12
        two_K: int - twice the nonnegative BF projection K
        E_int: float - summed channel threshold relative to both monomer zeros,
            in Hartree
    """

    block_X: int
    tau_X: int
    block_Y: int
    tau_Y: int
    two_j12: int
    two_K: int
    E_int: float


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSDiatomDiatomBasis(Sequence[FSDiatomDiatomChannel]):
    """
    Complete two-fine-structure-diatom channel basis at fixed total J and parity.

    Members:
        channels: tuple[FSDiatomDiatomChannel,...] - energy-ordered BF channels
        monomer_X: FSMonomerBasis - first fine-structure diatom eigenbasis
        monomer_Y: FSMonomerBasis - second fine-structure diatom eigenbasis
        two_J: int - twice the conserved total angular momentum J
        system_parity: int - conserved total spatial parity P, -1 or 1
    """

    channels: tuple[FSDiatomDiatomChannel, ...]
    monomer_X: FSMonomerBasis
    monomer_Y: FSMonomerBasis
    two_J: int
    system_parity: int

    @property
    def n_channel(self) -> int:
        """Return the number of scattering channels."""
        return len(self.channels)

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
    def __getitem__(self, index: int) -> FSDiatomDiatomChannel: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FSDiatomDiatomChannel, ...]: ...

    def __getitem__(self, index: int | slice) -> FSDiatomDiatomChannel | tuple[FSDiatomDiatomChannel, ...]:
        return self.channels[index]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_fs_diatom_diatom_channels(
    monomer_X: FSMonomerBasis,
    monomer_Y: FSMonomerBasis,
    two_J: int,
    system_parity: int,
    *,
    E_X_cut: float = np.inf,
    E_Y_cut: float = np.inf,
    two_K_cut: int | None = None,
) -> FSDiatomDiatomBasis:
    r"""
    Build parity-adapted BF channels for two fine-structure diatoms.

    Formula:
        The two monomer angular momenta are coupled according to

        |j_X-j_Y| <= j_12 <= j_X+j_Y,

        and the end-over-end angular momentum is an integer, so ``2*j_12`` and
        ``2*J`` must have the same parity. Retained nonnegative helicities obey

        0 <= K <= min(J,j_12).

        At the K=0 parity boundary, which exists only for integer J and j_12,
        the channel survives exactly when

        P epsilon_X epsilon_Y (-1)^(J+j_12) = +1.

        The channel threshold is

        E_int = (E_X-E_X,zero) + (E_Y-E_Y,zero).

        Monomer eigenstates use the normalization and parity conventions stored
        by ``FSMonomerBasis``. Energies and cutoffs are in Hartree. Channels are
        returned in increasing ``E_int`` order.

    Inputs:
        monomer_X: FSMonomerBasis - first diagonalized molecular basis
        monomer_Y: FSMonomerBasis - second diagonalized molecular basis
        two_J: int - twice total angular momentum J
        system_parity: int - total spatial parity P, -1 or 1
        E_X_cut: float - largest retained monomer-X relative threshold, Hartree
        E_Y_cut: float - largest retained monomer-Y relative threshold, Hartree
        two_K_cut: int | None - optional largest retained twice-helicity

    Returns:
        basis: FSDiatomDiatomBasis - complete energy-ordered channel basis
    """
    if two_J < 0 or system_parity not in (-1, 1):
        message = "two_J must be nonnegative and system_parity must be -1 or 1"
        logger.error(message)
        raise ValueError(message)
    if two_K_cut is not None and two_K_cut < 0:
        message = f"two_K_cut must be nonnegative, but got {two_K_cut}"
        logger.error(message)
        raise ValueError(message)

    channels: list[FSDiatomDiatomChannel] = []
    for block_X_index, block_X in enumerate(monomer_X.blocks):
        for tau_X, energy_X in enumerate(block_X.energies):
            threshold_X = float(energy_X - monomer_X.energy_zero)
            if threshold_X > E_X_cut:
                continue
            for block_Y_index, block_Y in enumerate(monomer_Y.blocks):
                for tau_Y, energy_Y in enumerate(block_Y.energies):
                    threshold_Y = float(energy_Y - monomer_Y.energy_zero)
                    if threshold_Y > E_Y_cut:
                        continue
                    for two_j12 in range(abs(block_X.two_j - block_Y.two_j), block_X.two_j + block_Y.two_j + 1, 2):
                        if two_j12 % 2 != two_J % 2:
                            continue
                        two_K_max = min(two_J, two_j12)
                        if two_K_cut is not None:
                            two_K_max = min(two_K_max, two_K_cut)
                        for two_K in range(two_J % 2, two_K_max + 1, 2):
                            if two_K == 0:
                                exponent = (two_J + two_j12) // 2
                                parity_phase = system_parity * block_X.parity * block_Y.parity * (-1) ** exponent
                                if parity_phase != 1:
                                    continue
                            channels.append(
                                FSDiatomDiatomChannel(
                                    block_X=block_X_index,
                                    tau_X=tau_X,
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
            channel.block_X,
            channel.tau_X,
            channel.block_Y,
            channel.tau_Y,
            channel.two_j12,
            channel.two_K,
        )
    )
    return FSDiatomDiatomBasis(tuple(channels), monomer_X, monomer_Y, two_J, system_parity)


# ----------------------------------------------------------------------------------------
