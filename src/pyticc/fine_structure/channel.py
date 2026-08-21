from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.channel import OpenClosedChannels
from pyticc.basis.dvr import build_SineDVR
from pyticc.basis.podvr import VibPODVR, build_VibPODVR
from pyticc.energy import EnergyInput, get_Etot
from pyticc.fine_structure.basis import build_primitive_states
from pyticc.fine_structure.constants import FSConstantsTable, load_fs_constants_csv
from pyticc.fine_structure.monomer import FSLevelBlock, diagonalize_block
from pyticc.fine_structure.operators import FSConstants


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSMonomerBasis:
    """
    Vibrational and fine-structure eigenbasis of one open-shell diatom.

    Members:
        vib: VibPODVR - shared PODVR vibration basis
        blocks: tuple[FSLevelBlock,...] - fixed-(v,j,parity) eigenblocks
        energy_zero: float - energy subtracted from scattering thresholds,
            in Hartree
        two_lambda_abs: int - twice |Lambda|
        two_S: int - twice electronic spin S
    """

    vib: VibPODVR
    blocks: tuple[FSLevelBlock, ...]
    energy_zero: float
    two_lambda_abs: int
    two_S: int


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_fs_monomer_basis(
    vib: VibPODVR,
    two_j_values: Sequence[int],
    two_lambda_abs: int,
    two_S: int,
    constants: Sequence[FSConstants] | FSConstants | FSConstantsTable | str | Path,
    *,
    reflection_parity: int = 1,
    energy_zero: float | None = None,
) -> FSMonomerBasis:
    """
    Diagonalize every retained vibrational, rotational, and parity block.

    Inputs:
        vib: VibPODVR - shared vibrational PODVR basis
        two_j_values: Sequence[int] - retained values of twice j
        two_lambda_abs: int - twice |Lambda|
        two_S: int - twice spin S
        constants: Sequence[FSConstants] | FSConstants | FSConstantsTable | path
            - one shared set, one set per v, or a long-format CSV table
        reflection_parity: int - Sigma electronic reflection symmetry
        energy_zero: float | None - threshold zero; None uses the lowest level

    Returns:
        basis: FSMonomerBasis - fine-structure monomer eigenbasis
    """
    table = load_fs_constants_csv(constants) if isinstance(constants, str | Path) else constants
    if isinstance(table, FSConstantsTable):
        per_v = tuple(table.for_v(v) for v in range(vib.energies.size))
    elif isinstance(table, FSConstants):
        per_v = (table,) * vib.energies.size
    else:
        per_v = tuple(table)
    if len(per_v) != vib.energies.size:
        message = f"Expected {vib.energies.size} fine-structure constant sets, but got {len(per_v)}"
        logger.error(message)
        raise ValueError(message)
    blocks: list[FSLevelBlock] = []
    for v, vibrational_energy in enumerate(vib.energies):
        for two_j in sorted(set(two_j_values)):
            states = build_primitive_states((v,), (two_j,), two_lambda_abs, two_S)
            if not states:
                continue
            for parity in (-1, 1):
                block = diagonalize_block(
                    states,
                    per_v[v],
                    parity,
                    vibrational_energy=float(vibrational_energy),
                    reflection_parity=reflection_parity,
                )
                if block.energies.size:
                    blocks.append(block)
    if not blocks:
        message = "No fine-structure monomer blocks survived the quantum-number and parity selections"
        logger.error(message)
        raise ValueError(message)
    zero = min(float(np.min(block.energies)) for block in blocks) if energy_zero is None else float(energy_zero)
    return FSMonomerBasis(vib, tuple(blocks), zero, two_lambda_abs, two_S)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare_fs_monomer(
    potential: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    r: tuple[float, float],
    n_dvr: int,
    n_podvr: int,
    vmax: int,
    mass: float,
    two_j_values: Sequence[int],
    two_lambda_abs: int,
    two_S: int,
    constants: Sequence[FSConstants] | FSConstants | FSConstantsTable | str | Path,
    reflection_parity: int = 1,
    energy_zero: float | None = None,
) -> FSMonomerBasis:
    """
    Prepare vibration through the existing DVR/PODVR path and add fine structure.

    Inputs:
        potential: Callable - isolated-diatom potential in bohr and Hartree
        r: tuple[float,float] - sine-DVR interval in bohr
        n_dvr: int - primitive sine-DVR size
        n_podvr: int - contracted PODVR size
        vmax: int - largest retained vibrational quantum number
        mass: float - diatomic reduced mass in atomic units
        two_j_values: Sequence[int] - retained twice-j values
        two_lambda_abs: int - twice |Lambda|
        two_S: int - twice spin S
        constants: Sequence[FSConstants] | FSConstants | FSConstantsTable | path
            - effective constants or their long-format CSV file
        reflection_parity: int - Sigma electronic reflection symmetry
        energy_zero: float | None - scattering threshold zero in Hartree

    Returns:
        basis: FSMonomerBasis - PODVR vibration and fine-structure eigenlevels
    """
    dvr = build_SineDVR(r[0], r[1], n_dvr, mass, potential)
    vib = build_VibPODVR(dvr, n_podvr, vmax)
    return build_fs_monomer_basis(
        vib,
        two_j_values,
        two_lambda_abs,
        two_S,
        constants,
        reflection_parity=reflection_parity,
        energy_zero=energy_zero,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSChannel:
    """
    One parity-adapted body-fixed open-shell atom-diatom channel.

    Members:
        block: int - monomer block index
        tau: int - eigenlevel index within the block
        two_K: int - twice the nonnegative BF projection K
        E_int: float - channel threshold relative to the monomer zero, Hartree
    """

    block: int
    tau: int
    two_K: int
    E_int: float


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSChannelBasis(Sequence[FSChannel]):
    """
    Complete fine-structure channel basis for fixed total J and parity.

    Members:
        channels: tuple[FSChannel,...] - energy-ordered channels
        monomer: FSMonomerBasis - open-shell diatom eigenbasis
        two_J: int - twice conserved total angular momentum J
        system_parity: int - conserved total spatial parity
    """

    channels: tuple[FSChannel, ...]
    monomer: FSMonomerBasis
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
        """Return channel thresholds in Hartree, shape (n_channel,)."""
        return np.asarray([channel.E_int for channel in self.channels], dtype=np.float64)

    def open_closed(self, total_energies: EnergyInput) -> OpenClosedChannels:
        """Classify fine-structure channels as open or closed at each energy."""
        energies = get_Etot(total_energies)
        return OpenClosedChannels(self.E_int[np.newaxis, :] < energies[:, np.newaxis])

    def __len__(self) -> int:
        return self.n_channel

    @overload
    def __getitem__(self, index: int) -> FSChannel: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FSChannel, ...]: ...

    def __getitem__(self, index: int | slice) -> FSChannel | tuple[FSChannel, ...]:
        return self.channels[index]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_fs_channels(
    monomer: FSMonomerBasis,
    two_J: int,
    system_parity: int,
    *,
    E_cut: float = np.inf,
    two_K_cut: int | None = None,
) -> FSChannelBasis:
    r"""
    Build parity-adapted BF channels for an atom plus open-shell diatom.

    Formula:
        0 <= K <= min(J,j). For K=0, which exists only for integer J and j,
        the parity-adapted channel survives when

        P epsilon (-1)^(J+j) = +1.

    Inputs:
        monomer: FSMonomerBasis - diagonalized molecular basis
        two_J: int - twice total J
        system_parity: int - total parity P, -1 or 1
        E_cut: float - largest retained relative threshold in Hartree
        two_K_cut: int | None - optional twice-K truncation

    Returns:
        basis: FSChannelBasis - complete energy-ordered channel basis
    """
    if two_J < 0 or system_parity not in (-1, 1):
        message = "two_J must be nonnegative and system_parity must be -1 or 1"
        logger.error(message)
        raise ValueError(message)
    channels: list[FSChannel] = []
    for block_index, block in enumerate(monomer.blocks):
        two_K_max = min(two_J, block.two_j)
        if two_K_cut is not None:
            two_K_max = min(two_K_max, two_K_cut)
        first_two_K = two_J % 2
        for tau, energy in enumerate(block.energies):
            threshold = float(energy - monomer.energy_zero)
            if threshold > E_cut:
                continue
            for two_K in range(first_two_K, two_K_max + 1, 2):
                if two_K == 0:
                    exponent = (two_J + block.two_j) // 2
                    if system_parity * block.parity * (-1) ** exponent != 1:
                        continue
                channels.append(FSChannel(block_index, tau, two_K, threshold))
    channels.sort(key=lambda channel: (channel.E_int, channel.block, channel.tau, channel.two_K))
    return FSChannelBasis(tuple(channels), monomer, two_J, system_parity)


# ----------------------------------------------------------------------------------------
