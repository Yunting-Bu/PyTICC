from collections.abc import Sequence
from dataclasses import dataclass, replace
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
class FSExchangeAdaptation:
    r"""Sparse complete-molecule exchange expansion for two FS diatoms.

    Formula:
        For c=(a,b,j12,K;J,P), a=(block_X,tau_X), b=(block_Y,tau_Y),
        exchange maps |c> to s_c |bar(c)> with

        s_c = P epsilon_a epsilon_b (-1)^(j_a+j_b-j12).

        This follows from the SF phase (-1)^(j_a+j_b-j12+L) and
        P=epsilon_a epsilon_b (-1)^L. The integer exponent is evaluated with
        doubled angular momenta. bar(c) exchanges a and b without changing K.
        For eta=+/-1 retain a<=b and use

        |c;eta> = [|c>+eta s_c |bar(c)>]/sqrt[2(1+delta_ab)].

        For a=b retain only eta s_c=1, with coefficient one. Thus the real,
        dimensionless expansion T satisfies T.T T=I. No nuclear-spin state
        or statistical weight is included.

    Members:
        eta: int - exchange eigenvalue +/-1
        source_channels: tuple[FSDiatomDiatomChannel,...] - labeled BF basis
        source_indices: NDArray[np.int64] - two source positions per retained
            channel, shape (n_adapted,2)
        coefficients: NDArray[np.float64] - expansion weights with the same
            shape; second weight zero for a=b
        permutation: NDArray[np.int64] - exchanged source position, (n_source,)
        phases: NDArray[np.float64] - s_c in source order, (n_source,)
    """

    eta: int
    source_channels: tuple[FSDiatomDiatomChannel, ...]
    source_indices: NDArray[np.int64]
    coefficients: NDArray[np.float64]
    permutation: NDArray[np.int64]
    phases: NDArray[np.float64]


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
        exchange: FSExchangeAdaptation | None - normalized exchange expansion;
            channels represent canonical monomer state pairs when present
    """

    channels: tuple[FSDiatomDiatomChannel, ...]
    monomer_X: FSMonomerBasis
    monomer_Y: FSMonomerBasis
    two_J: int
    system_parity: int
    exchange: FSExchangeAdaptation | None = None

    @property
    def molecule_exchange(self) -> int:
        """Return the complete-molecule exchange eigenvalue, or zero if unused."""
        return 0 if self.exchange is None else self.exchange.eta

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
def adapt_fs_molecule_exchange(basis: FSDiatomDiatomBasis, eta: int) -> FSDiatomDiatomBasis:
    r"""Construct normalized, nonzero exchange eigenchannels.

    Formula:
        Use the phase and normalized expansion defined by FSExchangeAdaptation.
        Canonical ordering is lexicographic in (block,tau), which uniquely
        identifies a monomer eigenstate even for degenerate levels.

    Inputs:
        basis: FSDiatomDiatomBasis - labeled, exchange-closed BF channels using
            the same FSMonomerBasis object for both monomers
        eta: int - exchange eigenvalue +/-1

    Returns:
        adapted: FSDiatomDiatomBasis - energy-ordered canonical channels and
            dimensionless expansion; forbidden blocks may have zero channels
    """
    if isinstance(eta, bool) or not isinstance(eta, int) or eta not in (-1, 1):
        raise ValueError("Molecule exchange eigenvalue must be -1 or 1")
    if basis.exchange is not None:
        raise ValueError("Channel basis is already molecule-exchange adapted")
    if basis.monomer_X is not basis.monomer_Y:
        raise ValueError("Molecule exchange requires the same FS monomer basis object for X and Y")
    source = tuple(basis)
    lookup = {(c.block_X, c.tau_X, c.block_Y, c.tau_Y, c.two_j12, c.two_K): i for i, c in enumerate(source)}
    retained: list[FSDiatomDiatomChannel] = []
    positions: list[tuple[int, int]] = []
    weights: list[tuple[float, float]] = []
    permutation = np.empty(len(source), dtype=np.int64)
    phases = np.empty(len(source), dtype=np.float64)
    for i, c in enumerate(source):
        partner = lookup.get((c.block_Y, c.tau_Y, c.block_X, c.tau_X, c.two_j12, c.two_K))
        if partner is None:
            raise ValueError("Molecule exchange requires a channel basis closed under X/Y exchange")
        block_X, block_Y = basis.monomer_X.blocks[c.block_X], basis.monomer_Y.blocks[c.block_Y]
        phase = basis.system_parity * block_X.parity * block_Y.parity * (-1) ** ((block_X.two_j + block_Y.two_j - c.two_j12) // 2)
        permutation[i], phases[i] = partner, phase
        a, b = (c.block_X, c.tau_X), (c.block_Y, c.tau_Y)
        if a > b or (a == b and eta * phase != 1):
            continue
        retained.append(c)
        positions.append((i, partner))
        weights.append((1.0, 0.0) if a == b else (1.0 / np.sqrt(2.0), eta * phase / np.sqrt(2.0)))
    source_indices = np.asarray(positions, dtype=np.int64).reshape(-1, 2)
    coefficients = np.asarray(weights, dtype=np.float64).reshape(-1, 2)
    for array in (source_indices, coefficients, permutation, phases):
        array.setflags(write=False)
    exchange = FSExchangeAdaptation(eta, source, source_indices, coefficients, permutation, phases)
    return replace(basis, channels=tuple(retained), exchange=exchange)


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
    molecule_exchange: int = 0,
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
        molecule_exchange: int - 0 keeps labeled channels; +/-1 selects the
            normalized exchange eigenchannels defined by FSExchangeAdaptation

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
    if isinstance(molecule_exchange, bool) or not isinstance(molecule_exchange, int) or molecule_exchange not in (-1, 0, 1):
        raise ValueError("molecule_exchange must be -1, 0, or 1")
    if molecule_exchange:
        if monomer_X is not monomer_Y:
            raise ValueError("Molecule exchange requires the same FS monomer basis object for X and Y")
        selected_X = {(i, t) for i, b in enumerate(monomer_X.blocks) for t, e in enumerate(b.energies) if e - monomer_X.energy_zero <= E_X_cut}
        selected_Y = {(i, t) for i, b in enumerate(monomer_Y.blocks) for t, e in enumerate(b.energies) if e - monomer_Y.energy_zero <= E_Y_cut}
        if selected_X != selected_Y:
            raise ValueError("Molecule exchange requires identical retained FS monomer states (exchange-closed energy cutoffs)")

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
    basis = FSDiatomDiatomBasis(tuple(channels), monomer_X, monomer_Y, two_J, system_parity)
    return adapt_fs_molecule_exchange(basis, molecule_exchange) if molecule_exchange else basis


# ----------------------------------------------------------------------------------------
