import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Protocol, overload

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.energy import EnergyInput, get_Etot
from pyticc.system import MolInnerState, MonomerType, ScattSystem


# ----------------------------------------------------------------------------------------
@lru_cache
def set_Kmax(j_couple: int, Jtot: int, Kcut: int | None = None) -> int:
    """Return the maximum helicity for one coupled angular state."""
    Kmax = min(j_couple, Jtot)
    if Kcut is not None:
        Kmax = min(Kmax, Kcut)
    return Kmax


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TruncSpec:
    """
    Monomer-energy and helicity truncations.

    Members:
        E_X_cut: float - X-monomer internal-energy cutoff in atomic units
        E_Y_cut: float - Y-monomer internal-energy cutoff in atomic units
        K_cut: int | None - maximum retained helicity, or None to retain every allowed K
    """

    E_X_cut: float = math.inf
    E_Y_cut: float = math.inf
    K_cut: int | None = None

    def __post_init__(self) -> None:
        if self.E_X_cut < 0.0 or self.E_Y_cut < 0.0:
            message = f"Energy cutoffs must be non-negative, but got E_X_cut={self.E_X_cut}, E_Y_cut={self.E_Y_cut}"
            logger.error(message)
            raise ValueError(message)
        if self.K_cut is not None and self.K_cut < 0:
            message = f"K_cut must be non-negative, but got K_cut={self.K_cut}"
            logger.error(message)
            raise ValueError(message)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Channel:
    mis_X: MolInnerState
    mis_Y: MolInnerState
    j_couple: int
    K: int
    Jtot: int
    system_parity: int
    E_int: float
    index: int = -1

    def __str__(self) -> str:
        qn_X = f"t={self.mis_X.t}" if self.mis_X.t is not None else f"v={'-' if self.mis_X.v is None else self.mis_X.v}"
        qn_Y = f"t={self.mis_Y.t}" if self.mis_Y.t is not None else f"v={'-' if self.mis_Y.v is None else self.mis_Y.v}"
        electronic_X = "" if self.mis_X.electronic_state is None else f"e={self.mis_X.electronic_state}, "
        electronic_Y = "" if self.mis_Y.electronic_state is None else f"e={self.mis_Y.electronic_state}, "
        return (
            f"Channel[{self.index}] "
            f"X({electronic_X}{qn_X}, j={self.mis_X.j}) "
            f"Y({electronic_Y}{qn_Y}, j={self.mis_Y.j}) "
            f"j_couple={self.j_couple} K={self.K} Jtot={self.Jtot} "
            f"parity={self.system_parity:+d} E_int={self.E_int:.10f} a.u."
        )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OpenClosedChannels:
    """
    Open and closed channel information over a total-energy grid.

    Members:
        total_energies: NDArray[np.float64] - total energies in atomic units, shape
            (n_energy,)
        open_mask: NDArray[np.bool_] - open-channel mask with shape (n_energy, n_channel)
        n_open: NDArray[np.int64] - number of open channels at each energy, shape
            (n_energy,)
        n_closed: NDArray[np.int64] - number of closed channels at each energy, shape
            (n_energy,)
    """

    total_energies: NDArray[np.float64]
    open_mask: NDArray[np.bool_]
    n_open: NDArray[np.int64]
    n_closed: NDArray[np.int64]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelBasis(Sequence[Channel]):
    """
    Complete channel basis for one field-free scattering block.

    Members:
        channels: tuple[Channel, ...] - channels ordered by increasing internal energy
        n_channel: int - total number of channels
    """

    channels: tuple[Channel, ...]

    @property
    def n_channel(self) -> int:
        """Return the number of channels in this basis."""
        return len(self.channels)

    @property
    def E_int(self) -> NDArray[np.float64]:
        """Return channel internal energies with shape (n_channel,)."""
        return np.asarray([channel.E_int for channel in self.channels], dtype=np.float64)

    def __len__(self) -> int:
        return self.n_channel

    @overload
    def __getitem__(self, index: int) -> Channel: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Channel, ...]: ...

    def __getitem__(self, index: int | slice) -> Channel | tuple[Channel, ...]:
        return self.channels[index]

    def open_closed(self, total_energies: EnergyInput) -> OpenClosedChannels:
        """
        Classify channels as open or closed at each total energy.

        Inputs:
            total_energies: EnergyInput - total-energy array with shape (n_energy,),
                or a one-column text file in atomic units

        Returns:
            result: OpenClosedChannels - energies and counts with shape (n_energy,),
                and an open-channel mask with shape (n_energy, n_channel)
        """
        energies = get_Etot(total_energies)
        open_mask = self.E_int[np.newaxis, :] < energies[:, np.newaxis]
        n_open = np.asarray(np.sum(open_mask, axis=1), dtype=np.int64)
        n_closed = np.asarray(self.n_channel - n_open, dtype=np.int64)

        return OpenClosedChannels(
            total_energies=energies,
            open_mask=open_mask,
            n_open=n_open,
            n_closed=n_closed,
        )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
class ParityRule(Protocol):
    def allow_K0(self, mis_X: MolInnerState, mis_Y: MolInnerState, j_couple: int) -> bool:
        """Return whether one coupled monomer state is allowed at K=0."""
        ...


@dataclass(frozen=True)
class ClosedShellParity:
    """Apply the field-free closed-shell parity condition to K=0 channels."""

    system_parity: int
    Jtot: int

    def allow_K0(self, mis_X: MolInnerState, mis_Y: MolInnerState, j_couple: int) -> bool:
        """Return whether the coupled monomer state belongs to this K=0 parity block."""
        phase = self.system_parity * (-1) ** (mis_X.j + mis_Y.j + j_couple + self.Jtot)
        return phase == 1


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelBuilder:
    """Construct and energy-order one field-free channel basis."""

    sys: ScattSystem
    trunc: TruncSpec

    def build(self) -> ChannelBasis:
        """Enumerate channels allowed by angular momentum, parity, energy, and helicity."""
        if self.sys.Jtot is None or self.sys.system_parity is None:
            message = "Field-free channel construction requires Jtot and system_parity"
            logger.error(message)
            raise ValueError(message)

        parity = ClosedShellParity(self.sys.system_parity, self.sys.Jtot)
        monomer_types = (self.sys.monomer_X.type, self.sys.monomer_Y.type)
        atom_triatom = monomer_types in (
            (MonomerType.ATOM, MonomerType.TRIATOM),
            (MonomerType.TRIATOM, MonomerType.ATOM),
        )
        parity_block_sign = self.sys.system_parity * (-1) ** self.sys.Jtot
        if atom_triatom:
            triatom = self.sys.monomer_X if self.sys.monomer_X.type is MonomerType.TRIATOM else self.sys.monomer_Y
            if getattr(triatom, "parity_block_sign", parity_block_sign) != parity_block_sign:
                message = "Triatomic basis parity_block_sign does not match system_parity*(-1)^Jtot"
                logger.error(message)
                raise ValueError(message)
        channels: list[Channel] = []

        for mis_X in self.sys.monomer_X.mis_iter(self.trunc.E_X_cut):
            for mis_Y in self.sys.monomer_Y.mis_iter(self.trunc.E_Y_cut):
                for j_couple in range(abs(mis_X.j - mis_Y.j), mis_X.j + mis_Y.j + 1):
                    Kmax = set_Kmax(j_couple, self.sys.Jtot, self.trunc.K_cut)
                    for K in range(Kmax + 1):
                        if not self.sys.monomer_X.allows_K(mis_X, K) or not self.sys.monomer_Y.allows_K(mis_Y, K):
                            continue
                        if K == 0:
                            if not atom_triatom and not parity.allow_K0(mis_X, mis_Y, j_couple):
                                continue

                        E_int = float(self.sys.monomer_X.energy(mis_X, K) + self.sys.monomer_Y.energy(mis_Y, K))
                        channels.append(
                            Channel(
                                mis_X=mis_X,
                                mis_Y=mis_Y,
                                j_couple=j_couple,
                                K=K,
                                Jtot=self.sys.Jtot,
                                system_parity=self.sys.system_parity,
                                E_int=E_int,
                            )
                        )

        channels.sort(key=lambda channel: channel.E_int)
        indexed_channels = tuple(replace(channel, index=index) for index, channel in enumerate(channels))
        return ChannelBasis(channels=indexed_channels)


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    from pyticc.basis.monomer import AtomSpec, DiatomSpec
    from pyticc.system import ScattSystem

    def A_plus_BC() -> None:
        """Print a minimal atom-diatom channel example."""
        atom = AtomSpec()
        diatom = DiatomSpec(Eint=np.array([[0.0, 0.01, 0.03]]), vmax=0, jmax=2)

        system = ScattSystem(monomer_X=atom, monomer_Y=diatom, Jtot=1, system_parity=-1)
        channels = ChannelBuilder(system, TruncSpec()).build()

        for channel in channels:
            print(channel)

    def AB_plus_CD() -> None:
        """Print a minimal diatom-diatom channel example."""
        diatom_X = DiatomSpec(Eint=np.array([[0.0, 0.01]]), vmax=0, jmax=1, jpar=-1)
        diatom_Y = DiatomSpec(Eint=np.array([[0.0, 0.02]]), vmax=0, jmax=1, jpar=-1)

        system = ScattSystem(monomer_X=diatom_X, monomer_Y=diatom_Y, Jtot=1, system_parity=1)
        channels = ChannelBuilder(system, TruncSpec()).build()

        for channel in channels:
            print(channel)

    print("Test case: A + BC")
    A_plus_BC()
    print("\nTest case: AB + CD")
    AB_plus_CD()
