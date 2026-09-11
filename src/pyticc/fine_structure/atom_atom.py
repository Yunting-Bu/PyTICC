from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import NDArray

from pyticc.basis.channel import OpenClosedChannels
from pyticc.energy import EnergyInput, get_Etot
from pyticc.fine_structure.atom import FSAtomBasis


@dataclass(frozen=True)
class FSAtomAtomChannel:
    """One parity-adapted atomic pair channel.

    Members:
        level_X, level_Y: int - indices of experimental atomic levels
        two_j12: int - twice the coupled electronic angular momentum
        two_K: int - twice the nonnegative BF helicity
        E_int: float - summed relative atomic threshold in Hartree
    """

    level_X: int
    level_Y: int
    two_j12: int
    two_K: int
    E_int: float

    @property
    def K(self) -> float:
        """Return physical helicity for shared K blocking."""
        return self.two_K / 2

    @property
    def ladder(self) -> tuple[int, int, int]:
        """Return the internal labels preserved by the centrifugal operator."""
        return self.level_X, self.level_Y, self.two_j12


@dataclass(frozen=True)
class FSAtomAtomBasis(Sequence[FSAtomAtomChannel]):
    """Two labeled fixed-term atoms at conserved J and total parity P.

    Members:
        channels: tuple - energy-ordered channels; matrix axes follow this order
        monomer_X, monomer_Y: FSAtomBasis - experimental atomic eigenlevels
        two_J: int - twice total angular momentum
        system_parity: int - total spatial inversion eigenvalue +/-1
    """

    channels: tuple[FSAtomAtomChannel, ...]
    monomer_X: FSAtomBasis
    monomer_Y: FSAtomBasis
    two_J: int
    system_parity: int

    @property
    def n_channel(self) -> int:
        """Return channel count."""
        return len(self.channels)

    @property
    def molecule_exchange(self) -> int:
        """Return zero for labeled atomic channels."""
        return 0

    @property
    def Jtot(self) -> float:
        """Return physical total angular momentum."""
        return self.two_J / 2

    @property
    def E_int(self) -> NDArray[np.float64]:
        """Return relative thresholds in Hartree, shape (n_channel,)."""
        return np.asarray([c.E_int for c in self.channels], dtype=np.float64)

    def open_closed(self, total_energies: EnergyInput) -> OpenClosedChannels:
        """Classify each threshold at each total energy in Hartree."""
        return OpenClosedChannels(self.E_int[None, :] < get_Etot(total_energies)[:, None])

    def __len__(self) -> int:
        return self.n_channel

    @overload
    def __getitem__(self, index: int) -> FSAtomAtomChannel: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FSAtomAtomChannel, ...]: ...

    def __getitem__(self, index: int | slice) -> FSAtomAtomChannel | tuple[FSAtomAtomChannel, ...]:
        return self.channels[index]


def build_fs_atom_atom_channels(
    monomer_X: FSAtomBasis,
    monomer_Y: FSAtomBasis,
    two_J: int,
    system_parity: int,
    *,
    E_X_cut: float = np.inf,
    E_Y_cut: float = np.inf,
    two_K_cut: int | None = None,
) -> FSAtomAtomBasis:
    r"""Build normalized parity-adapted BF channels for two structured atoms.

    Formula:
        beta=(j_X,j_Y,j_12), |j_X-j_Y|<=j_12<=j_X+j_Y and
        0<=K<=min(J,j_12), with J-j_12 integral. The normalized parity pair is
        [|beta,K>+s|beta,-K>]/sqrt(2(1+delta_K0)),
        s=P epsilon_X epsilon_Y (-1)^(j_12-J).
        At K=0 only s=+1 survives. Thresholds are
        E_int=(E_X-E_X,zero)+(E_Y-E_Y,zero).

    Inputs:
        monomer_X, monomer_Y: FSAtomBasis - fixed-L,S atomic bases
        two_J: int - nonnegative twice-J; supports half-integer J
        system_parity: int - spatial parity P=+/-1
        E_X_cut, E_Y_cut: float - individual relative energy cutoffs, Hartree
        two_K_cut: int | None - optional maximum twice-helicity

    Returns:
        basis: FSAtomAtomBasis - channels sorted by energy then internal labels
    """
    if isinstance(two_J, bool) or not isinstance(two_J, int) or two_J < 0 or system_parity not in (-1, 1):
        raise ValueError("two_J must be a nonnegative integer and system_parity must be +/-1")
    if two_K_cut is not None and (isinstance(two_K_cut, bool) or not isinstance(two_K_cut, int) or two_K_cut < 0):
        raise ValueError("two_K_cut must be a nonnegative integer")
    if np.isnan(E_X_cut) or np.isnan(E_Y_cut):
        raise ValueError("Atomic energy cutoffs cannot be NaN")
    if two_J % 2 != (monomer_X.two_S + monomer_Y.two_S) % 2:
        raise ValueError("J and j_X+j_Y must have the same integer/half-integer character")
    channels = []
    for ix, jx in enumerate(monomer_X.two_j_levels):
        ex = float(monomer_X.thresholds[ix])
        if ex > E_X_cut:
            continue
        for iy, jy in enumerate(monomer_Y.two_j_levels):
            ey = float(monomer_Y.thresholds[iy])
            if ey > E_Y_cut:
                continue
            for j12 in range(abs(jx - jy), jx + jy + 1, 2):
                kmax = min(two_J, j12) if two_K_cut is None else min(two_J, j12, two_K_cut)
                for k in range(two_J % 2, kmax + 1, 2):
                    phase = system_parity * monomer_X.atom_parity * monomer_Y.atom_parity * (-1) ** ((j12 - two_J) // 2)
                    if k == 0 and phase != 1:
                        continue
                    channels.append(FSAtomAtomChannel(ix, iy, j12, k, ex + ey))
    channels.sort(key=lambda c: (c.E_int, c.level_X, c.level_Y, c.two_j12, c.two_K))
    return FSAtomAtomBasis(tuple(channels), monomer_X, monomer_Y, two_J, system_parity)
