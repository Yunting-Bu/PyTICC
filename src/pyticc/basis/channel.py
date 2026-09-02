from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import cast, overload

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.monomer.diatom_electric import DiatomElectricBasis
from pyticc.energy import EnergyInput, get_Etot
from pyticc.system import ChannelSpec, MolInnerState, MonomerSpec, MonomerType, ScattSystem


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
class Channel:
    """
    One field-free body-fixed scattering channel.

    Members:
        mis_X: MolInnerState - internal state of monomer X
        mis_Y: MolInnerState - internal state of monomer Y
        j_couple: int - coupled monomer angular momentum
        K: int - body-fixed helicity
        E_int: float - total channel threshold in atomic units
    """

    mis_X: MolInnerState
    mis_Y: MolInnerState
    j_couple: int
    K: int
    E_int: float

    def __str__(self) -> str:
        qn_X = f"t={self.mis_X.t}" if self.mis_X.t is not None else f"v={'-' if self.mis_X.v is None else self.mis_X.v}"
        qn_Y = f"t={self.mis_Y.t}" if self.mis_Y.t is not None else f"v={'-' if self.mis_Y.v is None else self.mis_Y.v}"
        electronic_X = "" if self.mis_X.electronic_state is None else f"e={self.mis_X.electronic_state}, "
        electronic_Y = "" if self.mis_Y.electronic_state is None else f"e={self.mis_Y.electronic_state}, "
        return (
            f"X({electronic_X}{qn_X}, j={self.mis_X.j}) "
            f"Y({electronic_Y}{qn_Y}, j={self.mis_Y.j}) "
            f"j_couple={self.j_couple} K={self.K} E_int={self.E_int:.10f} a.u."
        )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OpenClosedChannels:
    """
    Open and closed channel information over a total-energy grid.

    Members:
        open_mask: NDArray[np.bool_] - open-channel mask with shape (n_energy, n_channel)
    """

    open_mask: NDArray[np.bool_]

    @property
    def n_open(self) -> NDArray[np.int64]:
        """Return the number of open channels at each energy, shape (n_energy,)."""
        return np.asarray(np.sum(self.open_mask, axis=1), dtype=np.int64)

    @property
    def n_closed(self) -> NDArray[np.int64]:
        """Return the number of closed channels at each energy, shape (n_energy,)."""
        return np.asarray(self.open_mask.shape[1] - self.n_open, dtype=np.int64)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelBasis(Sequence[Channel]):
    """
    Complete channel basis for one field-free scattering block.

    Members:
        channels: tuple[Channel, ...] - channels ordered by increasing internal energy
        Jtot: int - conserved total angular momentum
        system_parity: int - conserved total parity, -1 or 1
        channel_spec: ChannelSpec - selections used to construct the basis
        n_channel: int - total number of channels
    """

    channels: tuple[Channel, ...]
    Jtot: int
    system_parity: int
    channel_spec: ChannelSpec = field(default_factory=ChannelSpec)

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
            result: OpenClosedChannels - open-channel mask with shape
                (n_energy,n_channel), with derived counts of shape (n_energy,)
        """
        energies = get_Etot(total_energies)
        open_mask = self.E_int[np.newaxis, :] < energies[:, np.newaxis]
        return OpenClosedChannels(open_mask=open_mask)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelElectricSF:
    r"""
    One electric-field atom-diatom channel in the space-fixed representation.

    Formula:
        The channel function is

        |eta; M> = |phi_{alpha m}> |l m_l>,

        where

        M = m + m_l,    |m_l| <= l.

        M is a property of ChannelBasisElectricSF rather than one monomer state.

    Members:
        alpha: int - zero-based dressed-monomer eigenstate index within the
            fixed-m block
        m: int - SF projection of the dressed diatomic angular momentum
        l: int - end-over-end angular momentum
        m_l: int - SF projection of the end-over-end angular momentum
        E_int: float - channel internal energy relative to the common monomer
            energy zero, in atomic units
    """

    alpha: int
    m: int
    l: int  # noqa: E741 - l is the conventional end-over-end angular momentum.
    m_l: int
    E_int: float

    def __str__(self) -> str:
        return f"alpha={self.alpha} m={self.m} l={self.l} m_l={self.m_l} E_int={self.E_int:.10f} a.u."


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelBasisElectricSF(Sequence[ChannelElectricSF]):
    r"""
    Complete channel basis for one electric-field SF scattering block.

    Formula:
        Every retained channel eta satisfies

        M = m_eta + m_{l,eta}.

        Channels are ordered by increasing internal threshold energy
        epsilon_{alpha m}.

    Members:
        channels: tuple[ChannelElectricSF, ...] - energy-ordered
            electric-field SF channels
        M: int - conserved total projection on the electric-field axis
    """

    channels: tuple[ChannelElectricSF, ...]
    M: int

    @property
    def n_channel(self) -> int:
        """Return the number of SF channels in this basis."""
        return len(self.channels)

    @property
    def E_int(self) -> NDArray[np.float64]:
        """Return channel internal energies in atomic units, shape (n_channel,)."""
        return np.asarray([channel.E_int for channel in self.channels], dtype=np.float64)

    @property
    def m_values(self) -> tuple[int, ...]:
        """Return the retained monomer projections in ascending order."""
        return tuple(sorted({channel.m for channel in self.channels}))

    def __len__(self) -> int:
        return self.n_channel

    @overload
    def __getitem__(self, index: int) -> ChannelElectricSF: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ChannelElectricSF, ...]: ...

    def __getitem__(self, index: int | slice) -> ChannelElectricSF | tuple[ChannelElectricSF, ...]:
        return self.channels[index]

    def open_closed(self, total_energies: EnergyInput) -> OpenClosedChannels:
        """
        Classify SF channels as open or closed at each total energy.

        Inputs:
            total_energies: EnergyInput - total-energy array with shape
                (n_energy,), or a one-column text file in atomic units

        Returns:
            result: OpenClosedChannels - open-channel mask with shape
                (n_energy,n_channel), with derived counts of shape (n_energy,)
        """
        energies = get_Etot(total_energies)
        open_mask = self.E_int[np.newaxis, :] < energies[:, np.newaxis]
        return OpenClosedChannels(open_mask=open_mask)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_ChannelBasisElectricSF(
    monomer_basis: DiatomElectricBasis,
    *,
    M: int,
    lmax: int,
    channel: ChannelSpec | None = None,
) -> ChannelBasisElectricSF:
    r"""
    Build one energy-ordered electric-field SF channel basis.

    Formula:
        For each

        l = 0, ..., lmax,    m_l = -l, ..., l,

        the monomer projection is determined rather than independently
        enumerated:

        m = M - m_l.

        A channel is retained when |m| <= jmax and

        epsilon_{alpha m}
          = E_{alpha m} - E_zero
          <= E_cut.

        No field-free total-J or parity restriction is applied. A dc electric
        field aligned with SF-Z conserves M, but in general mixes j and parity.

    Inputs:
        monomer_basis: DiatomElectricBasis - electric-field-dressed monomer
            eigenstates grouped by fixed m
        M: int - conserved total SF projection
        lmax: int - largest retained end-over-end angular momentum
        channel: ChannelSpec | None - channel-energy selection; None uses the
            default untruncated specification

    Returns:
        basis: ChannelBasisElectricSF - channels ordered by increasing E_int
    """
    channel_spec = ChannelSpec() if channel is None else channel
    channels: list[ChannelElectricSF] = []
    for ell in range(lmax + 1):
        for m_l in range(-ell, ell + 1):
            m = M - m_l
            if abs(m) > monomer_basis.jmax:
                continue
            for alpha, E_int in enumerate(monomer_basis.relative_energies(m)):
                if E_int <= channel_spec.E_Y_cut:
                    channels.append(
                        ChannelElectricSF(
                            alpha=alpha,
                            m=m,
                            l=ell,
                            m_l=m_l,
                            E_int=float(E_int),
                        )
                    )

    channels.sort(key=lambda channel: (channel.E_int, channel.l, channel.m_l, channel.alpha))
    return ChannelBasisElectricSF(channels=tuple(channels), M=M)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _allow_closed_shell_K0(
    mis_X: MolInnerState,
    mis_Y: MolInnerState,
    j_couple: int,
    Jtot: int,
    system_parity: int,
) -> bool:
    """Return whether one coupled monomer state belongs to the K=0 parity block."""
    return system_parity * (-1) ** (mis_X.j + mis_Y.j + j_couple + Jtot) == 1


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _state_values(values: int | tuple[int, ...], n_state: int, name: str) -> tuple[int, ...]:
    """Expand one channel selection to one value per electronic state."""
    selected = (values,) * n_state if isinstance(values, int) else values
    if len(selected) != n_state:
        message = f"{name} must provide one value per electronic state; expected {n_state}, got {len(selected)}"
        logger.error(message)
        raise ValueError(message)
    return selected


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _selected_diatom_states(
    monomer: MonomerSpec,
    E_cut: float,
    vmin: int | tuple[int, ...],
    exchange_parity: int | tuple[int, ...],
) -> tuple[MolInnerState, ...]:
    """Return full monomer states filtered by channel vibrational and rotational symmetry selections."""
    states = tuple(monomer.mis_iter(E_cut))
    if monomer.type is not MonomerType.DIATOM:
        return states

    n_state = int(getattr(monomer, "n_state", 1))
    vmin_values = _state_values(vmin, n_state, "vmin")
    parity_values = _state_values(exchange_parity, n_state, "exchange_parity")
    selected: list[MolInnerState] = []
    for state in states:
        electronic_state = 0 if state.electronic_state is None else state.electronic_state
        if state.v is None:
            message = "Diatomic channel state requires a vibrational quantum number"
            logger.error(message)
            raise ValueError(message)
        if state.v < vmin_values[electronic_state]:
            continue
        parity = parity_values[electronic_state]
        if parity != 0 and (-1) ** state.j != parity:
            continue
        selected.append(state)
    return tuple(selected)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_ChannelBasis(system: ScattSystem, channel: ChannelSpec | None = None) -> ChannelBasis:
    """
    Construct and energy-order one field-free channel basis.

    Inputs:
        system: ScattSystem - field-free monomer bases and conserved block
            quantum numbers
        channel: ChannelSpec | None - vibrational, exchange-parity, energy, and
            helicity selections; None uses ``system.channel_spec`` or defaults

    Returns:
        basis: ChannelBasis - channels allowed by angular momentum, parity,
            energy, and helicity
    """
    if isinstance(system.monomer_Y, DiatomElectricBasis):
        message = "Field-free channel construction does not accept a dressed electric monomer basis"
        logger.error(message)
        raise TypeError(message)
    monomer_X = cast(MonomerSpec, system.monomer_X)
    monomer_Y = cast(MonomerSpec, system.monomer_Y)
    if system.Jtot is None or system.system_parity is None:
        message = "Field-free channel construction requires Jtot and system_parity"
        logger.error(message)
        raise ValueError(message)

    channel_spec = channel if channel is not None else system.channel_spec
    if channel_spec is None:
        channel_spec = ChannelSpec()
    monomer_types = (monomer_X.type, monomer_Y.type)
    atom_triatom = monomer_types in (
        (MonomerType.ATOM, MonomerType.TRIATOM),
        (MonomerType.TRIATOM, MonomerType.ATOM),
    )
    parity_block_sign = system.system_parity * (-1) ** system.Jtot
    if atom_triatom:
        triatom = monomer_X if monomer_X.type is MonomerType.TRIATOM else monomer_Y
        if getattr(triatom, "parity_block_sign", parity_block_sign) != parity_block_sign:
            message = "Triatomic basis parity_block_sign does not match system_parity*(-1)^Jtot"
            logger.error(message)
            raise ValueError(message)
    channels: list[Channel] = []

    states_X = _selected_diatom_states(
        monomer_X,
        channel_spec.E_X_cut,
        channel_spec.vmin_X,
        channel_spec.exchange_parity_X,
    )
    states_Y = _selected_diatom_states(
        monomer_Y,
        channel_spec.E_Y_cut,
        channel_spec.vmin_Y,
        channel_spec.exchange_parity_Y,
    )
    for mis_X in states_X:
        for mis_Y in states_Y:
            for j_couple in range(abs(mis_X.j - mis_Y.j), mis_X.j + mis_Y.j + 1):
                Kmax = set_Kmax(j_couple, system.Jtot, channel_spec.K_cut)
                for K in range(Kmax + 1):
                    if not monomer_X.allows_K(mis_X, K) or not monomer_Y.allows_K(mis_Y, K):
                        continue
                    if K == 0 and not atom_triatom:
                        if not _allow_closed_shell_K0(mis_X, mis_Y, j_couple, system.Jtot, system.system_parity):
                            continue

                    E_int = float(monomer_X.energy(mis_X, K) + monomer_Y.energy(mis_Y, K))
                    channels.append(
                        Channel(
                            mis_X=mis_X,
                            mis_Y=mis_Y,
                            j_couple=j_couple,
                            K=K,
                            E_int=E_int,
                        )
                    )

    channels.sort(key=lambda channel: channel.E_int)
    return ChannelBasis(
        channels=tuple(channels),
        Jtot=system.Jtot,
        system_parity=system.system_parity,
        channel_spec=channel_spec,
    )


# ----------------------------------------------------------------------------------------
