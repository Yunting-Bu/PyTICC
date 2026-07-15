import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Protocol

from pyticc.system import MolInnerState, ScattSystem


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
    """Monomer-energy and helicity truncations in atomic units."""

    E_X_cut: float = math.inf
    E_Y_cut: float = math.inf
    K_cut: int | None = None

    def __post_init__(self) -> None:
        if self.E_X_cut < 0.0 or self.E_Y_cut < 0.0:
            raise ValueError("Energy cutoffs must be non-negative")
        if self.K_cut is not None and self.K_cut < 0:
            raise ValueError("K_cut must be non-negative")


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
        v_X = "-" if self.mis_X.v is None else str(self.mis_X.v)
        v_Y = "-" if self.mis_Y.v is None else str(self.mis_Y.v)
        return (
            f"Channel[{self.index}] "
            f"X(v={v_X}, j={self.mis_X.j}) "
            f"Y(v={v_Y}, j={self.mis_Y.j}) "
            f"j_couple={self.j_couple} K={self.K} Jtot={self.Jtot} "
            f"parity={self.system_parity:+d} E_int={self.E_int:.10f} a.u."
        )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
class ParityRule(Protocol):
    def allow_K0(self, mis_X: MolInnerState, mis_Y: MolInnerState, j_couple: int) -> bool: ...


@dataclass(frozen=True)
class ClosedShellParity:
    system_parity: int
    Jtot: int

    def allow_K0(self, mis_X: MolInnerState, mis_Y: MolInnerState, j_couple: int) -> bool:
        phase = self.system_parity * (-1) ** (mis_X.j + mis_Y.j + j_couple + self.Jtot)
        return phase == 1


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelBuilder:
    sys: ScattSystem
    trunc: TruncSpec

    def build(self) -> list[Channel]:
        if self.sys.Jtot is None or self.sys.system_parity is None:
            raise ValueError("Field-free channel construction requires Jtot and system_parity")

        parity = ClosedShellParity(self.sys.system_parity, self.sys.Jtot)
        channels: list[Channel] = []

        for mis_X in self.sys.monomer_X.mis_iter(self.trunc.E_X_cut):
            for mis_Y in self.sys.monomer_Y.mis_iter(self.trunc.E_Y_cut):
                for j_couple in range(abs(mis_X.j - mis_Y.j), mis_X.j + mis_Y.j + 1):
                    Kmax = set_Kmax(j_couple, self.sys.Jtot, self.trunc.K_cut)
                    for K in range(Kmax + 1):
                        if K == 0 and not parity.allow_K0(mis_X, mis_Y, j_couple):
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
        return [replace(channel, index=index) for index, channel in enumerate(channels)]

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    from pyticc.basis.monomer import AtomSpec, DiatomSpec
    from pyticc.system import ScattSystem

    def A_plus_BC() -> None:
        atom = AtomSpec()
        diatom = DiatomSpec(Eint=np.array([[0.0, 0.01, 0.03]]), vmax=0, jmax=2)

        system = ScattSystem(monomer_X=atom, monomer_Y=diatom, Jtot=1, system_parity=-1)
        channels = ChannelBuilder(system, TruncSpec()).build()

        for channel in channels:
            print(channel)

    def AB_plus_CD() -> None:
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
