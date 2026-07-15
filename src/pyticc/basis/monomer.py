from dataclasses import dataclass

from numpy.typing import NDArray

from pyticc.system import MolInnerState, MonomerType


# ----------------------------------------------------------------------------------------
def set_j_parity(jpar: int) -> tuple[int, int]:
    """Return the first allowed j and increment for a rotational parity."""
    try:
        return {-1: (1, 2), 0: (0, 1), 1: (0, 2)}[jpar]
    except KeyError as error:
        raise ValueError("jpar must be -1, 0, or 1") from error


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
class AtomSpec:
    type = MonomerType.ATOM
    jpar: int = 0

    def mis_iter(self, E_cut: float):
        yield MolInnerState(j=0, Eint=0.0)

    def energy(self, mis: MolInnerState, K: int) -> float:
        return 0.0


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiatomSpec:
    type = MonomerType.DIATOM
    Eint: NDArray
    vmax: int
    jmax: int
    vmin: int = 0
    jpar: int = 0

    @property
    def jmin(self) -> int:
        return set_j_parity(self.jpar)[0]

    @property
    def jinc(self) -> int:
        return set_j_parity(self.jpar)[1]

    def mis_iter(self, E_cut: float):
        for v in range(self.vmin, self.vmax + 1):
            for j in range(self.jmin, self.jmax + 1, self.jinc):
                if self.Eint[v, j] <= E_cut:
                    yield MolInnerState(j=j, v=v, Eint=self.Eint[v, j])

    def energy(self, mis: MolInnerState, K: int) -> float:
        if mis.v is None:
            raise ValueError("Diatomic inner state requires v")
        return self.Eint[mis.v, mis.j]


# ----------------------------------------------------------------------------------------
