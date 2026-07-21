from pyticc.system import MolInnerState, MonomerType


class AtomSpec:
    """Structureless atomic monomer with one zero-energy internal state."""

    type = MonomerType.ATOM
    jpar: int = 0

    def mis_iter(self, E_cut: float):
        """Yield the atom's only internal state."""
        yield MolInnerState(j=0, Eint=0.0)

    def energy(self, mis: MolInnerState, K: int) -> float:
        """Return the zero internal energy of a structureless atom."""
        return 0.0

    def allows_K(self, mis: MolInnerState, K: int) -> bool:
        """Allow every system helicity because the atomic monomer has j=0."""
        return True
