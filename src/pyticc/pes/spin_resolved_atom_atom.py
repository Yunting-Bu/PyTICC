from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyticc.pes.spin_resolved_diatom_diatom import allowed_total_spins

ElectronicValues = NDArray[np.float64] | NDArray[np.complex128]


@dataclass(frozen=True, order=True)
class AtomAtomOrbitalState:
    """BF orbital product label a=(lambda_X,lambda_Y).

    Members:
        two_lambda_X, two_lambda_Y: int - twice signed integral projections
    """

    two_lambda_X: int
    two_lambda_Y: int


def atom_atom_orbital_states(two_L_X: int, two_L_Y: int) -> tuple[AtomAtomOrbitalState, ...]:
    """Return all signed orbital products in X-major order.

    Inputs:
        two_L_X, two_L_Y: int - nonnegative even twice-orbital angular momenta
    Returns:
        states: tuple - lambda_X ascending, then lambda_Y ascending
    """
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 or v % 2 for v in (two_L_X, two_L_Y)):
        raise ValueError("Atomic two_L values must be nonnegative even integers")
    return tuple(AtomAtomOrbitalState(x, y) for x in range(-two_L_X, two_L_X + 1, 2) for y in range(-two_L_Y, two_L_Y + 1, 2))


@dataclass(frozen=True)
class SpinResolvedAtomAtomPES:
    r"""Total-spin-resolved BF orbital diabatic interaction for two atoms.

    Formula:
        V_el(R)=sum_S P_S tensor V_orb^S(R), a=(lambda_X,lambda_Y).
        Returned W[s,a',a]=<a'|V_orb^(S_s)|a> is in Hartree, excluding
        experimental atomic thresholds. Each W is Hermitian, conserves
        lambda_X+lambda_Y and obeys simultaneous orbital-reflection symmetry
        W[-a',-a]=W[a',a] for fixed atomic L terms. This is the field-free
        parity-conserving scalar operator required by positive-K propagation.
        No spherical-harmonic expansion or adiabatic-to-diabatic transform is
        imposed on the callback; the PES provider supplies this representation.

    Members:
        interaction: Callable - scalar R in bohr -> (n_spin,n_orbital,n_orbital)
        two_total_spins: tuple[int,...] - twice total spin in callback order
        orbital_states: tuple[AtomAtomOrbitalState,...] - both matrix-axis labels
        interaction_many: Callable | None - R batch -> (n_R,n_spin,n_orbital,n_orbital)
    """

    interaction: Callable[[float], ElectronicValues]
    two_total_spins: tuple[int, ...]
    orbital_states: tuple[AtomAtomOrbitalState, ...]
    interaction_many: Callable[[NDArray[np.float64]], ElectronicValues] | None = None

    def __post_init__(self) -> None:
        if not self.two_total_spins or len(set(self.two_total_spins)) != len(self.two_total_spins):
            raise ValueError("two_total_spins must be nonempty and unique")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.two_total_spins):
            raise ValueError("two_total_spins must contain nonnegative integers")
        if not self.orbital_states or len(set(self.orbital_states)) != len(self.orbital_states):
            raise ValueError("orbital_states must be nonempty and unique")

    def validate_basis(self, two_L_X: int, two_S_X: int, two_L_Y: int, two_S_Y: int) -> None:
        """Validate complete spin and orbital manifolds against both atoms."""
        if set(self.two_total_spins) != set(allowed_total_spins(two_S_X, two_S_Y)):
            raise ValueError("PES total spins do not match the atomic spins")
        if set(self.orbital_states) != set(atom_atom_orbital_states(two_L_X, two_L_Y)):
            raise ValueError("PES orbital states do not match the atomic L manifolds")

    def evaluate(self, R: float | Sequence[float] | NDArray[np.float64]) -> ElectronicValues:
        """Evaluate and validate the orbital matrices at R in bohr.

        Inputs:
            R: float | array - positive scalar or one-dimensional radial batch
        Returns:
            values: array - (n_spin,n_orbital,n_orbital), with leading n_R for batches;
                Hartree, real symmetric or complex Hermitian
        """
        radial = np.asarray(R, dtype=np.float64)
        if radial.ndim > 1 or not np.all(np.isfinite(radial)) or np.any(radial <= 0):
            raise ValueError("R must be positive finite scalar or one-dimensional array")
        shape = (len(self.two_total_spins), len(self.orbital_states), len(self.orbital_states))
        if radial.ndim == 0:
            values = np.asarray(self.interaction(float(radial)))
            expected = shape
        else:
            expected = (radial.size, *shape)
            if not radial.size:
                return np.empty(expected)
            values = np.asarray(self.interaction_many(radial)) if self.interaction_many else np.stack([self.interaction(float(r)) for r in radial])
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(f"Atomic PES must return finite values with shape {expected}; got {values.shape}")
        if not np.allclose(values, values.swapaxes(-1, -2).conj(), atol=1e-12, rtol=0):
            raise ValueError("Atomic orbital PES must be Hermitian")
        projection = np.array([s.two_lambda_X + s.two_lambda_Y for s in self.orbital_states])
        if np.any(np.abs(values[..., projection[:, None] != projection[None, :]]) > 1e-12):
            raise ValueError("Field-free atomic PES must conserve lambda_X+lambda_Y")
        lookup = {s: i for i, s in enumerate(self.orbital_states)}
        try:
            reflected = [lookup[AtomAtomOrbitalState(-s.two_lambda_X, -s.two_lambda_Y)] for s in self.orbital_states]
        except KeyError as error:
            raise ValueError("Orbital basis must contain simultaneous reflection partners") from error
        if not np.allclose(values, values[..., reflected, :][..., :, reflected], atol=1e-12, rtol=1e-10):
            raise ValueError("Atomic orbital PES violates the specified BF reflection convention")
        return values
