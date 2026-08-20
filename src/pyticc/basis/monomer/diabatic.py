from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.dvr import SineDVR, build_RovibDVR, build_SineDVR
from pyticc.basis.podvr import build_RovibPODVR
from pyticc.basis.rovib import RovibBasis
from pyticc.system import MolInnerState, MonomerType


# Diabatic diatom
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiabaticDiatomState:
    """
    One diabatic electronic state's primitive and contracted bases.

    Members:
        electronic_state: int - zero-based diabatic electronic-state label
        contracted: RovibBasis - state-specific contracted radial basis
        primitive: RovibBasis - state-specific rovibrational wavefunctions on the
            common primitive grid
    """

    electronic_state: int
    contracted: RovibBasis
    primitive: RovibBasis

    def __post_init__(self) -> None:
        if self.electronic_state < 0:
            message = f"electronic_state must be non-negative, but got {self.electronic_state}"
            logger.error(message)
            raise ValueError(message)
        if self.primitive.E_vj.shape != self.contracted.E_vj.shape:
            message = f"Primitive energy shape {self.primitive.E_vj.shape} does not match contracted shape {self.contracted.E_vj.shape}"
            logger.error(message)
            raise ValueError(message)

    @property
    def vmax(self) -> int:
        """Return the largest available vibrational quantum number."""
        return self.contracted.E_vj.shape[0] - 1

    @property
    def jmax(self) -> int:
        """Return the largest available rotational quantum number."""
        return self.contracted.E_vj.shape[1] - 1


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiabaticDiatomBasis:
    """Diatomic basis spanning several diabatic electronic states."""

    states: tuple[DiabaticDiatomState, ...]
    energy_zero: float
    type = MonomerType.DIATOM

    def __post_init__(self) -> None:
        if not self.states:
            message = "At least one diabatic electronic state is required"
            logger.error(message)
            raise ValueError(message)
        if not np.isfinite(self.energy_zero):
            message = f"energy_zero must be finite, but got {self.energy_zero}"
            logger.error(message)
            raise ValueError(message)

        labels = tuple(state.electronic_state for state in self.states)
        expected = tuple(range(len(self.states)))
        if labels != expected:
            message = f"electronic_state labels must be consecutive from zero, but got {labels}"
            logger.error(message)
            raise ValueError(message)

        reference_grid = self.states[0].primitive.grids
        for state in self.states[1:]:
            if state.primitive.grids.shape != reference_grid.shape or not np.array_equal(state.primitive.grids, reference_grid):
                message = "All diabatic electronic states must use the same primitive DVR grid"
                logger.error(message)
                raise ValueError(message)

    @property
    def n_state(self) -> int:
        """Return the number of diabatic electronic states."""
        return len(self.states)

    def state(self, electronic_state: int) -> DiabaticDiatomState:
        """Return one electronic-state basis by its zero-based label."""
        if not 0 <= electronic_state < self.n_state:
            message = f"Electronic state {electronic_state} is outside 0..{self.n_state - 1}"
            logger.error(message)
            raise ValueError(message)
        return self.states[electronic_state]

    def relative_energies(self, electronic_state: int) -> NDArray[np.float64]:
        """
        Return one electronic state's energies relative to the common zero.

        Inputs:
            electronic_state: int - zero-based electronic-state label

        Returns:
            energies: NDArray[np.float64] - relative energies indexed as
                energies[v,j], shape (vmax + 1,jmax + 1)
        """
        return np.asarray(self.state(electronic_state).contracted.E_vj - self.energy_zero, dtype=np.float64)

    def mis_iter(self, E_cut: float) -> Iterator[MolInnerState]:
        """Yield retained rovibrational states from every electronic state."""
        for state in self.states:
            for v in range(state.vmax + 1):
                for j in range(state.jmax + 1):
                    energy = float(state.contracted.E_vj[v, j] - self.energy_zero)
                    if np.isfinite(energy) and energy <= E_cut:
                        yield MolInnerState(j=j, v=v, Eint=energy, electronic_state=state.electronic_state)

    def energy(self, mis: MolInnerState, K: int) -> float:
        """Return one threshold relative to the common energy zero."""
        if mis.electronic_state is None:
            message = "Diabatic diatomic inner state requires electronic_state"
            logger.error(message)
            raise ValueError(message)
        if mis.v is None:
            message = "Diabatic diatomic inner state requires v"
            logger.error(message)
            raise ValueError(message)
        return float(self.state(mis.electronic_state).contracted.E_vj[mis.v, mis.j] - self.energy_zero)

    def allows_K(self, mis: MolInnerState, K: int) -> bool:
        """Allow every helicity admitted by angular momentum coupling."""
        if mis.electronic_state is None:
            return False
        return True


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _per_state(value: int | Sequence[int], n_state: int, name: str) -> tuple[int, ...]:
    values = (value,) * n_state if isinstance(value, int) else tuple(value)
    if len(values) != n_state:
        message = f"{name} must provide one value per electronic state; expected {n_state}, got {len(values)}"
        logger.error(message)
        raise ValueError(message)
    return values


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_DiabaticDiatomBasis(
    dvrs: Sequence[SineDVR],
    *,
    n_podvr: int | Sequence[int],
    vmax: int | Sequence[int],
    jmax: int | Sequence[int],
    mass: float,
    energy_zero: float | None = None,
) -> DiabaticDiatomBasis:
    """Build state-specific diatomic bases with one common energy zero."""
    dvr_states = tuple(dvrs)
    if not dvr_states:
        message = "At least one primitive DVR state is required"
        logger.error(message)
        raise ValueError(message)

    n_state = len(dvr_states)
    n_podvr_values = _per_state(n_podvr, n_state, "n_podvr")
    vmax_values = _per_state(vmax, n_state, "vmax")
    jmax_values = _per_state(jmax, n_state, "jmax")
    rovib_states = tuple(
        build_RovibPODVR(dvr, n_po, vmax_state, jmax_state, mass)
        for dvr, n_po, vmax_state, jmax_state in zip(dvr_states, n_podvr_values, vmax_values, jmax_values, strict=True)
    )
    primitive_states = tuple(
        build_RovibDVR(dvr, vmax_state, jmax_state, mass) for dvr, vmax_state, jmax_state in zip(dvr_states, vmax_values, jmax_values, strict=True)
    )
    zero = float(rovib_states[0].E_vj[0, 0]) if energy_zero is None else float(energy_zero)
    if not np.isfinite(zero):
        message = f"energy_zero must be finite, but got {zero}"
        logger.error(message)
        raise ValueError(message)

    states = tuple(
        DiabaticDiatomState(
            electronic_state=index,
            contracted=rovib,
            primitive=rovib_dvr,
        )
        for index, (rovib, rovib_dvr) in enumerate(zip(rovib_states, primitive_states, strict=True))
    )
    return DiabaticDiatomBasis(states=states, energy_zero=zero)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare_DiabaticDiatom(
    potential: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    n_state: int,
    r: tuple[float, float],
    n_dvr: int,
    n_podvr: int | Sequence[int],
    vmax: int | Sequence[int],
    jmax: int | Sequence[int],
    mass: float,
    energy_zero: float | None = None,
) -> DiabaticDiatomBasis:
    """
    Prepare a diabatic diatom through the DVR and PODVR steps.

    Inputs:
        potential: Callable - all-state monomer PES mapping bond-length grids with
            shape (n_dvr,) to energies with shape (n_dvr,n_state)
        n_state: int - number of diabatic electronic states
        r: tuple[float,float] - common primitive-DVR boundaries
        n_dvr: int - number of common primitive-DVR points
        n_podvr: int | Sequence[int] - contracted-grid sizes by electronic state
        vmax: int | Sequence[int] - largest vibrational quantum numbers by state
        jmax: int | Sequence[int] - largest rotational quantum numbers by state
        mass: float - diatomic reduced mass in atomic units
        energy_zero: float | None - common absolute channel-energy zero

    Returns:
        monomer: DiabaticDiatomBasis - prepared multi-state rovibrational basis
    """
    if n_state < 1:
        message = f"n_state must be positive, but got {n_state}"
        logger.error(message)
        raise ValueError(message)

    cached_grids: NDArray[np.float64] | None = None
    cached_values: NDArray[np.float64] | None = None

    def evaluate_all(grids: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal cached_grids, cached_values
        if cached_grids is None or not np.array_equal(grids, cached_grids):
            values = np.asarray(potential(grids), dtype=np.float64)
            expected = (grids.size, n_state)
            if values.shape != expected:
                message = f"Diabatic monomer potential returned shape {values.shape}, but expected {expected}"
                logger.error(message)
                raise ValueError(message)
            if not np.all(np.isfinite(values)):
                message = "Diabatic monomer potential returned non-finite values"
                logger.error(message)
                raise ValueError(message)
            cached_grids = grids.copy()
            cached_values = values
        assert cached_values is not None
        return cached_values

    def state_potential(electronic_state: int) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        def evaluate(grids: NDArray[np.float64]) -> NDArray[np.float64]:
            return evaluate_all(grids)[:, electronic_state]

        return evaluate

    dvrs = tuple(build_SineDVR(r[0], r[1], n_dvr, mass, state_potential(state)) for state in range(n_state))
    return build_DiabaticDiatomBasis(
        dvrs,
        n_podvr=n_podvr,
        vmax=vmax,
        jmax=jmax,
        mass=mass,
        energy_zero=energy_zero,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    from pyticc.constants import AU2CM
    from pyticc.pes import load_fortran_diabatic_pes

    pes_dir = Path(__file__).resolve().parents[4] / "example" / "HO2_diabatic" / "pes"
    oxygen_reduced_mass = 15.99492 * 1822.88853 / 2.0
    pes = load_fortran_diabatic_pes(
        [pes_dir / "ho2-dpme.f", pes_dir / "long_range_H_O2.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
        lapack=True,
    )
    try:
        ho2_basis = prepare_DiabaticDiatom(
            pes.monomer_values,
            n_state=pes.n_state,
            r=(1.2, 5.0),
            n_dvr=135,
            n_podvr=80,
            vmax=0,
            jmax=(3, 2),
            mass=oxygen_reduced_mass,
        )
        labels = ("D0", "state 1: v=0, j=1", "state 1: v=0, j=3", "state 2: v=0, j=0", "state 2: v=0, j=2")
        calculated = (
            np.array(
                [
                    ho2_basis.energy_zero,
                    ho2_basis.relative_energies(0)[0, 1],
                    ho2_basis.relative_energies(0)[0, 3],
                    ho2_basis.relative_energies(1)[0, 0],
                    ho2_basis.relative_energies(1)[0, 2],
                ]
            )
            * AU2CM
        )
        reference = np.array([784.9778, 2.8631, 17.1778, 7756.3073, 7764.7756])
        np.testing.assert_allclose(calculated, reference, atol=5.0e-5)

        print("HO2 diabatic rovibrational levels (cm-1)")
        print("level                         PyTICC       reference")
        for label, value, expected in zip(labels, calculated, reference, strict=True):
            print(f"{label:27s} {value:12.6f} {expected:12.6f}")
        print("All HO2 level checks passed.")
    finally:
        pes.close()
# ----------------------------------------------------------------------------------------
