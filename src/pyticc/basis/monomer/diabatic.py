from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from loguru import logger

from pyticc.basis.dvr import RovibDVR, SineDVR, build_RovibDVR
from pyticc.basis.monomer.diatom import DiatomSpec
from pyticc.basis.podvr import RovibPODVR, build_RovibPODVR
from pyticc.system import MolInnerState, MonomerType


@dataclass(frozen=True)
class DiabaticDiatomState:
    """One diabatic electronic state's primitive and contracted bases."""

    electronic_state: int
    dvr: SineDVR
    rovib: RovibPODVR
    rovib_dvr: RovibDVR
    spec: DiatomSpec

    def __post_init__(self) -> None:
        if self.electronic_state < 0:
            message = f"electronic_state must be non-negative, but got {self.electronic_state}"
            logger.error(message)
            raise ValueError(message)
        if self.rovib.E_vj.shape != self.spec.Eint.shape:
            message = f"Rovibrational energy shape {self.rovib.E_vj.shape} does not match DiatomSpec shape {self.spec.Eint.shape}"
            logger.error(message)
            raise ValueError(message)
        if self.rovib_dvr.E_vj.shape != self.rovib.E_vj.shape:
            message = f"Primitive-DVR energy shape {self.rovib_dvr.E_vj.shape} does not match PODVR shape {self.rovib.E_vj.shape}"
            logger.error(message)
            raise ValueError(message)
        if not np.array_equal(self.rovib_dvr.grids, self.dvr.grids):
            message = "Primitive rovibrational and SineDVR grids must match"
            logger.error(message)
            raise ValueError(message)


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

        reference_dvr = self.states[0].dvr
        for state in self.states[1:]:
            if state.dvr.grids.shape != reference_dvr.grids.shape or not np.array_equal(state.dvr.grids, reference_dvr.grids):
                message = "All diabatic electronic states must use the same primitive DVR grid"
                logger.error(message)
                raise ValueError(message)

    @property
    def n_state(self) -> int:
        """Return the number of diabatic electronic states."""
        return len(self.states)

    @property
    def rotational_parities(self) -> tuple[int, ...]:
        """Return the rotational parity selector for every electronic state."""
        return tuple(state.spec.jpar for state in self.states)

    def state(self, electronic_state: int) -> DiabaticDiatomState:
        """Return one electronic-state basis by its zero-based label."""
        if not 0 <= electronic_state < self.n_state:
            message = f"Electronic state {electronic_state} is outside 0..{self.n_state - 1}"
            logger.error(message)
            raise ValueError(message)
        return self.states[electronic_state]

    def mis_iter(self, E_cut: float):
        """Yield retained rovibrational states from every electronic state."""
        for state in self.states:
            for inner_state in state.spec.mis_iter(E_cut):
                yield replace(inner_state, electronic_state=state.electronic_state)

    def energy(self, mis: MolInnerState, K: int) -> float:
        """Return one threshold relative to the common energy zero."""
        if mis.electronic_state is None:
            message = "Diabatic diatomic inner state requires electronic_state"
            logger.error(message)
            raise ValueError(message)
        return self.state(mis.electronic_state).spec.energy(mis, K)

    def allows_K(self, mis: MolInnerState, K: int) -> bool:
        """Allow every helicity admitted by angular momentum coupling."""
        if mis.electronic_state is None:
            return False
        return self.state(mis.electronic_state).spec.allows_K(mis, K)


def _per_state(value: int | Sequence[int], n_state: int, name: str) -> tuple[int, ...]:
    values = (value,) * n_state if isinstance(value, int) else tuple(value)
    if len(values) != n_state:
        message = f"{name} must provide one value per electronic state; expected {n_state}, got {len(values)}"
        logger.error(message)
        raise ValueError(message)
    return values


def build_DiabaticDiatomBasis(
    dvrs: Sequence[SineDVR],
    *,
    n_podvr: int | Sequence[int],
    vmax: int | Sequence[int],
    jmax: int | Sequence[int],
    mass: float,
    vmin: int | Sequence[int] = 0,
    jpar: int | Sequence[int] = 0,
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
    vmin_values = _per_state(vmin, n_state, "vmin")
    jpar_values = _per_state(jpar, n_state, "jpar")
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
            dvr=dvr,
            rovib=rovib,
            rovib_dvr=rovib_dvr,
            spec=DiatomSpec(
                Eint=np.asarray(rovib.E_vj - zero, dtype=np.float64),
                vmax=vmax_state,
                jmax=jmax_state,
                vmin=vmin_state,
                jpar=jpar_state,
            ),
        )
        for index, (dvr, rovib, rovib_dvr, vmax_state, jmax_state, vmin_state, jpar_state) in enumerate(
            zip(dvr_states, rovib_states, primitive_states, vmax_values, jmax_values, vmin_values, jpar_values, strict=True)
        )
    )
    return DiabaticDiatomBasis(states=states, energy_zero=zero)


if __name__ == "__main__":
    from pathlib import Path

    from pyticc.basis.dvr import build_SineDVR
    from pyticc.constants import AU2CM
    from pyticc.pes import load_fortran_diabatic_pes

    pes_dir = Path(__file__).resolve().parents[4] / "example" / "dia_HO2" / "pes"
    oxygen_reduced_mass = 15.99492 * 1822.88853 / 2.0
    pes = load_fortran_diabatic_pes(
        [pes_dir / "ho2-dpme.f", pes_dir / "long_range_H_O2.f"],
        pes_dir / "pyticc_wrapper.f90",
        workdir=pes_dir,
    )
    try:
        ho2_dvrs = tuple(build_SineDVR(1.2, 5.0, 135, oxygen_reduced_mass, pes.monomer_state(state)) for state in range(2))
        ho2_basis = build_DiabaticDiatomBasis(
            ho2_dvrs,
            n_podvr=80,
            vmax=0,
            jmax=(3, 2),
            mass=oxygen_reduced_mass,
            jpar=(-1, 1),
        )
        labels = ("D0", "state 1: v=0, j=1", "state 1: v=0, j=3", "state 2: v=0, j=0", "state 2: v=0, j=2")
        calculated = (
            np.array(
                [
                    ho2_basis.energy_zero,
                    ho2_basis.state(0).spec.Eint[0, 1],
                    ho2_basis.state(0).spec.Eint[0, 3],
                    ho2_basis.state(1).spec.Eint[0, 0],
                    ho2_basis.state(1).spec.Eint[0, 2],
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
