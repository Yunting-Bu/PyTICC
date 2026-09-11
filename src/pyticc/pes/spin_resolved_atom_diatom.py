from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.pes.adiabatic import MonomerPES, PESWrapper
from pyticc.pes.spin_resolved_diatom_diatom import allowed_total_spins, signed_lambda_values

ElectronicValues: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
SpinResolvedInteraction = Callable[[float, NDArray[np.float64]], ElectronicValues]
SpinResolvedInteractionMany = Callable[[NDArray[np.float64], NDArray[np.float64]], ElectronicValues]
SpinResolvedInteractionManyProcesses = Callable[[NDArray[np.float64], NDArray[np.float64], int], ElectronicValues]
RadialInput: TypeAlias = float | Sequence[float] | NDArray[np.float64]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class AtomDiatomOrbitalState:
    """
    One signed orbital product state for an atom--diatom electronic PES.

    Members:
        two_lambda_atom: int - twice the signed projection lambda of the atomic
            orbital angular momentum L on the body-fixed axis; L is integral so
            lambda runs over all values from -L to L
        two_lambda_Y: int - twice the signed molecular projection Lambda of the
            diatom
    """

    two_lambda_atom: int
    two_lambda_Y: int


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpinResolvedAtomDiatomPES:
    r"""
    Total-spin-resolved orbital PES for a fine-structure atom and a
    fine-structure diatom.

    Formula:
        At each geometry q, ``interaction`` returns

        W[q,s,alpha_bra,alpha_ket]
          = <alpha_bra|V_orb^(S_s)(q)|alpha_ket>,

        where ``S_s=two_total_spins[s]/2`` and
        ``alpha=(lambda, Lambda)`` follows ``orbital_states``. Every orbital
        matrix may be real symmetric or complex Hermitian. Together with the
        total-spin projectors,

        V_el(q) = sum_s P_(S_s) tensor W^(S_s)(q).

        The callback receives coordinates with shape ``(2,n_grid)`` ordered as
        ``(r_Y,theta)`` in bohr and radians, and returns Hartree values with
        shape ``(n_grid,n_spin,n_orbital,n_orbital)``.

    Members:
        interaction: SpinResolvedInteraction - scalar-R electronic PES callback
        two_total_spins: tuple[int,...] - twice total electronic spins, defining
            the spin-surface axis
        orbital_states: tuple[AtomDiatomOrbitalState,...] - signed orbital
            product basis, defining both orbital matrix axes
        interaction_many: SpinResolvedInteractionMany | None - optional radial
            batch callback returning ``(n_R,n_grid,n_spin,n_orbital,n_orbital)``
        monomer_Y: MonomerPES | None - isolated diatom potential used to build
            its vibrational basis
    """

    interaction: SpinResolvedInteraction
    two_total_spins: tuple[int, ...] = ()
    orbital_states: tuple[AtomDiatomOrbitalState, ...] = ()
    interaction_many: SpinResolvedInteractionMany | None = None
    monomer_Y: MonomerPES | None = None
    _interaction_many_processes: SpinResolvedInteractionManyProcesses | None = field(default=None, repr=False, compare=False)
    _close: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if bool(self.two_total_spins) != bool(self.orbital_states):
            message = "two_total_spins and orbital_states must either both be supplied or both be omitted for later TOML binding"
            logger.error(message)
            raise ValueError(message)
        if not self.two_total_spins:
            return
        if len(set(self.two_total_spins)) != len(self.two_total_spins):
            message = "two_total_spins must contain unique values"
            logger.error(message)
            raise ValueError(message)
        if any(value < 0 for value in self.two_total_spins):
            message = "two_total_spins must be nonnegative"
            logger.error(message)
            raise ValueError(message)
        if not self.orbital_states or len(set(self.orbital_states)) != len(self.orbital_states):
            message = "orbital_states must contain unique signed orbital product states"
            logger.error(message)
            raise ValueError(message)

    def close(self) -> None:
        """Release persistent resources owned by the PES callback."""
        if self._close is not None:
            self._close()


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def atom_diatom_orbital_states(two_L_atom: int, two_lambda_Y_abs: int) -> tuple[AtomDiatomOrbitalState, ...]:
    """Return the canonical signed orbital product basis in atom-major order."""
    return tuple(
        AtomDiatomOrbitalState(two_lambda_atom, two_lambda_Y)
        for two_lambda_atom in range(-two_L_atom, two_L_atom + 1, 2)
        for two_lambda_Y in signed_lambda_values(two_lambda_Y_abs)
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def as_spin_resolved_atom_diatom_pes(
    pes: PESWrapper,
    *,
    two_S_atom: int,
    two_L_atom: int,
    two_S_Y: int,
    two_lambda_Y_abs: int,
) -> SpinResolvedAtomDiatomPES:
    r"""
    Promote a scalar PES to identical total-spin surfaces and orbital identity.

    Formula:
        For every allowed total spin S and signed orbital product states,

        W^(S)_(alpha',alpha)(q) = V(q) delta_(alpha'alpha).

        Since ``sum_S P_S=I_spin``, this representation is exactly the scalar
        spin-independent operator in the complete primitive electronic space.
        It is the exact interaction only for an S-state atom (L=0); for L>0 the
        physical interaction generally mixes atomic orbital projections.

    Inputs:
        pes: PESWrapper - scalar atom-diatom interaction PES
        two_S_atom: int - twice atomic electronic spin
        two_L_atom: int - twice atomic orbital angular momentum
        two_S_Y: int - twice diatom electronic spin
        two_lambda_Y_abs: int - twice absolute molecular Lambda

    Returns:
        promoted: SpinResolvedAtomDiatomPES - dense spin/orbital view
    """
    spins = allowed_total_spins(two_S_atom, two_S_Y)
    orbitals = atom_diatom_orbital_states(two_L_atom, two_lambda_Y_abs)
    identity = np.eye(len(orbitals), dtype=np.float64)

    def promote(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values[..., None, None, None] * np.broadcast_to(identity, (len(spins), *identity.shape))

    def interaction(R: float, coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
        return promote(np.asarray(pes.interaction(R, coordinates), dtype=np.float64))

    interaction_many: SpinResolvedInteractionMany | None = None
    if pes.interaction_many is not None:
        source_many = pes.interaction_many

        def evaluate_many(R: NDArray[np.float64], coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
            return promote(np.asarray(source_many(R, coordinates), dtype=np.float64))

        interaction_many = evaluate_many

    interaction_many_processes: SpinResolvedInteractionManyProcesses | None = None
    if pes._interaction_many_processes is not None:
        source_many_processes = pes._interaction_many_processes

        def evaluate_many_processes(
            R: NDArray[np.float64],
            coordinates: NDArray[np.float64],
            processes: int,
        ) -> NDArray[np.float64]:
            return promote(np.asarray(source_many_processes(R, coordinates, processes)))

        interaction_many_processes = evaluate_many_processes

    return SpinResolvedAtomDiatomPES(
        interaction=interaction,
        two_total_spins=spins,
        orbital_states=orbitals,
        interaction_many=interaction_many,
        monomer_Y=pes.monomer_Y,
        _interaction_many_processes=interaction_many_processes,
        _close=pes.close,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _validate_values(values: ElectronicValues, expected_shape: tuple[int, ...]) -> None:
    if values.shape != expected_shape:
        message = f"Spin-resolved A+BC PES returned shape {values.shape}, but expected {expected_shape}"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(values)):
        message = "Spin-resolved A+BC PES returned non-finite values"
        logger.error(message)
        raise ValueError(message)
    if not np.allclose(values, np.swapaxes(np.conjugate(values), -1, -2), rtol=0.0, atol=1.0e-12):
        message = "Every spin-resolved orbital PES matrix must be Hermitian"
        logger.error(message)
        raise ValueError(message)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _evaluate(
    pes: SpinResolvedAtomDiatomPES,
    R: RadialInput,
    coordinates: NDArray[np.float64],
    grid_shape: tuple[int, int],
    *,
    processes: int,
) -> ElectronicValues:
    """Evaluate scalar or radial-batched spin-resolved values."""
    if not pes.two_total_spins or not pes.orbital_states:
        message = "Spin-resolved A+BC PES electronic ordering has not been bound"
        logger.error(message)
        raise ValueError(message)
    if not isinstance(processes, int) or isinstance(processes, bool) or processes < 1:
        message = f"processes must be a positive integer, but got {processes!r}"
        logger.error(message)
        raise ValueError(message)
    radial_points = np.asarray(R, dtype=np.float64)
    electronic_shape = (len(pes.two_total_spins), len(pes.orbital_states), len(pes.orbital_states))
    if radial_points.ndim == 0:
        values = np.asarray(pes.interaction(float(radial_points), coordinates))
        expected_shape = (coordinates.shape[1], *electronic_shape)
        output_shape = (*grid_shape, *electronic_shape)
    elif radial_points.ndim == 1:
        if radial_points.size == 0:
            values = np.empty((0, coordinates.shape[1], *electronic_shape), dtype=np.complex128)
        elif pes._interaction_many_processes is not None:
            values = np.asarray(pes._interaction_many_processes(radial_points, coordinates, processes))
        elif pes.interaction_many is None:
            values = np.stack([pes.interaction(float(RR), coordinates) for RR in radial_points])
        else:
            values = np.asarray(pes.interaction_many(radial_points, coordinates))
        expected_shape = (radial_points.size, coordinates.shape[1], *electronic_shape)
        output_shape = (radial_points.size, *grid_shape, *electronic_shape)
    else:
        message = f"R must be scalar or one-dimensional, but got shape {radial_points.shape}"
        logger.error(message)
        raise ValueError(message)
    _validate_values(values, expected_shape)
    return values.reshape(output_shape)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def get_spin_resolved_grid_atom_diatom(
    pes: SpinResolvedAtomDiatomPES,
    R: RadialInput,
    r_Y: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    processes: int = 1,
) -> ElectronicValues:
    """Evaluate total-spin-resolved orbital matrices on the A+BC tensor grid.

    Inputs:
        pes: SpinResolvedAtomDiatomPES - electronic interaction model
        R: RadialInput - scalar separation or radial batch in bohr
        r_Y: NDArray[np.float64] - diatomic bond grid in bohr
        theta: NDArray[np.float64] - Jacobi-angle grid in radians
        processes: int - temporary worker process count

    Returns:
        values: ElectronicValues - real or complex values with shape
            ``(n_r,n_theta,n_spin,n_orb,n_orb)``, optionally preceded by ``n_R``
    """
    r_values = np.asarray(r_Y, dtype=np.float64)
    theta_values = np.asarray(theta, dtype=np.float64)
    if r_values.ndim != 1 or theta_values.ndim != 1:
        message = f"r_Y and theta must be one-dimensional, but got shapes {r_values.shape} and {theta_values.shape}"
        logger.error(message)
        raise ValueError(message)
    grids = np.meshgrid(r_values, theta_values, indexing="ij")
    coordinates = np.asfortranarray(np.stack(tuple(grid.reshape(-1) for grid in grids)))
    return _evaluate(pes, R, coordinates, grids[0].shape, processes=processes)


# ----------------------------------------------------------------------------------------
