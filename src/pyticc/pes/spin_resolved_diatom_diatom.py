from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.pes.adiabatic import PESWrapper

ElectronicValues: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
SpinResolvedInteraction = Callable[[float, NDArray[np.float64]], ElectronicValues]
SpinResolvedInteractionMany = Callable[[NDArray[np.float64], NDArray[np.float64]], ElectronicValues]
SpinResolvedInteractionManyProcesses = Callable[[NDArray[np.float64], NDArray[np.float64], int], ElectronicValues]
RadialInput: TypeAlias = float | Sequence[float] | NDArray[np.float64]


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class OrbitalState:
    """
    One signed-Lambda product state for a diatom--diatom electronic PES.

    Members:
        two_lambda_X: int - twice signed Lambda_X of the first diatom
        two_lambda_Y: int - twice signed Lambda_Y of the second diatom
    """

    two_lambda_X: int
    two_lambda_Y: int


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpinResolvedDiatomDiatomPES:
    r"""
    Total-spin-resolved orbital PES for two fine-structure diatoms.

    Formula:
        At each geometry q, ``interaction`` returns

        W[q,s,alpha_bra,alpha_ket]
          = <alpha_bra|V_orb^(S_s)(q)|alpha_ket>,

        where ``S_s=two_total_spins[s]/2`` and
        ``alpha=(Lambda_X,Lambda_Y)`` follows ``orbital_states``. Every orbital
        matrix may be real symmetric or complex Hermitian. Together with the
        total-spin projectors,

        V_el(q) = sum_s P_(S_s) tensor W^(S_s)(q).

        The callback receives coordinates with shape ``(5,n_grid)`` ordered as
        ``(r_X,r_Y,theta_X,theta_Y,phi)`` in bohr and radians, and returns
        Hartree values with shape ``(n_grid,n_spin,n_orbital,n_orbital)``.

    Members:
        interaction: SpinResolvedInteraction - scalar-R electronic PES callback
        two_total_spins: tuple[int,...] - twice total electronic spins, defining
            the spin-surface axis
        orbital_states: tuple[OrbitalState,...] - signed-Lambda product basis,
            defining both orbital matrix axes
        interaction_many: SpinResolvedInteractionMany | None - optional radial
            batch callback returning
            ``(n_R,n_grid,n_spin,n_orbital,n_orbital)``
    """

    interaction: SpinResolvedInteraction
    two_total_spins: tuple[int, ...]
    orbital_states: tuple[OrbitalState, ...]
    interaction_many: SpinResolvedInteractionMany | None = None
    _interaction_many_processes: SpinResolvedInteractionManyProcesses | None = field(default=None, repr=False, compare=False)
    _close: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.two_total_spins or len(set(self.two_total_spins)) != len(self.two_total_spins):
            message = "two_total_spins must contain unique values"
            logger.error(message)
            raise ValueError(message)
        if any(value < 0 for value in self.two_total_spins):
            message = "two_total_spins must be nonnegative"
            logger.error(message)
            raise ValueError(message)
        if not self.orbital_states or len(set(self.orbital_states)) != len(self.orbital_states):
            message = "orbital_states must contain unique signed-Lambda product states"
            logger.error(message)
            raise ValueError(message)

    def close(self) -> None:
        """Release persistent resources owned by the PES callback."""
        if self._close is not None:
            self._close()


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def signed_lambda_values(two_lambda_abs: int) -> tuple[int, ...]:
    """Return the signed doubled Lambda values for one electronic manifold."""
    if two_lambda_abs < 0:
        message = f"two_lambda_abs must be nonnegative, but got {two_lambda_abs}"
        logger.error(message)
        raise ValueError(message)
    return (0,) if two_lambda_abs == 0 else (-two_lambda_abs, two_lambda_abs)


# ----------------------------------------------------------------------------------------
def orbital_product_states(two_lambda_X_abs: int, two_lambda_Y_abs: int) -> tuple[OrbitalState, ...]:
    """Return the canonical signed-Lambda product basis in X-major order."""
    return tuple(
        OrbitalState(two_lambda_X, two_lambda_Y)
        for two_lambda_X in signed_lambda_values(two_lambda_X_abs)
        for two_lambda_Y in signed_lambda_values(two_lambda_Y_abs)
    )


# ----------------------------------------------------------------------------------------
def allowed_total_spins(two_S_X: int, two_S_Y: int) -> tuple[int, ...]:
    """Return all allowed doubled total spins from coupling S_X and S_Y."""
    if two_S_X < 0 or two_S_Y < 0:
        message = "Doubled monomer spins must be nonnegative"
        logger.error(message)
        raise ValueError(message)
    return tuple(range(abs(two_S_X - two_S_Y), two_S_X + two_S_Y + 1, 2))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def as_spin_resolved_diatom_diatom_pes(
    pes: PESWrapper,
    *,
    two_S_X: int,
    two_lambda_X_abs: int,
    two_S_Y: int,
    two_lambda_Y_abs: int,
) -> SpinResolvedDiatomDiatomPES:
    r"""
    Promote a scalar PES to identical total-spin surfaces and orbital identity.

    Formula:
        For every allowed total spin S and signed-Lambda product states,

        W^(S)_(alpha',alpha)(q) = V(q) delta_(alpha'alpha).

        Since ``sum_S P_S=I_spin``, this representation is exactly the scalar
        spin-independent operator in the complete primitive electronic space.

    Inputs:
        pes: PESWrapper - scalar AB+CD interaction PES
        two_S_X: int - twice spin S_X
        two_lambda_X_abs: int - twice absolute Lambda_X
        two_S_Y: int - twice spin S_Y
        two_lambda_Y_abs: int - twice absolute Lambda_Y

    Returns:
        promoted: SpinResolvedDiatomDiatomPES - dense spin/orbital view
    """
    spins = allowed_total_spins(two_S_X, two_S_Y)
    orbitals = orbital_product_states(two_lambda_X_abs, two_lambda_Y_abs)
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
            return promote(np.asarray(source_many_processes(R, coordinates, processes), dtype=np.float64))

        interaction_many_processes = evaluate_many_processes

    return SpinResolvedDiatomDiatomPES(
        interaction=interaction,
        two_total_spins=spins,
        orbital_states=orbitals,
        interaction_many=interaction_many,
        _interaction_many_processes=interaction_many_processes,
        _close=pes.close,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _validate_values(values: ElectronicValues, expected_shape: tuple[int, ...]) -> None:
    if values.shape != expected_shape:
        message = f"Spin-resolved AB+CD PES returned shape {values.shape}, but expected {expected_shape}"
        logger.error(message)
        raise ValueError(message)
    if not np.all(np.isfinite(values)):
        message = "Spin-resolved AB+CD PES returned non-finite values"
        logger.error(message)
        raise ValueError(message)
    if not np.allclose(values, np.swapaxes(np.conjugate(values), -1, -2), rtol=0.0, atol=1.0e-12):
        message = "Every spin-resolved orbital PES matrix must be Hermitian"
        logger.error(message)
        raise ValueError(message)


# ----------------------------------------------------------------------------------------
def _evaluate(
    pes: SpinResolvedDiatomDiatomPES,
    R: RadialInput,
    coordinates: NDArray[np.float64],
    grid_shape: tuple[int, ...],
    *,
    processes: int,
) -> ElectronicValues:
    """Evaluate scalar or radial-batched spin-resolved values."""
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
def get_spin_resolved_grid_diatom_diatom(
    pes: SpinResolvedDiatomDiatomPES,
    R: RadialInput,
    r_X: NDArray[np.float64],
    r_Y: NDArray[np.float64],
    theta_X: NDArray[np.float64],
    theta_Y: NDArray[np.float64],
    phi: NDArray[np.float64],
    *,
    processes: int = 1,
) -> ElectronicValues:
    """Evaluate total-spin-resolved orbital matrices on the AB+CD tensor grid.

    Inputs:
        pes: SpinResolvedDiatomDiatomPES - electronic interaction model
        R: RadialInput - scalar separation or radial batch in bohr
        r_X: NDArray[np.float64] - first bond grid in bohr
        r_Y: NDArray[np.float64] - second bond grid in bohr
        theta_X: NDArray[np.float64] - first polar grid in radians
        theta_Y: NDArray[np.float64] - second polar grid in radians
        phi: NDArray[np.float64] - torsional grid in radians
        processes: int - temporary worker process count

    Returns:
        values: ElectronicValues - real or complex values with shape
            ``(n_rX,n_rY,n_thetaX,n_thetaY,n_phi,n_spin,n_orb,n_orb)``,
            optionally preceded by ``n_R``
    """
    grids = np.meshgrid(r_X, r_Y, theta_X, theta_Y, phi, indexing="ij")
    coordinates = np.asfortranarray(np.stack(tuple(grid.reshape(-1) for grid in grids)))
    return _evaluate(pes, R, coordinates, grids[0].shape, processes=processes)


# ----------------------------------------------------------------------------------------
