from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.fine_structure.basis import FSState
from pyticc.fine_structure.operators import FSConstants, effective_hamiltonian


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSLevelBlock:
    """
    Fine-structure eigenlevels for one fixed (v,j,parity) block.

    Members:
        v: int - vibrational quantum number
        two_j: int - twice molecular total angular momentum
        parity: int - monomer inversion parity, -1 or 1
        primitive_states: tuple[FSState,...] - signed primitive basis
        transform: NDArray[np.float64] - primitive-to-parity basis transform,
            shape (n_primitive,n_parity)
        energies: NDArray[np.float64] - tau-ordered eigenenergies in Hartree
        coefficients: NDArray[np.float64] - parity-basis eigenvectors with
            columns indexed by tau
    """

    v: int
    two_j: int
    parity: int
    primitive_states: tuple[FSState, ...]
    transform: NDArray[np.float64]
    energies: NDArray[np.float64]
    coefficients: NDArray[np.float64]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def parity_transform(
    states: tuple[FSState, ...],
    parity: int,
    *,
    reflection_parity: int = 1,
) -> NDArray[np.float64]:
    r"""
    Construct the signed primitive-to-monomer-parity transformation.

    Formula:
        |a;epsilon> = N[|a> + epsilon q |abar>],

        q = (-1)^(j-S) for |Lambda|>0 and
        q = sigma_reflection (-1)^(j-S) for Lambda=0.

        Distinct partners have N=1/sqrt(2). A self-partner is retained only
        when epsilon q=+1, with unit normalization.

    Inputs:
        states: tuple[FSState,...] - complete inversion-closed signed basis
        parity: int - requested monomer inversion parity, -1 or 1
        reflection_parity: int - Sigma-state reflection symmetry, -1 or 1

    Returns:
        transform: NDArray[np.float64] - orthonormal columns, shape
            (n_primitive,n_parity_state)
    """
    index = {state: position for position, state in enumerate(states)}
    if len(index) != len(states) or any(state.partner not in index for state in states):
        message = "Primitive fine-structure basis must contain each inversion partner exactly once"
        logger.error(message)
        raise ValueError(message)

    columns: list[NDArray[np.float64]] = []
    consumed: set[FSState] = set()
    for state in states:
        if state in consumed:
            continue
        partner = state.partner
        consumed.update((state, partner))
        exponent = (state.two_j - state.two_S) // 2
        q = int((-1) ** exponent)
        if state.two_lambda == 0:
            q *= reflection_parity
        column = np.zeros(len(states), dtype=np.float64)
        if state == partner:
            if parity * q == 1:
                column[index[state]] = 1.0
                columns.append(column)
            continue
        representative = state if (state.two_lambda, state.two_sigma, state.two_omega) > (0, 0, 0) else partner
        representative_partner = representative.partner
        column[index[representative]] = 2.0**-0.5
        column[index[representative_partner]] = parity * q * 2.0**-0.5
        columns.append(column)
    return np.column_stack(columns) if columns else np.empty((len(states), 0), dtype=np.float64)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def diagonalize_block(
    states: tuple[FSState, ...],
    constants: FSConstants,
    parity: int,
    *,
    vibrational_energy: float = 0.0,
    reflection_parity: int = 1,
) -> FSLevelBlock:
    r"""
    Diagonalize one fixed-(v,j,parity) effective molecular Hamiltonian.

    Formula:
        H^epsilon = U_epsilon^T H_signed U_epsilon,
        H^epsilon C = C diag(E_tau).

    Inputs:
        states: tuple[FSState,...] - fixed-(v,j,S) signed basis
        constants: FSConstants - effective constants in Hartree
        parity: int - monomer parity epsilon
        vibrational_energy: float - vibrational term T_v in Hartree
        reflection_parity: int - Sigma electronic reflection symmetry

    Returns:
        block: FSLevelBlock - tau energies and transformations
    """
    if not states:
        message = "Fine-structure block requires at least one primitive state"
        logger.error(message)
        raise ValueError(message)
    labels = {(state.v, state.two_j, state.two_S) for state in states}
    if len(labels) != 1:
        message = "Fine-structure diagonalization requires one fixed (v,j,S) block"
        logger.error(message)
        raise ValueError(message)
    transform = parity_transform(states, parity, reflection_parity=reflection_parity)
    signed = effective_hamiltonian(states, constants, vibrational_energy)
    parity_matrix = transform.T @ signed @ transform
    energies, coefficients = np.linalg.eigh(parity_matrix)
    v, two_j, _ = next(iter(labels))
    return FSLevelBlock(v, two_j, parity, states, transform, energies, coefficients)


# ----------------------------------------------------------------------------------------
