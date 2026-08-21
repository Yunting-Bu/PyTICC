from dataclasses import dataclass
from functools import lru_cache
from math import sqrt

import numpy as np
from numpy.typing import NDArray
from sympy import Rational
from sympy.physics.wigner import wigner_3j

from pyticc.constants import EnergyUnit, energy_to_au
from pyticc.fine_structure.basis import FSState


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSConstants:
    """
    Effective molecular constants for one vibrational manifold.

    Members:
        A: float - spin-orbit constant in Hartree
        B: float - molecular rotational constant in Hartree
        gamma: float - spin-rotation constant in Hartree
        lambda_ss: float - spin-spin constant in Hartree
        O: float - O Lambda-doubling constant in Hartree
        P: float - P Lambda-doubling constant in Hartree
        Q: float - Q Lambda-doubling constant in Hartree
        M: float - M Delta-state Lambda-doubling constant in Hartree
        N: float - N Delta-state Lambda-doubling constant in Hartree
    """

    A: float = 0.0
    B: float = 0.0
    gamma: float = 0.0
    lambda_ss: float = 0.0
    O: float = 0.0  # noqa: E741 - O is the conventional spectroscopic constant.
    P: float = 0.0
    Q: float = 0.0
    M: float = 0.0
    N: float = 0.0

    @classmethod
    def from_unit(
        cls,
        unit: EnergyUnit,
        *,
        A: float = 0.0,
        B: float = 0.0,
        gamma: float = 0.0,
        lambda_ss: float = 0.0,
        O: float = 0.0,  # noqa: E741 - O is the conventional spectroscopic constant.
        P: float = 0.0,
        Q: float = 0.0,
        M: float = 0.0,
        N: float = 0.0,
    ) -> "FSConstants":
        """
        Construct one constants set from a shared spectroscopic unit.

        Stored values are always converted to Hartree. For mixed-unit tables,
        convert individual values with ``energy_to_au`` and use the ordinary
        constructor.

        Inputs:
            unit: EnergyUnit - au, cm-1, Hz, kHz, MHz, or GHz
            A: float - spin-orbit constant in the selected unit
            B: float - molecular rotational constant in the selected unit
            gamma: float - spin-rotation constant in the selected unit
            lambda_ss: float - spin-spin constant in the selected unit
            O: float - O Lambda-doubling constant in the selected unit
            P: float - P Lambda-doubling constant in the selected unit
            Q: float - Q Lambda-doubling constant in the selected unit
            M: float - M Delta-state Lambda-doubling constant in the selected unit
            N: float - N Delta-state Lambda-doubling constant in the selected unit

        Returns:
            constants: FSConstants - constants converted to Hartree
        """
        factor = energy_to_au(1.0, unit)
        return cls(
            A=A * factor,
            B=B * factor,
            gamma=gamma * factor,
            lambda_ss=lambda_ss * factor,
            O=O * factor,
            P=P * factor,
            Q=Q * factor,
            M=M * factor,
            N=N * factor,
        )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@lru_cache
def _wigner_3j(
    two_j1: int,
    two_j2: int,
    two_j3: int,
    two_m1: int,
    two_m2: int,
    two_m3: int,
) -> float:
    """Return one Wigner 3-j symbol using exact half-integer arguments."""
    return float(
        wigner_3j(
            Rational(two_j1, 2),
            Rational(two_j2, 2),
            Rational(two_j3, 2),
            Rational(two_m1, 2),
            Rational(two_m2, 2),
            Rational(two_m3, 2),
        )
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _same_block(bra: FSState, ket: FSState) -> bool:
    return bra.v == ket.v and bra.two_j == ket.two_j and bra.two_S == ket.two_S


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def spin_orbit_element(bra: FSState, ket: FSState, constant: float) -> float:
    r"""
    Return a Hund-case-(a) spin-orbit matrix element.

    Formula:
        <Lambda' S Sigma' j Omega'|H_SO|Lambda S Sigma j Omega>
        = A Lambda Sigma delta_{Lambda'Lambda} delta_{Sigma'Sigma}
          delta_{j'j} delta_{Omega'Omega}.
    """
    if bra != ket:
        return 0.0
    return constant * bra.two_lambda * bra.two_sigma / 4.0


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def spin_spin_element(bra: FSState, ket: FSState, constant: float) -> float:
    r"""
    Return the axial spin-spin matrix element.

    Formula:
        <a'|H_SS|a> = (2 lambda/3)[3 Sigma^2-S(S+1)] delta_{a'a}.
    """
    if bra != ket:
        return 0.0
    S = bra.two_S / 2.0
    sigma = bra.two_sigma / 2.0
    return 2.0 * constant * (3.0 * sigma**2 - S * (S + 1.0)) / 3.0


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _js_off_diagonal(bra: FSState, ket: FSState) -> float:
    """Return the shared off-diagonal angular factor in H_MR and H_SMR."""
    if not _same_block(bra, ket) or bra.two_lambda != ket.two_lambda:
        return 0.0
    two_j = ket.two_j
    two_S = ket.two_S
    j = two_j / 2.0
    S = two_S / 2.0
    factor = sqrt(j * (j + 1.0) * (two_j + 1.0) * S * (S + 1.0) * (two_S + 1.0))
    value = 0.0
    for q in (-1, 1):
        value += _wigner_3j(two_j, 2, two_j, -bra.two_omega, 2 * q, ket.two_omega) * _wigner_3j(two_S, 2, two_S, -bra.two_sigma, 2 * q, ket.two_sigma)
    exponent = (two_j - bra.two_omega + two_S - bra.two_sigma) // 2
    return (-1.0) ** exponent * factor * value


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def molecular_rotation_element(bra: FSState, ket: FSState, constant: float) -> float:
    r"""
    Return the molecular rotational matrix element B N^2.

    Formula:
        The diagonal part is

        B[j(j+1)+S(S+1)-2 Omega Sigma],

        and the off-diagonal spin-uncoupling part is -2 B times the shared
        rank-one J-S angular factor implemented by ``_js_off_diagonal``.
    """
    if not _same_block(bra, ket) or bra.two_lambda != ket.two_lambda:
        return 0.0
    value = -2.0 * constant * _js_off_diagonal(bra, ket)
    if bra == ket:
        j = ket.two_j / 2.0
        S = ket.two_S / 2.0
        omega = ket.two_omega / 2.0
        sigma = ket.two_sigma / 2.0
        value += constant * (j * (j + 1.0) + S * (S + 1.0) - 2.0 * omega * sigma)
    return value


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def spin_rotation_element(bra: FSState, ket: FSState, constant: float) -> float:
    r"""
    Return the spin-molecular-rotation matrix element gamma N dot S.

    Formula:
        The diagonal part is gamma[Omega Sigma-S(S+1)], and the off-diagonal
        part is gamma times the shared rank-one J-S angular factor.
    """
    if not _same_block(bra, ket) or bra.two_lambda != ket.two_lambda:
        return 0.0
    value = constant * _js_off_diagonal(bra, ket)
    if bra == ket:
        S = ket.two_S / 2.0
        omega = ket.two_omega / 2.0
        sigma = ket.two_sigma / 2.0
        value += constant * (omega * sigma - S * (S + 1.0))
    return value


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def pi_lambda_doubling_element(bra: FSState, ket: FSState, constants: FSConstants) -> float:
    r"""
    Return the O, P, Q Lambda-doubling element for a Pi state.

    Formula:
        The Brown tensor expression is evaluated for q=+-1 with the selection
        Lambda'=Lambda-2q. Its rank-two spin, rank-two rotational, and rank-one
        spin-rotational terms carry O+P+Q, Q, and P+2Q, respectively.
    """
    if not _same_block(bra, ket) or abs(ket.two_lambda) != 2 or abs(bra.two_lambda) != 2:
        return 0.0
    j = ket.two_j / 2.0
    S = ket.two_S / 2.0
    value = 0.0
    for q in (-1, 1):
        if bra.two_lambda != ket.two_lambda - 4 * q:
            continue
        spin_phase = (-1.0) ** ((ket.two_S - bra.two_sigma) // 2)
        if bra.two_omega == ket.two_omega and S >= 1.0:
            spin_rank_two = sqrt((2.0 * S - 1.0) * (2.0 * S) * (2.0 * S + 1.0) * (2.0 * S + 2.0) * (2.0 * S + 3.0))
            value += (
                (constants.O + constants.P + constants.Q)
                * spin_phase
                * spin_rank_two
                * _wigner_3j(ket.two_S, 4, ket.two_S, -bra.two_sigma, 4 * q, ket.two_sigma)
                / (2.0 * sqrt(6.0))
            )
        rotational_phase = (-1.0) ** ((ket.two_j - bra.two_omega) // 2)
        rotational_prefactor = rotational_phase * sqrt(j * (2.0 * j + 1.0))
        if bra.two_sigma == ket.two_sigma and j >= 1.0:
            value += (
                rotational_prefactor
                * constants.Q
                * sqrt((2.0 * j - 1.0) * (2.0 * j + 2.0) * (2.0 * j + 3.0))
                * _wigner_3j(ket.two_j, 4, ket.two_j, -bra.two_omega, -4 * q, ket.two_omega)
                / (2.0 * sqrt(3.0))
            )
        if S > 0.0:
            value += (
                rotational_prefactor
                * (constants.P + 2.0 * constants.Q)
                * spin_phase
                * sqrt((j + 1.0) * S * (S + 1.0) * (2.0 * S + 1.0))
                * _wigner_3j(ket.two_j, 2, ket.two_j, -bra.two_omega, -2 * q, ket.two_omega)
                * _wigner_3j(ket.two_S, 2, ket.two_S, -bra.two_sigma, 2 * q, ket.two_sigma)
            )
    return value


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _ladder_factor(order: int, angular: float, projection: float, direction: int) -> float:
    """Return a repeated raising or lowering ladder factor."""
    value = 1.0
    for step in range(order):
        if direction == 1:
            value *= (angular - projection - step) * (angular + projection + step + 1.0)
        else:
            value *= (angular + projection - step) * (angular - projection + step + 1.0)
    return sqrt(max(value, 0.0))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def delta_lambda_doubling_element(bra: FSState, ket: FSState, constants: FSConstants) -> float:
    r"""
    Return the M, N, O, P, Q Lambda-doubling element for a Delta state.

    Formula:
        For eta=+-1 and k=0,...,min(4,2S),

        <a'|H_LD|a> = (1/2) sum_eta sum_k (-1)^k c_k
        delta_{Lambda',Lambda-4eta} delta_{Sigma',Sigma+k eta}
        delta_{Omega',Omega-(4-k)eta}
        F_k^eta(S,Sigma) F_{4-k}^{-eta}(j,Omega),

        where c=(Q,P+4Q,O+3P+6Q,N+2O+3P+4Q,M+N+O+P+Q).
    """
    if not _same_block(bra, ket) or abs(ket.two_lambda) != 4 or abs(bra.two_lambda) != 4:
        return 0.0
    coefficients = (
        constants.Q,
        constants.P + 4.0 * constants.Q,
        constants.O + 3.0 * constants.P + 6.0 * constants.Q,
        constants.N + 2.0 * constants.O + 3.0 * constants.P + 4.0 * constants.Q,
        constants.M + constants.N + constants.O + constants.P + constants.Q,
    )
    j = ket.two_j / 2.0
    S = ket.two_S / 2.0
    sigma = ket.two_sigma / 2.0
    omega = ket.two_omega / 2.0
    value = 0.0
    for eta in (-1, 1):
        if bra.two_lambda != ket.two_lambda - 8 * eta:
            continue
        for order in range(min(4, ket.two_S) + 1):
            if bra.two_sigma != ket.two_sigma + 2 * order * eta:
                continue
            if bra.two_omega != ket.two_omega - 2 * (4 - order) * eta:
                continue
            value += 0.5 * (-1.0) ** order * coefficients[order] * _ladder_factor(order, S, sigma, eta) * _ladder_factor(4 - order, j, omega, -eta)
    return value


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def lambda_doubling_element(bra: FSState, ket: FSState, constants: FSConstants) -> float:
    """Dispatch the Pi or Delta Lambda-doubling operator from |Lambda|."""
    if abs(ket.two_lambda) == 2:
        return pi_lambda_doubling_element(bra, ket, constants)
    if abs(ket.two_lambda) == 4:
        return delta_lambda_doubling_element(bra, ket, constants)
    return 0.0


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def effective_hamiltonian(states: tuple[FSState, ...], constants: FSConstants, vibrational_energy: float = 0.0) -> NDArray[np.float64]:
    r"""
    Construct the signed-projection effective molecular Hamiltonian.

    Formula:
        H_eff = T_v + H_SO + H_MR + H_SMR + H_SS + H_LD.

    Inputs:
        states: tuple[FSState,...] - one fixed-(v,j,S) signed basis
        constants: FSConstants - constants for that vibrational manifold
        vibrational_energy: float - T_v in Hartree

    Returns:
        matrix: NDArray[np.float64] - real symmetric matrix, shape (n,n)
    """
    size = len(states)
    matrix = np.empty((size, size), dtype=np.float64)
    for row, bra in enumerate(states):
        for column, ket in enumerate(states):
            matrix[row, column] = (
                spin_orbit_element(bra, ket, constants.A)
                + molecular_rotation_element(bra, ket, constants.B)
                + spin_rotation_element(bra, ket, constants.gamma)
                + spin_spin_element(bra, ket, constants.lambda_ss)
                + lambda_doubling_element(bra, ket, constants)
            )
            if row == column:
                matrix[row, column] += vibrational_energy
    return 0.5 * (matrix + matrix.T)


# ----------------------------------------------------------------------------------------
