from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from loguru import logger


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class FSState:
    r"""
    One signed Hund-case-(a) primitive molecular state.

    Formula:
        The state label represents

        |v j Omega Lambda S Sigma>,

        with Omega = Lambda + Sigma. Twice-angular-momentum integers are used
        so integer and half-integer states are represented exactly.

    Members:
        v: int - vibrational quantum number
        two_j: int - twice the molecular total angular momentum j
        two_omega: int - twice its molecule-fixed projection Omega
        two_lambda: int - twice the electronic orbital projection Lambda;
            necessarily even
        two_S: int - twice the electronic spin S
        two_sigma: int - twice the electronic spin projection Sigma
    """

    v: int
    two_j: int
    two_omega: int
    two_lambda: int
    two_S: int
    two_sigma: int

    def __post_init__(self) -> None:
        if self.two_lambda % 2:
            message = f"Lambda must be integral, but two_lambda={self.two_lambda}"
            logger.error(message)
            raise ValueError(message)
        if abs(self.two_sigma) > self.two_S or (self.two_S - self.two_sigma) % 2:
            message = f"Sigma={self.two_sigma}/2 is incompatible with S={self.two_S}/2"
            logger.error(message)
            raise ValueError(message)
        if self.two_omega != self.two_lambda + self.two_sigma:
            message = "Hund-case-(a) state requires Omega = Lambda + Sigma"
            logger.error(message)
            raise ValueError(message)
        if abs(self.two_omega) > self.two_j or (self.two_j - self.two_omega) % 2:
            message = f"Omega={self.two_omega}/2 is incompatible with j={self.two_j}/2"
            logger.error(message)
            raise ValueError(message)

    @property
    def partner(self) -> "FSState":
        """Return the signed-projection partner under spatial inversion."""
        return FSState(
            v=self.v,
            two_j=self.two_j,
            two_omega=-self.two_omega,
            two_lambda=-self.two_lambda,
            two_S=self.two_S,
            two_sigma=-self.two_sigma,
        )

    @property
    def is_self_partner(self) -> bool:
        """Return whether inversion leaves every signed projection unchanged."""
        return self == self.partner


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParityPair:
    r"""
    Canonical primitive pair used to build one parity-adapted state.

    Formula:
        For distinct partners a and abar, the normalized state has the form

        |a; epsilon> = (|a> + epsilon phase |abar>)/sqrt(2).

        A self-partner has only one primitive component and unit normalization.
        The angular-momentum-dependent phase is deliberately applied by the
        parity transformation layer, where the total basis convention is known.

    Members:
        state: FSState - canonical representative a
        partner: FSState - signed-projection partner abar
        normalization: float - 1/sqrt(2) for a distinct pair and 1 otherwise
    """

    state: FSState
    partner: FSState
    normalization: float


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def parity_pair(state: FSState) -> ParityPair:
    """Return the canonical inversion pair containing ``state``."""
    partner = state.partner
    canonical = min(state, partner)
    canonical_partner = canonical.partner
    normalization = 1.0 if canonical == canonical_partner else 2.0**-0.5
    return ParityPair(canonical, canonical_partner, normalization)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _signed_lambda_values(two_lambda_abs: int) -> tuple[int, ...]:
    if two_lambda_abs == 0:
        return (0,)
    return (-two_lambda_abs, two_lambda_abs)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_primitive_states(
    v_values: Sequence[int],
    two_j_values: Sequence[int],
    two_lambda_abs: int,
    two_S: int,
) -> tuple[FSState, ...]:
    r"""
    Enumerate valid signed Hund-case-(a) primitive states.

    Formula:
        Lambda belongs to {-|Lambda|,+|Lambda|}, with a single value for
        Lambda=0. Sigma runs from -S to S in unit steps, and

        Omega = Lambda + Sigma,  |Omega| <= j.

        States for which j-Omega is not integral are excluded.

    Inputs:
        v_values: Sequence[int] - retained vibrational quantum numbers
        two_j_values: Sequence[int] - retained values of twice j
        two_lambda_abs: int - twice the nonnegative |Lambda|
        two_S: int - twice the nonnegative spin S

    Returns:
        states: tuple[FSState, ...] - valid states ordered by
            (v,two_j,two_lambda,two_sigma)
    """

    def states() -> Iterator[FSState]:
        for v in sorted(set(v_values)):
            for two_j in sorted(set(two_j_values)):
                for two_lambda in _signed_lambda_values(two_lambda_abs):
                    for two_sigma in range(-two_S, two_S + 1, 2):
                        two_omega = two_lambda + two_sigma
                        if abs(two_omega) <= two_j and (two_j - two_omega) % 2 == 0:
                            yield FSState(v, two_j, two_omega, two_lambda, two_S, two_sigma)

    return tuple(states())


# ----------------------------------------------------------------------------------------
