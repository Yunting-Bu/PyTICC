from functools import lru_cache

import numpy as np
from numpy.typing import NDArray


# ----------------------------------------------------------------------------------------
def required_m_values(M: int, lmax: int, jmax: int) -> tuple[int, ...]:
    r"""
    Get all diatomic projections required by the SF projection constraint.

    Formula:
        The SF projections satisfy

        M = m + m_l,

        with |m| <= jmax and |m_l| <= lmax. Therefore,

        max(-jmax, M-lmax) <= m <= min(jmax, M+lmax).

    Inputs:
        M: int - conserved total SF projection
        lmax: int - largest end-over-end angular momentum
        jmax: int - largest primitive diatomic angular momentum

    Returns:
        m_values: tuple[int, ...] - consecutive required values of m
    """
    m_min = max(-jmax, M - lmax)
    m_max = min(jmax, M + lmax)
    return tuple(range(m_min, m_max + 1))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _cosine_step(j: int, m: int) -> float:
    if j < 0 or j < abs(m):
        return 0.0
    return float(np.sqrt(((j + 1) ** 2 - m**2) / ((2 * j + 1) * (2 * j + 3))))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@lru_cache
def rotor_orientation_moment_matrices(jmax: int, m: int) -> tuple[NDArray[np.float64], ...]:
    r"""
    Get the first three rotor orientation-moment matrices.

    Formula:
        In the normalized associated-Legendre basis |jm>, define

        b_j^m = sqrt{[(j+1)^2-m^2]/[(2j+1)(2j+3)]}.

        Multiplication by x = cos(theta) gives

        x |jm> = b_j^m |j+1,m> + b_{j-1}^m |j-1,m>.

        Hence, with X_{j'j}^{(n,m)} = <j'm|x^n|jm>,

        X_{j'j}^{(0,m)} = delta_{j'j},

        X_{j'j}^{(1,m)}
          = b_j^m delta_{j',j+1}
            + b_{j-1}^m delta_{j',j-1},

        X_{j'j}^{(2,m)}
          = b_j^m b_{j+1}^m delta_{j',j+2}
            + [(b_j^m)^2+(b_{j-1}^m)^2] delta_{j'j}
            + b_{j-1}^m b_{j-2}^m delta_{j',j-2},

        X_{j'j}^{(3,m)}
          = b_j^m b_{j+1}^m b_{j+2}^m delta_{j',j+3}
            + b_j^m [(b_{j-1}^m)^2+(b_j^m)^2+(b_{j+1}^m)^2]
              delta_{j',j+1}
            + b_{j-1}^m [(b_{j-2}^m)^2+(b_{j-1}^m)^2+(b_j^m)^2]
              delta_{j',j-1}
            + b_{j-1}^m b_{j-2}^m b_{j-3}^m delta_{j',j-3}.

        The retained basis is j = |m|, ..., jmax. These expressions are
        evaluated before projection, so the diagonal x^2 contribution through
        the virtual state jmax+1 is not lost.

    Inputs:
        jmax: int - largest primitive rotor angular momentum
        m: int - conserved SF projection of the diatomic angular momentum

    Returns:
        orientation_matrices: tuple[NDArray[np.float64], ...] - identity and
            orientation-moment matrices X^(n,m) for n = 0, 1, 2, 3, each with shape
            (jmax-|m|+1, jmax-|m|+1)
    """
    j_values = np.arange(abs(m), jmax + 1, dtype=np.int64)
    matrices = [np.zeros((j_values.size, j_values.size), dtype=np.float64) for _ in range(4)]
    matrices[0] = np.eye(j_values.size, dtype=np.float64)
    index = {int(j): i for i, j in enumerate(j_values)}

    for column, j_value in enumerate(j_values):
        j = int(j_value)
        entries = (
            {j + 1: _cosine_step(j, m), j - 1: _cosine_step(j - 1, m)},
            {
                j + 2: _cosine_step(j, m) * _cosine_step(j + 1, m),
                j: _cosine_step(j, m) ** 2 + _cosine_step(j - 1, m) ** 2,
                j - 2: _cosine_step(j - 1, m) * _cosine_step(j - 2, m),
            },
            {
                j + 3: _cosine_step(j, m) * _cosine_step(j + 1, m) * _cosine_step(j + 2, m),
                j + 1: _cosine_step(j, m) * (_cosine_step(j - 1, m) ** 2 + _cosine_step(j, m) ** 2 + _cosine_step(j + 1, m) ** 2),
                j - 1: _cosine_step(j - 1, m) * (_cosine_step(j - 2, m) ** 2 + _cosine_step(j - 1, m) ** 2 + _cosine_step(j, m) ** 2),
                j - 3: _cosine_step(j - 1, m) * _cosine_step(j - 2, m) * _cosine_step(j - 3, m),
            },
        )
        for power, power_entries in enumerate(entries, start=1):
            for target_j, value in power_entries.items():
                if target_j in index:
                    matrices[power][index[target_j], column] += value

    result = tuple(matrices)
    for matrix in result:
        matrix.setflags(write=False)
    return result


# ----------------------------------------------------------------------------------------
