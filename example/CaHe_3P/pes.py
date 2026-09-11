"""HIBRIDON cahe NPOT=1 model, not an ab initio Ca--He surface.

Source: hibridon/hibridon tests/cahe/pot_cahe.F90 (potmsv1) and
Cahe_test.inp at commit 27b79222d0e09d1ea39de6651d465c932125654b.
Atomic Ca(4s5p 3P) and He(1S) are fixed terms. Only the triplet block is used.
"""

import numpy as np

from pyticc import SpinResolvedAtomAtomPES, atom_atom_orbital_states


def msv(r: float, de: float, re: float, be: float, rl: float, c6: float) -> float:
    r"""Evaluate the Morse--spline--van der Waals model in atomic units.

    Formula:
        V=de exp[-be(R-re)](exp[-be(R-re)]-2), R<=re;
        V=-de+a x^2+b x^3, re<R<=rl, x=R-re, d=rl-re;
        a=3(de-c6/rl^6)/d^2-6 c6/(rl^7 d),
        b=-2(de-c6/rl^6)/d^3+6 c6/(rl^7 d^2);
        V=-c6/R^6, R>rl. Value and first derivative match at both joins.

    Inputs:
        r, re, rl: float - separation, minimum and matching radius in bohr
        de: float - well depth in Hartree
        be: float - Morse inverse length, bohr^-1
        c6: float - dispersion coefficient, Hartree bohr^6
    Returns:
        energy: float - interaction in Hartree, zero at infinity
    """
    if r <= re:
        exponential = np.exp(-be * (r - re))
        return float(de * exponential * (exponential - 2))
    if r > rl:
        return -c6 / r**6
    d, x = rl - re, r - re
    a = 3 * (de - c6 / rl**6) / d**2 - 6 * c6 / (rl**7 * d)
    b = -2 * (de - c6 / rl**6) / d**3 + 6 * c6 / (rl**7 * d**2)
    return -de + a * x * x + b * x**3


def interaction(r: float) -> np.ndarray:
    """Return W[S=1,lambda_bra,lambda_ket], Hartree, lambda=(-1,0,1)."""
    pi = msv(r, 1e-4, 9.0, 0.4, 13.0, 40.0)
    sigma = msv(r, 8e-6, 13.8, 0.47, 18.0, 50.0)
    return np.diag([pi, sigma, pi])[None, ...]


pes = SpinResolvedAtomAtomPES(interaction, (2,), atom_atom_orbital_states(2, 0))
