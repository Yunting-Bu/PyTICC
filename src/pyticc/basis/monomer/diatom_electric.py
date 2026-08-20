from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from pyticc.basis.angle import norm_YjK
from pyticc.basis.dvr import SineDVR, build_SineDVR
from pyticc.basis.podvr import podvr_grids
from pyticc.electric import ElectricResponseTable, electric_coefficients, load_electric_response_csv, required_m_values
from pyticc.electric.hamiltonian import solve_diatom_electric_block


def _readonly_array(values: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiatomElectricBlock:
    r"""
    One fixed-m electric-field-dressed diatomic block.

    Formula:
        The dressed monomer eigenstate is expanded as

        |phi_{alpha m}> = sum_p sum_{j=|m|}^{jmax}
                          C_{pj}^{alpha m} |p>|jm>,

        where p labels the PODVR radial grid and alpha labels the eigenstates
        in ascending energy order.

    Members:
        m: int - SF projection of the diatomic angular momentum
        j_values: NDArray[np.int64] - retained values |m|, ..., jmax,
            shape (n_j,)
        energies: NDArray[np.float64] - absolute dressed-state energies
            E_{alpha m}, shape (n_alpha,)
        coefficients: NDArray[np.float64] - expansion coefficients
            C_{pj}^{alpha m}, indexed as coefficients[p, j_index, alpha],
            shape (n_podvr, n_j, n_alpha)
    """

    m: int
    j_values: NDArray[np.int64]
    energies: NDArray[np.float64]
    coefficients: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "j_values", _readonly_array(self.j_values, np.int64))
        object.__setattr__(self, "energies", _readonly_array(self.energies))
        object.__setattr__(self, "coefficients", _readonly_array(self.coefficients))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiatomElectricBasis:
    """
    Electric-field-dressed diatomic eigenstates grouped into fixed-m blocks.

    The scattering quantities M and lmax are used only while selecting which m
    blocks to build. They are not properties of the resulting monomer basis.

    Members:
        grids: NDArray[np.float64] - PODVR bond-length grids in bohr,
            shape (n_podvr,)
        blocks: tuple[DiatomElectricBlock, ...] - dressed monomer blocks in
            ascending m order
        energy_zero: float - common absolute energy subtracted from channel
            thresholds, in atomic units
        electric_strength: float - electric-field strength in atomic units
        jmax: int - largest primitive diatomic angular momentum
        mass: float - diatomic reduced mass in atomic units
    """

    grids: NDArray[np.float64]
    blocks: tuple[DiatomElectricBlock, ...]
    energy_zero: float
    electric_strength: float
    jmax: int
    mass: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "grids", _readonly_array(self.grids))

    @property
    def m_values(self) -> tuple[int, ...]:
        """
        Get the available fixed-m projections.

        Returns:
            m_values: tuple[int, ...] - m values in the same order as blocks
        """
        return tuple(block.m for block in self.blocks)

    def block(self, m: int) -> DiatomElectricBlock:
        """
        Get one fixed-m dressed monomer block.

        Inputs:
            m: int - monomer SF projection

        Returns:
            block: DiatomElectricBlock - requested fixed-m block
        """
        for block in self.blocks:
            if block.m == m:
                return block
        message = f"m={m} is unavailable; available blocks are {self.m_values}"
        logger.error(message)
        raise KeyError(message)

    def relative_energies(self, m: int) -> NDArray[np.float64]:
        r"""
        Get one block's dressed energies relative to the common energy zero.

        Formula:
            epsilon_{alpha m} = E_{alpha m} - E_zero.

        Inputs:
            m: int - monomer SF projection

        Returns:
            energies: NDArray[np.float64] - relative energies epsilon_{alpha m},
                shape (n_alpha,)
        """
        return self.block(m).energies - self.energy_zero


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_DiatomElectricBasis(
    dvr: SineDVR,
    response: ElectricResponseTable | str | Path,
    *,
    electric_strength: float,
    n_podvr: int,
    jmax: int,
    M: int,
    lmax: int,
    n_alpha: int,
    mass: float,
    energy_zero: float | None = None,
) -> DiatomElectricBasis:
    r"""
    Build electric-field-dressed diatomic states in the PODVR times j basis.

    Formula:
        Let E_v and U_{vp} be the retained zero-field vibrational eigenvalues
        and the PODVR-to-contracted-FBR transformation. The radial reference
        Hamiltonian on the PODVR grid is

        H_{p'p}^{PO} = sum_v U_{vp'} E_v U_{vp}.

        For every m allowed by

        max(-jmax, M-lmax) <= m <= min(jmax, M+lmax),

        the fixed-m Hamiltonian is

        H_{p'j',pj}^{(m)}
          = delta_{j'j} H_{p'p}^{PO}
            + delta_{p'p} delta_{j'j}
              j(j+1)/(2 mu_r r_p^2)
            + delta_{p'p} sum_{n=0}^{3}
              a_n(r_p) X_{j'j}^{(n,m)}.

        Its eigenvectors define

        |phi_{alpha m}> = sum_p sum_j
                          C_{pj}^{alpha m}|p>|jm>.

        Because X^(n,m) depends on m only through |m| and m^2,

        H^(m) = H^(-m),   E_{alpha m} = E_{alpha,-m}.

        Therefore, the fixed-m Hamiltonian is diagonalized only once for each
        required |m|. Signed blocks are retained for later SF channel assembly.

        M and lmax select which m blocks are solved but are not stored in the
        returned monomer basis.

    Inputs:
        dvr: SineDVR - zero-field vibrational SineDVR basis
        response: ElectricResponseTable | str | Path - electric-response table
            or fixed-schema CSV path
        electric_strength: float - electric-field strength in atomic units
        n_podvr: int - number of retained PODVR radial points
        jmax: int - largest primitive diatomic angular momentum
        M: int - conserved total SF projection used to select m
        lmax: int - largest end-over-end angular momentum used to select m
        n_alpha: int - number of lowest dressed states retained in each
            fixed-m block
        mass: float - diatomic reduced mass mu_r in atomic units
        energy_zero: float | None - common absolute energy zero; if None, use
            the lowest eigenvalue of the m=0 Hamiltonian

    Returns:
        basis: DiatomElectricBasis - dressed energies and coefficients
            C_{pj}^{alpha m} on the PODVR grid
    """
    m_values = required_m_values(M, lmax, jmax)
    response_table = load_electric_response_csv(response) if isinstance(response, str | Path) else response
    grids, _, po_to_cfbr = podvr_grids(dvr.grids, dvr.eigen_vec, n_podvr)
    h_reference = np.einsum("ia,i,ib->ab", po_to_cfbr, dvr.eigen_val[:n_podvr], po_to_cfbr, optimize=True)
    radial_coefficients = electric_coefficients(response_table.evaluate(grids), electric_strength)
    solved_blocks: dict[int, tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]] = {}

    def build_block(m: int, n_states: int) -> DiatomElectricBlock:
        abs_m = abs(m)
        if abs_m not in solved_blocks:
            solved_blocks[abs_m] = solve_diatom_electric_block(
                h_reference,
                grids,
                radial_coefficients,
                m=abs_m,
                jmax=jmax,
                mass=mass,
                n_alpha=n_states,
            )
        j_values, energies, state_coefficients = solved_blocks[abs_m]
        return DiatomElectricBlock(m=m, j_values=j_values, energies=energies, coefficients=state_coefficients)

    blocks = tuple(build_block(m, n_alpha) for m in m_values)
    if energy_zero is None:
        if 0 in m_values:
            zero = float(blocks[m_values.index(0)].energies[0])
        else:
            zero = float(build_block(0, 1).energies[0])
    else:
        zero = float(energy_zero)

    return DiatomElectricBasis(
        grids=grids,
        blocks=blocks,
        energy_zero=zero,
        electric_strength=electric_strength,
        jmax=jmax,
        mass=mass,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def prepare_DiatomElectric(
    potential: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
    response: ElectricResponseTable | str | Path,
    *,
    r: tuple[float, float],
    n_dvr: int,
    n_podvr: int,
    electric_strength: float,
    jmax: int,
    M: int,
    lmax: int,
    n_alpha: int,
    mass: float,
    energy_zero: float | None = None,
) -> DiatomElectricBasis:
    """
    Prepare an electric-field-dressed diatom through the DVR and PODVR steps.

    Inputs:
        potential: Callable | None - zero-field monomer PES mapping bond-length
            grids with shape (n_dvr,) to energies with shape (n_dvr,); None
            raises an error
        response: ElectricResponseTable | str | Path - electric-response table
            or fixed-schema CSV path
        r: tuple[float,float] - left and right sine-DVR boundaries in bohr
        n_dvr: int - number of primitive sine-DVR points
        n_podvr: int - number of contracted PODVR points
        electric_strength: float - electric-field strength in atomic units
        jmax: int - largest primitive diatomic angular momentum
        M: int - conserved total SF projection used to select m
        lmax: int - largest end-over-end angular momentum used to select m
        n_alpha: int - number of lowest dressed states retained in each
            fixed-m block
        mass: float - diatomic reduced mass in atomic units
        energy_zero: float | None - common absolute energy zero; if None, use
            the lowest eigenvalue of the m=0 Hamiltonian

    Returns:
        monomer: DiatomElectricBasis - prepared electric-field-dressed monomer
            basis on the PODVR grid
    """
    if potential is None:
        message = "Electric diatomic monomer preparation requires a monomer potential"
        logger.error(message)
        raise ValueError(message)

    dvr = build_SineDVR(r[0], r[1], n_dvr, mass, potential)
    return build_DiatomElectricBasis(
        dvr,
        response,
        electric_strength=electric_strength,
        n_podvr=n_podvr,
        jmax=jmax,
        M=M,
        lmax=lmax,
        n_alpha=n_alpha,
        mass=mass,
        energy_zero=energy_zero,
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def diatom_electric_amplitude(
    block: DiatomElectricBlock,
    cos_theta: ArrayLike,
    weights: ArrayLike | None = None,
) -> NDArray[np.float64]:
    r"""
    Get the reduced dressed-monomer amplitude on angular grids.

    Formula:
        With x_q = cos(theta_q),

        A_{alpha m}(p,x_q)
          = sum_{j=|m|}^{jmax}
            C_{pj}^{alpha m} P_tilde_j^m(x_q),

        where

        integral_{-1}^{1}
          P_tilde_{j'}^m(x) P_tilde_j^m(x) dx
          = delta_{j'j}.

        The complete coordinate representation is

        <r_p,theta,phi|phi_{alpha m}>
          = A_{alpha m}(p,cos(theta))
            exp(i m phi)/sqrt(2 pi).

        If Gauss-Legendre weights w_q are supplied, the returned grid
        representation is

        A_bar_{alpha m}(p,q)
          = sqrt(w_q) A_{alpha m}(p,x_q).

    Inputs:
        block: DiatomElectricBlock - one fixed-m dressed-state block
        cos_theta: ArrayLike - angular nodes x_q = cos(theta_q), shape (n_theta,)
        weights: ArrayLike | None - optional Gauss-Legendre weights w_q,
            shape (n_theta,)

    Returns:
        amplitudes: NDArray[np.float64] - reduced amplitudes A_{alpha m}(p,x_q),
            or weighted amplitudes A_bar_{alpha m}(p,q), indexed as
            amplitudes[alpha, p, q], shape (n_alpha, n_podvr, n_theta)
    """
    nodes = np.asarray(cos_theta, dtype=np.float64)
    angular_values = np.stack([norm_YjK(int(j), block.m, nodes) for j in block.j_values], axis=0)
    amplitudes = np.einsum("pja,jq->apq", block.coefficients, angular_values, optimize=True)
    if weights is not None:
        angular_weights = np.asarray(weights, dtype=np.float64)
        amplitudes *= np.sqrt(angular_weights)[None, None, :]
    return amplitudes


# ----------------------------------------------------------------------------------------
