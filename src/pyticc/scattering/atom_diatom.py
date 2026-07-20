from collections.abc import Sequence
from math import prod
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import ChannelBuilder, TruncSpec
from pyticc.basis.diabatic import DiabaticDiatomBasis
from pyticc.basis.monomer import AtomSpec, DiatomSpec
from pyticc.basis.podvr import RovibPODVR
from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.finalize import finalize_scattering
from pyticc.matrix.diabatic import get_DiabaticVgrid_BF_atom_diatom, get_DiabaticVmat_BF, prepare_DiabaticVmat_BF_atom_diatom
from pyticc.matrix.interaction import get_Vmat_BF, prepare_Vmat_BF_atom_diatom
from pyticc.pes.diabatic import DiabaticPESWrapper
from pyticc.pes.wrapper import PESWrapper, get_Vgrid_atom_diatom
from pyticc.propagation.runner import propagate_BF
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.coupled_states import run_coupled_states_BF
from pyticc.system import Approx, ScattSystem


# ----------------------------------------------------------------------------------------
def run_atom_diatom(
    diatom: DiatomSpec,
    rovib: RovibPODVR,
    pes: PESWrapper,
    *,
    Jtot: int,
    system_parity: int,
    Etot: EnergyInput,
    reduced_mass: float,
    radial_boundaries: Sequence[float],
    radial_half_steps: Sequence[float],
    trunc: TruncSpec | None = None,
    n_theta: int = 16,
    mode: Literal["inelastic", "capture"] = "inelastic",
    approx: Approx = Approx.EXACT,
    K_delta: int = 1,
    memory_limit_mb: float = 512.0,
) -> ScatteringResult | CoupledStatesResult:
    """
    Run one field-free atom-diatom scattering block from channels through matching.

    The diatomic internal energies in ``diatom`` and total energies in ``Etot`` must
    use the same energy zero. Angular quadrature, interaction matrices, propagation,
    the BF-to-SF transformation, and asymptotic matching are handled internally.

    Inputs:
        diatom: DiatomSpec - diatomic states and internal energies
        rovib: RovibPODVR - diatomic PODVR grids and wavefunctions
        pes: PESWrapper - monomer and interaction potential interfaces
        Jtot: int - total angular momentum
        system_parity: int - field-free parity block, -1 or 1
        Etot: EnergyInput - total energies with shape (n_energy,) in atomic units,
            or a one-column text file
        reduced_mass: float - atom-diatom collision reduced mass in atomic units
        radial_boundaries: Sequence[float] - radial interval boundaries with shape
            (n_interval + 1,) in atomic units
        radial_half_steps: Sequence[float] - nominal LDMD half-step for each radial
            interval, shape (n_interval,)
        trunc: TruncSpec | None - channel-energy and helicity truncations
        n_theta: int - retained Gauss-Legendre points for the Jacobi angle; a
            homonuclear diatom uses one half of a rule with twice this order
        mode: Literal["inelastic", "capture"] - inner-boundary condition
        approx: Approx - exact CC, CS, or NNCC propagation
        K_delta: int - neighboring K range retained on each side in NNCC
        memory_limit_mb: float - target transient-memory limit in MiB

    Returns:
        result: ScatteringResult | CoupledStatesResult - exact result containing
            log-derivative arrays of shape (n_energy, n_channel, n_channel), or
            separated CS/NNCC block arrays with their corresponding block dimensions
    """
    energies = get_Etot(Etot)
    system = ScattSystem(AtomSpec(), diatom, Jtot=Jtot, system_parity=system_parity, approx=approx)
    basis = ChannelBuilder(system, TruncSpec() if trunc is None else trunc).build()
    cos_theta, theta_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta, symmetry=diatom.jpar != 0)
    theta = np.arccos(cos_theta)
    V_basis = prepare_Vmat_BF_atom_diatom(basis, rovib, cos_theta, theta_weights)

    def Vgrid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate PES grids, returning (*grid_shape,) or (n_R, *grid_shape)."""
        return get_Vgrid_atom_diatom(pes, radial_points, rovib.grids, theta)

    def Vmat(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Contract PES grids into shape (n_channel, n_channel), optionally preceded by n_R."""
        return get_Vmat_BF(V_basis, Vgrid(radial_points))

    message = f"Running atom-diatom block Jtot={Jtot}, parity={system_parity:+d}, channels={basis.n_channel}, energies={energies.size}"
    logger.info(message)
    if approx is not Approx.EXACT:
        return run_coupled_states_BF(
            basis=basis,
            V_basis=V_basis,
            Vgrid=Vgrid,
            Etot=energies,
            reduced_mass=reduced_mass,
            radial_boundaries=radial_boundaries,
            radial_half_steps=radial_half_steps,
            approx=approx,
            K_delta=K_delta,
            mode=mode,
            memory_limit_mb=memory_limit_mb,
        )

    Y_BF = propagate_BF(
        basis=basis,
        Vmat=Vmat,
        Etot=energies,
        reduced_mass=reduced_mass,
        radial_boundaries=radial_boundaries,
        radial_half_steps=radial_half_steps,
        mode=mode,
        batch_Vmat=True,
        memory_limit_mb=memory_limit_mb,
        potential_grid_size=prod(V_basis.grid_shape),
    )
    return finalize_scattering(basis, np.asarray(Y_BF), energies, reduced_mass, float(radial_boundaries[-1]))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def run_diabatic_atom_diatom(
    diatom: DiabaticDiatomBasis,
    pes: DiabaticPESWrapper,
    *,
    Jtot: int,
    system_parity: int,
    Etot: EnergyInput,
    reduced_mass: float,
    radial_boundaries: Sequence[float],
    radial_half_steps: Sequence[float],
    trunc: TruncSpec | None = None,
    n_theta: int = 16,
    mode: Literal["inelastic", "capture"] = "inelastic",
    memory_limit_mb: float = 512.0,
) -> ScatteringResult:
    """Run one exact-CC diabatic atom-diatom scattering block through matching.

    Electronic-state labels are retained in the channel basis. State-diagonal DPEM
    elements are contracted on each state's PODVR grid, while off-diagonal elements
    are contracted on the shared primitive DVR grid.

    Inputs:
        diatom: DiabaticDiatomBasis - state-resolved diatomic asymptotic basis
        pes: DiabaticPESWrapper - monomer potentials and interaction DPEM
        Jtot: int - total angular momentum
        system_parity: int - field-free parity block, -1 or 1
        Etot: EnergyInput - total energies in atomic units, or a one-column file
        reduced_mass: float - atom-diatom collision reduced mass in atomic units
        radial_boundaries: Sequence[float] - radial interval boundaries in bohr
        radial_half_steps: Sequence[float] - nominal LDMD half-step per interval
        trunc: TruncSpec | None - channel-energy and helicity truncations
        n_theta: int - retained Gauss-Legendre points for the Jacobi angle; when
            every state has rotational exchange parity, one half of a rule with
            twice this order is used as in ABCdia
        mode: Literal["inelastic", "capture"] - inner-boundary condition
        memory_limit_mb: float - target transient-memory limit in MiB

    Returns:
        result: ScatteringResult - matched result in the complete diabatic channel basis
    """
    if pes.n_state != diatom.n_state:
        message = f"PES has {pes.n_state} electronic states, but the diatomic basis has {diatom.n_state}"
        logger.error(message)
        raise ValueError(message)
    if n_theta < 1:
        message = f"n_theta must be positive, but got {n_theta}"
        logger.error(message)
        raise ValueError(message)

    energies = get_Etot(Etot)
    system = ScattSystem(AtomSpec(), diatom, Jtot=Jtot, system_parity=system_parity, approx=Approx.EXACT)
    basis = ChannelBuilder(system, TruncSpec() if trunc is None else trunc).build()
    angular_symmetry = all(jpar != 0 for jpar in diatom.rotational_parities)
    cos_theta, theta_weights = gauss_legendre_dvr(-1.0, 1.0, n_theta, symmetry=angular_symmetry)
    V_basis = prepare_DiabaticVmat_BF_atom_diatom(basis, diatom, cos_theta, theta_weights)

    def Vmat(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Sample and contract the DPEM, optionally for a radial batch."""
        potential = get_DiabaticVgrid_BF_atom_diatom(pes, radial_points, V_basis)
        return get_DiabaticVmat_BF(V_basis, potential)

    logger.info(
        f"Running diabatic atom-diatom block Jtot={Jtot}, parity={system_parity:+d}, "
        f"states={diatom.n_state}, channels={basis.n_channel}, energies={energies.size}"
    )
    potential_grid_size = V_basis.theta.size * (sum(grid.size for grid in V_basis.diagonal_grids) + V_basis.coupling_grid.size) * diatom.n_state**2
    Y_BF = propagate_BF(
        basis=basis,
        Vmat=Vmat,
        Etot=energies,
        reduced_mass=reduced_mass,
        radial_boundaries=radial_boundaries,
        radial_half_steps=radial_half_steps,
        mode=mode,
        batch_Vmat=True,
        memory_limit_mb=memory_limit_mb,
        potential_grid_size=potential_grid_size,
    )
    return finalize_scattering(basis, np.asarray(Y_BF), energies, reduced_mass, float(radial_boundaries[-1]))
