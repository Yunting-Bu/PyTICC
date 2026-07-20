from collections.abc import Sequence
from math import prod
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.special import roots_legendre

from pyticc.basis.channel import ChannelBuilder, TruncSpec
from pyticc.basis.monomer import DiatomSpec
from pyticc.basis.podvr import RovibPODVR
from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.finalize import finalize_scattering
from pyticc.matrix.interaction import get_Vmat_BF, prepare_Vmat_BF_diatom_diatom
from pyticc.pes.wrapper import PESWrapper, get_Vgrid_diatom_diatom
from pyticc.propagation.runner import propagate_BF
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.coupled_states import run_coupled_states_BF
from pyticc.system import Approx, ScattSystem


# ----------------------------------------------------------------------------------------
def run_diatom_diatom(
    diatom_X: DiatomSpec,
    rovib_X: RovibPODVR,
    diatom_Y: DiatomSpec,
    rovib_Y: RovibPODVR,
    pes: PESWrapper,
    *,
    Jtot: int,
    system_parity: int,
    Etot: EnergyInput,
    reduced_mass: float,
    radial_boundaries: Sequence[float],
    radial_half_steps: Sequence[float],
    trunc: TruncSpec | None = None,
    n_theta_X: int = 15,
    n_theta_Y: int = 15,
    n_phi: int = 12,
    mode: Literal["inelastic", "capture"] = "inelastic",
    approx: Approx = Approx.EXACT,
    K_delta: int = 1,
    memory_limit_mb: float = 512.0,
) -> ScatteringResult | CoupledStatesResult:
    """
    Run one field-free diatom-diatom scattering block from channels through matching.

    Both monomer internal-energy arrays and ``Etot`` must use the same energy zero.
    Angular quadrature, interaction matrices, propagation, the BF-to-SF transformation,
    and asymptotic matching are handled internally.

    Inputs:
        diatom_X: DiatomSpec - first diatom states and internal energies
        rovib_X: RovibPODVR - first diatom PODVR grids and wavefunctions
        diatom_Y: DiatomSpec - second diatom states and internal energies
        rovib_Y: RovibPODVR - second diatom PODVR grids and wavefunctions
        pes: PESWrapper - monomer and interaction potential interfaces
        Jtot: int - total angular momentum
        system_parity: int - field-free parity block, -1 or 1
        Etot: EnergyInput - total energies with shape (n_energy,) in atomic units,
            or a one-column text file
        reduced_mass: float - diatom-diatom collision reduced mass in atomic units
        radial_boundaries: Sequence[float] - radial interval boundaries with shape
            (n_interval + 1,) in atomic units
        radial_half_steps: Sequence[float] - nominal LDMD half-step for each radial
            interval, shape (n_interval,)
        trunc: TruncSpec | None - channel-energy and helicity truncations
        n_theta_X: int - Gauss-Legendre points for the first polar angle
        n_theta_Y: int - Gauss-Legendre points for the second polar angle
        n_phi: int - Gauss-Legendre points for the dihedral angle on [0, pi]
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
    system = ScattSystem(diatom_X, diatom_Y, Jtot=Jtot, system_parity=system_parity, approx=approx)
    basis = ChannelBuilder(system, TruncSpec() if trunc is None else trunc).build()
    cos_theta_X, theta_weights_X = roots_legendre(n_theta_X)
    cos_theta_Y, theta_weights_Y = roots_legendre(n_theta_Y)
    phi_grid, phi_weights = roots_legendre(n_phi)
    theta_X = np.arccos(cos_theta_X)
    theta_Y = np.arccos(cos_theta_Y)
    phi = 0.5 * np.pi * (phi_grid + 1.0)
    phi_weights *= 0.5 * np.pi
    V_basis = prepare_Vmat_BF_diatom_diatom(
        basis,
        rovib_X,
        rovib_Y,
        cos_theta_X,
        theta_weights_X,
        cos_theta_Y,
        theta_weights_Y,
        phi,
        phi_weights,
    )

    def Vgrid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate PES grids, returning (*grid_shape,) or (n_R, *grid_shape)."""
        return get_Vgrid_diatom_diatom(pes, radial_points, rovib_X.grids, rovib_Y.grids, theta_X, theta_Y, phi)

    def Vmat(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Contract PES grids into shape (n_channel, n_channel), optionally preceded by n_R."""
        return get_Vmat_BF(V_basis, Vgrid(radial_points))

    message = f"Running diatom-diatom block J={Jtot}, parity={system_parity:+d}, channels={basis.n_channel}, energies={energies.size}"
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
