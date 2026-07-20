from collections.abc import Sequence
from math import prod
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.special import roots_legendre

from pyticc.basis.channel import ChannelBuilder, TruncSpec
from pyticc.basis.monomer import AtomSpec, DiatomSpec
from pyticc.basis.podvr import RovibPODVR
from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.finalize import finalize_scattering
from pyticc.matrix.interaction import get_Vmat_BF, prepare_Vmat_BF_atom_diatom
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
        n_theta: int - Gauss-Legendre points for the Jacobi angle
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
    cos_theta, theta_weights = roots_legendre(n_theta)
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
