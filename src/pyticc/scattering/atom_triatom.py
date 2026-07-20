from collections.abc import Sequence
from math import prod
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pyticc.basis.angle import gauss_legendre_dvr
from pyticc.basis.channel import ChannelBuilder, TruncSpec
from pyticc.basis.monomer import AtomSpec
from pyticc.basis.triatom import TriatomBasis
from pyticc.energy import EnergyInput, get_Etot
from pyticc.match.finalize import finalize_scattering
from pyticc.matrix.atom_triatom import prepare_Vmat_BF_atom_triatom
from pyticc.matrix.interaction import get_Vmat_BF
from pyticc.pes.wrapper import PESWrapper, get_Vgrid_atom_triatom
from pyticc.propagation.runner import propagate_BF
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.coupled_states import run_coupled_states_BF
from pyticc.system import Approx, ScattSystem


# ----------------------------------------------------------------------------------------
def run_atom_triatom(
    triatom: TriatomBasis,
    pes: PESWrapper,
    *,
    Jtot: int,
    system_parity: int,
    Etot: EnergyInput,
    reduced_mass: float,
    radial_boundaries: Sequence[float],
    radial_half_steps: Sequence[float],
    trunc: TruncSpec | None = None,
    n_theta_1: int | None = None,
    n_theta_2: int = 16,
    n_phi: int = 16,
    mode: Literal["inelastic", "capture"] = "inelastic",
    approx: Approx = Approx.EXACT,
    K_delta: int = 1,
    memory_limit_mb: float = 512.0,
) -> ScatteringResult | CoupledStatesResult:
    """
    Run one field-free atom-triatom scattering block from channels through matching.

    ``triatom`` must have been solved for the same value of
    ``system_parity * (-1)**Jtot``. By default, the interaction integral reuses its
    monomer bending quadrature; ``n_theta_1`` may request a separate quadrature.

    Inputs:
        triatom: TriatomBasis - contracted triatomic eigenstates and PODVR data
        pes: PESWrapper - atom-triatom interaction potential interface
        Jtot: int - total angular momentum
        system_parity: int - field-free parity block, -1 or 1
        Etot: EnergyInput - total energies with shape (n_energy,) in atomic units,
            or a one-column text file
        reduced_mass: float - atom-triatom collision reduced mass in atomic units
        radial_boundaries: Sequence[float] - radial interval boundaries with shape
            (n_interval + 1,) in atomic units
        radial_half_steps: Sequence[float] - nominal LDMD half-step for each radial
            interval, shape (n_interval,)
        trunc: TruncSpec | None - channel-energy and helicity truncations
        n_theta_1: int | None - optional interaction quadrature size for the
            triatomic bend; None reuses the monomer quadrature
        n_theta_2: int - Gauss-Legendre points for the external polar angle
        n_phi: int - Gauss-Legendre points for the dihedral angle on [0, pi]
        mode: Literal["inelastic", "capture"] - inner-boundary condition
        approx: Approx - exact CC, CS, or NNCC propagation
        K_delta: int - neighboring K range retained on each side in NNCC
        memory_limit_mb: float - target transient-memory limit in MiB

    Returns:
        result: ScatteringResult | CoupledStatesResult - exact result containing
            arrays with shape (n_energy, n_channel, n_channel), or separated
            CS/NNCC block arrays with their corresponding block dimensions
    """
    energies = get_Etot(Etot)
    system = ScattSystem(AtomSpec(), triatom, Jtot=Jtot, system_parity=system_parity, approx=approx)
    basis = ChannelBuilder(system, TruncSpec() if trunc is None else trunc).build()

    if n_theta_1 is None:
        if triatom.cos_theta is None or triatom.theta_weights is None:
            message = "TriatomBasis has no stored bending quadrature; provide n_theta_1"
            logger.error(message)
            raise ValueError(message)
        cos_theta_1 = triatom.cos_theta
        theta_weights_1 = triatom.theta_weights
    else:
        cos_theta_1, theta_weights_1 = gauss_legendre_dvr(-1.0, 1.0, n_theta_1)

    cos_theta_2, theta_weights_2 = gauss_legendre_dvr(-1.0, 1.0, n_theta_2)
    phi, phi_weights = gauss_legendre_dvr(0.0, np.pi, n_phi)
    theta_1 = np.arccos(cos_theta_1)
    theta_2 = np.arccos(cos_theta_2)
    V_basis = prepare_Vmat_BF_atom_triatom(
        basis,
        triatom,
        cos_theta_1,
        theta_weights_1,
        cos_theta_2,
        theta_weights_2,
        phi,
        phi_weights,
    )

    radial_1 = triatom.radial_1
    radial_2 = triatom.radial_2
    if radial_1 is None or radial_2 is None:
        message = "Atom-triatom scattering requires radial PODVR data in TriatomBasis"
        logger.error(message)
        raise ValueError(message)

    def Vgrid(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate PES grids, returning (*grid_shape,) or (n_R, *grid_shape)."""
        return get_Vgrid_atom_triatom(
            pes,
            radial_points,
            radial_1.grids,
            radial_2.grids,
            theta_1,
            theta_2,
            phi,
        )

    def Vmat(radial_points: float | Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
        """Contract PES grids into shape (n_channel, n_channel), optionally preceded by n_R."""
        return get_Vmat_BF(V_basis, Vgrid(radial_points))

    message = f"Running atom-triatom block Jtot={Jtot}, parity={system_parity:+d}, channels={basis.n_channel}, energies={energies.size}"
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
