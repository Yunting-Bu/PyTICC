from pathlib import Path

from loguru import logger

from pyticc.constants import CM2AU
from pyticc.input.common import (
    TomlTable,
    approximation,
    build_diatom,
    diatom_symbols,
    energies,
    k_cut,
    potential_grid_settings,
    propagation,
    required,
    section,
)
from pyticc.pes.adiabatic import PESWrapper
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.potential import prepare_potential
from pyticc.scattering.solver import solve
from pyticc.system import ChannelSpec, ScatteringType, build_ScattSystem, reduced_mass


# ----------------------------------------------------------------------------------------
def run(config: TomlTable, base: Path, pes: PESWrapper) -> ScatteringResult | CoupledStatesResult:
    """Run a diatom-diatom calculation from parsed TOML data."""
    if config.get("molecule_exchange", 0) != 0:
        raise NotImplementedError("molecule_exchange currently requires the Python API with a shared monomer basis object")
    if pes.monomer_X is None or pes.monomer_Y is None:
        message = "Diatom-diatom calculation requires both pes.monomer_X and pes.monomer_Y"
        logger.error(message)
        raise ValueError(message)

    basis_X, mass_X = build_diatom(diatom_symbols(config, "diatom_X"), section(config, "basis_X"), pes.monomer_X)
    basis_Y, mass_Y = build_diatom(diatom_symbols(config, "diatom_Y"), section(config, "basis_Y"), pes.monomer_Y)
    quadrature = section(config, "quadrature")
    channels = section(config, "channels")
    approx, K_delta = approximation(config)
    system = build_ScattSystem(
        basis_X,
        basis_Y,
        scattering_type=ScatteringType.DIATOM_DIATOM,
        Jtot=int(required(config, "Jtot")),
        system_parity=int(required(config, "system_parity")),
        approx=approx,
        K_delta=K_delta,
        channel=ChannelSpec(
            vmin_X=int(channels.get("vmin_X", 0)),
            vmin_Y=int(channels.get("vmin_Y", 0)),
            exchange_parity_X=int(channels.get("exchange_parity_X", 0)),
            exchange_parity_Y=int(channels.get("exchange_parity_Y", 0)),
            E_X_cut=float(required(channels, "E_X_cut_cm")) * CM2AU,
            E_Y_cut=float(required(channels, "E_Y_cut_cm")) * CM2AU,
            K_cut=k_cut(channels),
        ),
        potential=pes,
        reduced_mass=reduced_mass(mass_X, mass_Y),
    )
    boundaries, half_steps, processes = potential_grid_settings(config)
    potential_grid = prepare_potential(
        system,
        boundaries,
        half_steps,
        n_theta_X=int(required(quadrature, "n_theta_X")),
        n_theta_Y=int(required(quadrature, "n_theta_Y")),
        n_phi=int(required(quadrature, "n_phi")),
        processes=processes,
    )
    return solve(system, energies(required(config, "energies_cm"), base), potential_grid, propagation(config))


# ----------------------------------------------------------------------------------------
