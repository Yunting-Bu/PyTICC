from pathlib import Path

from loguru import logger

from pyticc.constants import CM2AU
from pyticc.input.common import (
    TomlTable,
    approximation,
    diatom_symbols,
    energies,
    k_cut,
    potential_grid_settings,
    propagation,
    required,
    section,
)
from pyticc.input.fine_structure_common import bind_diatom_diatom_order, build_fs_diatom
from pyticc.pes.spin_resolved_diatom_diatom import SpinResolvedDiatomDiatomPES
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.potential import prepare_potential
from pyticc.scattering.solver import solve
from pyticc.system import ChannelSpec, ScattSystem, build_ScattSystem, reduced_mass


# ----------------------------------------------------------------------------------------
def build_system(config: TomlTable, base: Path, pes: SpinResolvedDiatomDiatomPES) -> ScattSystem:
    """Build a two-fine-structure-diatom system from TOML data."""
    if config.get("molecule_exchange", 0) != 0:
        raise NotImplementedError("molecule_exchange currently requires the Python API with a shared monomer basis object")
    if pes.monomer_X is None or pes.monomer_Y is None:
        message = "AB+CD fine-structure input requires isolated diatom potentials as pes.monomer_X and pes.monomer_Y"
        logger.error(message)
        raise ValueError(message)

    fine_structure = section(config, "fine_structure")
    monomer_X, mass_X = build_fs_diatom(
        diatom_symbols(config, "diatom_X"),
        section(config, "basis_X"),
        section(fine_structure, "diatom_X"),
        pes.monomer_X,
        base,
    )
    monomer_Y, mass_Y = build_fs_diatom(
        diatom_symbols(config, "diatom_Y"),
        section(config, "basis_Y"),
        section(fine_structure, "diatom_Y"),
        pes.monomer_Y,
        base,
    )
    channels = section(config, "channels")
    approx, K_delta = approximation(config)
    ordered_pes = bind_diatom_diatom_order(config, pes)
    return build_ScattSystem(
        monomer_X,
        monomer_Y,
        two_J=int(required(config, "two_J")),
        system_parity=int(required(config, "system_parity")),
        approx=approx,
        K_delta=K_delta,
        channel=ChannelSpec(
            E_X_cut=float(required(channels, "E_X_cut_cm")) * CM2AU,
            E_Y_cut=float(required(channels, "E_Y_cut_cm")) * CM2AU,
            K_cut=k_cut(channels),
        ),
        potential=ordered_pes,
        reduced_mass=reduced_mass(mass_X, mass_Y),
        magnetic_dipole_coefficient=float(fine_structure.get("magnetic_dipole_coefficient_au_bohr3", 0.0)),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def run(
    config: TomlTable,
    base: Path,
    pes: SpinResolvedDiatomDiatomPES,
) -> ScatteringResult | CoupledStatesResult:
    """Run a two-fine-structure-diatom calculation from parsed TOML data."""
    system = build_system(config, base, pes)
    quadrature = section(config, "quadrature")
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
