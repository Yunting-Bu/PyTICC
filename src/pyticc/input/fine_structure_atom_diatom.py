from pathlib import Path
from typing import cast

from loguru import logger

from pyticc.constants import CM2AU, EnergyUnit
from pyticc.fine_structure import build_fs_atom_basis
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
from pyticc.input.fine_structure_common import bind_atom_diatom_order, build_fs_diatom, integer_sequence
from pyticc.pes.spin_resolved_atom_diatom import SpinResolvedAtomDiatomPES
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.potential import prepare_potential
from pyticc.scattering.solver import solve
from pyticc.system import ChannelSpec, ScattSystem, build_ScattSystem, element_masses_au, reduced_mass


# ----------------------------------------------------------------------------------------
def _fine_structure_sections(config: TomlTable) -> tuple[TomlTable, TomlTable]:
    """Return the atomic and diatomic fine-structure input tables."""
    values = section(config, "fine_structure")
    atom = section(values, "atom")
    diatom = section(values, "diatom")
    return atom, diatom


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_system(config: TomlTable, base: Path, pes: SpinResolvedAtomDiatomPES) -> ScattSystem:
    """Build a fine-structure atom plus fine-structure diatom system from TOML data."""
    monomer_potential = pes.monomer_Y
    if monomer_potential is None:
        message = "A+BC both-fine-structure input requires the isolated diatom potential as pes.monomer_Y"
        logger.error(message)
        raise ValueError(message)

    atom_values, diatom_values = _fine_structure_sections(config)
    channels = section(config, "channels")
    symbols = diatom_symbols(config, "diatom")
    atom_mass = element_masses_au(str(required(config, "atom")))[0]
    atom = build_fs_atom_basis(
        two_L=int(required(atom_values, "two_L")),
        two_S=int(required(atom_values, "two_S")),
        atom_parity=int(required(atom_values, "atom_parity")),
        two_j_levels=integer_sequence(atom_values, "two_j_levels"),
        level_energies=required(atom_values, "level_energies"),
        unit=cast(EnergyUnit, str(atom_values.get("energy_unit", "cm-1"))),
        energy_zero=None if "energy_zero" not in atom_values else float(atom_values["energy_zero"]),
    )
    diatom, diatom_mass = build_fs_diatom(symbols, section(config, "basis"), diatom_values, monomer_potential, base)
    ordered_pes = bind_atom_diatom_order(config, pes)
    approx, K_delta = approximation(config)
    return build_ScattSystem(
        atom,
        diatom,
        two_J=int(required(config, "two_J")),
        system_parity=int(required(config, "system_parity")),
        approx=approx,
        K_delta=K_delta,
        channel=ChannelSpec(
            E_X_cut=float(channels.get("E_X_cut_cm", float("inf"))) * CM2AU,
            E_Y_cut=float(required(channels, "E_Y_cut_cm")) * CM2AU,
            K_cut=k_cut(channels),
        ),
        potential=ordered_pes,
        reduced_mass=reduced_mass(atom_mass, diatom_mass),
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def run(
    config: TomlTable,
    base: Path,
    pes: SpinResolvedAtomDiatomPES,
) -> ScatteringResult | CoupledStatesResult:
    """Run a fine-structure atom plus fine-structure diatom calculation."""
    system = build_system(config, base, pes)
    quadrature = section(config, "quadrature")
    boundaries, half_steps, processes = potential_grid_settings(config)
    potential_grid = prepare_potential(
        system,
        boundaries,
        half_steps,
        n_theta=int(required(quadrature, "n_theta")),
        processes=processes,
    )
    return solve(system, energies(required(config, "energies_cm"), base), potential_grid, propagation(config))


# ----------------------------------------------------------------------------------------
