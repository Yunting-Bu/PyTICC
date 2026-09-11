from pathlib import Path
from typing import cast

from pyticc.constants import AMU2AU, CM2AU, EnergyUnit
from pyticc.fine_structure.atom import build_fs_atom_basis
from pyticc.input.common import TomlTable, approximation, energies, k_cut, potential_grid_settings, propagation, required, section
from pyticc.input.fine_structure_common import integer_sequence
from pyticc.pes.spin_resolved_atom_atom import SpinResolvedAtomAtomPES
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.potential import prepare_potential
from pyticc.scattering.solver import solve
from pyticc.system import ChannelSpec, ScattSystem, build_ScattSystem


def build_system(config: TomlTable, base: Path, pes: SpinResolvedAtomAtomPES) -> ScattSystem:
    """Build two fixed-term atoms from experimental-energy TOML tables.

    Inputs:
        config: TomlTable - fine_structure.atom_X/atom_Y and reduced_mass_amu
        base: Path - input directory; atomic energies are inline values
        pes: SpinResolvedAtomAtomPES - explicitly labeled diabatic callback
    Returns:
        system: ScattSystem - parity-adapted atomic pair, atomic-unit energies
    """
    fs = section(config, "fine_structure")
    monomers = []
    for key in ("atom_X", "atom_Y"):
        values = section(fs, key)
        monomers.append(
            build_fs_atom_basis(
                two_L=int(required(values, "two_L")),
                two_S=int(required(values, "two_S")),
                atom_parity=int(required(values, "atom_parity")),
                two_j_levels=integer_sequence(values, "two_j_levels"),
                level_energies=required(values, "level_energies"),
                unit=cast(EnergyUnit, str(values.get("energy_unit", "cm-1"))),
                energy_zero=None if "energy_zero" not in values else float(values["energy_zero"]),
            )
        )
    channels = section(config, "channels")
    approx, K_delta = approximation(config)
    return build_ScattSystem(
        *monomers,
        two_J=int(required(config, "two_J")),
        system_parity=int(required(config, "system_parity")),
        approx=approx,
        K_delta=K_delta,
        potential=pes,
        reduced_mass=float(required(config, "reduced_mass_amu")) * AMU2AU,
        magnetic_dipole_coefficient=float(config.get("magnetic_dipole_coefficient", 0.0)),
        molecule_exchange=int(config.get("molecule_exchange", 0)),
        channel=ChannelSpec(
            E_X_cut=float(channels.get("E_X_cut_cm", float("inf"))) * CM2AU,
            E_Y_cut=float(channels.get("E_Y_cut_cm", float("inf"))) * CM2AU,
            K_cut=k_cut(channels) if "K_cut" in channels else None,
        ),
    )


def run(config: TomlTable, base: Path, pes: SpinResolvedAtomAtomPES) -> ScatteringResult | CoupledStatesResult:
    """Run atomic scattering from parsed TOML using shared propagation.

    Inputs:
        config: TomlTable - calculation settings and experimental atom levels
        base: Path - input directory for relative energy-list paths
        pes: SpinResolvedAtomAtomPES - orbital diabatic callback in Hartree
    Returns:
        result: ScatteringResult | CoupledStatesResult - matched S matrices
    """
    system = build_system(config, base, pes)
    boundaries, half_steps, processes = potential_grid_settings(config)
    grid = prepare_potential(system, boundaries, half_steps, processes=processes)
    return solve(system, energies(required(config, "energies_cm"), base), grid, propagation(config))
