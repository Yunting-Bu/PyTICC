from pathlib import Path
from typing import cast

from loguru import logger

from pyticc.basis.monomer import AtomSpec, prepare_DiatomElectric
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
    resolve_path,
    section,
)
from pyticc.pes.adiabatic import PESWrapper
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.potential import prepare_potential
from pyticc.scattering.solver import solve
from pyticc.system import Approx, ChannelSpec, ScatteringType, build_ScattSystem, element_masses_au, reduced_mass

# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def run_electric(config: TomlTable, base: Path, pes: PESWrapper) -> ScatteringResult:
    r"""
    Run an exact electric-field atom-diatom calculation from parsed TOML data.

    Formula:
        For a dc electric field along SF-Z, channels are

        |eta;M> = |phi_{alpha m}(E)> |l m_l>,

        M = m + m_l,

        with l=0,...,lmax and m_l=-l,...,l. The dressed monomer states
        |phi_{alpha m}(E)> are obtained from the electric-response CSV, and the
        resulting Electric-SF Hamiltonian is propagated with exact coupled
        channels.

    Inputs:
        config: TomlTable - parsed electric-atom-diatom TOML data
        base: Path - directory containing the input file
        pes: PESWrapper - scalar atom-diatom interaction and zero-field
            diatomic monomer PES

    Returns:
        result: ScatteringResult - exact fixed-M Electric-SF scattering result
    """
    if "approximation" in config:
        approx, _ = approximation(config)
        if approx is not Approx.EXACT:
            message = f"Electric atom-diatom calculations require exact coupled channels, but got {approx.value!r}"
            logger.error(message)
            raise ValueError(message)

    potential = pes.monomer_Y
    if potential is None:
        message = "Electric atom-diatom calculation requires the diatomic monomer potential as pes.monomer_Y"
        logger.error(message)
        raise ValueError(message)

    basis_values = section(config, "basis")
    electric_values = section(config, "electric")
    quadrature = section(config, "quadrature")
    channels = section(config, "channels")
    interval = tuple(float(value) for value in required(basis_values, "r"))
    if len(interval) != 2:
        message = f"basis r must contain the two DVR boundaries, but got {interval}"
        logger.error(message)
        raise ValueError(message)

    mass_1, mass_2 = element_masses_au(*diatom_symbols(config, "diatom"))
    diatom_mass = mass_1 + mass_2
    diatom_reduced_mass = reduced_mass(mass_1, mass_2)
    atom_mass = element_masses_au(str(required(config, "atom")))[0]
    M = int(required(config, "M"))
    lmax = int(required(basis_values, "lmax"))
    n_podvr = int(required(basis_values, "n_podvr"))
    jmax = int(required(basis_values, "jmax"))
    electric_diatom = prepare_DiatomElectric(
        potential,
        resolve_path(base, str(required(electric_values, "response_csv"))),
        r=(interval[0], interval[1]),
        n_dvr=int(required(basis_values, "n_dvr")),
        electric_strength=float(required(electric_values, "strength_au")),
        n_podvr=n_podvr,
        jmax=jmax,
        M=M,
        lmax=lmax,
        n_alpha=int(required(basis_values, "n_alpha")),
        mass=diatom_reduced_mass,
    )
    system = build_ScattSystem(
        AtomSpec(),
        electric_diatom,
        scattering_type=ScatteringType.ATOM_DIATOM_ELECTRIC,
        M=M,
        lmax=lmax,
        channel=ChannelSpec(E_Y_cut=float(required(channels, "E_Y_cut_cm")) * CM2AU),
        potential=pes,
        reduced_mass=reduced_mass(atom_mass, diatom_mass),
    )
    boundaries, half_steps, processes = potential_grid_settings(config)
    potential_grid = prepare_potential(
        system,
        boundaries,
        half_steps,
        n_theta_r=int(required(quadrature, "n_theta_r")),
        n_theta_R=int(required(quadrature, "n_theta_R")),
        n_delta=int(required(quadrature, "n_delta")),
        delta_symmetry=cast(bool, quadrature.get("delta_symmetry", True)),
        processes=processes,
    )
    result = solve(system, energies(required(config, "energies_cm"), base), potential_grid, propagation(config))
    if not isinstance(result, ScatteringResult):
        message = "Electric atom-diatom solver returned a coupled-states result"
        logger.error(message)
        raise TypeError(message)
    return result


# ----------------------------------------------------------------------------------------
def run(config: TomlTable, base: Path, pes: PESWrapper) -> ScatteringResult | CoupledStatesResult:
    """Run an atom-diatom calculation from parsed TOML data."""
    potential = pes.monomer_Y
    if potential is None:
        message = "Atom-diatom calculation requires the diatomic monomer potential as pes.monomer_Y"
        logger.error(message)
        raise ValueError(message)

    basis_values = section(config, "basis")
    quadrature = section(config, "quadrature")
    channels = section(config, "channels")
    diatom, diatom_mass = build_diatom(diatom_symbols(config, "diatom"), basis_values, potential)
    atom_mass = element_masses_au(str(required(config, "atom")))[0]
    approx, K_delta = approximation(config)
    system = build_ScattSystem(
        AtomSpec(),
        diatom,
        scattering_type=ScatteringType.ATOM_DIATOM,
        Jtot=int(required(config, "Jtot")),
        system_parity=int(required(config, "system_parity")),
        approx=approx,
        K_delta=K_delta,
        channel=ChannelSpec(
            vmin_Y=int(channels.get("vmin_Y", 0)),
            exchange_parity_Y=int(channels.get("exchange_parity_Y", 0)),
            E_Y_cut=float(required(channels, "E_Y_cut_cm")) * CM2AU,
            K_cut=k_cut(channels),
        ),
        potential=pes,
        reduced_mass=reduced_mass(atom_mass, diatom_mass),
    )
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
