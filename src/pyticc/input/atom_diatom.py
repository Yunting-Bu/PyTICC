from pathlib import Path
from typing import cast

from loguru import logger

import pyticc.scattering.atom_diatom as geometry
from pyticc.basis.channel import TruncSpec
from pyticc.basis.dvr import build_SineDVR
from pyticc.basis.monomer import AtomSpec, build_DiatomElectricBasis
from pyticc.constants import CM2AU
from pyticc.input.common import (
    TomlTable,
    approximation,
    build_diatom,
    diatom_symbols,
    energies,
    k_cut,
    propagation,
    required,
    resolve_path,
    section,
)
from pyticc.pes.adiabatic import PESWrapper
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.solver import solve
from pyticc.system import Approx, ScattSystem, element_masses_au, reduced_mass

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
    truncation = section(config, "truncation")
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
    dvr = build_SineDVR(
        interval[0],
        interval[1],
        int(required(basis_values, "n_dvr")),
        diatom_reduced_mass,
        potential,
    )
    electric_diatom = build_DiatomElectricBasis(
        dvr,
        resolve_path(base, str(required(electric_values, "response_csv"))),
        electric_strength=float(required(electric_values, "strength_au")),
        n_podvr=n_podvr,
        jmax=jmax,
        M=M,
        lmax=lmax,
        n_alpha=int(required(basis_values, "n_alpha")),
        mass=diatom_reduced_mass,
    )
    system = ScattSystem(
        AtomSpec(),
        electric_diatom,
        M=M,
        potential=pes,
        reduced_mass=reduced_mass(atom_mass, diatom_mass),
    )
    hamiltonian = geometry.build_hamiltonian_electric_sf(
        system,
        lmax=lmax,
        E_cut=float(required(truncation, "E_Y_cut_cm")) * CM2AU,
        n_theta_r=int(required(quadrature, "n_theta_r")),
        n_theta_R=int(required(quadrature, "n_theta_R")),
        n_delta=int(required(quadrature, "n_delta")),
        delta_symmetry=cast(bool, quadrature.get("delta_symmetry", True)),
    )
    result = solve(hamiltonian, energies(required(config, "energies_cm"), base), propagation(config))
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
    truncation = section(config, "truncation")
    diatom, diatom_mass = build_diatom(diatom_symbols(config, "diatom"), basis_values, potential)
    atom_mass = element_masses_au(str(required(config, "atom")))[0]
    approx, K_delta = approximation(config)
    system = ScattSystem(
        AtomSpec(),
        diatom,
        Jtot=int(required(config, "Jtot")),
        system_parity=int(required(config, "system_parity")),
        approx=approx,
        K_delta=K_delta,
        potential=pes,
        reduced_mass=reduced_mass(atom_mass, diatom_mass),
    )
    hamiltonian = geometry.build_hamiltonian(
        system,
        trunc=TruncSpec(
            E_Y_cut=float(required(truncation, "E_Y_cut_cm")) * CM2AU,
            K_cut=k_cut(truncation),
        ),
        n_theta=int(required(quadrature, "n_theta")),
    )
    return solve(hamiltonian, energies(required(config, "energies_cm"), base), propagation(config))


# ----------------------------------------------------------------------------------------
