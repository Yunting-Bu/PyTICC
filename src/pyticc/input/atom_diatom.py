from pathlib import Path

from loguru import logger

import pyticc.scattering.atom_diatom as geometry
from pyticc.basis.channel import TruncSpec
from pyticc.basis.monomer import AtomSpec
from pyticc.constants import CM2AU
from pyticc.input.common import TomlTable, approximation, build_diatom, diatom_symbols, energies, k_cut, propagation, required, section
from pyticc.pes.wrapper import PESWrapper
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.solver import solve
from pyticc.system import ScattSystem, element_masses_au, reduced_mass


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
