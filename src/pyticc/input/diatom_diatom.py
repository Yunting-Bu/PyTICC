from pathlib import Path

from loguru import logger

import pyticc.scattering.diatom_diatom as geometry
from pyticc.basis.channel import TruncSpec
from pyticc.constants import CM2AU
from pyticc.input.common import TomlTable, approximation, build_diatom, diatom_symbols, energies, k_cut, propagation, required, section
from pyticc.pes.adiabatic import PESWrapper
from pyticc.result import CoupledStatesResult, ScatteringResult
from pyticc.scattering.solver import solve
from pyticc.system import ScattSystem, reduced_mass


def run(config: TomlTable, base: Path, pes: PESWrapper) -> ScatteringResult | CoupledStatesResult:
    """Run a diatom-diatom calculation from parsed TOML data."""
    if pes.monomer_X is None or pes.monomer_Y is None:
        message = "Diatom-diatom calculation requires both pes.monomer_X and pes.monomer_Y"
        logger.error(message)
        raise ValueError(message)

    basis_X, mass_X = build_diatom(diatom_symbols(config, "diatom_X"), section(config, "basis_X"), pes.monomer_X)
    basis_Y, mass_Y = build_diatom(diatom_symbols(config, "diatom_Y"), section(config, "basis_Y"), pes.monomer_Y)
    quadrature = section(config, "quadrature")
    truncation = section(config, "truncation")
    approx, K_delta = approximation(config)
    system = ScattSystem(
        basis_X,
        basis_Y,
        Jtot=int(required(config, "Jtot")),
        system_parity=int(required(config, "system_parity")),
        approx=approx,
        K_delta=K_delta,
        potential=pes,
        reduced_mass=reduced_mass(mass_X, mass_Y),
    )
    hamiltonian = geometry.build_hamiltonian(
        system,
        trunc=TruncSpec(
            E_X_cut=float(required(truncation, "E_X_cut_cm")) * CM2AU,
            E_Y_cut=float(required(truncation, "E_Y_cut_cm")) * CM2AU,
            K_cut=k_cut(truncation),
        ),
        n_theta_X=int(required(quadrature, "n_theta_X")),
        n_theta_Y=int(required(quadrature, "n_theta_Y")),
        n_phi=int(required(quadrature, "n_phi")),
    )
    return solve(hamiltonian, energies(required(config, "energies_cm"), base), propagation(config))
