from pathlib import Path

from loguru import logger

import pyticc.scattering.diabatic_atom_diatom as geometry
from pyticc.basis.channel import TruncSpec
from pyticc.basis.dvr import build_SineDVR
from pyticc.basis.monomer import AtomSpec, build_DiabaticDiatomBasis
from pyticc.constants import CM2AU
from pyticc.input.common import TomlTable, approximation, diatom_symbols, energies, k_cut, propagation, required, section, state_int
from pyticc.pes.diabatic import DiabaticPESWrapper
from pyticc.result import ScatteringResult
from pyticc.scattering.solver import solve
from pyticc.system import Approx, ScattSystem, element_masses_au, reduced_mass


# ----------------------------------------------------------------------------------------
def run(config: TomlTable, base: Path, pes: DiabaticPESWrapper) -> ScatteringResult:
    """Run a diabatic atom-diatom calculation from parsed TOML data."""
    approx, _ = approximation(config)
    if approx is not Approx.EXACT:
        message = f"Diabatic atom-diatom calculations currently require approximation method 'exact', but got {approx.value!r}"
        logger.error(message)
        raise ValueError(message)

    values = section(config, "basis")
    interval = tuple(float(value) for value in required(values, "r"))
    if len(interval) != 2:
        message = f"basis r must contain the two DVR boundaries, but got {interval}"
        logger.error(message)
        raise ValueError(message)

    atom_mass = element_masses_au(str(required(config, "atom")))[0]
    mass_1, mass_2 = element_masses_au(*diatom_symbols(config, "diatom"))
    diatom_mass = mass_1 + mass_2
    mass = reduced_mass(mass_1, mass_2)
    dvrs = tuple(
        build_SineDVR(interval[0], interval[1], int(required(values, "n_dvr")), mass, pes.monomer_state(state)) for state in range(pes.n_state)
    )
    diatom = build_DiabaticDiatomBasis(
        dvrs,
        n_podvr=state_int(values, "n_podvr", pes.n_state),
        vmax=state_int(values, "vmax", pes.n_state),
        jmax=state_int(values, "jmax", pes.n_state),
        mass=mass,
        vmin=state_int(values, "vmin", pes.n_state, 0),
        jpar=state_int(values, "jpar", pes.n_state, 0),
    )
    system = ScattSystem(
        AtomSpec(),
        diatom,
        Jtot=int(required(config, "Jtot")),
        system_parity=int(required(config, "system_parity")),
        potential=pes,
        reduced_mass=reduced_mass(atom_mass, diatom_mass),
    )
    quadrature = section(config, "quadrature")
    truncation = section(config, "truncation")
    hamiltonian = geometry.build_hamiltonian(
        system,
        trunc=TruncSpec(
            E_Y_cut=float(required(truncation, "E_Y_cut_cm")) * CM2AU,
            K_cut=k_cut(truncation),
        ),
        n_theta=int(required(quadrature, "n_theta")),
    )
    result = solve(hamiltonian, energies(required(config, "energies_cm"), base), propagation(config))
    if not isinstance(result, ScatteringResult):
        message = "Diabatic atom-diatom solver returned a coupled-states result"
        logger.error(message)
        raise TypeError(message)
    return result


# ----------------------------------------------------------------------------------------
