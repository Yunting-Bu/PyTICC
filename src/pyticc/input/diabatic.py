from pathlib import Path

from loguru import logger

from pyticc.basis.monomer import AtomSpec, prepare_DiabaticDiatom
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
    state_int,
)
from pyticc.pes.diabatic import DiabaticPESWrapper
from pyticc.result import ScatteringResult
from pyticc.scattering.potential import prepare_potential
from pyticc.scattering.solver import solve
from pyticc.system import Approx, ChannelSpec, ScatteringType, build_ScattSystem, element_masses_au, reduced_mass


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
    diatom = prepare_DiabaticDiatom(
        pes.monomer_values,
        n_state=pes.n_state,
        r=(interval[0], interval[1]),
        n_dvr=int(required(values, "n_dvr")),
        n_podvr=state_int(values, "n_podvr", pes.n_state),
        vmax=state_int(values, "vmax", pes.n_state),
        jmax=state_int(values, "jmax", pes.n_state),
        mass=mass,
    )
    quadrature = section(config, "quadrature")
    channels = section(config, "channels")
    system = build_ScattSystem(
        AtomSpec(),
        diatom,
        scattering_type=ScatteringType.ATOM_DIATOM_DIABATIC,
        Jtot=int(required(config, "Jtot")),
        system_parity=int(required(config, "system_parity")),
        channel=ChannelSpec(
            vmin_Y=state_int(channels, "vmin_Y", pes.n_state, 0),
            exchange_parity_Y=state_int(channels, "exchange_parity_Y", pes.n_state, 0),
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
    result = solve(system, energies(required(config, "energies_cm"), base), potential_grid, propagation(config))
    if not isinstance(result, ScatteringResult):
        message = "Diabatic atom-diatom solver returned a coupled-states result"
        logger.error(message)
        raise TypeError(message)
    return result


# ----------------------------------------------------------------------------------------
