from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from loguru import logger

from pyticc.system import ScatteringType

if TYPE_CHECKING:
    from pyticc.scattering.hamiltonian import ScattHamiltonian
    from pyticc.scattering.potential import PotentialGrid


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _ScatteringModel:
    """Geometry-specific potential preparation and Hamiltonian construction."""

    prepare_potential: Callable[..., PotentialGrid]
    build_hamiltonian: Callable[..., ScattHamiltonian]


# ----------------------------------------------------------------------------------------
@cache
def _models() -> dict[ScatteringType, _ScatteringModel]:
    """Build the fixed-arrangement scattering-model registry once."""
    from pyticc.scattering.energy_transfer import (
        atom_diatom,
        atom_triatom,
        diabatic_atom_diatom,
        diatom_diatom,
        fine_structure_atom_diatom,
        fine_structure_diatom_diatom,
    )

    return {
        ScatteringType.ATOM_DIATOM: _ScatteringModel(atom_diatom.prepare_potential, atom_diatom.build_hamiltonian),
        ScatteringType.ATOM_DIATOM_ELECTRIC: _ScatteringModel(
            atom_diatom.prepare_potential_electric_sf,
            atom_diatom.build_hamiltonian_electric_sf,
        ),
        ScatteringType.ATOM_DIATOM_FINE_STRUCTURE: _ScatteringModel(
            fine_structure_atom_diatom.prepare_potential,
            fine_structure_atom_diatom.build_hamiltonian,
        ),
        ScatteringType.ATOM_DIATOM_DIABATIC: _ScatteringModel(
            diabatic_atom_diatom.prepare_potential,
            diabatic_atom_diatom.build_hamiltonian,
        ),
        ScatteringType.DIATOM_DIATOM: _ScatteringModel(diatom_diatom.prepare_potential, diatom_diatom.build_hamiltonian),
        ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE: _ScatteringModel(
            fine_structure_diatom_diatom.prepare_potential,
            fine_structure_diatom_diatom.build_hamiltonian,
        ),
        ScatteringType.ATOM_TRIATOM: _ScatteringModel(atom_triatom.prepare_potential, atom_triatom.build_hamiltonian),
    }


# ----------------------------------------------------------------------------------------
def get_scattering_model(scattering_type: ScatteringType) -> _ScatteringModel:
    """Return the fixed-arrangement implementation selected by the user."""
    model = _models().get(scattering_type)
    if model is None:
        message = f"{scattering_type.value} does not use a fixed-arrangement PotentialGrid"
        logger.error(message)
        raise TypeError(message)
    return model


# ----------------------------------------------------------------------------------------
