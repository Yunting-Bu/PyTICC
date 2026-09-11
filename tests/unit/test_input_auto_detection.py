import tomllib
from pathlib import Path

import numpy as np
import pytest

import pyticc as ticc
from pyticc.basis.podvr import VibPODVR
from pyticc.fine_structure import FSConstants, build_fs_monomer_basis
from pyticc.input.driver import _resolve_scattering_type
from pyticc.input.fine_structure_atom_diatom import build_system
from pyticc.input.fine_structure_diatom_diatom import build_system as build_diatom_diatom_system


# ----------------------------------------------------------------------------------------
def _fs_diatom() -> ticc.FSMonomerBasis:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    return build_fs_monomer_basis(vib, (0,), 0, 0, FSConstants())


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def test_build_scatt_system_infers_both_fine_structure() -> None:
    atom = ticc.build_fs_atom_basis(2, 1, -1, (1, 3), (0.0, 10.0))
    diatom = _fs_diatom()
    orbitals = ticc.atom_diatom_orbital_states(atom.two_L, diatom.two_lambda_abs)

    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        return np.zeros((coordinates.shape[1], 1, len(orbitals), len(orbitals)))

    pes = ticc.SpinResolvedAtomDiatomPES(interaction, (1,), orbitals)
    system = ticc.build_ScattSystem(
        atom,
        diatom,
        two_J=1,
        system_parity=1,
        potential=pes,
        reduced_mass=2.0,
    )

    assert system.scattering_type is ticc.ScatteringType.ATOM_DIATOM_BOTH_FS


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def test_toml_type_detection_uses_physics_sections() -> None:
    assert _resolve_scattering_type({"atom": "Ar", "diatom": ["H", "F"]}) is ticc.ScatteringType.ATOM_DIATOM
    assert _resolve_scattering_type({"type": "A+BC", "atom": "Ar", "diatom": ["H", "F"], "electric": {}}) is ticc.ScatteringType.ATOM_DIATOM_ELECTRIC
    assert (
        _resolve_scattering_type(
            {
                "type": "A+BC",
                "atom": "O",
                "diatom": ["O", "H"],
                "fine_structure": {"atom": {}, "diatom": {}},
            }
        )
        is ticc.ScatteringType.ATOM_DIATOM_BOTH_FS
    )
    with pytest.raises(ValueError, match="Combined external-field and fine-structure"):
        _resolve_scattering_type(
            {
                "atom": "O",
                "diatom": ["O", "H"],
                "electric": {},
                "fine_structure": {"atom": {}, "diatom": {}},
            }
        )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def test_both_fine_structure_toml_builds_system(tmp_path: Path) -> None:
    config = {
        "type": "A+BC",
        "atom": "O",
        "diatom": ["O", "H"],
        "two_J": 1,
        "system_parity": 1,
        "basis": {"r": [1.2, 3.5], "n_dvr": 12, "n_podvr": 2, "vmax": 0},
        "channels": {"E_X_cut_cm": 1000.0, "E_Y_cut_cm": 1000.0, "K_cut": "none"},
        "fine_structure": {
            "atom": {
                "two_L": 2,
                "two_S": 1,
                "atom_parity": -1,
                "two_j_levels": [1, 3],
                "level_energies": [0.0, 158.0],
                "energy_unit": "cm-1",
            },
            "diatom": {
                "two_lambda_abs": 0,
                "two_S": 0,
                "two_j_values": [0],
                "reflection_parity": 1,
                "constants": {"unit": "cm-1", "B": 18.5},
            },
            "pes": {
                "two_total_spins": [1],
                "orbital_states": [
                    {"two_lambda_atom": 0, "two_lambda_Y": 0},
                    {"two_lambda_atom": 2, "two_lambda_Y": 0},
                    {"two_lambda_atom": -2, "two_lambda_Y": 0},
                ],
            },
        },
    }
    orbitals = ticc.atom_diatom_orbital_states(2, 0)

    def monomer_Y(r: np.ndarray) -> np.ndarray:
        return 0.1 * (r - 1.8) ** 2

    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        return np.zeros((coordinates.shape[1], 1, len(orbitals), len(orbitals)))

    pes = ticc.SpinResolvedAtomDiatomPES(interaction, monomer_Y=monomer_Y)
    system = build_system(config, tmp_path, pes)

    assert system.scattering_type is ticc.ScatteringType.ATOM_DIATOM_BOTH_FS
    assert isinstance(system.monomer_X, ticc.FSAtomBasis)
    assert isinstance(system.monomer_Y, ticc.FSMonomerBasis)
    assert system.potential.two_total_spins == (1,)
    assert [(state.two_lambda_atom, state.two_lambda_Y) for state in system.potential.orbital_states] == [(0, 0), (2, 0), (-2, 0)]
    assert system.n_channel > 0


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def test_diatom_diatom_toml_binds_electronic_order() -> None:
    root = Path(__file__).parents[2] / "example" / "H2OH_2Pi"
    with (root / "input.toml").open("rb") as stream:
        config = tomllib.load(stream)

    def monomer(r: np.ndarray) -> np.ndarray:
        return 0.1 * (r - 1.8) ** 2

    def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
        return np.zeros((coordinates.shape[1], 1, 2, 2))

    pes = ticc.SpinResolvedDiatomDiatomPES(interaction, monomer_X=monomer, monomer_Y=monomer)
    system = build_diatom_diatom_system(config, root, pes)

    assert system.scattering_type is ticc.ScatteringType.DIATOM_DIATOM_FINE_STRUCTURE
    assert system.potential.two_total_spins == (1,)
    assert [(state.two_lambda_X, state.two_lambda_Y) for state in system.potential.orbital_states] == [
        (0, -2),
        (0, 2),
    ]


# ----------------------------------------------------------------------------------------
