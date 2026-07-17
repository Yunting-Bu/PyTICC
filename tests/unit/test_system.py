import pytest

import pyticc as ticc


def test_element_masses_au_preserves_symbol_order() -> None:
    masses = ticc.element_masses_au("H", "F", "Ar")

    assert masses == (
        ticc.element_mass_au("H"),
        ticc.element_mass_au("F"),
        ticc.element_mass_au("Ar"),
    )


def test_element_mass_au_converts_from_amu_to_atomic_units() -> None:
    assert ticc.element_mass_au("H") == pytest.approx(1.00782503223 * 1822.888486209)


def test_element_masses_au_reuses_symbol_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported element symbol"):
        ticc.element_masses_au("H", "Xx")
