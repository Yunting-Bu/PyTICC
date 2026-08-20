from pathlib import Path

import numpy as np
import pytest

import pyticc as pt
from pyticc.basis.channel import ChannelBasis, build_ChannelBasis
from pyticc.basis.monomer import DiatomSpec


def build_test_basis() -> ChannelBasis:
    diatom = DiatomSpec(Eint=np.array([[0.0, 1.0, 2.0]]))
    system = pt.ScattSystem(monomer_X=pt.AtomSpec(), monomer_Y=diatom, Jtot=0, system_parity=1)
    return build_ChannelBasis(system, pt.ChannelSpec())


def test_open_closed_channels_accept_array_input() -> None:
    basis = build_test_basis()

    result = basis.open_closed([1.0, 1.5, 2.0])

    np.testing.assert_array_equal(result.n_open, [1, 2, 2])
    np.testing.assert_array_equal(result.n_closed, [2, 1, 1])
    np.testing.assert_array_equal(
        result.open_mask,
        [
            [True, False, False],
            [True, True, False],
            [True, True, False],
        ],
    )


def test_open_closed_channels_accept_file_input(tmp_path: Path) -> None:
    path = tmp_path / "total_energies.dat"
    path.write_text("# energy/a.u.\n1.0\n1.5\n2.0\n")
    basis = build_test_basis()

    result = basis.open_closed(path)

    np.testing.assert_array_equal(result.n_open, [1, 2, 2])


def test_total_energies_may_be_negative() -> None:
    basis = build_test_basis()

    result = basis.open_closed([-1.0, 0.0])

    np.testing.assert_array_equal(result.n_open, [0, 0])


@pytest.mark.parametrize("energies", [[], [0.0, np.inf], [[0.0, 1.0]]])
def test_open_closed_channels_reject_invalid_energies(energies: list[float]) -> None:
    basis = build_test_basis()

    with pytest.raises(ValueError):
        basis.open_closed(energies)
