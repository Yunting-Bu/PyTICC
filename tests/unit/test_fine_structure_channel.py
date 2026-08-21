import numpy as np

from pyticc.basis.podvr import VibPODVR
from pyticc.fine_structure import FSConstants, build_fs_channels, build_fs_monomer_basis


def test_half_integer_channels_use_half_integer_K_ladder() -> None:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    monomer = build_fs_monomer_basis(vib, (1, 3), 2, 1, FSConstants(A=0.01, B=0.001))
    basis = build_fs_channels(monomer, two_J=3, system_parity=1)

    assert basis.n_channel > 0
    assert {channel.two_K for channel in basis} == {1, 3}


def test_integer_K_zero_is_selected_by_total_and_monomer_parity() -> None:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    monomer = build_fs_monomer_basis(vib, (0, 2), 0, 0, FSConstants(B=0.001))
    plus = build_fs_channels(monomer, two_J=2, system_parity=1)

    for channel in plus:
        if channel.two_K == 0:
            block = monomer.blocks[channel.block]
            assert plus.system_parity * block.parity * (-1) ** ((plus.two_J + block.two_j) // 2) == 1
