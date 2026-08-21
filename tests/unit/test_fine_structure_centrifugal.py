import numpy as np

from pyticc.basis.podvr import VibPODVR
from pyticc.fine_structure import FSConstants, build_fs_channels, build_fs_monomer_basis
from pyticc.match.asymptotic import get_Bmat_FS_BF_to_SF


def test_half_integer_parity_boundary_produces_integral_orbital_angular_momentum() -> None:
    vib = VibPODVR(np.array([2.0]), np.array([0.0]), np.ones((1, 1)))
    monomer = build_fs_monomer_basis(vib, (1, 3), 2, 1, FSConstants())
    basis = build_fs_channels(monomer, two_J=1, system_parity=1)

    _, orbital_angular_momenta = get_Bmat_FS_BF_to_SF(basis)

    for channel, orbital_angular_momentum in zip(basis, orbital_angular_momenta, strict=True):
        block = monomer.blocks[channel.block]
        expected = 0 if (block.two_j, block.parity) == (1, 1) else 1 if block.parity == -1 else 2
        assert orbital_angular_momentum == expected
