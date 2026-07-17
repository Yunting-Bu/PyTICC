import numpy as np

from pyticc.basis.channel import ChannelBuilder, TruncSpec
from pyticc.basis.kblock import build_cs_blocks, build_nncc_blocks
from pyticc.basis.monomer import AtomSpec, DiatomSpec
from pyticc.system import ScattSystem


def _channels():
    diatom = DiatomSpec(Eint=np.zeros((1, 5)), vmax=0, jmax=4)
    system = ScattSystem(AtomSpec(), diatom, Jtot=4, system_parity=1)
    return ChannelBuilder(system, TruncSpec()).build()


def test_cs_builds_one_owned_block_per_K() -> None:
    channels = _channels()

    blocks = build_cs_blocks(channels)

    assert [block.K_values for block in blocks] == [(0,), (1,), (2,), (3,), (4,)]
    assert [block.owned_K_values for block in blocks] == [(0,), (1,), (2,), (3,), (4,)]
    assert sorted(index for block in blocks for index in block.owned_channel_indices) == list(range(len(channels)))


def test_nncc_delta_one_builds_overlapping_three_K_windows() -> None:
    channels = _channels()

    blocks = build_nncc_blocks(channels, K_delta=1)

    assert [block.K_values for block in blocks] == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert [block.owned_K_values for block in blocks] == [(0, 1), (2,), (3, 4)]
    assert sorted(index for block in blocks for index in block.owned_channel_indices) == list(range(len(channels)))
    assert all(set(block.owned_channel_indices) <= set(block.channel_indices) for block in blocks)
