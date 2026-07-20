from collections.abc import Sequence
from dataclasses import dataclass

from loguru import logger

from pyticc.basis.channel import Channel


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KBlock:
    """
    NNCC propagation block centered at one K.

    Members:
        index: int - sequential index of this propagation block
        center_K: int - central helicity that labels this propagation block
        K_delta: int - number of neighboring K blocks included on each side of center_K
        K_values: tuple[int, ...] - helicities included in the propagated channel basis
        channel_indices: tuple[int, ...] - positions of the included channels in the complete channel list
        owned_K_values: tuple[int, ...] - helicities whose scattering results are taken from this block
        owned_channel_indices: tuple[int, ...] - positions of the owned incoming channels in the complete channel list
    """

    index: int
    center_K: int
    K_delta: int
    K_values: tuple[int, ...]
    channel_indices: tuple[int, ...]
    owned_K_values: tuple[int, ...]
    owned_channel_indices: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"KBlock[{self.index}] center_K={self.center_K} K_delta={self.K_delta} "
            f"K_values={self.K_values} N_channel={len(self.channel_indices)} "
            f"owned_K_values={self.owned_K_values} N_owned={len(self.owned_channel_indices)}"
        )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_cs_blocks(channels: Sequence[Channel]) -> list[KBlock]:
    """
    Build independent single-K propagation blocks for the coupled-states approximation.

    Inputs:
        channels: Sequence[Channel] - complete field-free channel basis

    Returns:
        blocks: list[KBlock] - one propagation block for each retained K
    """
    K_values = sorted({channel.K for channel in channels})
    blocks: list[KBlock] = []
    for block_index, K in enumerate(K_values):
        channel_indices = tuple(index for index, channel in enumerate(channels) if channel.K == K)
        blocks.append(
            KBlock(
                index=block_index,
                center_K=K,
                K_delta=0,
                K_values=(K,),
                channel_indices=channel_indices,
                owned_K_values=(K,),
                owned_channel_indices=channel_indices,
            )
        )
    return blocks


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_nncc_blocks(channels: Sequence[Channel], K_delta: int = 1) -> list[KBlock]:
    """
    Build overlapping NNCC propagation blocks from a complete channel basis.

    Inputs:
        channels: Sequence[Channel] - complete field-free channel basis
        K_delta: int - number of neighboring K blocks included on each side

    Returns:
        blocks: list[KBlock] - NNCC propagation blocks
    """
    if K_delta < 1:
        message = f"NNCC requires K_delta >= 1, but got K_delta={K_delta}"
        logger.error(message)
        raise ValueError(message)
    if not channels:
        return []

    Kmin = min(channel.K for channel in channels)
    Kmax = max(channel.K for channel in channels)
    number_of_K = Kmax - Kmin + 1

    if 2 * K_delta + 1 >= number_of_K:
        centers = ((Kmin + Kmax) // 2,)
    else:
        centers = tuple(range(Kmin + K_delta, Kmax - K_delta + 1))

    first_center = centers[0]
    last_center = centers[-1]

    def owner_center(K: int) -> int:
        """Assign each physical K to the unique NNCC block that owns its result."""
        if K <= first_center:
            return first_center
        if K >= last_center:
            return last_center
        return K

    blocks: list[KBlock] = []
    for block_index, center_K in enumerate(centers):
        K_low = max(Kmin, center_K - K_delta)
        K_high = min(Kmax, center_K + K_delta)
        K_values = tuple(range(K_low, K_high + 1))
        owned_K_values = tuple(K for K in range(Kmin, Kmax + 1) if owner_center(K) == center_K)

        channel_indices = tuple(index for index, channel in enumerate(channels) if K_low <= channel.K <= K_high)
        owned_channel_indices = tuple(index for index, channel in enumerate(channels) if owner_center(channel.K) == center_K)

        blocks.append(
            KBlock(
                index=block_index,
                center_K=center_K,
                K_delta=K_delta,
                K_values=K_values,
                channel_indices=channel_indices,
                owned_K_values=owned_K_values,
                owned_channel_indices=owned_channel_indices,
            )
        )

    return blocks


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    from pyticc.basis.channel import ChannelBuilder, TruncSpec
    from pyticc.basis.monomer import AtomSpec, DiatomSpec
    from pyticc.system import Approx, ScattSystem

    atom = AtomSpec()
    diatom = DiatomSpec(Eint=np.zeros((1, 7)), vmax=0, jmax=6)
    system = ScattSystem(
        monomer_X=atom,
        monomer_Y=diatom,
        Jtot=6,
        system_parity=1,
        approx=Approx.NNCC,
    )
    channels = ChannelBuilder(system, TruncSpec()).build()

    for K_delta in (1, 2):
        print(f"Test case: NNCC with K_delta={K_delta}")
        K_blocks = build_nncc_blocks(channels, K_delta)
        for K_block in K_blocks:
            print(K_block)

        owned_indices = sorted(index for K_block in K_blocks for index in K_block.owned_channel_indices)
        assert owned_indices == list(range(len(channels)))
        assert all(set(K_block.owned_channel_indices) <= set(K_block.channel_indices) for K_block in K_blocks)
        print()
