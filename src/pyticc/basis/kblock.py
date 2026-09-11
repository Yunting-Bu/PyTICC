from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from loguru import logger


class HelicityChannel(Protocol):
    """Channel exposing its integer or half-integer BF projection."""

    @property
    def K(self) -> float:
        """Return the physical nonnegative helicity K."""
        ...


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KBlock:
    """
    NNCC propagation block centered at one K.

    Members:
        index: int - sequential index of this propagation block
        center_K: float - integer or half-integer central helicity
        K_delta: int - number of neighboring K blocks included on each side of center_K
        K_values: tuple[float, ...] - helicities included in the propagated channel basis
        channel_indices: tuple[int, ...] - positions of the included channels in the complete channel list
        owned_K_values: tuple[float, ...] - helicities whose scattering results are taken from this block
        owned_channel_indices: tuple[int, ...] - positions of the owned incoming channels in the complete channel list
    """

    index: int
    center_K: float
    K_delta: int
    K_values: tuple[float, ...]
    channel_indices: tuple[int, ...]
    owned_K_values: tuple[float, ...]
    owned_channel_indices: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"KBlock[{self.index}] center_K={self.center_K} K_delta={self.K_delta} "
            f"K_values={self.K_values} N_channel={len(self.channel_indices)} "
            f"owned_K_values={self.owned_K_values} N_owned={len(self.owned_channel_indices)}"
        )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_cs_blocks(channels: Sequence[HelicityChannel]) -> tuple[KBlock, ...]:
    """
    Build independent single-K propagation blocks for the coupled-states approximation.

    Inputs:
        channels: Sequence[HelicityChannel] - complete field-free channel basis

    Returns:
        blocks: tuple[KBlock, ...] - one propagation block for each retained K
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
    return tuple(blocks)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def build_nncc_blocks(channels: Sequence[HelicityChannel], K_delta: int = 1) -> tuple[KBlock, ...]:
    r"""
    Build overlapping NNCC propagation blocks from a complete channel basis.

    Formula:
        A window centered at K_c retains |K-K_c| <= K_delta, with K advancing
        in unit steps from K_min (either integer or half-integer). Incoming
        K is owned by the center clamp(K,first_center,last_center), so every
        incoming channel contributes once. A window spanning the full ladder
        reduces to exact CC. All helicities are dimensionless.

    Inputs:
        channels: Sequence[HelicityChannel] - complete field-free channel basis
        K_delta: int - number of neighboring K blocks included on each side

    Returns:
        blocks: tuple[KBlock, ...] - NNCC propagation blocks
    """
    if not isinstance(K_delta, int) or K_delta < 1:
        message = f"NNCC requires K_delta >= 1, but got K_delta={K_delta}"
        logger.error(message)
        raise ValueError(message)
    if not channels:
        return ()

    Kmin = min(channel.K for channel in channels)
    Kmax = max(channel.K for channel in channels)
    if 2 * Kmin != round(2 * Kmin) or any(channel.K < 0 or channel.K - Kmin != round(channel.K - Kmin) for channel in channels):
        raise ValueError("NNCC channels must belong to one nonnegative integer or half-integer K ladder")
    number_of_K = int(Kmax - Kmin) + 1
    ladder = tuple(Kmin + offset for offset in range(number_of_K))

    if 2 * K_delta + 1 >= number_of_K:
        centers = (ladder[(number_of_K - 1) // 2],)
    else:
        centers = ladder[K_delta : number_of_K - K_delta]

    first_center = centers[0]
    last_center = centers[-1]

    def owner_center(K: float) -> float:
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
        K_values = tuple(K for K in ladder if K_low <= K <= K_high)
        owned_K_values = tuple(K for K in ladder if owner_center(K) == center_K)

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

    return tuple(blocks)


# ----------------------------------------------------------------------------------------
