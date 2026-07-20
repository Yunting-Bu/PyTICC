import numpy as np
import pytest

from pyticc.basis.channel import ChannelBuilder, TruncSpec
from pyticc.basis.monomer import AtomSpec, DiatomSpec
from pyticc.basis.triatom import TriatomBasis
from pyticc.system import ScattSystem


def test_atom_diatom_keeps_internal_angular_momentum_above_J() -> None:
    diatom = DiatomSpec(Eint=np.array([[0.0, 1.0, 2.0]]), vmax=0, jmax=2)
    system = ScattSystem(monomer_X=AtomSpec(), monomer_Y=diatom, Jtot=0, system_parity=1)

    channels = ChannelBuilder(system, TruncSpec()).build()

    assert channels.n_channel == 3
    assert [(channel.j_couple, channel.K) for channel in channels] == [(0, 0), (1, 0), (2, 0)]


def test_channel_string_shows_quantum_numbers_and_energy() -> None:
    diatom = DiatomSpec(Eint=np.array([[0.0]]), vmax=0, jmax=0)
    system = ScattSystem(monomer_X=AtomSpec(), monomer_Y=diatom, Jtot=0, system_parity=1)

    channel = ChannelBuilder(system, TruncSpec()).build()[0]

    assert str(channel) == "Channel[0] X(v=-, j=0) Y(v=0, j=0) j_couple=0 K=0 Jtot=0 parity=+1 E_int=0.0000000000 a.u."


def test_diatom_diatom_filters_K0_for_every_rotational_state() -> None:
    odd_diatom = DiatomSpec(Eint=np.array([[0.0, 0.0]]), vmax=0, jmax=1, jpar=-1)
    even_diatom = DiatomSpec(Eint=np.array([[0.0]]), vmax=0, jmax=0, jpar=1)

    positive_parity = ScattSystem(
        monomer_X=odd_diatom,
        monomer_Y=even_diatom,
        Jtot=1,
        system_parity=1,
    )
    negative_parity = ScattSystem(
        monomer_X=odd_diatom,
        monomer_Y=even_diatom,
        Jtot=1,
        system_parity=-1,
    )

    positive_channels = ChannelBuilder(positive_parity, TruncSpec()).build()
    negative_channels = ChannelBuilder(negative_parity, TruncSpec()).build()

    assert [(channel.j_couple, channel.K) for channel in positive_channels] == [(1, 1)]
    assert [(channel.j_couple, channel.K) for channel in negative_channels] == [(1, 0), (1, 1)]


@pytest.mark.parametrize(
    ("Jtot", "system_parity"),
    [(None, 1), (0, None), (None, None)],
)
def test_field_free_builder_requires_J_and_parity(Jtot: int | None, system_parity: int | None) -> None:
    system = ScattSystem(
        monomer_X=AtomSpec(),
        monomer_Y=AtomSpec(),
        Jtot=Jtot,
        system_parity=system_parity,
    )

    with pytest.raises(ValueError, match="requires Jtot and system_parity"):
        ChannelBuilder(system, TruncSpec()).build()


def test_K_cut_limits_helicity_without_limiting_j_couple() -> None:
    diatom = DiatomSpec(Eint=np.zeros((1, 3)), vmax=0, jmax=2, jpar=1)
    system = ScattSystem(monomer_X=AtomSpec(), monomer_Y=diatom, Jtot=2, system_parity=1)

    channels = ChannelBuilder(system, TruncSpec(K_cut=0)).build()

    assert {channel.j_couple for channel in channels} == {0, 2}
    assert {channel.K for channel in channels} == {0}


def test_atom_triatom_odd_parity_keeps_K0_for_positive_j() -> None:
    triatom = TriatomBasis(Eint=np.array([[0.0], [0.01]]), jmax=1, tmax=0, parity_block_sign=-1)
    system = ScattSystem(monomer_X=AtomSpec(), monomer_Y=triatom, Jtot=1, system_parity=1)

    channels = ChannelBuilder(system, TruncSpec()).build()

    assert [(channel.mis_Y.j, channel.mis_Y.t, channel.K) for channel in channels] == [
        (1, 0, 0),
        (1, 0, 1),
    ]
    assert "Y(t=0, j=1)" in str(channels[0])


def test_atom_triatom_even_parity_keeps_j_zero_and_all_allowed_K() -> None:
    triatom = TriatomBasis(Eint=np.array([[0.0], [0.01]]), jmax=1, tmax=0)
    system = ScattSystem(monomer_X=AtomSpec(), monomer_Y=triatom, Jtot=1, system_parity=-1)

    channels = ChannelBuilder(system, TruncSpec()).build()

    assert [(channel.mis_Y.j, channel.mis_Y.t, channel.K) for channel in channels] == [
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 1),
    ]


def test_atom_triatom_parity_is_independent_of_monomer_order() -> None:
    triatom = TriatomBasis(Eint=np.array([[0.0], [0.01]]), jmax=1, tmax=0, parity_block_sign=-1)
    system = ScattSystem(monomer_X=triatom, monomer_Y=AtomSpec(), Jtot=1, system_parity=1)

    channels = ChannelBuilder(system, TruncSpec(K_cut=0)).build()

    assert [(channel.mis_X.j, channel.mis_X.t, channel.K) for channel in channels] == [(1, 0, 0)]


def test_atom_triatom_requires_matching_parity_block() -> None:
    triatom = TriatomBasis(Eint=np.array([[0.0]]), jmax=0, tmax=0, parity_block_sign=1)
    system = ScattSystem(monomer_X=AtomSpec(), monomer_Y=triatom, Jtot=1, system_parity=1)

    with pytest.raises(ValueError, match="parity_block_sign"):
        ChannelBuilder(system, TruncSpec()).build()
