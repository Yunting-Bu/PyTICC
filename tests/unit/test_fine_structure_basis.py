import numpy as np
import pytest

from pyticc.fine_structure import FSState, build_primitive_states, parity_pair


def test_build_primitive_states_enforces_omega_relation() -> None:
    states = build_primitive_states(v_values=(0,), two_j_values=(1, 3), two_lambda_abs=2, two_S=1)

    assert len(states) == 6
    assert all(state.two_omega == state.two_lambda + state.two_sigma for state in states)
    assert all(abs(state.two_omega) <= state.two_j for state in states)


def test_inversion_partner_reverses_signed_projections() -> None:
    state = FSState(v=2, two_j=3, two_omega=1, two_lambda=2, two_S=1, two_sigma=-1)

    assert state.partner == FSState(v=2, two_j=3, two_omega=-1, two_lambda=-2, two_S=1, two_sigma=1)
    assert state.partner.partner == state
    assert np.isclose(parity_pair(state).normalization, 1.0 / np.sqrt(2.0))


def test_sigma_singlet_is_a_self_partner() -> None:
    state = FSState(v=0, two_j=0, two_omega=0, two_lambda=0, two_S=0, two_sigma=0)

    pair = parity_pair(state)
    assert pair.state == pair.partner == state
    assert pair.normalization == 1.0


def test_invalid_primitive_state_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"Omega = Lambda \+ Sigma"):
        FSState(v=0, two_j=1, two_omega=-1, two_lambda=2, two_S=1, two_sigma=-1)
