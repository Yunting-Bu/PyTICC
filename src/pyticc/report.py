from collections.abc import Sequence
from typing import TypeAlias, cast

import numpy as np

from pyticc.basis.channel import Channel, ChannelBasis, ChannelBasisElectricSF
from pyticc.basis.kblock import KBlock
from pyticc.basis.monomer import DiabaticDiatomBasis, DiatomBasis
from pyticc.constants import AU2CM
from pyticc.energy import EnergyInput, get_Etot
from pyticc.result import CoupledStatesResult, KBlockResult, ScatteringResult
from pyticc.system import MolInnerState

QuantumSelection: TypeAlias = int | range | Sequence[int] | None
EnergySelection: TypeAlias = int | slice | Sequence[int] | None


# ----------------------------------------------------------------------------------------
def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    header = "  ".join(value.rjust(width) for value, width in zip(headers, widths, strict=True))
    separator = "  ".join("-" * width for width in widths)
    body = ["  ".join(value.rjust(width) for value, width in zip(row, widths, strict=True)) for row in rows]
    return "\n".join((header, separator, *body))


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def rovib_levels(basis: DiatomBasis | DiabaticDiatomBasis) -> str:
    """
    Format the retained diatomic rovibrational levels as a text table.

    Electronic-state labels are included for a diabatic basis. Levels are
    sorted by internal energy, and energies are converted to cm-1.

    Inputs:
        basis: DiatomBasis | DiabaticDiatomBasis - diatomic basis to report

    Returns:
        output: str - formatted rovibrational-level table
    """
    states = sorted(basis.mis_iter(float("inf")), key=lambda state: state.Eint)
    include_electronic_state = isinstance(basis, DiabaticDiatomBasis)
    headers = ["state", "v", "j", "E_int/cm-1"] if include_electronic_state else ["v", "j", "E_int/cm-1"]
    rows: list[list[str]] = []
    for state in states:
        row = []
        if include_electronic_state:
            row.append(str(state.electronic_state))
        row.extend((str(state.v), str(state.j), f"{state.Eint * AU2CM:.6f}"))
        rows.append(row)
    return _table(headers, rows)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _is_atom_state(state: MolInnerState) -> bool:
    return state.v is None and state.t is None and state.electronic_state is None and state.j == 0


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _state_columns(states: Sequence[MolInnerState], suffix: str = "") -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    if any(state.electronic_state is not None for state in states):
        columns.append((f"state{suffix}", "electronic_state"))
    if any(state.v is not None for state in states):
        columns.append((f"v{suffix}", "v"))
    if any(state.t is not None for state in states):
        columns.append((f"t{suffix}", "t"))
    columns.append((f"j{suffix}", "j"))
    return columns


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _state_value(state: MolInnerState, attribute: str) -> str:
    value = getattr(state, attribute)
    return "-" if value is None else str(value)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def channels(basis: ChannelBasis | ChannelBasisElectricSF) -> str:
    """
    Format the channel quantum numbers and internal energies as a text table.

    Only quantum numbers belonging to active monomers are included. Electronic
    states are shown when present, while Jtot and parity are omitted.

    Inputs:
        basis: ChannelBasis | ChannelBasisElectricSF - channel basis to report

    Returns:
        output: str - formatted channel table with energies in cm-1
    """
    if isinstance(basis, ChannelBasisElectricSF):
        rows = [
            [
                str(index),
                str(channel.alpha),
                str(channel.m),
                str(channel.l),
                str(channel.m_l),
                f"{channel.E_int * AU2CM:.6f}",
            ]
            for index, channel in enumerate(basis, start=1)
        ]
        return _table(("n", "alpha", "m", "l", "m_l", "E_int/cm-1"), rows)

    if basis.n_channel == 0:
        return _table(("n", "K", "E_int/cm-1"), ())

    states_X = [channel.mis_X for channel in basis]
    states_Y = [channel.mis_Y for channel in basis]
    active_X = not all(_is_atom_state(state) for state in states_X)
    active_Y = not all(_is_atom_state(state) for state in states_Y)
    columns: list[tuple[str, str, str]] = []

    if active_X and active_Y:
        columns.extend((header, "X", attribute) for header, attribute in _state_columns(states_X, "_X"))
        columns.extend((header, "Y", attribute) for header, attribute in _state_columns(states_Y, "_Y"))
        include_coupled_j = True
    else:
        side = "X" if active_X else "Y"
        states = states_X if active_X else states_Y
        columns.extend((header, side, attribute) for header, attribute in _state_columns(states))
        include_coupled_j = any(channel.j_couple != (channel.mis_X.j if active_X else channel.mis_Y.j) for channel in basis)

    headers = ["n", *(header for header, _, _ in columns)]
    if include_coupled_j:
        headers.append("j_couple")
    headers.extend(("K", "E_int/cm-1"))

    rows: list[list[str]] = []
    for index, channel in enumerate(basis, start=1):
        row = [str(index)]
        for _, side, attribute in columns:
            state = channel.mis_X if side == "X" else channel.mis_Y
            row.append(_state_value(state, attribute))
        if include_coupled_j:
            row.append(str(channel.j_couple))
        row.extend((str(channel.K), f"{channel.E_int * AU2CM:.6f}"))
        rows.append(row)
    return _table(headers, rows)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def open_closed(basis: ChannelBasis | ChannelBasisElectricSF, energies: EnergyInput) -> str:
    """
    Format open and closed channel counts at each total energy.

    Inputs:
        basis: ChannelBasis | ChannelBasisElectricSF - channel basis used to
            classify thresholds
        energies: EnergyInput - total energies in atomic units, or a path to a
            one-column energy file

    Returns:
        output: str - formatted channel-count table with energies in cm-1
    """
    values = get_Etot(energies)
    counts = basis.open_closed(values)
    rows = [
        [str(index), f"{energy * AU2CM:.8f}", str(n_open), str(n_closed)]
        for index, (energy, n_open, n_closed) in enumerate(
            zip(values, counts.n_open, counts.n_closed, strict=True),
            start=1,
        )
    ]
    return _table(("n", "Etot/cm-1", "open", "closed"), rows)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def k_blocks(blocks: Sequence[KBlock | KBlockResult]) -> str:
    """
    Format CS or NNCC K-block membership and ownership information.

    Inputs:
        blocks: Sequence[KBlock | KBlockResult] - K blocks or completed block
            results to report

    Returns:
        output: str - formatted K-block table
    """
    values = [item.block if isinstance(item, KBlockResult) else item for item in blocks]
    rows = [
        [
            str(block.index),
            str(block.center_K),
            ",".join(str(value) for value in block.K_values),
            str(len(block.channel_indices)),
            ",".join(str(value) for value in block.owned_K_values),
            str(len(block.owned_channel_indices)),
        ]
        for block in values
    ]
    return _table(("block", "center_K", "K_values", "channels", "owned_K", "owned_channels"), rows)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _selection(value: QuantumSelection, name: str) -> set[int] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, range, integer sequence, or None")
    if isinstance(value, int):
        return {value}
    if isinstance(value, str | bytes):
        raise TypeError(f"{name} must be an integer, range, integer sequence, or None")
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, range, integer sequence, or None") from error
    if any(not isinstance(item, int) or isinstance(item, bool) for item in values):
        raise TypeError(f"{name} must contain only integers")
    return set(values)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _energy_indices(selection: EnergySelection, size: int) -> tuple[int, ...]:
    available = np.arange(size, dtype=np.int64)
    if selection is None:
        return tuple(int(index) for index in available)
    if isinstance(selection, bool):
        raise TypeError("energy_indices must be an integer, slice, integer sequence, or None")
    if isinstance(selection, int):
        selection = [selection]
    if isinstance(selection, str | bytes):
        raise TypeError("energy_indices must be an integer, slice, integer sequence, or None")
    try:
        selected = np.atleast_1d(available[selection])
    except (IndexError, TypeError) as error:
        raise type(error)(f"Invalid energy_indices {selection!r}: {error}") from error
    return tuple(int(index) for index in selected)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _internal_state(channel: Channel) -> MolInnerState:
    states = tuple(state for state in (channel.mis_X, channel.mis_Y) if not _is_atom_state(state))
    if len(states) != 1 or states[0].v is None:
        raise ValueError("smatrix currently requires an atom-diatom channel basis with one vibrational quantum number")
    return states[0]


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _matches(state: MolInnerState, electronic: set[int] | None, v: set[int] | None, j: set[int] | None) -> bool:
    return (electronic is None or state.electronic_state in electronic) and (v is None or state.v in v) and (j is None or state.j in j)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _orbital_angular_momentum(value: float) -> str:
    rounded = round(value)
    if not np.isclose(value, rounded, rtol=0.0, atol=1.0e-8):
        raise ValueError(f"Orbital angular momentum must be integral, but got L={value}")
    return str(rounded)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _smatrix_electric(result: ScatteringResult, energy_indices: EnergySelection) -> str:
    basis = cast(ChannelBasisElectricSF, result.basis)
    rows: list[list[str]] = []
    for energy_index in _energy_indices(energy_indices, result.Etot.size):
        indices = result.open_channel_indices[energy_index]
        matrix = result.Smat[energy_index]
        for incoming, incoming_global in enumerate(indices):
            initial = basis[int(incoming_global)]
            for outgoing, outgoing_global in enumerate(indices):
                final = basis[int(outgoing_global)]
                value = matrix[outgoing, incoming]
                rows.append(
                    [
                        f"{result.Etot[energy_index] * AU2CM:.8f}",
                        str(initial.alpha),
                        str(initial.m),
                        str(initial.l),
                        str(initial.m_l),
                        str(final.alpha),
                        str(final.m),
                        str(final.l),
                        str(final.m_l),
                        f"{value.real:.16E}",
                        f"{value.imag:.16E}",
                    ]
                )
    return _table(("Etot/cm-1", "alpha", "m", "l", "m_l", "alpha'", "m'", "l'", "m_l'", "Re(S)", "Im(S)"), rows)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def smatrix(
    result: ScatteringResult | CoupledStatesResult,
    *,
    energy_indices: EnergySelection = None,
    state: QuantumSelection = None,
    v: QuantumSelection = None,
    j: QuantumSelection = None,
    state_prime: QuantumSelection = None,
    v_prime: QuantumSelection = None,
    j_prime: QuantumSelection = None,
    block_index: int | None = None,
) -> str:
    """
    Format selected atom-diatom S-matrix elements as a text table.

    Initial-state filters select matrix columns and final-state filters select
    matrix rows, following ``S[out, in]``. A filter may be one integer, a range,
    an integer sequence, or None to retain every available value. The complex
    S-matrix elements are written with 16 digits after the decimal point.

    Inputs:
        result: ScatteringResult | CoupledStatesResult - completed scattering
            calculation to report
        energy_indices: EnergySelection - integer indices, slice, or integer
            sequence selecting total-energy entries; None selects all entries
        state: QuantumSelection - initial electronic-state filter
        v: QuantumSelection - initial vibrational-state filter
        j: QuantumSelection - initial rotational-state filter
        state_prime: QuantumSelection - final electronic-state filter
        v_prime: QuantumSelection - final vibrational-state filter
        j_prime: QuantumSelection - final rotational-state filter
        block_index: int | None - coupled-states K-block result to report; may be
            omitted only when the result contains one block

    Returns:
        output: str - formatted S-matrix table with total energies in cm-1
    """
    if isinstance(result, ScatteringResult) and isinstance(result.basis, ChannelBasisElectricSF):
        if any(value is not None for value in (state, v, j, state_prime, v_prime, j_prime, block_index)):
            raise ValueError("Electronic, vibrational, rotational, and K-block filters do not apply to Electric-SF results")
        return _smatrix_electric(result, energy_indices)

    state_values = _selection(state, "state")
    v_values = _selection(v, "v")
    j_values = _selection(j, "j")
    state_prime_values = _selection(state_prime, "state_prime")
    v_prime_values = _selection(v_prime, "v_prime")
    j_prime_values = _selection(j_prime, "j_prime")
    basis = cast(ChannelBasis, result.basis)
    internal_states = tuple(_internal_state(channel) for channel in basis)

    if not any(_matches(value, state_values, v_values, j_values) for value in internal_states):
        raise ValueError("The initial-state selection does not match any channel")
    if not any(_matches(value, state_prime_values, v_prime_values, j_prime_values) for value in internal_states):
        raise ValueError("The final-state selection does not match any channel")

    if isinstance(result, ScatteringResult):
        if block_index is not None:
            raise ValueError("block_index is only valid for coupled-states results")
        matrices = result.Smat
        open_indices = result.open_channel_indices
        angular_momenta = tuple(result.L[indices] for indices in open_indices)
    else:
        if block_index is None:
            if len(result.blocks) != 1:
                raise ValueError("block_index is required when a coupled-states result has more than one K block")
            block_index = 0
        try:
            block_result = result.blocks[block_index]
        except IndexError as error:
            raise IndexError(f"block_index {block_index} is outside the available K blocks") from error
        matrices = block_result.Smat_asymptotic
        open_indices = block_result.open_channel_indices
        local_positions = {global_index: local_index for local_index, global_index in enumerate(block_result.block.channel_indices)}
        angular_momenta = tuple(
            np.asarray([block_result.L[local_positions[int(global_index)]] for global_index in indices]) for indices in open_indices
        )

    include_electronic_state = any(value.electronic_state is not None for value in internal_states)
    headers = ["Etot/cm-1"]
    if include_electronic_state:
        headers.append("state")
    headers.extend(("v", "j", "L"))
    if include_electronic_state:
        headers.append("state'")
    headers.extend(("v'", "j'", "L'", "Re(S)", "Im(S)"))

    rows: list[list[str]] = []
    for energy_index in _energy_indices(energy_indices, result.Etot.size):
        indices = open_indices[energy_index]
        L_values = angular_momenta[energy_index]
        matrix = matrices[energy_index]
        for incoming, incoming_global in enumerate(indices):
            initial = internal_states[int(incoming_global)]
            if not _matches(initial, state_values, v_values, j_values):
                continue
            for outgoing, outgoing_global in enumerate(indices):
                final = internal_states[int(outgoing_global)]
                if not _matches(final, state_prime_values, v_prime_values, j_prime_values):
                    continue
                value = matrix[outgoing, incoming]
                row = [f"{result.Etot[energy_index] * AU2CM:.8f}"]
                if include_electronic_state:
                    row.append(str(initial.electronic_state))
                row.extend((str(initial.v), str(initial.j), _orbital_angular_momentum(float(L_values[incoming]))))
                if include_electronic_state:
                    row.append(str(final.electronic_state))
                row.extend(
                    (
                        str(final.v),
                        str(final.j),
                        _orbital_angular_momentum(float(L_values[outgoing])),
                        f"{value.real:.16E}",
                        f"{value.imag:.16E}",
                    )
                )
                rows.append(row)
    return _table(headers, rows)


# ----------------------------------------------------------------------------------------
