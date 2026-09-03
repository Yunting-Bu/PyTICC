from collections.abc import Sequence
from typing import TypeAlias, cast

import numpy as np

from pyticc.basis.channel import Channel, ChannelBasis, ChannelBasisElectricSF, ChannelElectricSF
from pyticc.basis.kblock import KBlock
from pyticc.basis.monomer import DiabaticDiatomBasis, DiatomBasis
from pyticc.constants import AU2CM
from pyticc.energy import EnergyInput, get_Etot
from pyticc.fine_structure.channel import FSChannelBasis, FSMonomerBasis
from pyticc.fine_structure.diatom_diatom import FSDiatomDiatomBasis
from pyticc.match.delves import DelvesAsymptoticBasis
from pyticc.result import CoupledStatesResult, KBlockResult, ReactiveScatteringResult, ScatteringResult
from pyticc.system import MolInnerState

QuantumSelection: TypeAlias = int | range | Sequence[int] | None
EnergySelection: TypeAlias = int | slice | Sequence[int] | None
ReportBasis: TypeAlias = ChannelBasis | ChannelBasisElectricSF | FSChannelBasis | FSDiatomDiatomBasis | DelvesAsymptoticBasis


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
def fine_structure_levels(basis: FSMonomerBasis) -> str:
    """
    Format retained open-shell diatomic fine-structure levels.

    Levels are reported relative to the monomer scattering zero and sorted in
    the stored fixed-(v,j,parity) block order.

    Inputs:
        basis: FSMonomerBasis - vibrational and fine-structure eigenbasis

    Returns:
        output: str - level table with energies in cm-1
    """
    rows = [
        [
            str(block.v),
            str(block.two_j / 2),
            str(tau),
            str(block.parity),
            f"{(energy - basis.energy_zero) * AU2CM:.8f}",
        ]
        for block in basis.blocks
        for tau, energy in enumerate(block.energies)
    ]
    return _table(("v", "j", "tau", "epsilon", "E_int/cm-1"), rows)


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
def channels(basis: ReportBasis) -> str:
    """
    Format the channel quantum numbers and internal energies as a text table.

    Only quantum numbers belonging to active monomers are included. Electronic
    states are shown when present, while Jtot and parity are omitted.

    Inputs:
        basis: ReportBasis - channel basis to report

    Returns:
        output: str - formatted channel table with energies in cm-1
    """
    if isinstance(basis, DelvesAsymptoticBasis):
        rows = [
            [
                str(index),
                str(arrangement),
                str(v),
                str(j),
                str(K),
                f"{energy * AU2CM:.6f}",
            ]
            for index, ((arrangement, v, j, K), energy) in enumerate(zip(basis.qns, basis.energies, strict=True), start=1)
        ]
        return _table(("n", "a", "v", "j", "K", "E_int/cm-1"), rows)

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

    if isinstance(basis, FSChannelBasis):
        rows = []
        for index, channel in enumerate(basis, start=1):
            block = basis.monomer.blocks[channel.block]
            rows.append(
                [
                    str(index),
                    str(block.v),
                    str(block.two_j / 2),
                    str(channel.tau),
                    str(block.parity),
                    str(channel.two_K / 2),
                    f"{channel.E_int * AU2CM:.6f}",
                ]
            )
        return _table(("n", "v", "j", "tau", "epsilon", "K", "E_int/cm-1"), rows)

    if isinstance(basis, FSDiatomDiatomBasis):
        rows = []
        for index, channel in enumerate(basis, start=1):
            block_X = basis.monomer_X.blocks[channel.block_X]
            block_Y = basis.monomer_Y.blocks[channel.block_Y]
            rows.append(
                [
                    str(index),
                    str(block_X.v),
                    str(block_X.two_j / 2),
                    str(channel.tau_X),
                    str(block_X.parity),
                    str(block_Y.v),
                    str(block_Y.two_j / 2),
                    str(channel.tau_Y),
                    str(block_Y.parity),
                    str(channel.two_j12 / 2),
                    str(channel.two_K / 2),
                    f"{channel.E_int * AU2CM:.6f}",
                ]
            )
        label = f"molecule_exchange={basis.molecule_exchange:+d}; X/Y label canonical state pairs\n" if basis.molecule_exchange else ""
        return label + _table(("n", "v_X", "j_X", "tau_X", "epsilon_X", "v_Y", "j_Y", "tau_Y", "epsilon_Y", "j_12", "K", "E_int/cm-1"), rows)

    exchange_label = f"molecule_exchange={basis.molecule_exchange:+d}; X/Y label canonical state pairs\n" if basis.molecule_exchange else ""
    if basis.n_channel == 0:
        return exchange_label + _table(("n", "K", "E_int/cm-1"), ())

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
    return exchange_label + _table(headers, rows)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def open_closed(basis: ReportBasis, energies: EnergyInput) -> str:
    """
    Format open and closed channel counts at each total energy.

    Inputs:
        basis: ReportBasis - channel basis used to classify thresholds
        energies: EnergyInput - total energies in atomic units, or a path to a
            one-column energy file

    Returns:
        output: str - formatted channel-count table with energies in cm-1
    """
    values = get_Etot(energies)
    if isinstance(basis, DelvesAsymptoticBasis):
        open_mask = basis.energies[np.newaxis, :] < values[:, np.newaxis]
        n_open = np.sum(open_mask, axis=1)
        n_closed = basis.n_channel - n_open
    else:
        counts = basis.open_closed(values)
        n_open = counts.n_open
        n_closed = counts.n_closed
    rows = [
        [str(index), f"{energy * AU2CM:.8f}", str(n_open_value), str(n_closed_value)]
        for index, (energy, n_open_value, n_closed_value) in enumerate(
            zip(values, n_open, n_closed, strict=True),
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
def _matches_state(state: MolInnerState, electronic: set[int] | None, v: set[int] | None, j: set[int] | None) -> bool:
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
def _reject_filters(context: str, filters: Sequence[tuple[str, object]]) -> None:
    selected = [name for name, value in filters if value is not None]
    if selected:
        raise ValueError(f"Filters {', '.join(selected)} do not apply to {context} results")


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _matches_electric(
    channel: ChannelElectricSF,
    alpha: set[int] | None,
    m: set[int] | None,
    ell: set[int] | None,
    m_l: set[int] | None,
) -> bool:
    return (
        (alpha is None or channel.alpha in alpha)
        and (m is None or channel.m in m)
        and (ell is None or channel.l in ell)
        and (m_l is None or channel.m_l in m_l)
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _smatrix_fine_structure(result: ScatteringResult, energy_indices: EnergySelection) -> str:
    """Format an open-shell atom--diatom S matrix in its asymptotic SF basis."""
    basis = cast(FSChannelBasis, result.basis)
    rows: list[list[str]] = []
    for energy_index in _energy_indices(energy_indices, result.Etot.size):
        indices = result.open_channel_indices[energy_index]
        matrix = result.Smat[energy_index]
        for incoming, incoming_global in enumerate(indices):
            initial_channel = basis[int(incoming_global)]
            initial_block = basis.monomer.blocks[initial_channel.block]
            for outgoing, outgoing_global in enumerate(indices):
                final_channel = basis[int(outgoing_global)]
                final_block = basis.monomer.blocks[final_channel.block]
                value = matrix[outgoing, incoming]
                rows.append(
                    [
                        f"{result.Etot[energy_index] * AU2CM:.8f}",
                        str(initial_block.v),
                        str(initial_block.two_j / 2),
                        str(initial_channel.tau),
                        str(initial_block.parity),
                        f"{result.L[int(incoming_global)]:.8f}",
                        str(final_block.v),
                        str(final_block.two_j / 2),
                        str(final_channel.tau),
                        str(final_block.parity),
                        f"{result.L[int(outgoing_global)]:.8f}",
                        f"{value.real:.16E}",
                        f"{value.imag:.16E}",
                    ]
                )
    headers = ("Etot/cm-1", "v", "j", "tau", "epsilon", "L", "v'", "j'", "tau'", "epsilon'", "L'", "Re(S)", "Im(S)")
    return _table(headers, rows)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _smatrix_fine_structure_diatom_diatom(result: ScatteringResult, energy_indices: EnergySelection) -> str:
    """Format a two-fine-structure-diatom S matrix in its asymptotic SF basis."""
    basis = cast(FSDiatomDiatomBasis, result.basis)
    label = f"molecule_exchange={basis.molecule_exchange:+d}; X/Y label canonical state pairs\n" if basis.molecule_exchange else ""
    if basis.molecule_exchange and basis.n_channel == 0:
        return label + "No allowed channels in this molecule-exchange block."
    rows: list[list[str]] = []
    for energy_index in _energy_indices(energy_indices, result.Etot.size):
        indices = result.open_channel_indices[energy_index]
        matrix = result.Smat[energy_index]
        for incoming, incoming_global in enumerate(indices):
            initial = basis[int(incoming_global)]
            initial_X = basis.monomer_X.blocks[initial.block_X]
            initial_Y = basis.monomer_Y.blocks[initial.block_Y]
            for outgoing, outgoing_global in enumerate(indices):
                final = basis[int(outgoing_global)]
                final_X = basis.monomer_X.blocks[final.block_X]
                final_Y = basis.monomer_Y.blocks[final.block_Y]
                value = matrix[outgoing, incoming]
                rows.append(
                    [
                        f"{result.Etot[energy_index] * AU2CM:.8f}",
                        str(initial_X.v),
                        str(initial_X.two_j / 2),
                        str(initial.tau_X),
                        str(initial_X.parity),
                        str(initial_Y.v),
                        str(initial_Y.two_j / 2),
                        str(initial.tau_Y),
                        str(initial_Y.parity),
                        str(initial.two_j12 / 2),
                        f"{result.L[int(incoming_global)]:.8f}",
                        str(final_X.v),
                        str(final_X.two_j / 2),
                        str(final.tau_X),
                        str(final_X.parity),
                        str(final_Y.v),
                        str(final_Y.two_j / 2),
                        str(final.tau_Y),
                        str(final_Y.parity),
                        str(final.two_j12 / 2),
                        f"{result.L[int(outgoing_global)]:.8f}",
                        f"{value.real:.16E}",
                        f"{value.imag:.16E}",
                    ]
                )
    headers = (
        "Etot/cm-1",
        "v_X",
        "j_X",
        "tau_X",
        "epsilon_X",
        "v_Y",
        "j_Y",
        "tau_Y",
        "epsilon_Y",
        "j_12",
        "L",
        "v_X'",
        "j_X'",
        "tau_X'",
        "epsilon_X'",
        "v_Y'",
        "j_Y'",
        "tau_Y'",
        "epsilon_Y'",
        "j_12'",
        "L'",
        "Re(S)",
        "Im(S)",
    )
    return label + _table(headers, rows)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _smatrix_electric(
    result: ScatteringResult,
    energy_indices: EnergySelection,
    *,
    alpha: QuantumSelection,
    m: QuantumSelection,
    ell: QuantumSelection,
    m_l: QuantumSelection,
    alpha_prime: QuantumSelection,
    m_prime: QuantumSelection,
    ell_prime: QuantumSelection,
    m_l_prime: QuantumSelection,
) -> str:
    basis = cast(ChannelBasisElectricSF, result.basis)
    initial_filters = (
        _selection(alpha, "alpha"),
        _selection(m, "m"),
        _selection(ell, "l"),
        _selection(m_l, "m_l"),
    )
    final_filters = (
        _selection(alpha_prime, "alpha_prime"),
        _selection(m_prime, "m_prime"),
        _selection(ell_prime, "l_prime"),
        _selection(m_l_prime, "m_l_prime"),
    )
    if not any(_matches_electric(channel, *initial_filters) for channel in basis):
        raise ValueError("The initial Electric-SF selection does not match any channel")
    if not any(_matches_electric(channel, *final_filters) for channel in basis):
        raise ValueError("The final Electric-SF selection does not match any channel")

    rows: list[list[str]] = []
    for energy_index in _energy_indices(energy_indices, result.Etot.size):
        indices = result.open_channel_indices[energy_index]
        matrix = result.Smat[energy_index]
        for incoming, incoming_global in enumerate(indices):
            initial = basis[int(incoming_global)]
            if not _matches_electric(initial, *initial_filters):
                continue
            for outgoing, outgoing_global in enumerate(indices):
                final = basis[int(outgoing_global)]
                if not _matches_electric(final, *final_filters):
                    continue
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
def _matches_delves(
    qns: tuple[int, int, int, int],
    arrangement: set[int] | None,
    v: set[int] | None,
    j: set[int] | None,
    K: set[int] | None,
) -> bool:
    a_value, v_value, j_value, K_value = qns
    return (
        (arrangement is None or a_value in arrangement)
        and (v is None or v_value in v)
        and (j is None or j_value in j)
        and (K is None or K_value in K)
    )


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _smatrix_delves(
    result: ReactiveScatteringResult,
    energy_indices: EnergySelection,
    *,
    arrangement: QuantumSelection,
    v: QuantumSelection,
    j: QuantumSelection,
    K: QuantumSelection,
    arrangement_prime: QuantumSelection,
    v_prime: QuantumSelection,
    j_prime: QuantumSelection,
    K_prime: QuantumSelection,
) -> str:
    initial_filters = (
        _selection(arrangement, "arrangement"),
        _selection(v, "v"),
        _selection(j, "j"),
        _selection(K, "K"),
    )
    final_filters = (
        _selection(arrangement_prime, "arrangement_prime"),
        _selection(v_prime, "v_prime"),
        _selection(j_prime, "j_prime"),
        _selection(K_prime, "K_prime"),
    )
    if not any(_matches_delves(qns, *initial_filters) for qns in result.basis.qns):
        raise ValueError("The initial Delves selection does not match any channel")
    if not any(_matches_delves(qns, *final_filters) for qns in result.basis.qns):
        raise ValueError("The final Delves selection does not match any channel")

    rows: list[list[str]] = []
    for energy_index in _energy_indices(energy_indices, result.Etot.size):
        indices = result.open_channel_indices[energy_index]
        matrix = result.Smat[energy_index]
        for incoming, incoming_global in enumerate(indices):
            initial = result.basis.qns[int(incoming_global)]
            if not _matches_delves(initial, *initial_filters):
                continue
            for outgoing, outgoing_global in enumerate(indices):
                final = result.basis.qns[int(outgoing_global)]
                if not _matches_delves(final, *final_filters):
                    continue
                value = matrix[outgoing, incoming]
                rows.append(
                    [
                        f"{result.Etot[energy_index] * AU2CM:.8f}",
                        *(str(number) for number in initial),
                        *(str(number) for number in final),
                        f"{value.real:.16E}",
                        f"{value.imag:.16E}",
                    ]
                )
    return _table(("Etot/cm-1", "a", "v", "j", "K", "a'", "v'", "j'", "K'", "Re(S)", "Im(S)"), rows)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _field_free_data(
    result: ScatteringResult | CoupledStatesResult,
    block_index: int | None,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if isinstance(result, ScatteringResult):
        if block_index is not None:
            raise ValueError("block_index is only valid for coupled-states results")
        open_indices = result.open_channel_indices
        angular_momenta = tuple(result.L[indices] for indices in open_indices)
        return result.Smat, open_indices, angular_momenta

    if block_index is None:
        if len(result.blocks) != 1:
            raise ValueError("block_index is required when a coupled-states result has more than one K block")
        block_index = 0
    try:
        block_result = result.blocks[block_index]
    except IndexError as error:
        raise IndexError(f"block_index {block_index} is outside the available K blocks") from error
    local_positions = {global_index: local_index for local_index, global_index in enumerate(block_result.block.channel_indices)}
    angular_momenta = tuple(
        np.asarray([block_result.L[local_positions[int(global_index)]] for global_index in indices]) for indices in block_result.open_channel_indices
    )
    return block_result.Smat_asymptotic, block_result.open_channel_indices, angular_momenta


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def _smatrix_atom_diatom(
    result: ScatteringResult | CoupledStatesResult,
    energy_indices: EnergySelection,
    *,
    state: QuantumSelection,
    v: QuantumSelection,
    j: QuantumSelection,
    state_prime: QuantumSelection,
    v_prime: QuantumSelection,
    j_prime: QuantumSelection,
    block_index: int | None,
) -> str:
    basis = cast(ChannelBasis, result.basis)
    internal_states: list[MolInnerState] = []
    for channel in basis:
        states = tuple(state_value for state_value in (channel.mis_X, channel.mis_Y) if not _is_atom_state(state_value))
        if len(states) != 1 or states[0].v is None:
            raise ValueError("The atom-diatom S-matrix report requires one diatomic internal state per channel")
        internal_states.append(states[0])

    initial_filters = (_selection(state, "state"), _selection(v, "v"), _selection(j, "j"))
    final_filters = (
        _selection(state_prime, "state_prime"),
        _selection(v_prime, "v_prime"),
        _selection(j_prime, "j_prime"),
    )
    if not any(_matches_state(value, *initial_filters) for value in internal_states):
        raise ValueError("The initial-state selection does not match any channel")
    if not any(_matches_state(value, *final_filters) for value in internal_states):
        raise ValueError("The final-state selection does not match any channel")

    matrices, open_indices, angular_momenta = _field_free_data(result, block_index)
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
            if not _matches_state(initial, *initial_filters):
                continue
            for outgoing, outgoing_global in enumerate(indices):
                final = internal_states[int(outgoing_global)]
                if not _matches_state(final, *final_filters):
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


# ----------------------------------------------------------------------------------------
def _smatrix_diatom_diatom(
    result: ScatteringResult | CoupledStatesResult,
    energy_indices: EnergySelection,
    *,
    state_X: QuantumSelection,
    v_X: QuantumSelection,
    j_X: QuantumSelection,
    state_Y: QuantumSelection,
    v_Y: QuantumSelection,
    j_Y: QuantumSelection,
    j_couple: QuantumSelection,
    state_X_prime: QuantumSelection,
    v_X_prime: QuantumSelection,
    j_X_prime: QuantumSelection,
    state_Y_prime: QuantumSelection,
    v_Y_prime: QuantumSelection,
    j_Y_prime: QuantumSelection,
    j_couple_prime: QuantumSelection,
    block_index: int | None,
) -> str:
    basis = cast(ChannelBasis, result.basis)
    if any(channel.mis_X.v is None or channel.mis_Y.v is None for channel in basis):
        raise ValueError("The diatom-diatom S-matrix report requires two diatomic internal states per channel")
    if basis.molecule_exchange and basis.n_channel == 0:
        return "No allowed channels in this molecule-exchange block."

    initial_X = (_selection(state_X, "state_X"), _selection(v_X, "v_X"), _selection(j_X, "j_X"))
    initial_Y = (_selection(state_Y, "state_Y"), _selection(v_Y, "v_Y"), _selection(j_Y, "j_Y"))
    initial_j_couple = _selection(j_couple, "j_couple")
    final_X = (
        _selection(state_X_prime, "state_X_prime"),
        _selection(v_X_prime, "v_X_prime"),
        _selection(j_X_prime, "j_X_prime"),
    )
    final_Y = (
        _selection(state_Y_prime, "state_Y_prime"),
        _selection(v_Y_prime, "v_Y_prime"),
        _selection(j_Y_prime, "j_Y_prime"),
    )
    final_j_couple = _selection(j_couple_prime, "j_couple_prime")

    def matches(channel: Channel, X_filters: tuple[set[int] | None, ...], Y_filters: tuple[set[int] | None, ...], coupled: set[int] | None) -> bool:
        return (
            _matches_state(channel.mis_X, *X_filters)
            and _matches_state(channel.mis_Y, *Y_filters)
            and (coupled is None or channel.j_couple in coupled)
        )

    if not any(matches(channel, initial_X, initial_Y, initial_j_couple) for channel in basis):
        raise ValueError("The initial diatom-diatom selection does not match any channel")
    if not any(matches(channel, final_X, final_Y, final_j_couple) for channel in basis):
        raise ValueError("The final diatom-diatom selection does not match any channel")

    matrices, open_indices, angular_momenta = _field_free_data(result, block_index)
    include_state_X = any(channel.mis_X.electronic_state is not None for channel in basis)
    include_state_Y = any(channel.mis_Y.electronic_state is not None for channel in basis)
    initial_headers = (["state_X"] if include_state_X else []) + ["v_X", "j_X"] + (["state_Y"] if include_state_Y else []) + ["v_Y", "j_Y"]
    final_headers = (["state_X'"] if include_state_X else []) + ["v_X'", "j_X'"] + (["state_Y'"] if include_state_Y else []) + ["v_Y'", "j_Y'"]
    headers = ["Etot/cm-1", *initial_headers, "j_couple", "L", *final_headers, "j_couple'", "L'", "Re(S)", "Im(S)"]

    def state_values(state_value: MolInnerState, include_electronic: bool) -> list[str]:
        values = [str(state_value.electronic_state)] if include_electronic else []
        values.extend((str(state_value.v), str(state_value.j)))
        return values

    rows: list[list[str]] = []
    for energy_index in _energy_indices(energy_indices, result.Etot.size):
        indices = open_indices[energy_index]
        L_values = angular_momenta[energy_index]
        matrix = matrices[energy_index]
        for incoming, incoming_global in enumerate(indices):
            initial = basis[int(incoming_global)]
            if not matches(initial, initial_X, initial_Y, initial_j_couple):
                continue
            for outgoing, outgoing_global in enumerate(indices):
                final = basis[int(outgoing_global)]
                if not matches(final, final_X, final_Y, final_j_couple):
                    continue
                value = matrix[outgoing, incoming]
                rows.append(
                    [
                        f"{result.Etot[energy_index] * AU2CM:.8f}",
                        *state_values(initial.mis_X, include_state_X),
                        *state_values(initial.mis_Y, include_state_Y),
                        str(initial.j_couple),
                        _orbital_angular_momentum(float(L_values[incoming])),
                        *state_values(final.mis_X, include_state_X),
                        *state_values(final.mis_Y, include_state_Y),
                        str(final.j_couple),
                        _orbital_angular_momentum(float(L_values[outgoing])),
                        f"{value.real:.16E}",
                        f"{value.imag:.16E}",
                    ]
                )
    return _table(headers, rows)


# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
def smatrix(
    result: ScatteringResult | CoupledStatesResult | ReactiveScatteringResult,
    *,
    energy_indices: EnergySelection = None,
    state: QuantumSelection = None,
    v: QuantumSelection = None,
    j: QuantumSelection = None,
    state_prime: QuantumSelection = None,
    v_prime: QuantumSelection = None,
    j_prime: QuantumSelection = None,
    state_X: QuantumSelection = None,
    v_X: QuantumSelection = None,
    j_X: QuantumSelection = None,
    state_Y: QuantumSelection = None,
    v_Y: QuantumSelection = None,
    j_Y: QuantumSelection = None,
    j_couple: QuantumSelection = None,
    state_X_prime: QuantumSelection = None,
    v_X_prime: QuantumSelection = None,
    j_X_prime: QuantumSelection = None,
    state_Y_prime: QuantumSelection = None,
    v_Y_prime: QuantumSelection = None,
    j_Y_prime: QuantumSelection = None,
    j_couple_prime: QuantumSelection = None,
    alpha: QuantumSelection = None,
    m: QuantumSelection = None,
    l: QuantumSelection = None,  # noqa: E741 - l is the conventional end-over-end angular momentum.
    m_l: QuantumSelection = None,
    alpha_prime: QuantumSelection = None,
    m_prime: QuantumSelection = None,
    l_prime: QuantumSelection = None,
    m_l_prime: QuantumSelection = None,
    arrangement: QuantumSelection = None,
    K: QuantumSelection = None,
    arrangement_prime: QuantumSelection = None,
    K_prime: QuantumSelection = None,
    block_index: int | None = None,
) -> str:
    """
    Format selected S-matrix elements as a text table.

    Initial-state filters select matrix columns and final-state filters select
    matrix rows, following ``S[out, in]``. A filter may be one integer, a range,
    an integer sequence, or None to retain every available value. Atom-diatom,
    diatom-diatom, Electric-SF, and Delves results use their corresponding filter
    families. The complex elements are written with 16 digits after the decimal
    point.

    Inputs:
        result: ScatteringResult | CoupledStatesResult |
            ReactiveScatteringResult - completed scattering calculation
        energy_indices: EnergySelection - integer indices, slice, or integer
            sequence selecting total-energy entries; None selects all entries
        state, v, j: QuantumSelection - atom-diatom initial-state filters; v and
            j also label initial Delves channels
        state_prime, v_prime, j_prime: QuantumSelection - corresponding final-state
            filters
        state_X, v_X, j_X, state_Y, v_Y, j_Y, j_couple: QuantumSelection -
            diatom-diatom initial-state filters
        state_X_prime, v_X_prime, j_X_prime, state_Y_prime, v_Y_prime,
            j_Y_prime, j_couple_prime: QuantumSelection - diatom-diatom
            final-state filters
        alpha, m, l, m_l: QuantumSelection - Electric-SF initial-channel filters
        alpha_prime, m_prime, l_prime, m_l_prime: QuantumSelection - Electric-SF
            final-channel filters
        arrangement, K: QuantumSelection - Delves initial-channel filters
        arrangement_prime, K_prime: QuantumSelection - Delves final-channel filters
        block_index: int | None - coupled-states K-block result to report; may be
            omitted only when the result contains one block

    Returns:
        output: str - formatted S-matrix table with total energies in cm-1
    """
    diatom_filters = (
        ("state_X", state_X),
        ("v_X", v_X),
        ("j_X", j_X),
        ("state_Y", state_Y),
        ("v_Y", v_Y),
        ("j_Y", j_Y),
        ("j_couple", j_couple),
        ("state_X_prime", state_X_prime),
        ("v_X_prime", v_X_prime),
        ("j_X_prime", j_X_prime),
        ("state_Y_prime", state_Y_prime),
        ("v_Y_prime", v_Y_prime),
        ("j_Y_prime", j_Y_prime),
        ("j_couple_prime", j_couple_prime),
    )
    electric_filters = (
        ("alpha", alpha),
        ("m", m),
        ("l", l),
        ("m_l", m_l),
        ("alpha_prime", alpha_prime),
        ("m_prime", m_prime),
        ("l_prime", l_prime),
        ("m_l_prime", m_l_prime),
    )
    delves_filters = (("arrangement", arrangement), ("K", K), ("arrangement_prime", arrangement_prime), ("K_prime", K_prime))

    if isinstance(result, ReactiveScatteringResult):
        _reject_filters(
            "Delves",
            (("state", state), ("state_prime", state_prime), *diatom_filters, *electric_filters, ("block_index", block_index)),
        )
        return _smatrix_delves(
            result,
            energy_indices,
            arrangement=arrangement,
            v=v,
            j=j,
            K=K,
            arrangement_prime=arrangement_prime,
            v_prime=v_prime,
            j_prime=j_prime,
            K_prime=K_prime,
        )

    if isinstance(result, ScatteringResult) and isinstance(result.basis, FSChannelBasis):
        _reject_filters(
            "fine-structure",
            (
                ("state", state),
                ("v", v),
                ("j", j),
                ("state_prime", state_prime),
                ("v_prime", v_prime),
                ("j_prime", j_prime),
                *diatom_filters,
                *electric_filters,
                *delves_filters,
                ("block_index", block_index),
            ),
        )
        return _smatrix_fine_structure(result, energy_indices)

    if isinstance(result, ScatteringResult) and isinstance(result.basis, FSDiatomDiatomBasis):
        _reject_filters(
            "two-diatom fine-structure",
            (
                ("state", state),
                ("v", v),
                ("j", j),
                ("state_prime", state_prime),
                ("v_prime", v_prime),
                ("j_prime", j_prime),
                *diatom_filters,
                *electric_filters,
                *delves_filters,
                ("block_index", block_index),
            ),
        )
        return _smatrix_fine_structure_diatom_diatom(result, energy_indices)

    if isinstance(result, ScatteringResult) and isinstance(result.basis, ChannelBasisElectricSF):
        _reject_filters(
            "Electric-SF",
            (
                ("state", state),
                ("v", v),
                ("j", j),
                ("state_prime", state_prime),
                ("v_prime", v_prime),
                ("j_prime", j_prime),
                *diatom_filters,
                *delves_filters,
                ("block_index", block_index),
            ),
        )
        return _smatrix_electric(
            result,
            energy_indices,
            alpha=alpha,
            m=m,
            ell=l,
            m_l=m_l,
            alpha_prime=alpha_prime,
            m_prime=m_prime,
            ell_prime=l_prime,
            m_l_prime=m_l_prime,
        )

    _reject_filters("field-free", (*electric_filters, *delves_filters))
    basis = cast(ChannelBasis, result.basis)
    active_X = any(not _is_atom_state(channel.mis_X) for channel in basis)
    active_Y = any(not _is_atom_state(channel.mis_Y) for channel in basis)
    if active_X and active_Y or basis.molecule_exchange:
        _reject_filters(
            "diatom-diatom",
            (("state", state), ("v", v), ("j", j), ("state_prime", state_prime), ("v_prime", v_prime), ("j_prime", j_prime)),
        )
        output = _smatrix_diatom_diatom(
            result,
            energy_indices,
            state_X=state_X,
            v_X=v_X,
            j_X=j_X,
            state_Y=state_Y,
            v_Y=v_Y,
            j_Y=j_Y,
            j_couple=j_couple,
            state_X_prime=state_X_prime,
            v_X_prime=v_X_prime,
            j_X_prime=j_X_prime,
            state_Y_prime=state_Y_prime,
            v_Y_prime=v_Y_prime,
            j_Y_prime=j_Y_prime,
            j_couple_prime=j_couple_prime,
            block_index=block_index,
        )
        if basis.molecule_exchange:
            return f"molecule_exchange={basis.molecule_exchange:+d}; X/Y label canonical state pairs\n" + output
        return output

    _reject_filters("atom-diatom", diatom_filters)
    return _smatrix_atom_diatom(
        result,
        energy_indices,
        state=state,
        v=v,
        j=j,
        state_prime=state_prime,
        v_prime=v_prime,
        j_prime=j_prime,
        block_index=block_index,
    )


# ----------------------------------------------------------------------------------------
