from pathlib import Path
from threading import Event

import numpy as np
from loguru import logger

import pyticc as ticc
from pyticc.basis.channel import ChannelBasis, build_ChannelBasis
from pyticc.basis.monomer import DiatomSpec
from pyticc.propagation.device import resolve_device
from pyticc.propagation.grid import RadialSector
from pyticc.propagation.runner import _evaluate_interaction_windows, propagate, propagate_blocks


def _basis() -> ChannelBasis:
    atom = ticc.AtomSpec()
    diatom = DiatomSpec(Eint=np.array([[0.0, 0.2]]))
    system = ticc.ScattSystem(atom, diatom, Jtot=0, system_parity=1)
    return build_ChannelBasis(system, ticc.ChannelSpec())


def _hamiltonian(Vmat) -> ticc.ScattHamiltonian:
    return ticc.ScattHamiltonian(basis=_basis(), reduced_mass=2.0, interaction=Vmat)


def test_propagate_builds_radial_matrices_on_the_requested_device() -> None:
    basis = _basis()
    evaluated_R: list[np.ndarray] = []

    def Vmat(RR: float | np.ndarray) -> np.ndarray:
        radial_points = np.atleast_1d(RR)
        evaluated_R.append(radial_points.copy())
        return np.zeros((radial_points.size, basis.n_channel, basis.n_channel))

    result = propagate(
        _hamiltonian(Vmat),
        [0.1, 0.3],
        ticc.Propagation((3.0, 3.4), (0.1,), device="cpu"),
    )

    assert result.shape == (2, basis.n_channel, basis.n_channel)
    assert {device.platform for device in result.devices()} == {"cpu"}
    np.testing.assert_allclose(evaluated_R[0], [3.0, 3.1, 3.2, 3.3, 3.4])


def test_propagate_reads_energies_from_file_and_supports_capture(tmp_path: Path) -> None:
    basis = _basis()
    energy_file = tmp_path / "energies.dat"
    np.savetxt(energy_file, [0.1, 0.3])

    result = propagate(
        _hamiltonian(lambda RR: np.zeros((np.atleast_1d(RR).size, basis.n_channel, basis.n_channel))),
        energy_file,
        ticc.Propagation((3.0, 3.2), (0.1,), mode="capture"),
    )

    assert result.shape == (2, basis.n_channel, basis.n_channel)
    assert result.dtype == np.complex128
    np.testing.assert_allclose(result[1, 0, 0], -1.0j * np.sqrt(1.2), rtol=1.0e-13, atol=1.0e-13)


def test_propagate_batches_distinct_radial_points() -> None:
    basis = _basis()
    evaluated_R: list[np.ndarray] = []

    def Vmat(RR: float | np.ndarray) -> np.ndarray:
        radial_points = np.asarray(RR)
        evaluated_R.append(radial_points)
        return np.zeros((radial_points.size, basis.n_channel, basis.n_channel))

    result = propagate(_hamiltonian(Vmat), [0.3], ticc.Propagation((3.0, 3.4), (0.1,)))

    assert result.shape == (1, basis.n_channel, basis.n_channel)
    assert len(evaluated_R) == 1
    np.testing.assert_allclose(evaluated_R[0], [3.0, 3.1, 3.2, 3.3, 3.4])


def test_propagate_streams_small_windows_without_repeating_endpoints() -> None:
    basis = _basis()
    evaluated_R: list[np.ndarray] = []

    def Vmat(RR: np.ndarray) -> np.ndarray:
        radial_points = np.asarray(RR)
        evaluated_R.append(radial_points.copy())
        return np.zeros((radial_points.size, basis.n_channel, basis.n_channel))

    streamed = propagate(
        _hamiltonian(Vmat),
        [0.3],
        ticc.Propagation((3.0, 3.4), (0.1,), memory_mb=1.0e-6),
    )
    full = propagate(
        _hamiltonian(lambda RR: np.zeros((np.atleast_1d(RR).size, basis.n_channel, basis.n_channel))),
        [0.3],
        ticc.Propagation((3.0, 3.4), (0.1,)),
    )

    assert len(evaluated_R) == 2
    np.testing.assert_allclose(evaluated_R[0], [3.0, 3.1, 3.2])
    np.testing.assert_allclose(evaluated_R[1], [3.3, 3.4])
    np.testing.assert_allclose(streamed, full, rtol=1.0e-13, atol=1.0e-13)


def test_gpu_interaction_prefetch_starts_next_window_before_current_is_consumed() -> None:
    first_window = (RadialSector(3.0, 3.2),)
    second_window = (RadialSector(3.2, 3.4),)
    windows = iter(
        (
            (first_window, np.array([3.0, 3.1, 3.2])),
            (second_window, np.array([3.2, 3.3, 3.4])),
        )
    )
    prefetch_started = Event()
    release_prefetch = Event()
    evaluated_points: list[np.ndarray] = []

    def interaction_provider(radial_points, blocks, device):
        evaluated_points.append(radial_points.copy())
        if len(evaluated_points) == 2:
            prefetch_started.set()
            assert release_prefetch.wait(timeout=2.0)
        return (np.zeros((radial_points.size, 1, 1)),)

    interaction_windows = _evaluate_interaction_windows(
        windows,
        interaction_provider,
        ((0,),),
        resolve_device("cpu").device,
        prefetch=True,
    )
    current = next(interaction_windows)

    assert current[0] == first_window
    assert prefetch_started.wait(timeout=2.0)
    release_prefetch.set()
    following = next(interaction_windows)
    assert following[0] == second_window
    np.testing.assert_allclose(evaluated_points[0], [3.0, 3.1, 3.2])
    np.testing.assert_allclose(evaluated_points[1], [3.3, 3.4])


def test_propagate_blocks_selects_one_nncc_block() -> None:
    basis = _basis()
    indices = (1,)

    hamiltonian = _hamiltonian(lambda RR: np.zeros((np.atleast_1d(RR).size, basis.n_channel, basis.n_channel)))
    result = propagate_blocks(hamiltonian, (indices,), [0.3], ticc.Propagation((3.0, 3.2), (0.1,)))[0]

    assert result.shape == (1, 1, 1)


def test_propagate_logs_completed_sector_count_radius_and_wall_time() -> None:
    basis = _basis()
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        propagate(
            _hamiltonian(lambda RR: np.zeros((np.atleast_1d(RR).size, basis.n_channel, basis.n_channel))),
            [0.3],
            ticc.Propagation((3.0, 3.4), (0.1,), print_verbose=True),
        )
    finally:
        logger.remove(sink)

    assert any("Propagation started" in message and "sectors=2" in message for message in messages)
    assert any("Propagation: 2/2 sectors" in message and "R=3.400000 bohr" in message and "wall=" in message for message in messages)
