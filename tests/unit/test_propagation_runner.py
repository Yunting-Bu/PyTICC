from pathlib import Path

import numpy as np
from loguru import logger

import pyticc as ticc
from pyticc.basis.channel import ChannelBuilder
from pyticc.basis.monomer import DiatomSpec
from pyticc.propagation import propagate_BF


def _basis() -> ticc.ChannelBasis:
    atom = ticc.AtomSpec()
    diatom = DiatomSpec(Eint=np.array([[0.0, 0.2]]), vmax=0, jmax=1)
    system = ticc.ScattSystem(atom, diatom, Jtot=0, system_parity=1)
    return ChannelBuilder(system, ticc.TruncSpec()).build()


def test_propagate_BF_builds_radial_matrices_and_caches_shared_points() -> None:
    basis = _basis()
    evaluated_R: list[float] = []

    def Vmat(RR: float) -> np.ndarray:
        evaluated_R.append(RR)
        return np.zeros((basis.n_channel, basis.n_channel))

    result = propagate_BF(
        basis=basis,
        Vmat=Vmat,
        Etot=[0.1, 0.3],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.4],
        radial_half_steps=[0.1],
        device="cpu",
    )

    assert result.shape == (2, basis.n_channel, basis.n_channel)
    assert {device.platform for device in result.devices()} == {"cpu"}
    np.testing.assert_allclose(evaluated_R, [3.0, 3.1, 3.2, 3.3, 3.4])


def test_propagate_BF_reads_energies_from_file_and_supports_capture(tmp_path: Path) -> None:
    basis = _basis()
    energy_file = tmp_path / "energies.dat"
    np.savetxt(energy_file, [0.1, 0.3])

    result = propagate_BF(
        basis=basis,
        Vmat=lambda RR: np.zeros((basis.n_channel, basis.n_channel)),
        Etot=energy_file,
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.2],
        radial_half_steps=[0.1],
        mode="capture",
    )

    assert result.shape == (2, basis.n_channel, basis.n_channel)
    assert result.dtype == np.complex128
    np.testing.assert_allclose(result[1, 0, 0], -1.0j * np.sqrt(1.2), rtol=1.0e-13, atol=1.0e-13)


def test_propagate_BF_batches_distinct_radial_points() -> None:
    basis = _basis()
    evaluated_R: list[np.ndarray] = []

    def Vmat(RR: float | np.ndarray) -> np.ndarray:
        radial_points = np.asarray(RR)
        evaluated_R.append(radial_points)
        return np.zeros((radial_points.size, basis.n_channel, basis.n_channel))

    result = propagate_BF(
        basis=basis,
        Vmat=Vmat,
        Etot=[0.3],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.4],
        radial_half_steps=[0.1],
        batch_Vmat=True,
    )

    assert result.shape == (1, basis.n_channel, basis.n_channel)
    assert len(evaluated_R) == 1
    np.testing.assert_allclose(evaluated_R[0], [3.0, 3.1, 3.2, 3.3, 3.4])


def test_propagate_BF_streams_small_windows_without_repeating_endpoints() -> None:
    basis = _basis()
    evaluated_R: list[np.ndarray] = []

    def Vmat(RR: np.ndarray) -> np.ndarray:
        radial_points = np.asarray(RR)
        evaluated_R.append(radial_points.copy())
        return np.zeros((radial_points.size, basis.n_channel, basis.n_channel))

    streamed = propagate_BF(
        basis=basis,
        Vmat=Vmat,
        Etot=[0.3],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.4],
        radial_half_steps=[0.1],
        batch_Vmat=True,
        memory_limit_mb=1.0e-6,
    )
    full = propagate_BF(
        basis=basis,
        Vmat=lambda RR: np.zeros((np.asarray(RR).size, basis.n_channel, basis.n_channel)),
        Etot=[0.3],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.4],
        radial_half_steps=[0.1],
        batch_Vmat=True,
    )

    assert len(evaluated_R) == 2
    np.testing.assert_allclose(evaluated_R[0], [3.0, 3.1, 3.2])
    np.testing.assert_allclose(evaluated_R[1], [3.3, 3.4])
    np.testing.assert_allclose(streamed, full, rtol=1.0e-13, atol=1.0e-13)


def test_propagate_BF_selects_one_nncc_block() -> None:
    basis = _basis()
    indices = (1,)

    result = propagate_BF(
        basis=basis,
        Vmat=lambda RR: np.zeros((1, 1)),
        Etot=[0.3],
        reduced_mass=2.0,
        radial_boundaries=[3.0, 3.2],
        radial_half_steps=[0.1],
        channel_indices=indices,
    )

    assert result.shape == (1, 1, 1)


def test_propagate_BF_logs_completed_sector_count_radius_and_wall_time() -> None:
    basis = _basis()
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    try:
        propagate_BF(
            basis=basis,
            Vmat=lambda RR: np.zeros((basis.n_channel, basis.n_channel)),
            Etot=[0.3],
            reduced_mass=2.0,
            radial_boundaries=[3.0, 3.4],
            radial_half_steps=[0.1],
            print_verbose=True,
        )
    finally:
        logger.remove(sink)

    assert any("Propagation started" in message and "sectors=2" in message for message in messages)
    assert any("Propagation: 2/2 sectors" in message and "R=3.400000 bohr" in message and "wall=" in message for message in messages)
