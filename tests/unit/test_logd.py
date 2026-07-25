import numpy as np

from pyticc.propagation.logd import initialize_logD_capture, initialize_logD_inelastic, propagate_logD, propagate_logD_sector


def _constant_logD(Y0: complex, W: float, length: float) -> complex:
    if W > 0.0:
        p = np.sqrt(W)
        tangent = np.tanh(p * length)
        return p * (Y0 + p * tangent) / (p + Y0 * tangent)
    k = np.sqrt(-W)
    tangent = np.tan(k * length)
    return k * (Y0 - k * tangent) / (k + Y0 * tangent)


def _dense_ldmd_sector(Ymat: np.ndarray, radial_half_step: float, W_start: np.ndarray, W_mid: np.ndarray, W_end: np.ndarray) -> np.ndarray:
    reference_values = np.diag(W_mid)
    magnitude = np.sqrt(np.abs(reference_values))
    argument = magnitude * radial_half_step
    y1_values = np.where(reference_values >= 0.0, magnitude / np.tanh(argument), magnitude / np.tan(argument))
    y2_values = np.where(reference_values >= 0.0, magnitude / np.sinh(argument), magnitude / np.sin(argument))
    y1 = np.diag(y1_values)
    y2 = np.diag(y2_values)
    reference = np.diag(reference_values)
    identity = np.eye(Ymat.shape[-1])
    Q_start = radial_half_step * (W_start - reference) / 3.0
    Q_mid = 4.0 / radial_half_step * (np.linalg.solve(identity - radial_half_step**2 * (W_mid - reference) / 6.0, identity) - identity)
    Q_end = radial_half_step * (W_end - reference) / 3.0
    Y_mid = y1 + Q_mid - y2 @ np.linalg.solve(Ymat + y1 + Q_start, y2)
    Y_end = y1 + Q_end - y2 @ np.linalg.solve(Y_mid + y1 + Q_mid, y2)
    return 0.5 * (Y_end + Y_end.T)


def test_logD_initial_conditions_distinguish_inelastic_and_capture() -> None:
    Wmat = np.diag([4.0, -9.0])

    inelastic = initialize_logD_inelastic(Wmat)
    capture = initialize_logD_capture(Wmat)

    np.testing.assert_allclose(inelastic, np.diag([2.0, 3.0]))
    np.testing.assert_allclose(capture, np.diag([2.0, -3.0j]))
    assert inelastic.dtype == np.float64
    assert capture.dtype == np.complex128


def test_propagate_logD_sector_is_exact_for_constant_reference_potential() -> None:
    W = 4.0
    radial_half_step = 0.15
    Y0 = 1.3
    matrix = np.array([[W]])

    result = propagate_logD_sector(np.array([[Y0]]), radial_half_step, matrix, matrix, matrix)
    expected = _constant_logD(Y0, W, 2.0 * radial_half_step)

    np.testing.assert_allclose(result, [[expected]], rtol=1.0e-13, atol=1.0e-13)


def test_propagate_logD_sector_preserves_capture_incoming_wave() -> None:
    W = -9.0
    radial_half_step = 0.1
    Y0 = -3.0j
    matrix = np.array([[W]])

    result = propagate_logD_sector(np.array([[Y0]]), radial_half_step, matrix, matrix, matrix)

    np.testing.assert_allclose(result, [[Y0]], rtol=1.0e-13, atol=1.0e-13)
    assert result.dtype == np.complex128


def test_propagate_logD_sector_matches_dense_coupled_capture_reference() -> None:
    radial_half_step = 0.08
    Y_initial = np.array([[1.2 - 0.7j, 0.04 + 0.02j], [0.04 + 0.02j, 1.5 - 0.4j]])
    W_start = np.array([[-1.1, 0.12], [0.12, 0.9]])
    W_mid = np.array([[-1.0, 0.15], [0.15, 0.8]])
    W_end = np.array([[-0.9, 0.11], [0.11, 0.7]])

    result = propagate_logD_sector(Y_initial, radial_half_step, W_start, W_mid, W_end)
    expected = _dense_ldmd_sector(Y_initial, radial_half_step, W_start, W_mid, W_end)

    np.testing.assert_allclose(result, expected, rtol=1.0e-12, atol=1.0e-12)


def test_propagate_logD_uses_vmap_over_energies_and_scan_over_sectors() -> None:
    total_energies = np.array([0.1, 0.2])
    reduced_mass = 2.0
    radial_half_steps = np.array([0.05, 0.10])
    W_base = np.full((2, 1, 1), 3.0)
    W_initial = 3.0 - 2.0 * reduced_mass * total_energies[:, None, None]
    Y_initial = np.full((2, 1, 1), 1.25)

    result = propagate_logD(Y_initial, total_energies, reduced_mass, radial_half_steps, W_base, W_base, W_base)
    total_length = 2.0 * np.sum(radial_half_steps)
    expected = np.array([[_constant_logD(1.25, W.item(), total_length)] for W in W_initial]).reshape(2, 1, 1)

    np.testing.assert_allclose(result, expected, rtol=1.0e-12, atol=1.0e-12)


def test_propagate_logD_matches_sequential_coupled_sector_updates() -> None:
    total_energies = np.array([0.03, 0.07])
    reduced_mass = 1.7
    radial_half_steps = np.array([0.04, 0.06, 0.05])
    W_base_start = np.array(
        [
            [[1.8, 0.10], [0.10, 2.3]],
            [[1.9, 0.12], [0.12, 2.2]],
            [[2.0, 0.08], [0.08, 2.1]],
        ]
    )
    W_base_mid = W_base_start + np.array([[[0.05, 0.02], [0.02, -0.03]]])
    W_base_end = W_base_start + np.array([[[0.09, -0.01], [-0.01, -0.06]]])
    Y_initial = np.broadcast_to(np.array([[1.2, 0.03], [0.03, 1.5]]), (total_energies.size, 2, 2)).copy()

    result = propagate_logD(
        Y_initial,
        total_energies,
        reduced_mass,
        radial_half_steps,
        W_base_start,
        W_base_mid,
        W_base_end,
    )

    expected = []
    identity = np.eye(2)
    for energy, initial in zip(total_energies, Y_initial, strict=True):
        Y_current = initial
        energy_shift = 2.0 * reduced_mass * energy * identity
        for half_step, W_start, W_mid, W_end in zip(
            radial_half_steps,
            W_base_start,
            W_base_mid,
            W_base_end,
            strict=True,
        ):
            Y_current = propagate_logD_sector(
                Y_current,
                half_step,
                W_start - energy_shift,
                W_mid - energy_shift,
                W_end - energy_shift,
            )
        expected.append(np.asarray(Y_current))

    np.testing.assert_allclose(result, np.stack(expected), rtol=1.0e-12, atol=1.0e-12)
