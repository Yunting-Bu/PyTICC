import numpy as np


def h2_potential(r: np.ndarray) -> np.ndarray:
    """Return a small Morse model for the isolated H2 vibration."""
    return 0.17 * (1.0 - np.exp(-1.0 * (r - 1.4))) ** 2


def oh_potential(r: np.ndarray) -> np.ndarray:
    """Return a small Morse model for the isolated OH vibration."""
    return 0.16 * (1.0 - np.exp(-1.1 * (r - 1.83))) ** 2


def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
    """Return an illustrative H2--OH(2Pi) signed-Lambda interaction matrix."""
    r_X, r_Y, theta_X, theta_Y, phi = coordinates
    radial = 1.5e-3 * np.exp(-0.8 * (R - 6.0))
    angular = 1.0 + 0.15 * np.cos(theta_X) * np.cos(theta_Y) + 0.05 * np.cos(phi)
    vibration = 1.0 + 0.08 * (r_X - 1.4) + 0.08 * (r_Y - 1.83)
    base = radial * angular * vibration
    values = np.zeros((r_X.size, 1, 2, 2))
    values[:, 0, 0, 0] = 1.05 * base
    values[:, 0, 1, 1] = 0.95 * base
    coupling = 0.04 * radial * np.sin(theta_X) * np.sin(theta_Y) * np.cos(phi)
    values[:, 0, 0, 1] = values[:, 0, 1, 0] = coupling
    return values
