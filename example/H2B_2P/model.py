import numpy as np


def h2_potential(r: np.ndarray) -> np.ndarray:
    """Return a small Morse model for the isolated H2 vibration."""
    return 0.17 * (1.0 - np.exp(-1.0 * (r - 1.4))) ** 2


def interaction(R: float, coordinates: np.ndarray) -> np.ndarray:
    """Return an illustrative B(2P)--H2 signed-orbital interaction matrix."""
    r, theta = coordinates
    radial = 2.0e-3 * np.exp(-0.9 * (R - 5.0))
    anisotropy = radial * (1.0 + 0.15 * (r - 1.4) + 0.25 * np.cos(theta))
    values = np.zeros((r.size, 1, 3, 3))
    values[:, 0, 0, 0] = 1.1 * anisotropy
    values[:, 0, 1, 1] = anisotropy
    values[:, 0, 2, 2] = 0.9 * anisotropy
    coupling = 0.08 * radial * np.sin(theta)
    values[:, 0, 0, 1] = values[:, 0, 1, 0] = coupling
    values[:, 0, 1, 2] = values[:, 0, 2, 1] = -coupling
    return values
