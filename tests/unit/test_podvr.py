import numpy as np

from pyticc.basis.dvr import build_SineDVR
from pyticc.basis.podvr import build_VibPODVR


def test_build_vib_podvr_returns_requested_contracted_basis() -> None:
    def potential(x: np.ndarray) -> np.ndarray:
        return 0.01 * (x - 2.0) ** 2

    dvr = build_SineDVR(1.0, 3.0, 16, 1000.0, potential)
    vib = build_VibPODVR(dvr, n_podvr=4, vmax=2)

    assert vib.grids.shape == (4,)
    assert vib.energies.shape == (3,)
    assert vib.wavefunctions.shape == (4, 3)
    assert np.allclose(vib.wavefunctions.T @ vib.wavefunctions, np.eye(3))
