import numpy as np
import pytest

from pyticc.basis.dvr import build_RovibDVR, build_SineDVR


def test_build_rovib_dvr_keeps_the_primitive_grid_and_adds_rotation() -> None:
    def potential(r: np.ndarray) -> np.ndarray:
        return 0.03 * (r - 2.0) ** 2

    dvr = build_SineDVR(1.0, 4.0, 24, 900.0, potential)
    rovib = build_RovibDVR(dvr, vmax=1, jmax=2, mass=900.0)

    assert rovib.grids is dvr.grids
    assert rovib.E_vj.shape == (2, 3)
    assert rovib.WF_vj.shape == (24, 2, 3)
    np.testing.assert_allclose(rovib.E_vj[:, 0], dvr.eigen_val[:2], atol=1.0e-13)
    np.testing.assert_allclose(rovib.WF_vj[:, :, 0], dvr.eigen_vec[:, :2], atol=1.0e-12)
    for j in range(3):
        np.testing.assert_allclose(rovib.WF_vj[:, :, j].T @ rovib.WF_vj[:, :, j], np.eye(2), atol=1.0e-13)


def test_build_rovib_dvr_validates_quantum_numbers_and_mass() -> None:
    dvr = build_SineDVR(1.0, 3.0, 8, 500.0, lambda r: np.zeros_like(r))

    with pytest.raises(ValueError, match="non-negative"):
        build_RovibDVR(dvr, vmax=-1, jmax=0, mass=500.0)
    with pytest.raises(ValueError, match="positive and finite"):
        build_RovibDVR(dvr, vmax=0, jmax=0, mass=0.0)
