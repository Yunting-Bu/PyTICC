import numpy as np
import pytest
from scipy.special import roots_legendre

from pyticc.basis.angle import gauss_legendre_dvr


def test_gauss_legendre_dvr_matches_full_interval_rule() -> None:
    grids, weights = gauss_legendre_dvr(-2.0, 3.0, 7)
    reference_grids, reference_weights = roots_legendre(7)

    np.testing.assert_allclose(grids, 0.5 + 2.5 * reference_grids)
    np.testing.assert_allclose(weights, 2.5 * reference_weights)


def test_gauss_legendre_dvr_symmetry_retains_half_of_double_order_rule() -> None:
    grids, weights = gauss_legendre_dvr(-1.0, 1.0, 6, symmetry=True)
    full_grids, full_weights = roots_legendre(12)

    np.testing.assert_allclose(grids, full_grids[:6])
    np.testing.assert_allclose(weights, 2.0 * full_weights[:6])
    np.testing.assert_allclose(np.sum(weights * grids**4), 2.0 / 5.0)


@pytest.mark.parametrize("bounds", [(0.0, 0.0), (1.0, -1.0), (np.nan, 1.0)])
def test_gauss_legendre_dvr_rejects_invalid_bounds(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="bounds"):
        gauss_legendre_dvr(*bounds, 4)


def test_gauss_legendre_dvr_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        gauss_legendre_dvr(-1.0, 1.0, 0)
