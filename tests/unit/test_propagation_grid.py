import numpy as np

from pyticc.propagation.grid import build_radial_sectors


def test_build_radial_sectors_supports_any_number_of_intervals() -> None:
    sectors = build_radial_sectors(radial_boundaries=[3.0, 3.5, 4.1], radial_half_steps=[0.1, 0.2])

    np.testing.assert_allclose([sector.radial_start for sector in sectors], [3.0, 3.2, 3.4, 3.5, 3.9])
    np.testing.assert_allclose([sector.radial_end for sector in sectors], [3.2, 3.4, 3.5, 3.9, 4.1])
    np.testing.assert_allclose([sector.radial_mid for sector in sectors], [3.1, 3.3, 3.45, 3.7, 4.0])
    np.testing.assert_allclose([sector.radial_half_step for sector in sectors], [0.1, 0.1, 0.05, 0.2, 0.1])


def test_build_radial_sectors_has_contiguous_exact_endpoints() -> None:
    sectors = build_radial_sectors(radial_boundaries=[2.0, 2.7, 5.0, 8.0, 9.1], radial_half_steps=[0.2, 0.3, 0.5, 0.4])

    assert sectors[0].radial_start == 2.0
    assert sectors[-1].radial_end == 9.1
    np.testing.assert_allclose([left.radial_end for left in sectors[:-1]], [right.radial_start for right in sectors[1:]])


def test_build_radial_sectors_does_not_add_a_roundoff_sized_final_sector() -> None:
    sectors = build_radial_sectors([0.0, 1.0], [0.05])

    assert len(sectors) == 10
    assert sectors[-1].radial_end == 1.0
