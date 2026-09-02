import jax
import numpy as np

from pyticc.propagation.grid import build_radial_sectors
from pyticc.scattering.potential import PotentialGrid
from pyticc.system import ScatteringType


# ----------------------------------------------------------------------------------------
def test_potential_grid_copies_complete_values_once_and_slices_on_device() -> None:
    device = jax.devices("cpu")[0]
    sectors = build_radial_sectors((3.0, 3.4), (0.1,))
    radial_points = np.array([3.0, 3.1, 3.2, 3.3, 3.4])
    values = np.arange(10.0).reshape(5, 2)
    potential_grid = PotentialGrid(
        boundaries=(3.0, 3.4),
        half_steps=(0.1,),
        sectors=sectors,
        radial_points=radial_points,
        scattering_type=ScatteringType.ATOM_DIATOM,
        coordinates=(),
        weights=(),
        values=values,
    )

    first = potential_grid.take_device(np.array([3.0, 3.2]), device)
    second = potential_grid.take_device(np.array([3.3, 3.4]), device)

    assert isinstance(first, jax.Array)
    assert isinstance(second, jax.Array)
    assert first.devices() == {device}
    assert len(potential_grid._device_cache) == 1
    np.testing.assert_allclose(first, values[[0, 2]])
    np.testing.assert_allclose(second, values[[3, 4]])


# ----------------------------------------------------------------------------------------
