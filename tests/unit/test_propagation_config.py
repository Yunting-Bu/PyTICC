import pytest

import pyticc as ticc


def test_propagation_normalizes_sequences() -> None:
    propagation = ticc.Propagation(
        boundaries=(3, 4, 6),
        half_steps=(0.1, 0.2),
        mode="capture",
        memory_mb=256,
        progress_every_sectors=25,
    )

    assert propagation.boundaries == (3.0, 4.0, 6.0)
    assert propagation.half_steps == (0.1, 0.2)
    assert propagation.Rmatch == 6.0
    assert propagation.mode == "capture"
    assert propagation.progress_every_sectors == 25


@pytest.mark.parametrize(
    ("boundaries", "half_steps", "match"),
    [
        ((3.0,), (), "sizes N\\+1 and N"),
        ((3.0, 2.0), (0.1,), "strictly increasing"),
        ((3.0, 4.0), (0.0,), "half_steps"),
    ],
)
def test_propagation_validates_grid(boundaries: tuple[float, ...], half_steps: tuple[float, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ticc.Propagation(boundaries=boundaries, half_steps=half_steps)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_propagation_validates_progress_interval(value: object) -> None:
    with pytest.raises(ValueError, match="progress_every_sectors"):
        ticc.Propagation(boundaries=(3.0, 4.0), half_steps=(0.1,), progress_every_sectors=value)  # type: ignore[arg-type]
