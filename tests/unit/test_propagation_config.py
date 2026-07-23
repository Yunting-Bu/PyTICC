from types import SimpleNamespace

import pytest

import pyticc as ticc
import pyticc.propagation.device as device_module


def test_propagation_normalizes_sequences() -> None:
    propagation = ticc.Propagation(
        boundaries=(3, 4, 6),
        half_steps=(0.1, 0.2),
        mode="capture",
        memory_mb=256,
        device=" CPU:0 ",
        print_verbose=True,
    )

    assert propagation.boundaries == (3.0, 4.0, 6.0)
    assert propagation.half_steps == (0.1, 0.2)
    assert propagation.Rmatch == 6.0
    assert propagation.mode == "capture"
    assert propagation.device == "cpu:0"
    assert propagation.print_verbose is True


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


@pytest.mark.parametrize("value", [-1, 1.5, None, "true"])
def test_propagation_validates_print_verbose(value: object) -> None:
    with pytest.raises(ValueError, match="print_verbose"):
        ticc.Propagation(boundaries=(3.0, 4.0), half_steps=(0.1,), print_verbose=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["cuda", "gpu:-1", "cpu:any", 0, None])
def test_propagation_validates_device(value: object) -> None:
    with pytest.raises(ValueError, match="device"):
        ticc.Propagation(boundaries=(3.0, 4.0), half_steps=(0.1,), device=value)  # type: ignore[arg-type]


def test_resolve_device_auto_prefers_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = SimpleNamespace(platform="gpu", id=2)
    cpu = SimpleNamespace(platform="cpu", id=0)

    monkeypatch.setattr(device_module, "_platform_devices", lambda platform: (gpu,) if platform == "gpu" else (cpu,))

    selected = device_module.resolve_device("auto")

    assert selected.device is gpu
    assert selected.label == "gpu:2"


def test_resolve_device_rejects_unavailable_explicit_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "_platform_devices", lambda platform: ())

    with pytest.raises(RuntimeError, match="available gpu devices are: none"):
        device_module.resolve_device("gpu:1")
