from types import SimpleNamespace

import pytest

import pyticc as ticc
import pyticc.propagation.device as device_module


def test_propagation_normalizes_runtime_settings() -> None:
    propagation = ticc.Propagation(
        mode="capture",
        memory_mb=256,
        device=" CPU:0 ",
        print_verbose=True,
    )

    assert propagation.mode == "capture"
    assert propagation.memory_mb == 256
    assert propagation.device == "cpu:0"
    assert propagation.print_verbose is True


@pytest.mark.parametrize("value", [-1, 1.5, None, "true"])
def test_propagation_validates_print_verbose(value: object) -> None:
    with pytest.raises(ValueError, match="print_verbose"):
        ticc.Propagation(print_verbose=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["cuda", "gpu:-1", "cpu:any", 0, None])
def test_propagation_validates_device(value: object) -> None:
    with pytest.raises(ValueError, match="device"):
        ticc.Propagation(device=value)  # type: ignore[arg-type]


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
