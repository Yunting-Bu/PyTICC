"""Compare HO2 propagation on CPU and GPU up to a selected sector."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pyticc as ticc

_DEFAULT_TARGET = "165/815"
_DEVICE_LINE = re.compile(r'(?m)^device\s*=\s*"[^"]+"\s*$')


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Timing measured when one propagation run reaches the target sector.

    Members:
        device: str - requested JAX propagation device
        elapsed_seconds: float - total subprocess wall time before the target line
        propagation_seconds: float - propagation wall time reported by PyTICC
    """

    device: str
    elapsed_seconds: float
    propagation_seconds: float


def _input_with_device(source: Path, device: str) -> Path:
    """Create a temporary input beside the original with one explicit device."""
    text = source.read_text()
    replacement = f'device = "{device}"'
    if _DEVICE_LINE.search(text) is None:
        raise ValueError(f"No propagation device entry found in {source}")
    configured = _DEVICE_LINE.sub(replacement, text, count=1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", prefix="benchmark_", dir=source.parent, delete=False) as file:
        file.write(configured)
        return Path(file.name)


def _terminate(process: subprocess.Popen[str]) -> None:
    """Terminate a benchmark process group and wait for all PES workers."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_benchmark(source: Path, device: str, target: str, gpu_id: int) -> BenchmarkResult:
    """Run one device until PyTICC reports the requested sector count."""
    configured_input = _input_with_device(source, device)
    target_pattern = re.compile(rf"Propagation:\s+{re.escape(target)} sectors,.*wall=([0-9.]+) s")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else str(gpu_id)
    if device == "cpu":
        environment["JAX_PLATFORMS"] = "cpu"
    else:
        environment.pop("JAX_PLATFORMS", None)
    started = perf_counter()
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(configured_input)],
        cwd=source.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    propagation_seconds: float | None = None
    try:
        if process.stdout is None:
            raise RuntimeError("Benchmark subprocess has no output pipe")
        for line in process.stdout:
            print(f"[{device}] {line}", end="", flush=True)
            match = target_pattern.search(line)
            if match is not None:
                propagation_seconds = float(match.group(1))
                break
        if propagation_seconds is None:
            return_code = process.wait()
            raise RuntimeError(f"{device} run exited with status {return_code} before reaching {target} sectors")
    finally:
        _terminate(process)
        configured_input.unlink(missing_ok=True)
    return BenchmarkResult(device, perf_counter() - started, propagation_seconds)


def _worker(source: Path) -> None:
    """Run one temporary PyTICC input for the benchmark controller."""
    ticc.run(source)


def main() -> None:
    """Run matching CPU and GPU propagation benchmarks and print the speedup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("input.toml"))
    parser.add_argument("--target", default=_DEFAULT_TARGET, help="reported sector count, for example 165/815")
    parser.add_argument("--gpu-id", type=int, default=0, help="physical GPU selected through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker is not None:
        _worker(arguments.worker)
        return

    source = arguments.input.expanduser().resolve()
    cpu = _run_benchmark(source, "cpu", arguments.target, arguments.gpu_id)
    gpu = _run_benchmark(source, "gpu", arguments.target, arguments.gpu_id)
    print("\nBenchmark summary")
    print(f"CPU: propagation={cpu.propagation_seconds:.3f} s, total={cpu.elapsed_seconds:.3f} s")
    print(f"GPU: propagation={gpu.propagation_seconds:.3f} s, total={gpu.elapsed_seconds:.3f} s")
    print(f"GPU speedup: {cpu.propagation_seconds / gpu.propagation_seconds:.3f}x")


if __name__ == "__main__":
    main()
