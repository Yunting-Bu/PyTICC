import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pyticc as ticc
import pyticc.pes.fortran.compiler as compiler_module

PES_DIR = Path(__file__).parents[2] / "example" / "ClH2O" / "pes"
EXAMPLE = PES_DIR.parent / "python_run.py"


@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_clh2o_pes_uses_corrected_radau_embedding() -> None:
    pes = ticc.load_fortran_pes(PES_DIR / "pes.toml")
    theta = np.deg2rad(105.0)
    coordinates = np.array(
        [
            [1.75, 1.95, 1.75],
            [1.95, 1.75, 1.95],
            [theta, theta, theta],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    values = pes.interaction(6.5, coordinates)
    asymptotic = pes.interaction(60.0, coordinates[:, :1])

    assert np.all(np.isfinite(values))
    assert values[0] == pytest.approx(values[1], abs=1.0e-11)
    assert values[0] == pytest.approx(values[2], abs=1.0e-13)
    assert asymptotic[0] == pytest.approx(0.0, abs=1.0e-8)

    equilibrium = np.array(
        [
            [1.8, 1.8],
            [1.8, 1.8],
            [theta, theta],
            [0.0, 0.5 * np.pi],
            [0.0, 0.0],
        ]
    )
    corrected_reference_cm = np.array([-798.13467028, -706.45385118])

    np.testing.assert_allclose(pes.interaction(6.5, equilibrium), corrected_reference_cm * ticc.CM2AU, atol=1.0e-6 * ticc.CM2AU)


@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_clh2o_example_runs_end_to_end_scattering(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTICC_CACHE_DIR"] = str(tmp_path / "pes-cache")
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")

    completed = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=EXAMPLE.parents[2],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "channels: 2" in completed.stdout
    assert "open channels at 300 cm^-1: 2" in completed.stdout
    match = re.search(r"\|\|S\^dagger S-I\|\|:\s+([0-9.eE+-]+)", completed.stdout)
    assert match is not None
    assert float(match.group(1)) < 1.0e-12
