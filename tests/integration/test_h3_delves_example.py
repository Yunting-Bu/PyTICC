import os
import subprocess
import sys
from pathlib import Path

import pytest

import pyticc.pes.fortran.compiler as compiler_module

EXAMPLE_DIR = Path(__file__).parents[2] / "example" / "H3_Delves"


@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_h3_delves_example_runs_the_public_reactive_workflow(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTICC_CACHE_DIR"] = str(tmp_path / "pes-cache")
    environment["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")

    completed = subprocess.run(
        [sys.executable, str(EXAMPLE_DIR / "python_run.py")],
        cwd=EXAMPLE_DIR.parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "n_primitive=20" in completed.stdout
    assert "propagation sectors=120" in completed.stdout
    assert "a=1, v=0, j=0, K=0" in completed.stdout
    assert "a=2, v=0, j=0, K=0" in completed.stdout
    assert "|S(2 <- 1)|^2 = 9.847" in completed.stdout
