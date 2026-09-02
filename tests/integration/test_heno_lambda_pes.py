from pathlib import Path

import numpy as np
import pytest

import pyticc.pes.fortran.compiler as compiler_module
from pyticc.constants import CM2AU
from pyticc.pes import get_lambda_grid_atom_diatom, load_fortran_lambda_pes


@pytest.mark.skipif(not all(compiler_module._build_tools()), reason="Fortran f2py/Meson toolchain is unavailable")
def test_published_arno_surfaces_use_lambda_interface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pes_dir = Path(__file__).parents[2] / "example" / "ArNO_3D_2Pi" / "pes"
    monkeypatch.setenv("PYTICC_CACHE_DIR", str(tmp_path / "cache"))
    pes = load_fortran_lambda_pes(pes_dir / "pes.toml")
    R = 6.5
    theta = np.array([0.0, np.pi / 3.0, np.pi / 2.0, np.pi])
    values = get_lambda_grid_atom_diatom(pes, R, np.array([2.175]), theta)[0]

    assert values.shape == (theta.size, 2)
    assert np.all(np.isfinite(values))
    assert np.any(values[:, 1] != 0.0)
    assert np.max(np.abs(values)) < 1.0e4 * CM2AU
    pes.close()
