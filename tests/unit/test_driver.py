from pathlib import Path

import numpy as np

import pyticc as ticc


def test_run_reads_compact_atom_diatom_input(tmp_path: Path) -> None:
    np.savetxt(tmp_path / "energies.dat", [100.0, 200.0])
    input_file = tmp_path / "input.toml"
    input_file.write_text(
        """
type = "atom-diatom"
atom = "Ar"
diatom = ["H", "F"]
J = 1
parity = -1
energies_cm = "energies.dat"

[approximation]
method = "cs"

[basis]
r = [1.5, 4.5]
n_dvr = 20
n_podvr = 1
vmin = 0
vmax = 0
jmax = 1
jpar = 0

[quadrature]
n_theta = 4

[truncation]
E_Y_cut_cm = 1000.0
K_cut = "none"

[propagation]
radial_boundaries = [3.0, 3.2]
radial_half_steps = [0.1]
mode = "inelastic"
""",
        encoding="utf-8",
    )
    pes = ticc.PESWrapper(
        interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]),
        monomer_Y=lambda r: np.zeros_like(r),
    )

    result = ticc.run(input_file, pes=pes)

    assert result.basis.n_channel == 3
    assert isinstance(result, ticc.CoupledStatesResult)
    assert result.approx is ticc.Approx.CS
    assert any(channel.K == 1 for channel in result.basis)
    assert result.Etot.shape == (2,)
    np.testing.assert_allclose(result.Etot * ticc.AU2CM, [100.0, 200.0])
    assert len(result.blocks) == 2


def test_run_reads_diatom_diatom_input(tmp_path: Path) -> None:
    input_file = tmp_path / "input.toml"
    input_file.write_text(
        """
type = "diatom-diatom"
diatom_X = ["H", "H"]
diatom_Y = ["H", "F"]
J = 0
parity = 1
energies_cm = [100.0]

[approximation]
method = "nncc"
K_delta = 1

[basis_X]
r = [0.4, 3.5]
n_dvr = 20
n_podvr = 1
vmin = 0
vmax = 0
jmax = 0
jpar = 1

[basis_Y]
r = [0.7, 4.7]
n_dvr = 20
n_podvr = 1
vmin = 0
vmax = 0
jmax = 0
jpar = 0

[quadrature]
n_theta_X = 3
n_theta_Y = 3
n_phi = 4

[truncation]
E_X_cut_cm = 1000.0
E_Y_cut_cm = 1000.0
K_cut = "none"

[propagation]
radial_boundaries = [3.0, 3.2]
radial_half_steps = [0.1]
mode = "inelastic"
""",
        encoding="utf-8",
    )
    pes = ticc.PESWrapper(
        interaction=lambda RR, coordinates: np.zeros(coordinates.shape[1]),
        monomer_X=lambda r: np.zeros_like(r),
        monomer_Y=lambda r: np.zeros_like(r),
    )

    result = ticc.run(input_file, pes=pes)

    assert result.basis.n_channel == 1
    assert isinstance(result, ticc.CoupledStatesResult)
    assert result.approx is ticc.Approx.NNCC
    assert result.Etot.shape == (1,)
    assert len(result.blocks) == 1
