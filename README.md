# PyTICC

Time-independent quantum scattering in Python, NumPy, and JAX. PyTICC treats
rovibrational, electronic, and fine-structure inelastic scattering and
short-range capture for atom--atom, atom--molecule, and molecule--molecule
systems.

## Features

| System | Capabilities |
| --- | --- |
| `A+BC` | Full-dimensional single-surface inelastic scattering; exact CC, CS, and NNCC |
| `A+BC_diabatic` | Coupled electronic, vibrational, and rotational motion on real symmetric diabatic surfaces; exact CC |
| `A+BC_fine_structure` | $^{2S+1}\Sigma/\Pi/\Delta$ diatoms with spin--orbit, spin--rotation, spin--spin, $\Lambda$-doubling, and signed-$\Lambda$ surfaces; exact CC, CS, and NNCC |
| `A+B_fine_structure` | Two structured atoms with experimental $j$ levels and magnetic dipole coupling; exact CC, CS, and NNCC |
| `A+BC_both_fs` | Fine structure on both the atom and the diatom; exact CC, CS, and NNCC |
| `A+BC_electric` | Stark levels and Electric-SF scattering in a static electric field |
| `A+BC_Delves` | Three-arrangement reactive scattering in Delves hyperspherical coordinates |
| `AB+CD` | Full-dimensional rovibrational scattering of two diatoms; exact CC, CS, and NNCC; identical-molecule exchange with exact CC |
| `AB+CD_fine_structure` | Two open-shell diatoms with scalar or total-spin-resolved complex Hermitian orbital surfaces and direct spin dipole coupling; exact CC, CS, and NNCC |
| `A+BCD` | Full-dimensional atom--triatom scattering in Radau coordinates with contracted triatom rovibrational states |

Fixed-arrangement and Delves calculations support ordinary inelastic/reactive
inner boundaries and incoming-wave capture boundaries:

```python
ticc.Propagation(mode="inelastic")
ticc.Propagation(mode="capture")
```

Other features:

- Sine DVR and PODVR diatom bases, plus contracted internal triatom bases.
- Python PES callbacks and compiled Fortran loaders for scalar, diabatic, signed-$\Lambda$, and total-triatom surfaces.
- Radial PES batching, multiprocessing, and reusable `PotentialGrid` caches.
- JAX 64-bit LogD propagation on CPU and NVIDIA GPU, with energy and channel blocking.
- Channel thresholds, open/closed-channel metadata, S matrices, and text reports.

## Quick start

Run a TOML input file:

```python
import pyticc as ticc

result = ticc.run("input.toml")
print(ticc.report.smatrix(result))
```

TOML front ends cover `A+BC`, `A+BC_electric`, `A+BC_diabatic`, `A+B_fine_structure`,
`A+BC_both_fs`, `AB+CD`, and `AB+CD_fine_structure`; fine-structure inputs take
the electronic PES through `pes=...`. Other capabilities use the Python API:

```python
system = ticc.build_ScattSystem(monomer_X, monomer_Y, two_J=2, system_parity=1, potential=pes, reduced_mass=reduced_mass)
grid = ticc.prepare_potential(system, boundaries=(10.0, 20.0, 50.0), half_steps=(0.05, 0.20), n_theta_X=15, n_theta_Y=15, n_phi=12)
result = ticc.solve(system, total_energies, grid, ticc.Propagation(mode="capture"))
```

The builder infers the scattering implementation from the monomer bases; no
scattering-type string is needed. See [`example/`](example/) for runnable
calculations, including [H2 + B(2P)](example/H2B_2P/),
[H2 + OH(2Pi)](example/H2OH_2Pi/), [O2 + O2](example/O2O2_spin/), and
[Ca + He(3P)](example/CaHe_3P/).

## Installation

Install the CPU backend on macOS or Linux:

```bash
uv sync
```

On Linux with an NVIDIA GPU, select one CUDA backend (CUDA 12 and CUDA 13 are
mutually exclusive):

```bash
uv sync --extra cuda12  # NVIDIA driver 525 or newer
uv sync --extra cuda13  # NVIDIA driver 580 or newer
```

Pass the same extra to `uv run`. Set `device = "gpu"` to treat a missing GPU as
an error, or `device = "auto"` to fall back to the CPU.

## Documentation

See the [Chinese API reference](docs/API_zh.md) for array shapes, units, and
limitations, and [atomic fine structure](docs/atomic_fine_structure.md) for the
`A+B_fine_structure` interface.
