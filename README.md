# PyTICC

PyTICC is a time-independent quantum scattering package built with Python,
NumPy, and JAX. It supports rovibrational, electronic, and fine-structure
inelastic scattering and short-range capture for atom--molecule and
molecule--molecule systems.

## Features

| System | Capabilities |
| --- | --- |
| `A+BC` | Full-dimensional single-surface inelastic scattering; exact CC, CS, and NNCC |
| `A+BC_diabatic` | Coupled electronic, vibrational, and rotational motion on real symmetric diabatic surfaces; exact CC |
| `A+BC_fine_structure` | General $^{2S+1}\Sigma/\Pi/\Delta$ diatoms with spin--orbit, spin--rotation, spin--spin, $\Lambda$-doubling, and signed-$\Lambda$ surfaces; exact CC |
| `A+BC_electric` | Stark levels and Electric-SF scattering in a static electric field |
| `A+BC_Delves` | Three-arrangement reactive scattering in Delves hyperspherical coordinates |
| `AB+CD` | Full-dimensional rovibrational scattering of two diatoms; exact CC, CS, and NNCC; identical-molecule exchange blocks with exact CC |
| `AB+CD_fine_structure` | Field-free scattering of two general open-shell diatoms with scalar or total-spin-resolved complex Hermitian orbital surfaces and direct electron spin dipole $V_{\rm dd}$; exact CC |
| `A+BCD` | Full-dimensional atom--triatom scattering in Radau coordinates with contracted triatom rovibrational states |

Both fixed-arrangement and Delves calculations support ordinary
inelastic/reactive inner boundaries and incoming-wave capture boundaries:

```python
ticc.Propagation(mode="inelastic")
ticc.Propagation(mode="capture")
```

Additional capabilities include:

- Sine DVR and PODVR diatom bases, plus contracted internal triatom bases.
- Python PES callbacks and compiled Fortran loaders for scalar, diabatic,
  signed-$\Lambda$, and total-triatom surfaces.
- Radial PES batching, multiprocessing, and reusable `PotentialGrid` caches.
- JAX 64-bit LogD propagation on CPU and NVIDIA GPU, with energy and channel
  blocking for memory control.
- Channel thresholds, open/closed-channel metadata, S matrices, text reports,
  and conversions among cm$^{-1}$, frequency, length, and atomic units.
- TOML front ends for `A+BC`, `A+BC_electric`, `A+BC_diabatic`, and `AB+CD`;
  other capabilities use the compositional Python API.

`AB+CD_fine_structure` currently uses field-free fixed-$(J,P)$ blocks and does
not yet include external electric or magnetic fields. Each total-spin-resolved
orbital matrix must be finite and Hermitian; real symmetric matrices remain a
supported special case. See the [Chinese API reference](docs/API_zh.md) for
array shapes, units, and current limitations.

For identical diatoms, with or without fine structure, the Python API accepts `molecule_exchange=+1`
or `-1` in `build_ScattSystem`. Reuse the **same monomer basis object** for X
and Y, retain the same monomer states, and use equal X/Y polar quadratures.
The default `0` preserves labeled-molecule calculations. This is distinct from
`ChannelSpec.exchange_parity_X/Y`, which filter individual monomer rotational
states. Exchange adaptation currently supports exact CC with both inelastic
and capture boundaries, but not CS/NNCC or TOML inputs. The fine-structure
path includes scalar PESs, total-spin-resolved complex Hermitian orbital
PESs, and direct spin dipole coupling. Scalar PES symmetry is checked on the
grid; FS interactions are also checked in the full retained labeled basis
before projection. This finite-basis check does not prove continuum PES symmetry.
FS potential contraction currently retains the labeled-basis cost, while
propagation uses the smaller exchange block. No PES is averaged or modified. Results are
single exchange blocks; nuclear-spin weights and identical-particle cross-section
counting factors are not applied automatically. See the runnable
[spin-free example](example/identical_diatoms/python_run.py) and
[fine-structure example with spin-resolved PES and Vdd](example/identical_diatoms/fine_structure.py).

## Quick start

Run a TOML input file:

```python
import pyticc as ticc

result = ticc.run("input.toml")
print(ticc.report.smatrix(result))
```

The compositional Python API constructs the PES, monomer bases, scattering
system, and potential grid before solving:

```python
system = ticc.build_ScattSystem(
    monomer_X,
    monomer_Y,
    scattering_type="AB+CD_fine_structure",
    two_J=2,
    system_parity=1,
    potential=pes,
    reduced_mass=collision_mass,
    magnetic_dipole_coefficient=C_dd,
)
grid = ticc.prepare_potential(
    system,
    boundaries=(10.0, 20.0, 50.0),
    half_steps=(0.05, 0.20),
    n_theta_X=15,
    n_theta_Y=15,
    n_phi=12,
)
result = ticc.solve(system, total_energies, grid, ticc.Propagation(mode="capture"))
```

Runnable examples are available in [`example/`](example/), including ordinary
inelastic, electric-field, fine-structure, diabatic, capture, atom--triatom,
and Delves reactive calculations.

## Installation

Install the default CPU backend on macOS or Linux:

```bash
uv sync
```

On Linux with an NVIDIA GPU, select one CUDA backend:

```bash
# NVIDIA driver 525 or newer
uv sync --extra cuda12

# NVIDIA driver 580 or newer
uv sync --extra cuda13
```

CUDA 12 and CUDA 13 are mutually exclusive. Use the same extra when running a
calculation so that the selected backend remains part of the environment:

```bash
uv run --extra cuda12 python path/to/input_run.py
```

Set `device = "gpu"` in the input file when a missing GPU should be treated as
an error. Set `device = "auto"` to allow PyTICC to fall back to the CPU.
