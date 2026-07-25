# PyTICC

Time-independent coupled-channel quantum scattering code in Python and JAX.

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
