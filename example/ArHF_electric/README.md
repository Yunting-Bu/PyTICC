# Ar–HF scattering in a dc electric field

This directory contains the electric-field-specific Ar–HF examples.

- `monomer_run.py` builds the field-dressed HF monomer states and compares the
  lowest levels with the reference Fortran calculation.
- `python_run.py` builds the fixed-\(M\) space-fixed channel basis, contracts the
  Ar–HF interaction PES, performs exact coupled-channel propagation, and prints
  the channel and \(S\)-matrix tables.
- `input.toml` and `input_run.py` run the same calculation through the PyTICC
  TOML input interface.
- `compare.py` projects the zero-field Electric-SF result onto the
  field-free $J=0,p=+1$ block and prints the two $S$ matrices side by side.
- `pes/HF_ele.csv` contains the HF dipole, polarizability, and
  hyperpolarizability curves used by the monomer Hamiltonian.

The scalar Ar–HF interaction PES is shared with `../ArHF/pes`; it is not copied
here because the electric field is included only in the asymptotic HF monomer
Hamiltonian.

Run the examples from the repository root:

```bash
uv run python example/ArHF_electric/monomer_run.py
uv run python example/ArHF_electric/python_run.py
uv run python example/ArHF_electric/input_run.py
uv run python example/ArHF_electric/compare.py
```

The scattering example uses

\[
M=m+m_l
\]

and propagates the Electric-SF channels

\[
\lvert \eta;M\rangle
=\lvert\phi_{\alpha m}(\mathcal E)\rangle
 \lvert l m_l\rangle
\]

with the exact coupled-channel method. CS and NNCC are not used for the
electric-field calculation.

## TOML input

Use `type = "electric-atom-diatom"`. The external-field-specific entries are:

```toml
M = 0

[basis]
jmax = 8
lmax = 1
n_alpha = 3

[electric]
strength_au = 1.0e-3
response_csv = "pes/HF_ele.csv"

[quadrature]
n_theta_r = 16
n_theta_R = 16
n_delta = 16
delta_symmetry = true

[truncation]
E_Y_cut_cm = 2000.0
```

Here `jmax` truncates the primitive rotational expansion of each dressed HF
state, `lmax` truncates end-over-end rotation, and `n_alpha` is the number of
dressed states retained in each required \(m\) block. The required \(m\) blocks
are generated automatically from \(M\) and `lmax`.

Do not specify `Jtot`, `system_parity`, or `K_cut`. An `[approximation]` table is
also unnecessary; Electric-SF calculations use exact coupled channels, and an
explicit `cs` or `nncc` request is rejected.
