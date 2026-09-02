# Ar--NO(X 2Pi) 3D potential model

## Interaction potential

`arnopes.f` is the public module of Teplukhin & Kendrick, J. Chem. Phys.
**152**, 114309 (2020) (doi: 10.1063/1.5145011), "Three-dimensional potential
energy surfaces of ArNO (X~ 2Pi)". The surfaces are CCSD(T)/CBS for the
two-dimensional (R, theta) part with an analytic 3D extension:

```text
V(R,theta,r) = V2D(R,theta) + V_NO(r) + V_3body(R,theta,r)
```

- `surf_tk_jac(jac, v)` returns `v(1) = Va = (A' + A'')/2` and
  `v(2) = Vd = (A' - A'')/2` in cm-1 for Jacobi input
  `jac = (rNO, R, theta[deg])` in bohr. theta=0 is collinear Ar-NO,
  theta=180 is collinear Ar-ON.
- The wrapper subtracts the NO monomer Morse term `V_NO(r)` (parameters from
  `parm-3d.in`: De=53434 cm-1, re=2.174644592 bohr, beta=1.451769543 bohr-1)
  from `Va` and converts to the PyTICC convention in Hartree:

```text
V_sum = Va - V_NO,        V_dif = (A'' - A')/2 = -Vd
```

- `parm-3d.in` is read from the working directory (`workdir = "."` in
  `pes.toml`), where PyTICC's Fortran executor runs the PES routine.
- This is a genuinely 3D surface: the NO bond coordinate is explicitly
  included (verified: varying rNO at fixed R changes Va by thousands of
  cm-1).

## Isolated NO potential

`pyticc_monomer_y_grid` supplies the same Morse model used by the PES itself:

```text
V_NO(r) = De [1 - exp(-beta (r - re))]^2
```

with the `parm-3d.in` parameters listed above (in Hartree).

## Fine-structure constants

`constant_2Pi_NO.csv` holds v=0 and v=1 effective constants of NO(X 2Pi)
from the high-precision isotopic-invariant fit of Mueller et al., J. Mol.
Spectrosc. **310**, 92 (2015) (doi: 10.1016/j.jms.2014.12.002), in MHz:

```text
v=0:  A=-3691437, B=50856.3, gamma=-189.66, p=350.62, q=2.8447 MHz
v=1:  A=-3684101, B=50329.5, gamma=-182.19, p=350.02, q=2.778 MHz
```

Provenance is recorded in the CSV file itself (comment lines). Note: the
A=123.26 cm-1 value embedded in `arnopes.f` is the Teplukhin & Kendrick
module constant; the Mueller fit above is the more accurate experimental
value (A = -123.13 cm-1 for v=0).
