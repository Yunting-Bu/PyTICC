# He--NO potential models

## Interaction potential

`heno_PES_export.f` is the published rigid-rotor RCCSD(T)/AVTZ+332
He--NO(X 2Pi) interaction by Klos et al., J. Chem. Phys. **112**, 2195
(2000), DOI: [10.1063/1.480785](https://doi.org/10.1063/1.480785). The
unmodified source and its original README are included in this directory.

The native functions accept `R` in bohr and `cos(theta)` and return cm-1. The
PyTICC wrapper converts the output to Hartree and exposes

```text
V_sum = (V_A'' + V_A') / 2
V_dif = (V_A'' - V_A') / 2
```

The interaction was calculated at a fixed NO bond length, so it is independent
of the bond coordinate supplied by PyTICC.

## Isolated NO potential

The interaction source does not contain an isolated NO potential. To construct
`v=0` through the same sine-DVR/PODVR route used by other PyTICC examples, the
wrapper supplies the spectroscopic Morse model

```text
V_NO(r) = De [1 - exp(-a(r-re))]^2
```

with `re=1.15077 Angstrom`, `omega_e=1904.2 cm-1`, and
`omega_e x_e=14.075 cm-1`. The corresponding Morse parameters are
`De=64404.5762 cm-1` and `a=1.32124596 bohr^-1`. This monomer model provides
the vibrational basis only; it does not add bond-length dependence to the
rigid-rotor interaction.
