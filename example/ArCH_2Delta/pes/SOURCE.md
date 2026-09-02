# Ar--CH(A 2Delta) potential model

## Interaction potential

`pot_archa.f` implements the CH(A 2Delta)-Ar surfaces of Kerenskaya, Kaledin
and Heaven, J. Chem. Phys. **115**, 2123 (2001) (doi: 10.1063/1.1382647),
from the Morse expansion coefficients of Table IV of that paper. The wrapper
uses the empirically modified "surface 1" (D scale factors V00*1.62,
V10*0.2, V20*1.568) with the recommended uniform R_e shift `dre = -0.25 A`.

The native routine returns, in cm-1 with R in Angstrom,

```text
V_a  = (V_A' + V_A'')/2
V_d  = (V_A' - V_A'')/2
```

which the wrapper converts to the PyTICC convention

```text
V_sum = V_a,     V_dif = (V_A'' - V_A')/2 = -V_d
```

in Hartree. The surface is a rigid-rotor 2D surface (CH bond fixed at
1.1021 A), so the bond coordinate supplied by PyTICC is ignored.

## Isolated CH(A 2Delta) potential

The wrapper supplies the spectroscopic Morse model

```text
V_CH(r) = De [1 - exp(-a(r-re))]^2
```

with `re=1.1021 Angstrom` (Kerenskaya et al. 2001), `omega_e=2931 cm-1`,
`omega_e x_e=83.6 cm-1`, giving `De=25691 cm-1` and `a=1.1362 bohr^-1`.

## Fine-structure constants

`constant_2Delta_CH.csv` holds v=0 and v=1 effective constants of
CH(A 2Delta) from experiment (Zachwieja, J. Mol. Spectrosc. **170**, 285
(1995); Masseron et al., Astron. Astrophys. **571**, A47 (2014)):

```text
v=0:  A=-1.10 cm-1, B=14.58 cm-1, gamma=0.0421 cm-1
v=1:  A=-1.07 cm-1, B=13.91 cm-1, gamma=0.0406 cm-1
```

(13CH values of Masseron et al. converted to 12CH for B.) The Lambda-doubling
constants M, N, O, P, Q are undetectably small for v=0,1 (p ~ 1e-6 cm-1) and
are left zero. Note: the A=-10 cm-1 and B=15.09 cm-1 values used in the
Kerenskaya et al. 2001 dynamics were rough computational estimates; the
experimental values above should be preferred.
