from pathlib import Path

import numpy as np

from pyticc import AU2CM, build_SineDVR, element_masses_au, reduced_mass
from pyticc.basis.monomer.diatom_electric import DiatomElectricBasis, build_DiatomElectricBasis

ELECTRIC_STRENGTH = 1.0e-3
REFERENCE_M0_CM = np.array([0.0, 135.557, 238.545, 348.192, 505.735])
REFERENCE_ABS_M1_CM = np.array([81.723, 212.784, 342.942, 504.401, 706.960])


def _peshf_reference(r: np.ndarray) -> np.ndarray:
    r"""
    Get the zero-field HF potential used by rovib.f::peshf.

    Formula:
        With Delta r = 0.529177249 r - 0.9168 in angstrom,

        V_0(r) = -6.123
                 [1 + 4.216 Delta r + 3.965 Delta r^2
                    + 3.835 Delta r^3]
                 exp(-4.216 Delta r) / 27.21138.

    Inputs:
        r: np.ndarray - HF bond lengths in bohr, shape (n_r,)

    Returns:
        potential: np.ndarray - zero-field HF potential in hartree,
            shape (n_r,)
    """
    displacement = 0.529177249 * r - 0.9168
    polynomial = 1.0 + 4.216 * displacement + 3.965 * displacement**2 + 3.835 * displacement**3
    return -6.123 * polynomial * np.exp(-4.216 * displacement) / 27.21138


def _print_comparison(electric_HF: DiatomElectricBasis) -> None:
    pyticc_m0_cm = electric_HF.relative_energies(0)[: REFERENCE_M0_CM.size] * AU2CM
    pyticc_abs_m1_cm = electric_HF.relative_energies(1)[: REFERENCE_ABS_M1_CM.size] * AU2CM

    print("\nHF electric-field-dressed monomer")
    print(f"Electric field strength : {electric_HF.electric_strength:.8e} a.u.")
    print(f"HF reduced mass         : {electric_HF.mass:.12f} m_e")
    print(f"Common energy zero      : {electric_HF.energy_zero * AU2CM:.8f} cm^-1")
    print(f"Solved m blocks         : {electric_HF.m_values}")
    print("Energy convention       : E(alpha,m) - E(alpha=0,m=0)")
    print()
    print(f"{'alpha':>5} {'PyTICC(m=0)':>16} {'ref(m=0)':>14} {'PyTICC(|m|=1)':>18} {'ref(|m|=1)':>16}")
    print("-" * 75)
    for alpha, (pyticc_m0, ref_m0, pyticc_abs_m1, ref_abs_m1) in enumerate(
        zip(pyticc_m0_cm, REFERENCE_M0_CM, pyticc_abs_m1_cm, REFERENCE_ABS_M1_CM, strict=True)
    ):
        print(f"{alpha:5d} {pyticc_m0:16.8f} {ref_m0:14.8f} {pyticc_abs_m1:18.8f} {ref_abs_m1:16.8f}")


def main() -> None:
    """Build the electric-field-dressed HF monomer basis for the Ar-HF example."""
    pes_dir = Path(__file__).with_name("pes")
    _, mass_H, mass_F = element_masses_au("Ar", "H", "F")
    reduced_mass_HF = reduced_mass(mass_H, mass_F)

    dvr_HF = build_SineDVR(
        a=0.75,
        b=6.55,
        n_dvr=50,
        mass=reduced_mass_HF,
        pot_func=_peshf_reference,
    )

    electric_HF = build_DiatomElectricBasis(
        dvr=dvr_HF,
        response=pes_dir / "HF_ele.csv",
        electric_strength=ELECTRIC_STRENGTH,
        n_podvr=5,
        jmax=25,
        M=0,
        lmax=1,
        n_alpha=5,
        mass=reduced_mass_HF,
    )

    _print_comparison(electric_HF)


if __name__ == "__main__":
    main()
