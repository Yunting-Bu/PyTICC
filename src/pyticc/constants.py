from typing import Final, Literal, TypeAlias

# Energy
AU2EV = 27.211386245988
AU2CM = 219474.6313705
AU2K = 315775.02480407
AU2KCAL = 627.5094740631
AU2HZ = 6.5796839204999e15
AU2KHZ = AU2HZ / 1.0e3
AU2MHZ = AU2HZ / 1.0e6
AU2GHZ = AU2HZ / 1.0e9

EV2AU = 1.0 / AU2EV
CM2AU = 1.0 / AU2CM
K2AU = 1.0 / AU2K
KCAL2AU = 1.0 / AU2KCAL
HZ2AU = 1.0 / AU2HZ
KHZ2AU = 1.0 / AU2KHZ
MHZ2AU = 1.0 / AU2MHZ
GHZ2AU = 1.0 / AU2GHZ

EnergyUnit: TypeAlias = Literal["au", "cm-1", "Hz", "kHz", "MHz", "GHz"]
ENERGY_TO_AU: Final[dict[EnergyUnit, float]] = {
    "au": 1.0,
    "cm-1": CM2AU,
    "Hz": HZ2AU,
    "kHz": KHZ2AU,
    "MHz": MHZ2AU,
    "GHz": GHZ2AU,
}


# ----------------------------------------------------------------------------------------
def energy_to_au(value: float, unit: EnergyUnit) -> float:
    """
    Convert one energy or frequency value to Hartree.

    Inputs:
        value: float - value expressed in the selected unit
        unit: EnergyUnit - au, cm-1, Hz, kHz, MHz, or GHz

    Returns:
        converted: float - value in Hartree
    """
    return float(value) * ENERGY_TO_AU[unit]


# ----------------------------------------------------------------------------------------
# Length
AU2ANG = 0.529177210903
ANG2AU = 1.0 / AU2ANG

# Mass
AMU2AU = 1822.888486209
AU2AMU = 1.0 / AMU2AU
