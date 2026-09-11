from pathlib import Path

from model import h2_potential, interaction, oh_potential

import pyticc as ticc


def main() -> None:
    """Run H2 + OH(2Pi) from the TOML system and electronic-order input."""
    pes = ticc.SpinResolvedDiatomDiatomPES(
        interaction=interaction,
        monomer_X=h2_potential,
        monomer_Y=oh_potential,
    )
    result = ticc.run(Path(__file__).with_name("input.toml"), pes=pes)
    print(ticc.report.smatrix(result))


if __name__ == "__main__":
    main()
