from pathlib import Path

from model import h2_potential, interaction

import pyticc as ticc


def main() -> None:
    """Run B(2P) + H2 from the TOML system and electronic-order input."""
    pes = ticc.SpinResolvedAtomDiatomPES(
        interaction=interaction,
        monomer_Y=h2_potential,
    )
    result = ticc.run(Path(__file__).with_name("input.toml"), pes=pes)
    print(ticc.report.smatrix(result))


if __name__ == "__main__":
    main()
