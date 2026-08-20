from pathlib import Path

import pyticc as ticc


def main() -> None:
    result = ticc.run(Path(__file__).with_name("input.toml"))
    print(ticc.report.open_closed(result.basis, result.Etot))
    if isinstance(result, ticc.CoupledStatesResult):
        print(ticc.report.k_blocks(result.blocks))


if __name__ == "__main__":
    main()
