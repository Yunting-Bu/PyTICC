from pathlib import Path

import pyticc as ticc


def main() -> None:
    """Run the Electric-SF Ar-HF example from its TOML input."""
    result = ticc.run(Path(__file__).with_name("input.toml"))
    print(ticc.report.channels(result.basis))
    print(ticc.report.open_closed(result.basis, result.Etot))
    print(ticc.report.smatrix(result))


if __name__ == "__main__":
    main()
