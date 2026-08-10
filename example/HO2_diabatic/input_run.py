from pathlib import Path

import pyticc as ticc


def main() -> None:
    input_file = Path(__file__).with_name("input.toml")
    result = ticc.run(input_file)
    print(ticc.report.open_closed(result.basis, result.Etot))
    print(ticc.report.smatrix(result))


if __name__ == "__main__":
    main()
