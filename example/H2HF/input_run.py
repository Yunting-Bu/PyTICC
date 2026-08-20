from pathlib import Path

import pyticc as ticc


def main() -> None:
    input_file = Path(__file__).with_name("input.toml")
    result = ticc.run(input_file)
    print(ticc.report.channels(result.basis))
    print(ticc.report.open_closed(result.basis, result.Etot))
    print(ticc.report.smatrix(result, energy_indices=0, v_X=0, v_Y=0, v_X_prime=0, v_Y_prime=0))


if __name__ == "__main__":
    main()
