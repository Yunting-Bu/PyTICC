from pathlib import Path

import pyticc as ticc


def main() -> None:
    input_file = Path(__file__).with_name("input.toml")
    result = ticc.run(input_file)
    result.print_summary()


if __name__ == "__main__":
    main()
