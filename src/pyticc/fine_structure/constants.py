import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast

from loguru import logger

from pyticc.constants import ENERGY_TO_AU, EnergyUnit, energy_to_au
from pyticc.fine_structure.operators import FSConstants

FS_CONSTANT_COLUMNS = ("v", "constant", "value", "unit")
FS_CONSTANT_NAMES = ("A", "B", "D", "H", "gamma", "lambda_ss", "O", "P", "Q", "M", "N")


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FSConstantsTable:
    """Vibrationally resolved effective molecular constants, stored in Hartree."""

    entries: tuple[tuple[int, FSConstants], ...]

    def __post_init__(self) -> None:
        vibrational_levels = tuple(v for v, _ in self.entries)
        if any(v < 0 for v in vibrational_levels):
            raise ValueError("Fine-structure vibrational quantum numbers must be nonnegative")
        if vibrational_levels != tuple(sorted(set(vibrational_levels))):
            raise ValueError("Fine-structure constant-table entries must have unique, sorted v values")

    @property
    def vibrational_levels(self) -> tuple[int, ...]:
        """Return the available vibrational quantum numbers."""
        return tuple(v for v, _ in self.entries)

    def for_v(self, v: int) -> FSConstants:
        """Return the constants for one discrete vibrational manifold."""
        for table_v, constants in self.entries:
            if table_v == v:
                return constants
        available = ", ".join(str(value) for value in self.vibrational_levels) or "none"
        message = f"No fine-structure constants were supplied for v={v}; available v values: {available}"
        logger.error(message)
        raise ValueError(message)


# ----------------------------------------------------------------------------------------
def load_fs_constants_csv(path: str | Path) -> FSConstantsTable:
    """
    Load vibrationally resolved molecular constants from a long-format CSV.

    The required header is ``v,constant,value,unit``. Supported constants are
    A, B, D, H, gamma, lambda_ss, O, P, Q, M, and N. Units may be au, cm-1, Hz, kHz,
    MHz, or GHz. Each value is converted to Hartree; omitted constants within a
    supplied vibrational manifold are zero. Lines beginning with ``#`` are
    ignored, which allows provenance notes to be kept in the file.
    """
    csv_path = Path(path).expanduser()
    values_by_v: dict[int, dict[str, float]] = {}
    seen: set[tuple[int, str]] = set()

    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        data_rows = (line for line in stream if not line.lstrip().startswith("#"))
        reader = csv.DictReader(data_rows)
        columns = tuple(name.strip() for name in reader.fieldnames or ())
        if columns != FS_CONSTANT_COLUMNS:
            expected = ",".join(FS_CONSTANT_COLUMNS)
            message = f"Fine-structure constants CSV header must be {expected}, but got {','.join(columns)}"
            logger.error(message)
            raise ValueError(message)

        for line_number, row in enumerate(reader, start=2):
            try:
                v_text = row["v"].strip()
                v = int(v_text)
                if str(v) != v_text or v < 0:
                    raise ValueError
                name = row["constant"].strip()
                if name not in FS_CONSTANT_NAMES:
                    supported = ", ".join(FS_CONSTANT_NAMES)
                    raise ValueError(f"unknown constant {name!r}; supported constants: {supported}")
                value = float(row["value"])
                if not isfinite(value):
                    raise ValueError("value must be finite")
                unit_text = row["unit"].strip()
                if unit_text not in ENERGY_TO_AU:
                    supported_units = ", ".join(ENERGY_TO_AU)
                    raise ValueError(f"unknown unit {unit_text!r}; supported units: {supported_units}")
                key = (v, name)
                if key in seen:
                    raise ValueError(f"duplicate entry for v={v}, constant={name}")
            except (AttributeError, TypeError, ValueError) as error:
                detail = f": {error}" if str(error) else ""
                message = f"Invalid fine-structure constant in {csv_path} at line {line_number}{detail}"
                logger.error(message)
                raise ValueError(message) from error

            seen.add(key)
            unit = cast(EnergyUnit, unit_text)
            values_by_v.setdefault(v, {})[name] = energy_to_au(value, unit)

    if not values_by_v:
        message = f"Fine-structure constants CSV {csv_path} contains no data rows"
        logger.error(message)
        raise ValueError(message)

    entries: list[tuple[int, FSConstants]] = []
    for v, values in sorted(values_by_v.items()):
        entries.append(
            (
                v,
                FSConstants(
                    A=values.get("A", 0.0),
                    B=values.get("B", 0.0),
                    D=values.get("D", 0.0),
                    H=values.get("H", 0.0),
                    gamma=values.get("gamma", 0.0),
                    lambda_ss=values.get("lambda_ss", 0.0),
                    O=values.get("O", 0.0),
                    P=values.get("P", 0.0),
                    Q=values.get("Q", 0.0),
                    M=values.get("M", 0.0),
                    N=values.get("N", 0.0),
                ),
            )
        )
    return FSConstantsTable(tuple(entries))


# ----------------------------------------------------------------------------------------
