from pathlib import Path

import pytest

from pyticc.constants import CM2AU, MHZ2AU
from pyticc.fine_structure import load_fs_constants_csv


def test_load_fs_constants_csv_supports_mixed_units_and_sparse_constants(tmp_path: Path) -> None:
    path = tmp_path / "NO_constants.csv"
    path.write_text(
        "v,constant,value,unit\n"
        "1,Q,-38.9,MHz\n"
        "0,A,123.146,cm-1\n"
        "0,B,508083.0,MHz\n"
        "1,A,122.831,cm-1\n",
        encoding="utf-8",
    )

    table = load_fs_constants_csv(path)

    assert table.vibrational_levels == (0, 1)
    assert table.for_v(0).A == pytest.approx(123.146 * CM2AU)
    assert table.for_v(0).B == pytest.approx(508083.0 * MHZ2AU)
    assert table.for_v(0).Q == 0.0
    assert table.for_v(1).Q == pytest.approx(-38.9 * MHZ2AU)


@pytest.mark.parametrize(
    "body",
    [
        "0,A,1.0,bad-unit\n",
        "0,bad-name,1.0,MHz\n",
        "0,A,1.0,MHz\n0,A,2.0,MHz\n",
        "-1,A,1.0,MHz\n",
        "0,A,nan,MHz\n",
    ],
)
def test_load_fs_constants_csv_rejects_invalid_rows(tmp_path: Path, body: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("v,constant,value,unit\n" + body, encoding="utf-8")

    with pytest.raises(ValueError):
        load_fs_constants_csv(path)


def test_load_fs_constants_csv_reports_missing_v(tmp_path: Path) -> None:
    path = tmp_path / "constants.csv"
    path.write_text("v,constant,value,unit\n0,A,1.0,MHz\n", encoding="utf-8")

    table = load_fs_constants_csv(path)

    with pytest.raises(ValueError, match="v=1"):
        table.for_v(1)
