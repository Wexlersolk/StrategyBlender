from pathlib import Path

from engine import data_loader


def test_load_bars_resamples_m1_to_h1(tmp_path, monkeypatch):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    file_path = export_dir / "TEST.cash_M1_202401010000_202401010059.csv"
    file_path.write_text(
        "\n".join(
            [
                "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>",
                "2024.01.01\t00:00:00\t100\t101\t99\t100.5\t10\t1\t2",
                "2024.01.01\t00:15:00\t100.5\t103\t100\t102\t20\t2\t3",
                "2024.01.01\t00:30:00\t102\t104\t101\t103\t30\t3\t4",
                "2024.01.01\t00:45:00\t103\t105\t102\t104\t40\t4\t5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(data_loader, "EXPORT_DIR", export_dir)
    data_loader._load_minute_bars.cache_clear()

    bars = data_loader.load_bars("TEST.cash", "H1")

    assert list(bars.columns) == ["open", "high", "low", "close", "volume", "real_volume", "spread"]
    assert len(bars) == 1
    row = bars.iloc[0]
    assert row["open"] == 100
    assert row["high"] == 105
    assert row["low"] == 99
    assert row["close"] == 104
    assert row["volume"] == 100
    assert row["real_volume"] == 10
    assert row["spread"] == 3.5
    assert data_loader.available_symbols() == ["TEST.cash"]


def test_load_bars_filters_date_range(tmp_path, monkeypatch):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    file_path = export_dir / "RANGE.cash_M1_202401010000_202401020059.csv"
    file_path.write_text(
        "\n".join(
            [
                "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>",
                "2024.01.01\t00:00:00\t1\t2\t0.5\t1.5\t10",
                "2024.01.02\t00:00:00\t2\t3\t1.5\t2.5\t20",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(data_loader, "EXPORT_DIR", export_dir)
    data_loader._load_minute_bars.cache_clear()

    bars = data_loader.load_bars("RANGE.cash", "M1", date_from="2024-01-02", date_to="2024-01-02 23:59:59")

    assert len(bars) == 1
    assert float(bars.iloc[0]["open"]) == 2.0
