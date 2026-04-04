from pathlib import Path

from convert.mt5_to_python import convert_mql5_to_python


FIXTURES = Path(__file__).parent / "fixtures"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_simple_review_snapshot():
    source_path = FIXTURES / "mq5" / "simple_review.mq5"
    snapshot_path = FIXTURES / "snapshots" / "simple_review.py"

    result = convert_mql5_to_python(
        source=_read(source_path),
        strategy_name="simple_review",
        symbol="EURUSD",
        timeframe="H1",
    )

    assert result.warnings == []
    assert result.review_source == _read(snapshot_path)


def test_unsupported_loop_snapshot():
    source_path = FIXTURES / "mq5" / "unsupported_loop.mq5"
    snapshot_path = FIXTURES / "snapshots" / "unsupported_loop.py"

    result = convert_mql5_to_python(
        source=_read(source_path),
        strategy_name="unsupported_loop",
        symbol="EURUSD",
        timeframe="M15",
    )

    assert result.warnings == [
        "OnTick() was not found; generated local review entrypoint is a placeholder."
    ]
    assert result.review_source == _read(snapshot_path)
