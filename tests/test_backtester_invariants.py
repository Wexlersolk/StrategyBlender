import pandas as pd

from engine.backtester import Backtester
from engine.base_strategy import BaseStrategy


class NoTradeStrategy(BaseStrategy):
    params = {}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, ctx):
        return


def test_no_trade_strategy_keeps_balance_constant():
    index = pd.date_range("2024-01-01", periods=6, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [1, 1, 1, 1, 1, 1],
        },
        index=index,
    )

    result = Backtester(initial_capital=50_000).run(NoTradeStrategy(), df)

    assert result.n_trades == 0
    assert result.net_profit == 0.0
    assert result.win_rate == 0.0
    assert result.max_drawdown == (0.0, 0.0)
    assert len(result.equity_curve) == 1
    assert result.equity_curve.iloc[0] == 50_000.0
