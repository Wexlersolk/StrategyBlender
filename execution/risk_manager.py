import config.settings as settings

class RiskManager:
    """Enforces portfolio-level risk limits."""
    
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.equity_history = []
    
    def update_equity(self, current_equity):
        self.equity_history.append(current_equity)
    
    def check_drawdown(self):
        """Return True if drawdown exceeds max allowed."""
        if not self.equity_history:
            return False
        peak = max(self.equity_history)
        drawdown = (peak - self.equity_history[-1]) / peak
        return drawdown > settings.MAX_DRAWDOWN_PERCENT
    
    def reduce_risk(self, params):
        """Halve lot sizes if drawdown limit exceeded."""
        reduced = params.copy()
        reduced['lot_size'] = params['lot_size'] / 2
        return reduced
