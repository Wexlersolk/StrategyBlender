# Meta-Learning for Adaptive Algorithmic Trading

This project implements a meta-learning layer on top of existing MetaTrader 5 trading bots.  
It enables rapid adaptation of trading parameters to changing market conditions using only a few days of new data.

See the project overview document for full details.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Configure symbols and MT5 connection in `config/settings.py` and `config/symbols.yaml`
3. Fetch historical data: `python scripts/update_data.py`
4. Train the meta-model: `python scripts/train_meta.py`
5. Run live adaptation: `python main.py`

## Project Structure

- `config/` – configuration files
- `data/` – data fetching, storage, feature engineering, task creation
- `meta_learning/` – meta-learning model, MAML implementation, training & adaptation
- `execution/` – MT5 bridge, risk management, parameter validation
- `monitoring/` – logging and optional dashboard
- `scripts/` – utility scripts
- `tests/` – unit tests (to be written)
