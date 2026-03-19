import sys
sys.path.append('.')  # allow imports from project root

from data.storage import DataStorage
from data.tasks import TaskGenerator
from meta_learning.train import train_meta
import config.settings as settings
import yaml

def main():
    # Load symbols
    with open(settings.SYMBOLS_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    symbols = [s['name'] for s in config['symbols'] if s.get('tradable', True)]
    
    # Initialize storage and task generator
    storage = DataStorage()
    tg = TaskGenerator(storage, symbols)
    
    # Define date range for training (e.g., last 5 years)
    import datetime
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5*365)
    
    tasks = tg.create_tasks(start_date, end_date)
    print(f"Created {len(tasks)} tasks")
    
    # Determine input/output dimensions
    feature_cols = tg.feature_cols
    num_features = len(feature_cols)
    num_symbols = len(symbols)
    input_dim = num_features * num_symbols   # when averaging over time
    output_dim = len(settings.PARAMETER_BOUNDS)  # number of parameters per symbol
    
    # Train
    model = train_meta(tasks, input_dim, output_dim)
    print("Meta-training complete. Model saved.")

if __name__ == "__main__":
    main()
