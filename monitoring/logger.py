import logging
import json
from datetime import datetime

class AdaptationLogger:
    """Logs adaptation events and performance."""
    
    def __init__(self, log_file="adaptation.log"):
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s %(message)s')
        self.logger = logging.getLogger()
    
    def log_adaptation(self, symbol, old_params, new_params, reason=""):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'old_params': old_params,
            'new_params': new_params,
            'reason': reason
        }
        self.logger.info(json.dumps(entry))
    
    def log_performance(self, metrics):
        self.logger.info(f"PERF: {json.dumps(metrics)}")
