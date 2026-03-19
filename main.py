# Entry point for live operation (could be run continuously with scheduler)
import sys
from scripts.run_live import main as live_main

if __name__ == "__main__":
    live_main()
