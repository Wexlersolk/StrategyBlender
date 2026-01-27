#!/usr/bin/env python3
"""
StrategyBlender - Main Application
Multi-Strategy Trading System for Arch Linux
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Any
import yaml
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Local imports
from core.config import load_config, Settings
from core.database import init_db, get_session
from adapters.metatrader5_adapter import create_mt5_adapter
from core.processor import SignalProcessor
from core.allocation_engine import AllocationEngine
from execution.execution_engine import ExecutionEngine

class StrategyBlender:
    """Main StrategyBlender application"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.settings = None
        self.running = False
        self.adapters = {}
        self.processor = None
        self.allocation_engine = None
        self.execution_engine = None
        self.db_session = None
        self.redis = None
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/strategyblender.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize all components"""
        self.logger.info("🚀 Initializing StrategyBlender...")
        
        try:
            # Load configuration
            self.settings = load_config(self.config_path)
            self.logger.info(f"✅ Loaded config: {self.settings.project.name} v{self.settings.project.version}")
            
            # Initialize database
            await self.init_database()
            
            # Initialize Redis
            await self.init_redis()
            
            # Initialize adapters
            await self.init_adapters()
            
            # Initialize core components
            await self.init_core_components()
            
            self.logger.info("✅ StrategyBlender initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            return False
    
    async def init_database(self):
        """Initialize PostgreSQL database"""
        self.logger.info("📊 Initializing database...")
        
        db_config = self.settings.database.postgresql
        database_url = (
            f"postgresql+asyncpg://{db_config.username}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )
        
        engine = create_async_engine(database_url, echo=False)
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        self.db_session = async_session
        
        # Initialize database schema
        await init_db(engine)
        
        self.logger.info("✅ Database initialized")
    
    async def init_redis(self):
        """Initialize Redis connection"""
        self.logger.info("🔴 Initializing Redis...")
        
        redis_config = self.settings.database.redis
        self.redis = await aioredis.from_url(
            f"redis://{redis_config.host}:{redis_config.port}/{redis_config.db}",
            encoding="utf-8",
            decode_responses=True
        )
        
        # Test connection
        await self.redis.ping()
        self.logger.info("✅ Redis connected")
    
    async def init_adapters(self):
        """Initialize trading platform adapters"""
        self.logger.info("🔌 Initializing adapters...")
        
        for strategy_config in self.settings.strategies:
            if not strategy_config.enabled:
                continue
            
            source = strategy_config.source
            strategy_id = strategy_config.id
            
            if source == "metatrader5":
                # Configure MT5 adapter
                mt5_config = {
                    "login": self.settings.trading.account.get("mt5_login", 0),
                    "password": self.settings.trading.account.get("mt5_password", ""),
                    "server": self.settings.trading.account.get("mt5_server", ""),
                    "signal_file": strategy_config.files.signal_file,
                    "order_file": strategy_config.files.order_file,
                    "symbols": ["EURUSD", "GBPUSD", "XAUUSD"]
                }
                
                adapter = create_mt5_adapter(mt5_config)
                
                if await adapter.connect():
                    self.adapters[strategy_id] = adapter
                    self.logger.info(f"✅ Connected adapter for {strategy_id}")
                else:
                    self.logger.warning(f"⚠️ Failed to connect adapter for {strategy_id}")
            
            elif source == "python":
                # TODO: Implement Python strategy adapter
                self.logger.info(f"Python strategy: {strategy_id}")
            
            elif source == "tradingview":
                # TODO: Implement TradingView adapter
                self.logger.info(f"TradingView strategy: {strategy_id}")
    
    async def init_core_components(self):
        """Initialize core processing components"""
        self.logger.info("⚙️ Initializing core components...")
        
        # Signal processor
        self.processor = SignalProcessor(
            db_session=self.db_session,
            redis=self.redis
        )
        
        # Allocation engine
        self.allocation_engine = AllocationEngine(
            db_session=self.db_session,
            ml_models={},  # Will be initialized separately
            settings=self.settings
        )
        
        # Execution engine
        self.execution_engine = ExecutionEngine(
            adapters=self.adapters,
            settings=self.settings
        )
        
        self.logger.info("✅ Core components initialized")
    
    async def run_cycle(self):
        """Run one processing cycle"""
        try:
            self.logger.debug("🔄 Running processing cycle...")
            
            # 1. Fetch signals from all adapters
            all_signals = []
            for strategy_id, adapter in self.adapters.items():
                signals = await adapter.fetch_signals()
                all_signals.extend(signals)
            
            if all_signals:
                self.logger.info(f"📨 Processing {len(all_signals)} signal(s)")
            
            # 2. Process signals
            await self.processor.process_signals(all_signals)
            
            # 3. Calculate optimal allocation
            allocation = await self.allocation_engine.calculate_optimal_allocation()
            
            # 4. Execute orders based on allocation
            if allocation:
                await self.execution_engine.execute_orders(allocation)
            
            # 5. Update monitoring metrics
            await self.update_monitoring()
            
        except Exception as e:
            self.logger.error(f"❌ Error in processing cycle: {e}")
    
    async def update_monitoring(self):
        """Update monitoring metrics and dashboard"""
        # Store metrics in Redis for dashboard
        current_time = datetime.utcnow().isoformat()
        
        # Get current portfolio value (simulated)
        portfolio_value = 100000  # TODO: Calculate actual value
        
        await self.redis.hset(
            "strategyblender:metrics",
            mapping={
                "last_update": current_time,
                "portfolio_value": str(portfolio_value),
                "active_strategies": str(len(self.adapters))
            }
        )
    
    async def run(self):
        """Main application loop"""
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("🚀 StrategyBlender starting...")
        
        # Main loop
        while self.running:
            try:
                await self.run_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(self.settings.trading.execution.allocation_frequency)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.running = False
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🔴 Shutting down StrategyBlender...")
        
        # Close all adapters
        for adapter in self.adapters.values():
            await adapter.disconnect()
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        self.logger.info("✅ Shutdown complete")

async def main():
    """Application entry point"""
    app = StrategyBlender()
    
    try:
        # Initialize
        if not await app.initialize():
            sys.exit(1)
        
        # Run main loop
        await app.run()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        await app.shutdown()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    import os
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Run the application
    asyncio.run(main())
