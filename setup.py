# StrategyBlender - Setup Script
# This creates the basic structure for your trading system

print("🚀 Welcome to StrategyBlender!")
print("Setting up your trading system...")

# Create basic configuration
config = {
    "project_name": "StrategyBlender",
    "version": "1.0.0",
    "description": "Multi-Strategy Trading Platform",
    "author": "Your Name",
    "strategies": []
}

# Display configuration
print("\n📋 Project Configuration:")
print(f"Name: {config['project_name']}")
print(f"Version: {config['version']}")
print(f"Description: {config['description']}")

# Create strategy examples
strategies = [
    {
        "name": "Trend Follower",
        "source": "MetaTrader",
        "description": "Follows market trends"
    },
    {
        "name": "Mean Reversion",
        "source": "Python Script",
        "description": "Trades price reversions"
    },
    {
        "name": "Breakout Strategy",
        "source": "TradingView",
        "description": "Trades breakout patterns"
    }
]

print("\n🎯 Available Strategies:")
for i, strategy in enumerate(strategies, 1):
    print(f"{i}. {strategy['name']} ({strategy['source']})")
    print(f"   {strategy['description']}")

print("\n✅ Setup complete!")
print("\nNext steps:")
print("1. Run this file by double-clicking it")
print("2. Create your first strategy adapter")
print("3. Connect to MetaTrader or your trading platform")

input("\nPress Enter to exit...")
