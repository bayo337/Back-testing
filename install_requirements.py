#!/usr/bin/env python3
"""
Installation script for the Backtesting Project
This script installs all required dependencies and sets up the project environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install requirements from requirements.txt."""
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt --quiet",
        "Installing Python requirements"
    )

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'historical-DATA',
        'daily-historical-DATA',
        'analysis_charts',
        'GAP_FADE/equity_charts',
        'GAP_FADE/reports_txt',
        'GAP_FADE/trade_logs_csv',
        'GAP_FADE/climactic_reversal_test',
        'GAP_FADE/acoutposition back test'
    ]
    
    print("\nCreating necessary directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_environment_file():
    """Check if polygon.env file exists."""
    if not os.path.exists('polygon.env'):
        print("\n⚠️  Warning: polygon.env file not found")
        print("You may need to create this file with your Polygon API key")
        print("Format: POLYGON_API_KEY=your_api_key_here")
    else:
        print("✅ polygon.env file found")

def main():
    """Main installation function."""
    print("🚀 Starting Backtesting Project Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Installation failed. Please check the errors above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check environment file
    check_environment_file()
    
    print("\n" + "=" * 50)
    print("✅ Installation completed successfully!")
    print("\nNext steps:")
    print("1. Set up your Polygon API key in polygon.env file")
    print("2. Run your backtesting scripts")
    print("3. Check the README.md for usage instructions")
    print("\nHappy backtesting! 📈")

if __name__ == "__main__":
    main() 