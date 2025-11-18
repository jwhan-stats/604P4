#!/usr/bin/env python3
"""
Complete prediction workflow for Elastic Net model
1. Prepare test data for given date
2. Make predictions using trained models
3. Display predictions in required format
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors"""
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return False

def main():
    # Get target date from command line or use today
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')

    # Change to codes directory
    codes_dir = Path(__file__).parent

    # Step 1: Prepare test data
    cmd1 = f"cd {codes_dir} && python3 prepare_test_data.py {target_date}"
    if not run_command(cmd1, "preparing test data"):
        sys.exit(1)

    # Step 2: Make predictions
    cmd2 = f"cd {codes_dir} && python3 make_predictions.py"
    if not run_command(cmd2, "making predictions"):
        sys.exit(1)

    # Step 3: Display predictions
    cmd3 = f"cd {codes_dir} && python3 show_predictions.py"
    if not run_command(cmd3, "displaying predictions"):
        sys.exit(1)

if __name__ == '__main__':
    main()
