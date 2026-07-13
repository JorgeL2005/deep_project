"""Interactive demonstration collection runner.

Usage:
    python -m src.collect_demonstrations --config configs/collect_demonstrations.yaml
"""

from __future__ import annotations

import argparse
import json

from src.config import load_yaml
from src.runner import run_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/collect_demonstrations.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    result = run_from_config(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
