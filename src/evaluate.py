"""Evaluation runner.

Usage:
    python -m src.evaluate --config configs/evaluate.yaml
"""

from __future__ import annotations

import argparse
import json

from src.config import load_yaml
from src.runner import run_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/evaluate.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    result = run_from_config(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
