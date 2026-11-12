"""Manual smoke script for starter/traffic_core.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter.traffic_core import accuracy_against_ground_truth, forward, viterbi  # noqa: E402


def main() -> None:
    with (PROJECT_ROOT / "data" / "hmm_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    observations = ["LOW", "LOW", "MEDIUM", "HIGH", "HIGH"]

    result = forward(observations, config)
    for t, probabilities in enumerate(result):
        print(f"t = {t}")
        print(probabilities)
        print("sum =", sum(probabilities.values()))
        print()

    path, probability = viterbi(observations, config)
    print("Viterbi path:", path)
    print("Probability:", probability)
    print()

    estimated = ["LOW", "LOW", "MEDIUM", "HIGH", "HIGH"]
    true = ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH"]
    accuracy = accuracy_against_ground_truth(estimated, true)
    print("Accuracy:", accuracy)
    print("Accuracy %:", accuracy * 100)


if __name__ == "__main__":
    main()
