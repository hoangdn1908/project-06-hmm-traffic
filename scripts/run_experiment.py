from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_sequence import generate_sequence, noisy_emission_matrix  # noqa: E402
from starter import loader, traffic_core  # noqa: E402

NOISE_LEVELS = [0.05, 0.15, 0.30]


def build_noisy_config(base_config: dict, noise_level: float) -> dict:
    """Return a copy of base_config with emission_matrix blended for noise_level."""
    config = dict(base_config)
    config["emission_matrix"] = noisy_emission_matrix(
        base_config["base_emission_matrix"],
        noise_level,
    )
    return config


def evaluate_sequence(
    sequence: List[Dict[str, object]],
    hmm_config: dict,
    noise_level: float,
) -> Dict[str, float]:
    """Run Forward, Viterbi, and accuracy metrics on a synthetic sequence."""
    observations = [str(row["observation"]) for row in sequence]
    true_states = [str(row["hidden_state"]) for row in sequence]
    states = hmm_config["hidden_states"]

    run_config = build_noisy_config(hmm_config, noise_level)

    forward_result = traffic_core.forward(observations, run_config)
    if len(forward_result) != len(observations):
        raise AssertionError("Forward did not return one distribution per timestep.")

    final_sum = sum(forward_result[-1].values())
    if abs(final_sum - 1.0) > 1e-9:
        raise AssertionError(f"Final Forward posterior does not sum to 1. Got {final_sum}")

    forward_map_states = [max(states, key=lambda s: dist[s]) for dist in forward_result]
    estimated_states, path_probability = traffic_core.viterbi(observations, run_config)
    if len(estimated_states) != len(true_states):
        raise AssertionError("Viterbi path length does not match ground truth length.")

    return {
        "steps": len(observations),
        "viterbi_acc": traffic_core.accuracy_against_ground_truth(estimated_states, true_states),
        "forward_acc": traffic_core.accuracy_against_ground_truth(forward_map_states, true_states),
        "naive_acc": traffic_core.accuracy_against_ground_truth(observations, true_states),
        "path_probability": path_probability,
    }


def main() -> None:
    hmm_config = loader.load_hmm_config()
    scenarios = loader.load_scenarios()

    print("=" * 78)
    print("Project 06 - Controlled Noise Experiment (Committed seed=42 Sequences)")
    print("=" * 78)
    print(f"{'Noise':<8}{'Steps':<8}{'Viterbi Acc':<15}{'Forward Acc':<15}{'Naive Acc':<13}{'Path Prob'}")
    print("-" * 78)

    for noise_level in NOISE_LEVELS:
        sequence = loader.load_sequence(noise_level)
        metrics = evaluate_sequence(sequence, hmm_config, noise_level)
        print(
            f"{noise_level * 100:>4.0f}%   "
            f"{metrics['steps']:<8}"
            f"{metrics['viterbi_acc'] * 100:>6.2f}%        "
            f"{metrics['forward_acc'] * 100:>6.2f}%        "
            f"{metrics['naive_acc'] * 100:>6.2f}%      "
            f"{metrics['path_probability']:.4e}"
        )

    print()
    print("=" * 78)
    print("Project 06 - Benchmark Stress & Edge Scenarios (data/scenarios.json)")
    print("=" * 78)
    print(f"{'Scenario ID':<29}{'Noise':<8}{'Steps':<7}{'Viterbi':<10}{'Forward':<10}{'Naive'}")
    print("-" * 78)

    for scenario in scenarios:
        gen = scenario["generator"]
        noise_level = float(gen["noise_level"])
        sequence = generate_sequence(
            length=int(gen["length"]),
            seed=int(gen["seed"]),
            noise_level=noise_level,
        )
        metrics = evaluate_sequence(sequence, hmm_config, noise_level)
        print(
            f"{scenario['id']:<29}"
            f"{noise_level * 100:>4.0f}%   "
            f"{metrics['steps']:<7}"
            f"{metrics['viterbi_acc'] * 100:>6.2f}%   "
            f"{metrics['forward_acc'] * 100:>6.2f}%   "
            f"{metrics['naive_acc'] * 100:>6.2f}%"
        )

    print()
    print("All Forward, Viterbi, and Accuracy checks PASSED.")


if __name__ == "__main__":
    main()
