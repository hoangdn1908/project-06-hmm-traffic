from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starter import loader, traffic_core


NOISE_LEVELS = [0.05, 0.15, 0.30]


def build_noisy_config(base_config: dict, noise_level: float) -> dict:
    states = base_config["hidden_states"]
    observations = base_config["observation_symbols"]
    base_emission = base_config["base_emission_matrix"]

    config = dict(base_config)
    noisy_emission = {}

    for state in states:
        noisy_emission[state] = {}

        for observation in observations:
            noisy_emission[state][observation] = (
                (1 - noise_level) * base_emission[state][observation]
                + noise_level / len(observations)
            )

    config["emission_matrix"] = noisy_emission
    return config


def main():
    hmm_config = loader.load_hmm_config()

    print("=" * 70)
    print("Project 06 - Controlled Experiment")
    print("=" * 70)
    print()

    print(f"{'Noise':<10}{'Steps':<10}{'Accuracy':<15}{'Path probability'}")
    print("-" * 70)

    for noise_level in NOISE_LEVELS:
        sequence = loader.load_sequence(noise_level)

        observations = [row["observation"] for row in sequence]
        true_states = [row["hidden_state"] for row in sequence]

        run_config = build_noisy_config(hmm_config, noise_level)

        # 1. Forward
        forward_result = traffic_core.forward(
            observations,
            run_config,
        )

        if len(forward_result) != len(observations):
            raise AssertionError(
                "Forward did not return one distribution per timestep."
            )

        final_sum = sum(forward_result[-1].values())

        if abs(final_sum - 1.0) > 1e-9:
            raise AssertionError(
                f"Final Forward posterior does not sum to 1. Got {final_sum}"
            )

        # 2. Viterbi
        estimated_states, path_probability = traffic_core.viterbi(
            observations,
            run_config,
        )

        if len(estimated_states) != len(true_states):
            raise AssertionError(
                "Viterbi path length does not match ground truth length."
            )

        # 3. Accuracy
        accuracy = traffic_core.accuracy_against_ground_truth(
            estimated_states,
            true_states,
        )

        print(
            f"{noise_level * 100:>5.0f}%"
            f"{len(observations):>10}"
            f"{accuracy * 100:>10.2f}%"
            f"        {path_probability:.6e}"
        )

    print()
    print("Forward check: PASSED")
    print("Viterbi check: PASSED")
    print("Accuracy check: PASSED")
    print()
    print("These accuracy values are the results to use for the Midterm experiment.")


if __name__ == "__main__":
    main()
