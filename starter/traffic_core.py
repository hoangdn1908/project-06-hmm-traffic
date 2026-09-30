"""Graded AI core for Project 06 — YOUR implementation goes here.

This module intentionally ships EMPTY. Implementing ``forward`` and ``viterbi`` is the graded
midterm AI core (hidden-state estimation in a Hidden Markov Model). Do not import a third-party
HMM library to replace these functions: write and explain your own implementation.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
"""Graded AI core for Project 06 — Hidden Markov Model inference."""

import math


def _get_model_parts(hmm_config: dict):
    """Get the HMM components from the project configuration."""

    states = hmm_config["hidden_states"]
    initial = hmm_config["initial_distribution"]
    transition = hmm_config["transition_matrix"]
    emission = hmm_config.get(
        "emission_matrix",
        hmm_config["base_emission_matrix"]
    )

    return states, initial, transition, emission

def forward(observations: Sequence[str], hmm_config: dict) -> List[Dict[str, float]]:
    """Run the Forward algorithm.

    Returns:
        A list containing one probability distribution for each timestep.
    """

    if not observations:
        return []

    states, initial, transition, emission = _get_model_parts(hmm_config)

    results = []

    # ---------------------------------------------------------
    # Step 1: First observation
    # ---------------------------------------------------------

    alpha = {}

    first_observation = observations[0]

    for state in states:
        alpha[state] = (
            initial[state]
            * emission[state][first_observation]
        )

    # Normalize so that probabilities sum to 1
    total = sum(alpha.values())

    if total == 0:
        raise ValueError("Forward probabilities became zero.")

    for state in states:
        alpha[state] /= total

    results.append(alpha.copy())

    # ---------------------------------------------------------
    # Step 2: Remaining observations
    # ---------------------------------------------------------

    for observation in observations[1:]:

        new_alpha = {}

        for current_state in states:

            # Probability of reaching current_state
            # from all possible previous states.
            probability_from_previous_states = 0.0

            for previous_state in states:
                probability_from_previous_states += (
                    alpha[previous_state]
                    * transition[previous_state][current_state]
                )

            # Add probability of observing this observation
            new_alpha[current_state] = (
                probability_from_previous_states
                * emission[current_state][observation]
            )

        # Normalize
        total = sum(new_alpha.values())

        if total == 0:
            raise ValueError("Forward probabilities became zero.")

        for state in states:
            new_alpha[state] /= total

        alpha = new_alpha

        results.append(alpha.copy())

    return results


def viterbi( observations: Sequence[str],hmm_config: dict) -> Tuple[List[str], float]:
    """Run the Viterbi algorithm.

    Returns:
        (most_likely_path, joint_probability)
    """

    if not observations:
        return [], 0.0

    states, initial, transition, emission = _get_model_parts(hmm_config)

    # We use logarithms internally to avoid numerical underflow.
    log_delta = {}
    backpointers = []

    first_observation = observations[0]

    # ---------------------------------------------------------
    # Step 1: Initialization
    # ---------------------------------------------------------

    for state in states:

        probability = (
            initial[state]
            * emission[state][first_observation]
        )

        if probability > 0:
            log_delta[state] = math.log(probability)
        else:
            log_delta[state] = float("-inf")

    # ---------------------------------------------------------
    # Step 2: Recursion
    # ---------------------------------------------------------

    for observation in observations[1:]:

        new_log_delta = {}
        current_backpointers = {}

        for current_state in states:

            best_previous_state = None
            best_probability = float("-inf")

            for previous_state in states:

                transition_probability = transition[
                    previous_state
                ][current_state]

                emission_probability = emission[
                    current_state
                ][observation]

                if (
                    log_delta[previous_state] == float("-inf")
                    or transition_probability <= 0
                    or emission_probability <= 0
                ):
                    candidate = float("-inf")
                else:
                    candidate = (
                        log_delta[previous_state]
                        + math.log(transition_probability)
                        + math.log(emission_probability)
                    )

                if candidate > best_probability:
                    best_probability = candidate
                    best_previous_state = previous_state

            new_log_delta[current_state] = best_probability
            current_backpointers[current_state] = best_previous_state

        log_delta = new_log_delta
        backpointers.append(current_backpointers)

    # ---------------------------------------------------------
    # Step 3: Find best final state
    # ---------------------------------------------------------

    best_final_state = max(
        states,
        key=lambda state: log_delta[state]
    )

    best_log_probability = log_delta[best_final_state]

    # ---------------------------------------------------------
    # Step 4: Backtracking
    # ---------------------------------------------------------

    path = [best_final_state]

    for backpointer in reversed(backpointers):

        previous_state = backpointer[path[-1]]

        path.append(previous_state)

    path.reverse()

    # Convert log probability back to normal probability.
    if best_log_probability == float("-inf"):
        probability = 0.0
    else:
        probability = math.exp(best_log_probability)

    return path, probability


def accuracy_against_ground_truth(estimated_states: Sequence[str], true_states: Sequence[str]) -> float:
    """Calculate the percentage of correctly estimated states."""

    if len(estimated_states) != len(true_states):
        raise ValueError(
            "Estimated states and true states must have the same length."
        )

    if len(true_states) == 0:
        return 0.0

    correct = 0

    for estimated, true in zip(estimated_states, true_states):
        if estimated == true:
            correct += 1

    return correct / len(true_states)