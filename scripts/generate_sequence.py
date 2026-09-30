"""Generate a synthetic hidden-state / observation sequence from the course HMM.

This script only *samples data* from the model described in ``data/hmm_config.json``. It does
not perform inference (forward or Viterbi) and must not be imported by the graded student core
as a shortcut: the AI core the students must write is hidden-state estimation
(``forward`` / ``viterbi`` in ``starter/traffic_core.py``), not this generator.

All sequences produced by this script are synthetic educational data. They do not describe any
real traffic segment or real queueing system.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_hmm_config(path: Path = DATA_DIR / "hmm_config.json") -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def noisy_emission_matrix(base_emission: Dict[str, Dict[str, float]], noise_level: float) -> Dict[str, Dict[str, float]]:
    """Blend the base emission matrix with a uniform distribution.

    ``emission[state][obs] = (1 - noise_level) * base[state][obs] + noise_level * uniform``

    Every row of the result still sums to 1.0 because both the base row and the uniform row do.
    """
    if not 0.0 <= noise_level <= 1.0:
        raise ValueError("noise_level must be between 0.0 and 1.0")

    states = list(base_emission.keys())
    symbols = list(next(iter(base_emission.values())).keys())
    uniform_probability = 1.0 / len(symbols)

    blended: Dict[str, Dict[str, float]] = {}
    for state in states:
        blended[state] = {
            symbol: (1.0 - noise_level) * base_emission[state][symbol] + noise_level * uniform_probability
            for symbol in symbols
        }
    return blended


def _sample_categorical(rng: random.Random, distribution: Dict[str, float]) -> str:
    roll = rng.random()
    cumulative = 0.0
    last_key = None
    for key, probability in distribution.items():
        cumulative += probability
        last_key = key
        if roll < cumulative:
            return key
    return last_key  # floating-point safety net


def generate_sequence(length: int, seed: int, noise_level: float) -> List[Dict[str, object]]:
    """Sample a synthetic ``length``-step (hidden_state, observation) sequence.

    Sampling is deterministic for a fixed ``(length, seed, noise_level)`` triple: it draws the
    initial hidden state, then one transition draw and one emission draw per subsequent step, in
    that fixed order, from a single ``random.Random(seed)`` stream.
    """
    if length < 1:
        raise ValueError("length must be a positive integer")

    config = load_hmm_config()
    initial_distribution = config["initial_distribution"]
    transition_matrix = config["transition_matrix"]
    emission_matrix = noisy_emission_matrix(config["base_emission_matrix"], noise_level)

    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []

    hidden_state = _sample_categorical(rng, initial_distribution)
    for step in range(length):
        if step > 0:
            hidden_state = _sample_categorical(rng, transition_matrix[hidden_state])
        observation = _sample_categorical(rng, emission_matrix[hidden_state])
        rows.append({"step": step, "hidden_state": hidden_state, "observation": observation})

    return rows


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "hidden_state", "observation"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: List[Dict[str, object]], output_path: Path, length: int, seed: int, noise_level: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"length": length, "seed": seed, "noise_level": noise_level, "sequence": rows}
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _noise_tag(noise_level: float) -> str:
    return f"{round(noise_level * 100):02d}"


def _default_output(length: int, seed: int, noise_level: float, fmt: str) -> Path:
    return DATA_DIR / f"sequence_len{length}_noise{_noise_tag(noise_level)}_seed{seed}.{fmt}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a synthetic hidden-state / observation sequence from the course HMM. "
            "All output is synthetic educational data, not a real sensor feed."
        )
    )
    parser.add_argument("--length", type=int, default=200, help="Sequence length (default: 200).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--noise", type=float, default=0.15, help="Noise level in [0, 1] (default: 0.15).")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format (default: csv).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to data/sequence_len<length>_noise<NN>_seed<seed>.<format>.",
    )
    args = parser.parse_args(argv)

    rows = generate_sequence(args.length, args.seed, args.noise)
    output_path = args.output or _default_output(args.length, args.seed, args.noise, args.format)

    if args.format == "csv":
        write_csv(rows, output_path)
    else:
        write_json(rows, output_path, args.length, args.seed, args.noise)

    print(
        f"Wrote {len(rows)} synthetic steps (seed={args.seed}, noise={args.noise}) to {output_path}"
    )


if __name__ == "__main__":
    main()
