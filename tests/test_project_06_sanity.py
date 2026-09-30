"""Sanity tests for Project 06. These check data integrity and starter imports.

They do NOT grade the student's forward/Viterbi implementation.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _load_module(name: str, path: Path):
    """Load a module from an explicit file path under a project-unique name.

    Avoids sys.modules collisions with same-named ``starter`` packages or
    same-named generator scripts from sibling projects when tests from multiple
    project trees run inside one pytest session.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


loader = _load_module("project06_starter_loader", PROJECT_ROOT / "starter" / "loader.py")
traffic_core = _load_module("project06_starter_traffic_core", PROJECT_ROOT / "starter" / "traffic_core.py")
generate_sequence = _load_module("project06_generate_sequence", PROJECT_ROOT / "scripts" / "generate_sequence.py")


# ---------------------------------------------------------------------------
# hmm_config.json
# ---------------------------------------------------------------------------


def test_hmm_config_has_required_states():
    config = loader.load_hmm_config()
    assert config["hidden_states"] == ["LOW", "MEDIUM", "HIGH"]
    assert config["observation_symbols"] == ["LOW", "MEDIUM", "HIGH"]


def test_hmm_config_has_synthetic_notice():
    config = loader.load_hmm_config()
    assert "synthetic" in config["synthetic_data_notice"].lower()


def test_initial_distribution_sums_to_one():
    config = loader.load_hmm_config()
    total = sum(config["initial_distribution"].values())
    assert total == pytest.approx(1.0, abs=1e-6)


def test_transition_matrix_rows_sum_to_one():
    config = loader.load_hmm_config()
    for state, row in config["transition_matrix"].items():
        total = sum(row.values())
        assert total == pytest.approx(1.0, abs=1e-6), f"transition row {state} sums to {total}"


def test_base_emission_matrix_rows_sum_to_one():
    config = loader.load_hmm_config()
    for state, row in config["base_emission_matrix"].items():
        total = sum(row.values())
        assert total == pytest.approx(1.0, abs=1e-6), f"emission row {state} sums to {total}"


@pytest.mark.parametrize("noise_level", [0.0, 0.05, 0.15, 0.30, 1.0])
def test_noisy_emission_matrix_rows_still_sum_to_one(noise_level):
    config = loader.load_hmm_config()
    blended = generate_sequence.noisy_emission_matrix(config["base_emission_matrix"], noise_level)
    for state, row in blended.items():
        total = sum(row.values())
        assert total == pytest.approx(1.0, abs=1e-6), f"blended row {state} sums to {total}"


# ---------------------------------------------------------------------------
# scenarios.json
# ---------------------------------------------------------------------------


def test_scenarios_cover_required_classes():
    scenarios = loader.load_scenarios()
    classes = {scenario["class"] for scenario in scenarios}
    assert classes == {
        "stable_low",
        "rapid_transition",
        "high_noise",
        "misleading_observations",
    }


def test_scenarios_have_unique_ids():
    scenarios = loader.load_scenarios()
    ids = [scenario["id"] for scenario in scenarios]
    assert len(ids) == len(set(ids))
    for scenario_id in ids:
        assert scenario_id.startswith("P06-")


# ---------------------------------------------------------------------------
# generate_sequence.py
# ---------------------------------------------------------------------------


def test_generate_sequence_is_deterministic():
    seq_a = generate_sequence.generate_sequence(length=50, seed=42, noise_level=0.15)
    seq_b = generate_sequence.generate_sequence(length=50, seed=42, noise_level=0.15)
    assert seq_a == seq_b


def test_generate_sequence_length_and_alphabet():
    config = loader.load_hmm_config()
    seq = generate_sequence.generate_sequence(length=40, seed=7, noise_level=0.1)
    assert len(seq) == 40
    for i, row in enumerate(seq):
        assert row["step"] == i
        assert row["hidden_state"] in config["hidden_states"]
        assert row["observation"] in config["observation_symbols"]


@pytest.mark.parametrize(
    "noise_level,filename",
    [
        (0.05, "sequence_len200_noise05_seed42.csv"),
        (0.15, "sequence_len200_noise15_seed42.csv"),
        (0.30, "sequence_len200_noise30_seed42.csv"),
    ],
)
def test_committed_sequences_match_regeneration(noise_level, filename):
    committed_path = DATA_DIR / filename
    with committed_path.open("r", encoding="utf-8", newline="") as handle:
        committed_rows = list(csv.DictReader(handle))

    regenerated = generate_sequence.generate_sequence(length=200, seed=42, noise_level=noise_level)

    assert len(committed_rows) == len(regenerated) == 200
    for committed, fresh in zip(committed_rows, regenerated):
        assert int(committed["step"]) == fresh["step"]
        assert committed["hidden_state"] == fresh["hidden_state"]
        assert committed["observation"] == fresh["observation"]


# ---------------------------------------------------------------------------
# starter loader + traffic_core stub
# ---------------------------------------------------------------------------


def test_loader_reads_committed_sequences():
    for noise_level in (0.05, 0.15, 0.30):
        rows = loader.load_sequence(noise_level)
        assert len(rows) == 200


def test_forward_is_not_implemented():
    config = loader.load_hmm_config()
    observations = ["LOW", "LOW", "MEDIUM"]
    with pytest.raises(NotImplementedError):
        traffic_core.forward(observations, config)


def test_viterbi_is_not_implemented():
    config = loader.load_hmm_config()
    observations = ["LOW", "LOW", "MEDIUM"]
    with pytest.raises(NotImplementedError):
        traffic_core.viterbi(observations, config)


def test_starter_does_not_ship_a_working_solver():
    source = (PROJECT_ROOT / "starter" / "traffic_core.py").read_text(encoding="utf-8")
    assert source.count("raise NotImplementedError") >= 2
