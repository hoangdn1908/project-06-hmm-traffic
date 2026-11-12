"""Unit tests for Project 06 AI core and Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_sequence import generate_sequence, noisy_emission_matrix  # noqa: E402
from starter import loader, traffic_core  # noqa: E402


def test_forward_empty_observations():
    config = loader.load_hmm_config()
    assert traffic_core.forward([], config) == []


def test_forward_matches_manual_first_step_and_normalizes():
    config = loader.load_hmm_config()
    observations = ["LOW", "MEDIUM", "HIGH"]
    posteriors = traffic_core.forward(observations, config)

    assert len(posteriors) == 3
    for dist in posteriors:
        assert set(dist.keys()) == {"LOW", "MEDIUM", "HIGH"}
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 <= p <= 1.0 for p in dist.values())

    # Step 0 unnormalized: LOW=0.5*0.90=0.45, MEDIUM=0.35*0.10=0.035, HIGH=0.15*0.02=0.003
    z0 = 0.45 + 0.035 + 0.003
    assert posteriors[0]["LOW"] == pytest.approx(0.45 / z0, abs=1e-9)
    assert posteriors[0]["MEDIUM"] == pytest.approx(0.035 / z0, abs=1e-9)
    assert posteriors[0]["HIGH"] == pytest.approx(0.003 / z0, abs=1e-9)


def test_viterbi_empty_observations():
    config = loader.load_hmm_config()
    path, prob = traffic_core.viterbi([], config)
    assert path == []
    assert prob == 0.0


def test_viterbi_matches_expected_joint_probability():
    config = loader.load_hmm_config()
    observations = ["LOW", "LOW"]
    path, prob = traffic_core.viterbi(observations, config)

    # Best path is ["LOW", "LOW"]: 0.5 * 0.9 * 0.75 * 0.9 = 0.30375
    assert path == ["LOW", "LOW"]
    assert prob == pytest.approx(0.5 * 0.9 * 0.75 * 0.9, abs=1e-9)


def test_viterbi_uses_blended_emission_matrix_when_present():
    config = dict(loader.load_hmm_config())
    config["emission_matrix"] = noisy_emission_matrix(config["base_emission_matrix"], 0.30)
    path, prob = traffic_core.viterbi(["HIGH", "HIGH", "HIGH"], config)

    assert len(path) == 3
    assert path == ["HIGH", "HIGH", "HIGH"]
    assert prob > 0.0


def test_viterbi_handles_zero_probability_safely():
    zero_config = {
        "hidden_states": ["LOW", "HIGH"],
        "observation_symbols": ["LOW", "HIGH"],
        "initial_distribution": {"LOW": 1.0, "HIGH": 0.0},
        "transition_matrix": {
            "LOW": {"LOW": 0.0, "HIGH": 0.0},
            "HIGH": {"LOW": 0.0, "HIGH": 0.0},
        },
        "base_emission_matrix": {
            "LOW": {"LOW": 1.0, "HIGH": 0.0},
            "HIGH": {"LOW": 0.0, "HIGH": 1.0},
        },
    }
    path, prob = traffic_core.viterbi(["LOW", "HIGH"], zero_config)
    assert len(path) == 2
    assert prob == 0.0


def test_accuracy_against_ground_truth():
    assert traffic_core.accuracy_against_ground_truth([], []) == 0.0
    assert traffic_core.accuracy_against_ground_truth(["LOW", "HIGH"], ["LOW", "MEDIUM"]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        traffic_core.accuracy_against_ground_truth(["LOW"], ["LOW", "HIGH"])


def test_scenarios_run_with_hmm_core():
    base_config = loader.load_hmm_config()
    for scenario in loader.load_scenarios():
        gen = scenario["generator"]
        seq = generate_sequence(int(gen["length"]), int(gen["seed"]), float(gen["noise_level"]))
        obs = [str(r["observation"]) for r in seq]
        truth = [str(r["hidden_state"]) for r in seq]

        cfg = dict(base_config)
        cfg["emission_matrix"] = noisy_emission_matrix(base_config["base_emission_matrix"], float(gen["noise_level"]))

        posteriors = traffic_core.forward(obs, cfg)
        path, prob = traffic_core.viterbi(obs, cfg)
        acc = traffic_core.accuracy_against_ground_truth(path, truth)

        assert len(posteriors) == len(seq)
        assert len(path) == len(seq)
        assert prob >= 0.0
        assert 0.0 <= acc <= 1.0


def test_streamlit_app_renders_controls_scenarios_and_eval_mode():
    at = AppTest.from_file(str(PROJECT_ROOT / "starter" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    assert len(at.warning) >= 1  # persistent synthetic data notice

    # Step forward, backward, and reset
    next_btn = next(b for b in at.button if "+1" in b.label)
    next_btn.click().run()
    assert not at.exception
    assert at.session_state["playback_step"] == 1

    prev_btn = next(b for b in at.button if "-1" in b.label)
    prev_btn.click().run()
    assert not at.exception
    assert at.session_state["playback_step"] == 0

    next_btn = next(b for b in at.button if "+1" in b.label)
    next_btn.click().run()
    reset_btn = next(b for b in at.button if "Reset" in b.label)
    reset_btn.click().run()
    assert not at.exception
    assert at.session_state["playback_step"] == 0

    eval_cb = next(c for c in at.checkbox if "Evaluation mode" in c.label)
    eval_cb.check().run()
    assert not at.exception

    # Switch to Benchmark Scenarios and verify all 4 scenarios render cleanly
    radio = at.radio[0]
    radio.set_value("Benchmark Scenarios").run()
    assert not at.exception
    assert at.session_state["playback_step"] == 0

    for scenario in loader.load_scenarios():
        scenario_box = next(s for s in at.selectbox if s.label == "Scenario")
        scenario_box.set_value(scenario["id"]).run()
        assert not at.exception
