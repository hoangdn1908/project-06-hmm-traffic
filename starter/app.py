"""Streamlit interactive dashboard for Project 06 — Campus Crowd State Estimator.

Calls the graded HMM core (`forward`, `viterbi`, `accuracy_against_ground_truth`) in
`starter/traffic_core.py` and renders real-time playback, posterior beliefs, and evaluation overlays.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_sequence import generate_sequence, noisy_emission_matrix  # noqa: E402
from starter import loader, traffic_core  # noqa: E402

STATE_COLORS = {
    "LOW": "#2e8b57",
    "MEDIUM": "#e09f3e",
    "HIGH": "#c1121f",
}

SPEED_OPTIONS = {
    "Fast (0.05s / step)": 0.05,
    "Normal (0.15s / step)": 0.15,
    "Slow (0.35s / step)": 0.35,
    "Step-by-step (0.60s / step)": 0.60,
}


def _build_active_config(base_config: dict, noise_level: float) -> dict:
    """Return a copy of hmm_config with emission_matrix blended for noise_level."""
    config = dict(base_config)
    config["emission_matrix"] = noisy_emission_matrix(
        base_config["base_emission_matrix"],
        noise_level,
    )
    return config


@st.cache_data
def _load_committed(noise_level: float) -> List[Dict[str, object]]:
    return loader.load_sequence(noise_level)


@st.cache_data
def _generate_scenario_sequence(length: int, seed: int, noise_level: float) -> List[Dict[str, object]]:
    return generate_sequence(length=length, seed=seed, noise_level=noise_level)


def _chart_confidence_bar(distribution: Dict[str, float], states: List[str]) -> alt.LayerChart:
    """Build current-step posterior confidence bar chart P(X_t | y_{1:t})."""
    df = pd.DataFrame(
        {
            "State": states,
            "Probability": [distribution[state] for state in states],
        }
    )
    color_scale = alt.Scale(domain=states, range=[STATE_COLORS[s] for s in states])
    bars = (
        alt.Chart(df)
        .mark_bar(size=48)
        .encode(
            x=alt.X(
                "State:N",
                sort=states,
                title="Crowd state",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "Probability:Q",
                scale=alt.Scale(domain=[0.0, 1.05]),
                title="Posterior P(X_t | y_1:t)",
            ),
            color=alt.Color("State:N", scale=color_scale, legend=None),
            tooltip=["State:N", alt.Tooltip("Probability:Q", format=".4f")],
        )
    )
    labels = bars.mark_text(dy=-8, fontWeight="bold").encode(
        text=alt.Text("Probability:Q", format=".1%"),
        color=alt.value("#222222"),
    )
    return (bars + labels).properties(height=280, title="Current Step Confidence P(X_t | y_1:t)")


def _chart_belief_shifts(
    posteriors: List[Dict[str, float]],
    states: List[str],
) -> alt.Chart:
    """Build temporal line plot tracking posterior belief shifts over time."""
    rows = [
        {"step": t, "State": state, "Probability": dist[state]}
        for t, dist in enumerate(posteriors)
        for state in states
    ]
    df = pd.DataFrame(rows)
    color_scale = alt.Scale(domain=states, range=[STATE_COLORS[s] for s in states])
    current_max = max(1, len(posteriors) - 1)
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2.0)
        .encode(
            x=alt.X("step:Q", scale=alt.Scale(domain=[0, current_max]), title="Time step (t)"),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0.0, 1.05]), title="Posterior probability"),
            color=alt.Color("State:N", scale=color_scale, title="State"),
            tooltip=["step:Q", "State:N", alt.Tooltip("Probability:Q", format=".4f")],
        )
        .properties(height=280, title="Belief Shifts Over Time")
    )


def _chart_trajectory_timeline(
    states: List[str],
    observations: List[str],
    viterbi_path: List[str],
    true_states: List[str] | None,
    max_step: int,
) -> alt.LayerChart:
    """Build observation + Viterbi timeline with optional ground-truth overlay."""
    state_to_idx = {state: idx for idx, state in enumerate(states)}

    obs_df = pd.DataFrame(
        {
            "step": list(range(len(observations))),
            "State": observations,
            "StateIdx": [state_to_idx[s] for s in observations],
            "Series": "Observation (noisy)",
        }
    )
    line_rows = [
        {"step": t, "State": s, "StateIdx": state_to_idx[s], "Series": "Viterbi Estimate"}
        for t, s in enumerate(viterbi_path)
    ]
    series_domain = ["Observation (noisy)", "Viterbi Estimate"]
    series_colors = ["#6c757d", "#1d3557"]

    if true_states is not None:
        line_rows.extend(
            {"step": t, "State": s, "StateIdx": state_to_idx[s], "Series": "Ground Truth"}
            for t, s in enumerate(true_states)
        )
        series_domain.append("Ground Truth")
        series_colors.append("#2a9d8f")

    lines_df = pd.DataFrame(line_rows)
    color_scale = alt.Scale(domain=series_domain, range=series_colors)
    x_encoding = alt.X("step:Q", scale=alt.Scale(domain=[0, max(1, max_step)]), title="Time step (t)")
    label_expr = " : ".join(
        f"datum.value === {idx} ? '{state}'" for idx, state in enumerate(states[:-1])
    ) + f" : '{states[-1]}'"
    y_encoding = alt.Y(
        "StateIdx:Q",
        scale=alt.Scale(domain=[-0.35, len(states) - 0.65]),
        axis=alt.Axis(
            values=list(range(len(states))),
            labelExpr=label_expr,
        ),
        title="Crowd state",
    )

    obs_layer = (
        alt.Chart(obs_df)
        .mark_point(size=45, opacity=0.65, filled=True)
        .encode(
            x=x_encoding,
            y=y_encoding,
            color=alt.Color("Series:N", scale=color_scale, title="Series"),
            tooltip=["step:Q", "Series:N", "State:N"],
        )
    )
    line_layer = (
        alt.Chart(lines_df)
        .mark_line(interpolate="step-after", strokeWidth=2.4)
        .encode(
            x=x_encoding,
            y=y_encoding,
            order="step:Q",
            color=alt.Color("Series:N", scale=color_scale, title="Series"),
            strokeDash=alt.StrokeDash(
                "Series:N",
                scale=alt.Scale(
                    domain=["Viterbi Estimate", "Ground Truth"],
                    range=[[1, 0], [6, 4]],
                ),
                legend=None,
            ),
            tooltip=["step:Q", "Series:N", "State:N"],
        )
    )
    return (obs_layer + line_layer).properties(
        height=270,
        title="Observation Stream & Decoded Hidden State Trajectory",
    )


def main() -> None:
    st.set_page_config(page_title="Campus Crowd State Estimator", layout="wide")
    st.title("Campus Crowd State Estimator")
    st.warning(loader.SYNTHETIC_DATA_NOTICE, icon="⚠️")

    hmm_config = loader.load_hmm_config()
    scenarios = loader.load_scenarios()
    states = hmm_config["hidden_states"]

    with st.sidebar:
        st.header("Data Source")
        source_mode = st.radio(
            "Select trace source",
            ["Committed Noise Sequences", "Benchmark Scenarios"],
            index=0,
        )

        if source_mode == "Committed Noise Sequences":
            noise_level = st.selectbox(
                "Committed noise level",
                [0.05, 0.15, 0.30],
                index=1,
                format_func=lambda value: f"{int(round(value * 100))}%",
            )
            sequence = _load_committed(noise_level)
            source_key = ("committed", noise_level)
        else:
            scenario_ids = [scenario["id"] for scenario in scenarios]
            scenario_id = st.selectbox("Scenario", scenario_ids, index=0)
            chosen_scenario = next(s for s in scenarios if s["id"] == scenario_id)
            gen_cfg = chosen_scenario["generator"]
            noise_level = float(gen_cfg["noise_level"])
            sequence = _generate_scenario_sequence(
                length=int(gen_cfg["length"]),
                seed=int(gen_cfg["seed"]),
                noise_level=noise_level,
            )
            st.caption(chosen_scenario["description"])
            st.info(f"Expected behavior: {chosen_scenario['expected_direction']}")
            st.json(gen_cfg)
            source_key = ("scenario", scenario_id)

        st.divider()
        st.header("Playback & Evaluation")
        speed_label = st.selectbox("Playback speed", list(SPEED_OPTIONS.keys()), index=1)
        step_delay = SPEED_OPTIONS[speed_label]
        evaluation_mode = st.checkbox(
            "Evaluation mode (overlay synthetic ground truth)",
            value=False,
        )

    max_step = len(sequence) - 1

    # Reset playback state when switching dataset or scenario.
    if st.session_state.get("active_source_key") != source_key:
        st.session_state["active_source_key"] = source_key
        st.session_state["playback_step"] = 0
        st.session_state["slider_step"] = 0
        st.session_state["is_playing"] = False
        st.session_state["advance_next"] = False

    st.session_state.setdefault("playback_step", 0)
    st.session_state.setdefault("is_playing", False)

    if st.session_state.pop("advance_next", False) and st.session_state["is_playing"]:
        if st.session_state["playback_step"] < max_step:
            st.session_state["playback_step"] += 1
        else:
            st.session_state["is_playing"] = False

    st.session_state["playback_step"] = max(0, min(st.session_state["playback_step"], max_step))
    st.session_state["slider_step"] = st.session_state["playback_step"]

    def _toggle_play() -> None:
        if st.session_state["is_playing"]:
            st.session_state["is_playing"] = False
        else:
            if st.session_state["playback_step"] >= max_step:
                st.session_state["playback_step"] = 0
            st.session_state["is_playing"] = True

    def _step_prev() -> None:
        st.session_state["is_playing"] = False
        st.session_state["playback_step"] = max(0, st.session_state["playback_step"] - 1)

    def _step_next() -> None:
        st.session_state["is_playing"] = False
        st.session_state["playback_step"] = min(max_step, st.session_state["playback_step"] + 1)

    def _reset_playback() -> None:
        st.session_state["is_playing"] = False
        st.session_state["playback_step"] = 0

    def _on_slider_change() -> None:
        st.session_state["is_playing"] = False
        st.session_state["playback_step"] = st.session_state["slider_step"]

    ctrl_cols = st.columns([1.25, 0.95, 0.95, 1.25, 5.6])
    with ctrl_cols[0]:
        play_label = "⏸ Pause" if st.session_state["is_playing"] else "▶ Play"
        st.button(play_label, on_click=_toggle_play, use_container_width=True)
    with ctrl_cols[1]:
        st.button("⏮ -1", on_click=_step_prev, use_container_width=True)
    with ctrl_cols[2]:
        st.button("+1 ⏭", on_click=_step_next, use_container_width=True)
    with ctrl_cols[3]:
        st.button("↺ Reset", on_click=_reset_playback, use_container_width=True)
    with ctrl_cols[4]:
        st.slider(
            "Playback step",
            min_value=0,
            max_value=max_step,
            key="slider_step",
            on_change=_on_slider_change,
            label_visibility="collapsed",
        )

    step = st.session_state["playback_step"]
    active_config = _build_active_config(hmm_config, noise_level)

    observations_so_far = [str(row["observation"]) for row in sequence[: step + 1]]
    posteriors = traffic_core.forward(observations_so_far, active_config)
    viterbi_path, path_probability = traffic_core.viterbi(observations_so_far, active_config)
    forward_map_path = [max(states, key=lambda s: dist[s]) for dist in posteriors]

    current_posterior = posteriors[-1]
    current_obs = observations_so_far[-1]
    current_viterbi = viterbi_path[-1]
    current_forward_map = forward_map_path[-1]

    metric_cols = st.columns(5)
    metric_cols[0].metric("Time Step", f"{step} / {max_step}")
    metric_cols[1].metric("Sensor Observation", current_obs)
    metric_cols[2].metric("Viterbi Estimate", current_viterbi)
    metric_cols[3].metric("Forward MAP", current_forward_map)
    metric_cols[4].metric("Viterbi Joint Prob.", f"{path_probability:.2e}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.altair_chart(_chart_confidence_bar(current_posterior, states), use_container_width=True)

    with chart_right:
        st.altair_chart(_chart_belief_shifts(posteriors, states), use_container_width=True)

    true_states_so_far = (
        [str(row["hidden_state"]) for row in sequence[: step + 1]]
        if evaluation_mode
        else None
    )
    st.altair_chart(
        _chart_trajectory_timeline(
            states=states,
            observations=observations_so_far,
            viterbi_path=viterbi_path,
            true_states=true_states_so_far,
            max_step=max_step,
        ),
        use_container_width=True,
    )

    if evaluation_mode and true_states_so_far is not None:
        viterbi_acc = traffic_core.accuracy_against_ground_truth(viterbi_path, true_states_so_far)
        forward_acc = traffic_core.accuracy_against_ground_truth(forward_map_path, true_states_so_far)
        naive_acc = traffic_core.accuracy_against_ground_truth(observations_so_far, true_states_so_far)

        eval_cols = st.columns(4)
        eval_cols[0].metric("Ground Truth (step)", true_states_so_far[-1])
        eval_cols[1].metric(
            "Viterbi Accuracy",
            f"{viterbi_acc * 100:.1f}%",
            delta=f"{(viterbi_acc - naive_acc) * 100:+.1f}% vs naive",
        )
        eval_cols[2].metric(
            "Forward MAP Accuracy",
            f"{forward_acc * 100:.1f}%",
            delta=f"{(forward_acc - naive_acc) * 100:+.1f}% vs naive",
        )
        eval_cols[3].metric("Naive Observation Accuracy", f"{naive_acc * 100:.1f}%")

    with st.expander("Full synthetic sequence data", expanded=False):
        st.dataframe(sequence, use_container_width=True)

    if st.session_state["is_playing"]:
        if step < max_step:
            st.session_state["advance_next"] = True
            time.sleep(step_delay)
            st.rerun()
        else:
            st.session_state["is_playing"] = False
            st.rerun()


if __name__ == "__main__":
    main()
