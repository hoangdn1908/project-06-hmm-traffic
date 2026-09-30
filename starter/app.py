"""Streamlit starter for Project 06. Plays back observations; does NOT estimate hidden states.

The graded AI core (`forward`, `viterbi`) lives in `traffic_core.py` and is called from here, but
this file must not implement inference itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter import loader, traffic_core  # noqa: E402

st.set_page_config(page_title="Campus Crowd State Estimator — starter", layout="wide")

st.title("Campus Crowd State Estimator — starter")
st.warning(
    "⚠️ " + loader.SYNTHETIC_DATA_NOTICE,
    icon="⚠️",
)

hmm_config = loader.load_hmm_config()
scenarios = loader.load_scenarios()

with st.sidebar:
    st.header("Noise control")
    noise_level = st.selectbox(
        "Committed noise level",
        [0.05, 0.15, 0.30],
        index=1,
        format_func=lambda value: f"{int(value * 100)}%",
    )

    st.divider()
    st.header("Load a scenario")
    scenario_id = st.selectbox(
        "Scenario",
        ["(none)"] + [scenario["id"] for scenario in scenarios],
    )
    if scenario_id != "(none)":
        chosen = next(s for s in scenarios if s["id"] == scenario_id)
        st.caption(chosen["description"])
        st.json(chosen["generator"])

    st.divider()
    evaluation_mode = st.checkbox("Evaluation mode (show synthetic ground truth)", value=False)

sequence = loader.load_sequence(noise_level)
max_step = len(sequence) - 1
step = st.slider("Playback step", min_value=0, max_value=max_step, value=0)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Observation stream so far")
    observations_so_far = [row["observation"] for row in sequence[: step + 1]]
    st.line_chart(
        {"observation": [hmm_config["observation_symbols"].index(obs) for obs in observations_so_far]}
    )
    st.caption("Y-axis is the index into observation_symbols: " + str(hmm_config["observation_symbols"]))
    st.write("Current observation:", sequence[step]["observation"])

with col_right:
    st.subheader("Estimated hidden state")
    try:
        path, probability = traffic_core.viterbi(observations_so_far, hmm_config)
        st.write("Estimated path:", path)
        st.write("Path probability:", probability)
    except NotImplementedError as exc:
        st.error(f"Not implemented yet: {exc}\nImplement `viterbi`/`forward` in `starter/traffic_core.py`.")

    if evaluation_mode:
        st.caption("Synthetic ground-truth hidden state (evaluation mode only):")
        st.write(sequence[step]["hidden_state"])

st.divider()
st.subheader("Full committed sequence (synthetic)")
st.dataframe(sequence)
