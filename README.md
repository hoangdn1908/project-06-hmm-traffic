# Project 06 — Campus Crowd State Estimator

## 1. Problem Description

Temporal probabilistic reasoning in Artificial Intelligence (AI) addresses state estimation in dynamic systems where the true underlying physical condition cannot be observed directly. A Hidden Markov Model (HMM) formalizes this paradigm by assuming a system satisfies the First-Order Markov Property—wherein the probability distribution of the future hidden state depends exclusively on the current hidden state—coupled with a noisy observation emission process.

This project focuses on estimating pedestrian congestion states across campus corridors. At any discrete time step $t$, the true crowd condition exists in one of three unobserved categorical states:
$$\mathcal{S} = \{\text{LOW}, \text{MEDIUM}, \text{HIGH}\}$$

Due to environmental occlusions and sensory jitter, deployed sensors yield imperfect, noisy discrete observations drawn from the same alphabet:
$$\mathcal{O} = \{\text{LOW}, \text{MEDIUM}, \text{HIGH}\}$$

Given a temporal sequence of noisy sensor readings $\mathbf{y}_{1:T}$, the system must filter instantaneous state probabilities and reconstruct the single most likely hidden state trajectory over time. All sensor traces are synthetic benchmark datasets generated for pedagogical evaluation.

---

## 2. Provided Materials & Starter Resources

The project package provides canonical transition and emission configurations, synthetic multi-noise benchmark sequences, sequence generation scripts, and an interactive Streamlit application.

### File Structure
```text
project-06-hmm-traffic/
├── data/
│   ├── README.md                          # Data dictionary and noise model specifications
│   ├── hmm_config.json                    # Hidden states, alphabet, priors, and transition matrices
│   ├── scenarios.json                     # Benchmark scenario descriptions (P06-STABLE-LOW to P06-MISLEADING-OBSERVATIONS)
│   ├── sequence_len200_noise05_seed42.csv # 200-step trajectory with 5% sensor noise
│   ├── sequence_len200_noise15_seed42.csv # 200-step trajectory with 15% sensor noise
│   ├── sequence_len200_noise30_seed42.csv # 200-step trajectory with 30% sensor noise
│   └── SHA256SUMS                         # Cryptographic checksums ensuring data integrity
├── scripts/
│   └── generate_sequence.py               # Configurable sequence generator with noise injection
├── starter/
│   ├── loader.py                          # Parser for HMM configurations and sequence datasets
│   ├── traffic_core.py                    # Algorithmic stubs for Forward and Viterbi inference
│   └── app.py                            # Streamlit interactive playback dashboard
├── tests/
│   └── test_project_06_sanity.py          # Unit tests verifying stochastic matrix validity and APIs
└── requirements.txt                       # Pinned Python package dependencies (including streamlit)
```

### Starter Infrastructure vs. Student Implementation
The module `starter/loader.py` loads and verifies stochastic consistency (all transition and emission rows sum to 1.0). The module `starter/app.py` renders the visual playback timeline. Students implement the probabilistic core in `starter/traffic_core.py`, specifically: the Forward algorithm (`forward`), the Viterbi dynamic programming algorithm (`viterbi`), and sequence evaluation metrics (`accuracy_against_ground_truth`). Third-party HMM libraries (e.g., `hmmlearn`) and Large Language Model (LLM) APIs are strictly prohibited.

### Installation and Execution
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run starter/app.py
```

---

## 3. Requirements & Deliverables

### 3.1 Midterm Milestone — AI Core
- **Algorithmic Implementation**: Implement exact inference over discrete Hidden Markov Models from first principles in `starter/traffic_core.py`. Implement `forward(observations, hmm_config)` to compute normalized state-occupancy probabilities $\mathbf{P}(X_t \mid \mathbf{y}_{1:t})$ for each time step $t \in [1, T]$. Implement `viterbi(observations, hmm_config)` using dynamic programming with backpointers to identify the globally optimal joint hidden state trajectory $\arg\max_{\mathbf{x}_{1:T}} P(\mathbf{x}_{1:T}, \mathbf{y}_{1:T})$ and its joint log-likelihood.
- **Mandatory Controlled Experiment**: Execute empirical decoding evaluations across all three committed noise regimes (5%, 15%, and 30% sensor error) using `accuracy_against_ground_truth`. Measure decoding accuracy against synthetic ground truth, quantify degradation rates under escalating sensory noise, and critically discuss whether the first-order Markov assumption holds under realistic high-noise conditions.
- **Deliverables**: Fully implemented `traffic_core.py`, experimental evaluation scripts/notebooks, and an oral presentation slide deck (`presentation.pdf`).
- **Grading Criteria**: Mathematical formalization and inference correctness (15 pts); noise experiment depth and critical assumptions analysis (10 pts); oral defense and live sequence decoding demo (15 pts). Total: 40 points (40% weight).

### 3.2 Final Milestone — AI Product
- **Interactive Application**: Package the HMM estimator into a real-time playback dashboard using Streamlit. The application must feature: interactive play/pause controls over sensor streams; real-time confidence bar charts showing posterior state distributions; dynamic temporal plots tracking belief shifts; an "Evaluation Mode" toggle overlaying ground truth against decoded estimates; and a persistent disclaimer regarding synthetic data.
- **Stress & Edge Scenarios**: Demonstrate decoding resilience across all four benchmark fixtures in `data/scenarios.json`: stationary low traffic (`P06-STABLE-LOW`), rapid transient spikes (`P06-RAPID-TRANSITION`), high-noise sensor regimes (`P06-HIGH-NOISE`), and adversarial observations contradictory to the hidden state (`P06-MISLEADING-OBSERVATIONS`).
- **Deliverables**: Functional Streamlit application, clean GitHub repository, and an IEEE-formatted engineering technical report (`report.pdf`).
- **Grading Criteria**: Product user interface ergonomics, animation smoothness, and usability (25 pts); technical report mathematical derivations and error analysis (15 pts); oral defense against complex noise-distorted traces (10 pts). Total: 50 points (50% weight).

---

## 4. References

- [1] L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," *Proceedings of the IEEE*, vol. 77, no. 2, pp. 257–286, 1989, doi: 10.1109/5.18626.
- [2] A. Viterbi, "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm," *IEEE Transactions on Information Theory*, vol. 13, no. 2, pp. 260–269, 1967, doi: 10.1109/TIT.1967.1054010.
- [3] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ, USA: Pearson, 2020, pp. 468–507.
