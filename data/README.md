# Project 06 data dictionary

**All files in this folder are synthetic educational data.** They describe a fictional Hidden
Markov Model simulation, not a real traffic feed or a real queueing system.

## Files

| File | Contents |
|---|---|
| `hmm_config.json` | Hidden states, observation symbols, initial distribution, transition matrix, base emission matrix, and the noise-blending rule. Every transition/emission row sums to 1.0. |
| `scenarios.json` | Four demo scenarios (stable low, rapid transition, high-noise, misleading observations) plus the required noise-level experiment. |
| `sequence_len200_noise05_seed42.csv` | 200-step sequence, seed 42, noise level 5%. |
| `sequence_len200_noise15_seed42.csv` | 200-step sequence, seed 42, noise level 15%. |
| `sequence_len200_noise30_seed42.csv` | 200-step sequence, seed 42, noise level 30%. |
| `SHA256SUMS` | Checksums of the files above, so regenerated data can be verified byte-for-byte. |

## Columns in `sequence_len200_noise*_seed42.csv`

`step, hidden_state, observation`

- `hidden_state` is the synthetic ground-truth hidden state (`LOW`, `MEDIUM`, `HIGH`) used only
  to generate the sequence and to evaluate your estimator's accuracy. It is not something your
  `forward`/`viterbi` implementation may read as an input.
- `observation` is the noisy sensor reading your estimator consumes.

## Regenerating the data

```bash
python3.11 scripts/generate_sequence.py --length 200 --seed 42 --noise 0.05
python3.11 scripts/generate_sequence.py --length 200 --seed 42 --noise 0.15
python3.11 scripts/generate_sequence.py --length 200 --seed 42 --noise 0.30
```

Each command overwrites the matching committed file with a byte-identical copy (verify with
`shasum -a 256 -c SHA256SUMS` from inside `data/`).

## Verifying checksums

```bash
cd data
shasum -a 256 -c SHA256SUMS
```
