# NET-V4-MV: multisequence response diagnostic

This separately versioned extension retains the original NET-V4 12-to-12 task. It provides 11,424 constructed scenarios, 2,856 stochastic backgrounds and six input views per scenario. Nine numeric slots encode a center, its duplicate and seven other profile-conditioned sequences. Profile similarity supplies the seven sequences; their adjacency and physical event responses are **not observed**. Models requiring global road identities or a measured graph remain outside this contract.

The 102 conditioning profiles are retained numeric inputs from the earlier construction companion. `pins`, `PINS.json`, `construct.py` and the protocol identify every input and setting needed to reconstruct the new arrays. Exact conditional regeneration does not recover the missing raw-source lineage of the original snapshots or turn these profiles into contemporaneous observations.

The reference week is a synthetic calendar coordinate. All scenarios use 12 historical and 12 future nominal 15-minute steps. Four fixed timings cover history-completed, cutoff-spanning, future-early and future-late pulses. Each background and amplitude draw is shared across timings. B/P/R/F remain the original construction arms. C changes only the center history and its copy; N changes only the other seven histories. C/N are functional prediction views scored on the common full-arm target, not physically validated interventions. Full-arm recovery is also scored under one fixed alternative target law, without regenerating forecasts.

## Use a coupled predictor

Python 3.8+, NumPy 1.20+ for construction and scoring. PyTorch is needed only to execute the bundled frozen STID models.

```text
python -B predict.py --inputs task/inputs.npz --adapter my_model.py:predict --out my_forecasts.npz
python -B score.py --predictions my_forecasts.npz --out my_scores
```

The adapter receives only `history[N,12,9]` and returns finite `prediction[N,12]`. Input IDs are shuffled opaque transport keys, excluded from the adapter call. IDs, targets, scenario mappings, phase masks and settings belong to the evaluator. This is an interface boundary, not a security sandbox or an independent blind test. Reader submission accepts all 68,544 rows exactly once, in any order. No evaluation-time fitting or tuning is implied.

## Inspect and reproduce

```text
python -B construct.py --verify task
python -B verify.py --out my_verification.json
python -B score.py --out replayed_scores
```

`responses.csv` reports fixed-target total response and conditional center/other-sequence responses for every model, target law, timing and phase. Positive values mean lower error under the declared comparison. `accuracy.csv` contains B/P/R/F diagonal scores. Empty phases are explicit. `profile_response.csv` retains all 102 profile-group results, including harm; scenarios sharing profiles and backgrounds are dependent. These are finite conditional losses, not confidence intervals or incident causal effects.

All thirteen forecast configurations are frozen: three simple rules and two original-training variants of center Ridge, joint-input Ridge, and three STID seeds. Saved weights and their scalers are hash-bound. `predictions/SEAL.json` precedes scoring. The generic scorer certifies neither a submitted model's determinism nor its exclusive use of history.

The protocol's main empirical check asks whether a joint-input model's neighbor response changes loss while center history and target stay fixed. It does not require favorable neighbors or establish a new ablation theory. Reproducibility, horizon support and interface coverage are material improvements, but do not by themselves establish WWW acceptance readiness. No new observed traffic records, real-road graph or calibrated incident mechanism is claimed.
