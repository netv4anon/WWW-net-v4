# NET-V4 external construction-diagnostic companion

This is a constrained-source use check of a pinned Context is Key traffic task.
It is not the full CiK benchmark or a reproduction of its published model scores.
Original NET-V4 remains 12-to-12; this component has 168 hourly history values and
24 future values. Do not pool their scores.

Environment used: Python 3.8.10, NumPy 1.24.4, pandas 2.0.3. The reference operation
uses Series integer indexing semantics of that pandas version. Original source
bodies are extracted through AST; only data loading is substituted. There are no
network requests in the replay. See SOURCE_ATTRIBUTION.md and upstream LICENSE.

From this extracted directory:

    python -X utf8 construct_probe.py
    python -X utf8 predict_probe.py
    python -X utf8 score_probe.py
    python -X utf8 verify_probe.py

Saved-result verification needs NumPy plus the standard library; pandas is only
needed to replay reference construction. The verifier independently reconstructs
source selection and all values with scalar operations.

To evaluate your own deterministic point predictor, read probe/INPUTS.npz (history,
event_start, event_duration, event_ratio, seed) and optionally probe/SCENARIOS.json.
Do not read probe/TARGETS.npz while predicting. Save an NPZ with integer seed [32]
and finite prediction [32,24]. Order may vary, but each fixed seed must occur once.
Then the offline evaluator can access both target laws:

    python -X utf8 score_user_forecast.py --predictions my_forecast.npz --out my_scores.json

The included USER_FORECAST_EXAMPLE.npz has reversed IDs and reproduces the
mean-profile anchored response. Both event-region weighting conventions are
reported; the primary measure weights instances equally. Original and alternative
targets are hypothetical, with no asserted physical ground truth. Predictions
are identical across target laws. Full manifests and code/input seals are included.
