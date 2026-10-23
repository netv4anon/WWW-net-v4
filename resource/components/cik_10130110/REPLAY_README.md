# Fixed second-sensor construction replay

Local research companion, not a complete CiK reproduction or an uploaded submission.
Primary result files are retained under primary_evidence; this package's probe directory
contains only sensor 10130110. Both use eight identical rules and seeds 0..31.

Tested runtime: Python 3.8.10, NumPy 1.24.4, pandas 2.0.3.
From an extracted directory, execute in order:

```text
python -X utf8 construct_probe.py
python -X utf8 predict_probe.py
python -X utf8 score_probe.py
python -X utf8 verify_probe.py
python -X utf8 score_user_forecast.py --predictions probe/USER_FORECAST_EXAMPLE.npz --out replay_user_scores.json
```

The independent verifier reconstructs source selection from the raw complete prefix,
then reconstructs every event, target, forecast and score. prepare_probe.py records
original preparation and is not needed for this saved-series replay. Its sanitized
HTTP header rendering is a provenance display, not an experiment input.
PRE_SCORE_SEAL.json records the original pre-loss files; do not regenerate that seal
during replay. The original TARGET_PROBE_PROTOCOL.md describes the first station;
NEXT_FIXED_REPLICATION_PROTOCOL.md explicitly supersedes only station selection and
the mean-profile success question for the sequential second check.

To score another point predictor, supply an NPZ with integer seed [32] and finite
real prediction [32,24]. Inputs are probe/INPUTS.npz and SCENARIOS.json; do not use
TARGETS.npz or background futures to generate predictions. The scorer realigns IDs.
Results are separately reported by station and target law, without a pooled score.

The original 12-to-12 NET-V4 data remain a distinct component. Source attributions,
fixed source snapshots, original code license and all limits accompany this package.
A later CiK source commit is not asserted to be the historical published implementation.
No new license for TrafficFresh observations is asserted.
