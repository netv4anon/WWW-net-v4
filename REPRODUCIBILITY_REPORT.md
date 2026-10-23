# Reproducibility and technical support report

The packaged offline replay completed all seven checks successfully. This is
validation of the supplied resource, retained-profile construction and saved
forecast calculations; it is not a new experiment or a claim of raw-source
completeness. The final manuscript governs the scientific claims.

## Executed checks

| Check | Result | Scope |
|---|---|---|
| Full package inventory | PASS | SHA-256 and file sizes |
| Resource inventory | PASS | Component distribution bytes |
| Center task | PASS | Retained profiles, matched construction and saved scores |
| Multisequence task | PASS | Complete original task regeneration and saved-model scores |
| Basel confirmation | PASS | Admission, 11,953,896 exact array elements, 24 principal effects and six checkpoint bindings |
| Representation follow-up | PASS | 16,702,704 exact array elements, 12 decompositions and 48 crossed responses |
| Unrounded/configuration evidence | PASS | Saved losses and actual model/scaler/log configuration checks |

Basel remains **7 of 12** conjunctions retained, with five conditional-response
reversals. All twelve total responses remain positive. Passing replay does not
mean that the original all-twelve scientific hypothesis passed. The maximum
follow-up scoring discrepancy was approximately 2.89e-15. The post-confirmation
hourly-representation analysis is not an independent replacement confirmation.

The release retains both original 33,076-sequence snapshots. The original archive
contains `data/{packaged,working}/{history,targets,splits,analysis_metadata}.npz`.
It also retains the constructed center and multisequence tasks, Basel profiles,
all frozen model weights and forecasts, adverse results, matching witnesses,
training-support masks, probe failures and source contracts.

All 814 retained files outside the declared documentation/inventory changes
matched the verified anonymous resource byte for byte. This includes the
datasets, task arrays, targets, weights, saved predictions and historical seals.
All 107 unpacked Python files passed syntax parsing. The new `reproduce.py`
entry point was actually executed; it writes outputs outside the resource.
No retraining, model selection or new neural inference was performed.

## Environment and repeat commands

Validation used Python 3.8.10, NumPy 1.24.4 and pandas 2.0.3. The optional
configuration inspection used PyTorch 2.4.1+cu118; the default task/score replay
requires no GPU. See `requirements-replay.txt` for minimal requirements and
`requirements-validation-recorded.txt` for the recorded versions.

```text
python -B reproduce.py --out ../netv4_replay_results
```

See `REPRODUCIBILITY_RESULTS.json` for the machine-readable summary, and
`PAPER_EVIDENCE_MAP.md` for the paper-to-file map. The detailed scientific report
is `technical_companion/NET_V4_technical_companion.pdf`; its numbering is local.
The manuscript and its typesetting source are not distributed in this
repository. The technical report documents the supporting evidence.

## Limits and source access

The exact IGSTGNN raw input is not redistributed. Its download source, size and
SHA-256 are in the README; acquire it separately to rerun the raw-source match
search. Matching witnesses and saved audit outputs are included. The historical
NYC raw-source snapshot has not been recovered, so complete regeneration from
that source is unsupported. Task regeneration from retained profiles and
saved-prediction recomputation are separate, supported operations.

Synthetic events are not validated real incident dynamics. Basel's within-hour
values are assumed uniform quarter-hour allocations, not observed quarter-hour
counts. Profiles and constructed cases are dependent. Upstream attribution and
licenses remain separate; the bundle adds no blanket source-data license.

Some frozen README files describe the historical source handoff and retain that
wording for hash verification. The current package-level README and this report
describe the anonymous review distribution. No historical placeholder DOI is a
current deposit address. No online upload or publication is asserted here.
