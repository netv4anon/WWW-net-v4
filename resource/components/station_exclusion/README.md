# NET-V4 station-input exclusion companion

This companion reproduces four fixed station-input exclusion conditions.
It performs no fitting, forecasting, future-target decoding or score selection.
All 839 query rows and 496 node positions remain represented.

Tested with Python 3.8.10 and NumPy 1.24.4. The raw source is not redistributed.
Use the exact IGSTGNN Contra_Costa v1 source identified in SOURCE.json:
865971481 bytes, SHA256 bc6c5608457f88864625d683f9a66d17bc0fc31fe5acba366b6033708a099279.
No pickle execution is required; only bound historical array offsets are decoded.

From this directory:
    python audit_exclusion.py --source /path/to/incident_all.npy --out replay_result
    python verify_replay.py --out replay_result

The output directory must not exist before execution. Full arrays, earliest
removal witnesses, memberships and per-node statistics are compared with expected/.
START and OUTPUT_SEAL contain run-specific provenance. Their timestamps are not
expected to equal the original execution; their declared file digests are checked.
The primary result retains its original pending-verification status as a historical
record. VERIFICATION.json records the completed independent audit.

TargetSupport was constructed using query targets before this audit. The new
input-only rule does not remove that development exposure. Zero declared-key
overlap is guaranteed by construction, not a new algorithm or proof of chronology.
Constant-flow collisions alone do not identify physical source reuse.
An empty station view prevents independently fitting that station from this view,
not every possible shared model or alternative protocol.
Nonempty views and row-count bins do not demonstrate statistical adequacy.

CALENDAR_INTERSECTION_CHECK.json and CALENDAR_VERIFICATION.json describe an
independently checked post-result explanation, not a prespecified new policy.
They are supplied as evidence; the portable command replays the four fixed primary
conditions only. This package is a companion, not the full NET-V4 submission.
