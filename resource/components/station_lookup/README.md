# Station-history retrieval after full-network exclusion

This companion extends NET-V4's WWW revision with a fixed audit on one
IGSTGNN Contra_Costa v1 release. It is not a new forecasting model, a
chronological repair, or an estimate of published neural-model inflation.

Requires Python 3.8+ and NumPy (verified locally with NumPy 1.24.4).
No Torch, model training, network access, or pickle execution is required.

Run from this directory:

    python replay_station_lookup.py --source "PATH/TO/incident_all.npy" --out "NEW/OUTPUT/DIRECTORY"

The source must have exactly 865,971,481 bytes and SHA256
bc6c5608457f88864625d683f9a66d17bc0fc31fe5acba366b6033708a099279.
It is the authors' Kaggle data4igstgnn v1 Contra_Costa release; obtain it
under upstream terms. The source is not included. The script verifies its
digest before using the saved numeric array offsets. The existing
TargetSupport mask is fixed and target-derived; no claim of prospective
evaluation follows from its use.

The two keys are the same station's 12 flow values, and those values plus
12 historical calendar pairs. Never use future calendars to form keys.
A history hit returns the mean of all matching training futures; a miss
uses LastValue. Original and TargetSupport views give four fixed
configurations. Predictions are saved before query-future scoring.
primary_reference.py is the sealed research implementation; run the wrapper
above, not that file's original project-specific command entry.

Output includes four prediction arrays, match/conflict counts, group scores
and a verification against expected prediction bytes and aggregate scores.
Approximately 200 MB of output and under 1 GB RAM are expected.
All/variable/constant and hit/miss/ambiguous groups are retained.
No approximate equality or cross-station search is used.

This is a development-exposed diagnostic. The same source, cohort and
TargetSupport mask were inspected in previous stages. Same-machine replay
does not certify installation in a fresh dependency environment.
