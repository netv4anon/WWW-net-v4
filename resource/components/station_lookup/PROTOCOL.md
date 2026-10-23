# Fixed station-history retrieval after a network-support exclusion

Written before this follow-on's predictive outcome inspection. The release,
TargetSupport mask, structural matches and earlier model results have already
been inspected. This is a development-exposed diagnostic, not a blind test.

Question: Does the already verified zero full-network target-state overlap
under TargetSupport prevent a local, history-only rule from exploiting
remaining traffic sequence reuse? The 30,685 prior witnesses were selected
using targets; they alone do not demonstrate prediction-time retrieval.

No new data, target laws, masks, neural models, optimization fits, random
seeds or threshold searches. Original NET-V4 12-to-12 remains untouched.
This follow-on does not change or rerun the completed fixed ridge experiment.

Use the exact existing Contra_Costa v1 source and two existing training views:
Original (3,909 rows) and TargetSupport (2,665). Query all 839 test rows at
all 496 fixed node positions, forecasting 12 flow steps from 12 history steps.
Do not match across node positions.

One fixed, conventional lookup rule, matching the prior full-network lookup:
- An exact history hit returns the coordinatewise arithmetic mean of ALL
  matching training futures, including repeated rows with their multiplicity.
- A miss returns the last observed flow at all twelve horizons.
- No clipping or missing-value masking.
- Primary key: raw float32 12-step flow history, same station position.
  This matches the information unit of the existing stationwise AR-Ridge.
- Prespecified sensitivity key: the same flow plus the 12 historical
  time-of-day/weekday channel pairs. No future calendar enters the key.
Two keys times two training views = four new prediction configurations;
the unchanged LastValue reference is reused, not counted as a new rule.

Inputs to prediction are training histories, training futures, query histories
and optional historical calendars. Save and hash predictions, hit membership,
number of matching training rows, distinct training-future count and the first
training representative BEFORE scoring query futures. The inherited mask
itself uses test targets and prior outcomes are exposed; the separation of
prediction/scoring cannot undo this. Hashing a source file is not model access
to its test labels.

Scoring uses inverse-normalized flow with the pinned published statistics,
MAE and RMSE over fixed 12-step cells. Report all cells and varying-target
cells separately, as well as near-zero and other constants; 0.001 is ONLY
the previous inverse-flow constant-zero classification, never a match rule.
Report hit/miss and train-future-unique/conflicting hit groups, plus the
previous 30,685 residual witnesses. Counts based on query targets are
descriptive analysis only, never prediction-time features.

Primary use gate (TargetSupport, flow-only key):
1. MAE lower than LastValue on all 416,144 cells.
2. MAE lower on all 341,680 varying-target cells.
3. A strictly positive MAE gain on varying targets at more than half of
   stations with at least one varying-target history hit.
Use 1e-10 only for numerical zero in gain reporting. These are a fixed local
research gate, not statistical significance or venue acceptance criteria.
Also count raw-exact recovery among unique-training-future hits; do not
conflate presence of a matching future with unique recoverability.

Retain all four configurations and every adverse/ambiguous result. No further
lookup variants, site selection or model additions after seeing outcomes.

Independently verify with sorted fixed-width records, original source bytes,
and separately reconstructed scores, not the primary dictionary routine.
Check every prediction and hit/conflict count, and unchanged misses.

If the primary gate passes, integrate only the qualified empirical consequence:
network-level support exclusion did not remove an exploitable local shortcut
for this rule/source/view. If it fails, preserve the negative result; do not
promote structural reuse to predictive benefit. In neither case establish
historical IGSTGNN inflation, real-date recovery, chronological repair,
physical source identity, a new algorithm or high-probability WWW acceptance.
