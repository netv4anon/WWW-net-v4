# Fixed audit of remaining reuse at the predictor's input granularity

Written before this audit's outcome inspection on 2026-10-08.
This is a post hoc scope audit, not an independent blind experiment.

The prior TargetSupport mask excludes exact test target states containing all
496 stations and their calendar channels. The fitted AR-Ridge uses one station's
12 flow values at a time. A network-state exclusion is not automatically a
station-sequence exclusion. Test the distinction on the existing source only.

Compare all six already fixed training masks: Original, WholeTraffic,
TargetSupport, Random4101, Random4102, Random4103. No new masks, models, training,
sampling, threshold search, data acquisition, or predictions are allowed.

For each fixed node position separately, define:
1. History reuse: exactly equal raw float32 12-flow history.
2. Paired reuse: exactly equal history AND 12-flow future from one same training row.
3. Calendar-aware paired reuse: paired reuse AND equal released 24-step calendar.
Do not match across different node positions. No absolute physical station/date
identity is assumed from numerical equality.

Report all pairs, then stratify by whether the 12-flow target varies (max>min)
or is constant. Also split constant targets into exactly-zero inverse-flow
(with tolerance 0.001, matching the prior inverse-normalization precision) and
other constant targets. This is a descriptive degeneracy partition, not a new
near-match rule. Every reuse match itself must be complete byte equality.
Do not remove constant nodes or alter the denominator.

Denominator: 839 test queries x 496 node positions = 416144 node-query cells.
Report query and node counts separately. Retain all matching training row
indices for witnesses, per-node counts and six boolean hit maps.
Also determine how many varying-target cells retain calendar-aware paired reuse
under TargetSupport, and the number of such matches in each original
93/466/280 query group.

Re-score only the already frozen Original and TargetSupport predictions:
per-node 12-horizon MAE, comparing remaining calendar-aware varying-target reuse
with its complement. The membership is target-defined and post hoc.
No new predictive configuration or causal interpretation is allowed.

Independently verify membership using record arrays / NumPy equality rather
than the main dictionary-of-byte-strings implementation. Verify every positive
TargetSupport witness directly against original source spans (both x and y).
Results establish the scope of the earlier exclusion contract, not a source
identity, causal memorization effect, prospective split or new algorithm.

Repair the two known unreadable Chinese protocol documents by a NEW English
reconstruction/notice bound to their unchanged hashes and previously sealed
implementation. Do not rewrite the original files or label recovery as an
intact prior human-readable protocol.
