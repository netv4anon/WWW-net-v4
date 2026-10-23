# Bounded external target-construction probe

Written after reading the selected source operation and before requesting traffic
values or evaluating any predictor. This is not a blind confirmatory study.

Question: In the fixed CiK DecreaseInTrafficInPredictionTask implementation, does
replacing its constant event plateau with pointwise proportional scaling change
same-input forecast comparisons? Both constructions are hypothetical. Neither is
asserted to be real incident truth or the author's intended meaning of 'usual'.
This does not test whether CiK is incorrect or reproduce its published scores.

Bind CiK Git commit 73f46016f8c5643bf6799ca31875eab3e8d0d075 and the exact source
blobs already selected by relevance to traffic. Bind TrafficFresh source commit
72f9682ad1ce00006090e61adedff5898ec3f963 before reading values. Acquire at most
1 MiB from the beginning of traffic_merged_sensor_data.csv, requiring range 206
and the 304418602-byte full-object size. Trim only an incomplete trailing record.
The repository identifies the full LFS object as SHA256
6fb160963102d271993349de2bad9c339ed68925b1965246892eea6c129a9f2b;
we cannot independently check a full-object hash from a prefix.

Select the smallest sensor ID with a run of at least 193 hourly, finite,
nonnegative occupancy values in this prefix, using its earliest such run. This
selection precedes losses. If no such run exists, stop without broadening source
acquisition. Do not select regions/stations by forecast outcomes.

Execute the original random_instance method and original random-window helper by
AST extraction, preserving executable function bodies. Supply only the selected
series through a clearly disclosed local data-loader substitute; thus this is
reference-operation execution with a constrained sampler, not full CiK execution.
Use numpy.RandomState seeds 0..31 once. Require 168 hourly history and 24 future
positions; preserve the original 1..7 hour duration, 0.1..0.5 magnitude selection,
top-five start candidate rule and rendered scenario. Record failed instances and
do not replace seeds. Disclose the 168-to-24 external task as distinct from the
original NET-V4 12-to-12 task, which is unchanged.

Save each unmodified background future before executing its event operation.
Verify the exact reference target against an independent scalar construction.
For each fixed event interval [s,s+d), compare original A_t=r*v_s with a single
alternative S_t=r*v_t; outside the interval both equal v_t. Keep histories and
scenario inputs identical. This is target-law sensitivity, not causal inference.

Freeze nine deterministic rules: three base predictions (last value, previous
24 hours, mean of the seven preceding daily profiles) crossed with three event
responses (no response, plateau r*q_s, pointwise r*q_t inside the supplied event
interval). Predictor inputs are history and the synthetic scenario's start,
duration and percentage; no background future/target values. These interpretable
rules are diagnostic references, not learned models or replicas of CiK's LLMs.
Save predictions before scoring either target. If constant rules coincide, count
actual distinct arrays separately from declared configurations.

Report all nine rules under both targets: full-horizon and event-region MAE;
within each base family report the paired response-rule comparison and whether
its aggregate sign changes across target laws. Also count target variance and
pointwise construction differences, including whether a plateau exceeds the
unmodified future at any event position. The latter is descriptive, not proof
that the original stated task is violated. No significance claim from these 32
dependent constructions or single source station. Preserve every negative result.

Independent stdlib/scalar verification must check source binding, reference
operation equality, fixed histories, forecast inputs and all score rows. A
successful use case requires exact reference reproduction and an aggregate
comparison changing across the two laws; otherwise retain only the structural
finding and do not strengthen the manuscript's utility claim. In neither case
claim an effect on the published CiK rankings or solved WWW competitiveness.
No model fitting, neural search, APIs to LLMs, new months, external messages or
submission. Acquisition and 9 diagnostic prediction configurations are logged
separately from existing 100 fit/calibration configurations and 67/87 prediction
counters. Further data expansion requires new scientific evidence, not a failed
or inconvenient score.
