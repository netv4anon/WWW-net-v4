# Construction rationale and evidence status

The purpose is to make a traffic-history response claim inspectable while holding
the forecast and scoring population fixed. This is a controlled diagnostic, not
an estimated law of how real emergency incidents alter traffic volume. The
original NET-V4 12-to-12 snapshots and all old results remain intact.

| Choice | Why the diagnostic needs it | Actual evidence status |
|---|---|---|
| Retained weekly volume profiles | Repeated temporal context across prescribed calendar positions | 102 retained distinct numeric profiles regenerate MV; complete historical raw lineage remains incomplete |
| Weekday/quarter-hour averaging and gap processing | Obtain finite periodic background profiles | Pinned builder documents averaging, interpolation, fallback means, smoothing and clipping; these are processed profiles |
| AR(1) residuals | Correlated rather than independently perturbed history values | Legacy implementation retained; phi .92, noise .025 and smoothing sigma 2 are prescribed, not empirically fitted here |
| Pulse/severity amplitudes | Controllable perturbation strength with shared draws in paired views | Exact ranges [.06,.15], [.14,.25], [.22,.36], [.32,.48], [.42,.58] are construction settings; not measured severity-to-volume effects |
| Dispatch multiplier, duration and attenuation | Reproducible variation of the specified pulse | FDNY provides no measured road response or trajectories; degree decay .015 and these mappings have no calibration claim |
| Exponential recovery and clipping | Define explicit futures after the marked interval | Legacy and residual-preserving alternatives are hypothetical laws; the second tests target sensitivity, not physical truth |
| Four pulse timings | Inspect event visibility relative to the prediction origin and avoid a single timing judgment | Prescribed coverage across 24 steps; no claim that the four groups estimate real event frequencies |
| B/P/R/F paired views | Isolate background, pulse and recovery operations with shared context | Finite construction contrasts; loss accounting is established arithmetic, not a new causal estimator |
| C/N functional views | Ask whether other-sequence response is helpful conditional on changed center inputs | Group ablations for joint predictors; not physically realizable interventions or unique attributions |
| Similarity-selected other profiles | Supply reproducible complete inputs for joint models | Numeric profile similarity only; not seven verified adjacent roads or a propagation model |
| Center duplicate | Preserve the actual original model input convention | Compatibility slot, not an additional independent road |
| Two source variants | Retain what the original task actually supplied | Constructed populations differ in more than one factor; no source-factor causal comparison |
| History proximity rule | Test persistence under one declared relation to original training inputs | Validation-calibrated nearest-history distance, not equal distributions, physical validation or calibrated targets |

Use references at the level they support. CheckList and Context is Key motivate
behavioral/contextual evaluation and explicit supplied targets. Traffic resources
motivate the application. Cao et al. study incident-related **speed reduction**
with matched controls; they do not establish our exact **volume** pulse law,
amplitude intervals, dispatch factor or recovery coefficient. No extra citation
is attached to a numerical parameter merely to imply it was calibrated.

The design can show that an aggregate score does not logically certify helpful
history response, and show the concrete consequence for these saved predictors.
It cannot estimate the frequency or magnitude of this failure on real roads.
The observed reference remains a separate descriptive study; source tracing and
external reuse witnesses retain their own identities and limitations.
