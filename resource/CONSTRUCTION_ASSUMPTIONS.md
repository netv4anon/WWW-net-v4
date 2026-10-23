# Construction assumptions and what can be traced

Two NET-V4 snapshots are retained as numerical evaluation objects, not verified
measurements of traffic responding to fire dispatches. The generated snapshot
has 20,000 generated events and 13,076 generated controls. The mixed snapshot has
20,000 generated events, 12,608 processed observation-tagged controls and 468
bootstrap controls. Their arrays and partitions differ. Observation-tagged does
not mean untouched measurements. Exact identities and file hashes are in the
stage's `RESOURCE_IDENTITY_AND_CONTRACT.json`.

## Pinned code assumptions

`CONSTRUCTION_SETTINGS.json` binds settings to the legacy source SHA256.
`source_excerpts/legacy_construction_functions.txt` retains its profile,
AR(1) and perturbation functions as documentation. That source digest agrees
with the perturbation provenance in the original archived candidate. These
excerpts are inspected code, not a rerun proving each row's historic lineage.

Profiles average volume by weekday and quarter-hour over the supplied source
pool. Missing slots are interpolated; remaining gaps use segment/global means.
Each day's profile is Gaussian-smoothed (sigma 2, wrap mode) and clipped below
at 5. Generated background follows the profile plus AR(1) residuals (phi .92).
The function's noise-scale default is .03, but event and generated-control
call sites use .025. A modeled year drift multiplies the profile by
`1+(year-2020)*Uniform(-.008,.008)`; background is clipped below at 3. These
parameters prescribe construction; they are not measured incident parameters.

Severity levels 1–5 draw amplitudes from [.06,.15], [.14,.25], [.22,.36],
[.32,.48], [.42,.58]; level 0 uses zero. The fallback severity range is [.08,.18].
Positive dispatch seconds multiply amplitude by
`clip(1+(300-dispatch_seconds)/600,.7,1.5)`; otherwise the factor is 1.
Spatial attenuation is `exp(-distance_degrees/.015)`, not a fitted propagation
law. A .6 parabolic/.4 Laplace envelope subtracts the modeled drop from
background during the declared pulse. The builder derives duration/start using
response seconds and a small random shift (exact rules in the settings file).

For first recovery index e, final unclipped pulse value a, background v and
q(t)=exp(-(t-e)/tau), the legacy recovery is
`v(t)+(a-v(t))*q(t)`, tau=max(2,response_seconds/900). The fixed alternative
uses `v(t)+(a-v(e-1))*q(t)`. Both apply the same [3,1.3*max(v)] clipping.
The alternative preserves background variation before clipping. Neither law
is physical ground truth. The mask marks only the pulse, not all processed
recovery points. The paired 256-context task preserves the old draws/parameters;
recovery starts at 12,13,14, so its replacement changes targets, not histories.
Eleven saved configurations can be re-scored without inference.

## Recovery boundary

| Object | Retained evidence | Remaining limit |
|---|---|---|
| Original snapshots | Arrays, source tags, indices and hashes | Complete historical raw-response pool not recovered |
| Perturbation mechanism | Verbatim original archived function, code checks, pinned documentation | Commented realism/calibration claims unverified |
| Paired scoring | Shuffled center histories, opaque IDs, separate targets/masks, saved forecasts | Dependent/exposed contexts; no paired spatial neighbors |
| Alternative recovery | Frozen targets and sealed protocol | One hypothetical sensitivity check, not model selection |
| Observation-tagged controls | Processed arrays and population count | Interpolation/clipping/reconstructed dates; no complete pointwise processing lineage |
| Observation reference | Pinned case/control origins and fixed forecasts | Descriptive comparison, no validation of synthetic incidents |

Full raw-source-to-snapshot regeneration is incomplete. Do not interpret
successfully running scores, retained function defaults, or present-day source
queries as recovering all historical observations, processing decisions or
physical effects. FDNY dispatch data provides no EMS trajectory, vehicle arrival,
lane occupation or observed road response. Original notice files remain in the
unchanged archive; this documentation assigns no new upstream license.

The unchanged finite paired-construction companion is now included in
`archives/NET_V4_paired_diagnostic_companion.zip`. It retains profiles,
all 256 context parameters, fixed seed, original functions and four-arm
float32 objects. See the reader guide for extraction/replay. This
supplies finite diagnostic regeneration, not the missing historical
raw-source pool for the complete snapshots.
