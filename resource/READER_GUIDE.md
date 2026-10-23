# Choose the prediction claim before the score

NET-V4 supports controlled evaluation and inspection of shared traffic releases. Keep its original snapshots, center diagnostic, multisequence diagnostic, observed comparison and external audits separate. The constructed cases are dependent and development-exposed; no result is an independent real-incident effect estimate.

| Reader's question | Supplied cases | Compare |
|---|---|---|
| Did changed history help on the same target? | 256 center contexts, four histories each | Crossed losses and fixed-target response `A` |
| Does a lower-error judgment depend on the recovery target? | The same fixed forecasts and two frozen full-arm targets | `D`, `T`, `A` under each law; complete target-segment bounds |
| Does helpful total response imply helpful other-sequence inputs? | 11,424 MV scenarios, complete nine-slot histories and six input views | `L_CF-L_FF` versus `L_BF-L_FF` |
| Does pooled adverse response hold at every event timing? | Four prescribed timings for each of 2,856 backgrounds | All timing groups, both target laws and structural controls |
| Do distinct incident records imply unseen traffic inputs? | Original and external release witnesses | Input-key hits, hit/miss losses and lookup consequences |
| Does zero overlap after exclusion leave usable support? | Fixed network masks and station-input checks | Remaining local matches and training rows at each station |

The machine-readable [evaluation decision map](EVALUATION_DECISIONS.json) links these six questions to exact saved-evidence paths, commands, score selections and interpretation limits. The original and external reuse outcomes remain separate. The map adds navigation, not a new experiment or evaluation algorithm.

Read the two complete use examples: [response interpretation](worked_examples/RESPONSE_USE_CASE_ZH.md), including the joint neural extension, and [input-unit reuse](worked_examples/REUSE_USE_CASE_ZH.md), starting with the original snapshots. The [focused related-resource map](RESOURCE_INCREMENT_MAP.json) connects four inspected works to the specific delivered objects and decisions. It records a bounded comparison, not an exhaustive novelty search or evidence that the related tools cannot express these tests.

## Run the saved center example

Run these commands from this resource directory, using output paths that do not exist:

```text
python -B resource.py score netv4 --predictions components/netv4/saved_predictions/working__STID_NoCalendar_seed1.npz --out ../center_seed1
python -B resource.py score netv4_checks --predictions components/netv4/saved_predictions/working__STID_NoCalendar_seed1.npz --out ../center_seed1_bounds
python -B resource.py replay netv4 --out ../center_all_models
python -B resource.py replay netv4_checks --out ../core_lookup_and_bounds
```

In `center_seed1/original/decompositions.csv`, select `population=all_steps`, `cohort=all`, `timing_group=all`. In `recovery/metrics.csv`, select `population=all_steps`, `cohort=all` for each law. These are one seed's scores; the paper also gives the mean over all three seeds. `recovery/summary.json` binds the laws to the same forecast digest.

`D=L_FF-L_BB` is the diagonal full-arm error change. `T=L_BF-L_BB` changes the target while keeping the background-history prediction; `A=L_BF-L_FF` changes the prediction on a common full target. Thus `D=T-A`. Positive `A` means helpful response for that target. A negative `D` alone does not establish helpful response. Mixed STID's three-seed original-law means are `D=-0.562829` and `A=-0.108118`: full-arm error is lower even though changed history raises it. The alternative recovery law cancels that diagonal advantage. Bounds cover only the convex segment between the two frozen clipped targets, not all generators or a physical uncertainty range.

The `netv4_checks` replay reads fourteen bound files from the unchanged original archive. It recomputes sixteen fixed lookup-rule outputs and bounds for all eleven center configurations and two seed-loss means. Original mixed lookup improves overall LastValue MAE by only 0.12%; all hits are controls and constant/conflicting-future groups worsen. The full-nine-slot key has no hits. Read `lookup/metrics.csv`, `lookup/witnesses.csv` and `target_segment/SUMMARY.json` together.

## Run the complete-input example

```text
python -B resource.py replay netv4_mv --out ../mv_all_models
python -B resource.py score netv4_mv --predictions components/netv4_mv/predictions/working__RidgeLocal9.npz --out ../mv_joint_example
```

`mv_all_models/mv` contains exact regeneration and direct verification reports plus all thirteen saved-model scores. `mv_joint_example/scores` contains your submitted-model scores. The adapter uses 12 historical steps and nine slots; slot 0 is center, slot 1 its exact copy, and slots 2--8 use seven other retained weekly profiles selected by numeric similarity. Global road identities and measured topology are absent. Four fixed timings yield 11,424 scenarios and six histories per scenario (68,544 rows). They are conditional scenes, not 68,544 independent incidents.

For `responses.csv`, select `scope=cutoff_spanning`, `phase=all_points` and one `target_law`. `neighbor_given_center` is `L_CF-L_FF`: C keeps the full-arm center and background other histories; F uses full-arm histories everywhere. `neighbor_given_background_center` is `L_BF-L_NF`: the center stays at background while other histories change. Both use the same full-arm target. These are functional input contrasts, not measured interventions on roads.

With the supplied mixed Ridge and legacy target, total response is `A=+3.308983`, but `L_CF=16.610767` and `L_FF=16.693043`, giving conditional other-response `-0.082276` (0.50% higher MAE). Both target laws and both Ridge training variants retain this small conditional harm. Helpful profile groups are also retained (42 of 102 mixed groups, 45 generated groups); grouped scenarios are dependent.

Read `accuracy.csv`, `responses.csv`, `structural_controls.csv` and `profile_response.csv` together. All six STID checkpoints have adverse pooled but helpful cutoff-spanning response under both laws; the main text reports the legacy law. The two future-starting groups have equal complete histories across all views and zero response for all thirteen saved deterministic configurations. Eleven center-only configurations have exactly zero other-sequence response. These controls qualify the interpretation of the supplied forecasts; they do not certify arbitrary user code as deterministic.

## Submit your predictor

From the resource directory, the executable coupled example is:

```text
python -B components/netv4_mv/predict.py --inputs components/netv4_mv/task/inputs.npz --adapter components/netv4_mv/ridge_adapter.py:predict --out ../my_mv_forecasts.npz
python -B resource.py score netv4_mv --predictions ../my_mv_forecasts.npz --out ../my_mv_scores
```

Replace the adapter with your own `file.py:predict`. It receives finite `history[N,12,9]` and returns finite numeric `[N,12]`. The supplied example loads `weights.npz` beside the adapter. Keep model weights and settings fixed. The predictor receives no IDs, targets, mappings or phase labels as arguments; opaque IDs are attached afterward. This is an interface boundary, not a security sandbox or a blind test. Your NPZ must contain exactly `sample_id` and `prediction`, every supplied ID exactly once, in any order.

For a center-history model, from `components/netv4`:

```text
python -B predict.py --inputs task/inputs.npz --adapter last_value.py:predict --out ../../../my_center_forecasts.npz
```

That adapter takes `[N,12]` and returns `[N,12]`. Submit its NPZ from the resource root using `score netv4` and `score netv4_checks`. Use `--prediction-mode single_realization_stochastic` with `score netv4` when appropriate; a single random realization does not establish a deterministic-input verdict. Graph-dependent models and probability-distribution scoring are outside these contracts.

## Use a neural predictor that combines complete inputs

The separately documented `netv4_gwn` extension supplies six fixed primary Graph WaveNet checkpoints, two training snapshots and seeds 1--3. These models use adaptive connections between nine local numeric slots, without a measured road graph. No MV target was used for fitting or checkpoint selection. A superseded FP32-layout run and one interrupted training attempt are preserved separately. They are not extra independent seeds.

```text
python -B resource.py score netv4_mv --predictions components/netv4_gwn/runs/working__GraphWaveNet_seed1/mv_predictions.npz --out ../gwn_working_seed1
python -B components/netv4_gwn/predict_reader.py --inputs components/netv4_mv/task/inputs.npz --variant working --seed 1 --out ../gwn_reader_forecasts.npz
```

The first command scores one saved seed. The second runs the history-only model on CPU; its 822,528 forecast values match the CUDA forecasts within 0.000489, below the declared 0.001 tolerance. Read `components/netv4_gwn/evaluation/primary_seed_rows.csv` with `primary_seed_loss_means.csv`: all six primary checkpoints have helpful total but harmful conditional other-sequence response on spanning scenes under both target laws. Three-seed mean conditional error increases range from 2.55% to 3.54%. All timing groups, profile groups and original-test scores remain reported. Graph WaveNet improves original Generated overall MAE relative to STID but worsens Mixed overall MAE; this is not a universal accuracy benefit.

Use the unchanged `score netv4_mv` contract for a new predictor. The original `replay netv4_mv` continues to replay its thirteen inherited configurations; this extension keeps its own six primary configurations and technical sensitivity. See `components/netv4_gwn/README.md` for source pins and training staging.

## Inspect external reuse and retained support

```text
python -B resource.py show release_audit
python -B resource.py replay release_audit --source /path/to/incident_all.npy --out ../release_audit
python -B resource.py replay station_lookup --source /path/to/incident_all.npy --out ../station_lookup
python -B resource.py replay station_exclusion --source /path/to/incident_all.npy --out ../station_exclusion
```

The external file is not bundled. It must match the source digest and byte size given by `show`. Keep all 839 test queries and 496 node positions. Distinct incident records coexist with 93 repeated traffic history/future pairs; a training-history lookup reduces all-test MAE by 11.79%. This demonstrates a shortcut in this pinned release, not its use by a published neural model.

Network target-support exclusion uses test targets offline. It leaves station-level retrieval benefits; station flow-history exclusion empties 51 station training views. Including calendar history changes the input key and preserves rows but leaves flow-only hits. Read lookup losses and retained support together. These exclusions do not recover a chronological test, and equal values do not identify the same physical source observation.

## Observed and contextual reference tasks

```text
python -B resource.py replay observed --out ../observed_saved
python -B resource.py score observed --predictions components/observed/saved_predictions.npz --out ../observed_reader
python -B resource.py replay cik_10130110 --out ../context_second_station
python -B resource.py score cik_423339 --predictions components/cik_423339/probe/USER_FORECAST_EXAMPLE.npz --out ../context_first_station_reader
```

The observed task is a descriptive calendar-matched comparison at 2,062 five-minute origins. CiK probes use separate hourly 168-to-24 tasks with 32 keyed constructions each. Neither validates the synthetic incident law or shares NET-V4's scoring population. `show COMPONENT` states the schema; observed scoring accepts multiple models. CiK station IDs and source versions must remain distinct even when seeds coincide.

## Reproduce saved examples and retain provenance

```text
python -B reproduce_examples.py --out ../paper_example_results
python -m zipfile -e archives/NET_V4_paired_diagnostic_companion.zip ../paired_construction_run
python -B ../paired_construction_run/replay_construction.py
```

The saved-example arithmetic is replayed without the external release file. It does not replace source matching or direct-loss verification against that release. The archived center companion conditionally regenerates its four arms from retained profiles and settings. MV additionally reconstructs all of its new arrays and verifies 16,702,704 elements exactly. Neither restores the historical raw-source pool of the original snapshots.

Read `CONSTRUCTION_ASSUMPTIONS.md`, `CONSTRUCTION_SETTINGS.json`, MV `PROTOCOL_ZH.md`, and component notices. Original arrays and partitions remain bound in the archive. Current source coverage, processing indicators, observed clocks and affected-sensor assignments have declared limits. No command turns development-exposed groups into independent incidents.

Python 3.8+ and NumPy 1.20+ run the main checks; CiK replay also needs pandas. The auxiliary metadata archive has its own Python 3.12 requirements. PyTorch is optional for executing neural checkpoints, not required for saved-forecast replay. Output paths must be new and outside the resource; no network or package installation occurs. Hash verification establishes integrity; the separate scientific checks establish the stated arithmetic and reconstruction claims.
