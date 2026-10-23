# NET-V4 evaluation resource

Start with [READER_GUIDE.md](READER_GUIDE.md). It connects the paper's research decisions to supported inputs, commands, losses and limitations. This repository distributes supporting data and code. The technical evidence report is in `../technical_companion`; see the package-level README and evidence map. The manuscript is not included.

The original NET-V4 12-to-12 snapshots, the 256-context center diagnostic and the separately identified NET-V4-MV task are distinct resources. MV provides complete nine-slot histories and 11,424 controlled scenarios, regenerable from 102 retained weekly numeric profiles. The layout is center, its duplicate and seven profile-similarity sequences. Their physical adjacency and incident responses are not measured; conditional regeneration does not recover the legacy raw-source pool.

| Research decision | Entry | Predictor coverage |
|---|---|---|
| Lower event-arm error versus helpful history response | `netv4` | 1,024 keyed center-history forecasts, each with 12 future values |
| Response bounds on one declared target segment; original-split lookup | `netv4_checks` | Same center forecasts for bounds; saved rules for lookup |
| Total versus conditional input response; pooled versus timing response | `netv4_mv` | 68,544 keyed forecasts from full `[N,12,9]` histories |
| Joint neural use example on the unchanged MV task | `netv4_gwn` via `score netv4_mv` | Six primary checkpoints; all seeds, timing groups and target laws |
| Unique incident records versus repeated predictor inputs | `release_audit`, `station_lookup`, `station_exclusion` | The pinned external release is required |
| Report-time errors relative to calendar controls | `observed` | 2,062 keyed forecast origins |
| Target sensitivity in a separate contextual construction | `cik_423339`, `cik_10130110` | 32 keyed 24-hour forecasts per station |

From this directory:

```text
python -B resource.py list
python -B resource.py verify
python -B resource.py replay netv4_mv --out ../mv_replay
python -B resource.py score netv4_mv --predictions components/netv4_mv/predictions/working__RidgeLocal9.npz --out ../mv_reader_scores
python -B resource.py score netv4_mv --predictions components/netv4_gwn/runs/working__GraphWaveNet_seed1/mv_predictions.npz --out ../gwn_reader_scores
```

Output directories must be new and outside this resource directory. `show COMPONENT` gives the component schema and scope. Nine components have executable replay entries and six accept forecasts; catalog-only components are inspectable with `show` and `verify --component`. There is no pooled score across resources or a single combined population.

The separate Graph WaveNet use study preserves six primary fits and all adverse accuracy results. Helpful total response coexists with harmful conditional other-sequence response on spanning cases under both laws, with three-seed loss-mean error increases of 2.55–3.54%. All six checkpoints also have adverse pooled but helpful spanning response. The thirteen original MV configurations remain unchanged; `replay netv4_mv` replays those inherited results. See [the neural component](components/netv4_gwn/README.md) for saved-forecast scoring, optional CPU inference and zero-fit training staging.

Array construction and saved-forecast scoring use Python 3.8+ and NumPy 1.20+. CiK reconstruction also uses pandas; PyTorch is needed only to execute the saved neural weights, not to replay their scores. No automatic installation or network access occurs. Hashes verify stored bytes. Scientific verification reports separately identify exact conditional regeneration, direct-loss checks and reader-interface checks.

Read `CONSTRUCTION_ASSUMPTIONS.md`, `CONSTRUCTION_SETTINGS.json`, the MV protocol, and the component notices. Upstream licenses remain separate; this local bundle does not assert public deposit or grant a replacement license. External release matching requires the exact upstream file, which is not included. Actual reviewer access is provided by the submission attachment or anonymous artifact link; no public deposit is asserted here.
