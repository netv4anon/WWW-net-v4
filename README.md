# NET-V4: data, code and supporting evidence

Repository: <https://github.com/netv4anon/WWW-net-v4>

This package accompanies **NET-V4: Diagnosing History Response and Input Reuse in Traffic Forecasting**. It contains the original 12-to-12 datasets, controlled tasks, frozen model weights and forecasts, source checks, and the complete new-source and representation follow-up evidence. Read `PAPER_EVIDENCE_MAP.md` for paper-to-file locations and `ERRATA_AND_SCOPE.md` for source and licensing boundaries.


## Reviewer quick start

This repository contains supplementary datasets, code and supporting evidence. The manuscript and its typesetting sources are not distributed here. Start here; no account, API
key or network access is required for the offline checks. This distribution
contains no author contact information or manuscript registration identifier.
Third-party scientific citations, publisher attribution and software notices
remain intact.

```text
python -m pip install -r requirements-replay.txt
python -B reproduce.py --out ../netv4_replay_results
```

Dependency installation is a separate, optional environment-setup step. The
replay command itself does not install packages or download data. Use a **new**
output directory outside this folder. The seven checks verify the inventories,
reconstruct center and multisequence tasks, reconstruct Basel admission and
arrays, recompute hourly-representation results and check unrounded evidence.
They do not retrain or generate new neural forecasts. For an individual check:

```text
python -B reproduce.py --list
python -B reproduce.py --checks basel representation --out ../netv4_followup_results
```

`REPRODUCIBILITY_REPORT.md` gives the actual validation scope and limits.
`ANONYMITY_REPORT.md` describes the identity and metadata inspection.
`PAPER_EVIDENCE_MAP.md` maps paper claims to data and executables.
`technical_companion/NET_V4_technical_companion.pdf` is detailed supporting
documentation, not an extension of the submitted paper's page budget.
The two complete original datasets are inside
`resource/archives/netv4_review_candidate_v0_3.zip`, under `data/packaged/`
and `data/working/`; each snapshot contains 33,076 sequences.

Some pinned documents preserve historical development context or machine paths
without a personal account name. Their wording about an earlier handoff is a
historical record, not the access status of this distribution. Use the portable
entry points listed here. Historical placeholder DOIs are not access links;
see `ERRATA_AND_SCOPE.md`. This folder does not create a hosting URL by itself.

## Included data and evidence

| Directory | Contents |
|---|---|
| `resource/archives/netv4_review_candidate_v0_3.zip` | Both complete original snapshots, each with 33,076 sequences: histories, targets, metadata, splits, original forecasts, STID weights, training and replay code |
| `resource/components/netv4` | 256 paired center contexts, four arms, fixed predictions, recovery-target scores |
| `resource/components/netv4_mv` | 102 processed weekly profiles; 11,424 scenarios; 68,544 input rows with nine numeric slots; targets, construction code and baseline forecasts |
| `resource/components/netv4_gwn` | Six primary GWN checkpoints, scalers, epoch logs, saved forecasts, CPU adapter and upstream license |
| `resource/components/netv4_input_scope` | Fixed proximity rule, selected inputs, coverage gates, all adverse results |
| `resource/components/{release_audit,station_lookup,station_exclusion}` and `resource/evidence` | Release contracts, matching offsets, retained-row masks, lookup/exclusion results and audit code |
| `resource/components/observed` | Descriptive observed-traffic comparison, retained inputs/targets, matched controls, saved forecasts and scorer |
| `resource/components/cik_*` | Contextual-model probe, retained source prefix, fixed constructions, raw outputs, fallback and failure results |
| `followup/baseline_confirmation` | Basel hourly source response and metadata, 73 admitted profiles, all six GWN forecasts, two target rules, all timing/profile results, frozen protocol |
| `followup/analysis` | Prediction increments, errors, per-profile and horizon contributions, site-omission results, aligned C/F forecasts and targets |
| `followup/paired` | Original and hourly-uniform representations, complete inputs/targets/forecasts, both paths of crossed scoring, all 12 primary comparisons |
| `verification/verification` | Match query IDs and witnesses, unrounded losses, actual GWN configurations and weight/log bindings |
| `technical_companion` | Technical evidence report and supporting vector figures; document numbering is local |

## Reproduce without training or downloading

Use Python 3.8+ and NumPy 1.20+. Pandas is additionally needed for the contextual-model probe; PyTorch is optional for executing model weights and checking their shapes. No package installation or network request is triggered automatically. All commands below use new output directories **outside** this resource package.

```text
python -B resource/resource.py list
python -B resource/resource.py verify
python -B resource/resource.py replay netv4 --out ../replay_center
python -B resource/resource.py replay netv4_mv --out ../replay_mv
python -B followup/baseline_confirmation/replay_resource.py
python -B followup/replay_followup.py --arrays
python -B verify_evidence.py --out ../verified_losses
```

`replay_resource.py` replays Basel admission, arrays and saved-forecast primary losses. It reports the failed all-twelve conjunction, with **7 passing and 5 reversing conditional responses**. `replay_followup.py --arrays` regenerates all 16,702,704 hourly-uniform task elements and recomputes decomposition and crossed scores. The paired old-profile result does not replace the Basel result. The scripts do not retrain or choose models.

The original snapshots remain inside a ZIP to preserve their frozen source binding. Unzip that archive separately for direct access to `data/packaged` and `data/working`; `resource.py` accesses it as needed. `resource/READER_GUIDE.md` documents nine replay and six forecast-scoring entries. Optional model inference saves new outputs separately; it never overwrites the frozen forecasts.

## External-source audit

The external IGSTGNN raw source is **not included**, as stated in the paper. Obtain `Contra_Costa/incident_all.npy`, version 1 of `lixiangfan/data4igstgnn`, from <https://www.kaggle.com/datasets/lixiangfan/data4igstgnn>. Its exact contract is 865,971,481 bytes, SHA-256:

```text
bc6c5608457f88864625d683f9a66d17bc0fc31fe5acba366b6033708a099279
```

Then run:

```text
python -B verify_evidence.py --source /path/to/incident_all.npy --out ../verified_with_source
python -B resource/resource.py replay release_audit --source /path/to/incident_all.npy --out ../release_replay
```

Saved audit outputs, matching witnesses and training masks are included for inspection without that download. Full regeneration from the missing historical NYC raw-source snapshot is not supported. Conditional reconstruction from retained profiles and saved-forecast scoring are supported and are distinct from raw-source regeneration.

## Repository contents and complete download

All data, model weights, saved forecasts and replay code are included in this
repository tree; Git LFS is not required for these files. The original data ZIP
inside `resource/archives/` is part of the resource and must remain included.
The repository's `.gitattributes` disables line-ending conversion to preserve
the byte-level inventories on Windows, macOS and Linux.

A separately supplied `NETV4_GitHub_anonymous.zip` is a complete download of
this same tree, suitable as a GitHub Release attachment. It is not an extra file
to commit inside the repository. No manuscript PDF or LaTeX typesetting source
is included. The technical evidence report and scientific figure PDFs remain.

## Integrity and distribution

Run `python -B verify_package.py` to verify the current package inventory. Each scientific array, target, model weight and saved forecast was copied without modification. Distribution manifests bind this review copy, including anonymized historical path strings; they do not constitute new preregistrations. Original source hashes and scientific freeze records retain their historical meaning. Historical lists may identify obsolete manuscripts or private handoff notes that are intentionally absent; omissions are explicitly recorded. Actual reviewer access depends on the submission's artifact attachment or anonymous access link.

Upstream notices and licenses apply separately. This bundle does not assign a new blanket license to all source data. No assertion of acceptance, physical incident validation or public deposit is made.
