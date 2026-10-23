# Fixed Graph WaveNet resource-use study

This component adds one existing joint-input architecture to the unchanged NET-V4-MV task. It is a use-validation study of the evaluation resource, not a new forecasting algorithm. `packaged` means Generated training; `working` means Mixed training. Seeds 1, 2 and 3 are the six primary fits. They use the original train/validation indices, never MV targets for model selection. All primary forecasts were sealed before MV scoring; previous development exposure remains.

The official source is pinned to `6b162e80c59a1d494809252eca055cff93dc66b1`, with its MIT license in `vendor`. Three Conv1d declarations with two-dimensional kernels are mechanically replaced by Conv2d. The adaptive-only model has 272,640 parameters, nine local numeric slots, 4 blocks with 2 layers each, channels 32/32/256/512 and dropout 0.3. Learned slot connections are not measured road adjacency. The fixed training recipe is in `PROTOCOL_ZH.md` and `run_study.py`.

On all six primary checkpoints, total response is helpful but other-sequence response conditioned on the event center is harmful on cutoff-spanning cases under both target laws. Three-seed loss means imply a conditional error increase of 2.55–3.54%; helpful profile groups remain. All six have adverse pooled but helpful spanning response. Original-test accuracy improves relative to STID for Generated, but worsens for Mixed. MV spanning legacy accuracy is worse than STID for both. No universal accuracy improvement is claimed.

## Score delivered forecasts

From the resource root, with Python 3.8+ and NumPy:

```text
python -B resource.py verify --component netv4_gwn
python -B resource.py score netv4_mv --predictions components/netv4_gwn/runs/working__GraphWaveNet_seed1/mv_predictions.npz --out ../gwn_working_seed1
```

The scoring output uses the existing MV interface; there is no new `score netv4_gwn` command. The output directory must be new and outside the resource. Substitute any of the six primary run directories. The prediction files have `sample_id[68544]` and `prediction[68544,12]`; row IDs bind six views for each of 11,424 scenarios. MV task inputs and evaluator targets remain in `components/netv4_mv/task`, without duplication here.

## Execute a delivered predictor

The input-only CPU reader was tested with PyTorch 2.4.1 and NumPy 1.24.4:

```text
python -B components/netv4_gwn/predict_reader.py --inputs components/netv4_mv/task/inputs.npz --variant working --seed 1 --out ../gwn_reader_forecasts.npz
```

The adapter consumes historical arrays and loads weights/scaling; it does not consume evaluator targets. `--device cuda` is optional. The working-seed-1 CPU replay checks 822,528 forecast values against sealed CUDA forecasts: maximum absolute difference 0.00048828125, below the predeclared 0.001 tolerance. This is numerical agreement, not bitwise equality or a security sandbox. See `READER_VERIFICATION.json`.

## Optional training reproduction

The immutable component contains saved results, not a writable training workspace. Prepare a fresh staging directory from the delivered original archive and MV task:

```text
python -B components/netv4_gwn/prepare_training_staging.py --archive archives/netv4_review_candidate_v0_3.zip --mv-task components/netv4_mv/task --out ../gwn_training_reproduction
```

This checks all 20 pinned inputs/source files and fits zero models. To train the six primary configurations, run `python -B run_study.py` inside that new staging directory, then `python -B evaluate_study.py` after its forecast seal exists. These commands require PyTorch, compute and new outputs. The original technical attempts are inherited as recorded technical sensitivity, not refitted or counted as new scientific replication. Exact GPU training reproduction across software/hardware versions is not asserted.

## Evidence and complete accounting

- `FORECAST_SEAL.json`: six primary weight/forecast bindings, plus a separate superseded completed run.
- `evaluation/primary_seed_rows.csv`: each primary checkpoint's pooled and spanning losses under both laws.
- `evaluation/responses.csv`, `accuracy.csv`, per-run scores and profile files: all timings, phases, laws and profiles.
- `evaluation/primary_seed_loss_means.csv`: means of three seed losses, not scores of a mean forecast.
- `evaluation/functional_input_checks.csv`: actual other-input forecast changes and nonzero gradients on 32 regularly spaced spanning contexts per checkpoint.
- `evaluation/FINDINGS.json`: direct-loss verification of 240 values (maximum discrepancy 5.33e-15), original-test scores and favorable/adverse outcomes.
- `technical_attempts`: the completed superseded contiguous-layout seed 1, seed 2 interrupted at epoch 22, original code versions and technical logs. The superseded forecast is scored in `evaluation/technical_contiguous_seed1`, separately from primary means.

There were 8 fit starts, 7 completed fits, 1 interrupted attempt, 6 primary fits and 1 architecture. There are zero new scenarios and zero new observed records. Storage-layout profiling and reader replay are technical checks, not additional architectures or independent experiments. The shared numeric profiles and constructed scenarios are dependent; this study establishes neither physical incident effects nor independently observed generalization.
