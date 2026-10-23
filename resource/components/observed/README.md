# Observed calendar-pair companion for the NET-V4 WWW revision

This companion exports the already evaluated February TraffiDent cohort: 268 cases, 763 controls, 94 stations and 2,062 forecast origins. It adds no observed records or fitted models. The input-only command requires Python 3.8+ and NumPy 1.20+; PyTorch is used during export only.

Recompute the archived scores:

    python evaluate.py --predictions saved_predictions.npz --data data --output scores.json

Replay all seven January-frozen predictors without access to targets or pair labels:

    python infer.py --inputs data/inputs.npz --weights models/weights.npz --scaler models/scaler.json --output replayed_predictions.npz

Evaluate the replayed or user-supplied predictions using the same evaluate command. Output files must not already exist. Prediction archives contain ids, models and predictions [models,2062,12]; IDs may be reordered, but missing, unknown or duplicate IDs and nonfinite values are rejected. Predictor inputs contain only opaque IDs and 12 flow observations. These file boundaries are an interface convention, not a security sandbox.

The scorer weights cases equally and all eligible controls equally within each case. It reports pre/post case-minus-control MAE and their difference G, weekly scores, and station-omission checks. STID_seed_mean averages seed losses, not forecasts. Archived G is 3.9046171197843784; it is a selected-cohort descriptive association, not an incident causal effect or an independent blind confirmation. Source points overlap between some episodes.

The paired timing uses pre history [c-25,c-13), pre target [c-13,c-1), post history [c-12,c) and post target [c,c+12). Bin c-1 contains the report. Nominal clocks are inherited and not independently calibrated.

BUILD_BINDINGS.json identifies original retained files. The package replays frozen inference and scoring; it does not contain all raw source blocks, recreate acquisition or independently reselect the cohort. The original audit checked those source intervals separately. Use the upstream TraffiDent/XTraffic terms for upstream data; no new source ownership or redistribution license is asserted. Public release arrangements remain separate from this local review artifact.
