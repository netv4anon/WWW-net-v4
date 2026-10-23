# Scoring clarification before first loss computation

The reference construction and nine prediction cells are now saved; no target
losses have been computed. Primary full-horizon and event-region MAE give each
of the 32 instances equal weight. Also report pooled event-position MAE because
interval lengths vary. For the prespecified comparison, delta = anchor-response
MAE minus pointwise-response MAE within each base rule. A sign change requires
opposite signs with both magnitudes > 1e-12, to exclude floating-point noise.
Identical last-value response arrays are one distinct predictor for cumulative
accounting: nine table cells represent eight distinct prediction configurations.
This is a clarified aggregation rule, not retrospective model selection.
