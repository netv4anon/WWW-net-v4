# Fixed feasibility audit of station-input exclusion

The previous goal turn made progress: four fixed lookup configurations showed
a residual local retrieval benefit and were independently verified. This
follow-on evaluates a specific numerical exclusion operation, not a new model.

Freeze this protocol and implementation before reading new retention results.
The underlying release, existing masks and prior diagnostics are exposed.

Use the same IGSTGNN Contra_Costa v1 file, 3,909 original training rows and
839 fixed test rows (global indices 4,746:5,585), all 496 node positions.
The two candidate views are Original and existing TargetSupport (2,665 rows).

For each fixed node position independently:
- Primary input key: exactly 12 raw float32 historical flow values.
- Prespecified alternative key: those values plus the 12 HISTORICAL calendar
  pairs. No future calendar, query future, loss or incident label enters it.
- Remove every candidate training row whose key equals ANY test history key
  at this same node position. Keep all remaining rows for this node.
- Save the full [configuration, original training row, node] boolean masks.
- Save the earliest matching test row as a witness for each removed row;
  -1 means retained candidate and -2 means outside that candidate view.
No tolerance, cross-node search, input jitter, dropping query rows, dropping
stations, random sampling, fitting or prediction is permitted.

The inherited TargetSupport row set was defined using query targets; the NEW
history exclusion does not undo that exposure. Source-file hashes process
bytes but are not semantic access to query targets.

Report all four view/key combinations:
1. Each node's retained count, fraction, distinct full input keys and distinct
   flow histories; candidate/retained constant and varying flow windows.
2. Before/after query hit counts at the DECLARED key, and residual flow-only
   query hits after the calendar-key exclusion. The latter diagnoses scope;
   it does not invalidate the declared calendar-key contract.
3. Zero-support nodes, nodes retaining 1–12, 13–99, 100–499 and >=500 rows,
   and retention min/quantiles/max. These bins are descriptive, NOT a chosen
   trainability or acceptance threshold.
4. Query history variability and constant-near-zero status of depleted nodes.
   Numerical exact constancy defines constant windows. Inverse flow
   tolerance .001 is only a descriptive near-zero label, never a key match.
5. Intersection of all per-node keep masks: common rows that would satisfy
   this particular exclusion simultaneously at all nodes. This is stricter
   than independent station training and is not a required graph split.

An empty per-node set prevents fitting that node from this view alone;
nonempty support, many rows or many distinct keys does not establish
statistical adequacy or generalization. A normal constant history can recur
without shared physical source identity. Do not call all collisions leakage.

Independently verify masks and witnesses with sorted fixed-width records and
the original historical bytes, and bind each source/contract by SHA256.
Verify that future flow targets are not parsed or used in this audit.

If exclusions empty or collapse some nodes, retain every affected node in
the report. Do not drop those nodes to declare success. Do not claim to fix
chronology, raw-source independence or published model inflation.

No training or predictive-score experiment follows automatically from this
audit. Decide what is supported only after all four fixed views are reported.
