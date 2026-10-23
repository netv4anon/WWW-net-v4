# NET-V4 WWW revision: public-release audit companion

This local companion reproduces the calendar lower bound and two fixed lookup diagnostics on one pinned IGSTGNN release. It extends the NET-V4 revision; it is not the full NET-V4 dataset, a corrected chronological benchmark, or a reproduction of the IGSTGNN model.

Use Python 3.8+ and NumPy 1.20+. Obtain the Contra_Costa incident_all.npy from the version-1 resource in SOURCE.json; the large upstream file is not bundled. The program checks its size and SHA256 before using source-bound byte offsets. It does not execute pickle constructors and requires no parent project files or network access.

Run from any directory:

    python audit_release.py --source /path/to/incident_all.npy --output /path/to/new_output_directory

The output directory must be new. The program writes aggregate scores, per-row metrics, all training matches, and a comparison to the archived result. Expected: calendar minimum 19,906.7465 days; 99 validation and 93 test history hits; test MAE 16.166514 -> 14.260033. This is a history-lookup consequence, not estimated neural-model score inflation.

The offset manifest is valid only for the pinned source digest. Matching uses complete traffic-plus-calendar history bytes, training targets and last-value fallback. Scoring uses all 12 x 496 targets with published inverse normalization and no clipping/zero masking. Targets were inspected during prior research; this replay is not an independent statistical test. Numerical repetition does not establish absolute dates or original incident identity.

The failed 13-station source-trace experiment and full structural/native-decoder verification remain separate archived stages. This tool replays the calendar bound and lookup consequences, not every metadata/provenance finding. Source access and redistribution remain governed by upstream terms; no upstream ownership or license grant is asserted here.
