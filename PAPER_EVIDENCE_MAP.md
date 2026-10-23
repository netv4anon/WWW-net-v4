# Paper to evidence map

Paths are relative to this package. The final main manuscript governs the claims; historical component prose is supporting documentation.

| Main-paper content | Data / results / executable entry |
|---|---|
| §3; Table 1, original snapshots | `resource/archives/netv4_review_candidate_v0_3.zip`: `data/{packaged,working}/{history,targets,splits,analysis_metadata}.npz` |
| §3, generator assumptions and source identity | `resource/CONSTRUCTION_ASSUMPTIONS.md`, `CONSTRUCTION_SETTINGS.json`, `source_excerpts`, and `resource/components/netv4_mv/pins` |
| §4; Figure 1, controlled comparison | `resource/components/netv4/task`, `resource/components/netv4_mv/task`, scorers and evaluation contracts |
| §5.1; Table 2; Figure 2 | `resource/components/netv4/recovery_evidence/metrics.csv`, saved predictions and `resource/components/netv4_checks/saved_target_segment` |
| §5.2; Table 3; Figure 3 | `resource/components/netv4_mv/scores/responses.csv`, `resource/components/netv4_gwn/evaluation` and all six `runs` |
| §5.2, input-proximity restriction and failure | `resource/components/netv4_input_scope/frozen`: policy, coverage, per-seed results and selected arrays |
| §5.3; Table 4; Appendix E, Basel | `followup/baseline_confirmation/{raw,admission,profiles,task,predictions,scores}`; `replay_resource.py` |
| §5.3; Appendix F, decomposition and omissions | `followup/analysis/{saved_points,aligned_targets.npz,profile_contributions.csv,SUMMARY.json,OMISSION_SUMMARY.json}` |
| §5.3; Table 10; Appendix F, hourly averaging | `followup/paired/{original,hour_uniform,predictions,scores}` and `followup/admission/FOLLOWUP_DESIGN_FREEZE.json`; `replay_followup.py --arrays` |
| §6.1; Table 5, original lookup | `resource/components/netv4_checks` and `resource/evidence/original_snapshots` |
| §6.2; Table 6, release chronology/retrieval | `resource/components/release_audit`, including exact upstream source contract |
| §6.3; Table 7, retained support and station lookup | `resource/evidence/fixed_query_target_support`, `resource/components/station_lookup`, `resource/components/station_exclusion` |
| Appendix A; Table 8, original baseline scores | Original archive `tables/metrics_long.csv`, per-model `runs`, `resource/components/netv4` |
| Appendix B, contextual-model probe/failures | `resource/components/cik_423339` and `cik_10130110`; raw outputs and fallback scores retained |
| Appendix C, equal-history bound | `resource/components/netv4_checks` replay code and saved checks |
| Appendix D; Table 9; 559/642 matching scopes | `verification/verification/{MATCH_QUERY_IDS.json,MATCH_WITNESSES.csv,REUSE_SCOPE_VERIFIED.json}` |
| Appendix D, unrounded ridge/observed losses | `verification/verification/UNROUNDED_LOSSES_VERIFIED.json` and per-context/per-case CSVs; `verify_evidence.py` |
| Appendix D, actual GWN settings | `verification/verification/GWN_AND_CONSTRUCTION_VERIFIED.json`; actual weights/scalers/logs in `resource/components/netv4_gwn/runs` |
| Appendix G, observed reference | `resource/components/observed`, `verification/verification/OBSERVED_UNROUNDED_PER_CASE_SEED.csv`, technical companion |

The two corrected prose points in the final main manuscript reflect the retained code/results: the 48-context STID numbers are three-seed means; 2,856 MV backgrounds are 102 profiles × seven days × four start hours, each reused for four timing settings. Float64 re-accumulation of the hourly-uniform profiles yields maximum hourly-sum discrepancy 0.00048828125 and maximum relative discrepancy 1.1850625707112023e-7.
