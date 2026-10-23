# Source and code context

The new experimental profiles come from Basel-Stadt dataset 100006, not NYC.
Its hourly counts are uniformly divided into four nominal quarter-hour counts;
that transformation is an assumption, not observed subhour detail. Publisher:
Amt fuer Mobilitaet, Basel-Stadt. CC BY 4.0.
https://data.bs.ch/explore/dataset/100006/
https://creativecommons.org/licenses/by/4.0/

pins/legacy_generator.py preserves the original constructor function bodies.
Its inherited comments concerning incident realism or calibration are not
claims of the current paper. PROJECT_MIT_ORIGINAL.txt retains the original
project code notice verbatim, including any historical placeholders; it does
not grant a new upstream data license. SOURCE_ATTRIBUTION.md documents the
older NYC task, whose catalog is retained only for duplicate admission checks.
The original task and this Basel confirmation have distinct source identities.

The GWN implementation and its upstream license are in ../models/vendor/.
Actual six checkpoint weights, original scalers and logs are in ../models/runs/.
Default replay requires only Python and NumPy; optional inference needs PyTorch.
The exact author environment is listed in ../RECORDED_ENVIRONMENT.json.
The current paper is an author handoff, with author-machine paths and source
history retained. It is not an anonymous submission artifact.
