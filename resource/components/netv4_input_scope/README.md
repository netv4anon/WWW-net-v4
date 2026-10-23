# One fixed history-proximity audit

This component adds a posthoc input-only check to the existing NET-V4-MV
response example. `frozen/` contains the original policy, program, seals,
selection arrays and all full/included/excluded outcomes. `describe_reference.py`
adds only the source composition of the already-returned training neighbours;
it never changes masks or scores. `CONSTRUCTION_RATIONALE_EN.md` distinguishes
design purposes, retained material and uncalibrated settings.

From the resource root, with NumPy >=1.22 and PyTorch installed:

```text
python -B components/netv4_input_scope/reproduce.py --out <new-directory>
```

The output directory must not exist. A GPU is optional; an exhaustive CPU run
is slower. The adapter verifies frozen resource hashes, temporarily extracts
only original history/split arrays, imports the unchanged frozen program,
reproduces the results and compares CSVs and masks. No external download,
training, model inference or source acquisition is required.

The source metadata is bound to original archive members. Tied nearest histories
are not assigned physical identities. Mixed supports persistence under this
rule; Generated fails profile coverage and sign/weighting stability. Neither
branch establishes physical realism or equality of distributions.

One original analysis plus one mechanical portable rerun has been executed.
They are the same scientific policy, not independent replications. Full original
results and the old candidate remain unchanged. No additional threshold or
subgroup search is authorized by this component.
