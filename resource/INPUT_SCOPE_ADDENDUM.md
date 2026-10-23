# History-proximity addendum, 2026-10-11

The original 551 bound resource files and scientific objects are unchanged.
The additional standalone component is `components/netv4_input_scope`.
It supplies the fixed posthoc policy, all input selections, all seed/law
full/included/excluded scores, profile-weighting results and source composition.

Run from this resource directory, with NumPy >=1.22 and PyTorch:

```text
python -B components/netv4_input_scope/reproduce.py --out <fresh-directory>
```

Mixed supports persistence under the fixed training-reference rule. Generated
fails profile coverage and its signs depend on seeds and profile weighting.
These outcomes are not physical incident validation. The original full results,
observed reference and external reuse audit retain their separate contracts.
Read the component README and construction rationale for exact scope.
