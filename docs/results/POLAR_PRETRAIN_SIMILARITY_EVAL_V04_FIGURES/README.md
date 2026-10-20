# Polar similarity figures

Public figures contain only aggregate curves or unlabelled binned densities.
All panels use the same reused 512 validation diagnostic targets, Polar-only
nonchiral Morgan radius 2 / 2048-bit Tanimoto, and the original model-ranked
strict candidate prefixes. Missing candidates score zero.

- Figure A: Top-1/3/5/10 exceedance curves, `P(S_N >= threshold)`; higher is better.
- Figure B: paired best-template versus generated Top-N similarity, Beam20/50.
  Hexagons count targets; the diagonal is equal similarity.
- Figure C: best-template proximity versus generated-minus-template gain,
  final 50k/5,000 model. Lines mark zero gain and 0.4/0.7 template strata.

Each is supplied as PNG, SVG and PDF. SHA256 values are in `FIGURE_SHA256.json`.
The template-best selector and Top-N target-similarity maximum are posthoc
evaluation oracles, not deployment ranking rules. These are not MS/MS accuracies.

Figure D contains private molecular structures and is intentionally absent here.
It remains on G drive under the experiment's `private_Figure_D/` directory,
with original candidate ranks, selection rules, formulas and real similarities.
