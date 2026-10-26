# Whole glycerol/head generation with an 8,000-template TRAIN bank

The user's 2026-10-26 direction starts a new experiment on the accepted unified
main library. Historical Polar E1/E2/E3 training is complete. Its failed RDKit
generation job and stop lock are preserved and are not restarted here.

Use `configs/glycerol_core_templates8000_v1.json` and the independent root
`outputs/glycerol_core_templates8000_v1_20261026`. All private data remain on G.

The 189,864 main labels include fixed glycerol and two distinct chain ports,
101 and 102. Do not substitute the legacy `[*:1]` Polar fragment or reuse an
incompatible historical output layer. Fit from random initialization using the
existing fingerprint/projector/set-transformer/decoder dimensions. This is a
new whole-core formula boundary, supporting both P and nonP: formula counts
refer to the neutral H-capped glycerol/head, not the marked graph or full lipid.
The decoder positional length is96, covering every label (observed maximum94).

Split90/5/5 by neutral stereo-free capped connectivity groups before selecting
any templates. Equivalent role, stereo and protonation forms stay together.
The frozen counts are TRAIN170872 / validation9493 / test9499. The bank has
6244 P and1756 nonP structures; it mixes traceable reference, database
substructure and rule-derived candidates, without implying all are reported
natural lipids or have spectra.
All8,000 bank entries are selected exclusively from TRAIN and remain TRAIN
prediction labels. Mask the target itself and every equivalent bank entry at
retrieval time, including when predicting a bank member. A bank-member
reconstruction evaluation is distinct from held-out validation/test performance.

Select8,000 via P count /100Da mass / sugar-or-inositol strata with sqrt-size
quotas and Morgan radius2/2048-bit MaxMin selection. Similarity excludes invariant
glycerol centers from fingerprint enumeration. Exact exhaustive Tanimoto
retrieval with deterministic RDKit tie resolution is audited against512
independent full-bank oracle queries. K3 and threshold0.2 apply; when no
nonself template reaches0.2, retain the nearest legal template and report this
fallback rather than silently creating a Formula-only or skipped target.
Formula normalization is fitted only on actual TRAIN retrieval pairs.
Target-structure retrieval is an oracle structural-pretraining input, not an
MS/MS retriever or an independent spectral benchmark.

Use batch32 / FP32 / Math Attention / TF32off. The selected prior E3 schedule
warms up4,000 updates to3e-4 then cosine-decays to1e-5. AdamW weight_decay0.01,
gradient clip1.0, no accumulation. The resource cap is80 complete epochs, with
steps derived from this new split. Select only minimum complete-validation
token cross entropy; patience5 complete validation epochs, consistent with
the previously frozen MetGenX selection rule. A cap does not prove convergence.

Before long CUDA training, run new CPU and CUDA backward/full-state disk-resume
gates, including longest labels, a nonP batch and bounded batch16/32 timing.
Both states include optimizer, scheduler, RNG, cursor and selection history.
Only the matching new supervisor may start CUDA jobs. A fresh monitor epoch
keeps historical weights,73 frozen V2 source hashes and the old stop lock intact.
No automatic retry is allowed after a new GPU stop latch.

Only best plus latest2 full checkpoints are retained for this experiment,
after atomic save/hash/CPU reload. The new owned-output limit is8GiB. Never
prune old experiments, restores, source provenance or stop/monitor evidence.

After validation-CE selection, a serial bounded queue evaluates Beam20 on
fixed validation512, bank-target512 leave-one-out and test512. Report raw beam
order separately from posterior neutral-formula/port/assembly filtering. An
invalid RDKit candidate, empty result or missing EOS contributes failure and
remains in the denominator. In particular, kekulization errors must not abort
the cohort. Record validity, formula coverage, exact@1/5/10, Similarity@10,
functional-group recovery and improvement over the best input template.
Functional-group F1 is measured on the neutral whole core, including the fixed
glycerol; variable-head fingerprint similarity excludes the invariant glycerol
centers. Report these distinct measurement scopes when interpreting recovery.
No test-based optimization selection, extra optimizer search, MS/MS, CL or
PGCN is introduced by this experiment. Other bank sizes/K/seeds remain later
experiments using these frozen new splits.

Commands (dot-source `scripts/enter_cuda_v04.ps1` first):

```powershell
python scripts/prepare_core_templates8000.py
python -m pytest tests/test_core_templates.py tests/test_formal_v2.py -q
# Save the actual test result and current test SHA as unit_tests.json.
python scripts/gate_core_templates8000.py --device cpu --output outputs/glycerol_core_templates8000_v1_20261026/cpu_gate_run
python scripts/supervise_core_templates8000.py --job cuda_gate --run-name cuda_gate_run --timeout-seconds 1200
python scripts/run_core_templates8000.py
python scripts/report_core_templates8000.py
```

For background launch, use `Start-Process -WindowStyle Hidden`, redirecting logs
into the new G-drive root. Verify actual supervisor/child process existence and
recorded progress; a stale `state.json` alone is not proof a run is alive.
