# Fixed-cohort structure pretraining midterm protocol

This experiment resumes the existing FP and GINE step-1001 ROCm Math checkpoints.
It measures generalization to held-out polar graphs under structure-oracle retrieval;
it does not evaluate MS/MS conditioning or establish prospective novelty.

## Frozen data and comparison

- Keep the existing 20,000 training and 2,000 validation target rows, in their exact order.
  The original model, optimizer, LR, seed, formula standardizer, data contracts,
  FP32, Math SDPA and effective batch size 16 remain unchanged.
- Reuse the original first 128 validation targets for direct smoke comparison.
- Add 512 distinct validation head graphs, disjoint from those 128 target rows,
  sampled round-robin over joint chemical family, label length, chain linkage,
  and cross-head top-1 template similarity strata. This is a diagnostic cohort,
  not a population-weighted estimate. Test targets never participate.
- Freeze row-array and ordered target-ID hashes, label manifest and split hashes
  before new training. Never publish target IDs, structures, or row arrays.
- All validation references are training structures with different canonical heads.
  P/A linkage is absent from the frozen validation split; no sample is moved across
  splits to create artificial coverage. Report this limitation explicitly.

## Independent evaluation

`scripts/evaluate_checkpoint_v03.py` loads an immutable checkpoint in a separate
process. It verifies the data bindings and evaluates all 2,000 original validation
targets with teacher forcing, and both the 128 and 512 cohorts with BOS-only greedy
generation and beam size five. It never repairs SMILES or forces EOS. The full
raw strings, token IDs, log probabilities and per-beam decisions remain private.

Teacher-forced polar loss is token-weighted, chain loss is slot-weighted; their
configured weighted sum is comparable to the previous smoke evaluation. Report
teacher-forced token/EOS accuracy separately from free-generation behavior.

Free-generation stages are exclusive and ordered: missing EOS; SMILES syntax;
attachment topology; polar/builder chemistry; chain combination; assembled neutral
formula mismatch; formula matches but target graph differs; fully correct. RDKit
syntax, sanitized molecule, attachment topology and accepted builder polar are
separate flags. Use the original frozen builder rules; novel heads receive the
same conservative generic rule as the original evaluator. Morgan similarity uses
radius two, 2048 bits, without chirality, and zero for unassembled predictions.

Raw metrics include EOS, polar validity/graph exact/Top-5, template copying,
length distribution, non-NULL chain accuracy, chain count/set, assembly, neutral
formula, full graph exact and Morgan similarity. An optional formula filter chooses
the first formula-matching beam using the same predicted chain pair and reports
coverage plus overall and conditional exactness separately. It never replaces raw
scores or introduces target-structure matching as a selection criterion.
Non-NULL chain accuracy uses canonical slot agreement for non-NULL target slots;
chain-set exactness ignores the predicted slot order. Graph equality follows the
frozen nonisomeric builder representation and does not establish sn-specific or
stereochemical identity.

## Template transformation audit

`scripts/prepare_midterm_v03.py` inspects every Top-10 reference for every frozen
training target in both normal and polar-excluded caches. Self references and
identical head-plus-chain pairs are forbidden. H changes the head alone, C changes
chains alone, HC changes both. Preserve per-reference codes privately. Report
pair proportions, targets with at least one relation (overlapping denominators),
target relation bitmasks and similarity-bin pair/target counts. Keep the original
50/50 mixture, including its deterministic per-target assignment.

## Resume, checkpoints and safety

Runtime-only `--sample-rows-from`, `--train-only`, and
`--retain-periodic-checkpoints` flags leave the original checkpoint contract intact.
`--steps` is the cumulative optimizer count. Each phase reuses exact original
sample arrays, verifies the full contract on resume, and saves independent step
files. Evaluation cannot alter saved optimizer, RNG or sampler cursor state.
The serialized original `run.steps=1000` and `run.kind=rocm_smoke` labels remain
unchanged for compatibility. Actual phase endpoints come from CLI `--steps` and
the checkpoint's numeric `step`, independently audited against AdamW counters.

Run under the existing Windows dump and progress supervisor. New GPU fault dumps,
nonfinite training values, or abnormal progress stalls stop the process; do not
retry automatically or change attention, precision, driver or TDR settings.

Evaluate 1001, 1500 and 2000 before deciding on longer runs. Extend only with stable
hardware and a positive validation/generation trend; severe regression or fault
stops extension. Complete FP analysis before starting the GINE midterm run.
Full pretraining requires a separate decision; this experiment never starts it.

Example supervised phase (PowerShell 7; use a fresh run name each time):

```powershell
$linuxHome = (& wsl.exe -d Ubuntu-24.04 -- printenv HOME).Trim()
$share = '\\wsl.localhost\Ubuntu-24.04' + "$linuxHome/projects/phospholipid-structure-pretrain".Replace('/', '\')
$supervisor = [scriptblock]::Create((Get-Content -LiteralPath "$share/scripts/supervise_rocm_diagnostic.ps1" -Raw))
& $supervisor -Job midterm -Encoder fingerprint -Steps 1500 -RunName fp_1500_new -Resume outputs/pretraining_v03/rocm_smoke/fingerprint_resume/checkpoint/last.pt -TimeoutSeconds 3600
& $supervisor -Job evaluate -Encoder fingerprint -Steps 1500 -RunName fp_1500_new -Resume outputs/midterm_v03/training/fp_1500_new/checkpoint/step_001500.pt -TimeoutSeconds 3600
```

The supervisor invokes `scripts/run_rocm.sh`, retains FP32/Math settings and stops
on a new Windows GPU dump or abnormal progress stall. Repeat only after reviewing
the preceding phase; commands above do not launch further phases automatically.
