# Full-test online constraints and a Formula-only control

The user explicitly authorizes all9,499 frozen test targets and an independently
trained Formula-only model, and prioritizes the previously missing online formula
constraints. Keep all old runtime sources, labels, splits, banks, checkpoints and
stop latches intact. Use `configs/core_formula_comparison_20261027.json` and the
independent G-drive root `outputs/core_formula_comparison_20261027`.

The fixed template checkpoint is epoch43/step229620, SHA256
`f8d7f646f6bfda7459296f5025adc78a9695b8b8bff4c34e8ae80c6d779e17c2`.
Test composition is8225 P /1274 nonP. No test target or equivalent graph is a
TRAIN reference, and the8,000 bank entries remain TRAIN output labels.

Run three actual Beam20 search modes, each on the entire same test set:
raw, elements (plus the two port-count budgets), and elements+basic grammar.
Posterior formula/graph filtering is separately reported for each mode and is
never described as online constrained search. After the ordinary special-token
mask and log_softmax, mask tokens that would exceed C/N/O/P/S/F/Cl/Br/I counts
or repeat a chain port. EOS requires exact heavy counts and both ports once.
Grammar additionally checks balanced nonempty branches, ring closure/bond
consistency, one connected component, and single O-bound101/102 ports. Final
RDKit/core-frame/assembly checks, not lexical character counts, verify H and the
entire neutral H-capped glycerol/head formula. No repair, forced EOS, formula
alteration or catalog whitelist. Dead and unfinished targets remain failures.

The decoder uses the original mean-log-probability ranking, float64 accumulation
and stable finished-first tie handling. Feasible probabilities are not
renormalized. Batching16 queries changes throughput, not beam/length budgets.
Fake-decoder tests compare exactly with the original sequential algorithm; the
real checkpoint gate records FP32 batched/serial logits and shared-beam scores.
Very close beams may differ with batched FP32 GEMM. Both compared models use
the same implementation, and the new full result does not splice in the older
512-target prefix.

Formula-only uses byte-identical TRAIN170872/validation9493/test9499 labels,
tokenizer and lengths. It never loads reference/fingerprint/mask/rank arrays.
Only its TRAIN target-formula mean/std are fitted. A fixed coordinate expansion
of this formula feeds the original2048-to256 MLP; the original20D formula MLP,
fusion512D,2-layer set transformer with TCLS, and2-layer decoder stay trainable.
Three formula-only tokens match the memory size. No reference formula or delta
enters the model; the internal second formula slot is a constant zero, not a
retrieved value. Trainable parameter names, shapes, counts and seeded initial
values match the template model exactly. Input information content intentionally
differs; equal tensor count does not imply equal information or identifiability
of first-layer columns under a10-dimensional input.

Use random initialization/seed42, batch32, AdamW/weight_decay0.01/clip1.0,
FP32/Math/TF32off and the same4000-update warmup to3e-4 and cosine to1e-5.
Both have the same80-epoch/427200-update cap and patience5, selecting minimum
complete-validation token CE only. Actual early-stop counts can differ and must
be reported. No test-based optimizer/epoch selection or new hyperparameter sweep.

CPU/CUDA actual backward/full-state disk recovery, longest94-token and TRAIN
nonP batches must pass. Reachability has been checked for all9499 gold test
labels/482902 legal prefixes with zero pruned targets.47 new/relevant tests pass,
including no early EOS/over-budget atoms, bracket/map digits, ring/branch rules,
unchanged original probability scoring, dead-beam failure, and no Formula-only
reference data access. Use only `scripts/supervise_core_ablation.py` for native
CUDA. Preserve WATCHDOG access limits and all original protections. A new stop
latch stops the queue; never automatically retry it.

The serial queue starts template elements+grammar first, then raw and elements,
then Formula-only training, then the three matched full-test modes. After each
job, the CPU reporter publishes aggregate status with unfinished values null.
The reporter is deliberately outside the frozen numerical runtime contract.

For all/P/nonP report exact@1/5/10, Similarity@10, core-valid and RDKit parse
rates, formula-correct@1/5/10, full-formula candidate rates, target formula
coverage, no-qualified rates, and gains over the same best nonself template.
Formula-only receives this common baseline only in metrics after generation.
Report both actual-emitted and nominal9499x20 candidate denominators; unfilled
slots count as failure in budget rates. Raw and posterior filtered ranking are
separate; exact/formula checks include all elements/H. All compressed per-target
records and weights stay on G. Keep best+latest2, max3 full checkpoints,8GiB new
owned-output cap, no logits/attention/all-pairs matrix.

The primary paired comparison is elements+grammar, posterior-filtered Exact@10.
Additional raw/elements and similarity/P/nonP results remain separate. Report
paired intervals/McNemar; bootstrap whole target-formula groups so repeated
Formula-only queries are not treated as independent inputs. Distinguish target uncertainty at seed42 from
independent-training-seed variability. Template retrieval still uses the known
target structure; benefits concern oracle structural conditioning, not proven
real-MS/MS identification. No MS/MS, CL, PGCN or extra training seeds.

Dot-source `scripts/enter_cuda_v04.ps1`, then prepare data, save the real test
receipt, run `gate_core_ablation.py --device cpu`, supervised `--job cuda_gate`,
and finally `run_core_ablation.py` in a hidden background process with G logs.
Never launch a duplicate queue based on a stale status file; verify actual PIDs.

After all six full cohorts and Formula-only training finish, the CPU reporter
emits the predeclared primary comparison and calls the one-shot
`publish_formula_comparison_results.py`. This verifies9499/8225/1274 counts,
all saved raw-stream hashes, best-validation/checkpoint SHA, both source
contracts, exact parameter equality and private-field exclusion. It stages only
the two aggregate Markdown/JSON files, and only with an empty pre-existing Git
index on the expected main/origin. Network failure or other staged work leaves
`PUBLICATION_PENDING.json` on G with numerical results preserved. Successful
remote SHA verification produces `PUBLICATION_DONE.json`. It never starts or
retries a GPU job. Reports stay null while comparison data are incomplete.
