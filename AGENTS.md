# Project execution notes

- Current Windows workspace: `G:\yzx\linzhiyuce`. Keep project data, checkpoints,
  experiment outputs, downloads and caches on this G drive, as requested by the user.
- On this native NVIDIA machine, dot-source `scripts/enter_cuda_v04.ps1` to select
  `../environments/polar-v04-cu128/Scripts/python.exe` and G-drive caches. Do not use
  the unrelated copied venv or the hard-coded WSL/ROCm interpreter.
- `data/private`, `data/structure_library_v0.3` and `outputs/pretraining_v04` are
  junctions into `../restored_v04/phospholipid-structure-pretrain`. Preserve that
  verified restore tree and the original transfer archive.
- The user's 2026-10-20 FP stable plan now authorizes a separate, randomly
  initialized 50k / at most 5,000-step fingerprint experiment, with CPU and
  native CUDA backward/resume gates and staged 1,000 / 3,000 / 5,000 decisions.
  Use a new output root; never train into the restored historical directories.
  The two frozen v0.4 3,000-step checkpoints remain protected comparison inputs.
  Preserve GINE, Morgan, Formula Encoder, Template Set Transformer, TCLS,
  tokenizer, internal `[*:1]`, Builder, data splits and retrieval caches.
  New labels, CL, full-library training and MS/MS work still require new direction.
- Native CUDA work requires this interpreter's independent CPU gate and
  the matching supervisor: `scripts/supervise_formula_beam_cuda_v04.py` for
  historical inference, `scripts/supervise_polar_fp50k.py` for FP50k stages.
  Keep Math Attention and FP32,
  with TF32 disabled. Never silently fall back to CPU for a requested GPU run.
- Preserve AMD stop latches and history. NVIDIA uses a separate experiment root
  and persistent monitoring epoch. Never delete a stop latch or automatically retry
  a failed, timed-out or GPU-faulted job.
- Report monitoring access limits explicitly. This ordinary Windows process cannot
  read the protected WATCHDOG dump directory; System 4101, WER GPU timeout events,
  accessible dump directories and NVIDIA telemetry remain monitored.
- Use the frozen `diagnostic_512.npy`. Keep raw and filtered results separate.
  Label subsets with their actual target counts; never combine the AMD interrupted
  prefix with NVIDIA results or count overlapping slices twice.
- Keep private structures, predictions and weights out of Git. Publish source,
  configuration, aggregate JSON/Markdown and a commit SHA after validation.

See `docs/experiments/POLAR_FORMULA_BEAM_CUDA_V04.md` for commands and layout.
The repaired continuation uses `outputs/formula_constrained_v04_rtx4070_20261020_r1`;
the earlier native root retains the CPU-reproduced validation exception and stop latch.
Completed native jobs are reused only through the immutable 69,190-candidate CPU audit.

The authorized FP50k run is now complete at 5,000 steps, with all five checkpoints,
Greedy/Beam20/Beam50 on diagnostic512 and aggregate reports under
`docs/results/POLAR_FP_BASE_50K_RESULTS.*`. Private outputs are in
`outputs/polar_fp_base_50k_20261020/`; do not overwrite or resume beyond the cap.
Further pool expansion, controls or MS/MS training require new user direction.

The subsequent similarity-only evaluation uses
`outputs/polar_similarity_eval_v04_20261020/`. It reuses five cases and adds only
the missing 50k/3,000 Beam50 job under the original FP50k supervisor. No training.
Private Figure D, target metrics, templates and predictions must stay on G drive;
only aggregate Figure A/B/C may be published. Storage cleanup is limited to the
hashed dry-run of inactive Python/test caches and the verified incomplete pip
wheel; all checkpoints, prediction inputs, restore/transfer assets and fault
records remain protected. See `docs/experiments/POLAR_PRETRAIN_SIMILARITY_EVAL_V04.md`.

The reported-headgroups v0.1 collection is an independent, CPU-only reference
data task in `outputs/reported_headgroups_v01_20261020/`. Preserve original
source structures and unresolved records; distinguish verified literature from
database/rule-only evidence. Do not infer an attachment from the first P atom,
enumerate hypothetical heads, alter frozen assets or train any model. Complete
CSVs, workbook, source downloads and manual literature curation stay on G drive.
Only collection/validation/import code and non-sensitive aggregate reports may
be published. Import additions into a new directory, retaining prior provenance.

The 2026-10-21 single-anchor review is a derived, CPU-only selection in
`outputs/reported_headgroups_single_anchor_review_20261021/`. Preserve v0.1 and
all source graphs. Exclude obvious extra fatty-acyl/ether chains, multi-lipid
anchors and sterol anchors from the default model selection; keep short-acyl,
amino-acyl and uncertain cases visible for human review. Structure images,
HTML, workbook and per-graph decisions remain private on G drive. Do not train
or automatically write manual decisions into a formal template library.

The 2026-10-21 v0.2 expansion explicitly admits structurally defined PubChem
and other database glycerophospholipids without upgrading unverified literature
status. The user's imported review retains 52 graphs and rejects one; preserve
all decisions and notes. The expansion library and separate visual review are
in `outputs/reported_headgroups_v02_20261021/` and
`outputs/reported_headgroups_v02_review_20261021/`. Keep the v0.1 library and
review intact. Default review shows only new candidates; explicit extra lipid
anchors, including short-chain second diacylated glycerol arms, stay excluded.
Special phosphorus bonds and uncertain attachment positions require review,
not graph repair or guessed structures. This work does not authorize training,
spectra generation or automatic promotion into the formal template library.
Publish only processing code and aggregate reports, never full structures,
source caches or per-graph human decisions.

The 2026-10-22 mass/formula/length assessment is read-only in
`outputs/polar_size_scope_audit_20261022/`. Its 350/450/500/600 Da profiles and
the suggested 500 Da, C25, P-distance15 scope are proposals, not an authorized
replacement for the frozen library, splits or model labels. Keep every frozen
graph and the complete known-headgroup reference. Do not apply a selection or
train from this assessment without subsequent user direction. The v0.2 review
can be opened directly in a system browser from its unchanged index.html;
loopback port 8880 is only a temporary convenience and may stop after a restart.

The 2026-10-22 template proposal integrates the user's review under
`../头基预测人工审核`: 135 keep, 3 exclude, 2 review; preserve snapshots and notes.
One approved phosphate-protonation duplicate gives 134 conceptual seed templates.
Per the user's subsequent correction, choose supplementary templates from the
complete 195,306 pool regardless of historical splits; preserve the split files.
Private nested 200/250/300/350 proposals and full-core coverage are in
`outputs/headgroup_template_proposal_20261022/`. All additions remain proposals
pending human review; no automatic promotion or training. The 600 Da bound is for
new candidates, not grounds to delete larger known reference heads.
Distinguish template feature entry checks (127 approved records pass) from
Decoder target labels (120 pass the current vocabulary/84-token limit). Long
glycan heads may be valid fingerprint template inputs despite longer labels.
Any future training needs an independent fixed-template cache and evaluation
design; the old split results are not an independent test of this new bank.
See `docs/results/HEADGROUP_TEMPLATE_PROPOSAL_20261022.md`.

The user's subsequent 2026-10-22 V1.0 plan and direct amendments supersede the
earlier provisional 250-template recommendation as the formal training design.
The final first-version model is structure-template/formula conditioned Polar
generation; structure training itself is final, with no required MS/MS supervised
finetuning. Keep current architecture/optimization defaults; do not follow the
attachment's broad neural hyperparameter search or schedule PGCN casework.
Use 5,000 as the main structural training template pool, with 500/2,000/5,000/
10,000/20,000 pool-size and K1/3/5/10 experiments plus coverage and generation
evaluation. The 134-known/200-350 proposals remain curation seeds and diagnostics.
Actual later spectral application templates must have traceable spectra and
explicit structures from MS-DIAL, LipidGate or equivalent known sources; a
synthetic training template without spectra is not an actual spectral hit.
Keep the training reference pool, spectral reference pool and Restricted output
whitelist distinct. Restricted and Free share a generator; neither external
spectra independence nor closed-catalog accuracy alone proves unseen-head
generation. Formal held-out target cores must not be training labels or equivalent
input templates; preserve historical splits and audit any warm-start's seen graphs.
This revision records design only; no 5,000 pool or new training was completed.
See `docs/experiments/TEMPLATE_GUIDED_PHOSPHOLIPID_GENERATION_V1.md` and
`configs/template_guided_generation_v1_plan.json` for the current contract.

The user's latest authorization now starts the independent fixed-5,000-template
structure-conditioned training, initially K3/seed42, after real CPU/CUDA backward
and checkpoint-resume gates. Prefer the paper's main-experiment batch32 after a
bounded batch16/32 throughput and longest-sequence check. Keep the existing
network, AdamW1e-4/weight_decay0.01, FP32/Math/TF32-off configuration; these are
local settings, not a claim to reproduce the paper's optimizer parameters.
Select the best checkpoint ONLY by minimum full-validation token cross entropy;
early stop after five consecutive complete epochs without improvement, as in
MetGenX (doi:10.1038/s41467-026-72149-6). Generation validity, Similarity@10,
functional groups and gains over templates remain evaluation, not selectors.
The new local budget is at most 50 epochs / 300,000 optimizer steps / 48 hours;
50 is a local safety cap, not the paper's epoch setting. New data are independently
derived from the unchanged frozen graph-level split, excluding extra lipid
anchors, targets over the current label limit and self/equivalent references.
The initial `outputs/template_guided_fp5000_20261022` GPU benchmark exited at
the first loss due to a newly enabled strict deterministic flag unsupported by
torch2.9 CUDA nll_loss2d. No driver reset/new dump was detected; 12 old checkpoint
hashes were unchanged. Preserve this root, its data and GPU_STOP.json.
The repaired runtime restores historical non-strict FP32 settings. Its independent
CPU backward/exact-restart gate passed in `outputs/template_guided_fp5000_20261022_r1`,
whose data junction points to the original verified preparation. On 2026-10-23
the user explicitly authorized continuing in r1; the exact reply is retained in
GPU_RESTART_AUTHORIZATION.json. The supervised batch16/32 benchmark and real
CUDA backward/restart gates both passed. Batch32 was selected (about 1,080
targets/s versus 644 for batch16; longest-sequence backward passed). The main
fixed5000/K3/seed42 training has now been launched in
`outputs/template_guided_fp5000_20261022_r1/train_main_k3_seed42` under the matching
`scripts/supervise_template_guided_fp5000.py`. A latched fault forbids automatic
retries. Preserve every old checkpoint and monitor history. Other template/K
ablations and final generation evaluation remain distinct subsequent jobs.

The r1 main run completed normally on 2026-10-23 at 04:05 Shanghai, reaching
the authorized 50-epoch cap: 267,950 steps, best epoch50 validation token CE
0.1383161811 on all 9,527 validation cores. Patience5 early stopping was NOT
triggered, and reaching the local cap is not evidence of convergence. The best
and last checkpoint are step_267950_epoch_050.pt; its SHA matches the pointer
and minimum of all 50 validation epochs. Final host monitor exit0, no observed
display resets/new dumps and 12 historical checkpoint hashes unchanged;
WATCHDOG access remains unavailable. Do not extend the epoch cap or restart
training automatically. Generation/Builder/formula validity, Similarity@10,
functional-group evaluation and other pool/K/seed experiments remain pending.
The detailed CPU-only training report, all 50 validation epochs, weight SHA,
runtime statistics and aggregate PNG/SVG figures are in
`docs/results/TEMPLATE_GUIDED_FP5000_RESULTS_20261023.{md,json}` and
`docs/results/figures/template_guided_fp5000_20261023/`. Regenerate with
`scripts/report_template_guided_fp5000_results.py`; no GPU job is started.
Teacher-forced token accuracy95.3943% is not a molecular generation accuracy;
the still-unmeasured generation metrics are explicitly null in the report.

The user supplied a formal V2.0 plan on 2026-10-23. It authorizes independent
random-initialized E1/E2/E3 optimization experiments in
`outputs/template_guided_v2_20261023`, with frozen190553 eligible cores,
TRAIN171478/validation9527/test9548, bank5000/K3/batch32 and current architecture.
The new resource cap is80 epochs/428720 optimizer updates; select only minimum
complete-validation Token CE, patience5 complete epochs. E1 constant1e-4;
E2 warmup4000 to1e-4 then cosine1e-5; E3 warmup4000 to3e-4 then cosine1e-5.
This supersedes the old50-epoch cap only for the independent V2 experiment.
Preserve ALL exploratory files/weights/logs/code and all old fault latches.
New V2 data use separate exact-retrieval caches and TRAIN-only formula statistics;
source inputs, labels, bank identity, Builder and frozen architecture remain unchanged.
Independent A/B/C audit and CPU/CUDA full-state resume gates must pass before D.
Use only `scripts/supervise_template_guided_v2.py` for V2 CUDA, no automatic retry.
New retention keeps best plus latest2 full checkpoints, at most3 distinct files/run,
only prunes registered V2 obsolete checkpoints after atomic save/hash/CPU reload.
Immutable input hardlinks must never be overwritten. Stream evaluation and retain
all failures/empty targets in compressed raw records and metric denominators.
Private structures, raw predictions, IDs and weights remain on G. Public source,
aggregate JSON/Markdown and distribution figures may be pushed to GitHub.
Formula-only and extra seeds are conditional later stages after template stability
and generation assessment; never use an empty template mask as Formula-only.
No real-MS/MS or known-headgroup adaptation. See
`docs/experiments/TEMPLATE_GUIDED_FORMAL_V2.md` and `configs/template_guided_v2.json`.

V2 A/B/C,72 related tests, independent real-model CPU and CUDA backward/resume
gates have now passed. The serial supervised queue has started E1 from fresh
random initialization; E2/E3 and fixed generation evaluation are queued. The
first complete9527-core validation is CE0.4089735984 (epoch1); this is an early
training observation, not final performance. The CUDA gate observed no driver
fault/new dump, and all320 historical exploratory/probe checkpoint inventory
entries plus protected weight SHA checks remained unchanged. Compact retention
is active in the V2 root. Read `state.json`/run events for current status; never
launch a duplicate queue. `scripts/report_template_guided_v2.py` exports only
aggregate status and figures without changing frozen training sources or inputs.

The user's 2026-10-23 next-library task is CPU-only and independent in
`outputs/glycerol_core_collection_v01_20261023`. It expands to glycerol+head
cores, including nonphosphorus glyceroglycolipids and betaine lipids. Keep one
ID per connectivity graph; do not enumerate or separately number stereoisomers.
Original source stereo stays in provenance. Preserve named attachment positions;
stereo-free inositol graph symmetry can merge numbered positions, which remain
explicitly pending rather than silently becoming one position-resolved template.
Mass is neutral H-capped deacylated glycerol+head: sugar OR inositol <=600 Da,
other <=400 Da. Exclude extra lipid chains/anchors from default proposals;
hold unverified halogens and nonneutral structures, retain all source graphs.
Display O=P from the explicitly observed attachment-associated P only; never
select the first P in canonical SMILES or rewrite P=S as P=O. Natural core
literature, chain-bearing database structures, source rules and rule extensions
are separate evidence. A natural core does not verify every acyl composition.
The original195306 library receives only a derived sidecar. Never change active
V2 sources, labels, data splits, tokenizer, bank or weights for this future scope.
Private structure tables, galleries and Excel remain on G; publish source and
aggregate reports only. No automatic new template-bank selection or training.

The first future-core collection now contains352 connectivity graphs:130 pass,
216 excluded proposals and6 pending (including a source-numbered inositol
position ambiguity). Source records and all prior decisions remain private.
The complete195306-row sidecar passes66252 under this proposed400/600 Da,
extra-chain/halogen/protecting-group policy; no filter was applied to training.
The original library SHA and all73 V2 source/config hashes were unchanged.
See `docs/results/GLYCEROL_CORE_COLLECTION_V01_20261023.md`.
At the collection check, E2's training/supervisor PIDs were absent and the last
progress was2026-10-23 14:21:57, step36300/epoch7; its running state file is stale.
No automatic GPU restart was performed. Inspect the current process/monitor and
checkpoint state before claiming the queue is running or proposing continuation.

On2026-10-26 the user explicitly authorized restarting/continuing the interrupted
V2 queue. CPU loading verified the registered E2 step35000 checkpoint, full
optimizer/scheduler/RNG/cursor state and all73 frozen source/config hashes.
No new fault/dump or root stop latch was seen. The one-shot continuation is
`scripts/resume_template_guided_v2_20261026.py`, through the unchanged V2
supervisor, in `E2_seed42_resume_20261026`; then original E3 and evaluation.
The old E2 run/checkpoints and all earlier latches stay intact. E1 had already
completed normally at epoch56/patience5 and is reused. Read current state/events;
the historical process-absence note above does not describe this resumed run.

The user's FINAL2026-10-26 future-library mass rule supersedes the old400 Da
proposal: ordinary glycerol+head cores500 Da regardless of P; sugar and the
previously explicitly approved inositol exception600 Da. Do not change it
silently. Original195306 background and current V2 inputs remain untouched.
Independent proposed targets are in `outputs/headgroup_training_targets_v02_20261026`:
8351 anchored stereo-free graphs, comprising132 reference heads,19 additions
from the user's40-structure amino-acid atlas and8200 one-edit rule variants.
The initial8779 screening was refined to reject short-acyl BMP/CL/Acyl-PG
anchors and fatty NAPE/NAPS as well; preserve alpha-aminoacyl ester/peptide
exceptions. Keep all source records. Do not count background,5000 template
pool or external fatty-chain combinations in the8351 proposed target total.
P-direct heads serialize from the actually glycerol-connected P; otherwise
from the head-bearing glycerol carbon. Keep source stereo only as provenance.
All proposed targets have H-inclusive reference-chain assembly verification;
the background received only an independent audit/P-first sidecar, not edits.
New variants are candidates, not automatically reported natural molecules.
The new library has NOT been inserted into the resumed Polar training, and
new whole-core labels/tokenizer/splits have not been frozen for a new run.
See `docs/results/HEADGROUP_TRAINING_TARGETS_V02_20261026.md`. Complete CSV,
Parquet, compressed derivations, offline SVG review and Excel stay on G.

The user has accepted fixing the current8351-candidate scope. The full combined
future pool is now frozen separately in
`outputs/glycerol_core_training_pool_v02_20261026`:99192 anchored stereo-free
cores,97022 with P and2170 without P. This is91335 old-scope-pass graphs plus8351
expansion graphs minus494 exact attachment-aware duplicates. All99192 serialized
graphs pass rooted/Polar roundtrip and neutral full-assembly H formula checks.
Of original195306 records,77656 exceed the new mass/extra-chain/anchor scope and
26315 additional records are held only for chemical/evidence review;103971 total
do not enter the new default pool. Original records and hashes remain intact.
Before filtering the raw deduplicated union is203163. Current V2 still uses
TRAIN171478/validation9527/test9548, all P; do not confuse this with the new pool,
which has no new splits/tokenizer/length contract and has not started training.
Freeze hashes and source-ID mapping are private. Do not enlarge the accepted
8351-candidate expansion automatically. See
`docs/results/TRAINING_POOL_COUNTS_V02_20261026.md` and reproducible CPU script
`scripts/freeze_combined_core_pool_v02.py`. Syntax/assembly validity does not
prove natural occurrence, stability or every upstream attachment assignment.

The user's subsequent v0.3 direction expands only database-backed glycerol/head
substructures, not additional one-edit enumeration. Keep the old8351 and195306
snapshots. Allow the existing C/H/N/O/P/S/F/Cl/Br/I vocabulary, with no halogen
atom-count cap; the limit is on target elements, not clipped source groups or
counterions. The new PubChem branch excludes P sources and retains terminal
sn1/sn3 head attachment only; central sn2 candidates remain in exclusion data.
Without stereo, use terminal-role identity, retaining original source naming.
The declared acyclic glycerol3O search retrieved833819 CIDs, of which353646 P
sources were removed; all480173 nonP property records are cached on G in
`outputs/glycerol_core_collection_v03_20261026/raw/nonP_properties`.
Other glycerol O substituents may be explicitly cut to obtain observed terminal
head substructures, even when they are not fatty chains. Preserve the entire
chosen head graph, source cut types/atom indices and graph/H formula roundtrip.
Distinguish these fragments from whole observed deacylated cores and natural
lipids. Never erase a head's actual fatty chain to make it pass. Crosslinks stay
pending. Connected glycerol-bearing source components may be used, preserving
the parent multicomponent record and exact component used for the source proof.
All195306 old core masses were independently recomputed:68050 exceed the current
500/600 bounds;43100 of these had Polar alone within the corresponding bound.
The user explicitly requests keeping those43100 in a separate standby table,
not adding them to the main pool. Old ring paths were wrongly counted as long
fatty chains in6298 flags; use acyclic carbon paths instead. Allowing halogens
and correcting this recovers22281 old cores (18954 halogen-only,3036 ring-only,
291 both), giving113616 old pass /74232 scope-excluded /7458 other holds.
The new final pool must include the target-only element/source-component recheck
before publication; `supplement_target_elements_v03.py` verifies and merges that
transitional pass idempotently. Current V2 training remains unchanged; read its
state rather than starting another queue. No new-library training is authorized
by this data collection. All complete structures/provenance/caches stay on G;
publish processing code, aggregate counts and reports only.

The v0.3 collection and target-element/component recheck are complete. Main
candidate pool:144797 unique role graphs,119303 P and25494 nonP (23324 net new
nonP over the prior2170). Separate43100 standby graphs are disjoint, giving
187897 distinct main-plus-standby graphs; standby is not in the training pool.
All480173 source records were processed. The full202801 provenance edges refer
to165816 admitted source CIDs and valid target IDs. Of the nonP graph set,23373
have this PubChem-substructure evidence,10275 have an uncut whole-core source,
and13016 have an observation without clipping other nonfatty glycerol groups.
Other candidates remain explicit source-substructure derivations, not reported
natural lipid structures. All retained graphs have root/assembly/H formula proof;
43 related tests pass. Original library,8351 snapshot and all73 protected V2
source/config hashes are unchanged. V2 has progressed to E3 in its original queue;
no new-pool training was started. See
`docs/results/GLYCEROL_SUBSTRUCTURE_V03_20261026.{md,json}` and private root
`outputs/glycerol_core_collection_v03_20261026`. The offline index is a clearly
labelled small visual sample, not the complete graph count; full data are in the
combined Parquet and provenance mapping. Do not silently change mass/element/
terminal-position rules or include the standby set in subsequent training.

The user's latest2026-10-26 admission decision supersedes the v0.3 exclusions for
37609 mass-only standby graphs and7458 risk-only graphs (4990 protecting-group
only,2358 acyl-phosphate only,110 both). Preserve all original atoms and flags;
these are accepted potential candidates, not newly verified natural lipids.
The authoritative unified snapshot is now
`outputs/glycerol_core_unified_v04_20261026/main_core_library_v04.parquet`:
189864 main cores,164370 P and25494 nonP. The exact25494 nonP ID set is frozen;
do not automatically enumerate or search more. The remaining5491 mass-plus-other
graphs stay in standby, and31132 others remain excluded/review; all three tables
share the same glycerol-plus-head schema and disjoint HT IDs (226487 total).
Every target_smiles includes the fixed glycerol with chain roles101/102.
Direct-P heads start at the actual glycerol-connected P; others at the head-side
glycerol carbon. Legacy Polar fragments are separately named and are not the
new whole-core labels. Ordinary500/sugar-or-inositol600 remain nominal bounds;
only the explicitly restored37609 IDs are approved mass exceptions. Do not
silently reapply old v0.3 hard filters to remove these accepted main records.
All226487 root/port/graph checks and45067 new H-inclusive reassemblies passed,
with prior unchanged proofs reused.48 related tests pass. Original195306,
v0.3 and8351 snapshots and all73 frozen V2 source/config hashes remain unchanged.

User-authorized hashed cleanup removed567 rebuildable cache files,3.047GB,
including the cached Torch installer and pip downloads, project pyc/test/lint
caches and superseded CPU intermediate files. The installed2.9.1+cu128 environment
is intact; the installer version/SHA and recovery constraints are in the new
cache cleanup receipt. Earlier storage-audit notes about retaining that cached
wheel are historical; its latest authorized deletion is not missing runtime or
lost checkpoint data. Keep all original recovery/transfer trees, checkpoints,
predictions, source records, manual reviews and stop/monitor evidence.

E1/E2/E3 training finished normally. The subsequent validation pilot stopped on
an unhandled RDKit KekulizeException in generated-candidate metrics, with the
supervisor failed and GPU_STOP.json retained; state.json's running value is
stale. This data curation did not restart the GPU evaluation or modify protected
generation code. No new-pool training was started. See
`docs/results/GLYCEROL_CORE_UNIFIED_V04_20261026.{md,json}` and the new manifest,
schema/report and cache cleanup receipts before continuing.

The user's subsequent2026-10-26 instruction now explicitly authorizes NEW-library
formal training with about8000 templates and a new target split, including all
8000 bank entries as prediction labels. This supersedes the earlier data-only
restriction for this independent experiment. Use
`configs/glycerol_core_templates8000_v1.json` and
`outputs/glycerol_core_templates8000_v1_20261026`; preserve all historical roots,
stop locks, weights and73 frozen V2 sources. Labels include glycerol and distinct
chain ports101/102, with neutral H-capped whole-core formulas, both P and nonP.
Train a freshly initialized output layer/model; do not pretend compatibility
with old Polar labels. Max label length96 covers all189864 main graphs; retain
approved mass/risk exceptions. Split by stereo-free neutral capped equivalence
before TRAIN-only bank selection. Keep all8000 bank entries as TRAIN targets,
excluding self/equivalent input templates for every prediction. Report their
leave-one-out reconstruction separately from held-out validation/test.
New CPU/CUDA backward, actual full-state disk recovery, longest-label and nonP
gates must pass under the new matching supervisor before formal training.
Reuse the prior best E3 optimization schedule: batch32/K3, warmup4000 to3e-4,
cosine to1e-5, FP32/Math/TF32off,80-epoch cap, minimum complete-validation Token
CE selection and patience5. The old stop lock is not cleared or retried.
New retention is best plus latest2, at most3 full checkpoints/run and8GiB owned
output cap. Predeclared validation512/bank-target512/test512 generation is
bounded, streams raw candidates, catches RDKit exceptions and retains failed
targets in denominators. No automatic extra bank/K/seed ablations, MS/MS, CL
or PGCN. See `docs/experiments/GLYCEROL_CORE_TEMPLATES8000_V1.md`.

New grouped split preparation,31 tests, CPU exact restart and CUDA real-state
restart gates have passed. Longest94-token and TRAIN nonP backward batches
passed; batch32 measured1541 targets/s vs908 for16 in the bounded gate.
The supervised queue is now running `train_main_k3_seed42` from fresh random
initialization. TRAIN170872/validation9493/test9499, bank6244 P+1756 nonP;
the complete first validation CE is0.3086855581 on9493 targets. All8000 bank
entries occur exactly once per training epoch, with self/equivalent inputs
masked. This is early training, not generated-molecule accuracy. Inspect actual
processes and events before any status claim; never launch a duplicate queue.
The new root retains all new gates and old-stop-lock/73-source preservation
proofs. WATCHDOG remains unreadable; accessible dumps/events/telemetry are
monitored. The startup aggregate report is
`docs/results/GLYCEROL_CORE_TEMPLATES8000_V1_20261026.{md,json}`; regenerate with
`scripts/report_core_templates8000.py` without modifying frozen active sources.

On2026-10-27 the user requests publishing BOTH completed training rounds.
The new8,000-template run and all three fixed512 generation jobs completed
normally on2026-10-26. Training stopped at48 epochs/256320 steps by patience5;
best epoch43/step229620, full-validation CE0.09594137142809196, weight SHA256
f8d7f646f6bfda7459296f5025adc78a9695b8b8bff4c34e8ae80c6d779e17c2.
Test512 raw/filtered exact@10 are0.642578125/0.69921875. This remains an oracle
structure-conditioned subset evaluation, not full9499-test or real-MS/MS accuracy.
Bank-target512 is TRAIN-label leave-one-out reconstruction, not independent test.
Do not restart a queue based on earlier running notes. Old E1/E2/E3 training
reports are complete (56/64/64 epochs); E3 has best CE0.12368985641566978.
Its old generation remains incomplete with its RDKit stop lock preserved;
do not invent completed generation metrics or rerun it for a publication task.
CPU-only publication and raw-aggregate replay use
`scripts/publish_training_rounds_20261027.py`, outside both frozen runtime
contracts. Final aggregate reports/JSON/PNG/SVG are under `docs/results/`,
indexed by `TRAINING_ROUNDS_FINAL_20261027.md`. Keep all structures, IDs, weights,
raw predictions and source caches on G; only aggregates/curves/code go to GitHub.

The user's latest2026-10-27 request authorizes full9499 test generation at the
frozen epoch43 checkpoint and a capacity-matched independent Formula-only
training/control. The user explicitly prioritizes fixing missing online formula
decoding: previous512 results were raw+posterior filters, not online constraints.
Use `configs/core_formula_comparison_20261027.json`, independent root
`outputs/core_formula_comparison_20261027`, and `core_ablation` modules/scripts.
Historical73-source and8000-model contracts and all old stop locks stay intact.
Whole-core online heavy-atom/port budgets and conservative SMILES grammar support
P and nonP, with final H-inclusive RDKit/assembly checks. Do not force EOS,
renormalize feasible probabilities, repair graphs or change formulas. All9499
gold labels/482902 prefixes pass reachability with zero legal targets pruned;
47 tests and a real CPU full-state restart gate have passed. Matching supervised
CUDA gate is required before the serial full-test/control queue.
Formula-only receives only target formulas, with no reference fingerprint,
reference formula/delta/rank/mask or graph-derived query feature. Same parameter
names/shapes/count/seeded initialization, labels/splits, batch32,80-epoch/427200
cap/E3 schedule/patience5; TRAIN target-formula statistics only. Raw/elements/
elements+grammar each use Beam20/length96 on the same9499 (8225P+1274nonP), with
posterior filters separate and missing targets/beam slots retained in denominators.
Epoch43 template weights are fixed. Formula-only checkpoint selection uses all
9493 validation targets only. Gains use a common structural baseline in metrics
after decoding; it is not passed to Formula-only. No real-MS/MS performance claim.
Use only `scripts/supervise_core_ablation.py`; no automatic retry of a latched job.
See `docs/experiments/CORE_FORMULA_COMPARISON_20261027.md` and aggregate
`docs/results/CORE_FORMULA_COMPARISON_20261027.{md,json}` for current status.

The new supervised CUDA backward/full-state restart and real epoch43 decoding
gates have now passed. Both models have16479940 trainable parameters; identical
seeded parameter tensors/shapes and no Formula-only reference-array access were
tested. CUDA model/optimizer/scheduler/RNG restart max difference0, longest94
and TRAIN nonP backward passed. Batched/serial BOS logits differ at most4.77e-6;
all40 beams from the first2 probe targets match in identity, Top1 agreement1.0,
shared logprob max difference1.27e-5. Raw/elements/grammar actual CUDA decoding
probes passed, no observed driver reset/new dump, WATCHDOG remains unreadable.
The serial queue is authorized to begin in this independent root with template
full elements+grammar first, then raw/elements, Formula-only training and matched
full-test modes. Never duplicate a started queue or mutate its numerical source
contract. The CPU-only reporter is intentionally excluded from that frozen
contract so presentation/paired-analysis corrections can be made without
changing training or generation. Read current PIDs/events/state for actual stage.

The queue is now launched in the independent control root; read its live stage
instead of treating these startup notes as completion. The CPU reporter can
update progress while numerical files remain frozen. After all six full-test
cohorts and Formula-only training finish, it automatically invokes the separate
`scripts/publish_formula_comparison_results.py` one-shot publisher. This only
stages the two allowed aggregate Markdown/JSON files, refuses other staged work,
verifies all stream/checkpoint/source hashes and full9499/P/nonP counts, and
checks the pushed remote SHA. `PUBLICATION_DONE.json` is the success receipt;
`PUBLICATION_PENDING.json` means numerical results are retained but upload needs
attention. No GPU auto-retry is permitted. Primary inference compares constrained
posterior Exact@10 with paired target-formula-cluster intervals, fixed seed42;
do not interpret it as real-MS/MS evidence or across-training-seed stability.
