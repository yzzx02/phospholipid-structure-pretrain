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
