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
