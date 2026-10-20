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
  `scripts/supervise_formula_beam_cuda_v04.py`. Keep Math Attention and FP32,
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
