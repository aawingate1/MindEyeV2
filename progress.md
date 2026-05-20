## 2026-05-19 18:16 EDT - Cycle 3 Diagnostic-Gated Sparse Adaptation

### Plan executed
- Read `/plan.md`; ignored older strategy notes as instructed.
- Located active training entrypoint at `/src/Train.py`; existing Slurm logs are under `/src/slurms` rather than `/slurms`.
- Added default-off config flags to `Train.py`:
  - `--reliability_mode {none,soft_weight,topk_mask}` default `none`
  - `--reliability_topk` default `0`
  - `--adapter_prior_weight` default `0.0`
  - `--adapter_prior_type {weights,outputs}` default `weights`
- Implemented reliability-aware voxel inputs. No NCSNR/reliability metadata was present in `/src`; only `brain_region_masks.hdf5` region masks were available. The implementation therefore logs and uses a fallback voxelwise training-beta standard-deviation proxy when reliability mode is enabled. `none` preserves baseline inputs.
- Implemented adapter drift regularization on the subject ridge adapter/projection. `weights` snapshots initialized ridge parameters after optional checkpoint loading and penalizes L2 drift; `outputs` snapshots a frozen ridge copy and penalizes output drift on the same fMRI batch.
- Added per-run reliability summaries at `../train_logs/<model_name>/reliability_summary.json` when checkpoint saving is enabled, and added `train/loss_adapter_prior` to epoch logs.
- Added Slurm scripts:
  - `/src/accel_cycle3_smoke.slurm`: 1-hour smoke, `num_sessions=1`, `num_epochs=2`, `topk_mask=8000`, `adapter_prior_weight=1e-3`.
  - `/src/accel_cycle3_grid.slurm`: primary array for baseline, reliability-only, adapter prior weights `1e-4/1e-3/1e-2`, and combined top-8k + `1e-3`.

### Verification
- Ran syntax check: `/src/fmri/bin/python -m py_compile /src/Train.py` succeeded.
- Submitted required pre-grid 1-hour Slurm smoke before any long run:
  - Command: `sbatch /src/accel_cycle3_smoke.slurm`
  - Initial job ID `8476878` failed before Python startup with `RaisedSignal:53` and no stdout/stderr; Slurm compute nodes do not use `/src` as the working/log path.
  - Patched Cycle 3 Slurm scripts to use the cluster scratch path used by the existing working `/src/accel.slurm`: `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`.
  - Resubmitted smoke as job ID `8476900`.
  - Final state: completed in `00:04:46`.
  - Logs confirm Python startup, `reliability_mode=topk_mask`, `active_voxels=8000`, saved `reliability_summary.json`, and initialized adapter prior `type=weights`, `weight=0.001`.
  - Final smoke metrics after 2 epochs / 1 session: `test/loss=4.62`, `test_fwd=0.113`, `test_bwd=0.0367`, `train/loss=1.16`, `train_fwd=0.845`, `train_bwd=0.671`, `train/loss_adapter_prior=9.89e-8`.
- After smoke passed, reduced primary grid walltime from 20h to 6h based on observed 1-session smoke runtime and submitted the primary array:
  - Command: `sbatch /src/accel_cycle3_grid.slurm`
  - Job ID: `8477001`
  - Array tasks `0-5` pending on `gpu` at last check.

### Existing log context
- Parsed recent `/src/slurms/*.err` logs for final reported metrics. Best recent final values found were around:
  - `6906702.err`: `test/loss=0.0883`, `test_fwd=0.990`, `test_bwd=0.953`
  - `6906703.err`: `test/loss=0.0835`, `test_fwd=0.977`, `test_bwd=0.957`
  - `6906705.err`: `test/loss=0.110`, `test_fwd=0.967`, `test_bwd=0.963`
- Later logs were worse, e.g. `7345430.err`: `test/loss=3.03`, `test_fwd=0.283`, `test_bwd=0.323`.
- These are only log-derived reference points; exact config comparability remains unresolved until the Cycle 3 grid runs with matched settings.

### Conclusions and next questions
- The code now supports the Cycle 3 diagnostic grid without changing flags-off baseline behavior.
- Because true NCSNR metadata is absent, reliability-aware results must be interpreted as a beta-variance proxy test, not a true reliability/NCSNR test.
- Next step: monitor `/src/slurms/c3_grid_8477001_<task>.err` and `.out` once tasks start. Compare final test loss, image retrieval (`test_fwd`), and brain retrieval (`test_bwd`) across baseline, reliability-only, adapter-prior weights, and combined condition. If `1e-2` collapses early, stop/ignore larger prior strengths.

### Telegram report-ready update
Cycle 3 implementation is in place. Added default-off reliability-aware voxel input flags and adapter drift regularization to `/src/Train.py`, plus logging for reliability summaries and adapter-prior loss. No NCSNR metadata exists in `/src`, so reliability mode uses a clearly logged fallback voxelwise training-beta std proxy. Syntax check passed. First smoke job `8476878` failed before Python due Slurm `/src` workdir/log path visibility; scripts were patched to the existing scratch path. Resubmitted smoke `8476900` completed in 4:46 with reliability summary saved, adapter prior initialized, and nonzero `train/loss_adapter_prior=9.89e-8`; final smoke metrics after 2 epochs were `test/loss=4.62`, `test_fwd=0.113`, `test_bwd=0.0367`. Submitted the 6-task primary grid as job array `8477001` on `gpu` after reducing walltime to 6h. Recent best reference logs show roughly `test/loss=0.0835-0.110`, `test_fwd=0.967-0.990`, `test_bwd=0.953-0.963`; Cycle 3 grid is now queued to compare matched baseline, reliability-only, adapter-prior sweep, and combined condition.

## 2026-05-20 03:19 EDT - Cycle 3 Grid Monitoring and Report

### Plan executed
- Read `/plan.md` and continued the prescribed Cycle 3 monitoring path only.
- Checked Slurm state for array `8477001`. `squeue -j 8477001` now reports no active jobs because the array has left the queue.
- Checked accounting with `sacct -j 8477001 --format=JobID,JobName%32,State,ExitCode,Elapsed,Timelimit,MaxRSS,ReqMem -P`.
- Parsed `/src/slurms/c3_grid_8477001_<task>.out` and `.err` for run setup, final metrics, reliability summaries, and failures.

### Job states
- `8477001_0` through `8477001_5` all completed with `State=COMPLETED`, `ExitCode=0:0`.
- Elapsed times were `00:38:01` to `00:38:48` against the requested `06:00:00`.
- Batch-step MaxRSS was about `21.77 GB` for all tasks, below requested `64G`.
- No Python failures were found. Each `.err` starts with a Slurm warning that `/src` was not visible as the initial working directory and Slurm moved to `/tmp`; the script then explicitly `cd`s to the scratch source path and proceeds normally.

### Comparability checks
- All six tasks used subject 1, `num_sessions=1`, `num_epochs=150`, `batch_size=24`, `global_batch_size=24`, `seed=42`, `max_lr=3e-4`, `mixup_pct=0.33`, `no-use_prior`, `no-blurry_recon`, and `no-use_image_aug`.
- All six tasks used the same train shard `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds/subj01/train/{0..0}.tar` and test shard `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds/subj01/new_test/0.tar`.
- All six tasks used the same model size, `469,462,680` total and trainable parameters, and `total_steps=4650`.
- The baseline run had `reliability_mode=none`, `active_voxels=15724`, and `adapter_prior_weight=0.0`, matching the flags-off condition.
- Reliability-enabled runs reported `mode=topk_mask`, `source=voxelwise_train_beta_std_proxy_no_ncsnr_found`, `active_voxels=8000`, and saved `reliability_summary.json`. The summary save paths are under the scratch train-log path printed by the jobs; direct file lookup from the login workspace did not expose `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs`, but the `.out` logs confirm the JSON save calls and include the summary contents.
- Prior-enabled runs printed `Initialized adapter prior: type=weights` with the intended weights. `train/loss_adapter_prior` was nonzero in all prior-enabled final logs.

### Final matched metrics

| Condition | Task | Final test/loss | test_fwd | test_bwd | train/loss | train_fwd | train_bwd | train/loss_adapter_prior | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0 | 2.72 | 0.493 | 0.320 | 0.000208 | 1.000 | 1.000 | 0 | Full 15,724 voxels |
| topk_mask | 1 | 2.77 | 0.490 | 0.327 | 0.000232 | 1.000 | 1.000 | 0 | Beta-std proxy top 8k; slightly better bwd, worse loss/fwd |
| adapter_prior_1e-4 | 2 | 2.73 | 0.483 | 0.317 | 0.000207 | 1.000 | 1.000 | 4.25e-6 | Worse retrieval than baseline |
| adapter_prior_1e-3 | 3 | 2.72 | 0.483 | 0.320 | 0.000206 | 1.000 | 1.000 | 4.24e-6 | Matches loss/bwd, worse fwd |
| adapter_prior_1e-2 | 4 | 2.73 | 0.487 | 0.317 | 0.000206 | 1.000 | 1.000 | 4.24e-6 | No benefit; not a useful stronger prior |
| topk_mask_adapter_prior_1e-3 | 5 | 2.77 | 0.497 | 0.323 | 0.000232 | 1.000 | 1.000 | 2.93e-6 | Best fwd by 0.004, but worse loss and only +0.003 bwd |

### Conclusions
- All conditions fully memorized the one-session training set, with train retrieval at `1.0` and train loss around `2e-4`. The train-test gap remains large and is not meaningfully reduced by the tested adapter-prior weights.
- Adapter drift regularization did not meet the success criteria. `1e-4` and `1e-3` did not improve retrieval over the matched baseline; `1e-3` only tied baseline loss and backward retrieval while reducing forward retrieval.
- The beta-std top-k proxy did not produce a clean win. It improved backward retrieval from `0.320` to `0.327` but worsened loss from `2.72` to `2.77` and slightly worsened forward retrieval.
- The combined top-k plus `1e-3` prior produced the best forward retrieval, `0.497` vs baseline `0.493`, and slight backward gain, `0.323` vs `0.320`, but worsened loss to `2.77`. Treat this as noise-level evidence or a weak directional signal, not a robust improvement.
- None of the Cycle 3 matched runs approach the older log-derived reference metrics around `test/loss=0.0835-0.110`, `test_fwd=0.967-0.990`, `test_bwd=0.953-0.963`. Those older logs likely differ materially in configuration and remain secondary context only.

### Recommended next research questions
- Do not advance the current adapter-weight prior as a Cycle 4 anchor based on these matched results.
- Do not treat beta-std top-k as a reliability/NCSNR win. If voxel gating remains interesting, replace the proxy with true repeatability, NCSNR, or ROI-stratified stable voxel selection.
- The most important bottleneck is still sparse-adaptation overfitting/generalization. Per `/plan.md`, the next cycle should pivot toward explicit functional alignment or subject-conditioned/meta-learning rather than diffusion-prior tuning, decoder enlargement, Fourier augmentation, or blurred CLIP target schedules.
- Add drift-norm and maybe output-drift logging only if a future prior formulation shows a real retrieval or train-test-gap benefit.

### Telegram report-ready update
Cycle 3 grid `8477001` completed cleanly: all 6 tasks finished with exit code `0:0` in ~38 minutes, using ~21.8 GB RSS under the 64 GB request. Matched baseline was `test/loss=2.72`, `test_fwd=0.493`, `test_bwd=0.320`, with full train memorization (`train_fwd=train_bwd=1.0`, train loss ~`2e-4`). Adapter prior did not help: `1e-4`, `1e-3`, and `1e-2` all matched or worsened retrieval, despite nonzero `train/loss_adapter_prior` around `4.24e-6`. Beta-std top-8k masking is not a clean win: it improved `test_bwd` to `0.327` but worsened loss to `2.77` and slightly reduced `test_fwd` to `0.490`. Combined top-8k + `1e-3` prior had the best `test_fwd=0.497` and `test_bwd=0.323`, but loss stayed worse at `2.77`, so this is weak/noisy rather than decisive. Reliability summaries were logged and saved for all runs; top-k runs used the documented fallback `voxelwise_train_beta_std_proxy_no_ncsnr_found` with `active_voxels=8000`, not true NCSNR. Recommendation: do not advance the current adapter-weight prior; do not claim beta-std reliability success; next cycle should pivot toward explicit functional alignment or subject-conditioned/meta-learning. No plots were generated in this cycle.
