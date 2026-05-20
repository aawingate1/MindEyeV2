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

## 2026-05-20 04:31 EDT - Cycle 3 Capacity-Controlled Functional Alignment Execution

### Plan executed
- Read `/plan.md` and implemented the prescribed capacity-control path in `/src/Train.py`.
- Added `--train_scope {all,adapter_only,adapter_head}`. Default `all` preserves flags-off behavior. `adapter_only` freezes all shared modules and trains only `model.ridge`; `adapter_head` trains `model.ridge`, `backbone_linear`, and `clip_proj`.
- Added parameter logging for total and trainable parameters in stdout, W&B config, and epoch logs (`params/total`, `params/trainable`).
- Added output-space alignment regularization:
  - `--alignment_output_prior_weight`, default `0.0`.
  - `--alignment_output_prior_type {distill,moments}`, default `distill`.
  - The prior snapshots a frozen initialized ridge adapter and penalizes drift after the adapter, before the shared mapper. `distill` uses full-output MSE; `moments` matches mean, variance, and feature norm.
  - Logged `train/loss_alignment_output_prior`.
- Added optional deterministic cached train-shard validation with `--val_fraction`; default `0.0` keeps baseline behavior unchanged. When enabled, it builds a fixed unique image/voxel cache, logs `val/loss`, `val/fwd_pct_correct`, and `val/bwd_pct_correct`, and saves `best_val.pth` on improvement.
- Added optional validation patience via `--early_stop_patience`; default `0` disables early stopping.
- Added Slurm scripts:
  - `/src/accel_cycle4_smoke.slurm`: 1-hour adapter-only smoke with output prior and validation enabled.
  - `/src/accel_cycle4_grid.slurm`: six-task compact grid for baseline-all, adapter-only, adapter-only prior `1e-3`, adapter-only prior `1e-2`, adapter-head, and adapter-head with early stopping.

### Verification and jobs
- Syntax check passed: `/src/fmri/bin/python -m py_compile /src/Train.py`.
- First smoke submission `8494843` failed immediately with `State=FAILED`, `ExitCode=0:53`, `Elapsed=00:00:00`. This was a Slurm setup/log-path failure, not Python startup; no stdout/stderr logs were created.
- Patched Cycle 4 Slurm scripts to match the successful Cycle 3 pattern: absolute scratch log paths and `SRC_DIR=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`.
- Resubmitted required 1-hour smoke as job `8494926`.
  - Final state: `COMPLETED`, `ExitCode=0:0`, `Elapsed=00:03:19`, batch `MaxRSS=21626748K`, under requested `64G`.
  - Smoke confirmed new code on compute: `train_scope=adapter_only total_params=469,462,680 trainable_params=16,102,400`.
  - Validation cache built: `75` unique image/voxel train-shard pairs.
  - Alignment prior initialized: `type=distill`, `weight=0.001`.
  - `best_val` and `last` checkpoints were saved successfully.
  - Final 2-epoch smoke metrics: `test/loss=5.79`, `test_fwd=0.0233`, `test_bwd=0.00333`, `train/loss=2.34`, `train_fwd=0.801`, `train_bwd=0.406`, `val/loss=2.59`, `val_fwd=0.933`, `val_bwd=0.560`, `train/loss_alignment_output_prior=0.697`.
- After smoke passed, submitted the six-task primary grid:
  - Command: `sbatch /src/accel_cycle4_grid.slurm`
  - Job array: `8495064`
  - Current state at last check: tasks `0-5` pending for resources on `gpu`, each with `02:00:00` time limit and `64G` memory.

### Conclusions
- The capacity-control implementation is wired and smoke-tested. Adapter-only reduces trainable capacity by about `29x` (`469.5M` total to `16.1M` trainable).
- Output prior loss is nonzero and validation/checkpoint plumbing works.
- Smoke retrieval is expectedly weak after only 2 epochs, but train/val curves show the training path is active and not frozen accidentally.
- The cached validation set is a deterministic diagnostic from the train shard, not a true held-out test substitute; primary comparison remains final test loss/retrieval from the standard new-test evaluation.
- No plots were generated.

### Recommended next research questions
- When grid `8495064` completes, compare each condition against the Cycle 3 matched baseline `test/loss=2.72`, `test_fwd=0.493`, `test_bwd=0.320`.
- Prioritize whether reduced trainable scope closes the train-test gap, even if train retrieval no longer reaches `1.0`.
- If adapter-only underfits final train metrics, use adapter-head as the next capacity point before expanding to explicit MindAligner-style low-rank transfer.

### Telegram report-ready update
Implemented the `/plan.md` capacity-control cycle in `/src/Train.py`: added `--train_scope all|adapter_only|adapter_head`, output-space alignment prior flags, trainable/total parameter logging, cached train-shard validation, `best_val` checkpointing, and optional validation patience. Syntax check passed. First smoke `8494843` failed before Python with Slurm `0:53` due log/workdir path setup; patched scripts to the Cycle 3 absolute scratch-log pattern. Resubmitted required 1-hour smoke `8494926`, which completed cleanly in `00:03:19` with exit `0:0` and ~21.6 GB RSS. Smoke confirmed `adapter_only` reduced trainables to `16,102,400` out of `469,462,680` total, built a 75-sample validation cache, initialized `alignment_output_prior type=distill weight=1e-3`, logged nonzero `train/loss_alignment_output_prior=0.697`, and saved both `best_val` and `last` checkpoints. Two-epoch smoke metrics were `test/loss=5.79`, `test_fwd=0.0233`, `test_bwd=0.00333`, `train/loss=2.34`, `train_fwd=0.801`, `train_bwd=0.406`, `val/loss=2.59`, `val_fwd=0.933`, `val_bwd=0.560`. Submitted six-task grid `8495064` for baseline-all, adapter-only, adapter-only prior `1e-3`, adapter-only prior `1e-2`, adapter-head, and adapter-head early-stop; all tasks were pending for GPU resources at last check. No plots generated yet.

## 2026-05-20 05:35 EDT - Cycle 3 Capacity-Controlled Grid Results

### Job states
- Grid `8495064` completed cleanly for all six tasks with `ExitCode=0:0`.
- Elapsed times: task 0 `00:47:29`, task 1 `00:38:48`, task 2 `00:38:44`, task 3 `00:39:05`, task 4 `00:46:17`, task 5 `00:07:53`.
- Batch MaxRSS was about `21.77 GB` for all tasks, below requested `64G`.
- Task 5 (`adapter_head_earlystop`) stopped after 21 epochs with `best_val_loss=0.0440092459321022`.

### Final matched metrics

| Condition | Task | Trainable params | Final test/loss | test_fwd | test_bwd | train/loss | train_fwd | train_bwd | val/loss | val_fwd | val_bwd | align prior loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_all | 0 | 469,462,680 | 2.69 | 0.487 | 0.343 | 0.000202 | 1.000 | 1.000 | 0.0000591 | 1.000 | 1.000 | 0 |
| adapter_only | 1 | 16,102,400 | 5.77 | 0.060 | 0.00667 | 0.106 | 1.000 | 1.000 | 0.0977 | 1.000 | 1.000 | 0 |
| adapter_only_prior_1e-3 | 2 | 16,102,400 | 5.77 | 0.0533 | 0.00667 | 0.107 | 1.000 | 1.000 | 0.0957 | 1.000 | 1.000 | 2.37 |
| adapter_only_prior_1e-2 | 3 | 16,102,400 | 5.81 | 0.0267 | 0.00333 | 0.112 | 1.000 | 1.000 | 0.0846 | 1.000 | 1.000 | 1.34 |
| adapter_head | 4 | 461,057,664 | 2.66 | 0.470 | 0.373 | 0.000197 | 1.000 | 1.000 | 0.0000701 | 1.000 | 1.000 | 0 |
| adapter_head_earlystop final | 5 | 461,057,664 | 3.78 | 0.243 | 0.187 | 0.293 | 0.695 | 0.754 | 0.0598 | 1.000 | 0.987 | 0 |

### Best-validation checkpoint-time metrics
- Because validation is logged every epoch, the epoch where `best_val` was saved can be read from the logs even though no separate eval-only reload script exists yet.
- For tasks 0-4, best validation occurred at the final epoch, so checkpoint-time metrics match the final metrics above.
- For task 5, the best-validation epoch had `val/loss=0.044`, `test/loss=3.71`, `test_fwd=0.310`, `test_bwd=0.157`, `train/loss=0.333`, `train_fwd=0.707`, `train_bwd=0.761`.

### Conclusions
- Reduced train scope did not meet success criteria.
- `adapter_only` and both adapter-only output-prior runs badly underfit the shared mapping path for test retrieval: `test_fwd <= 0.060`, `test_bwd <= 0.00667`, and `test/loss >= 5.77`, despite reaching train retrieval of `1.0`.
- Output-space alignment prior did not rescue adapter-only. Stronger prior `1e-2` worsened both forward and backward retrieval.
- `adapter_head` produced the best backward retrieval in this grid: `test_bwd=0.373`, a `+0.053` gain over the Cycle 3 matched baseline `0.320`, and improved loss modestly from `2.72` to `2.66`; however, it reduced forward retrieval from `0.493` to `0.470`.
- The new baseline-all run with validation plumbing was close but not identical to the prior matched baseline: `test/loss=2.69`, `test_fwd=0.487`, `test_bwd=0.343`. This suggests validation/cache code did not break training, but exact comparability has small run-to-run or code-path drift.
- Cached train-shard validation saturates to near-perfect retrieval for non-earlystop runs, so it is not a useful early-stopping proxy in the current form. Task 5 stopped early and reduced overfitting but also had poor test retrieval.
- The strongest actionable signal is adapter-head improving `test_bwd` and loss while hurting `test_fwd`; this is not a clean win but is more promising than adapter-only.

### Recommended next research questions
- Do not advance adapter-only as-is.
- For adapter-head, run a focused sweep across smaller head scopes or learning rates to recover forward retrieval while retaining the backward/loss gain. A lower LR for `clip_proj`/`backbone_linear` relative to ridge is the most direct next test.
- Replace the cached validation proxy with a true held-out shard or a deterministic split that is excluded from training; current validation saturates and does not predict new-test retrieval.
- If trainable-scope tuning remains mixed, pivot to an explicit low-rank functional alignment layer before the frozen shared stack rather than training the full projection head.

### Telegram report-ready update
Cycle capacity-control grid `8495064` completed: all six tasks exited `0:0`, with runtime `7:53` to `47:29` and ~21.8 GB RSS under 64 GB. Baseline-all with validation plumbing got `test/loss=2.69`, `test_fwd=0.487`, `test_bwd=0.343` versus previous matched baseline `2.72/0.493/0.320`. Adapter-only reduced trainables to `16.1M` but failed on test retrieval: no-prior `test/loss=5.77`, `test_fwd=0.060`, `test_bwd=0.0067`; prior `1e-3` was `5.77/0.053/0.0067`; prior `1e-2` was `5.81/0.0267/0.0033`. Adapter-head used `461.1M` trainables and was mixed: `test/loss=2.66`, `test_fwd=0.470`, `test_bwd=0.373`, so it improved loss and backward retrieval over Cycle 3 but hurt forward retrieval. Adapter-head early-stop stopped after 21 epochs (`best_val_loss=0.044`) but final test was poor: `test/loss=3.78`, `test_fwd=0.243`, `test_bwd=0.187`; checkpoint-time best-val metrics were `test/loss=3.71`, `test_fwd=0.310`, `test_bwd=0.157`. Cached train-shard validation saturated to near-perfect retrieval and is not a useful early-stop proxy. Recommendation: do not advance adapter-only; next test should refine adapter-head with differential/lower head LR or a smaller low-rank alignment layer before the frozen shared stack. No plots generated.

## 2026-05-20 06:02 EDT - Cycle 4 Constrained Adapter-Head Alignment Setup and Smoke

### Plan executed
- Read `/plan.md` at the start of the run and executed only its constrained adapter-head alignment plan. The plan file header says Cycle 5, but this is recorded as the requested Cycle 4 execution.
- Implemented differential optimizer groups in `/src/Train.py`:
  - Added `--head_lr_scale`, default `1.0`.
  - Added `--ridge_lr_scale`, default `1.0`.
  - In `train_scope=adapter_head`, `model.ridge` keeps the base LR while `backbone_linear` and `clip_proj` use `head_lr_scale * max_lr`.
  - Optimizer logs now print group names, parameter counts, weight decay, and effective LR.
- Implemented a default-off near-identity residual alignment block after ridge and before the shared mapper:
  - Added `--alignment_lora_rank`, default `0`.
  - Added `--alignment_lora_alpha`, default `0.0`, interpreted as rank when rank > 0.
  - Added `--alignment_lora_dropout`, default `0.0`.
  - The block computes `z + scale * B(A(LayerNorm(z)))`; `B` is zero-initialized so the residual starts as exact identity.
  - For `train_scope=adapter_only`, trainable parameters are ridge plus the LoRA residual while the dense shared head stays frozen.
  - For `train_scope=adapter_head`, ridge, LoRA if enabled, `backbone_linear`, and `clip_proj` are trainable.
- Routed train, cached validation, fixed-probe diagnostics, and held-out test evaluation through the residual alignment path.
- Added fixed-probe diagnostics from cached validation pairs: feature norm, mean, std, effective-rank summaries after ridge/alignment/`clip_proj`, plus drift MSE from initialization.
- Kept cached train-shard validation as diagnostics only with `--early_stop_patience=0`; grid comparisons are final-epoch matched comparisons.
- Added `/src/accel_cycle5_smoke.slurm` and `/src/accel_cycle5_grid.slurm`.

### Verification and smoke jobs
- Syntax checks passed after code fixes: `/src/fmri/bin/python -m py_compile /src/Train.py`.
- First smoke `8497741` failed in `00:01:44`: initial probe diagnostics ran outside autocast, causing fp16 cached voxels to hit fp32 ridge weights. Fixed by wrapping probe collection in fp16 autocast. Also fixed smoke array log naming from `%j` to `%A_%a`.
- Second smoke `8497807` failed in `00:02:07`: effective-rank eigensolve ran under fp16 autocast. Fixed by forcing covariance/eigensolve diagnostics into fp32 with autocast disabled.
- Third smoke `8497865` completed:
  - Task 0 `c5_smoke_adapter_head_headlr_0.1`: `COMPLETED`, `ExitCode=0:0`, `Elapsed=00:03:15`, `MaxRSS=21768640K`.
  - Task 1 `c5_smoke_alignment_lora_r4`: `COMPLETED`, `ExitCode=0:0`, `Elapsed=00:02:58`, `MaxRSS=21624200K`.
  - Both tasks saved `best_val` and `last` checkpoints.
  - Adapter-head smoke confirmed ridge LR `3e-4` and head LRs `3e-5`.
  - LoRA smoke confirmed rank 4, scale `1.0`, `10,240` LoRA params, and `16,112,640` trainable params total.
  - LoRA gradient diagnostic showed a nonfinite aggregate on epoch 1 but finite nonzero gradient by epoch 2 (`train/alignment_lora_grad_norm=24.7`). Patched the diagnostic afterward to separate finite gradient norms from nonfinite counts; this is logging-only.

### Smoke metrics
- `c5_smoke_adapter_head_headlr_0.1`, after 2 epochs: `test/loss=5.37`, `test_fwd=0.0433`, `test_bwd=0.0167`, `train/loss=1.94`, `train_fwd=0.817`, `train_bwd=0.706`.
- `c5_smoke_alignment_lora_r4`, after 2 epochs: `test/loss=5.70`, `test_fwd=0.0833`, `test_bwd=0.00333`, `train/loss=2.66`, `train_fwd=0.800`, `train_bwd=0.265`.
- These are smoke-only metrics and are not interpreted as final model quality.

### Primary grid
- Submitted matched six-task grid after the successful smoke:
  - Command: `sbatch /src/accel_cycle5_grid.slurm`
  - Job array: `8497957`
  - Conditions: `baseline_all`, `adapter_head`, `adapter_head_headlr_0.1`, `adapter_head_headlr_0.03`, `alignment_lora_r4`, `alignment_lora_r8`.
  - Last checked state: all six tasks pending on `gpu` for scheduler priority with `02:00:00` time limit and `64G` memory.

### Conclusions and next questions
- Differential LR, low-rank residual alignment, and diagnostics are implemented and smoke-tested without changing flags-off defaults.
- The required pre-grid smoke caught and resolved two diagnostic-only dtype bugs before any long run.
- The primary result is pending because grid `8497957` has not started. Once it completes, compare final `test/loss`, `test_fwd`, and `test_bwd` against Cycle 4 `baseline_all` (`2.69`, `0.487`, `0.343`) and `adapter_head` (`2.66`, `0.470`, `0.373`).
- Treat cached validation and fixed-probe diagnostics as explanatory signals only; do not use cached train-shard validation for model selection.

## 2026-05-20 07:13 EDT - Cycle 5 Constrained Alignment Results and Held-Out Validation Pivot

### Plan executed
- Read `/plan.md` and monitored primary Cycle 5 grid array `8497957`.
- Checked live state with `squeue -j 8497957` and final accounting with `sacct -j 8497957 --format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS,ReqMem -P`.
- Parsed `/src/slurms/c5_align_grid_8497957_<task>.out` and `.err` for final metrics, optimizer/trainable setup, and fixed-probe diagnostics.
- Applied the `/plan.md` branch rule after final matched metrics: no balanced constrained-alignment win, so pivoted toward true excluded validation infrastructure.
- Added default-off held-out validation flags in `/src/Train.py`:
  - `--heldout_val_sessions`, default `0`;
  - `--heldout_val_start_session`, default `-1`, which starts immediately after the training session range;
  - `--heldout_val_max_samples`, default `300`.
- Held-out validation now builds its cache from train-session tar shards not included in `num_sessions`, checks for overlap/missing shards, records the cache source, and prints parseable `val_metrics` lines.
- Added `/src/accel_cycle5_heldout_val_smoke.slurm` for a 1-hour, 2-epoch smoke using train session `0` and held-out validation session `1`.

### Job states
- Grid `8497957` completed cleanly for all six tasks with `ExitCode=0:0`.
- Elapsed times:
  - `baseline_all` task 0: `00:47:01`
  - `adapter_head` task 1: `00:46:57`
  - `adapter_head_headlr_0.1` task 2: `00:46:52`
  - `adapter_head_headlr_0.03` task 3: `00:48:06`
  - `alignment_lora_r4` task 4: `00:38:59`
  - `alignment_lora_r8` task 5: `00:40:11`
- Batch MaxRSS was about `21.6-21.8 GB`, under the requested `64G`.
- The held-out validation smoke `8499318` completed cleanly in `00:02:31`, `ExitCode=0:0`, MaxRSS `21626688K`, under the requested `32G`.

### Final matched metrics

| Condition | Task | Trainable params | Final test/loss | test_fwd | test_bwd | train/loss | train_fwd | train_bwd | Cached val |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline_all | 0 | 469,462,680 | 2.70 | 0.477 | 0.350 | 0.000211 | 1.000 | 1.000 | Built but not parseable in grid logs |
| adapter_head | 1 | 461,057,664 | 2.65 | 0.483 | 0.333 | 0.000202 | 1.000 | 1.000 | Built but not parseable in grid logs |
| adapter_head_headlr_0.1 | 2 | 461,057,664 | 3.99 | 0.0567 | 0.233 | 0.00228 | 1.000 | 1.000 | Built but not parseable in grid logs |
| adapter_head_headlr_0.03 | 3 | 461,057,664 | 4.42 | 0.0367 | 0.193 | 0.00626 | 1.000 | 1.000 | Built but not parseable in grid logs |
| alignment_lora_r4 | 4 | 16,112,640 | 5.76 | 0.0467 | 0.00333 | 0.109 | 1.000 | 1.000 | Built but not parseable in grid logs |
| alignment_lora_r8 | 5 | 16,120,832 | 5.74 | 0.0133 | 0.00333 | 0.105 | 1.000 | 1.000 | Built but not parseable in grid logs |

### Alignment diagnostics
- `baseline_all`: final ridge/aligned norm `91`, effective rank `38.6`, drift MSE `6.66`; final `clip_proj` norm `448`, effective rank `42.3`, drift MSE `0.510`.
- `adapter_head`: final ridge/aligned norm `91.9`, effective rank `38.4`, drift MSE `6.83`; final `clip_proj` norm `440`, effective rank `42.1`, drift MSE `0.499`.
- `adapter_head_headlr_0.1`: final ridge/aligned norm `95.5`, effective rank `38.7`, drift MSE `7.99`; final `clip_proj` norm `216`, effective rank `60.8`, drift MSE `0.197`. The shrunken `clip_proj` norm coincides with severe forward-retrieval collapse.
- `adapter_head_headlr_0.03`: final ridge/aligned norm `73.0`, effective rank `50.0`, drift MSE `4.73`; final `clip_proj` norm `204`, effective rank `66.3`, drift MSE `0.197`. Retrieval also collapsed.
- `alignment_lora_r4`: final ridge/aligned norm `109`, effective rank `68.5`, drift MSE `11.7`; final frozen `clip_proj` norm `245`, effective rank `73.0`, drift MSE `0.156`. LoRA finite grad norm ended at `17.3`; nonfinite gradient count was positive in 2 logged updates early, max `0.387`, then returned to `0`.
- `alignment_lora_r8`: final ridge/aligned norm `55.2`, effective rank `68.2`, drift MSE `3.13`; final frozen `clip_proj` norm `246`, effective rank `73.1`, drift MSE `0.158`. LoRA finite grad norm ended at `75.1`; nonfinite gradient count was positive in 2 logged updates early, max `0.387`, then returned to `0`.

### Conclusions
- Cycle 5 did not meet the primary success criterion. No condition improved `test_bwd` or loss while keeping `test_fwd` within `0.01` of the in-grid baseline.
- In-grid `baseline_all` was the strongest balanced condition: `test/loss=2.70`, `test_fwd=0.477`, `test_bwd=0.350`.
- `adapter_head` slightly improved loss (`2.65` vs `2.70`) and forward retrieval (`0.483` vs `0.477`) but reduced backward retrieval (`0.333` vs `0.350`). This is not a clean improvement over the in-grid baseline and does not retain the Cycle 4 backward gain.
- Lower head LR was decisively bad in this setup. `headlr_0.1` and `headlr_0.03` both fully memorized training retrieval but collapsed held-out retrieval, especially `test_fwd`.
- LoRA rank 4/8 behaved like adapter-only: training retrieval reached `1.0`, but new-test retrieval was near chance and loss stayed around `5.7`. Early nonfinite LoRA gradient diagnostics reinforce deprioritizing this exact implementation.
- Cached train-shard validation was built in the grid and saved `best_val`, but the grid logs did not expose parseable `val/*` metrics. The new explicit `val_metrics` print fixes this for future runs.

### Held-out validation smoke
- Command: `sbatch /src/accel_cycle5_heldout_val_smoke.slurm`
- Job ID: `8499318`.
- Smoke source: trained with `--num_sessions=1`, validating from excluded shard `/wds/subj01/train/{1..1}.tar`.
- Log confirmation:
  - `Validation cache ready: source=heldout_train_sessions n=75 unique image/voxel pairs`
  - Epoch 1 held-out val: `val/loss=4.28962`, `val_fwd=0.106667`, `val_bwd=0.0266667`
  - Epoch 2 held-out val: `val/loss=3.56854`, `val_fwd=0.186667`, `val_bwd=0.0666667`
- Final 2-epoch smoke test metrics: `test/loss=4.68`, `test_fwd=0.110`, `test_bwd=0.0367`; train metrics: `train/loss=1.10`, `train_fwd=0.843`, `train_bwd=0.696`.

### Recommended next research questions
- Stop advancing Cycle 5 differential-head-LR and adapter-only LoRA as primary branches.
- Use the new excluded-session validation path to measure whether validation from session `1` predicts `new_test` better than the previous train-shard cache.
- Next compact run should compare `baseline_all` and the previously mixed `adapter_head` with `heldout_val_sessions=1` and parse explicit `val_metrics`; use final new-test metrics for selection, with held-out validation as a diagnostic.
- If excluded validation tracks new-test behavior, use it to screen reliability/ROI-stratified voxel selection or explicit Procrustes/SRM-style functional alignment from common stimuli.
- Do not revisit adapter-only, adapter priors, beta-std top-k masking, diffusion-prior tuning, larger decoders, Fourier augmentation, or blurred-CLIP schedules unless a new validation/reliability result justifies it.
## 2026-05-20 08:35 EDT - Cycle 6 Held-Out Validation Selector Check

### Plan executed
- Read `/plan.md` at the start of the run and kept the cycle measurement-focused: no new architecture, no lower-head-LR, no LoRA, no reliability masks, no prior/diffusion changes.
- Added `/src/accel_cycle6_heldout_val_smoke.slurm` for the required 1-hour smoke using train session `0`, held-out validation session `1`, `heldout_val_max_samples=75`, and 2 epochs.
- Added `/src/accel_cycle6_heldout_val_grid.slurm` for the compact four-task grid:
  - `baseline_all_heldout1`
  - `adapter_head_heldout1`
  - `baseline_all_heldout2`
  - `adapter_head_heldout2`
- Syntax checks passed:
  - `/src/fmri/bin/python -m py_compile /src/Train.py`
  - `bash -n /src/accel_cycle6_heldout_val_smoke.slurm`
  - `bash -n /src/accel_cycle6_heldout_val_grid.slurm`
- Best-heldout checkpoints were saved by existing `best_val.pth` logic. Separate best-checkpoint reload/new-test evaluation was not implemented this cycle because it would require broader eval-loop extraction; final/new-test rank agreement was used as planned.

### Jobs launched
- Smoke: `sbatch /src/accel_cycle6_heldout_val_smoke.slurm` -> job `8499774`.
- Grid: `sbatch /src/accel_cycle6_heldout_val_grid.slurm` -> array `8499927`.

### Smoke result
- Job `8499774` completed cleanly: `COMPLETED`, `ExitCode=0:0`, `Elapsed=00:02:34`, batch `MaxRSS=21626104K`, under requested `32G`.
- Confirmed held-out validation source and no train/validation shard overlap:
  - training shard: `/wds/subj01/train/{0..0}.tar`
  - validation shard: `/wds/subj01/train/{1..1}.tar`
  - log: `Validation cache ready: source=heldout_train_sessions n=75 unique image/voxel pairs`
- Parseable smoke validation:
  - epoch 1: `val/loss=4.28962`, `val_fwd=0.106667`, `val_bwd=0.0266667`
  - epoch 2: `val/loss=3.56854`, `val_fwd=0.186667`, `val_bwd=0.0666667`
- Final smoke new-test metrics: `test/loss=4.68`, `test_fwd=0.110`, `test_bwd=0.0367`.
- Final smoke train metrics: `train/loss=1.10`, `train_fwd=0.843`, `train_bwd=0.696`.

### Grid job states
- Array `8499927` completed cleanly for all four tasks with `ExitCode=0:0`.
- Elapsed/RSS:
  - task 0: `00:47:15`, `MaxRSS=21771056K`
  - task 1: `00:46:19`, `MaxRSS=21626052K`
  - task 2: `00:48:06`, `MaxRSS=21769988K`
  - task 3: `00:48:34`, `MaxRSS=21766448K`
- All tasks stayed below requested `64G`.

### Final matched metrics

| Condition | Task | Held-out sessions | Val n | Trainable params | Final val/loss | val_fwd | val_bwd | Final test/loss | test_fwd | test_bwd | train/loss | train_fwd | train_bwd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_all_heldout1 | 0 | 1 | 75 | 469,462,680 | 2.09552 | 0.506667 | 0.400000 | 2.70 | 0.477 | 0.350 | 0.000211 | 1.000 | 1.000 |
| adapter_head_heldout1 | 1 | 1 | 75 | 461,057,664 | 2.05662 | 0.520000 | 0.413333 | 2.65 | 0.483 | 0.333 | 0.000202 | 1.000 | 1.000 |
| baseline_all_heldout2 | 2 | 2 | 150 | 469,462,680 | 2.87685 | 0.366667 | 0.286667 | 2.70 | 0.477 | 0.350 | 0.000211 | 1.000 | 1.000 |
| adapter_head_heldout2 | 3 | 2 | 150 | 461,057,664 | 2.85590 | 0.346667 | 0.260000 | 2.65 | 0.483 | 0.333 | 0.000202 | 1.000 | 1.000 |

### Best validation epochs
- `baseline_all_heldout1`: best val loss at epoch 150, `val/loss=2.09552`, `val_fwd=0.506667`, `val_bwd=0.400000`.
- `adapter_head_heldout1`: best val loss at epoch 148, `val/loss=2.05646`, `val_fwd=0.520000`, `val_bwd=0.413333`; final epoch was essentially tied at `2.05662`.
- `baseline_all_heldout2`: best val loss at epoch 149, `val/loss=2.87672`, `val_fwd=0.366667`, `val_bwd=0.286667`; final epoch was essentially tied at `2.87685`.
- `adapter_head_heldout2`: best val loss at epoch 150, `val/loss=2.85590`, `val_fwd=0.346667`, `val_bwd=0.260000`.

### Rank agreement
- Heldout-1:
  - val loss ranks `adapter_head` better; new-test loss also ranks `adapter_head` better.
  - val fwd ranks `adapter_head` better; new-test fwd also ranks `adapter_head` better.
  - val bwd ranks `adapter_head` better; new-test bwd ranks `baseline_all` better.
  - val mean retrieval ranks `adapter_head` better (`0.4667` vs `0.4533`); new-test mean retrieval ranks `baseline_all` slightly better (`0.4135` vs `0.4080`).
- Heldout-2:
  - val loss ranks `adapter_head` better; new-test loss also ranks `adapter_head` better.
  - val fwd ranks `baseline_all` better; new-test fwd ranks `adapter_head` better.
  - val bwd ranks `baseline_all` better; new-test bwd also ranks `baseline_all` better.
  - val mean retrieval ranks `baseline_all` better (`0.3267` vs `0.3033`); new-test mean retrieval also ranks `baseline_all` slightly better (`0.4135` vs `0.4080`).

### Conclusions
- Held-out validation did not saturate: all final held-out retrieval values stayed far below `1.0`, unlike the old train-shard validation cache.
- The validation selector is useful diagnostically but not yet reliable enough as a sole model-selection signal. It consistently agreed with final new-test loss, but retrieval agreement depended on the held-out window:
  - heldout-1 agreed on loss and forward retrieval but missed backward and mean retrieval;
  - heldout-2 agreed on loss, backward retrieval, and mean retrieval but missed forward retrieval.
- Final new-test metrics reproduced Cycle 5 exactly for the overlapping conditions:
  - `baseline_all`: `test/loss=2.70`, `test_fwd=0.477`, `test_bwd=0.350`
  - `adapter_head`: `test/loss=2.65`, `test_fwd=0.483`, `test_bwd=0.333`
- No Cycle 6 condition met model-side success criteria. `adapter_head` improved loss and forward retrieval but reduced backward retrieval by `0.017`; `baseline_all` remains the stronger balanced condition by mean retrieval and backward retrieval.
- Because heldout-1 and heldout-2 disagree on retrieval direction, do not use a single excluded session as the next-cycle selector without fixing or broadening validation construction.

### Recommended next research questions
- Stop architecture search until validation is made more stable across held-out session choices.
- Next validation work should enforce image-ID exclusion from the actual training loader and use either multiple excluded sessions, official repeat structure, or metadata-backed repeat/reliability splits.
- If compute is limited, use heldout-2 or a multi-session aggregate rather than heldout-1 alone, because heldout-2 better matched new-test backward and mean retrieval in this cycle.
- Continue to avoid diffusion-prior tuning, larger decoders, Fourier augmentation, blurred-CLIP schedules, adapter-only, adapter priors, beta-std top-k masking, lower-head-LR sweeps, and the current LoRA residual until a more reliable validation signal justifies them.
