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
