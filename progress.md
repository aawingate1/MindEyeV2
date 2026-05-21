## 2026-05-20 Cycle 1

Plan source: executed `/plan.md` only. The required gate is the paper-matched MindEye2 1-hour baseline on subjects 1/2/5/7 before any ablation.

Code/config changes:
- Replaced `/src/accel.slurm` with a 1-hour Slurm smoke/preflight for the official 1-session fine-tuning config for subject 1: `--use_prior`, `--blurry_recon`, `--hidden_dim=4096`, `--num_sessions=1`, `--batch_size=24`, and `--multisubject_ckpt=../train_logs/final_multisubject_subj01`.
- Added a fast checkpoint existence check so a non-baseline run is not accidentally launched if the required pretrained multisubject checkpoint is absent.
- No changes were made to `Train.py` or `models.py` in this cycle.

Commands/jobs launched:
- `sbatch accel.slurm`
- Slurm job: `8516467` (`c1_baseline_smoke`), requested `1x A100`, `64G`, `01:00:00`.
- Job result: `FAILED`, exit `3:0`, elapsed `00:00:01`.
- Logs: `/src/slurms/c1_baseline_smoke_8516467.out` and `/src/slurms/c1_baseline_smoke_8516467.err`.

Observed setup/results:
- The smoke job failed at preflight because `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/final_multisubject_subj01/last.pth` is missing.
- No baseline training or evaluation metrics were produced, so the baseline gate has not passed.
- No ablations were run, per plan.
- Local required subject beta files exist with voxel counts:
  - subj01: `15724`
  - subj02: `14278`
  - subj05: `13039`
  - subj07: `12682`
- ROI masks are present in `/src/brain_region_masks.hdf5`:
  - subj01: early `4657`, higher `11067`
  - subj02: early `3757`, higher `10521`
  - subj05: early `3661`, higher `9378`
  - subj07: early `3251`, higher `9431`
- Local files include NSD beta/image data and low-level/model support files, but not the required `train_logs/final_multisubject_subj0#` checkpoint folders. The original MindEyeV2 README says these pretrained models are in the HuggingFace dataset `pscotti/mindeyev2` under `train_logs`.

Conclusions:
- Baseline reproduction is blocked by missing paper-required multisubject checkpoints. Training from scratch or using the local `/src/gnet_multisubject.pt` would not be paper-matched: that file is a GNet-style dictionary with keys such as `best_params`, `final_params`, and ROI metadata, not a MindEye2 `last.pth` checkpoint compatible with `Train.py`.
- The next valid step is to stage/download `train_logs/final_multisubject_subj01`, `subj02`, `subj05`, and `subj07` before launching any long baseline training. Once present, rerun the 1-hour smoke for subj01, then launch the full 150-epoch 1-session baseline per subject only after the smoke succeeds.

Recommended next research questions:
- Can the four required pretrained checkpoint folders be staged from HuggingFace without exceeding storage limits?
- After the checkpoint issue is fixed, does the local evaluator reproduce the paper's 1-hour row within tolerance on subj01 before scaling to 1/2/5/7?
- Are refined reconstruction artifacts and the final evaluation notebooks/scripts present for full PixCorr/SSIM/AlexNet/Inception/CLIP/EfficientNet/SwAV/retrieval reporting?

Telegram-ready update:
Cycle 1 established the paper-matched baseline gate but did not produce metrics because the required MindEye2 multisubject checkpoint is missing locally. I replaced `/src/accel.slurm` with a 1-hour paper-config smoke/preflight and submitted job `8516467`; it failed fast in 1 second with missing `/train_logs/final_multisubject_subj01/last.pth`. Data files and ROI masks are present for subjects 1/2/5/7, with voxel counts 15724/14278/13039/12682. No ablations were run. Next action is to stage the official `final_multisubject_subj0#` checkpoint folders from HuggingFace, then rerun the subj01 smoke before full 1-hour baseline jobs.

## 2026-05-20 Cycle 2

Plan source: executed `/plan.md` only. Telegram report is not due.

Code/config changes:
- Staged official Hugging Face `pscotti/mindeyev2` multisubject checkpoints under `/src/train_logs/final_multisubject_subj0{1,2,5,7}/last.pth`, because compute nodes cannot resolve `huggingface.co` and cannot download directly.
- Updated `/src/accel.slurm` from the Cycle 1 missing-checkpoint preflight into a full official-checkpoint 1-session baseline array for subjects `1/2/5/7`.
- Added checkpoint byte-size checks, LFS-pointer checks, and a `torch.load(..., mmap=True)` sanity check in `accel.slurm`.
- Patched `/src/Train.py` so frozen target encoders used during training (`clip_img_embedder`, SD VAE target encoder, ConvNeXt target encoder) run under `torch.no_grad()`. This preserved the batch-24 paper smoke on a 40 GB A100; without it, batch 24 OOMed by ~588 MiB.
- Also patched `Train.py` to define `model_for_submodules` after `accelerator.prepare` for DDP-wrapped submodule access. DDP later hung after epoch 1, so full baselines were launched as single-GPU batch-24 jobs, not DDP.

Official checkpoint provenance:
- Source: Hugging Face dataset `pscotti/mindeyev2`, paths `train_logs/final_multisubject_subj01`, `subj02`, `subj05`, `subj07`.
- Staged files and verified sizes:
  - subj01: `/src/train_logs/final_multisubject_subj01/last.pth`, `10301091408` bytes
  - subj02: `/src/train_logs/final_multisubject_subj02/last.pth`, `10324782672` bytes
  - subj05: `/src/train_logs/final_multisubject_subj05/last.pth`, `10345082448` bytes
  - subj07: `/src/train_logs/final_multisubject_subj07/last.pth`, `10350931536` bytes
- All four load as dicts with `model_state_dict` and 261 state keys via the `/src/fmri/bin/python` environment.

Commands/jobs launched:
- Local download/verification commands used `curl -L --fail --retry 5 --continue-at -` and `/src/fmri/bin/python` checkpoint load checks.
- `sbatch accel.slurm` job `8521229`: compute-node Hugging Face staging attempt; failed in `00:00:51` because compute DNS could not resolve `huggingface.co`.
- `sbatch accel.slurm` job `8521307`: subject 1 smoke with staged checkpoint; failed in `00:02:04` because `num_epochs=1` made `OneCycleLR pct_start=2.0`.
- `sbatch accel.slurm` job `8521381`: subject 1 3-epoch batch-24 smoke on one 40 GB A100; failed in `00:01:45` with CUDA OOM during SD VAE target encoding.
- `sbatch accel.slurm` job `8521532`: GH200 smoke attempt; failed in `00:00:01` because the existing x86 Python/torch environment is not usable on the Grace node.
- `sbatch accel.slurm` job `8521563`: subject 1 3-epoch batch-12 smoke on one visible A100; completed in `00:05:50`. This was useful for checkpoint/path validation but not paper batch matched.
- `sbatch accel.slurm` job `8521762`: true two-process DDP smoke; failed in `00:02:42` because `DistributedDataParallel` hid direct `model.ridge` submodule access.
- `sbatch accel.slurm` job `8521942`: patched DDP smoke; passed the prior submodule error and produced epoch-1 metrics but hung after epoch 1, so it was cancelled.
- `sbatch accel.slurm` job `8522325`: paper batch-24, one-A100, 3-epoch smoke after `no_grad()` target-encoder patch; completed successfully. Final logged smoke metrics: test blurry PixCorr `0.288`, test loss `11.8`, test fwd top-1 `0.700`, test bwd top-1 `0.513`, train loss `11.2`, train blurry PixCorr `0.375`.
- Full baseline array submitted with `sbatch accel.slurm`: job `8522659_[0-3]`, subjects `1/2/5/7`, one A100 each, `05:00:00`, `64G`, `--num_epochs=150`, `--batch_size=24`, `--num_sessions=1`, `--hidden_dim=4096`, `--use_prior`, `--blurry_recon`, `--multisubject_ckpt=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/train_logs/final_multisubject_subj0#`. At write time all four array tasks were running.

Observed setup/results:
- Official checkpoint baseline path is now recovered for subject 1 and validated by a paper batch-24 smoke.
- Full per-subject metric vectors are not available yet; the 150-epoch jobs are running.
- The full evaluator for PixCorr/SSIM/AlexNet/Inception/CLIP/EfficientNet/SwAV/retrieval has not run yet.
- DDP is not currently a safe path for this code without more surgery: the initial direct-submodule failure was fixed, but the patched DDP smoke hung after epoch 1.

Conclusions:
- The original Cycle 1 blocker was checkpoint staging, not data availability. Official checkpoints are present and loadable.
- A 40 GB A100 can run the paper batch-24 smoke after avoiding autograd bookkeeping for frozen target encoders. This is the cleanest path for paper-matched full training on current hardware.
- No ablations were run. The baseline gate is still open until the submitted official-checkpoint full runs finish and the final evaluator reproduces the paper metric row within tolerance.

Recommended next research questions:
- Do the `8522659_[0-3]` jobs start and complete within the requested 5 hours, and do their training/test curves look like the successful subject 1 smoke?
- Which script/notebook should be used to produce the full paper metric vector from the completed `cycle2_subj0#_official_1sess_150ep` checkpoints?
- If full evaluator metrics miss the paper row, first debug refined/unrefined reconstruction selection, new-test split, retrieval candidate pool, and evaluator settings before any ablation.

## 2026-05-20 Cycle 3

Plan source: executed `/plan.md` only. Telegram report is not due.

Code/config changes:
- Generated executable Python scripts from the standard notebooks: `/src/recon_inference.py`, `/src/enhanced_recon_inference.py`, and `/src/final_evaluations.py`.
- Added execution-only fixes to the generated scripts:
  - default `plotting=False` for non-interactive runs;
  - local `/src/zavychromaxl_v30.safetensors` path for enhancement;
  - fallback checkpoint lookup in `/src/train_logs` for staged checkpoints;
  - optional `--max_images` smoke-test limit for `recon_inference.py` and `enhanced_recon_inference.py`;
  - non-interactive `final_evaluations.py` exits after saving the required paper metric table, before optional caption/UMAP sections that need extra tensors.
- Added Slurm scripts:
  - `/src/cycle3_eval_smoke.slurm` for the required 1-hour inference/refinement smoke;
  - `/src/cycle3_eval_full.slurm` for full recon, enhanced recon, and final evaluation;
  - `/src/cycle3_resume_subj12.slurm` to resume failed subjects 1/2.
- Patched `/src/Train.py` with `--resume_from_ckpt`, local checkpoint resume semantics, and `optimizer.zero_grad(set_to_none=True)`. Resume loads `../train_logs/{model_name}/last.pth` if present and continues at saved epoch + 1; otherwise it falls back to the official multisubject initialization. The resume Slurm also sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True`.

Commands/jobs launched and statuses:
- Checked original baseline array `8522659_[0-3]` with `squeue`, `sacct`, and log parsing.
- `8522659_0`, subject 1: `FAILED`, exit `1:0`, elapsed `00:53:00`, node `della-l02g7`. Failure class: runtime/memory. Traceback is CUDA OOM during `accelerator.backward(loss)` at `Train.py:953`, after epoch 78. Last logged metrics: test blurry PixCorr `0.296`, test loss `14.4`, test fwd top-1 `0.923`, test bwd top-1 `0.847`, train loss `7.3`, train blurry PixCorr `0.747`.
- `8522659_1`, subject 2: `FAILED`, exit `1:0`, elapsed `00:51:48`, node `della-l02g3`. Failure class: runtime/memory. Traceback is CUDA OOM during `accelerator.backward(loss)` at `Train.py:953`, after epoch 77. Last logged metrics: test blurry PixCorr `0.216`, test loss `15.0`, test fwd top-1 `0.840`, test bwd top-1 `0.797`, train loss `7.4`, train blurry PixCorr `0.739`.
- `8522659_2`, subject 5: `COMPLETED`, exit `0:0`, elapsed `01:37:02`, node `della-l02g1`. Final training log metrics: test blurry PixCorr `0.198`, test loss `14.5`, test fwd top-1 `0.657`, test bwd top-1 `0.553`, train loss `5.89`, train blurry PixCorr `0.800`.
- `8522659_3`, subject 7: `COMPLETED`, exit `0:0`, elapsed `01:36:11`, node `della-l01g15`. Final training log metrics: test blurry PixCorr `0.234`, test loss `14.3`, test fwd top-1 `0.747`, test bwd top-1 `0.570`, train loss `5.94`, train blurry PixCorr `0.788`.
- Submitted resume array `8526170_[0-1]` for subjects 1/2 with the same paper config plus local resume and allocator mitigation. Both tasks correctly found local checkpoints and resumed at epoch 76, but failed immediately with PyTorch CUDA allocator internal assertion `!block->expandable_segment_` caused by `PYTORCH_CUDA_ALLOC_CONF=...,expandable_segments:True`. Removed `expandable_segments:True` and relaunched as `8528252_[0-1]` with only `max_split_size_mb:128`; at final check both replacement tasks were `PENDING`, `02:00:00`.
- Staged missing evaluator cache models from Hugging Face into `/src/cache/hf_hub`: `microsoft/git-large-coco` and `openai/clip-vit-large-patch14`, because compute nodes run offline.
- Evaluation smoke attempts:
  - `8526682`: failed in `00:02:08`; missing offline `microsoft/git-large-coco`.
  - `8526991`: failed in `00:02:55`; `recon_inference` succeeded for two images, then `enhanced_recon_inference` failed on missing offline `openai/clip-vit-large-patch14`.
  - `8527561`: completed in `00:04:56` on subject 5. It ran two-image `recon_inference` and `enhanced_recon_inference`, loaded `/src/zavychromaxl_v30.safetensors`, and saved `evals/cycle2_subj05_official_1sess_150ep/cycle2_subj05_official_1sess_150ep_all_enhancedrecons.pt`.
- Submitted full evaluation for completed subjects only: `8528056_[2-3]` from `/src/cycle3_eval_full.slurm`, corresponding to subjects 5 and 7. At final check both tasks are `PENDING (Priority)`, `12:00:00`, not yet assigned.

Checkpoint/artifact provenance:
- Official multisubject initialization remains the Hugging Face `pscotti/mindeyev2` checkpoints staged under `/src/train_logs/final_multisubject_subj0{1,2,5,7}/last.pth`.
- Local subject 5/7 fine-tuned checkpoints were written by `Train.py` under the compute-visible path `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle2_subj0{5,7}_official_1sess_150ep/last.pth`.
- Subject 1/2 partial checkpoints should exist at the same compute-visible parent path and are the intended resume source for job `8526170`.

Baseline metric table status:
- Full paper-vector metrics are not available yet. The full evaluator has not started because only subjects 5/7 completed training and their full evaluation jobs are still pending; subjects 1/2 are pending resumed completion.
- Baseline gate is therefore still open. No ablations were launched.

Required voxel/ROI diagnostics:
- Voxel counts and ROI masks remain as previously verified:
  - subj01: `15724` total; early `4657`; higher `11067`
  - subj02: `14278` total; early `3757`; higher `10521`
  - subj05: `13039` total; early `3661`; higher `9378`
  - subj07: `12682` total; early `3251`; higher `9431`
- Weak-subject behavior is visible in training diagnostics: subject 5 finished with test fwd/bwd retrieval `0.657/0.553`; subject 7 finished with `0.747/0.570`. These are lower than subjects 1/2 partial runs before OOM, but final refined evaluator metrics are still pending.

Conclusions:
- Local training is not fully complete: subjects 5/7 passed, subjects 1/2 hit late-epoch GPU OOM. This is currently classified as operational runtime/memory drift, not model drift.
- The evaluator path had cache/environment drift on offline compute nodes. After staging `microsoft/git-large-coco` and `openai/clip-vit-large-patch14`, the 1-hour smoke passed end-to-end for subject 5.
- The next valid work is to let `8528252_[0-1]` finish subjects 1/2 and `8528056_[2-3]` produce full metrics for subjects 5/7, then run the same full evaluation for subjects 1/2 after their resumed checkpoints complete. Do not run ablations until the full four-subject paper-vector table exists and passes or fails the tolerance rule.

Recommended next research questions:
- Do the subject 1/2 resume jobs avoid the late-epoch OOM and preserve the strong partial retrieval diagnostics?
- Does full refined evaluation for completed subjects 5/7 match the paper row enough to justify completing 1/2 evaluation, or does it expose evaluator drift?
- If full local metrics miss tolerance, evaluate official Hugging Face 1-session checkpoints/artifacts before changing the model.

## 2026-05-20 Cycle 4

Plan source: executed `/plan.md` only. Telegram report is not due.

Code/config changes:
- Added an operational memory cleanup in `/src/Train.py`: after the training-loop blurry PixCorr metric decode, `blurry_recon_images` and `pixcorr` are deleted and `torch.cuda.empty_cache()` is called before `accelerator.backward(loss)`. This is intended to reduce allocator fragmentation at the exact late-epoch backward allocation that failed subjects 1/2, without changing the paper baseline loss, batch size, data, checkpoint initialization, or model architecture.
- Added `/src/cycle4_resume_subj12.slurm` for subject 1/2 paper-config resume jobs with a shorter `01:30:00` time request, `64G`, one A100, `batch_size=24`, `num_epochs=150`, `num_sessions=1`, `hidden_dim=4096`, `use_prior`, `blurry_recon`, official `final_multisubject_subj0#` initialization, local `--resume_from_ckpt`, and `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.
- Added `/src/cycle4_eval_full_s57.slurm` for full refined evaluation of completed subjects 5/7 with a shorter `04:00:00` time request rather than the prior 12-hour array. It runs the same `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py` path and uses enhanced reconstructions for the primary metric table.

Validation:
- `/src/fmri/bin/python -m py_compile /src/Train.py /src/recon_inference.py /src/enhanced_recon_inference.py /src/final_evaluations.py` completed successfully.
- `bash -n /src/cycle4_resume_subj12.slurm /src/cycle4_eval_full_s57.slurm` completed successfully.

Commands/jobs launched:
- Cancelled still-pending Cycle 3 duplicate arrays: `scancel 8528252 8528056`.
  - `8528252_[0-1]` subject 1/2 resume: cancelled before start; no elapsed time, no node, no metrics.
  - `8528056_[2-3]` subject 5/7 full eval: cancelled before start; no elapsed time, no node, no metrics.
- Submitted `sbatch /src/cycle4_resume_subj12.slurm`: job `8529134_[0-1]`, subjects 1/2, one A100 each, `01:30:00`, `64G`.
- Submitted `sbatch /src/cycle4_eval_full_s57.slurm`: job `8529135_[0-1]`, subjects 5/7, one A100 each, `04:00:00`, `64G`.
- At check time, `8529134_[0-1]` and `8529135_[0-1]` were `PENDING`, elapsed `00:00:00`, no node assigned, no exit code beyond pending `0:0`.

Observed metrics/results:
- No new training or full paper-vector evaluation metrics were produced in Cycle 4 because the replacement jobs had not started by the final check.
- Existing completed baseline status remains unchanged: subjects 5/7 have completed local 150-epoch one-session training; subjects 1/2 still require resumed completion; full refined evaluation for 1/2/5/7 remains pending.
- Baseline gate remains open. No ablations were launched.

Checkpoint/artifact provenance:
- Official multisubject initialization remains Hugging Face `pscotti/mindeyev2` staged under `/src/train_logs/final_multisubject_subj0{1,2,5,7}/last.pth`.
- Local fine-tuned checkpoint names remain `cycle2_subj0{1,2,5,7}_official_1sess_150ep`; subjects 1/2 resume from the last local checkpoint if present, otherwise the Slurm command still points at the official multisubject checkpoint.
- Subject 5 two-image smoke artifacts remain under `/src/evals/cycle2_subj05_official_1sess_150ep/`; full 1000-image artifacts are still pending.

Required voxel/ROI diagnostics:
- subj01: `15724` total; early `4657`; higher `11067`
- subj02: `14278` total; early `3757`; higher `10521`
- subj05: `13039` total; early `3661`; higher `9378`
- subj07: `12682` total; early `3251`; higher `9431`

Conclusions:
- Cycle 4 classified the unresolved subject 1/2 issue as operational CUDA allocator/memory failure and applied the smallest semantic-preserving mitigation available before relaunch: allocator split-size plus freeing metric-only decode outputs before backward.
- The completed-subject evaluator was made more schedulable by reducing requested wall time after the Cycle 3 smoke validated the end-to-end path.
- The next valid action is to inspect `8529134_[0-1]` and `8529135_[0-1]` logs once they start, parse completion metrics or failures, and then launch full evaluation for subjects 1/2 immediately after their resumed checkpoints finish.

Recommended next research questions:
- Did the Cycle 4 cleanup plus `max_split_size_mb:128` eliminate the late backward OOM for subjects 1/2?
- Do the subject 5/7 full refined metrics indicate evaluator/refinement drift before subject 1/2 evaluation is launched?
- If the refined metric table is still incomplete or outside tolerance, should official Hugging Face one-session fine-tuned artifacts be staged first to separate local training drift from evaluator drift?

## 2026-05-20 Cycle 5

Plan source: executed `/plan.md` only. Telegram report is not due.

Code/config changes:
- Corrected active Slurm path assumptions in `/src/cycle4_resume_subj12.slurm`, `/src/cycle4_eval_full_s57.slurm`, and `/src/accel.slurm`: `SRC_DIR` and Slurm stdout/stderr now use `/src` instead of missing `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`.
- Added `/src/cycle5_official_control_smoke.slurm`: 1-hour official one-session checkpoint smoke on subject 1 with `recon_inference.py --max_images=2`.
- Added `/src/cycle5_official_control_full.slurm`: official one-session control array for subjects 1/2/5/7. It preserves Hugging Face enhanced reconstructions, regenerates missing local evaluator intermediates (`all_blurryrecons`, `all_clipvoxels`, `all_predcaptions`, `all_recons`) through `recon_inference.py`, then runs `final_evaluations.py` against the official enhanced recon tensor.

Validation:
- `bash -n /src/cycle4_resume_subj12.slurm /src/cycle4_eval_full_s57.slurm /src/accel.slurm` passed.
- `bash -n /src/cycle5_official_control_smoke.slurm /src/cycle5_official_control_full.slurm` passed.
- Official one-session artifacts were checked for LFS pointer rejection by reading file headers; all are PyTorch zip archives beginning with `PK`, not text pointer files.
- Official checkpoints load with `/src/fmri/bin/python` and `torch.load(..., mmap=True)`:
  - `final_subj01_pretrained_1sess_24bs/last.pth`: dict with `model_state_dict`, 249 state keys
  - `final_subj02_pretrained_1sess_24bs/last.pth`: dict with `model_state_dict`, 249 state keys
  - `final_subj05_pretrained_1sess_24bs/last.pth`: dict with `model_state_dict`, 249 state keys
  - `final_subj07_pretrained_1sess_24bs/last.pth`: dict with `model_state_dict`, 249 state keys
- Official enhanced recon tensors load as `(1000, 3, 512, 512)` `torch.float32` for all four subjects.

Cycle 4 job outcomes inspected:
- Original `8529134_[0-1]` subject 1/2 resume array was still `PENDING`, elapsed `00:00:00`, no node assigned. It also had stdout/stderr and `SRC_DIR` pointing to absent `/scratch/...`, so it was cancelled before start.
- Original `8529135_[0-1]` subject 5/7 full-eval array was still `PENDING`, elapsed `00:00:00`, no node assigned. It had the same absent `/scratch/...` path issue, so it was cancelled before start.
- `scontrol` showed scheduled nodes before cancellation but no execution had begun; no new training/eval metrics or failures came from these original arrays.

Commands/jobs launched:
- Cancelled path-broken pending arrays: `scancel 8529134 8529135`.
- Resubmitted corrected local subject 1/2 resume array: `sbatch /src/cycle4_resume_subj12.slurm`, job `8529618_[0-1]`, subjects 1/2, one A100 each, `01:30:00`, `64G`. At write time: `PENDING (Priority)`.
- Resubmitted corrected local subject 5/7 full-eval array: `sbatch /src/cycle4_eval_full_s57.slurm`, job `8529619_[0-1]`, subjects 5/7, one A100 each, `04:00:00`, `64G`. At write time: `PENDING (Priority)`.
- Staged official one-session controls from Hugging Face dataset `pscotti/mindeyev2` with `snapshot_download(..., local_dir="/src", allow_patterns=[train_logs/final_subj*_pretrained_1sess_24bs/last.pth, evals/final_subj*_pretrained_1sess_24bs/*_all_enhancedrecons.pt])`.
- Submitted official-control 1-hour smoke: `sbatch /src/cycle5_official_control_smoke.slurm`, job `8529828`, subject 1, one A100, `01:00:00`, `64G`. At write time: `PENDING`.
- Submitted official-control full array with smoke dependency: `sbatch --dependency=afterok:8529828 /src/cycle5_official_control_full.slurm`, job `8529829_[0-3]`, subjects 1/2/5/7, one A100 each, `04:00:00`, `64G`. At write time: `PENDING (Dependency)`.

Official artifact provenance and sizes:
- Source: Hugging Face dataset `pscotti/mindeyev2`.
- Checkpoints:
  - `/src/train_logs/final_subj01_pretrained_1sess_24bs/last.pth`, `8909277522` bytes
  - `/src/train_logs/final_subj02_pretrained_1sess_24bs/last.pth`, `8885586258` bytes
  - `/src/train_logs/final_subj05_pretrained_1sess_24bs/last.pth`, `8865286482` bytes
  - `/src/train_logs/final_subj07_pretrained_1sess_24bs/last.pth`, `8859437394` bytes
- Enhanced reconstructions:
  - `/src/evals/final_subj01_pretrained_1sess_24bs/final_subj01_pretrained_1sess_24bs_all_enhancedrecons.pt`, `3145729538` bytes
  - `/src/evals/final_subj02_pretrained_1sess_24bs/final_subj02_pretrained_1sess_24bs_all_enhancedrecons.pt`, `3145729538` bytes
  - `/src/evals/final_subj05_pretrained_1sess_24bs/final_subj05_pretrained_1sess_24bs_all_enhancedrecons.pt`, `3145729538` bytes
  - `/src/evals/final_subj07_pretrained_1sess_24bs/final_subj07_pretrained_1sess_24bs_all_enhancedrecons.pt`, `3145729538` bytes

Observed metrics/results:
- No new paper-vector metrics were produced in Cycle 5 because all corrected Slurm work is still pending.
- The local baseline row remains incomplete: subjects 5/7 have trained local checkpoints, subjects 1/2 still require resumed completion, and full refined evaluation is pending.
- The official-control row is now staged and queued but not yet evaluated locally.
- No ablations were launched.

Baseline-control table status:
- Row A, paper target: unchanged from `/plan.md`.
- Row B, local fine-tuned checkpoints from official multisubject initialization: incomplete. Subjects 5/7 checkpoints exist from Cycle 2; subjects 1/2 are pending resume job `8529618`; refined metric table pending job `8529619` for subjects 5/7 and later 1/2 if resume succeeds.
- Row C, official one-session controls: checkpoints and enhanced recon tensors are staged and loadable; evaluator intermediates and metrics are pending jobs `8529828` and `8529829`.
- Acceptance gate remains open. No pass/fail call can be made until at least one four-subject metric vector is written.

Required voxel/ROI diagnostics:
- subj01: `15724` total; early `4657`; higher `11067`
- subj02: `14278` total; early `3757`; higher `10521`
- subj05: `13039` total; early `3661`; higher `9378`
- subj07: `12682` total; early `3251`; higher `9431`
- Early/higher brain-correlation summaries are still pending `final_evaluations.py` outputs. Weak-subject behavior for subjects 5 and 7 still rests only on Cycle 3 training diagnostics until full evaluator metrics finish.

Conclusions:
- Cycle 5 found the active operational blocker before compute was wasted: the pending Cycle 4 jobs were submitted with stale `/scratch/...` paths in a workspace where only `/src` is mounted.
- The official one-session control path is now available locally and verified. This gives a clean route to separate local training drift from evaluator/refinement drift once Slurm runs.
- Current mismatch classification: unresolved/incomplete baseline, with one fixed operational path issue. No model ablation is interpretable yet.

Recommended next research questions:
- Does official-control smoke job `8529828` complete and prove that local `recon_inference.py` can load `final_subj01_pretrained_1sess_24bs`?
- Do local subject 1/2 resume jobs `8529618_[0-1]` eliminate the late backward OOM after the Cycle 4 memory cleanup?
- Does local evaluation of official controls (`8529829_[0-3]`) reproduce the paper 1-hour row? If yes, prioritize local training/resume drift; if no, prioritize evaluator/refinement/candidate-pool debugging.

Cycle 5 correction after immediate Slurm feedback:
- The attempted `/src` Slurm runtime-path correction was wrong for compute nodes. Interactive Codex sees `/src`, but Slurm batch nodes require the historical `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src` path for stdout/stderr and `SRC_DIR`.
- Official smoke job `8529828` failed before user code ran: `FAILED`, exit `0:53`, elapsed `00:00:01`, node `della-l09g2`, Slurm reason `RaisedSignal:53(Real-time_signal_19)`. No stdout/stderr files were created at `/src/slurms`, consistent with the batch node not resolving the `/src` log path.
- Dependent official full array `8529829_[0-3]` became `DependencyNeverSatisfied` and was cancelled before start.
- The `/src`-path local arrays `8529618_[0-1]` and `8529619_[0-1]` were cancelled before start to avoid the same operational failure.
- Reverted Slurm runtime paths in `/src/cycle4_resume_subj12.slurm`, `/src/cycle4_eval_full_s57.slurm`, `/src/accel.slurm`, `/src/cycle5_official_control_smoke.slurm`, and `/src/cycle5_official_control_full.slurm` back to `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`; `bash -n` passed after the revert.
- Resubmitted clean scratch-path jobs:
  - local subject 1/2 resume: `8529878_[0-1]`, `PENDING`, `01:30:00`, `64G`
  - local subject 5/7 full eval: `8529879_[0-1]`, `PENDING`, `04:00:00`, `64G`
  - official-control smoke: `8529880`, `PENDING`, `01:00:00`, `64G`
  - official-control full eval: `8529881_[0-3]`, `PENDING (Dependency afterok:8529880)`, `04:00:00`, `64G`
- Updated next job IDs to watch: `8529878`, `8529879`, `8529880`, and `8529881`. Older Cycle 5 IDs `8529618`, `8529619`, `8529828`, and `8529829` are operationally superseded.
