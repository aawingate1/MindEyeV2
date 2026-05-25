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

## Cycle 6 - 2026-05-21

Plan executed:
- Read `/plan.md` and executed only the Cycle 6 baseline-control gate: reproduce a trustworthy MindEye2 one-hour control table for NSD subjects 1/2/5/7 before any ablation.
- Telegram report was not due.
- No ablation jobs were launched.

Code/config changes:
- Patched `/src/final_evaluations.py` so non-interactive metric runs can execute on Slurm compute nodes:
  - moved `sentence_transformers`, `transformers`, and `evaluate` imports below the non-interactive `sys.exit(0)` block, avoiding the stale `huggingface_hub.errors` import failure in this workflow;
  - set `TORCH_HOME`/`torch.hub` under `/src/cache/torch`;
  - pointed CLIP image-metric loading at `/src/cache/clip`.
- Added `/src/cycle6_official_final_only.slurm` to run final metrics only for the already-generated official-control intermediates.
- Added `/src/cycle6_local_final_s57.slurm` to finish final metrics for local subjects 5/7 after their enhancement tensors had already been written.
- Added `/src/cycle6_eval_full_s12.slurm` to run local subjects 1/2 through recon, enhancement, and final metrics after their resumed training completed.
- Staged evaluator weights offline for compute nodes:
  - `/src/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
  - `/src/cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth`
  - `/src/cache/torch/hub/checkpoints/efficientnet_b1-c27df63c.pth`
  - `/src/cache/torch/hub/checkpoints/swav_800ep_pretrain.pth.tar`
  - `/src/cache/clip/ViT-L-14.pt`
  - `/src/cache/torch/hub/facebookresearch_swav_main`

Commands/jobs launched and inspected:
- Official smoke `8529880`: completed successfully and proved subject 1 official `recon_inference.py` could write local evaluator intermediates.
- Official full eval `8529881_[0-3]`: generated official intermediates for subjects 1/2/5/7, then failed in `final_evaluations.py` on `ModuleNotFoundError: No module named 'huggingface_hub.errors'`.
- Official final-only retry `8534854_[0-3]`: failed quickly because compute nodes tried to download AlexNet weights and DNS failed.
- Official final-only retry `8535411_[0-3]`: completed all four subjects in about 8-9 minutes each and wrote the official-control CSVs.
- Local subject 1/2 resume `8529878_[0-1]`: completed both tasks after resuming from epoch 76, resolving the late CUDA OOM path.
  - subject 1 final resume diagnostics: train loss `5.19`, train blurry PixCorr `0.793`, test loss `15.9`, test blurry PixCorr `0.243`.
  - subject 2 final resume diagnostics: train loss `5.21`, train blurry PixCorr `0.797`, test loss `15.9`, test blurry PixCorr `0.254`.
- Local subject 5/7 full eval `8529879_[0-1]`: recon and enhancement succeeded, then final metrics failed on the same old `evaluate` import issue.
- Local subject 5/7 final-only retry `8535856_[0-1]`: completed both subjects and wrote CSVs.
- Local subject 1/2 full eval `8533288_[0-1]`: completed both subjects and wrote CSVs.
- Queue check after bookkeeping showed no remaining Cycle 6 batch jobs, only the interactive allocation.

Required voxel/ROI diagnostics:
- subj01: `15724` total; early `4657`; higher `11067`
- subj02: `14278` total; early `3757`; higher `10521`
- subj05: `13039` total; early `3661`; higher `9378`
- subj07: `12682` total; early `3251`; higher `9431`

Official-control outputs:
- CSVs:
  - `/src/tables/final_subj01_pretrained_1sess_24bs_all_enhancedrecons.csv`
  - `/src/tables/final_subj02_pretrained_1sess_24bs_all_enhancedrecons.csv`
  - `/src/tables/final_subj05_pretrained_1sess_24bs_all_enhancedrecons.csv`
  - `/src/tables/final_subj07_pretrained_1sess_24bs_all_enhancedrecons.csv`
- Metric order: PixCorr, SSIM, AlexNet-2, AlexNet-5, Inception, CLIP, EffNet dist, SwAV dist, image retrieval, brain retrieval, visual cortex, V1, V2, V3, V4, higher visual.
- subj01: `0.234518`, `0.428201`, `0.880245`, `0.933269`, `0.835659`, `0.807559`, `0.798230`, `0.458674`, `0.939556`, `0.776444`, `0.346744`, `0.318201`, `0.337088`, `0.340852`, `0.316322`, `0.344988`
- subj02: `0.200407`, `0.433062`, `0.850004`, `0.921283`, `0.818588`, `0.793912`, `0.807368`, `0.466684`, `0.905667`, `0.672000`, `0.350124`, `0.306053`, `0.296230`, `0.322867`, `0.336179`, `0.357373`
- subj05: `0.175138`, `0.405438`, `0.831098`, `0.910006`, `0.843287`, `0.825327`, `0.781252`, `0.444283`, `0.669222`, `0.469667`, `0.403527`, `0.328285`, `0.335762`, `0.323165`, `0.303573`, `0.414595`
- subj07: `0.169830`, `0.408449`, `0.807004`, `0.858992`, `0.749044`, `0.742868`, `0.854014`, `0.503689`, `0.644444`, `0.378111`, `0.293540`, `0.283393`, `0.285332`, `0.271559`, `0.243048`, `0.285348`

Official-control four-subject mean vs paper:
- PixCorr: paper `.195`, observed `.194973`, delta `-0.000027` (`-0.01%`)
- SSIM: paper `.419`, observed `.418787`, delta `-0.000213` (`-0.05%`)
- AlexNet-2: paper `.842`, observed `.842088`, delta `+0.000088` (`+0.01%`)
- AlexNet-5: paper `.906`, observed `.905888`, delta `-0.000112` (`-0.01%`)
- Inception: paper `.812`, observed `.811644`, delta `-0.000356` (`-0.04%`)
- CLIP: paper `.792`, observed `.792416`, delta `+0.000416` (`+0.05%`)
- EfficientNet distance: paper `.810`, observed `.810216`, delta `+0.000216` (`+0.03%`)
- SwAV distance: paper `.468`, observed `.468332`, delta `+0.000332` (`+0.07%`)
- Image retrieval: paper `.790`, observed `.789722`, delta `-0.000278` (`-0.04%`)
- Brain retrieval: paper `.574`, observed `.574056`, delta `+0.000056` (`+0.01%`)
- Visual cortex: paper `.348`, observed `.348483`, delta `+0.000483` (`+0.14%`)
- V1: paper `.309`, observed `.308983`, delta `-0.000017` (`-0.01%`)
- V2: paper `.314`, observed `.313603`, delta `-0.000397` (`-0.13%`)
- V3: paper `.315`, observed `.314611`, delta `-0.000389` (`-0.12%`)
- V4: paper `.300`, observed `.299781`, delta `-0.000219` (`-0.07%`)
- Higher visual: paper `.351`, observed `.350576`, delta `-0.000424` (`-0.12%`)

Official-control conclusion:
- PASS. The official one-session control reproduces the paper row to within rounding noise across reconstruction metrics, retrieval metrics, and brain-correlation ROIs.
- This validates the local evaluator, official enhanced tensors, `shared1000` handling, candidate pool, brain-correlation path, and refined-output selection.
- Weak-subject behavior is visible in the official row: subject 5 image/brain retrieval `0.669/0.470` and subject 7 `0.644/0.378`, much lower than subjects 1/2.

Local fine-tuned outputs:
- CSVs:
  - `/src/tables/cycle2_subj01_official_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle2_subj02_official_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle2_subj05_official_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle2_subj07_official_1sess_150ep_all_enhancedrecons.csv`
- subj01: `0.239807`, `0.418140`, `0.887531`, `0.935575`, `0.825261`, `0.812677`, `0.796079`, `0.447024`, `0.916000`, `0.846222`, `0.352725`, `0.322273`, `0.341952`, `0.344400`, `0.323520`, `0.352405`
- subj02: `0.209819`, `0.414476`, `0.856166`, `0.928121`, `0.818332`, `0.804996`, `0.810257`, `0.460600`, `0.895333`, `0.793556`, `0.364327`, `0.328214`, `0.312045`, `0.333570`, `0.355932`, `0.366316`
- subj05: `0.198452`, `0.412553`, `0.848701`, `0.918145`, `0.856925`, `0.846210`, `0.759550`, `0.430057`, `0.648778`, `0.535778`, `0.414329`, `0.345305`, `0.350450`, `0.334916`, `0.312673`, `0.423239`
- subj07: `0.200331`, `0.405877`, `0.826426`, `0.889542`, `0.775259`, `0.770100`, `0.830555`, `0.478162`, `0.691222`, `0.547222`, `0.317326`, `0.321064`, `0.323698`, `0.313173`, `0.281722`, `0.300001`

Local fine-tuned four-subject mean vs paper:
- PixCorr `.212102` (`+8.77%`)
- SSIM `.412762` (`-1.49%`)
- AlexNet-2 `.854706` (`+1.51%`)
- AlexNet-5 `.917846` (`+1.31%`)
- Inception `.818944` (`+0.86%`)
- CLIP `.808496` (`+2.08%`)
- EfficientNet distance `.799110` (`-1.34%`, lower is better)
- SwAV distance `.453961` (`-3.00%`, lower is better)
- Image retrieval `.787833` (`-0.27%`)
- Brain retrieval `.680694` (`+18.59%`)
- Visual cortex `.362177` (`+4.07%`)
- V1 `.329214` (`+6.54%`)
- V2 `.332036` (`+5.74%`)
- V3 `.331515` (`+5.24%`)
- V4 `.318462` (`+6.15%`)
- Higher visual `.360490` (`+2.70%`)

Mismatch classification:
- Official-control row: evaluator/control PASS.
- Initial Cycle 6 failures: operational/evaluator environment drift, fixed by lazy imports and offline metric weights.
- Local fine-tuned row: complete but non-paper-matched. Since the official-control row matches the paper, local differences should be treated as local training/resume/config drift rather than evaluator drift.

Conclusions:
- The baseline gate is now satisfied for the official one-session control row.
- Any future claimed improvement must be compared against the official-control row for paper reproduction and against the local fine-tuned row only with clear provenance separation.
- Subjects 5 and 7 remain the weakest held-out subjects on retrieval in the official baseline, making them the most useful diagnostics for early post-baseline ablations.

Recommended next research questions:
- First post-baseline ablation: projection-drift or ridge-prior regularization targeted at weak-subject generalization, with the same official-control evaluator path.
- Before treating the stronger local brain/retrieval row as a model improvement, debug why the local fine-tunes are substantially stronger than the paper brain-retrieval baseline.

## Cycle 7 - 2026-05-21

Plan executed:
- Read `/plan.md` and executed only the first constrained post-baseline ablation: projection-drift regularization for sparse one-session target-subject adaptation.
- Telegram report was not due.

Code/config changes:
- Patched `/src/Train.py` with disabled-by-default projection-drift regularization:
  - added `--proj_reg_lambda`, default `0.0`;
  - added `--proj_reg_modules`, default `ridge`;
  - snapshots selected trainable full-model parameter names after checkpoint load and before `accelerator.prepare`;
  - computes `sum(||theta_current - theta_source||_2^2)` over matched selected parameters and adds `proj_reg_lambda * proj_reg_loss` during training only when `proj_reg_lambda > 0`;
  - logs unscaled regularization loss, scaled regularization loss, absolute drift norm, relative drift norm, matched tensor count, and selected parameter count in the existing epoch log dictionary.
- Confirmed the selected default module is `ridge`; on subject 5 it matched two tensors and `53,411,840` trainable parameters.
- Preserved the existing paper config flags in Slurm wrappers: `--num_sessions=1`, `--batch_size=24`, `--hidden_dim=4096`, `--use_prior`, `--blurry_recon`, official multisubject initialization, and the validated evaluator path.
- Added `/src/cycle7_projreg_smoke.slurm` for the required one-hour subject 5 smoke.
- Added `/src/cycle7_projreg_train_s57.slurm` for the constrained weak-subject training sweep over subjects 5/7 and lambdas `0`, `1e-5`, `3e-5`, `1e-4`.
- Added `/src/cycle7_projreg_eval_s57.slurm` for the dependent refined evaluation path: `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py`.

Validation:
- `/src/fmri/bin/python -m py_compile /src/Train.py` passed.
- `bash -n /src/cycle7_projreg_smoke.slurm /src/cycle7_projreg_train_s57.slurm /src/cycle7_projreg_eval_s57.slurm` passed.
- `lambda=0` should preserve the default training path because source-parameter snapshotting, drift-stat computation, and loss addition are all guarded by `if proj_reg_lambda > 0`; a queued `lambda=0` local-control arm will verify the current code path against the ablation arms.

Commands/jobs launched:
- Submitted smoke: `sbatch /src/cycle7_projreg_smoke.slurm`, job `8539534`, subject 5, lambda `1e-4`, one A100, `64G`, `01:00:00`.
- Smoke job `8539534` completed: `COMPLETED`, exit `0:0`, elapsed `00:06:06`, node `della-l09g5`.
- Submitted weak-subject training sweep after the smoke passed: `sbatch /src/cycle7_projreg_train_s57.slurm`, job `8539988_[0-7]`, subjects 5/7, lambdas `0`, `1e-5`, `3e-5`, `1e-4`, one A100 each, `64G`, `03:00:00`.
- Submitted dependent weak-subject evaluation array: `sbatch --dependency=afterok:8539988 /src/cycle7_projreg_eval_s57.slurm`, job `8540137_[0-7]`, one A100 each, `64G`, `04:00:00`.
- At write time, `8539988_[0-7]` was `PENDING (Priority)` and `8540137_[0-7]` was `PENDING (Dependency afterok:8539988_*)`.

Smoke observed metrics:
- Checkpoint load and training started from `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/train_logs/final_multisubject_subj05/last.pth`.
- Regularizer initialization log: `lambda=0.0001`, modules `('ridge',)`, tensors `2`, params `53,411,840`.
- Final epoch smoke diagnostics:
  - train loss `11.4`; test loss `13.1`;
  - train blurry PixCorr `0.341`; test blurry PixCorr `0.250`;
  - train fwd/bwd top-1 `0.992/0.972`; test fwd/bwd top-1 `0.407/0.283`;
  - unscaled projection loss `37.7`; scaled projection loss `0.00377`;
  - absolute drift norm `6.13`; relative drift norm `0.165`;
  - matched tensors `2`; selected params `5.34e+7`.
- Smoke failure class: none. No checkpoint-load, loss-addition, logging, output-path, or CUDA-memory failure was observed.

Baseline and comparison status:
- Official-control paper-matched row from Cycle 6 remains the primary reference:
  - weak subject 5 image/brain retrieval `0.669222/0.469667`, visual cortex/higher visual `0.403527/0.414595`;
  - weak subject 7 image/brain retrieval `0.644444/0.378111`, visual cortex/higher visual `0.293540/0.285348`.
- The Cycle 7 sweep has not produced refined metrics yet. No claim is available until the dependent evaluation array writes per-subject CSVs.
- The stronger local Cycle 2 fine-tuned row remains classified as non-paper-matched local training/resume/config drift and is not the primary baseline for this ablation.

Artifact/checkpoint provenance:
- Official multisubject initialization for Cycle 7 training is Hugging Face `pscotti/mindeyev2`, staged under `/src/train_logs/final_multisubject_subj0{5,7}/last.pth` for the weak-subject first pass.
- Smoke model name: `cycle7_smoke_subj05_projreg1e-4_3ep`.
- Sweep model names: `cycle7_subj0{5,7}_projreg{0,1em5,3em5,1em4}_1sess_150ep`.
- Pending refined outputs and CSVs will be produced under `/src/evals/<model_name>/` and `/src/tables/<model_name>_all_enhancedrecons.csv` after job `8540137_[0-7]` runs.

Conclusions:
- The projection-drift plumbing is implemented and active only when requested.
- The required one-hour preflight passed on subject 5 with nonzero drift/loss diagnostics, so the weak-subject sweep was launched.
- The first sweep intentionally covers subjects 5 and 7 only because they are the weak-subject diagnostics and queue pressure is nontrivial. Any final four-subject claim still requires subjects 1/2/5/7.

Recommended next research questions:
- Do the queued weak-subject lambda arms complete within `03:00:00`, and does the `lambda=0` arm reproduce the current local training behavior closely enough to serve as the local-control row?
- Which lambda, if any, improves subjects 5/7 brain retrieval while preserving CLIP, Inception, and image retrieval against the official-control row?
- If all lambdas underfit or semantic/retrieval metrics degrade, reduce the lambda grid or restrict regularization to fewer ridge parameters before scaling to subjects 1/2.

## Cycle 8 - 2026-05-21

Plan executed:
- Read `/plan.md` and executed only the active projection-drift sweep readout path.
- Telegram report was not due.
- No generator/refiner, ROI-routing, temporal-decoding, caption/VLM, low-rank-adapter, or relational-consistency experiment was started.

Code/config changes:
- No changes were made to `Train.py`, `models.py`, or Slurm training/evaluation scripts in Cycle 8.
- Added the train-only diagnostic artifact `/workspace/myresearch/cycle8_train_repeat_reliability.json`.

Cycle 7 active job status:
- Training array `8539988_[0-7]` is still pending under Slurm priority; no task has started and no training stdout/stderr exists yet for the array.
- Evaluation array `8540137_[0-7]` is still pending on the `afterok:8539988` dependency; no evaluation task has started and no evaluation stdout/stderr exists yet.
- `squeue` status at Cycle 8 inspection:
  - `8539988_0`: subject 5, lambda `0`, `PENDING (Priority)`, exit `0:0`, node none, elapsed `00:00:00`, stdout `/src/slurms/c7_projreg_s57_8539988_0.out`, stderr `/src/slurms/c7_projreg_s57_8539988_0.err` not yet created.
  - `8539988_1`: subject 5, lambda `1e-5`, `PENDING (Priority)`, exit `0:0`, node none, elapsed `00:00:00`, stdout `/src/slurms/c7_projreg_s57_8539988_1.out`, stderr `/src/slurms/c7_projreg_s57_8539988_1.err` not yet created.
  - `8539988_2`: subject 5, lambda `3e-5`, `PENDING (Priority)`, exit `0:0`, node none, elapsed `00:00:00`, stdout `/src/slurms/c7_projreg_s57_8539988_2.out`, stderr `/src/slurms/c7_projreg_s57_8539988_2.err` not yet created.
  - `8539988_3`: subject 5, lambda `1e-4`, `PENDING (Priority)`, exit `0:0`, node none, elapsed `00:00:00`, stdout `/src/slurms/c7_projreg_s57_8539988_3.out`, stderr `/src/slurms/c7_projreg_s57_8539988_3.err` not yet created.
  - `8539988_4`: subject 7, lambda `0`, `PENDING (Priority)`, exit `0:0`, node none, elapsed `00:00:00`, stdout `/src/slurms/c7_projreg_s57_8539988_4.out`, stderr `/src/slurms/c7_projreg_s57_8539988_4.err` not yet created.
  - `8539988_5`: subject 7, lambda `1e-5`, `PENDING (Priority)`, exit `0:0`, node none, elapsed `00:00:00`, stdout `/src/slurms/c7_projreg_s57_8539988_5.out`, stderr `/src/slurms/c7_projreg_s57_8539988_5.err` not yet created.
  - `8539988_6`: subject 7, lambda `3e-5`, `PENDING (Priority)`, exit `0:0`, node none, elapsed `00:00:00`, stdout `/src/slurms/c7_projreg_s57_8539988_6.out`, stderr `/src/slurms/c7_projreg_s57_8539988_6.err` not yet created.
  - `8539988_7`: subject 7, lambda `1e-4`, `PENDING (Priority)`, exit `0:0`, node none, elapsed `00:00:00`, stdout `/src/slurms/c7_projreg_s57_8539988_7.out`, stderr `/src/slurms/c7_projreg_s57_8539988_7.err` not yet created.
  - `8540137_[0-7]`: dependent evaluation array, `PENDING (Dependency)`, exit `0:0`, node none, elapsed `00:00:00`; expected stdout/stderr `/src/slurms/c7_projreg_eval_s57_8540137_<task>.out/.err` not yet created.
- Failure class for all above tasks: none yet; operational state is queue-pending, not failed.

Completed projection-drift smoke diagnostics:
- Smoke job `8539534`, subject 5, lambda `1e-4`, completed before Cycle 8 and remains the only available projection-drift training trace.
- Drift curve by epoch:
  - epoch 1: train loss `15.2`, test loss `14.9`, train/test blurry PixCorr `0.183/0.231`, train/test fwd retrieval `0.387/0.157`, train/test bwd retrieval `0.159/0.040`, unscaled projection loss `1.03`, scaled projection loss `0.000103`, drift norm `0.762`, relative drift `0.0206`, matched tensors `2`, selected params `5.34e7`.
  - epoch 2: train loss `11.5`, test loss `11.4`, train/test blurry PixCorr `0.254/0.227`, train/test fwd retrieval `0.898/0.380`, train/test bwd retrieval `0.755/0.200`, unscaled projection loss `18.5`, scaled projection loss `0.00185`, drift norm `4.18`, relative drift `0.113`, matched tensors `2`, selected params `5.34e7`.
  - epoch 3: train loss `11.4`, test loss `13.1`, train/test blurry PixCorr `0.341/0.250`, train/test fwd retrieval `0.992/0.407`, train/test bwd retrieval `0.972/0.283`, unscaled projection loss `37.7`, scaled projection loss `0.00377`, drift norm `6.13`, relative drift `0.165`, matched tensors `2`, selected params `5.34e7`.
- There is not yet a `lambda=0` full-run drift/loss curve for same-code local control; it is task `8539988_0` and remains pending.

Available weak-subject control metric rows:

| row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist | SwAV dist | image retrieval | brain retrieval | visual cortex | V1 | V2 | V3 | V4 | higher visual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| official-control subj05 | 0.175138 | 0.405438 | 0.831098 | 0.910006 | 0.843287 | 0.825327 | 0.781252 | 0.444283 | 0.669222 | 0.469667 | 0.403527 | 0.328285 | 0.335762 | 0.323165 | 0.303573 | 0.414595 |
| official-control subj07 | 0.169830 | 0.408449 | 0.807004 | 0.858992 | 0.749044 | 0.742868 | 0.854014 | 0.503689 | 0.644444 | 0.378111 | 0.293540 | 0.283393 | 0.285332 | 0.271559 | 0.243048 | 0.285348 |
| local Cycle 2 subj05 | 0.198452 | 0.412553 | 0.848701 | 0.918145 | 0.856925 | 0.846210 | 0.759550 | 0.430057 | 0.648778 | 0.535778 | 0.414329 | 0.345305 | 0.350450 | 0.334916 | 0.312673 | 0.423239 |
| local Cycle 2 subj07 | 0.200331 | 0.405877 | 0.826426 | 0.889542 | 0.775259 | 0.770100 | 0.830555 | 0.478162 | 0.691222 | 0.547222 | 0.317326 | 0.321064 | 0.323698 | 0.313173 | 0.281722 | 0.300001 |
| official weak mean 5/7 | 0.172484 | 0.406944 | 0.819051 | 0.884499 | 0.796166 | 0.784098 | 0.817633 | 0.473986 | 0.656833 | 0.423889 | 0.348533 | 0.305839 | 0.310547 | 0.297362 | 0.273311 | 0.349971 |
| local Cycle 2 weak mean 5/7 | 0.199392 | 0.409215 | 0.837564 | 0.903843 | 0.816092 | 0.808155 | 0.795053 | 0.454109 | 0.670000 | 0.541500 | 0.365828 | 0.333184 | 0.337074 | 0.324045 | 0.297198 | 0.361620 |

Projection-drift sweep metric status:
- `lambda=0`, `1e-5`, `3e-5`, and `1e-4` CSVs do not exist yet for subjects 5 or 7 because training/evaluation arrays have not started.
- No Cycle 8 decision table can honestly rank lambdas yet. The official-control and Cycle 2 local rows above are only provenance controls; the required same-code local-control `lambda=0` row is still pending.
- EfficientNet and SwAV are distance metrics, so lower is better when the sweep rows become available.

Train-only reliability diagnostic:
- Method: for each subject, use only training betas and `COCO_73k_subj_indices.hdf5`; compute per-voxel Pearson correlation across repeated training image IDs between the first and second occurrence of each repeated ID, then summarize by voxel masks. No `shared1000`, `new_test`, or final-evaluator test repeats were used.
- One-session subset (`n=750`) has `136` repeated image IDs and `303` repeated-image trials for each subject.
- Full training set (`n=30000`) has `10000` repeated image IDs and `30000` repeated-image trials for each subject.
- Voxel counts match the required masks: subj01 total `15724`, early `4657`, higher `11067`; subj02 total `14278`, early `3757`, higher `10521`; subj05 total `13039`, early `3661`, higher `9378`; subj07 total `12682`, early `3251`, higher `9431`.

| subject/window | all mean r | early mean r | higher mean r | all median r | early median r | higher median r |
|---|---|---|---|---|---|---|
| subj01 one-session 750 | 0.162033 | 0.232150 | 0.132528 | 0.142471 | 0.230462 | 0.117591 |
| subj01 all-train 30000 | 0.164854 | 0.212194 | 0.144933 | 0.141908 | 0.200937 | 0.121222 |
| subj02 one-session 750 | 0.180506 | 0.281447 | 0.144460 | 0.145721 | 0.268344 | 0.122151 |
| subj02 all-train 30000 | 0.183543 | 0.228757 | 0.167398 | 0.157422 | 0.205648 | 0.146640 |
| subj05 one-session 750 | 0.221498 | 0.277334 | 0.199700 | 0.211197 | 0.277415 | 0.189570 |
| subj05 all-train 30000 | 0.202433 | 0.205930 | 0.201068 | 0.186557 | 0.192253 | 0.183862 |
| subj07 one-session 750 | 0.166190 | 0.255124 | 0.135533 | 0.145981 | 0.258435 | 0.123987 |
| subj07 all-train 30000 | 0.125508 | 0.151110 | 0.116682 | 0.112246 | 0.139390 | 0.104818 |

Interpretation:
- Subject 5 has comparatively high train-repeat reliability, including higher visual cortex, despite weak official-control image/brain retrieval. This supports the plan's interpretation of subject 5 as more likely a retrieval-geometry or adaptation bottleneck than a pure signal-quality bottleneck.
- Subject 7 has the weakest full-train reliability among the four subjects in both early and higher visual masks, matching the plan's signal/alignment-limited concern.
- These reliability values are interpretation-only for Cycle 8 and were not used to tune voxel weights or launch a new mechanism.

Decision:
- Do not scale to subjects 1/2 yet. The success criterion depends on completed `lambda=0`, `1e-5`, `3e-5`, and `1e-4` refined metrics for subjects 5/7, and those runs are still pending.
- Do not launch a second mechanism. The correct next action is to let `8539988_[0-7]` start/finish, then parse logs and run or repair dependent evaluation `8540137_[0-7]` if needed.
- If training completes but evaluation remains blocked, rerun only the missing `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py` steps using the Cycle 6 validated scratch-path/offline-cache setup.

## Cycle 9 - 2026-05-21

Plan executed:
- Read `/plan.md` and executed only the active Cycle 7 projection-drift sweep readout/monitoring path.
- Telegram report was not due.
- No generator/refiner, temporal decoding, ROI routing, caption/VLM correction, reliability weighting, low-rank adapter, relational-consistency, or second mechanism was started.

Code/config changes:
- No changes were made to `/src/Train.py`, `/src/models.py`, or Slurm scripts in Cycle 9.
- No jobs were cancelled, resubmitted, or newly launched. The active sweep is validly running, so the experiment design was left untouched.

Commands/status checks:
- Ran `squeue -j 8539988,8540137`.
- Ran `sacct -j 8539988,8540137 --format=JobID,JobName%35,State,ExitCode,Elapsed,NodeList,MaxRSS,ReqMem,Submit,Start,End -P`.
- Parsed `/src/slurms/c7_projreg_s57_8539988_<task>.out/.err` for task mapping, checkpoint load, early progress, and live training diagnostics.
- Checked that no Cycle 7 refined CSVs exist yet under `/src/tables/*cycle7_subj0*_projreg*_1sess_150ep*`, as expected because evaluation has not started.

Active Slurm state at final Cycle 9 check:

| task | subject | lambda | state | exit | node | elapsed | latest epoch | stdout | stderr | expected checkpoint |
|---|---:|---:|---|---|---|---|---:|---|---|---|
| `8539988_0` | 5 | `0` | RUNNING | `0:0` | `della-l03g3` | `00:51:22` | `53/150` | `/src/slurms/c7_projreg_s57_8539988_0.out` | `/src/slurms/c7_projreg_s57_8539988_0.err` | compute-visible `../train_logs/cycle7_subj05_projreg0_1sess_150ep/last.pth` |
| `8539988_1` | 5 | `1e-5` | RUNNING | `0:0` | `della-l03g6` | `00:47:47` | `49/150` | `/src/slurms/c7_projreg_s57_8539988_1.out` | `/src/slurms/c7_projreg_s57_8539988_1.err` | compute-visible `../train_logs/cycle7_subj05_projreg1em5_1sess_150ep/last.pth` |
| `8539988_2` | 5 | `3e-5` | RUNNING | `0:0` | `della-l02g3` | `00:46:15` | `48/150` | `/src/slurms/c7_projreg_s57_8539988_2.out` | `/src/slurms/c7_projreg_s57_8539988_2.err` | compute-visible `../train_logs/cycle7_subj05_projreg3em5_1sess_150ep/last.pth` |
| `8539988_3` | 5 | `1e-4` | RUNNING | `0:0` | `della-l04g7` | `00:43:43` | `45/150` | `/src/slurms/c7_projreg_s57_8539988_3.out` | `/src/slurms/c7_projreg_s57_8539988_3.err` | compute-visible `../train_logs/cycle7_subj05_projreg1em4_1sess_150ep/last.pth` |
| `8539988_4` | 7 | `0` | RUNNING | `0:0` | `della-l05g4` | `00:41:41` | `43/150` | `/src/slurms/c7_projreg_s57_8539988_4.out` | `/src/slurms/c7_projreg_s57_8539988_4.err` | compute-visible `../train_logs/cycle7_subj07_projreg0_1sess_150ep/last.pth` |
| `8539988_5` | 7 | `1e-5` | RUNNING | `0:0` | `della-l04g3` | `00:41:41` | `43/150` | `/src/slurms/c7_projreg_s57_8539988_5.out` | `/src/slurms/c7_projreg_s57_8539988_5.err` | compute-visible `../train_logs/cycle7_subj07_projreg1em5_1sess_150ep/last.pth` |
| `8539988_6` | 7 | `3e-5` | RUNNING | `0:0` | `della-l04g14` | `00:40:40` | `42/150` | `/src/slurms/c7_projreg_s57_8539988_6.out` | `/src/slurms/c7_projreg_s57_8539988_6.err` | compute-visible `../train_logs/cycle7_subj07_projreg3em5_1sess_150ep/last.pth` |
| `8539988_7` | 7 | `1e-4` | RUNNING | `0:0` | `della-l03g13` | `00:40:10` | `41/150` | `/src/slurms/c7_projreg_s57_8539988_7.out` | `/src/slurms/c7_projreg_s57_8539988_7.err` | compute-visible `../train_logs/cycle7_subj07_projreg1em4_1sess_150ep/last.pth` |
| `8540137_[0-7]` | 5/7 | `0,1e-5,3e-5,1e-4` | PENDING | `0:0` | none | `00:00:00` | not started | `/src/slurms/c7_projreg_eval_s57_8540137_<task>.out` | `/src/slurms/c7_projreg_eval_s57_8540137_<task>.err` | uses trained checkpoints above |

Failure classification:
- Training array `8539988_[0-7]`: no failure observed. All tasks are running, loaded the intended `final_multisubject_subj0{5,7}/last.pth` checkpoint, and reached epoch-loop logging.
- Evaluation array `8540137_[0-7]`: no failure observed. It is correctly pending on the `afterok:8539988` dependency and has not started.
- The Slurm warning `couldn't chdir to /workspace` appears in each training stderr, but each script immediately `cd`s to the scratch-visible `/src` path and proceeds normally. This is not currently a failure class.

Live training diagnostics snapshot:

| task | subject | lambda | epoch | train loss | test loss | train/test blurry PixCorr | train fwd/bwd | test fwd/bwd | unscaled reg | scaled reg | drift norm | relative drift | matched tensors | selected params |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| `8539988_0` | 5 | `0` | 53 | `9.8` | `18.9` | `0.670/0.187` | `1.000/1.000` | `0.533/0.423` | `0` | `0` | `0` | `0` | 0 | 0 |
| `8539988_1` | 5 | `1e-5` | 49 | `10.9` | `14.7` | `0.506/0.144` | `0.737/0.786` | `0.460/0.377` | `802` | `0.00802` | `28.3` | `0.764` | 2 | `5.34e7` |
| `8539988_2` | 5 | `3e-5` | 48 | `10.0` | `15.6` | `0.488/0.236` | `0.718/0.778` | `0.550/0.347` | `764` | `0.0229` | `27.6` | `0.746` | 2 | `5.34e7` |
| `8539988_3` | 5 | `1e-4` | 45 | `10.7` | `15.7` | `0.465/0.166` | `0.659/0.737` | `0.457/0.310` | `592` | `0.0592` | `24.3` | `0.657` | 2 | `5.34e7` |
| `8539988_4` | 7 | `0` | 43 | `11.0` | `16.2` | `0.431/0.129` | `0.702/0.765` | `0.523/0.330` | `0` | `0` | `0` | `0` | 0 | 0 |
| `8539988_5` | 7 | `1e-5` | 43 | `10.9` | `16.0` | `0.442/0.137` | `0.706/0.763` | `0.523/0.360` | `759` | `0.00759` | `27.5` | `0.744` | 2 | `5.19e7` |
| `8539988_6` | 7 | `3e-5` | 42 | `10.6` | `15.7` | `0.487/0.211` | `0.706/0.778` | `0.520/0.353` | `689` | `0.0207` | `26.2` | `0.709` | 2 | `5.19e7` |
| `8539988_7` | 7 | `1e-4` | 41 | `10.5` | `18.4` | `0.420/0.203` | `0.691/0.773` | `0.477/0.323` | `588` | `0.0588` | `24.3` | `0.655` | 2 | `5.19e7` |

Metric table status:
- No refined Cycle 7 projection-drift CSV rows exist yet for `lambda=0`, `1e-5`, `3e-5`, or `1e-4`.
- The official-control and Cycle 2 local metric rows from Cycle 8 remain the only completed weak-subject metric controls.
- Because no Cycle 7 refined metrics exist yet, there is no valid lambda ranking, no weak-subject mean table, and no scale/stop scientific decision in Cycle 9.

Conclusion:
- The Cycle 7 projection-drift sweep is no longer stale or invalid; it is actively running and producing regularization diagnostics.
- The correct action was to monitor and not mutate the experiment. The next cycle should inspect final state for `8539988_[0-7]`; if all tasks complete, let or inspect dependent evaluator `8540137_[0-7]`. If evaluation fails or never launches, rerun only the validated Cycle 6 evaluator path for the exact completed model names.

## Cycle 10 - 2026-05-21

Plan executed:
- Read `/plan.md` and executed only the Cycle 7 projection-drift recovery/readout path.
- Telegram report is due; concise report-ready update is appended below.
- No generator/refiner, temporal decoding, ROI routing, caption/VLM correction, reliability weighting, low-rank adapter, relational-consistency, or other second mechanism was started.

Code/config changes:
- No changes were made to `/src/Train.py`, `/src/models.py`, or any Slurm scripts.
- No jobs were cancelled, resubmitted, or newly launched. The active training sweep is validly running, so the plan's instruction was to record status and not mutate the experiment.

Commands/status checks:
- Ran `squeue -j 8539988,8540137`.
- Ran `sacct -j 8539988,8540137 --format=JobID,JobName%35,State,ExitCode,Elapsed,NodeList,MaxRSS,ReqMem,Submit,Start,End -P`.
- Parsed `/src/slurms/c7_projreg_s57_8539988_<task>.out/.err` for subject/lambda mapping, current epoch, losses, blurry PixCorr, retrieval diagnostics, and projection-drift diagnostics.
- Checked `/src/tables` for `cycle7_subj0*_projreg*_1sess_150ep` CSVs; none exist yet because evaluation has not started.

Active Slurm state at 2026-05-21 04:54 EDT:

| task | subject | lambda | state | exit | node | elapsed | stdout | stderr | expected checkpoint |
|---|---:|---:|---|---|---|---|---|---|---|
| `8539988_0` | 5 | `0` | RUNNING | `0:0` | `della-l03g3` | `01:11:17` | `/src/slurms/c7_projreg_s57_8539988_0.out` | `/src/slurms/c7_projreg_s57_8539988_0.err` | `../train_logs/cycle7_subj05_projreg0_1sess_150ep/last.pth` |
| `8539988_1` | 5 | `1e-5` | RUNNING | `0:0` | `della-l03g6` | `01:07:42` | `/src/slurms/c7_projreg_s57_8539988_1.out` | `/src/slurms/c7_projreg_s57_8539988_1.err` | `../train_logs/cycle7_subj05_projreg1em5_1sess_150ep/last.pth` |
| `8539988_2` | 5 | `3e-5` | RUNNING | `0:0` | `della-l02g3` | `01:06:10` | `/src/slurms/c7_projreg_s57_8539988_2.out` | `/src/slurms/c7_projreg_s57_8539988_2.err` | `../train_logs/cycle7_subj05_projreg3em5_1sess_150ep/last.pth` |
| `8539988_3` | 5 | `1e-4` | RUNNING | `0:0` | `della-l04g7` | `01:03:38` | `/src/slurms/c7_projreg_s57_8539988_3.out` | `/src/slurms/c7_projreg_s57_8539988_3.err` | `../train_logs/cycle7_subj05_projreg1em4_1sess_150ep/last.pth` |
| `8539988_4` | 7 | `0` | RUNNING | `0:0` | `della-l05g4` | `01:01:36` | `/src/slurms/c7_projreg_s57_8539988_4.out` | `/src/slurms/c7_projreg_s57_8539988_4.err` | `../train_logs/cycle7_subj07_projreg0_1sess_150ep/last.pth` |
| `8539988_5` | 7 | `1e-5` | RUNNING | `0:0` | `della-l04g3` | `01:01:36` | `/src/slurms/c7_projreg_s57_8539988_5.out` | `/src/slurms/c7_projreg_s57_8539988_5.err` | `../train_logs/cycle7_subj07_projreg1em5_1sess_150ep/last.pth` |
| `8539988_6` | 7 | `3e-5` | RUNNING | `0:0` | `della-l04g14` | `01:00:35` | `/src/slurms/c7_projreg_s57_8539988_6.out` | `/src/slurms/c7_projreg_s57_8539988_6.err` | `../train_logs/cycle7_subj07_projreg3em5_1sess_150ep/last.pth` |
| `8539988_7` | 7 | `1e-4` | RUNNING | `0:0` | `della-l03g13` | `01:00:05` | `/src/slurms/c7_projreg_s57_8539988_7.out` | `/src/slurms/c7_projreg_s57_8539988_7.err` | `../train_logs/cycle7_subj07_projreg1em4_1sess_150ep/last.pth` |
| `8540137_[0-7]` | 5/7 | `0,1e-5,3e-5,1e-4` | PENDING | `0:0` | none | `00:00:00` | `/src/slurms/c7_projreg_eval_s57_8540137_<task>.out` | `/src/slurms/c7_projreg_eval_s57_8540137_<task>.err` | dependent evaluator for completed checkpoints above |

Failure classification:
- Training array `8539988_[0-7]`: no failure observed. All tasks are running on assigned A100 nodes, loaded the intended `final_multisubject_subj0{5,7}/last.pth` checkpoint, and are writing epoch diagnostics.
- Evaluation array `8540137_[0-7]`: no failure observed. It is still correctly pending on `afterok:8539988` and has not started.
- Each training stderr includes Slurm's `couldn't chdir to /workspace` warning before the script `cd`s to `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`; this remains non-fatal because training proceeds normally.

Latest live training diagnostics:

| task | subject | lambda | latest epoch | train loss | test loss | train/test blurry PixCorr | train fwd/bwd | test fwd/bwd | unscaled reg | scaled reg | drift norm | relative drift | matched tensors | selected params |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| `8539988_0` | 5 | `0` | 75/150 | `7.27` | `15.1` | `0.747/0.222` | `1.000/1.000` | `0.603/0.507` | `0` | `0` | `0` | `0` | 0 | 0 |
| `8539988_1` | 5 | `1e-5` | 71/150 | `7.97` | `16.9` | `0.747/0.206` | `1.000/1.000` | `0.613/0.483` | `1.14e3` | `0.0114` | `33.8` | `0.911` | 2 | `5.34e7` |
| `8539988_2` | 5 | `3e-5` | 70/150 | `8.49` | `15.2` | `0.741/0.133` | `1.000/1.000` | `0.600/0.477` | `1.02e3` | `0.0307` | `32.0` | `0.863` | 2 | `5.34e7` |
| `8539988_3` | 5 | `1e-4` | 67/150 | `8.78` | `15.7` | `0.718/0.191` | `1.000/1.000` | `0.603/0.453` | `821` | `0.0821` | `28.7` | `0.773` | 2 | `5.34e7` |
| `8539988_4` | 7 | `0` | 64/150 | `8.13` | `13.0` | `0.726/0.228` | `1.000/1.000` | `0.687/0.500` | `0` | `0` | `0` | `0` | 0 | 0 |
| `8539988_5` | 7 | `1e-5` | 65/150 | `8.67` | `16.4` | `0.721/0.229` | `1.000/1.000` | `0.653/0.490` | `1.19e3` | `0.0119` | `34.4` | `0.930` | 2 | `5.19e7` |
| `8539988_6` | 7 | `3e-5` | 63/150 | `8.22` | `16.0` | `0.708/0.142` | `1.000/1.000` | `0.663/0.483` | `1.08e3` | `0.0324` | `32.9` | `0.887` | 2 | `5.19e7` |
| `8539988_7` | 7 | `1e-4` | 62/150 | `8.61` | `16.0` | `0.688/0.216` | `1.000/1.000` | `0.633/0.463` | `865` | `0.0865` | `29.4` | `0.794` | 2 | `5.19e7` |

Metric table status:
- No refined Cycle 7 projection-drift CSV rows exist yet for `lambda=0`, `1e-5`, `3e-5`, or `1e-4`.
- Because `8540137_[0-7]` has not started, there are still no PixCorr/SSIM/AlexNet/Inception/CLIP/EfficientNet/SwAV/image-retrieval/brain-retrieval/ROI-correlation rows for the projection-regularized sweep.
- The official-control and local Cycle 2 weak-subject rows from Cycle 8 remain the only completed controls; no valid lambda ranking or weak-subject mean table can be produced yet.

Conclusion and decision:
- The Cycle 7 projection-drift sweep is actively running and has not failed, so no repair or rerun was appropriate in Cycle 10.
- Early live diagnostics do not justify a scientific decision; several regularized arms show lower or noisier same-epoch test retrieval than `lambda=0`, but these are mid-run training diagnostics and not final refined metrics.
- Do not scale to subjects 1/2 and do not launch a new mechanism. The next valid action is to inspect final state for `8539988_[0-7]`; if all training tasks complete, let `8540137_[0-7]` run. If evaluation fails, rerun only the validated Cycle 6 evaluator path for the exact Cycle 7 model names.

Telegram-ready update:
Cycle 10 monitored the active Cycle 7 projection-drift sweep only. All eight training tasks `8539988_[0-7]` are running on A100 nodes with no observed failures; subjects 5 and 7 are being trained at lambdas `0`, `1e-5`, `3e-5`, and `1e-4`. The dependent evaluation array `8540137_[0-7]` is still pending correctly on `afterok:8539988`, so no refined CSV metrics or plots exist yet. Latest mid-run epochs are roughly 62-75/150. The regularizer is active in nonzero arms with 2 matched ridge tensors, about `5.19e7-5.34e7` selected params, drift norms around `28.7-34.4`, and scaled regularization losses around `0.011-0.087`. No jobs were changed or relaunched because the sweep is validly running. Next report should parse final training logs and the dependent refined evaluation tables once `8540137` finishes.

## Cycle 11 - 2026-05-21

Plan executed:
- Read `/plan.md` and executed only Cycle 7 projection-drift recovery/readout.
- Telegram report was not due.
- No reliability weighting, relational consistency, CLIP-layer fusion, low-rank adapter, generator/refiner, ROI routing, temporal decoding, caption/VLM correction, or other second mechanism was started.

Code/config changes:
- No changes were made to `/src/Train.py`, `/src/models.py`, or Slurm scripts in Cycle 11.
- No jobs were cancelled, resubmitted, or newly launched. The active training and evaluation arrays were valid, so the correct action was to monitor and parse them.

Operational visibility:
- This shell could query `squeue`/`sacct` and read `/src`, but the compute-side path `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/...` was not mounted locally. This did not block execution because all Slurm tasks ran on compute nodes and wrote logs/tables visible under `/src`.

Commands/status checks:
- Ran `squeue -j 8539988,8540137`.
- Ran `sacct -j 8539988,8540137 --format=JobID,JobName,State,ExitCode,NodeList,Elapsed,Start,End,ReqMem,MaxRSS -P`.
- Parsed `/src/slurms/c7_projreg_s57_8539988_<task>.err` for final train/test losses, blurry PixCorr, retrieval diagnostics, and projection-drift diagnostics.
- Parsed `/src/tables/cycle7_subj0{5,7}_projreg{0,1em5,3em5,1em4}_1sess_150ep_all_enhancedrecons.csv`.

Final Slurm status and failure classification:

| task | subject | lambda | state | exit | node | elapsed | stdout/stderr | expected checkpoint/artifacts | failure class |
|---|---:|---:|---|---|---|---|---|---|---|
| `8539988_0` | 5 | `0` | COMPLETED | `0:0` | `della-l03g3` | `02:18:43` | `/src/slurms/c7_projreg_s57_8539988_0.out/.err` | `cycle7_subj05_projreg0_1sess_150ep` checkpoint used by evaluator | none |
| `8539988_1` | 5 | `1e-5` | COMPLETED | `0:0` | `della-l03g6` | `02:18:39` | `/src/slurms/c7_projreg_s57_8539988_1.out/.err` | `cycle7_subj05_projreg1em5_1sess_150ep` checkpoint used by evaluator | none |
| `8539988_2` | 5 | `3e-5` | COMPLETED | `0:0` | `della-l02g3` | `02:18:02` | `/src/slurms/c7_projreg_s57_8539988_2.out/.err` | `cycle7_subj05_projreg3em5_1sess_150ep` checkpoint used by evaluator | none |
| `8539988_3` | 5 | `1e-4` | COMPLETED | `0:0` | `della-l04g7` | `02:18:11` | `/src/slurms/c7_projreg_s57_8539988_3.out/.err` | `cycle7_subj05_projreg1em4_1sess_150ep` checkpoint used by evaluator | none |
| `8539988_4` | 7 | `0` | COMPLETED | `0:0` | `della-l05g4` | `02:18:35` | `/src/slurms/c7_projreg_s57_8539988_4.out/.err` | `cycle7_subj07_projreg0_1sess_150ep` checkpoint used by evaluator | none |
| `8539988_5` | 7 | `1e-5` | COMPLETED | `0:0` | `della-l04g3` | `02:18:31` | `/src/slurms/c7_projreg_s57_8539988_5.out/.err` | `cycle7_subj07_projreg1em5_1sess_150ep` checkpoint used by evaluator | none |
| `8539988_6` | 7 | `3e-5` | COMPLETED | `0:0` | `della-l04g14` | `02:18:28` | `/src/slurms/c7_projreg_s57_8539988_6.out/.err` | `cycle7_subj07_projreg3em5_1sess_150ep` checkpoint used by evaluator | none |
| `8539988_7` | 7 | `1e-4` | COMPLETED | `0:0` | `della-l03g13` | `02:18:54` | `/src/slurms/c7_projreg_s57_8539988_7.out/.err` | `cycle7_subj07_projreg1em4_1sess_150ep` checkpoint used by evaluator | none |
| `8540137_0` | 5 | `0` | COMPLETED | `0:0` | `della-l05g5` | `02:23:30` | `/src/slurms/c7_projreg_eval_s57_8540137_0.out/.err` | CSV and enhanced recon tensor written | none |
| `8540137_1` | 5 | `1e-5` | COMPLETED | `0:0` | `della-l02g1` | `02:26:13` | `/src/slurms/c7_projreg_eval_s57_8540137_1.out/.err` | CSV and enhanced recon tensor written | none |
| `8540137_2` | 5 | `3e-5` | COMPLETED | `0:0` | `della-l02g4` | `02:26:04` | `/src/slurms/c7_projreg_eval_s57_8540137_2.out/.err` | CSV and enhanced recon tensor written | none |
| `8540137_3` | 5 | `1e-4` | COMPLETED | `0:0` | `della-l03g3` | `02:26:19` | `/src/slurms/c7_projreg_eval_s57_8540137_3.out/.err` | CSV and enhanced recon tensor written | none |
| `8540137_4` | 7 | `0` | COMPLETED | `0:0` | `della-l03g10` | `02:27:26` | `/src/slurms/c7_projreg_eval_s57_8540137_4.out/.err` | CSV and enhanced recon tensor written | none |
| `8540137_5` | 7 | `1e-5` | COMPLETED | `0:0` | `della-l03g9` | `02:23:00` | `/src/slurms/c7_projreg_eval_s57_8540137_5.out/.err` | CSV and enhanced recon tensor written | none |
| `8540137_6` | 7 | `3e-5` | COMPLETED | `0:0` | `della-l05g1` | `02:23:47` | `/src/slurms/c7_projreg_eval_s57_8540137_6.out/.err` | CSV and enhanced recon tensor written | none |
| `8540137_7` | 7 | `1e-4` | COMPLETED | `0:0` | `della-l04g8` | `02:25:32` | `/src/slurms/c7_projreg_eval_s57_8540137_7.out/.err` | CSV and enhanced recon tensor written | none |

Final training diagnostics:

| subject | lambda | test loss | test blurry PixCorr | test fwd/bwd | train loss | train blurry PixCorr | train fwd/bwd | unscaled reg | scaled reg | drift norm | relative drift | matched tensors | selected params |
|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | `0` | 14.5 | 0.198 | 0.657/0.553 | 5.89 | 0.800 | 1.000/1.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | `1e-5` | 14.5 | 0.202 | 0.663/0.543 | 5.90 | 0.800 | 1.000/1.000 | 1210 | 0.0121 | 34.8 | 0.938 | 2 | 5.34e7 |
| 5 | `3e-5` | 14.5 | 0.187 | 0.633/0.520 | 5.94 | 0.798 | 1.000/1.000 | 978 | 0.0293 | 31.3 | 0.843 | 2 | 5.34e7 |
| 5 | `1e-4` | 14.8 | 0.191 | 0.667/0.483 | 6.00 | 0.797 | 1.000/1.000 | 554 | 0.0554 | 23.5 | 0.635 | 2 | 5.34e7 |
| 7 | `0` | 14.3 | 0.234 | 0.747/0.570 | 5.94 | 0.788 | 1.000/1.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | `1e-5` | 14.3 | 0.252 | 0.717/0.560 | 5.95 | 0.787 | 1.000/1.000 | 1300 | 0.0130 | 36.1 | 0.975 | 2 | 5.19e7 |
| 7 | `3e-5` | 14.4 | 0.239 | 0.707/0.530 | 5.98 | 0.788 | 1.000/1.000 | 1060 | 0.0319 | 32.6 | 0.880 | 2 | 5.19e7 |
| 7 | `1e-4` | 14.5 | 0.223 | 0.713/0.480 | 6.06 | 0.783 | 1.000/1.000 | 596 | 0.0596 | 24.4 | 0.659 | 2 | 5.19e7 |

Subject-wise refined metrics:

| row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist | SwAV dist | image retrieval | brain retrieval | visual cortex | V1 | V2 | V3 | V4 | higher visual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| official subj05 | 0.175138 | 0.405438 | 0.831098 | 0.910006 | 0.843287 | 0.825327 | 0.781252 | 0.444283 | 0.669222 | 0.469667 | 0.403527 | 0.328285 | 0.335762 | 0.323165 | 0.303573 | 0.414595 |
| Cycle 2/local subj05 | 0.198452 | 0.412553 | 0.848701 | 0.918145 | 0.856925 | 0.846210 | 0.759550 | 0.430057 | 0.648778 | 0.535778 | 0.414329 | 0.345305 | 0.350450 | 0.334916 | 0.312673 | 0.423239 |
| lambda0 subj05 | 0.198452 | 0.412553 | 0.848701 | 0.918145 | 0.856925 | 0.846210 | 0.759550 | 0.430057 | 0.648778 | 0.535778 | 0.414329 | 0.345305 | 0.350450 | 0.334916 | 0.312673 | 0.423239 |
| lambda1e-5 subj05 | 0.187435 | 0.410105 | 0.836173 | 0.912912 | 0.859159 | 0.845757 | 0.770829 | 0.434232 | 0.655444 | 0.517889 | 0.412533 | 0.343205 | 0.348078 | 0.332631 | 0.310461 | 0.422432 |
| lambda3e-5 subj05 | 0.191910 | 0.409869 | 0.845924 | 0.915991 | 0.854605 | 0.844619 | 0.766937 | 0.432970 | 0.642333 | 0.499333 | 0.410567 | 0.339292 | 0.345890 | 0.330353 | 0.309711 | 0.419768 |
| lambda1e-4 subj05 | 0.191268 | 0.411556 | 0.822008 | 0.900380 | 0.845233 | 0.828622 | 0.781932 | 0.438265 | 0.646667 | 0.431778 | 0.405016 | 0.326635 | 0.336490 | 0.324731 | 0.309751 | 0.417789 |
| official subj07 | 0.169830 | 0.408449 | 0.807004 | 0.858992 | 0.749044 | 0.742868 | 0.854014 | 0.503689 | 0.644444 | 0.378111 | 0.293540 | 0.283393 | 0.285332 | 0.271559 | 0.243048 | 0.285348 |
| Cycle 2/local subj07 | 0.200331 | 0.405877 | 0.826426 | 0.889542 | 0.775259 | 0.770100 | 0.830555 | 0.478162 | 0.691222 | 0.547222 | 0.317326 | 0.321064 | 0.323698 | 0.313173 | 0.281722 | 0.300001 |
| lambda0 subj07 | 0.200331 | 0.405877 | 0.826426 | 0.889542 | 0.775259 | 0.770100 | 0.830555 | 0.478162 | 0.691222 | 0.547222 | 0.317326 | 0.321064 | 0.323698 | 0.313173 | 0.281722 | 0.300001 |
| lambda1e-5 subj07 | 0.188403 | 0.408402 | 0.820619 | 0.881425 | 0.779218 | 0.759334 | 0.833629 | 0.486234 | 0.675778 | 0.514444 | 0.313782 | 0.316000 | 0.320073 | 0.313825 | 0.274364 | 0.298592 |
| lambda3e-5 subj07 | 0.204717 | 0.405263 | 0.823024 | 0.883913 | 0.775582 | 0.773722 | 0.829963 | 0.478411 | 0.678000 | 0.509000 | 0.317400 | 0.317305 | 0.316743 | 0.310857 | 0.273364 | 0.302353 |
| lambda1e-4 subj07 | 0.194194 | 0.400118 | 0.803316 | 0.864714 | 0.757584 | 0.753395 | 0.845851 | 0.492464 | 0.665889 | 0.452222 | 0.304163 | 0.301064 | 0.301102 | 0.295333 | 0.259155 | 0.290992 |

Weak-subject means:

| row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist | SwAV dist | image retrieval | brain retrieval | visual cortex | V1 | V2 | V3 | V4 | higher visual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| official weak mean | 0.172484 | 0.406944 | 0.819051 | 0.884499 | 0.796166 | 0.784098 | 0.817633 | 0.473986 | 0.656833 | 0.423889 | 0.348533 | 0.305839 | 0.310547 | 0.297362 | 0.273311 | 0.349971 |
| Cycle 2/local weak mean | 0.199392 | 0.409215 | 0.837564 | 0.903843 | 0.816092 | 0.808155 | 0.795053 | 0.454109 | 0.670000 | 0.541500 | 0.365828 | 0.333184 | 0.337074 | 0.324045 | 0.297198 | 0.361620 |
| lambda0 weak mean | 0.199392 | 0.409215 | 0.837564 | 0.903843 | 0.816092 | 0.808155 | 0.795053 | 0.454109 | 0.670000 | 0.541500 | 0.365828 | 0.333184 | 0.337074 | 0.324045 | 0.297198 | 0.361620 |
| lambda1e-5 weak mean | 0.187919 | 0.409254 | 0.828396 | 0.897169 | 0.819189 | 0.802546 | 0.802229 | 0.460233 | 0.665611 | 0.516167 | 0.363157 | 0.329603 | 0.334075 | 0.323228 | 0.292412 | 0.360512 |
| lambda3e-5 weak mean | 0.198314 | 0.407566 | 0.834474 | 0.899952 | 0.815093 | 0.809170 | 0.798450 | 0.455690 | 0.660167 | 0.504167 | 0.363983 | 0.328298 | 0.331316 | 0.320605 | 0.291537 | 0.361060 |
| lambda1e-4 weak mean | 0.192731 | 0.405837 | 0.812662 | 0.882547 | 0.801408 | 0.791009 | 0.813891 | 0.465365 | 0.656278 | 0.442000 | 0.354590 | 0.313849 | 0.318796 | 0.310032 | 0.284453 | 0.354390 |

Primary deltas versus same-code `lambda=0`:
- Subject 5 brain retrieval: `1e-5` `-0.017889`, `3e-5` `-0.036444`, `1e-4` `-0.104000`.
- Subject 7 brain retrieval: `1e-5` `-0.032778`, `3e-5` `-0.038222`, `1e-4` `-0.095000`.
- Weak-subject mean brain retrieval: `1e-5` `-0.025333`, `3e-5` `-0.037333`, `1e-4` `-0.099500`.
- `1e-4` also damages CLIP, Inception, image retrieval, visual cortex, and higher visual on both subjects relative to `lambda=0`.
- `3e-5` slightly improves subject 7 PixCorr, CLIP, visual cortex, and higher visual, but reduces subject 7 brain retrieval by `0.038222`; by the plan, PixCorr/SSIM/ROI-only gains do not count.

Interpretation:
- The same-code `lambda=0` rows exactly reproduce the stronger Cycle 2 local rows for subjects 5 and 7, so the Cycle 7 readout is not suffering from a local-control regression.
- Moderate or strong ridge anchoring is not promising under the plan's success rule. No nonzero lambda improves brain retrieval by about `0.02` absolute on either weak subject; all nonzero lambdas reduce brain retrieval versus `lambda=0`.
- The pattern is consistent with anchoring acting as an underfitting or geometry-disrupting constraint rather than a useful anti-drift prior in this setup. Drift norms still become large for `1e-5` and `3e-5`, while brain retrieval worsens; `1e-4` reduces drift more but clearly damages retrieval and semantic metrics.

Decision:
- Do not scale projection-drift anchoring to subjects 1/2.
- Stop the current ridge-wide projection-anchoring grid as a candidate improvement.
- If this mechanism is revisited, narrow/debug rather than scale: test a smaller lambda such as `3e-6` or anchor a smaller ridge subset only after confirming that the anchored tensors are the intended effective bottleneck. Based on this cycle, the next main mechanism should not be ridge-wide projection anchoring.

Recommended next research questions:
- Why does the local `lambda=0` one-session row outperform the official-control row so strongly for brain retrieval on subjects 5 and 7, despite being provenance-labeled as local training/config drift?
- If projection anchoring is revisited, can a much smaller anchored subset preserve brain retrieval while reducing drift, or is the ridge map too coupled to sparse subject adaptation?
- For subject 7 specifically, does the train-repeat reliability pattern support moving next to a reliability-aware or ROI-aware adaptation method rather than projection anchoring?

## Cycle 12 - 2026-05-21

Plan executed:
- Read `/plan.md` and executed only the Cycle 12 relational-consistency implementation and experiment-launch path.
- Telegram report was not due.
- Preserved the Cycle 7 decision: ridge-wide `ridge` L2 projection anchoring at `1e-5`, `3e-5`, and `1e-4` failed the success rule and should not be scaled to subjects 1/2.
- Kept provenance labels distinct: official paper-matched control, local Cycle 2 fine-tuned row, Cycle 7 same-code `lambda=0`, and Cycle 7 nonzero projection-anchored rows.
- No generator/refiner changes, caption/VLM correction, temporal decoding, broad ROI routing, CLIP-layer fusion, high-capacity adapters, reliability weighting, or new projection-anchor grid was launched.

Code/config changes:
- Edited `/src/Train.py`.
  - Added disabled-by-default CLI flags: `--relational_consistency`, `--rel_lambda`, `--rel_target`, and `--rel_metric`.
  - Implemented `--rel_target=clip` and `--rel_metric=sim_mse`.
  - Reserved `--rel_target=teacher` with an explicit `NotImplementedError` rather than silently changing the training path.
  - Added off-diagonal batch similarity MSE on the normalized `clip_voxels.flatten(1)` prediction and normalized CLIP image target.
  - During MixCo epochs, relational targets are mixed only for selected samples using the same `perm`, `betas`, and `select` tensors, then renormalized.
  - Added logs for `train/rel_loss`, `train/rel_loss_scaled`, `train/rel_sim_corr`, `test/rel_loss`, and `test/rel_sim_corr`.
  - Default behavior with no relational flags remains unchanged; no-rel control logs zero relational diagnostics.
- Added Slurm scripts:
  - `/src/cycle12_rel_smoke.slurm`: 1-hour two-task subject 5 smoke, no-rel control plus `rel_lambda=1e-3`.
  - `/src/cycle12_rel_train_s57.slurm`: subjects 5/7, lambdas `0`, `1e-3`, `3e-3`, `1e-2`, 150 epochs, paper-matched one-session local settings.
  - `/src/cycle12_rel_eval_s57.slurm`: dependent validated evaluator path `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py`.

Validation commands:
- Ran `python -m py_compile /src/Train.py`: passed.
- Ran `bash -n /src/cycle12_rel_smoke.slurm`: passed.
- Ran `bash -n /src/cycle12_rel_train_s57.slurm`: passed.
- Ran `bash -n /src/cycle12_rel_eval_s57.slurm`: passed.

Smoke jobs:
- Submitted `sbatch /src/cycle12_rel_smoke.slurm`: job `8549659`.
- `8549659_0`: subject 5 no-rel control, model `cycle12_smoke_subj05_rel0_1sess_3ep`, `COMPLETED`, exit `0:0`, elapsed `00:05:59`, node `della-l05g7`, MaxRSS `22933044K`, stdout `/src/slurms/c12_rel_smoke_8549659_0.out`, stderr `/src/slurms/c12_rel_smoke_8549659_0.err`.
- `8549659_1`: subject 5 relational `1e-3`, model `cycle12_smoke_subj05_rel1em3_1sess_3ep`, `COMPLETED`, exit `0:0`, elapsed `00:05:59`, node `della-l04g12`, MaxRSS `21471900K`, stdout `/src/slurms/c12_rel_smoke_8549659_1.out`, stderr `/src/slurms/c12_rel_smoke_8549659_1.err`.
- Failure classification: none. The Slurm `couldn't chdir to /src` warning is the same non-fatal compute-mount warning observed in Cycle 7; scripts immediately `cd` to the scratch-visible source tree and completed normally.

Smoke diagnostics:

| smoke row | epoch | train loss | test loss | train/test blurry PixCorr | train fwd/bwd | test fwd/bwd | train rel loss | train rel scaled | train rel corr | test rel loss | test rel corr |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| subj05 rel0 | 1 | 15.2 | 14.9 | 0.183/0.231 | 0.387/0.159 | 0.153/0.040 | 0 | 0 | 0 | 0 | 0 |
| subj05 rel0 | 2 | 11.5 | 11.4 | 0.254/0.227 | 0.898/0.754 | 0.380/0.203 | 0 | 0 | 0 | 0 | 0 |
| subj05 rel0 | 3 | 11.4 | 13.1 | 0.341/0.250 | 0.992/0.972 | 0.407/0.283 | 0 | 0 | 0 | 0 | 0 |
| subj05 rel1e-3 | 1 | 15.2 | 14.9 | 0.183/0.231 | 0.388/0.159 | 0.157/0.040 | 0.485 | 0.000485 | 0.158 | 0.373 | 0.221 |
| subj05 rel1e-3 | 2 | 11.5 | 11.4 | 0.254/0.226 | 0.898/0.754 | 0.380/0.200 | 0.115 | 0.000115 | 0.246 | 0.180 | 0.271 |
| subj05 rel1e-3 | 3 | 11.4 | 13.1 | 0.341/0.250 | 0.992/0.972 | 0.407/0.283 | 0.0268 | 0.0000268 | 0.284 | 0.160 | 0.283 |

Smoke conclusion:
- The relational loss is finite, nonzero, and logged when enabled.
- The scaled relational term at `1e-3` is far below the base objective and does not dominate training.
- Same-code no-rel control keeps relational diagnostics at zero and follows the expected Cycle 7 `lambda=0` short-run path.
- Proceeded with the planned lambda grid without reducing lambdas.

Full jobs launched:
- Submitted `sbatch /src/cycle12_rel_train_s57.slurm`: training array job `8549927`.
- Submitted `sbatch --dependency=afterok:8549927 /src/cycle12_rel_eval_s57.slurm`: dependent evaluator array job `8549929`.
- Training model mapping:
  - `8549927_0`: subject 5, `rel0`, `cycle12_subj05_rel0_1sess_150ep`.
  - `8549927_1`: subject 5, `rel1em3`, `cycle12_subj05_rel1em3_1sess_150ep`.
  - `8549927_2`: subject 5, `rel3em3`, `cycle12_subj05_rel3em3_1sess_150ep`.
  - `8549927_3`: subject 5, `rel1em2`, `cycle12_subj05_rel1em2_1sess_150ep`.
  - `8549927_4`: subject 7, `rel0`, `cycle12_subj07_rel0_1sess_150ep`.
  - `8549927_5`: subject 7, `rel1em3`, `cycle12_subj07_rel1em3_1sess_150ep`.
  - `8549927_6`: subject 7, `rel3em3`, `cycle12_subj07_rel3em3_1sess_150ep`.
  - `8549927_7`: subject 7, `rel1em2`, `cycle12_subj07_rel1em2_1sess_150ep`.
- Evaluation job `8549929_[0-7]` maps to the same model order and is pending on `afterok:8549927`.

Slurm state at 2026-05-21 09:47 EDT:

| task | subject | lambda | state | exit | node/reason | elapsed | stdout/stderr |
|---|---:|---:|---|---|---|---|---|
| `8549927_0` | 5 | `0` | RUNNING | `0:0` | `della-l05g5` | `00:00:51` | `/src/slurms/c12_rel_s57_8549927_0.out/.err` |
| `8549927_1` | 5 | `1e-3` | RUNNING | `0:0` | `della-l04g15` | `00:00:51` | `/src/slurms/c12_rel_s57_8549927_1.out/.err` |
| `8549927_2` | 5 | `3e-3` | RUNNING | `0:0` | `della-l04g14` | `00:00:51` | `/src/slurms/c12_rel_s57_8549927_2.out/.err` |
| `8549927_3` | 5 | `1e-2` | RUNNING | `0:0` | `della-l03g3` | `00:00:18` | `/src/slurms/c12_rel_s57_8549927_3.out/.err` |
| `8549927_4` | 7 | `0` | RUNNING | `0:0` | `della-l03g2` | `00:00:18` | `/src/slurms/c12_rel_s57_8549927_4.out/.err` |
| `8549927_5` | 7 | `1e-3` | RUNNING | `0:0` | `della-l02g16` | `00:00:18` | `/src/slurms/c12_rel_s57_8549927_5.out/.err` |
| `8549927_6` | 7 | `3e-3` | PENDING | `0:0` | `Priority` | `00:00:00` | `/src/slurms/c12_rel_s57_8549927_6.out/.err` expected |
| `8549927_7` | 7 | `1e-2` | PENDING | `0:0` | `Priority` | `00:00:00` | `/src/slurms/c12_rel_s57_8549927_7.out/.err` expected |
| `8549929_[0-7]` | 5/7 | all | PENDING | `0:0` | `Dependency` | `00:00:00` | `/src/slurms/c12_rel_eval_s57_8549929_<task>.out/.err` expected |

Current metric status:
- No Cycle 12 refined evaluation CSVs exist yet under `/src/tables`.
- No PixCorr/SSIM/AlexNet/Inception/CLIP/EfficientNet/SwAV/image-retrieval/brain-retrieval/ROI-correlation rows are available yet for the full Cycle 12 models.
- The next cycle should parse final training logs for relational diagnostics and then parse `cycle12_subj0{5,7}_rel{0,1em3,3em3,1em2}_1sess_150ep_all_enhancedrecons.csv` after evaluator job `8549929` completes.

Recommended next research questions:
- If relational consistency improves subject 5 brain retrieval without semantic or ROI damage, does subject 7 require reliability-aware voxel adaptation rather than stronger relational weighting?
- If all relational rows match no-rel control, audit whether `clip_voxels.flatten(1)` is the only retrieval-relevant embedding used downstream in `recon_inference.py`.
- If `1e-2` damages semantic metrics while `1e-3` or `3e-3` is neutral, narrow the grid rather than increasing the penalty.

## Cycle 13 - 2026-05-21

Plan executed:
- Read `/plan.md` and executed only the Cycle 12 relational-consistency recovery/readout path.
- Telegram report was not due.
- No generator/refiner edits, caption/VLM correction, temporal decoding, broad ROI routing, CLIP-layer fusion, reliability weighting, high-capacity adapters, or new projection-anchor mechanism was started.

Code/config changes:
- No code or Slurm changes were made in Cycle 13.
- No jobs were cancelled or relaunched. The active Cycle 12 sweep was running validly, so the correct action was to monitor rather than mutate it.

Operational visibility:
- This shell could query Slurm with `squeue`/`sacct` and read `/src` logs/tables.
- The compute-side Slurm warning `couldn't chdir to /src` appears in stderr, but it is non-fatal: each script immediately changes to `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`, loads the intended environment, and trains.

Commands/status checks:
- Ran `squeue -j 8549927,8549929`.
- Ran `sacct -j 8549927,8549929 --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList,MaxRSS,ReqMem,Timelimit -P`.
- Parsed `/src/slurms/c12_rel_s57_8549927_<task>.out/.err` for subject/lambda mapping and live train/test diagnostics.
- Checked `/src/tables` for Cycle 12 CSVs; none existed yet because training had not completed and evaluator `8549929` had not started.

Slurm state at approximately 2026-05-21 10:35 EDT:

| task | subject | lambda | state | exit | node | elapsed | stdout/stderr | expected checkpoint/artifacts | failure class |
|---|---:|---:|---|---|---|---|---|---|---|
| `8549927_0` | 5 | `0` | RUNNING | `0:0` | `della-l05g5` | `00:48:34` | `/src/slurms/c12_rel_s57_8549927_0.out/.err` | `../train_logs/cycle12_subj05_rel0_1sess_150ep/last.pth` | none observed |
| `8549927_1` | 5 | `1e-3` | RUNNING | `0:0` | `della-l04g15` | `00:48:34` | `/src/slurms/c12_rel_s57_8549927_1.out/.err` | `../train_logs/cycle12_subj05_rel1em3_1sess_150ep/last.pth` | none observed |
| `8549927_2` | 5 | `3e-3` | RUNNING | `0:0` | `della-l04g14` | `00:48:34` | `/src/slurms/c12_rel_s57_8549927_2.out/.err` | `../train_logs/cycle12_subj05_rel3em3_1sess_150ep/last.pth` | none observed |
| `8549927_3` | 5 | `1e-2` | RUNNING | `0:0` | `della-l03g3` | `00:48:01` | `/src/slurms/c12_rel_s57_8549927_3.out/.err` | `../train_logs/cycle12_subj05_rel1em2_1sess_150ep/last.pth` | none observed |
| `8549927_4` | 7 | `0` | RUNNING | `0:0` | `della-l03g2` | `00:48:01` | `/src/slurms/c12_rel_s57_8549927_4.out/.err` | `../train_logs/cycle12_subj07_rel0_1sess_150ep/last.pth` | none observed |
| `8549927_5` | 7 | `1e-3` | RUNNING | `0:0` | `della-l02g16` | `00:48:01` | `/src/slurms/c12_rel_s57_8549927_5.out/.err` | `../train_logs/cycle12_subj07_rel1em3_1sess_150ep/last.pth` | none observed |
| `8549927_6` | 7 | `3e-3` | RUNNING | `0:0` | `della-l02g12` | `00:47:30` | `/src/slurms/c12_rel_s57_8549927_6.out/.err` | `../train_logs/cycle12_subj07_rel3em3_1sess_150ep/last.pth` | none observed |
| `8549927_7` | 7 | `1e-2` | RUNNING | `0:0` | `della-l02g11` | `00:47:30` | `/src/slurms/c12_rel_s57_8549927_7.out/.err` | `../train_logs/cycle12_subj07_rel1em2_1sess_150ep/last.pth` | none observed |
| `8549929_[0-7]` | 5/7 | all | PENDING | `0:0` | none | `00:00:00` | `/src/slurms/c12_rel_eval_s57_8549929_<task>.out/.err` expected | final CSVs expected under `/src/tables` | dependency-pending, no failure |

Live training diagnostics at approximately epoch 50:

| task | subj | lambda | epoch | test loss | test PixCorr | test fwd/bwd | train loss | train PixCorr | train fwd/bwd | train rel loss | scaled rel | train rel corr | test rel loss | test rel corr |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| `8549927_0` | 5 | `0` | 50 | 16.6 | 0.205 | 0.460/0.370 | 10.5 | 0.651 | 1.000/1.000 | 0 | 0 | 0 | 0 | 0 |
| `8549927_1` | 5 | `1e-3` | 50 | 16.6 | 0.213 | 0.500/0.413 | 10.5 | 0.647 | 1.000/1.000 | 0.0164 | 1.64e-5 | 0.269 | 0.197 | 0.308 |
| `8549927_2` | 5 | `3e-3` | 50 | 16.5 | 0.219 | 0.513/0.347 | 10.5 | 0.645 | 1.000/1.000 | 0.0162 | 4.86e-5 | 0.267 | 0.173 | 0.301 |
| `8549927_3` | 5 | `1e-2` | 50 | 16.5 | 0.194 | 0.430/0.403 | 10.5 | 0.646 | 1.000/1.000 | 0.0158 | 1.58e-4 | 0.266 | 0.157 | 0.303 |
| `8549927_4` | 7 | `0` | 50 | 13.6 | 0.214 | 0.600/0.400 | 9.77 | 0.642 | 1.000/1.000 | 0 | 0 | 0 | 0 | 0 |
| `8549927_5` | 7 | `1e-3` | 50 | 13.5 | 0.252 | 0.580/0.453 | 9.69 | 0.645 | 1.000/1.000 | 0.0158 | 1.58e-5 | 0.255 | 0.138 | 0.245 |
| `8549927_6` | 7 | `3e-3` | 50 | 16.8 | 0.164 | 0.533/0.380 | 10.4 | 0.625 | 1.000/1.000 | 0.0169 | 5.08e-5 | 0.257 | 0.159 | 0.257 |
| `8549927_7` | 7 | `1e-2` | 50 | 16.9 | 0.201 | 0.547/0.433 | 10.3 | 0.631 | 1.000/1.000 | 0.0172 | 1.72e-4 | 0.275 | 0.158 | 0.271 |

Metric table status:
- No Cycle 12 refined CSVs exist yet under `/src/tables`.
- Because `8549929_[0-7]` is still dependency-pending, PixCorr, SSIM, AlexNet-2, AlexNet-5, Inception, CLIP, EfficientNet distance, SwAV distance, image retrieval, brain retrieval, visual cortex, V1, V2, V3, V4, and higher-visual rows cannot yet be computed for Cycle 12.
- Therefore no deltas versus same-subject `rel0`, no weak-subject means, and no scientific success/failure decision can be made in this cycle.

Conclusion:
- Cycle 12 is not unrecoverable. It is actively running and producing expected relational diagnostics.
- Same-code `rel0` rows correctly log zero relational diagnostics; nonzero rows log finite relational losses and correlations.
- No operational repair or rerun was appropriate during Cycle 13. The next valid action is to inspect final state for `8549927_[0-7]`; if all tasks complete, let or inspect dependent evaluator `8549929_[0-7]`. If a task fails, rerun only that failed task or its missing evaluator path with the same model name and hyperparameters.

Recommended next research questions:
- Once `8549929` writes final CSVs, do any nonzero relational rows improve brain retrieval by about 0.02 absolute versus same-subject `rel0` without damaging CLIP, Inception, image retrieval, ROI correlations, EfficientNet distance, or SwAV distance?
- If all relational rows match `rel0`, audit whether `clip_voxels.flatten(1)` is the actual retrieval-relevant embedding used by `recon_inference.py`.
- If `1e-2` damages final semantic metrics while `1e-3` or `3e-3` is neutral, narrow the relational lambda grid downward rather than increasing it.

## Cycle 14 - 2026-05-21

Plan executed:
- Read `/plan.md` and executed only the Cycle 12 relational-consistency recovery/readout path.
- Telegram report was not due.
- No new mechanism was started. No code, model, evaluator, or Slurm script changes were made.
- No jobs were cancelled, relaunched, or newly submitted because the Cycle 12 jobs were still running validly.

Operational visibility:
- This shell could query Slurm with `squeue`/`sacct` and read `/src` logs/tables.
- The compute-side Slurm warning `couldn't chdir to /src` remains non-fatal; the scripts immediately `cd` to the scratch-visible source tree and continue training.

Commands/status checks:
- Ran `squeue -j 8549927,8549929`.
- Ran `sacct -j 8549927,8549929 --format=JobID,JobName%35,State,ExitCode,Elapsed,MaxRSS,ReqMem,Timelimit,NodeList%25 -P`.
- Parsed `/src/slurms/c12_rel_s57_8549927_<task>.err` for live train/test losses, blurry PixCorr, retrieval diagnostics, and relational diagnostics.
- Checked `/src/tables` for the eight expected Cycle 12 CSVs; none exist yet because the dependent evaluator has not started.
- Checked `/src/train_logs/cycle12_subj0{5,7}_rel*_1sess_150ep/last.pth`; final checkpoints were not visible yet at the snapshot time because training was still in progress.

Slurm state at 2026-05-21 11:14 EDT:

| task | subject | lambda | state | exit | node/reason | elapsed | timelimit | stdout/stderr | expected checkpoint/CSV | failure class |
|---|---:|---:|---|---|---|---|---|---|---|---|
| `8549927_0` | 5 | `0` | RUNNING | `0:0` | `della-l05g5` | `01:27:53` | `03:00:00` | `/src/slurms/c12_rel_s57_8549927_0.out/.err` | `cycle12_subj05_rel0_1sess_150ep` | none observed |
| `8549927_1` | 5 | `1e-3` | RUNNING | `0:0` | `della-l04g15` | `01:27:53` | `03:00:00` | `/src/slurms/c12_rel_s57_8549927_1.out/.err` | `cycle12_subj05_rel1em3_1sess_150ep` | none observed |
| `8549927_2` | 5 | `3e-3` | RUNNING | `0:0` | `della-l04g14` | `01:27:53` | `03:00:00` | `/src/slurms/c12_rel_s57_8549927_2.out/.err` | `cycle12_subj05_rel3em3_1sess_150ep` | none observed |
| `8549927_3` | 5 | `1e-2` | RUNNING | `0:0` | `della-l03g3` | `01:27:20` | `03:00:00` | `/src/slurms/c12_rel_s57_8549927_3.out/.err` | `cycle12_subj05_rel1em2_1sess_150ep` | none observed |
| `8549927_4` | 7 | `0` | RUNNING | `0:0` | `della-l03g2` | `01:27:20` | `03:00:00` | `/src/slurms/c12_rel_s57_8549927_4.out/.err` | `cycle12_subj07_rel0_1sess_150ep` | none observed |
| `8549927_5` | 7 | `1e-3` | RUNNING | `0:0` | `della-l02g16` | `01:27:20` | `03:00:00` | `/src/slurms/c12_rel_s57_8549927_5.out/.err` | `cycle12_subj07_rel1em3_1sess_150ep` | none observed |
| `8549927_6` | 7 | `3e-3` | RUNNING | `0:0` | `della-l02g12` | `01:26:49` | `03:00:00` | `/src/slurms/c12_rel_s57_8549927_6.out/.err` | `cycle12_subj07_rel3em3_1sess_150ep` | none observed |
| `8549927_7` | 7 | `1e-2` | RUNNING | `0:0` | `della-l02g11` | `01:26:49` | `03:00:00` | `/src/slurms/c12_rel_s57_8549927_7.out/.err` | `cycle12_subj07_rel1em2_1sess_150ep` | none observed |
| `8549929_[0-7]` | 5/7 | all | PENDING | `0:0` | `Dependency` | `00:00:00` | `04:00:00` | `/src/slurms/c12_rel_eval_s57_8549929_<task>.out/.err` expected | final CSVs expected under `/src/tables` | dependency-pending, no failure |

Latest live training diagnostics:

| task | subject | lambda | epoch | test loss | test PixCorr | test fwd/bwd | train loss | train PixCorr | train fwd/bwd | train rel loss | scaled rel | train rel corr | test rel loss | test rel corr |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| `8549927_0` | 5 | `0` | 93/150 | 17.1 | 0.195 | 0.637/0.510 | 6.81 | 0.781 | 1.000/1.000 | 0 | 0 | 0 | 0 | 0 |
| `8549927_1` | 5 | `1e-3` | 93/150 | 17.1 | 0.216 | 0.613/0.513 | 6.81 | 0.782 | 1.000/1.000 | 0.0205 | 2.05e-5 | 0.311 | 0.108 | 0.307 |
| `8549927_2` | 5 | `3e-3` | 93/150 | 17.1 | 0.208 | 0.673/0.530 | 6.83 | 0.781 | 1.000/1.000 | 0.0220 | 6.60e-5 | 0.305 | 0.0933 | 0.312 |
| `8549927_3` | 5 | `1e-2` | 93/150 | 17.2 | 0.197 | 0.677/0.553 | 6.84 | 0.781 | 1.000/1.000 | 0.0165 | 1.65e-4 | 0.308 | 0.117 | 0.301 |
| `8549927_4` | 7 | `0` | 94/150 | 14.1 | 0.164 | 0.723/0.567 | 6.66 | 0.773 | 1.000/1.000 | 0 | 0 | 0 | 0 | 0 |
| `8549927_5` | 7 | `1e-3` | 94/150 | 14.1 | 0.153 | 0.723/0.577 | 6.64 | 0.775 | 1.000/1.000 | 0.0280 | 2.80e-5 | 0.305 | 0.0842 | 0.257 |
| `8549927_6` | 7 | `3e-3` | 93/150 | 16.7 | 0.178 | 0.753/0.530 | 6.88 | 0.791 | 1.000/1.000 | 0.0261 | 7.82e-5 | 0.289 | 0.103 | 0.259 |
| `8549927_7` | 7 | `1e-2` | 93/150 | 16.9 | 0.180 | 0.723/0.523 | 6.87 | 0.790 | 1.000/1.000 | 0.0188 | 1.88e-4 | 0.297 | 0.129 | 0.269 |

Metric table status:
- No Cycle 12 refined CSVs exist yet under `/src/tables`.
- Because `8549929_[0-7]` remains dependency-pending, PixCorr, SSIM, AlexNet-2, AlexNet-5, Inception, CLIP, EfficientNet distance, SwAV distance, image retrieval, brain retrieval, visual cortex, V1, V2, V3, V4, and higher-visual rows are still unavailable for Cycle 12.
- No deltas versus same-subject `rel0`, weak-subject means, or scientific success/failure decision can be made yet.

Conclusion:
- Cycle 12 is still recoverable and actively running. It is not an all-rows-unrecoverable case.
- The same-code `rel0` rows continue to log zero relational diagnostics, and nonzero relational rows continue to log finite relational losses and correlations.
- The next valid action is to inspect `8549927_[0-7]` after completion. If all training tasks complete, let `8549929_[0-7]` run and parse the final CSVs. If a training task fails or times out, repair only operational issues and rerun only the failed task with the same model name, subject, lambda, and hyperparameters.

Recommended next research questions:
- Do all eight training tasks complete within the 3-hour request, and does the dependent evaluator start automatically?
- Once final CSVs are written, does any nonzero relational row improve brain retrieval by about 0.02 absolute versus same-subject `rel0` without semantic, distance, retrieval, or ROI damage?
- If training finishes but any evaluator row fails or is missing, rerun only `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py` for that exact model.

## Cycle 15 - 2026-05-21

Plan executed:
- Read `/plan.md` and executed only the Cycle 12 relational-consistency recovery/readout path.
- Telegram report was not due.
- No code, model, evaluator, or Slurm script changes were made.
- No new mechanism was launched.

Operational status:
- All eight Cycle 12 training tasks completed successfully.
- All eight dependent evaluator tasks completed successfully through `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py`.
- Failure classification for all eight rows: none.
- The recurring compute-side `couldn't chdir to /src` warning remained non-fatal; scripts changed to the scratch-visible source tree and completed.
- The planning shell could read final `/src/evals` and `/src/tables` artifacts after evaluation. It could not reliably stat live compute-side checkpoints during training, but each evaluator loaded its expected `../train_logs/<model_name>/last.pth` checkpoint before writing outputs.

Commands/status checks:
- `squeue -j 8549927,8549929`
- `sacct -j 8549927,8549929 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList,Reason --parsable2`
- Parsed `/src/slurms/c12_rel_s57_8549927_<task>.err` for final train/test diagnostics.
- Parsed `/src/slurms/c12_rel_eval_s57_8549929_<task>.out/.err` for evaluator completion and final metric prints.
- Read final CSVs from `/src/tables/cycle12_subj0{5,7}_rel{0,1em3,3em3,1em2}_1sess_150ep_all_enhancedrecons.csv`.

Training/evaluation job states:

| task | model | state | elapsed | MaxRSS | node | artifacts |
|---|---|---|---:|---:|---|---|
| `8549927_0` | `cycle12_subj05_rel0_1sess_150ep` | `COMPLETED 0:0` | `02:19:58` | `21465840K` | `della-l05g5` | final CSV and enhanced recon present |
| `8549927_1` | `cycle12_subj05_rel1em3_1sess_150ep` | `COMPLETED 0:0` | `02:20:00` | `21611572K` | `della-l04g15` | final CSV and enhanced recon present |
| `8549927_2` | `cycle12_subj05_rel3em3_1sess_150ep` | `COMPLETED 0:0` | `02:20:01` | `21467420K` | `della-l04g14` | final CSV and enhanced recon present |
| `8549927_3` | `cycle12_subj05_rel1em2_1sess_150ep` | `COMPLETED 0:0` | `02:19:25` | `21915092K` | `della-l03g3` | final CSV and enhanced recon present |
| `8549927_4` | `cycle12_subj07_rel0_1sess_150ep` | `COMPLETED 0:0` | `02:17:02` | `22894032K` | `della-l03g2` | final CSV and enhanced recon present |
| `8549927_5` | `cycle12_subj07_rel1em3_1sess_150ep` | `COMPLETED 0:0` | `02:17:06` | `21573536K` | `della-l02g16` | final CSV and enhanced recon present |
| `8549927_6` | `cycle12_subj07_rel3em3_1sess_150ep` | `COMPLETED 0:0` | `02:17:26` | `22942796K` | `della-l02g12` | final CSV and enhanced recon present |
| `8549927_7` | `cycle12_subj07_rel1em2_1sess_150ep` | `COMPLETED 0:0` | `02:17:31` | `21584748K` | `della-l02g11` | final CSV and enhanced recon present |
| `8549929_0` | `cycle12_subj05_rel0_1sess_150ep` | `COMPLETED 0:0` | `02:23:55` | `49258772K` | `della-l05g7` | `/src/tables/cycle12_subj05_rel0_1sess_150ep_all_enhancedrecons.csv` |
| `8549929_1` | `cycle12_subj05_rel1em3_1sess_150ep` | `COMPLETED 0:0` | `02:24:23` | `46261796K` | `della-l05g5` | `/src/tables/cycle12_subj05_rel1em3_1sess_150ep_all_enhancedrecons.csv` |
| `8549929_2` | `cycle12_subj05_rel3em3_1sess_150ep` | `COMPLETED 0:0` | `02:24:55` | `49358376K` | `della-l04g15` | `/src/tables/cycle12_subj05_rel3em3_1sess_150ep_all_enhancedrecons.csv` |
| `8549929_3` | `cycle12_subj05_rel1em2_1sess_150ep` | `COMPLETED 0:0` | `02:25:11` | `49406004K` | `della-l05g2` | `/src/tables/cycle12_subj05_rel1em2_1sess_150ep_all_enhancedrecons.csv` |
| `8549929_4` | `cycle12_subj07_rel0_1sess_150ep` | `COMPLETED 0:0` | `02:23:08` | `47235392K` | `della-l04g14` | `/src/tables/cycle12_subj07_rel0_1sess_150ep_all_enhancedrecons.csv` |
| `8549929_5` | `cycle12_subj07_rel1em3_1sess_150ep` | `COMPLETED 0:0` | `02:23:09` | `47502640K` | `della-l04g14` | `/src/tables/cycle12_subj07_rel1em3_1sess_150ep_all_enhancedrecons.csv` |
| `8549929_6` | `cycle12_subj07_rel3em3_1sess_150ep` | `COMPLETED 0:0` | `02:31:51` | `49444132K` | `della-l04g1` | `/src/tables/cycle12_subj07_rel3em3_1sess_150ep_all_enhancedrecons.csv` |
| `8549929_7` | `cycle12_subj07_rel1em2_1sess_150ep` | `COMPLETED 0:0` | `02:25:07` | `46342456K` | `della-l03g3` | `/src/tables/cycle12_subj07_rel1em2_1sess_150ep_all_enhancedrecons.csv` |

Final training diagnostics:

| row | test loss | test PixCorr | test fwd/bwd | train loss | train PixCorr | train rel loss | scaled rel | train rel corr | test rel loss | test rel corr |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| subj05 rel0 | 14.5 | 0.198 | 0.657/0.553 | 5.89 | 0.800 | 0 | 0 | 0 | 0 | 0 |
| subj05 rel1e-3 | 14.5 | 0.190 | 0.637/0.553 | 5.88 | 0.801 | 0.0286 | 2.86e-5 | 0.335 | 0.0821 | 0.318 |
| subj05 rel3e-3 | 14.4 | 0.186 | 0.677/0.543 | 5.91 | 0.800 | 0.0289 | 8.67e-5 | 0.332 | 0.0716 | 0.328 |
| subj05 rel1e-2 | 14.5 | 0.194 | 0.673/0.570 | 5.90 | 0.799 | 0.0208 | 2.08e-4 | 0.337 | 0.0959 | 0.320 |
| subj07 rel0 | 14.3 | 0.234 | 0.747/0.570 | 5.94 | 0.788 | 0 | 0 | 0 | 0 | 0 |
| subj07 rel1e-3 | 14.2 | 0.219 | 0.727/0.597 | 5.93 | 0.788 | 0.0318 | 3.18e-5 | 0.308 | 0.0814 | 0.268 |
| subj07 rel3e-3 | 14.3 | 0.221 | 0.757/0.543 | 5.93 | 0.781 | 0.0282 | 8.45e-5 | 0.316 | 0.0980 | 0.270 |
| subj07 rel1e-2 | 14.3 | 0.225 | 0.737/0.523 | 5.93 | 0.788 | 0.0202 | 2.02e-4 | 0.321 | 0.1200 | 0.277 |

Training-diagnostic checks:
- `rel0` rows logged zero relational diagnostics.
- Nonzero rows logged finite relational losses and finite train/test similarity correlations.
- The relational term stayed tiny relative to the base objective at all tested lambdas.

Final refined metrics, subject 5:

| row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist | SwAV dist | image retrieval | brain retrieval | visual cortex | V1 | V2 | V3 | V4 | higher visual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| subj05 rel0 | 0.198452 | 0.412553 | 0.848701 | 0.918145 | 0.856925 | 0.846210 | 0.759550 | 0.430057 | 0.648778 | 0.535778 | 0.414329 | 0.345305 | 0.350450 | 0.334916 | 0.312673 | 0.423239 |
| subj05 rel1e-3 | 0.187441 | 0.414308 | 0.847139 | 0.917372 | 0.859084 | 0.843295 | 0.767757 | 0.431824 | 0.639333 | 0.526000 | 0.412454 | 0.347917 | 0.354858 | 0.337126 | 0.312403 | 0.421132 |
| subj05 rel3e-3 | 0.191241 | 0.415813 | 0.842435 | 0.921072 | 0.857290 | 0.844147 | 0.768463 | 0.431831 | 0.660000 | 0.526222 | 0.411674 | 0.340972 | 0.350501 | 0.335578 | 0.312085 | 0.421737 |
| subj05 rel1e-2 | 0.200522 | 0.415293 | 0.839455 | 0.917371 | 0.865186 | 0.842430 | 0.767671 | 0.429910 | 0.660222 | 0.525333 | 0.413880 | 0.351519 | 0.356984 | 0.340144 | 0.314392 | 0.420878 |

Final refined metrics, subject 7:

| row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist | SwAV dist | image retrieval | brain retrieval | visual cortex | V1 | V2 | V3 | V4 | higher visual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| subj07 rel0 | 0.200331 | 0.405877 | 0.826426 | 0.889542 | 0.775259 | 0.770100 | 0.830555 | 0.478162 | 0.691222 | 0.547222 | 0.317326 | 0.321064 | 0.323698 | 0.313173 | 0.281722 | 0.300001 |
| subj07 rel1e-3 | 0.201250 | 0.403458 | 0.827532 | 0.893929 | 0.786837 | 0.782383 | 0.822688 | 0.473255 | 0.684444 | 0.561556 | 0.317221 | 0.318054 | 0.319379 | 0.312152 | 0.278774 | 0.299809 |
| subj07 rel3e-3 | 0.195962 | 0.401322 | 0.817856 | 0.887415 | 0.782259 | 0.776850 | 0.825701 | 0.478171 | 0.692222 | 0.532111 | 0.320165 | 0.322570 | 0.325068 | 0.315844 | 0.280332 | 0.304207 |
| subj07 rel1e-2 | 0.199021 | 0.404358 | 0.824430 | 0.891920 | 0.790353 | 0.783824 | 0.817865 | 0.471761 | 0.688000 | 0.504333 | 0.315187 | 0.309740 | 0.310752 | 0.304774 | 0.269459 | 0.301732 |

Primary deltas versus same-subject `rel0`:

| row | brain retrieval | image retrieval | CLIP | Inception | EffNet dist | SwAV dist | visual cortex | higher visual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 rel1e-3 | -0.009778 | -0.009444 | -0.002915 | +0.002159 | +0.008206 | +0.001766 | -0.001875 | -0.002108 |
| subj05 rel3e-3 | -0.009556 | +0.011222 | -0.002063 | +0.000365 | +0.008913 | +0.001774 | -0.002655 | -0.001502 |
| subj05 rel1e-2 | -0.010444 | +0.011444 | -0.003780 | +0.008261 | +0.008121 | -0.000148 | -0.000449 | -0.002361 |
| subj07 rel1e-3 | +0.014333 | -0.006778 | +0.012283 | +0.011578 | -0.007866 | -0.004907 | -0.000105 | -0.000192 |
| subj07 rel3e-3 | -0.015111 | +0.001000 | +0.006750 | +0.007000 | -0.004854 | +0.000009 | +0.002839 | +0.004206 |
| subj07 rel1e-2 | -0.042889 | -0.003222 | +0.013724 | +0.015094 | -0.012690 | -0.006401 | -0.002139 | +0.001731 |

Weak-subject means:

| row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist | SwAV dist | image retrieval | brain retrieval | visual cortex | V1 | V2 | V3 | V4 | higher visual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| weak rel0 | 0.199392 | 0.409215 | 0.837564 | 0.903843 | 0.816092 | 0.808155 | 0.795053 | 0.454109 | 0.670000 | 0.541500 | 0.365828 | 0.333184 | 0.337074 | 0.324045 | 0.297198 | 0.361620 |
| weak rel1e-3 | 0.194345 | 0.408883 | 0.837335 | 0.905651 | 0.822960 | 0.812839 | 0.795222 | 0.452539 | 0.661889 | 0.543778 | 0.364838 | 0.332985 | 0.337118 | 0.324639 | 0.295588 | 0.360470 |
| weak rel3e-3 | 0.193601 | 0.408568 | 0.830146 | 0.904244 | 0.819775 | 0.810498 | 0.797082 | 0.455001 | 0.676111 | 0.529167 | 0.365920 | 0.331771 | 0.337784 | 0.325711 | 0.296209 | 0.362972 |
| weak rel1e-2 | 0.199772 | 0.409825 | 0.831943 | 0.904646 | 0.827770 | 0.813127 | 0.792768 | 0.450835 | 0.674111 | 0.514833 | 0.364533 | 0.330629 | 0.333868 | 0.322459 | 0.291925 | 0.361305 |

Weak-subject brain retrieval deltas versus `rel0`:
- `rel1e-3`: `+0.002278`
- `rel3e-3`: `-0.012333`
- `rel1e-2`: `-0.026667`

Interpretation:
- Subject 5 did not support relational consistency. All nonzero lambdas reduced refined brain retrieval by about `0.0096` to `0.0104` absolute versus same-subject `rel0`. `rel3e-3` and `rel1e-2` improved image retrieval, but the plan does not count image retrieval gains when brain retrieval does not improve and EffNet distance worsens.
- Subject 7 had the only positive retrieval signal at `rel1e-3`: brain retrieval improved by `+0.014333`, with better CLIP, Inception, EffNet distance, and SwAV distance. This is below the plan's approximate `+0.02` success threshold and came with slightly lower image retrieval plus essentially flat to slightly lower visual/higher-visual correlations.
- Subject 7 `rel3e-3` and `rel1e-2` are not candidates: `rel3e-3` reduced brain retrieval by `0.015111`; `rel1e-2` reduced it by `0.042889`.
- Weak-subject mean brain retrieval is effectively neutral at `rel1e-3` (`+0.002278`) and worse at higher lambdas.

Scientific decision:
- Cycle 12 relational consistency is not a successful weak-subject improvement under the plan's criteria.
- Do not claim a four-subject MindEyeV2 improvement.
- Do not scale the relational lambda grid to subjects 1/2/5/7 as an improvement candidate.
- If revisited, `rel1e-3` on subject 7 is the only weak hint worth repeating, but it is below threshold and not enough to displace the next mechanism.
- The next mechanism should follow the plan's reliability-aware fallback: training-only reliability-aware voxel adaptation on subjects 5 and 7, with train-repeat reliability estimated only from training images, shrinkage toward ROI means, separate early and higher visual reporting, and a small no-reliability/low/mid grid.

Recommended next research questions:
- Does subject 7 `rel1e-3` repeat its sub-threshold brain-retrieval gain, or was it evaluation/training noise?
- Before abandoning the mechanism entirely, confirm whether `clip_voxels.flatten(1)` is the same embedding used by `recon_inference.py`, retrieval evaluation, and prior conditioning.
- For the next planned mechanism, what reliability estimate is stable when computed only from repeated training images after MindEye beta preprocessing?

## Cycle 16 - 2026-05-24

Plan executed:
- Read `/plan.md` and executed the Cycle 16 reliability-aware voxel adaptation plan only.
- Telegram report is due; report-ready update is included below.
- Implemented leakage-free train-repeat reliability estimation and a disabled-by-default tensor-backed reliability gate.
- Launched the required 1-hour smoke array before any full training.
- After smoke passed, launched the small subject 5/7 training grid and dependent fixed-path evaluator.

Code/config changes:
- Added `/src/estimate_reliability.py`.
  - Reads the exact one-session WebDataset train shard (`wds/subj0{5,7}/train/0.tar`), `COCO_73k_subj_indices.hdf5`, `betas_all_subj0{5,7}_fp32_renorm.hdf5`, and `brain_region_masks.hdf5`.
  - Verifies `behav[:,0,0]` image IDs against `COCO_73k_subj_indices[subj][behav[:,0,5]]`.
  - Computes voxelwise split-half repeat consistency from repeated training images only.
  - Excludes old test and `new_test` overlaps from reliability inputs.
  - Shrinks raw reliabilities 25% toward ROI means for early and higher visual masks.
  - Writes `/src/reliability/subj05_trainrepeat_reliability.pt`, `/src/reliability/subj07_trainrepeat_reliability.pt`, `/src/reliability/trainrepeat_reliability_summary.json`, and `/src/reliability/trainrepeat_reliability_summary.csv`.
- Updated `/src/Train.py`.
  - Replaced the prior variance-proxy/hard top-k reliability hook with `--reliability_mode {none,centered_scale}`, `--reliability_path`, `--reliability_strength`, and conservative clip bounds.
  - `relgate0` uses the reliability tensor with `strength=0`, verifies tensor shape, and applies scale exactly 1.0.
  - Nonzero rows use centered input scaling, clipped to `[0.75, 1.25]` and re-centered to mean scale 1.0.
  - Logs reliability provenance plus scale mean/min/max and early/higher reliability and scale means to `reliability_summary.json`.
- Added Slurm scripts:
  - `/src/cycle16_relgate_smoke.slurm`
  - `/src/cycle16_relgate_train_s57.slurm`
  - `/src/cycle16_relgate_eval_s57.slurm`

Preflight diagnostics:

| subject | voxel count | train rows | unique train images | repeat images | repeat trials | invalid/degenerate voxels | train-new_test overlap | repeat-new_test overlap | early/higher mask counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 13039 | 688 | 536 | 123 | 275 | 0/0 | 0 | 0 | 3661/9378 |
| 7 | 12682 | 688 | 536 | 123 | 275 | 0/0 | 0 | 0 | 3251/9431 |

Reliability summaries:

| subject | raw mean | raw median | raw std | raw early mean | raw higher mean | shrunk mean | shrunk median | shrunk std | shrunk p05/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 0.221351 | 0.209002 | 0.165043 | 0.275172 | 0.200341 | 0.221351 | 0.211106 | 0.125764 | 0.029087 / 0.446512 |
| 7 | 0.171797 | 0.154757 | 0.164491 | 0.257655 | 0.142201 | 0.171797 | 0.156331 | 0.127795 | -0.013230 / 0.413119 |

Validation:
- `/src/fmri/bin/python -m py_compile /src/estimate_reliability.py /src/Train.py` passed.
- `bash -n /src/cycle16_relgate_smoke.slurm`, `bash -n /src/cycle16_relgate_train_s57.slurm`, and `bash -n /src/cycle16_relgate_eval_s57.slurm` passed.

Smoke jobs:
- Submitted `sbatch /src/cycle16_relgate_smoke.slurm` as job `8670161`.
- Tasks:
  - `8670161_0`: subj05 `relgate0`, completed `0:0`, elapsed `00:06:34`, MaxRSS `21606288K`.
  - `8670161_1`: subj05 `relgate_low` (`alpha=0.05`), completed `0:0`, elapsed `00:06:43`, MaxRSS `22894108K`.
  - `8670161_2`: subj07 `relgate0`, completed `0:0`, elapsed `00:06:46`, MaxRSS `21520724K`.
  - `8670161_3`: subj07 `relgate_low` (`alpha=0.05`), completed `0:0`, elapsed `00:05:21`, MaxRSS `22798496K`.
- Failure class for all smoke rows: none.
- The recurring Slurm `couldn't chdir to /src` warning appeared and remained non-fatal, as in prior cycles.
- Smoke gate statistics:
  - subj05 `relgate0`: scale mean/min/max `1.000/1.000/1.000`, early/higher scale mean `1.000/1.000`.
  - subj05 `relgate_low`: scale mean/min/max `1.000/0.855/1.164`, early/higher scale mean `1.021/0.992`.
  - subj07 `relgate0`: scale mean/min/max `1.000/1.000/1.000`, early/higher scale mean `1.000/1.000`.
  - subj07 `relgate_low`: scale mean/min/max `1.000/0.864/1.173`, early/higher scale mean `1.034/0.988`.
- Smoke final 3-epoch diagnostics:
  - subj05 `relgate0`: test loss `13.1`, test PixCorr `0.250`, test fwd/bwd `0.407/0.283`; train loss `11.4`, train PixCorr `0.341`, train fwd/bwd `0.992/0.972`.
  - subj05 `relgate_low`: test loss `13.0`, test PixCorr `0.254`, test fwd/bwd `0.417/0.287`; train loss `11.4`, train PixCorr `0.342`, train fwd/bwd `0.992/0.972`.
  - subj07 `relgate0`: test loss `13.4`, test PixCorr `0.196`, test fwd/bwd `0.430/0.267`; train loss `11.5`, train PixCorr `0.286`, train fwd/bwd `0.976/0.966`.
  - subj07 `relgate_low`: test loss `13.3`, test PixCorr `0.198`, test fwd/bwd `0.430/0.273`; train loss `11.5`, train PixCorr `0.287`, train fwd/bwd `0.977/0.965`.

Full training/evaluation launched:
- Submitted `/src/cycle16_relgate_train_s57.slurm` as job `8671041`.
- Submitted `/src/cycle16_relgate_eval_s57.slurm` as dependent job `8671042` with `afterok:8671041`.
- Rows:
  - subj05 `relgate0`, `alpha=0.00`, model `cycle16_subj05_relgate0_1sess_150ep`.
  - subj05 `relgate_low`, `alpha=0.05`, model `cycle16_subj05_relgate_low_1sess_150ep`.
  - subj05 `relgate_mid`, `alpha=0.10`, model `cycle16_subj05_relgate_mid_1sess_150ep`.
  - subj07 `relgate0`, `alpha=0.00`, model `cycle16_subj07_relgate0_1sess_150ep`.
  - subj07 `relgate_low`, `alpha=0.05`, model `cycle16_subj07_relgate_low_1sess_150ep`.
  - subj07 `relgate_mid`, `alpha=0.10`, model `cycle16_subj07_relgate_mid_1sess_150ep`.
- At launch/status check, `8671041_[0-5]` was pending for priority, and `8671042_[0-5]` was dependency-pending.
- Expected final tables are `/src/tables/cycle16_subj0{5,7}_relgate{0,_low,_mid}_1sess_150ep_all_enhancedrecons.csv` after evaluator completion.

Current scientific decision:
- Operational preflight succeeded: train-repeat reliability is available, length-matched, ROI-aligned, and leakage checks passed for subjects 5 and 7.
- Smoke succeeded: no NaNs, finite losses, valid checkpoint writing, and conservative gate statistics.
- No refined metric decision is available yet because the 150-epoch training/evaluation grid has only been launched.
- Do not claim a MindEyeV2 improvement until dependent evaluator job `8671042` writes final CSVs and deltas versus same-subject `relgate0` are computed.

Telegram report-ready update:
- Cycle 16 implemented train-repeat reliability gating for MindEyeV2 weak subjects 5 and 7. Reliability was computed only from one-session training repeats, with zero overlap against old test or `new_test`; vectors match expected voxel counts (`13039`/`12682`) and ROI masks (`3661/9378` early/higher for subj05, `3251/9431` for subj07).
- Reliability tensors and diagnostics are in `/src/reliability/`. Mean shrunk reliability is `0.221` for subj05 and `0.172` for subj07; early visual reliability is higher than higher-visual in both subjects.
- The 1-hour smoke array `8670161` completed all four tasks successfully. Low gate `alpha=0.05` produced finite losses and conservative scales: subj05 scale range `0.855-1.164`, subj07 scale range `0.864-1.173`; `relgate0` scales were exactly `1.0`.
- Full training grid `8671041` is queued for subjects 5/7 and rows `relgate0`, `relgate_low`, `relgate_mid`; dependent evaluator `8671042` will run the fixed `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py` path. No final brain retrieval claim yet.

Recommended next research questions:
- After `8671042` completes, does `relgate_low` or `relgate_mid` improve refined brain retrieval by about `0.02` absolute versus same-subject `relgate0` without hurting CLIP, Inception, image retrieval, EfficientNet distance, SwAV distance, visual cortex, or higher-visual correlation?
- If centered scaling is neutral, should the next input-side variant use reliability-informed training-only dropout while preserving evaluation-time inputs?
- If one subject improves and the other does not, is the effect explained by the stronger early/higher reliability separation in subject 7 versus subject 5?

## Cycle 17 - 2026-05-24

Plan executed:
- Read `/plan.md` and executed only the Cycle 16 reliability-gating recovery/readout plan.
- Telegram report is not due.
- No new mechanism was launched.

Recovered Slurm/artifact state:
- Training array `8671041_[0-5]` failed on all six rows before completing epoch 0.
- Dependent evaluator `8671042_[0-5]` was stuck in `DependencyNeverSatisfied`; it was cancelled because no checkpoint row completed.
- No Cycle 16 final CSVs were present under `/src/tables` at recovery time.
- Failure class for all six training rows: operational CUDA memory failure in frozen SD VAE target encoding, not a reliability-gating logic failure.

Original failed rows:

| task | model | state | elapsed | MaxRSS | node | stdout/stderr | failure |
|---|---|---:|---:|---:|---|---|---|
| `8671041_0` | `cycle16_subj05_relgate0_1sess_150ep` | `FAILED 1:0` | `00:01:41` | `21608680K` | `della-i14g8` | `/src/slurms/c16_relgate_s57_8671041_0.out/.err` | OOM at `autoenc.encode` |
| `8671041_1` | `cycle16_subj05_relgate_low_1sess_150ep` | `FAILED 1:0` | `00:01:11` | `22809724K` | `della-i14g8` | `/src/slurms/c16_relgate_s57_8671041_1.out/.err` | OOM at `autoenc.encode` |
| `8671041_2` | `cycle16_subj05_relgate_mid_1sess_150ep` | `FAILED 1:0` | `00:01:11` | `22826096K` | `della-i14g8` | `/src/slurms/c16_relgate_s57_8671041_2.out/.err` | OOM at `autoenc.encode` |
| `8671041_3` | `cycle16_subj07_relgate0_1sess_150ep` | `FAILED 1:0` | `00:01:13` | `21442760K` | `della-i14g8` | `/src/slurms/c16_relgate_s57_8671041_3.out/.err` | OOM at `autoenc.encode` |
| `8671041_4` | `cycle16_subj07_relgate_low_1sess_150ep` | `FAILED 1:0` | `00:01:14` | `22811640K` | `della-i14g8` | `/src/slurms/c16_relgate_s57_8671041_4.out/.err` | OOM at `autoenc.encode` |
| `8671041_5` | `cycle16_subj07_relgate_mid_1sess_150ep` | `FAILED 1:0` | `00:01:13` | `22809616K` | `della-i14g8` | `/src/slurms/c16_relgate_s57_8671041_5.out/.err` | OOM at `autoenc.encode` |

Reliability-gate checks from the failed pre-epoch logs:
- `relgate0` rows loaded the reliability tensors but applied no effect:
  - subj05 scale mean/min/max `1.000/1.000/1.000`; early/higher scale mean `1.000/1.000`.
  - subj07 scale mean/min/max `1.000/1.000/1.000`; early/higher scale mean `1.000/1.000`.
- Nonzero rows used the intended strengths:
  - subj05 `relgate_low alpha=0.05`: scale mean/min/max `1.000/0.855/1.164`; early/higher scale mean `1.021/0.992`.
  - subj05 `relgate_mid alpha=0.10`: scale mean/min/max `1.000/0.750/1.250`; early/higher scale mean `1.043/0.983`.
  - subj07 `relgate_low alpha=0.05`: scale mean/min/max `1.000/0.864/1.173`; early/higher scale mean `1.034/0.988`.
  - subj07 `relgate_mid alpha=0.10`: scale mean/min/max `1.000/0.750/1.251`; early/higher scale mean `1.066/0.977`.

Operational repair:
- Patched `/src/Train.py` to encode SD VAE target latents in chunks of 8 under the existing `torch.no_grad()` block:
  - added `encode_autoenc_latents(autoenc, image_batch, chunk_size=8)`;
  - replaced the batch-24 `autoenc.encode(2*image-1)` call with the chunked helper.
- This does not change model inputs, labels, targets, losses, reliability tensors, checkpoint initialization, optimizer settings, or evaluation settings. It only reduces peak memory for a frozen target encoder.
- Updated `/src/cycle16_relgate_train_s57.slurm` wall time from `03:00:00` to `04:30:00` after the chunked smoke showed slower epoch timing. Memory remains `64G`, one A100.
- Added `/src/cycle17_relgate_fullmix_smoke.slurm` for a one-hour full-mix smoke with `num_epochs=150`, batch 24, subject 5 `relgate0`, so the early BiMixCo path matches the failed production run.

Validation and smoke:
- `/src/fmri/bin/python -m py_compile /src/Train.py` passed.
- `bash -n /src/cycle17_relgate_fullmix_smoke.slurm /src/cycle16_relgate_train_s57.slurm /src/cycle16_relgate_eval_s57.slurm` passed.
- Submitted smoke `8671597`; it started on `della-l09g7`, passed the previous OOM point, completed epoch 0 metrics, and was cancelled intentionally after `00:02:55` to avoid wasting the one-hour test allocation.
- Smoke epoch-0 metrics:
  - test loss `14.9`, test blurry PixCorr `0.244`, test fwd/bwd `0.157/0.0567`;
  - train loss `14.6`, train blurry PixCorr `0.165`, train fwd/bwd `0.297/0.147`.

Recovery jobs launched:
- Relaunched the exact six-row Cycle 16 grid with original model names: `sbatch /src/cycle16_relgate_train_s57.slurm`, job `8671676_[0-5]`.
- Relaunched dependent fixed-path evaluator: `sbatch --dependency=afterok:8671676 /src/cycle16_relgate_eval_s57.slurm`, job `8671677_[0-5]`.
- At write time, `8671676_[0-5]` is pending with `04:30:00` time limit and `8671677_[0-5]` is dependency-pending.

Metric table status:
- Final refined CSVs are still unavailable because the recovery training/evaluation arrays have not completed.
- Required pending CSVs remain:
  - `/src/tables/cycle16_subj05_relgate0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj05_relgate_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj05_relgate_mid_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate_mid_1sess_150ep_all_enhancedrecons.csv`

Scientific decision:
- No refined-metric decision can be made yet.
- The reliability-gate implementation remains operationally valid: leakage checks, tensor lengths, ROI masks, and no-effect `relgate0` scaling are still verified.
- Next action is to inspect `8671676_[0-5]`. If training completes, allow `8671677_[0-5]` to run and compute the subject-wise deltas versus same-code `relgate0`; if a row times out, resume or rerun only the affected row with the same subject, model name, reliability strength, and hyperparameters.

Recommended next research questions:
- Do the chunked-VAE recovery rows complete inside `04:30:00`, and do final training diagnostics match the Cycle 16 smoke/control behavior?
- Once CSVs exist, does `relgate_low` or `relgate_mid` improve refined brain retrieval by about `0.02` absolute on subject 5 or 7 without semantic or distance regressions?
- If the grid is neutral, the next input-side candidate remains reliability-informed training-only dropout using the same train-repeat tensors and same-code control.

## Cycle 18 - 2026-05-24

Plan executed:
- Read `/plan.md` and executed only the Cycle 18 recovery/readout plan.
- Telegram report is not due.
- No new mechanism, hyperparameter, reliability tensor, checkpoint initialization, evaluator setting, split, mask, or retrieval protocol was introduced.

Recovered scheduler/artifact state:
- Relaunched Cycle 16 training array `8671676_[0-5]` was partially successful at inspection time:
  - `8671676_1`: `cycle16_subj05_relgate_low_1sess_150ep`, `RUNNING`, elapsed `00:24:51`, node `della-l03g4`.
  - `8671676_5`: `cycle16_subj07_relgate_mid_1sess_150ep`, `RUNNING`, elapsed `00:20:07`, node `della-l02g12`.
  - `8671676_0`: `cycle16_subj05_relgate0_1sess_150ep`, `FAILED 1:0`, elapsed `00:01:43`, MaxRSS `21945336K`, node `della-i14g20`.
  - `8671676_2`: `cycle16_subj05_relgate_mid_1sess_150ep`, `FAILED 1:0`, elapsed `00:01:04`, MaxRSS `22823040K`, node `della-i14g20`.
  - `8671676_3`: `cycle16_subj07_relgate0_1sess_150ep`, `FAILED 1:0`, elapsed `00:01:14`, MaxRSS `22632576K`, node `della-i14g20`.
  - `8671676_4`: `cycle16_subj07_relgate_low_1sess_150ep`, `FAILED 1:0`, elapsed `00:01:13`, MaxRSS `22810816K`, node `della-i14g20`.
- Failure class for the four failed rows: operational CUDA memory failure at `accelerator.backward(loss)` during epoch 0, not reliability-gating logic. Each traceback reported a 774 MiB allocation failure with about 520-595 MiB free on a 39.49 GiB A100.
- The stale dependent evaluator array `8671677_[0-5]` remained dependency-pending after parent failures, so it was cancelled.
- No required Cycle 16 final CSVs were present under `/src/tables` at inspection time.
- Login-side `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2` is not visible, so checkpoint-path verification was limited to Slurm logs and scheduler state from the login container. This is an observability limit, not an experimental failure.

Reliability/scale diagnostics recovered from logs:
- `relgate0` rows loaded the reliability tensors and applied exact no-effect scaling:
  - subj05 `relgate0`: scale mean/min/max `1.000/1.000/1.000`, early/higher scale mean `1.000/1.000`.
  - subj07 `relgate0`: scale mean/min/max `1.000/1.000/1.000`, early/higher scale mean `1.000/1.000`.
- Nonzero rows used the intended strengths:
  - subj05 `relgate_low alpha=0.05`: scale mean/min/max `1.000/0.855/1.164`, early/higher scale mean `1.021/0.992`.
  - subj05 `relgate_mid alpha=0.10`: scale mean/min/max `1.000/0.750/1.250`, early/higher scale mean `1.043/0.983`.
  - subj07 `relgate_low alpha=0.05`: scale mean/min/max `1.000/0.864/1.173`, early/higher scale mean `1.034/0.988`.
  - subj07 `relgate_mid alpha=0.10`: scale mean/min/max `1.000/0.750/1.251`, early/higher scale mean `1.066/0.977`.

Operational repair/relaunch:
- Added `/src/cycle18_relgate_recover_oom_s57.slurm`.
  - Relaunches only failed tasks `0,2,3,4`.
  - Preserves the exact model names, subject mapping, reliability tensors, reliability strengths, batch size `24`, one-session `150` epoch protocol, official multisubject initialization, optimizer/training arguments, and checkpoint names.
  - Adds only scheduler-level recovery constraints: `--array=0,2,3,4`, `--exclude=della-i14g8,della-i14g20`, and `--chdir=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`.
- Validation: `bash -n /src/cycle18_relgate_recover_oom_s57.slurm /src/cycle16_relgate_eval_s57.slurm` passed.
- Submitted recovery array: `sbatch /src/cycle18_relgate_recover_oom_s57.slurm`, job `8672355_[0,2,3,4]`.
  - At status check, `8672355_0` (`cycle16_subj05_relgate0_1sess_150ep`) was `RUNNING`, elapsed `00:04:27`, node `della-l05g2`, and had cleared the original epoch-0 OOM point with logged epoch-1 metrics.
  - `8672355_2`, `8672355_3`, and `8672355_4` were `PENDING (Priority)`.
- Submitted replacement evaluator: `sbatch --dependency=afterok:8671676_1:8671676_5:8672355 /src/cycle16_relgate_eval_s57.slurm`, job `8672425_[0-5]`.
  - `scontrol` verified dependency as `afterok:8671676_1`, `afterok:8671676_5`, and `afterok:8672355_*`.
  - At status check, `8672425_[0-5]` was `PENDING (Dependency)`.

Observed training diagnostics so far:
- `8672355_0` subj05 `relgate0` epoch-1 diagnostic matched the Cycle 17 smoke pattern and verified no-effect reliability:
  - test loss `14.9`, test blurry PixCorr `0.244`, test fwd/bwd `0.157/0.0567`;
  - train loss `14.6`, train blurry PixCorr `0.165`, train fwd/bwd `0.297/0.147`;
  - train/test `rel_loss=0` and `rel_sim_corr=0`.
- `8671676_1` and `8671676_5` were both beyond epoch 0 and continuing to log finite losses/retrieval. No final training metrics were available yet.

Metric table status:
- Required refined CSVs are still unavailable:
  - `/src/tables/cycle16_subj05_relgate0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj05_relgate_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj05_relgate_mid_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate_mid_1sess_150ep_all_enhancedrecons.csv`
- Sensitivity diagnostic was not run because final checkpoints for all six rows are not yet complete/visible.

Scientific decision:
- No refined-metric decision can be made in Cycle 18 yet.
- The no-effect `relgate0` controls remain verified at scale level.
- The only active issue is operational completion of the six exact rows. Next action is to inspect `8671676_1`, `8671676_5`, `8672355_[0,2,3,4]`, then allow `8672425_[0-5]` to run if all training dependencies finish successfully.

Recommended next research questions:
- Do the recovery rows excluded from `della-i14g8/della-i14g20` complete within `04:30:00`, or is a second semantic-preserving memory mitigation needed?
- Once `8672425_[0-5]` writes CSVs, does `relgate_low` or `relgate_mid` improve refined brain retrieval by about `0.02` absolute versus same-subject `relgate0` without semantic/distance regressions?
- If centered reliability scaling is neutral or harmful, should the next mechanism be reliability-informed training-only dropout using the same leakage-free tensors and a same-code no-dropout control?

## Cycle 19 - 2026-05-24

Plan executed:
- Read `/plan.md` and executed only the centered reliability-gate recovery/readout plan.
- Telegram report is not due.
- No new method, reliability tensor, hyperparameter, evaluator setting, split, mask, checkpoint initialization, or retrieval protocol was introduced.

Scheduler/artifact state recovered:
- Original continuing rows remained active and had cleared the epoch-0 OOM point:
  - `8671676_1`: `cycle16_subj05_relgate_low_1sess_150ep`, `RUNNING`, elapsed `00:41:51`, node `della-l03g4`.
  - `8671676_5`: `cycle16_subj07_relgate_mid_1sess_150ep`, `RUNNING`, elapsed `00:37:07`, node `della-l02g12`.
- Cycle 18 recovery row still active:
  - `8672355_0`: `cycle16_subj05_relgate0_1sess_150ep`, `RUNNING`, elapsed `00:21:27`, node `della-l05g2`.
- Cycle 18 recovery rows `8672355_2`, `8672355_3`, and `8672355_4` failed quickly on `della-i14g19`:
  - `8672355_2`: `cycle16_subj05_relgate_mid_1sess_150ep`, `FAILED 1:0`, elapsed `00:01:42`, MaxRSS `22141164K`.
  - `8672355_3`: `cycle16_subj07_relgate0_1sess_150ep`, `FAILED 1:0`, elapsed `00:01:12`, MaxRSS `21443948K`.
  - `8672355_4`: `cycle16_subj07_relgate_low_1sess_150ep`, `FAILED 1:0`, elapsed `00:01:11`, MaxRSS `22805352K`.
- Failure class for the three newly failed rows: operational CUDA memory failure at `accelerator.backward(loss)` during epoch 0, same class as prior `della-i14g20` failures. Tracebacks show a 774 MiB allocation failure with only about 520-595 MiB free on a 39.49 GiB A100. This is not evidence of reliability-gate scientific failure.
- Stale evaluator `8672425_[0-5]` depended on the whole failed `8672355` array, so it was cancelled before running.
- No required final refined CSVs exist yet under `/src/tables`.

Code/config changes:
- Added `/src/cycle19_relgate_recover_oom_s57.slurm`.
  - Relaunches only tasks `2,3,4`: subj05 `relgate_mid`, subj07 `relgate0`, and subj07 `relgate_low`.
  - Preserves exact model names, subjects, reliability tensors, strengths, batch size `24`, one-session `150` epochs, official multisubject initialization, chunked frozen SD-VAE target encoding repair, and all existing training arguments.
  - Adds scheduler-only exclusions for observed low-headroom nodes: `della-i14g8,della-i14g19,della-i14g20`.
- Validation: `bash -n /src/cycle19_relgate_recover_oom_s57.slurm /src/cycle16_relgate_eval_s57.slurm` passed.

Commands/jobs launched:
- `scancel 8672425`
- `sbatch /src/cycle19_relgate_recover_oom_s57.slurm`
  - Submitted job `8673678_[2,3,4]`.
  - At final check all three were `PENDING`, with scheduled nodes visible for at least the tasks inspected and exclusions applied as `della-i14g[8,19-20]`.
- `sbatch --dependency=afterok:8671676_1:8671676_5:8672355_0:8673678 /src/cycle16_relgate_eval_s57.slurm`
  - Submitted replacement evaluator `8673688_[0-5]`.
  - `scontrol` verified dependency as `afterok:8671676_1`, `afterok:8671676_5`, `afterok:8672355_0`, and `afterok:8673678_*`.

Reliability/scale provenance:
- Reliability tensors remain:
  - `/src/reliability/subj05_trainrepeat_reliability.pt`
  - `/src/reliability/subj07_trainrepeat_reliability.pt`
- Verified from logs and previous summaries:
  - subj05 voxel count `13039`, early/higher `3661/9378`.
  - subj07 voxel count `12682`, early/higher `3251/9431`.
  - Reliability was computed from one-session training repeats with old-test and `new_test` overlaps excluded.
  - `relgate0` scale mean/min/max is `1.000/1.000/1.000`.
  - Nonzero strengths remain low `alpha=0.05` and mid `alpha=0.10`.
- Scale summaries remain:
  - subj05 `relgate_low`: mean/min/max `1.000/0.855/1.164`, early/higher `1.021/0.992`.
  - subj05 `relgate_mid`: mean/min/max `1.000/0.750/1.250`, early/higher `1.043/0.983`.
  - subj07 `relgate_low`: mean/min/max `1.000/0.864/1.173`, early/higher `1.034/0.988`.
  - subj07 `relgate_mid`: mean/min/max `1.000/0.750/1.251`, early/higher `1.066/0.977`.

Observed running diagnostics:
- Active rows are logging finite losses and retrieval diagnostics with train/test `rel_loss=0` and `rel_sim_corr=0`, as expected for this input-scaling experiment.
- Approximate last parsed progress before writeout:
  - `8671676_1` subj05 `relgate_low`: around epoch 45/150, test blurry PixCorr about `0.171`, test loss about `15.5`, test fwd/bwd about `0.487/0.397`, train loss about `10.7`.
  - `8671676_5` subj07 `relgate_mid`: around epoch 40/150, test blurry PixCorr about `0.160`, test loss about `16.3`, test fwd/bwd about `0.480/0.307`, train loss about `10.8`.
  - `8672355_0` subj05 `relgate0`: around epoch 22/150, test blurry PixCorr about `0.200`, test loss about `13.1`, test fwd/bwd about `0.520/0.280`, train loss about `11.4`.
- These are interim training diagnostics only and are not a refined evaluation result.

Metric table status:
- Required refined CSVs remain unavailable:
  - `/src/tables/cycle16_subj05_relgate0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj05_relgate_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj05_relgate_mid_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate_mid_1sess_150ep_all_enhancedrecons.csv`
- Sensitivity diagnostic was not run because the six checkpoints/final CSVs are not yet complete.

Scientific decision:
- No scientific success/failure call can be made in Cycle 19.
- The centered reliability-gate attempt remains operationally in recovery. The next valid action is to let `8671676_1`, `8671676_5`, `8672355_0`, and `8673678_[2,3,4]` finish, then allow evaluator `8673688_[0-5]` to run.

Recommended next research questions:
- Do the Cycle 19 reruns avoid the epoch-0 backward OOM on non-excluded nodes?
- If all six checkpoints complete, does evaluator `8673688_[0-5]` produce the six required refined CSVs through the fixed path?
- Once CSVs exist, do `relgate_low` or `relgate_mid` improve refined brain retrieval by about `0.02` absolute versus same-subject `relgate0` without semantic or distance regressions?

## Cycle 20 - 2026-05-24

Plan executed:
- Read `/plan.md` and executed only the centered reliability-gate recovery/readout plan.
- Telegram report is not due.
- No new training method, evaluator setting, reliability tensor, split, mask, checkpoint initialization, retrieval protocol, or architecture change was introduced.

Scheduler/artifact state:
- No Cycle 20 training or reconstruction reruns were needed. The Cycle 19 recovery and dependent evaluator completed before this readout.
- Required training rows:
  - `8671676_1` subj05 `relgate_low`: `COMPLETED 0:0`, elapsed `02:10:42`, MaxRSS `22318580K`, node `della-l03g4`, logs `/src/slurms/c16_relgate_s57_8671676_1.out/.err`.
  - `8671676_5` subj07 `relgate_mid`: `COMPLETED 0:0`, elapsed `02:11:58`, MaxRSS `22962676K`, node `della-l02g12`, logs `/src/slurms/c16_relgate_s57_8671676_5.out/.err`.
  - `8672355_0` subj05 `relgate0`: `COMPLETED 0:0`, elapsed `02:04:35`, MaxRSS `22975100K`, node `della-l05g2`, logs `/src/slurms/c18_relgate_recover_s57_8672355_0.out/.err`.
  - `8673678_2` subj05 `relgate_mid`: `COMPLETED 0:0`, elapsed `02:05:26`, MaxRSS `22967832K`, node `della-l04g1`, logs `/src/slurms/c19_relgate_recover_s57_8673678_2.out/.err`.
  - `8673678_3` subj07 `relgate0`: `COMPLETED 0:0`, elapsed `02:07:10`, MaxRSS `21595948K`, node `della-l03g16`, logs `/src/slurms/c19_relgate_recover_s57_8673678_3.out/.err`.
  - `8673678_4` subj07 `relgate_low`: `COMPLETED 0:0`, elapsed `02:10:49`, MaxRSS `22802700K`, node `della-l02g12`, logs `/src/slurms/c19_relgate_recover_s57_8673678_4.out/.err`.
- Historical failed attempts remain classified as operational epoch-0 CUDA memory failures on low-headroom `della-i14g*` nodes, recovered by exact-grid reruns with node exclusions and the Cycle 17 chunked frozen SD-VAE target-encoding repair.
- Dependent evaluator `8673688_[0-5]` completed all six rows through `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py`:
  - tasks `0-5`: all `COMPLETED 0:0`, elapsed about `02:23:55-02:25:29`, MaxRSS about `47074996K-49086628K`, nodes `della-l05g7`, `della-l05g5`, and `della-l05g4`.
  - Logs: `/src/slurms/c16_relgate_eval_s57_8673688_{0..5}.out/.err`.

Final artifact paths:
- Final CSVs are present:
  - `/src/tables/cycle16_subj05_relgate0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj05_relgate_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj05_relgate_mid_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle16_subj07_relgate_mid_1sess_150ep_all_enhancedrecons.csv`
- Enhanced recon tensors and evaluator intermediates are present under `/src/evals/cycle16_subj0{5,7}_relgate{0,_low,_mid}_1sess_150ep/`.
- Compute-visible checkpoint paths used by the evaluator and diagnostic are `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/<model_name>/last.pth`. Login-side `/scratch` remains invisible, so checkpoint file inspection from the container is limited; compute jobs successfully loaded them.

Reliability/scale provenance:
- Reliability tensors:
  - `/src/reliability/subj05_trainrepeat_reliability.pt`
  - `/src/reliability/subj07_trainrepeat_reliability.pt`
- Provenance remains leakage-free: one-session training repeats only, with old-test and `new_test` overlaps excluded.
- Voxel and ROI counts verified in logs and diagnostic: subj05 `13039`, early/higher `3661/9378`; subj07 `12682`, early/higher `3251/9431`.
- `relgate0` rows are exact no-effect controls: scale mean/min/max `1.000/1.000/1.000`, early/higher `1.000/1.000`.
- Nonzero scale summaries:
  - subj05 `low alpha=0.05`: mean/min/max `1.000/0.855/1.164`, early/higher `1.021/0.992`.
  - subj05 `mid alpha=0.10`: mean/min/max `1.000/0.750/1.250`, early/higher `1.043/0.983`.
  - subj07 `low alpha=0.05`: mean/min/max `1.000/0.864/1.173`, early/higher `1.034/0.988`.
  - subj07 `mid alpha=0.10`: mean/min/max `1.000/0.750/1.251`, early/higher `1.066/0.977`.

Final refined metrics:

| subj | row | PixCorr | SSIM | Alex2 | Alex5 | Incept | CLIP | EffNet | SwAV | ImgRet | BrainRet | VC | V1 | V2 | V3 | V4 | Higher |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 05 | relgate0 | 0.1927 | 0.4069 | 0.8468 | 0.9161 | 0.8562 | 0.8405 | 0.7681 | 0.4312 | 0.6488 | 0.5528 | 0.4164 | 0.3519 | 0.3560 | 0.3410 | 0.3151 | 0.4236 |
| 05 | low | 0.2012 | 0.4111 | 0.8485 | 0.9231 | 0.8600 | 0.8397 | 0.7674 | 0.4332 | 0.6421 | 0.5246 | 0.4101 | 0.3492 | 0.3549 | 0.3377 | 0.3114 | 0.4176 |
| 05 | mid | 0.1885 | 0.4087 | 0.8458 | 0.9200 | 0.8581 | 0.8451 | 0.7654 | 0.4339 | 0.6367 | 0.5382 | 0.4132 | 0.3438 | 0.3530 | 0.3376 | 0.3122 | 0.4224 |
| 07 | relgate0 | 0.1943 | 0.4047 | 0.8290 | 0.8940 | 0.7851 | 0.7710 | 0.8282 | 0.4779 | 0.6806 | 0.5420 | 0.3213 | 0.3217 | 0.3250 | 0.3160 | 0.2813 | 0.3070 |
| 07 | low | 0.1997 | 0.4012 | 0.8276 | 0.8875 | 0.7949 | 0.7844 | 0.8220 | 0.4741 | 0.6949 | 0.5518 | 0.3184 | 0.3153 | 0.3182 | 0.3109 | 0.2770 | 0.3038 |
| 07 | mid | 0.2009 | 0.4036 | 0.8175 | 0.8814 | 0.7868 | 0.7791 | 0.8273 | 0.4733 | 0.6724 | 0.5191 | 0.3130 | 0.3079 | 0.3107 | 0.3029 | 0.2718 | 0.2984 |

Deltas versus same-subject `relgate0`:
- subj05 `low`: BrainRet `-0.0282`; CLIP `-0.0008`; Inception `+0.0038`; ImgRet `-0.0067`; EffNet `-0.0007` lower/better; SwAV `+0.0020` higher/worse; VC `-0.0062`; Higher `-0.0060`.
- subj05 `mid`: BrainRet `-0.0146`; CLIP `+0.0045`; Inception `+0.0019`; ImgRet `-0.0121`; EffNet `-0.0027` lower/better; SwAV `+0.0027` higher/worse; VC `-0.0032`; Higher `-0.0013`.
- subj07 `low`: BrainRet `+0.0098`; CLIP `+0.0134`; Inception `+0.0098`; ImgRet `+0.0143`; EffNet `-0.0062` lower/better; SwAV `-0.0038` lower/better; VC `-0.0029`; Higher `-0.0032`.
- subj07 `mid`: BrainRet `-0.0229`; CLIP `+0.0081`; Inception `+0.0017`; ImgRet `-0.0081`; EffNet `-0.0010` lower/better; SwAV `-0.0046` lower/better; VC `-0.0083`; Higher `-0.0086`.
- Weak-subject mean BrainRet: `relgate0 0.5474`, `low 0.5382`, `mid 0.5287`.

Final training diagnostics from epoch 150:
- subj05 `relgate0`: test loss `14.3`, blurry PixCorr `0.193`, test fwd/bwd `0.670/0.583`; train loss `5.88`, train blurry PixCorr `0.801`, train fwd/bwd `1.000/1.000`, train/test `rel_loss=0`, `rel_sim_corr=0`.
- subj05 `low`: test loss `14.5`, blurry PixCorr `0.189`, test fwd/bwd `0.673/0.540`; train loss `5.87`, train blurry PixCorr `0.800`, train fwd/bwd `1.000/1.000`, train/test `rel_loss=0`, `rel_sim_corr=0`.
- subj05 `mid`: test loss `14.4`, blurry PixCorr `0.180`, test fwd/bwd `0.663/0.563`; train loss `5.87`, train blurry PixCorr `0.800`, train fwd/bwd `1.000/1.000`, train/test `rel_loss=0`, `rel_sim_corr=0`.
- subj07 `relgate0`: test loss `14.2`, blurry PixCorr `0.226`, test fwd/bwd `0.730/0.563`; train loss `5.95`, train blurry PixCorr `0.788`, train fwd/bwd `1.000/1.000`, train/test `rel_loss=0`, `rel_sim_corr=0`.
- subj07 `low`: test loss `14.1`, blurry PixCorr `0.231`, test fwd/bwd `0.767/0.583`; train loss `5.95`, train blurry PixCorr `0.789`, train fwd/bwd `1.000/1.000`, train/test `rel_loss=0`, `rel_sim_corr=0`.
- subj07 `mid`: test loss `14.2`, blurry PixCorr `0.243`, test fwd/bwd `0.743/0.567`; train loss `5.94`, train blurry PixCorr `0.788`, train fwd/bwd `1.000/1.000`, train/test `rel_loss=0`, `rel_sim_corr=0`.

Post-hoc mechanism diagnostic:
- Added `/src/cycle20_sensitivity_diagnostic.py` and `/src/cycle20_sensitivity_diagnostic.slurm`.
- Submitted `sbatch /src/cycle20_sensitivity_diagnostic.slurm`, job `8707430`; `COMPLETED 0:0`, elapsed `00:01:24`, MaxRSS `35068612K`, node `della-i13n25`.
- Output: `/src/tables/cycle20_relgate_sensitivity.csv`.
- Diagnostic uses `ridge.linears.0.weight.abs().mean(dim=0)` as learned voxel input sensitivity and Spearman correlation with the leakage-free reliability tensor.
- Spearman reliability/sensitivity summary:
  - subj05 `relgate0`: all `-0.041`, early `+0.270`, higher `-0.177`.
  - subj05 `low`: all `-0.059`, early `+0.243`, higher `-0.189`.
  - subj05 `mid`: all `-0.077`, early `+0.220`, higher `-0.199`.
  - subj07 `relgate0`: all `+0.189`, early `+0.457`, higher `+0.031`.
  - subj07 `low`: all `+0.175`, early `+0.435`, higher `+0.021`.
  - subj07 `mid`: all `+0.150`, early `+0.394`, higher `+0.006`.
- Interpretation: sensitivity is much more aligned with reliability in early visual cortex than higher visual cortex, especially for subj07; stronger centered scaling reduces, not increases, the reliability/sensitivity correlation while higher-visual refined correlations degrade. This does not support a semantic reliability-gate mechanism.

Scientific decision:
- Operational success criteria are satisfied: all six required centered-gate rows have final refined CSVs, `relgate0` is verified as same-code no-effect scaling, and deltas versus same-subject controls are recorded.
- Scientific success criteria are not met. No nonzero gate improves refined brain retrieval by about `0.02` absolute. Subject 5 is harmful for both strengths. Subject 7 `low` has a small BrainRet gain of `+0.0098` with slight VC/Higher drops and does not meet the decision threshold; subject 7 `mid` is harmful.
- Centered deterministic reliability scaling is therefore classified as neutral-to-harmful for weak subjects 5/7, not a MindEyeV2 improvement.

Recommended next research questions:
- Do not increase fixed centered scaling strength; `mid` already reaches the configured clipping bounds and worsens weak-subject mean BrainRet.
- The next valid input-side candidate is reliability-informed training-only dropout or multiplicative noise using the same leakage-free tensors and unchanged evaluation-time inputs, with same-code no-dropout controls.
- If pursuing subject 7 only, repeat subj07 `relgate0` and `relgate_low` first before treating the small `+0.0098` BrainRet movement as anything beyond noise.

## Cycle 21 - 2026-05-24

Plan executed:
- Read `/plan.md` and executed only the Cycle 21 reliability-uncertainty plan.
- Telegram report is due; report-ready summary is appended below.
- Did not use `/strategizing-chat.md` or researcher notes.

Code/config changes:
- Patched `/src/Train.py` with disabled-by-default `--reliability_mode=inverted_voxel_dropout`.
- Added CLI flags `--reldrop_p_base` and `--reldrop_p_span`.
- Preserved the existing `centered_scale` path for prior relgate rows; the new dropout path is separate and opt-in.
- New dropout mode loads the leakage-free train-repeat reliability tensor, converts lower reliability to higher bounded uncertainty, and applies inverted voxel dropout only to training batches on device immediately before the ridge/model forward pass.
- Validation/test fMRI remains unperturbed for `inverted_voxel_dropout`; reconstruction/evaluator inputs, targets, image latents, CLIP targets, blurry targets, captions, and saved beta data are not modified.
- Added epoch-level logging for assigned probability summaries in `reliability_summary.json` and realized train drop/keep rates for all, early visual, and higher visual voxels in train logs.
- Confirmed the frozen target-encoder `torch.no_grad()` and chunked SD-VAE target-encoding repairs remain present in `/src/Train.py`.

New Slurm files:
- `/src/cycle21_reldrop_smoke.slurm`: required 1-hour subject 7 smoke for `reldrop0` and `reldrop_low`, 3 epochs, batch 24, one A100, official multisubject initialization.
- `/src/cycle21_reldrop_train_s57.slurm`: six full required rows for subjects 5/7 and `reldrop0/low/mid`, one-session 150 epochs.
- `/src/cycle21_reldrop_eval_s57.slurm`: fixed evaluation path `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py`.

Validation:
- `/src/fmri/bin/python -m py_compile /src/Train.py /src/recon_inference.py /src/enhanced_recon_inference.py /src/final_evaluations.py` passed.
- `bash -n /src/cycle21_reldrop_smoke.slurm /src/cycle21_reldrop_train_s57.slurm /src/cycle21_reldrop_eval_s57.slurm` passed.

Assigned dropout probability diagnostics:
- subj05 `reldrop0`: all/early/higher mean `0.0000/0.0000/0.0000`, range `0.0000-0.0000`.
- subj05 `reldrop_low`: all mean `0.0625`, range `0.0200-0.1000`, early/higher mean `0.0570/0.0647`, reliability/probability corr `-1.000`.
- subj05 `reldrop_mid`: all mean `0.1144`, range `0.0400-0.1800`, early/higher mean `0.1047/0.1182`, reliability/probability corr about `-1.000`.
- subj07 `reldrop0`: all/early/higher mean `0.0000/0.0000/0.0000`, range `0.0000-0.0000`.
- subj07 `reldrop_low`: all mean `0.0648`, range `0.0200-0.1000`, early/higher mean `0.0561/0.0678`, reliability/probability corr `-1.000`.
- subj07 `reldrop_mid`: all mean `0.1184`, range `0.0400-0.1800`, early/higher mean `0.1031/0.1236`, reliability/probability corr about `-1.000`.
- Early/higher mean probability imbalance is under `2x` for both subjects and strengths, so no ROI-balanced normalization was introduced.
- Voxel/ROI counts remain: subj05 `13039` total, early/higher `3661/9378`; subj07 `12682` total, early/higher `3251/9431`.

Commands/jobs launched:
- Submitted required smoke only: `sbatch /src/cycle21_reldrop_smoke.slurm`.
- Smoke job: `8707727_[0-1]`.
  - `8707727_0`: subj07 `reldrop0`, model `cycle21_smoke_subj07_reldrop0_1sess_3ep`, pending at last check.
  - `8707727_1`: subj07 `reldrop_low`, model `cycle21_smoke_subj07_reldrop_low_1sess_3ep`, pending at last check.
- Last scheduler check: `2026-05-24 14:43:27 EDT`; both smoke tasks were `PENDING`, elapsed `00:00`, with no node assigned yet.
- Full six-row training was intentionally not submitted because `/plan.md` requires smoke to pass first.
- Evaluation was not submitted because full checkpoints do not exist yet.

Observed metrics/results:
- No Cycle 21 training metrics, realized dropout rates, GPU memory, or final refined CSVs are available yet because the smoke had not started by the last scheduler check.
- Required final CSV paths remain expected, not produced:
  - `/src/tables/cycle21_subj05_reldrop0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle21_subj05_reldrop_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle21_subj05_reldrop_mid_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle21_subj07_reldrop0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle21_subj07_reldrop_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle21_subj07_reldrop_mid_1sess_150ep_all_enhancedrecons.csv`

Conclusions:
- Cycle 21 implementation is ready and statically validated.
- The smoke gate is now queued. The next valid action is to inspect `8707727_[0-1]` logs after it runs, verify `reldrop0` no-effect behavior, stochastic realized dropout for `reldrop_low`, finite train/test losses, unperturbed validation/test behavior, and GPU memory, then submit `/src/cycle21_reldrop_train_s57.slurm` only if the smoke passes.

Recommended next research questions:
- Do the realized subj07 `reldrop_low` drop rates match assigned all/early/higher probabilities without fixed masks across batches?
- Does `reldrop0` reproduce the same-code no-effect training diagnostics while still loading/logging reliability?
- If the smoke passes, do the six full rows improve refined brain retrieval by about `0.02` absolute versus same-subject `reldrop0` without semantic or higher-visual regressions?

Telegram-ready update:
Cycle 21 implemented the planned training-only reliability dropout in `/src/Train.py` and queued the required 1-hour smoke. New mode is opt-in via `--reliability_mode=inverted_voxel_dropout`, applies inverted voxel dropout only to training batches immediately before the ridge forward, and leaves validation/test/reconstruction/evaluator fMRI unperturbed. Static probability checks match plan: low ranges `0.02-0.10`, mid ranges `0.04-0.18`; early/higher probability imbalance is below `2x` for subjects 5/7. Validation passed (`py_compile` and `bash -n`). Submitted smoke job `8707727_[0-1]` for subj07 `reldrop0` and `reldrop_low`; both were still pending at `2026-05-24 14:43 EDT`, so no train metrics, realized drop rates, or refined CSVs exist yet. Full six-row training has not been submitted because the plan requires smoke to pass first.

Cycle 21 smoke/full-launch addendum:
- Smoke `8707727_[0-1]` started after the first writeout and completed successfully on `della-l09g7`.
- `8707727_0` subj07 `reldrop0`: `COMPLETED 0:0`, elapsed `00:05:40`, MaxRSS `21830856K`, logs `/src/slurms/c21_reldrop_smoke_8707727_0.out/.err`.
- `8707727_1` subj07 `reldrop_low`: `COMPLETED 0:0`, elapsed `00:05:40`, MaxRSS `21876712K`, logs `/src/slurms/c21_reldrop_smoke_8707727_1.out/.err`.
- Smoke final epoch diagnostics:
  - `reldrop0`: test loss `12.3`, test blurry PixCorr `0.168`, test fwd/bwd `0.413/0.253`; train loss `11.4`, train blurry PixCorr `0.286`, train fwd/bwd `0.974/0.958`; realized train drop rates all/early/higher `0.0000/0.0000/0.0000`.
  - `reldrop_low`: test loss `12.3`, test blurry PixCorr `0.167`, test fwd/bwd `0.403/0.257`; train loss `11.4`, train blurry PixCorr `0.284`, train fwd/bwd `0.976/0.957`; realized train drop rates all/early/higher `0.0646/0.0562/0.0675`.
- Smoke pass/fail call: passed. `reldrop0` is a no-effect same-code control for dropout, `reldrop_low` realized stochastic nonzero train-only drops at the assigned all/early/higher rates, validation/test metrics remained finite, and GPU memory stayed below the prior OOM pattern.
- Added `#SBATCH --chdir=/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src` to Cycle 21 Slurm files after smoke to remove the harmless compute-side `/src` chdir warning.
- Launched full required training: `sbatch /src/cycle21_reldrop_train_s57.slurm`, job `8708124_[0-5]`, pending at last check.
- Launched dependent evaluator: `sbatch --dependency=afterok:8708124 /src/cycle21_reldrop_eval_s57.slurm`, job `8708125_[0-5]`, dependency-pending at last check. `scontrol` verified dependency `afterok:8708124_*` and workdir `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`.
- Updated Telegram-ready status: Smoke passed and full Cycle 21 is now queued. Full rows are `cycle21_subj05_reldrop0_1sess_150ep`, `cycle21_subj05_reldrop_low_1sess_150ep`, `cycle21_subj05_reldrop_mid_1sess_150ep`, `cycle21_subj07_reldrop0_1sess_150ep`, `cycle21_subj07_reldrop_low_1sess_150ep`, and `cycle21_subj07_reldrop_mid_1sess_150ep`; final refined CSVs are still pending training and dependent evaluation.

## Cycle 22 - 2026-05-24

Plan executed:
- Read `/plan.md` and executed only the Cycle 22 recovery/readout plan.
- Telegram report is not due.
- No code, reliability tensor, dropout strength, split, mask, initialization, evaluator setting, candidate pool, or model name was changed.

Scheduler state recovered:
- Required training array `8708124_[0-5]` remains `PENDING (Priority)`, elapsed `00:00:00`, exit `0:0`, no MaxRSS and no allocated node yet. The array is still the first source of truth for all six exact Cycle 21 rows.
- Training tasks map as:
  - `8708124_0`: `cycle21_subj05_reldrop0_1sess_150ep`, pending; scheduled start shown as `2026-05-24T16:24:00`, scheduled node `della-l02g13`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_0.out`, stderr `.err`.
  - `8708124_1`: `cycle21_subj05_reldrop_low_1sess_150ep`, pending; scheduled start `2026-05-24T16:54:00`, scheduled node `della-l05g7`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_1.out`, stderr `.err`.
  - `8708124_2`: `cycle21_subj05_reldrop_mid_1sess_150ep`, pending; scheduled start `2026-05-24T16:54:00`, scheduled node `della-l04g6`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_2.out`, stderr `.err`.
  - `8708124_3`: `cycle21_subj07_reldrop0_1sess_150ep`, pending; no scheduled start/node shown yet, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_3.out`, stderr `.err`.
  - `8708124_4`: `cycle21_subj07_reldrop_low_1sess_150ep`, pending; no scheduled start/node shown yet, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_4.out`, stderr `.err`.
  - `8708124_5`: `cycle21_subj07_reldrop_mid_1sess_150ep`, pending; no scheduled start/node shown yet, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_5.out`, stderr `.err`.
- Training array settings verified by `scontrol`: `04:30:00`, `64G`, one `a100`, `gpu-short`, workdir `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`, `--exclude=della-i14g8,della-i14g19,della-i14g20`, command `/src/cycle21_reldrop_train_s57.slurm`.
- Dependent evaluator array `8708125_[0-5]` remains `PENDING (Dependency)`, elapsed `00:00:00`, exit `0:0`, no MaxRSS and no node. Dependency is `afterok:8708124_*`, workdir `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`, time `04:00:00`, memory `64G`.
- Scheduler observability is partial from the login/container side: `/scratch/gpfs/...` paths used by compute jobs are not directly inspectable here for pending-job artifacts, so current artifact checks are based on visible `/src` paths plus Slurm metadata.

Artifact state:
- None of the six required full-row checkpoints are visible yet at `/src/train_logs/<model_name>/last.pth`, and no compute-visible `/scratch/.../train_logs/<model_name>/last.pth` can be confirmed before the pending jobs start.
- No enhanced reconstruction tensors exist yet under `/src/evals/<model_name>/` for the six required rows.
- No final CSVs exist yet at `/src/tables/<model_name>_all_enhancedrecons.csv`.
- Therefore no final evaluator row has used enhanced reconstructions yet; the evaluator is correctly waiting on successful training completion.

Smoke diagnostics carried forward:
- Smoke `8707727_[0-1]` completed successfully before Cycle 22 and remains the required preflight gate.
- `8707727_0`/`8707727_1` both completed on `della-l09g7` in `00:05:40`; MaxRSS was `21830856K` for `reldrop0` and `21876712K` for `reldrop_low`.
- Smoke `reldrop0` final realized train drop rates all/early/higher were `0.0000/0.0000/0.0000`.
- Smoke `reldrop_low` final realized train drop rates all/early/higher were `0.0646/0.0562/0.0675`, matching the assigned probability scale.
- Smoke final diagnostics were finite: `reldrop0` test loss `12.3`, blurry PixCorr `0.168`, test fwd/bwd `0.413/0.253`; `reldrop_low` test loss `12.3`, blurry PixCorr `0.167`, test fwd/bwd `0.403/0.257`.

Reliability/dropout provenance:
- Reliability tensors remain unchanged:
  - `/src/reliability/subj05_trainrepeat_reliability.pt`
  - `/src/reliability/subj07_trainrepeat_reliability.pt`
- Voxel/ROI counts remain: subj05 `13039` total, early/higher `3661/9378`; subj07 `12682` total, early/higher `3251/9431`.
- Assigned dropout probabilities remain:
  - subj05 `reldrop0`: all/early/higher `0.0000/0.0000/0.0000`.
  - subj05 `reldrop_low`: all mean `0.0625`, range `0.0200-0.1000`, early/higher mean `0.0570/0.0647`, reliability/probability corr `-1.000`.
  - subj05 `reldrop_mid`: all mean `0.1144`, range `0.0400-0.1800`, early/higher mean `0.1047/0.1182`, reliability/probability corr about `-1.000`.
  - subj07 `reldrop0`: all/early/higher `0.0000/0.0000/0.0000`.
  - subj07 `reldrop_low`: all mean `0.0648`, range `0.0200-0.1000`, early/higher mean `0.0561/0.0678`, reliability/probability corr `-1.000`.
  - subj07 `reldrop_mid`: all mean `0.1184`, range `0.0400-0.1800`, early/higher mean `0.1031/0.1236`, reliability/probability corr about `-1.000`.

Metrics/readout status:
- No Cycle 22 final training diagnostics, refined metric CSVs, deltas, or sensitivity diagnostic are available because the full training array is still pending.
- The mechanism diagnostic was not run, per plan, because the required full checkpoints do not yet exist.

Conclusion:
- Operational completion is still pending scheduler allocation, not blocked by a code or artifact failure detected in Cycle 22.
- The next valid action is to inspect `8708124_[0-5]` after launch, parse train logs for final train/test diagnostics and realized dropout rates, then allow dependency evaluator `8708125_[0-5]` to run. If any checkpoint completes but its CSV is missing, rerun only the fixed evaluator path for that exact model name.

Recommended next research questions:
- Do all six `8708124` training tasks start on the scheduled/non-excluded nodes and complete within `04:30:00`?
- Do the full rows preserve the smoke behavior: `reldrop0` realized rates exactly `0.0000/0.0000/0.0000` and nonzero rows realized rates close to assigned all/early/higher probabilities?
- Once evaluator `8708125_[0-5]` writes CSVs, does either nonzero strength improve same-subject BrainRet by about `0.02` absolute without degrading CLIP, Inception, retrieval, EfficientNet/SwAV distances, visual cortex, or higher visual correlation?

## Cycle 23 - 2026-05-24

Plan executed:
- Read `/plan.md` and executed only the Cycle 23 recovery/readout plan.
- Telegram report is not due.
- No code, reliability tensor, dropout strength, split, mask, initialization, evaluator setting, candidate pool, model name, or Slurm resource request was changed.

Scheduler state recovered at `2026-05-24 15:19:22 EDT`:
- Required training array `8708124_[0-5]` is still the first source of truth and remains `PENDING (Priority)`, elapsed `00:00:00`, exit `0:0`, no allocated node and no MaxRSS yet.
- `8708124_0`: `cycle21_subj05_reldrop0_1sess_150ep`, pending, scheduled node `della-l04g16`, scheduled start `2026-05-24T15:46:51`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_0.out`, stderr `.err`.
- `8708124_1`: `cycle21_subj05_reldrop_low_1sess_150ep`, pending, scheduled node `della-l04g12`, scheduled start `2026-05-24T15:47:28`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_1.out`, stderr `.err`.
- `8708124_2`: `cycle21_subj05_reldrop_mid_1sess_150ep`, pending, scheduled node `della-i14g18`, scheduled start `2026-05-24T15:48:28`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_2.out`, stderr `.err`.
- `8708124_3`: `cycle21_subj07_reldrop0_1sess_150ep`, pending, no scheduled node/start shown, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_3.out`, stderr `.err`.
- `8708124_4`: `cycle21_subj07_reldrop_low_1sess_150ep`, pending, no scheduled node/start shown, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_4.out`, stderr `.err`.
- `8708124_5`: `cycle21_subj07_reldrop_mid_1sess_150ep`, pending, no scheduled node/start shown, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_5.out`, stderr `.err`.
- Training array settings remain as intended by the prior smoke gate: `gpu-short`, one A100, `64G`, `04:30:00`, workdir `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`, command `/src/cycle21_reldrop_train_s57.slurm`, `--exclude=della-i14g8,della-i14g19,della-i14g20`, and no requeue.
- Dependent evaluator array `8708125_[0-5]` remains `PENDING (Dependency)`, dependency `afterok:8708124_*`, elapsed `00:00:00`, exit `0:0`, no node, no MaxRSS, time limit `04:00:00`, memory `64G`, command `/src/cycle21_reldrop_eval_s57.slurm`.
- Scheduler/artifact observability remains partial from the container: `/scratch/gpfs/...` is not mounted locally (`/scratch` is absent), so compute-visible paths cannot be directly inspected until artifacts are mirrored or visible under `/src`. `squeue`, `sacct`, and `scontrol` are available and were used.

Artifact state:
- No full-row Slurm logs are visible yet under `/src/slurms` for `c21_reldrop_s57_8708124*` or `c21_reldrop_eval_s57_8708125*`, consistent with the jobs not having started.
- None of the six required full-row checkpoints are visible at `/src/train_logs/<model_name>/last.pth`.
- No enhanced reconstruction directories or tensors are visible under `/src/evals/<model_name>/` for the six required full rows.
- No final CSVs exist at `/src/tables/<model_name>_all_enhancedrecons.csv`.
- Therefore no Cycle 21 full-row final evaluator has run yet, and no row can yet be confirmed as using enhanced reconstructions.

Reliability/dropout provenance:
- Reliability tensors remain the planned unchanged files:
  - `/src/reliability/subj05_trainrepeat_reliability.pt`
  - `/src/reliability/subj07_trainrepeat_reliability.pt`
- Voxel/ROI counts remain: subj05 `13039` total, early/higher `3661/9378`; subj07 `12682` total, early/higher `3251/9431`.
- Assigned dropout probability summaries remain those established before launch:
  - subj05 `reldrop0`: all/early/higher `0.0000/0.0000/0.0000`.
  - subj05 `reldrop_low`: all mean `0.0625`, range `0.0200-0.1000`, early/higher mean `0.0570/0.0647`, reliability/probability corr `-1.000`.
  - subj05 `reldrop_mid`: all mean `0.1144`, range `0.0400-0.1800`, early/higher mean `0.1047/0.1182`, reliability/probability corr about `-1.000`.
  - subj07 `reldrop0`: all/early/higher `0.0000/0.0000/0.0000`.
  - subj07 `reldrop_low`: all mean `0.0648`, range `0.0200-0.1000`, early/higher mean `0.0561/0.0678`, reliability/probability corr `-1.000`.
  - subj07 `reldrop_mid`: all mean `0.1184`, range `0.0400-0.1800`, early/higher mean `0.1031/0.1236`, reliability/probability corr about `-1.000`.

Metrics/readout status:
- No Cycle 23 final training diagnostics, realized full-run dropout rates, refined metric CSVs, deltas versus same-subject `reldrop0`, or mechanism sensitivity diagnostics are available because the required training array remains pending.
- The post-readout mechanism diagnostic was not run, per plan, because the six full checkpoints do not yet exist.
- No recovery rerun was submitted. The exact pending training array is healthy scheduler-side and remains the valid path; duplicating it would risk conflicting rows.

Conclusion:
- Cycle 23 is scheduler-limited, not code-limited or artifact-failure-limited. The six required `reldrop` rows are still queued with the intended semantics and dependency chain intact.
- The next valid action is to inspect `8708124_[0-5]` after allocation, parse the resulting train logs for final train/test diagnostics and realized all/early/higher drop rates, then let `8708125_[0-5]` run. If any checkpoint exists but its CSV is missing, rerun only the fixed evaluator path for that exact model name.

Recommended next research questions:
- Do the three scheduled subject 5 rows actually start on `della-l04g16`, `della-l04g12`, and `della-i14g18`, and do they complete within `04:30:00` without the prior low-headroom OOM pattern?
- Do all six full rows reproduce the smoke behavior: `reldrop0` realized rates exactly `0.0000/0.0000/0.0000`, and nonzero rows realized rates close to assigned probabilities?
- Once `8708125_[0-5]` writes final CSVs, does either nonzero `reldrop` strength improve same-subject BrainRet by about `0.02` absolute while preserving semantic retrieval, CLIP/Inception, distances, visual cortex, and higher visual correlation?

## Cycle 24 - 2026-05-24

Plan executed:
- Read `/plan.md` and executed only the Cycle 24 recovery/readout plan.
- Telegram report is not due.
- No code, reliability tensor, dropout strength, split, mask, initialization, evaluator setting, candidate pool, model name, or Slurm resource request was changed.

Scheduler state recovered at `2026-05-24 15:33:00 EDT`:
- Required training array `8708124_[0-5]` is still the first source of truth and remains `PENDING (Priority)`, elapsed `00:00:00`, exit `0:0`, no allocated node and no MaxRSS yet.
- `8708124_0`: `cycle21_subj05_reldrop0_1sess_150ep`, pending, scheduled node `della-i14g18`, scheduled start `2026-05-24T15:48:28`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_0.out`, stderr `.err`.
- `8708124_1`: `cycle21_subj05_reldrop_low_1sess_150ep`, pending, scheduled node `della-l05g7`, scheduled start `2026-05-24T16:54:00`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_1.out`, stderr `.err`.
- `8708124_2`: `cycle21_subj05_reldrop_mid_1sess_150ep`, pending, scheduled node `della-l04g6`, scheduled start `2026-05-24T16:54:00`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_2.out`, stderr `.err`.
- `8708124_3`: `cycle21_subj07_reldrop0_1sess_150ep`, pending, no scheduled node/start shown, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_3.out`, stderr `.err`.
- `8708124_4`: `cycle21_subj07_reldrop_low_1sess_150ep`, pending, no scheduled node/start shown, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_4.out`, stderr `.err`.
- `8708124_5`: `cycle21_subj07_reldrop_mid_1sess_150ep`, pending, no scheduled node/start shown, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_5.out`, stderr `.err`.
- Training array settings remain intended: `gpu-short`, one A100, `64G`, `04:30:00`, no requeue, workdir `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`, command `/src/cycle21_reldrop_train_s57.slurm`, and `--exclude=della-i14g8,della-i14g19,della-i14g20`.
- Dependent evaluator array `8708125_[0-5]` remains `PENDING (Dependency)`, dependency `afterok:8708124_*`, elapsed `00:00:00`, exit `0:0`, no node, no MaxRSS, time limit `04:00:00`, memory `64G`, command `/src/cycle21_reldrop_eval_s57.slurm`.
- `squeue`, `sacct`, and `scontrol` are available, but filesystem observability remains partial from this container: `/scratch` is not mounted locally, so compute-side `/scratch/gpfs/...` artifacts and pending-job stdout/stderr cannot be inspected directly unless mirrored under `/src`.

Artifact state:
- No full-row Slurm logs are visible under `/src/slurms` for `c21_reldrop_s57_8708124*` or `c21_reldrop_eval_s57_8708125*`, consistent with the jobs not having started.
- None of the six required full-row checkpoints are visible at `/src/train_logs/<model_name>/last.pth`.
- No enhanced reconstruction directories or tensors are visible under `/src/evals/<model_name>/`.
- No final CSVs exist at `/src/tables/<model_name>_all_enhancedrecons.csv`.
- Therefore no Cycle 21 full-row final evaluator has run yet, and no row can yet be confirmed as using enhanced reconstructions.

Reliability/dropout provenance:
- Reliability tensors remain the planned files and were not modified:
  - `/src/reliability/subj05_trainrepeat_reliability.pt`
  - `/src/reliability/subj07_trainrepeat_reliability.pt`
- Voxel/ROI counts remain: subj05 `13039` total, early/higher `3661/9378`; subj07 `12682` total, early/higher `3251/9431`.
- Assigned dropout probability summaries remain unchanged from launch:
  - subj05 `reldrop0`: all/early/higher `0.0000/0.0000/0.0000`.
  - subj05 `reldrop_low`: all mean `0.0625`, range `0.0200-0.1000`, early/higher mean `0.0570/0.0647`, reliability/probability corr `-1.000`.
  - subj05 `reldrop_mid`: all mean `0.1144`, range `0.0400-0.1800`, early/higher mean `0.1047/0.1182`, reliability/probability corr about `-1.000`.
  - subj07 `reldrop0`: all/early/higher `0.0000/0.0000/0.0000`.
  - subj07 `reldrop_low`: all mean `0.0648`, range `0.0200-0.1000`, early/higher mean `0.0561/0.0678`, reliability/probability corr `-1.000`.
  - subj07 `reldrop_mid`: all mean `0.1184`, range `0.0400-0.1800`, early/higher mean `0.1031/0.1236`, reliability/probability corr about `-1.000`.

Metrics/readout status:
- No Cycle 24 final training diagnostics, realized full-run dropout rates, refined metric CSVs, same-subject deltas, or mechanism sensitivity diagnostics are available because the required training array remains pending.
- The post-readout mechanism diagnostic was not run, per plan, because the six full checkpoints do not yet exist.
- No recovery rerun or new experiment was submitted. The exact pending training array remains the valid path; duplicating it would risk conflicting rows.

Conclusion:
- Cycle 24 remains scheduler-limited, not code-limited or artifact-failure-limited. The six required `reldrop` rows are still queued with the intended semantics and dependency chain intact.
- The next valid action is to inspect `8708124_[0-5]` after allocation, parse the resulting train logs for final train/test diagnostics and realized all/early/higher drop rates, then let `8708125_[0-5]` run. If any checkpoint exists but its CSV is missing, rerun only the fixed evaluator path for that exact model name.

Recommended next research questions:
- Do the scheduled subject 5 rows start on the currently reserved nodes and complete within `04:30:00` without the prior low-headroom OOM pattern?
- Do all six full rows reproduce the smoke behavior: `reldrop0` realized rates exactly `0.0000/0.0000/0.0000`, and nonzero rows realized rates close to assigned probabilities?
- Once `8708125_[0-5]` writes final CSVs, does either nonzero `reldrop` strength improve same-subject BrainRet by about `0.02` absolute while preserving semantic retrieval, CLIP/Inception, distances, visual cortex, and higher visual correlation?

## Cycle 25 - 2026-05-24

Plan executed:
- Read `/plan.md` and executed only the Cycle 25 recovery/readout plan.
- Telegram report is not due.
- No code, reliability tensor, dropout strength, split, mask, initialization, evaluator setting, candidate pool, model name, Slurm time, or Slurm memory request was changed.

Scheduler state recovered at `2026-05-24 15:48:31 EDT`:
- Required training array `8708124_[0-5]` remains the first source of truth and is still healthy `PENDING (Priority)`. All training tasks have elapsed `00:00:00`, exit code `0:0`, no MaxRSS, and no allocated node yet.
- `8708124_0`: `cycle21_subj05_reldrop0_1sess_150ep`, pending, scheduled node `della-l03g3`, scheduled start `2026-05-24T19:21:51`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_0.out`, stderr `.err`.
- `8708124_1`: `cycle21_subj05_reldrop_low_1sess_150ep`, pending, scheduled node `della-l03g12`, scheduled start `2026-05-24T19:23:13`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_1.out`, stderr `.err`.
- `8708124_2`: `cycle21_subj05_reldrop_mid_1sess_150ep`, pending, scheduled node `della-l02g11`, scheduled start `2026-05-24T19:24:43`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_2.out`, stderr `.err`.
- `8708124_3`: `cycle21_subj07_reldrop0_1sess_150ep`, pending, no scheduled node/start shown, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_3.out`, stderr `.err`.
- `8708124_4`: `cycle21_subj07_reldrop_low_1sess_150ep`, pending, no scheduled node/start shown, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_4.out`, stderr `.err`.
- `8708124_5`: `cycle21_subj07_reldrop_mid_1sess_150ep`, pending, no scheduled node/start shown, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_s57_8708124_5.out`, stderr `.err`.
- Training array settings remain intended: `gpu-short`, one A100, `64G`, `04:30:00`, no requeue, workdir `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`, command `/src/cycle21_reldrop_train_s57.slurm`, and exclude list `della-i14g[8,19-20]`.
- Dependent evaluator array `8708125_[0-5]` remains `PENDING (Dependency)` with `Dependency=afterok:8708124_*(unfulfilled)`. All evaluator tasks have elapsed `00:00:00`, exit code `0:0`, no MaxRSS, no allocated node, time limit `04:00:00`, memory `64G`, workdir `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src`, command `/src/cycle21_reldrop_eval_s57.slurm`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c21_reldrop_eval_s57_8708125_<task>.out`, stderr `.err`.
- `squeue`, `sacct`, and `scontrol` are available. Filesystem observability is still partial from this container because `/scratch` is not mounted locally, so compute-side pending-job artifacts cannot be inspected directly until visible under `/src` or the jobs write mirrored artifacts.

Artifact state:
- No full-row Slurm logs are visible under `/src/slurms` for `c21_reldrop_s57_8708124*` or `c21_reldrop_eval_s57_8708125*`, consistent with the jobs not having started.
- None of the six required full-row checkpoints are visible at `/src/train_logs/<model_name>/last.pth`.
- No enhanced reconstruction tensors or directories are visible under `/src/evals/<model_name>/` for the six required rows.
- No final CSVs exist at `/src/tables/<model_name>_all_enhancedrecons.csv`.
- Therefore no completed checkpoint can yet be parsed, no evaluator row has used enhanced reconstructions, and no same-subject deltas can be computed.

Recovery/rerun actions:
- No recovery job was submitted. The original `8708124_[0-5]` training array is still pending and recoverable scheduler-side, and launching duplicates would violate the plan.
- No evaluator rerun was submitted. The original `8708125_[0-5]` dependency is correctly waiting for all six training tasks.

Training diagnostics and dropout rates:
- No new full-run final train/test loss, blurry PixCorr, forward retrieval, backward retrieval, or realized all/early/higher dropout rates are available because all six training tasks are still pending.
- Smoke diagnostics remain the only executed Cycle 21 reliability-dropout diagnostics: `reldrop0` realized all/early/higher drop rates `0.0000/0.0000/0.0000`, and `reldrop_low` realized `0.0646/0.0562/0.0675` for subj07 in the 3-epoch smoke.

Metric table and sensitivity diagnostic:
- Final refined CSV metric tables are unavailable because no full-row checkpoints/evaluations exist yet.
- The mechanism sensitivity diagnostic was not run, per plan, because all six CSVs do not exist and no rows have been declared concretely unrecoverable.

Decision:
- Reliability dropout is operationally incomplete, scheduler-limited, and still queued through the exact original arrays. It has not passed, failed scientifically, or become unrecoverable.

Recommended next research questions:
- Do the scheduled subject 5 tasks start at the newly projected evening times and complete within `04:30:00`?
- Once training starts, do full-run realized dropout rates match the smoke behavior and assigned all/early/higher probabilities?
- If the evaluator dependency runs, do all six final CSVs appear at the required paths and show any same-subject BrainRet gain of about `+0.02` without semantic or higher-visual regression?

## Cycle 26 - 2026-05-25

Plan source: read and executed `/plan.md` only. Telegram report is not due.

Code/config changes:
- None. This was a recovery/readout cycle for the exact Cycle 21 reliability-dropout rows; no mechanism, grid, evaluator, generator/refiner, or row-name changes were made.

Scheduler and artifact state:
- Training array `8708124_[0-5]` completed successfully for all six required rows. `sacct` showed `COMPLETED`, exit `0:0`.
  - `8708124_0` `cycle21_subj05_reldrop0_1sess_150ep`: elapsed `02:08:23`, node `della-l04g5`, MaxRSS `22974988K`, logs `/src/slurms/c21_reldrop_s57_8708124_0.out/.err`.
  - `8708124_1` `cycle21_subj05_reldrop_low_1sess_150ep`: elapsed `02:08:22`, node `della-l03g1`, MaxRSS `22973724K`, logs `/src/slurms/c21_reldrop_s57_8708124_1.out/.err`.
  - `8708124_2` `cycle21_subj05_reldrop_mid_1sess_150ep`: elapsed `02:07:39`, node `della-l04g12`, MaxRSS `21953240K`, logs `/src/slurms/c21_reldrop_s57_8708124_2.out/.err`.
  - `8708124_3` `cycle21_subj07_reldrop0_1sess_150ep`: elapsed `02:07:36`, node `della-l02g6`, MaxRSS `22958540K`, logs `/src/slurms/c21_reldrop_s57_8708124_3.out/.err`.
  - `8708124_4` `cycle21_subj07_reldrop_low_1sess_150ep`: elapsed `02:04:29`, node `della-l05g7`, MaxRSS `22947608K`, logs `/src/slurms/c21_reldrop_s57_8708124_4.out/.err`.
  - `8708124_5` `cycle21_subj07_reldrop_mid_1sess_150ep`: elapsed `02:07:52`, node `della-l02g9`, MaxRSS `22953796K`, logs `/src/slurms/c21_reldrop_s57_8708124_5.out/.err`.
- Checkpoint visibility limitation: `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2` is not mounted in the current container, and `/src/train_logs/cycle21_*` does not show the completed `last.pth` files. All six dependent evaluator tasks passed their checkpoint preflight and are running, so the checkpoints are compute-visible; local absence is an observability limitation, not concrete failure.
- Evaluator array `8708125_[0-5]` is running through the required path `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py`.
  - At readout all tasks were `RUNNING`, exit `0:0`, time limit `04:00:00`, memory `64G`.
  - Nodes/elapsed: task 0 `della-l05g1` `01:35:35`; task 1 `della-i14g4` `01:34:34`; task 2 `della-i14g6` `01:34:04`; task 3 `della-l04g6` `01:34:04`; task 4 `della-i14g8` `01:15:49`; task 5 `della-l04g8` `01:15:49`.
  - No `/src/tables/*cycle21*reldrop*csv` existed yet. No enhanced recon tensors were visible yet; task 0 had written recon intermediates under `/src/evals/cycle21_subj05_reldrop0_1sess_150ep/`.

Recovery/rerun commands launched:
- None. The original training array completed and the original dependent evaluator array is healthy/running; duplicate jobs were not submitted.

Assigned dropout probability summaries:
- subj05 `reldrop0`: all/early/higher `0.0000/0.0000/0.0000`, range `0.0000-0.0000`, corr `nan`.
- subj05 `reldrop_low`: mean `0.0625`, range `0.0200-0.1000`, early/higher `0.0570/0.0647`, corr `-1.000`.
- subj05 `reldrop_mid`: mean `0.1144`, range `0.0400-0.1800`, early/higher `0.1047/0.1182`, corr about `-1.000`.
- subj07 `reldrop0`: all/early/higher `0.0000/0.0000/0.0000`, range `0.0000-0.0000`, corr `nan`.
- subj07 `reldrop_low`: mean `0.0648`, range `0.0200-0.1000`, early/higher `0.0561/0.0678`, corr about `-1.000`.
- subj07 `reldrop_mid`: mean `0.1184`, range `0.0400-0.1800`, early/higher `0.1031/0.1236`, corr `-1.000`.

Final training diagnostics and realized dropout rates:
- subj05 `reldrop0`: test loss `16.6`, blurry PixCorr `0.193`, test fwd/bwd `0.663/0.560`; train loss `5.60`, train blurry PixCorr `0.802`, train fwd/bwd `1.000/1.000`; realized drops `0.0000/0.0000/0.0000`.
- subj05 `reldrop_low`: test loss `16.7`, blurry PixCorr `0.193`, test fwd/bwd `0.667/0.583`; train loss `5.61`, train blurry PixCorr `0.798`, train fwd/bwd `1.000/1.000`; realized drops `0.0626/0.0571/0.0648`.
- subj05 `reldrop_mid`: test loss `16.6`, blurry PixCorr `0.208`, test fwd/bwd `0.670/0.553`; train loss `5.64`, train blurry PixCorr `0.797`, train fwd/bwd `1.000/1.000`; realized drops `0.1150/0.1050/0.1180`.
- subj07 `reldrop0`: test loss `16.7`, blurry PixCorr `0.233`, test fwd/bwd `0.733/0.573`; train loss `5.68`, train blurry PixCorr `0.788`, train fwd/bwd `1.000/1.000`; realized drops `0.0000/0.0000/0.0000`.
- subj07 `reldrop_low`: test loss `16.3`, blurry PixCorr `0.214`, test fwd/bwd `0.723/0.573`; train loss `5.70`, train blurry PixCorr `0.786`, train fwd/bwd `1.000/1.000`; realized drops `0.0649/0.0560/0.0680`.
- subj07 `reldrop_mid`: test loss `16.5`, blurry PixCorr `0.234`, test fwd/bwd `0.710/0.577`; train loss `5.71`, train blurry PixCorr `0.783`, train fwd/bwd `1.000/1.000`; realized drops `0.1190/0.1030/0.1240`.

Final CSV metric table and deltas:
- Not available yet. The evaluator array is still running and no final CSVs existed at readout.

Sensitivity diagnostic:
- Not run. Per plan, this waits until all six CSVs exist or missing rows are concretely unrecoverable.

Decision:
- Operationally incomplete. The six trainings completed and the fixed evaluator is running, but refined metrics and same-subject deltas are not yet available. No scientific pass/fail decision can be made yet.

Recommended next research questions:
- Did `8708125_[0-5]` finish within `04:00:00` and write all six enhanced recon tensors plus final CSVs?
- Once CSVs exist, do `reldrop_low` or `reldrop_mid` improve same-subject refined BrainRet by about `+0.02` for subjects 5 and 7 without semantic or higher-visual regressions?
- After CSV completion, does the sensitivity diagnostic show reliability dropout shifting learned sensitivity in a way that aligns with BrainRet and higher-visual outcomes?
