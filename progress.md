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

## Cycle 27 - 2026-05-25

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- No mechanism, dropout strength, generator/refiner, evaluator setting, row name, split, mask, initialization, batch size, hidden dim, epoch count, blurry branch, diffusion prior, or retrieval candidate pool was changed.

Code/config changes:
- Added `/src/cycle27_reldrop_sensitivity.py` and `/src/cycle27_reldrop_sensitivity.slurm` only for the post-CSV sensitivity diagnostic required by the plan.
- Corrected the diagnostic Spearman helper to handle tied and constant vectors; `reldrop0` assigned dropout probability is constant zero, so its probability/sensitivity Spearman is `nan` rather than an arbitrary tied-rank value.

Evaluator scheduler state for `8708125_[0-5]`:
- `8708125_0` `cycle21_subj05_reldrop0_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l05g1`, elapsed `02:25:01`, MaxRSS `49548972K`, stdout `/src/slurms/c21_reldrop_eval_s57_8708125_0.out`, stderr `/src/slurms/c21_reldrop_eval_s57_8708125_0.err`.
- `8708125_1` `cycle21_subj05_reldrop_low_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-i14g4`, elapsed `02:26:34`, MaxRSS `49489324K`, stdout `/src/slurms/c21_reldrop_eval_s57_8708125_1.out`, stderr `/src/slurms/c21_reldrop_eval_s57_8708125_1.err`.
- `8708125_2` `cycle21_subj05_reldrop_mid_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-i14g6`, elapsed `02:24:26`, MaxRSS `49440632K`, stdout `/src/slurms/c21_reldrop_eval_s57_8708125_2.out`, stderr `/src/slurms/c21_reldrop_eval_s57_8708125_2.err`.
- `8708125_3` `cycle21_subj07_reldrop0_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l04g6`, elapsed `02:24:14`, MaxRSS `49487324K`, stdout `/src/slurms/c21_reldrop_eval_s57_8708125_3.out`, stderr `/src/slurms/c21_reldrop_eval_s57_8708125_3.err`.
- `8708125_4` `cycle21_subj07_reldrop_low_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-i14g8`, elapsed `02:24:39`, MaxRSS `49399368K`, stdout `/src/slurms/c21_reldrop_eval_s57_8708125_4.out`, stderr `/src/slurms/c21_reldrop_eval_s57_8708125_4.err`.
- `8708125_5` `cycle21_subj07_reldrop_mid_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l04g8`, elapsed `02:22:32`, MaxRSS `49576608K`, stdout `/src/slurms/c21_reldrop_eval_s57_8708125_5.out`, stderr `/src/slurms/c21_reldrop_eval_s57_8708125_5.err`.
- `squeue` returned `slurm_load_jobs error: Invalid job id specified` after all tasks left the queue; `sacct` remained available and gave the completed states above.

Artifact state:
- Checkpoints are compute-visible at `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/<model_name>/last.pth`; local `/src/train_logs/cycle21_*` checkpoint paths remain absent because the current container does not mount the same `/scratch/gpfs` tree. The sensitivity job loaded all six compute-visible checkpoints successfully.
- Enhanced tensors exist and are 751M each:
  - `/src/evals/cycle21_subj05_reldrop0_1sess_150ep/cycle21_subj05_reldrop0_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle21_subj05_reldrop_low_1sess_150ep/cycle21_subj05_reldrop_low_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle21_subj05_reldrop_mid_1sess_150ep/cycle21_subj05_reldrop_mid_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle21_subj07_reldrop0_1sess_150ep/cycle21_subj07_reldrop0_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle21_subj07_reldrop_low_1sess_150ep/cycle21_subj07_reldrop_low_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle21_subj07_reldrop_mid_1sess_150ep/cycle21_subj07_reldrop_mid_1sess_150ep_all_enhancedrecons.pt`
- Final CSVs exist:
  - `/src/tables/cycle21_subj05_reldrop0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle21_subj05_reldrop_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle21_subj05_reldrop_mid_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle21_subj07_reldrop0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle21_subj07_reldrop_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle21_subj07_reldrop_mid_1sess_150ep_all_enhancedrecons.csv`
- The evaluator logs confirm `final_evaluations.py` consumed `evals/<model_name>/<model_name>_all_enhancedrecons.pt` for every row.

Recovery/rerun commands launched:
- No evaluator recovery reruns were submitted. The original dependent evaluator array completed cleanly.
- Sensitivity diagnostic commands:
  - `sbatch /src/cycle27_reldrop_sensitivity.slurm` -> job `8727693`, completed but superseded after fixing tied-rank handling.
  - `sbatch /src/cycle27_reldrop_sensitivity.slurm` -> job `8727737`, `COMPLETED`, exit `0:0`, node `della-i13n25`, elapsed `00:01:31`, MaxRSS `34946372K`, output `/src/tables/cycle27_reldrop_sensitivity.csv`.

Full refined metric table:

| subject | row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist | SwAV dist | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | reldrop0 | 0.194953 | 0.409673 | 0.839638 | 0.917622 | 0.865263 | 0.843651 | 0.768107 | 0.432059 | 0.664222 | 0.563556 | 0.410970 | 0.341505 | 0.349198 | 0.333538 | 0.313308 | 0.420221 |
| subj05 | reldrop_low | 0.193842 | 0.404299 | 0.843798 | 0.916389 | 0.856537 | 0.845423 | 0.768270 | 0.432230 | 0.649556 | 0.556111 | 0.416306 | 0.353121 | 0.355846 | 0.335339 | 0.315220 | 0.423606 |
| subj05 | reldrop_mid | 0.189794 | 0.410379 | 0.837471 | 0.915836 | 0.858889 | 0.848361 | 0.765696 | 0.433695 | 0.660556 | 0.563889 | 0.412196 | 0.340201 | 0.346989 | 0.331149 | 0.313066 | 0.422663 |
| subj07 | reldrop0 | 0.192812 | 0.402312 | 0.825233 | 0.884783 | 0.779239 | 0.781335 | 0.835155 | 0.478052 | 0.685000 | 0.535333 | 0.317103 | 0.320393 | 0.323790 | 0.313454 | 0.276887 | 0.301993 |
| subj07 | reldrop_low | 0.185776 | 0.404134 | 0.824459 | 0.880209 | 0.778906 | 0.783524 | 0.829921 | 0.477394 | 0.664778 | 0.557222 | 0.316120 | 0.316478 | 0.319120 | 0.313092 | 0.280742 | 0.301013 |
| subj07 | reldrop_mid | 0.187893 | 0.414382 | 0.817423 | 0.877122 | 0.782041 | 0.772345 | 0.836533 | 0.482801 | 0.686556 | 0.546222 | 0.312270 | 0.315216 | 0.318102 | 0.313352 | 0.274388 | 0.298637 |

Same-subject deltas versus `reldrop0`:
- subj05 `reldrop_low`: PixCorr `-0.001111`, SSIM `-0.005374`, AlexNet-2 `+0.004160`, AlexNet-5 `-0.001232`, Inception `-0.008727`, CLIP `+0.001773`, EffNet dist `+0.000162`, SwAV dist `+0.000171`, ImageRet `-0.014667`, BrainRet `-0.007444`, VC `+0.005336`, V1 `+0.011615`, V2 `+0.006647`, V3 `+0.001801`, V4 `+0.001912`, HigherVis `+0.003385`.
- subj05 `reldrop_mid`: PixCorr `-0.005159`, SSIM `+0.000706`, AlexNet-2 `-0.002166`, AlexNet-5 `-0.001786`, Inception `-0.006374`, CLIP `+0.004711`, EffNet dist `-0.002412`, SwAV dist `+0.001636`, ImageRet `-0.003667`, BrainRet `+0.000333`, VC `+0.001226`, V1 `-0.001305`, V2 `-0.002209`, V3 `-0.002389`, V4 `-0.000242`, HigherVis `+0.002442`.
- subj07 `reldrop_low`: PixCorr `-0.007037`, SSIM `+0.001822`, AlexNet-2 `-0.000774`, AlexNet-5 `-0.004574`, Inception `-0.000333`, CLIP `+0.002188`, EffNet dist `-0.005233`, SwAV dist `-0.000657`, ImageRet `-0.020222`, BrainRet `+0.021889`, VC `-0.000983`, V1 `-0.003915`, V2 `-0.004669`, V3 `-0.000362`, V4 `+0.003855`, HigherVis `-0.000980`.
- subj07 `reldrop_mid`: PixCorr `-0.004919`, SSIM `+0.012070`, AlexNet-2 `-0.007810`, AlexNet-5 `-0.007661`, Inception `+0.002802`, CLIP `-0.008990`, EffNet dist `+0.001379`, SwAV dist `+0.004749`, ImageRet `+0.001556`, BrainRet `+0.010889`, VC `-0.004833`, V1 `-0.005177`, V2 `-0.005687`, V3 `-0.000102`, V4 `-0.002499`, HigherVis `-0.003355`.

Sensitivity diagnostic:
- Results saved to `/src/tables/cycle27_reldrop_sensitivity.csv`.
- subj05 reliability/sensitivity Spearman:
  - `reldrop0`: all `-0.0338`, early `+0.2857`, higher `-0.1753`; dropout-probability/sensitivity `nan/nan/nan` because probability is constant zero.
  - `reldrop_low`: all `-0.0294`, early `+0.2841`, higher `-0.1678`; dropout-probability/sensitivity all `+0.0294`, early `-0.2841`, higher `+0.1678`.
  - `reldrop_mid`: all `-0.0089`, early `+0.2973`, higher `-0.1475`; dropout-probability/sensitivity all `+0.0089`, early `-0.2973`, higher `+0.1475`.
- subj07 reliability/sensitivity Spearman:
  - `reldrop0`: all `+0.1893`, early `+0.4554`, higher `+0.0314`; dropout-probability/sensitivity `nan/nan/nan` because probability is constant zero.
  - `reldrop_low`: all `+0.1965`, early `+0.4667`, higher `+0.0347`; dropout-probability/sensitivity all `-0.1965`, early `-0.4667`, higher `-0.0347`.
  - `reldrop_mid`: all `+0.2090`, early `+0.4771`, higher `+0.0445`; dropout-probability/sensitivity all `-0.2090`, early `-0.4771`, higher `-0.0445`.
- Interpretation: dropout does not create a consistent useful sensitivity shift that matches protected semantic gains. Subject 5 shows no BrainRet gain despite small higher-visual correlation increases. Subject 7 `reldrop_low` gains BrainRet but loses image retrieval by `-0.0202`, has lower PixCorr, and is not replicated in subject 5.

Decision:
- Reliability dropout is closed as neutral/harmful for this MindEyeV2 one-session weak-subject setting.
- It does not satisfy the plan's success rule. Subject 5 has no nonzero BrainRet improvement; subject 7 `reldrop_low` reaches `+0.0219` BrainRet but fails the preservation requirement because image retrieval materially worsens and the effect is not credible subject-wise across both weak subjects. Subject 7 `reldrop_mid` is only `+0.0109` BrainRet with CLIP, distances, VC, and higher-visual degradation.
- Per the plan, do not try stronger centered scaling, hard reliability pruning, broader dropout/noise grids, ROI-balanced evaluation-time scaling, generator/refiner edits, caption/VLM correction, temporal decoding, CLIP-layer fusion, broad ROI routing, cross-subject routers, or MoE architectures as the next step.

Recommended next research questions:
- If continuing after dropout closure, the defensible fallback is one small subject-functional-alignment adapter with same-code `adapter0` control and one conservative trainable setting on subjects 5 and 7, using the unchanged one-session/evaluator protocol and the same semantic-protected success rule.

## Cycle 28 - 2026-05-25 Opening Provenance

Plan source: read `/plan.md` and executing only the Cycle 28 zero-controlled subject-adapter plan. Telegram report is due this cycle.

Required reliability-dropout closure note before implementation:
- Cycle 21 reliability dropout is closed as neutral/harmful for the MindEyeV2 one-session weak-subject setting.
- All six Cycle 21 rows trained and evaluated successfully with exit `0:0`: `cycle21_subj05_reldrop{0,_low,_mid}_1sess_150ep` and `cycle21_subj07_reldrop{0,_low,_mid}_1sess_150ep`.
- Enhanced reconstruction tensors and final CSVs exist for all six rows under `/src/evals/<model_name>/<model_name>_all_enhancedrecons.pt` and `/src/tables/<model_name>_all_enhancedrecons.csv`.
- Evaluator logs confirm `final_evaluations.py` consumed each `*_all_enhancedrecons.pt` tensor.
- Sensitivity diagnostic exists at `/src/tables/cycle27_reldrop_sensitivity.csv`.
- Scientific decision: nonzero dropout did not pass. Subject 5 had no BrainRet gain (`reldrop_low -0.007444`, `reldrop_mid +0.000333`), and subject 7 `reldrop_low` traded BrainRet gain (`+0.021889`) for ImageRet/PixCorr/ROI-correlation degradation, while `reldrop_mid` was below the target BrainRet gain and degraded protected metrics.

## Cycle 28 - 2026-05-25

Plan source:
- Read and executed `/plan.md` only. Telegram report is due.
- Cycle 21 reliability dropout was explicitly closed before implementation; see the Cycle 28 opening provenance note above.

Code/config changes:
- Added `SubjectResidualAdapter` in `/src/models.py` at the model utility layer before `BrainNetwork`. Form: `x + gamma * W_up(GELU(W_down(LayerNorm(x))))`; `W_up.weight` and `W_up.bias` are zero-initialized, so enabled adapters start as exact no-ops.
- Added disabled-by-default adapter CLI flags in `/src/Train.py`: `--use_subject_adapter`, `--subject_adapter_dim`, `--freeze_subject_adapter`. Default behavior leaves the model graph unchanged and adds no adapter checkpoint keys.
- Inserted the adapter at the required point: immediately after subject-specific `model.ridge(...)` output is concatenated into `voxel_ridge`, and immediately before the shared `model.backbone(voxel_ridge)` call in both training and test paths.
- Added the same adapter flags and insertion point to `/src/recon_inference.py` so checkpoints from adapter rows can be loaded and evaluated with the validated enhanced path.
- Added Slurm scripts: `/src/cycle28_adapter_smoke.slurm`, `/src/cycle28_adapter_train_s57.slurm`, and `/src/cycle28_adapter_eval_s57.slurm`.
- After the first smoke landed on 40 GB A100 nodes and OOMed, constrained the Cycle 28 smoke/train/eval scripts to A100 `gpu80` nodes. Batch size, hidden dim, one-session split, initialization, refiner, evaluator, and model names were not changed.

Sanity checks:
- `/src/fmri/bin/python -m py_compile /src/models.py /src/Train.py /src/recon_inference.py /src/enhanced_recon_inference.py /src/final_evaluations.py` passed.
- `bash -n /src/cycle28_adapter_smoke.slurm /src/cycle28_adapter_train_s57.slurm /src/cycle28_adapter_eval_s57.slurm` passed.
- Lightweight tensor sanity check passed: disabled backbone shape path returned `(2,4,16)`/`(2,4,16)` on a reduced test model; full-size adapter init had `adapter_max_abs_delta=0.0`.
- Full-size adapter parameter count at `hidden_dim=4096`, `subject_adapter_dim=128`: `1,060,993` params. `adapter_low` trainable params: `1,060,993`; frozen `adapter0` trainable params after `requires_grad_(False)`: `0`.
- Smoke logs also confirmed adapter initialization on the real graph: `Subject adapter init max_abs_delta=0`; `adapter0` trainable `0`, `adapter_low` trainable `1,060,993`.

Commands/jobs launched:
- Initial smoke: `sbatch /src/cycle28_adapter_smoke.slurm` -> job `8727970_[0-1]`, subject 7, 3 epochs, rows `cycle28_smoke_subj07_adapter0_1sess_3ep` and `cycle28_smoke_subj07_adapter_low_1sess_3ep`, one A100, `64G`, `01:00:00`.
- Initial full train dependency: `sbatch --dependency=afterok:8727970 /src/cycle28_adapter_train_s57.slurm` -> job `8727971_[0-3]`; cancelled after the smoke failed before any task started.
- Initial evaluator dependency: `sbatch --dependency=afterok:8727971 /src/cycle28_adapter_eval_s57.slurm` -> job `8727972_[0-3]`; cancelled after the smoke failed before any task started.
- Recovery smoke after constraining to A100 `gpu80`: `sbatch /src/cycle28_adapter_smoke.slurm` -> job `8728063_[0-1]`.
- Recovery full train dependency: `sbatch --dependency=afterok:8728063 /src/cycle28_adapter_train_s57.slurm` -> job `8728064_[0-3]`.
- Recovery evaluator dependency: `sbatch --dependency=afterok:8728064 /src/cycle28_adapter_eval_s57.slurm` -> job `8728065_[0-3]`.

Initial smoke failure details:
- `8727970_0` `cycle28_smoke_subj07_adapter0_1sess_3ep`: `FAILED`, exit `1:0`, node `della-i14g18`, elapsed `00:01:32`, MaxRSS `21588440K`, logs `/src/slurms/c28_adapter_smoke_8727970_0.out/.err`.
- `8727970_1` `cycle28_smoke_subj07_adapter_low_1sess_3ep`: `FAILED`, exit `1:0`, node `della-i14g4`, elapsed `00:01:27`, MaxRSS `21448208K`, logs `/src/slurms/c28_adapter_smoke_8727970_1.out/.err`.
- Failure class: CUDA OOM on 40 GB A100 at first `accelerator.backward(loss)`, attempting a `774 MiB` allocation with only about `596 MiB` free. Both logs reached model build, official multisubject checkpoint load, adapter no-op check, and epoch start before OOM.
- Recovery action: no duplicate full rows were run. Blocked full/eval arrays were cancelled, and only the 1-hour smoke plus dependent exact full/eval rows were relaunched with A100 `gpu80` constraint.

Recovery scheduler state at readout:
- `8728063_0` smoke `adapter0`: `RUNNING`, node `della-l07g7`, elapsed `00:00:03`, time limit `01:00:00`, `64G`, A100 `gpu80`.
- `8728063_1` smoke `adapter_low`: `PENDING`, elapsed `00:00:00`, time limit `01:00:00`, `64G`, A100 `gpu80`.
- `8728064_[0-3]` full rows are `PENDING (Dependency=afterok:8728063_*)`, time limit `04:30:00`, `64G`, A100 `gpu80`.
- `8728065_[0-3]` evaluator rows are `PENDING (Dependency=afterok:8728064_*)`, time limit `04:00:00`, `64G`, A100 `gpu80`.
- Full row model names queued behind the smoke: `cycle28_subj05_adapter0_1sess_150ep`, `cycle28_subj05_adapter_low_1sess_150ep`, `cycle28_subj07_adapter0_1sess_150ep`, `cycle28_subj07_adapter_low_1sess_150ep`.

Metrics/artifact status:
- No Cycle 28 full training diagnostics, checkpoints, enhanced reconstruction tensors, final CSVs, refined metric table, or same-subject `adapter_low - adapter0` deltas exist yet because the first smoke failed and the recovery smoke/full/eval chain is still running/pending.
- Required evaluator path is encoded for every full row as `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py`, with `final_evaluations.py` pointed at `evals/<model_name>/<model_name>_all_enhancedrecons.pt`.

Decision:
- Operationally incomplete. The adapter code and sanity checks passed, but the initial 40 GB smoke exposed a CUDA OOM. Recovery is queued/running on A100 `gpu80` with unchanged MindEye2 protocol and exact requested full row names.

Recommended next research questions:
- Does `8728063_[0-1]` complete on `gpu80` and preserve the adapter no-op/parameter-count checks without OOM?
- If the recovery smoke passes, do the four dependent full rows `8728064_[0-3]` complete within `04:30:00` and produce final train/test loss, blurry PixCorr, forward retrieval, and backward retrieval diagnostics?
- After `8728065_[0-3]` writes enhanced tensors and CSVs, does `adapter_low` improve refined BrainRet by about `+0.02` separately for subjects 5 and 7 while preserving CLIP, Inception, ImageRet, distances, visual-cortex correlation, and HigherVis?

Telegram-ready update:
Cycle 28 implemented the zero-initialized subject residual adapter and queued the exact four planned rows for subjects 5 and 7. The adapter is inserted after the subject ridge and before the shared backbone, is disabled by default, and starts as an exact no-op (`max_abs_delta=0.0`); trainable size is `1,060,993` params at bottleneck dim `128`, while frozen `adapter0` has `0` trainable adapter params. The first 1-hour smoke on 40 GB A100 nodes failed at first backward with CUDA OOM after loading the official checkpoint, so the blocked full/eval arrays were cancelled before running. I relaunched the smoke and dependent full/eval chain constrained to A100 `gpu80`: smoke `8728063_[0-1]`, full train `8728064_[0-3]`, eval `8728065_[0-3]`. At readout, smoke task 0 was running on `della-l07g7`, task 1 was pending, and the full/eval arrays were dependency-gated. No Cycle 28 final metrics or plots exist yet.

## Cycle 29 - 2026-05-28

Plan source:
- Read and executed `/plan.md` only. Telegram report is due.
- No code, model mechanism, split, batch size, hidden dim, epoch count, adapter setting, evaluator path, refiner, or retrieval candidate pool was changed.

Scheduler recovery state:
- `squeue -j 8728063,8728064,8728065` returned no active rows because the chain had left the queue.
- `scontrol show job 8728063 8728064 8728065` and individual `scontrol show job <id>` calls returned `slurm_load_jobs error: Invalid job id specified`, consistent with aged completed jobs no longer being visible to `scontrol`; `sacct` remained the source of truth.
- Smoke `8728063_[0-1]` completed successfully:
  - `8728063_0` `adapter0`: `COMPLETED`, exit `0:0`, node `della-l07g7`, elapsed `00:05:33`, batch MaxRSS `21594120K`, logs `/src/slurms/c28_adapter_smoke_8728063_0.out/.err`.
  - `8728063_1` `adapter_low`: `COMPLETED`, exit `0:0`, node `della-l08g5`, elapsed `00:05:03`, batch MaxRSS `23051112K`, logs `/src/slurms/c28_adapter_smoke_8728063_1.out/.err`.
- Full training `8728064_[0-3]` completed successfully:
  - `8728064_0` `cycle28_subj05_adapter0_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l02g4`, elapsed `02:09:52`, batch MaxRSS `21614440K`.
  - `8728064_1` `cycle28_subj05_adapter_low_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l05g3`, elapsed `02:10:53`, batch MaxRSS `22066964K`.
  - `8728064_2` `cycle28_subj07_adapter0_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l04g6`, elapsed `02:04:20`, batch MaxRSS `22906888K`.
  - `8728064_3` `cycle28_subj07_adapter_low_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l03g1`, elapsed `02:10:06`, batch MaxRSS `21447672K`.
- Evaluation `8728065_[0-3]` completed successfully:
  - `8728065_0` `cycle28_subj05_adapter0_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l04g12`, elapsed `02:27:36`, batch MaxRSS `49241008K`.
  - `8728065_1` `cycle28_subj05_adapter_low_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l02g4`, elapsed `02:26:56`, batch MaxRSS `49197744K`.
  - `8728065_2` `cycle28_subj07_adapter0_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l02g11`, elapsed `02:26:55`, batch MaxRSS `49723620K`.
  - `8728065_3` `cycle28_subj07_adapter_low_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l04g6`, elapsed `02:22:22`, batch MaxRSS `46340584K`.

Smoke confirmation:
- Both recovery smoke rows loaded the official multisubject checkpoint, started epoch 0, reached backward/training progress, and saved checkpoints without OOM on `gpu80`.
- `adapter0`: `Subject adapter enabled: dim=128, params=1060993, trainable=0, freeze=True`; `Subject adapter init max_abs_delta=0`.
- `adapter_low`: `Subject adapter enabled: dim=128, params=1060993, trainable=1060993, freeze=False`; `Subject adapter init max_abs_delta=0`.
- Final 3-epoch smoke diagnostics:
  - `adapter0`: test loss `13.4`, blurry PixCorr `0.196`, test fwd/bwd `0.423/0.267`; train loss `11.5`, train blurry PixCorr `0.286`, train fwd/bwd `0.976/0.966`.
  - `adapter_low`: test loss `13.3`, blurry PixCorr `0.196`, test fwd/bwd `0.423/0.273`; train loss `11.5`, train blurry PixCorr `0.283`, train fwd/bwd `0.978/0.968`.

Full training diagnostics and checkpoint paths:
- `cycle28_subj05_adapter0_1sess_150ep`: final test loss `14.4`, blurry PixCorr `0.189`, test fwd/bwd `0.637/0.543`; train loss `5.89`, train blurry PixCorr `0.801`, train fwd/bwd `1.000/1.000`; checkpoint saved to `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle28_subj05_adapter0_1sess_150ep/last.pth`.
- `cycle28_subj05_adapter_low_1sess_150ep`: final test loss `14.3`, blurry PixCorr `0.200`, test fwd/bwd `0.663/0.540`; train loss `5.88`, train blurry PixCorr `0.800`, train fwd/bwd `1.000/1.000`; checkpoint saved to `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle28_subj05_adapter_low_1sess_150ep/last.pth`.
- `cycle28_subj07_adapter0_1sess_150ep`: final test loss `14.3`, blurry PixCorr `0.219`, test fwd/bwd `0.703/0.597`; train loss `5.94`, train blurry PixCorr `0.787`, train fwd/bwd `1.000/1.000`; checkpoint saved to `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle28_subj07_adapter0_1sess_150ep/last.pth`.
- `cycle28_subj07_adapter_low_1sess_150ep`: final test loss `14.2`, blurry PixCorr `0.222`, test fwd/bwd `0.700/0.590`; train loss `5.97`, train blurry PixCorr `0.787`, train fwd/bwd `1.000/1.000`; checkpoint saved to `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle28_subj07_adapter_low_1sess_150ep/last.pth`.
- Local `/src/train_logs/cycle28_*/last.pth` paths are not visible in this container, but all four evaluator tasks loaded the compute-visible `/scratch/gpfs/.../train_logs/<model>/last.pth` checkpoints successfully.

Evaluation artifacts and enhanced-consumption confirmation:
- The required path `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py` completed for all four rows.
- Evaluator logs confirm `final_evaluations.py` consumed `evals/<model_name>/<model_name>_all_enhancedrecons.pt` for every row via the `all_recons_path` line.
- Enhanced tensors:
  - `/src/evals/cycle28_subj05_adapter0_1sess_150ep/cycle28_subj05_adapter0_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle28_subj05_adapter_low_1sess_150ep/cycle28_subj05_adapter_low_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle28_subj07_adapter0_1sess_150ep/cycle28_subj07_adapter0_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle28_subj07_adapter_low_1sess_150ep/cycle28_subj07_adapter_low_1sess_150ep_all_enhancedrecons.pt`
- Final CSVs:
  - `/src/tables/cycle28_subj05_adapter0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle28_subj05_adapter_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle28_subj07_adapter0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle28_subj07_adapter_low_1sess_150ep_all_enhancedrecons.csv`

Full refined metric table:

| subject | row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist | SwAV dist | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | adapter0 | 0.195854 | 0.405062 | 0.847151 | 0.920398 | 0.856311 | 0.844255 | 0.769723 | 0.433867 | 0.649333 | 0.547111 | 0.419901 | 0.354024 | 0.359358 | 0.344285 | 0.321358 | 0.427158 |
| subj05 | adapter_low | 0.194194 | 0.411969 | 0.849475 | 0.917309 | 0.846193 | 0.839817 | 0.771197 | 0.435373 | 0.667778 | 0.550444 | 0.413983 | 0.350641 | 0.357130 | 0.337356 | 0.315990 | 0.420599 |
| subj07 | adapter0 | 0.192158 | 0.410032 | 0.820612 | 0.878952 | 0.784732 | 0.773060 | 0.825522 | 0.480245 | 0.678444 | 0.570889 | 0.314461 | 0.310144 | 0.311558 | 0.306064 | 0.271515 | 0.300882 |
| subj07 | adapter_low | 0.190809 | 0.406053 | 0.830622 | 0.895010 | 0.779578 | 0.773203 | 0.837802 | 0.483974 | 0.677444 | 0.567111 | 0.316526 | 0.320483 | 0.324429 | 0.312241 | 0.273492 | 0.299470 |

Same-subject deltas, `adapter_low - adapter0`:
- subj05: PixCorr `-0.001660`, SSIM `+0.006908`, AlexNet-2 `+0.002324`, AlexNet-5 `-0.003089`, Inception `-0.010118`, CLIP `-0.004438`, EffNet dist `+0.001473`, SwAV dist `+0.001506`, ImageRet `+0.018444`, BrainRet `+0.003333`, VC `-0.005919`, V1 `-0.003383`, V2 `-0.002228`, V3 `-0.006928`, V4 `-0.005367`, HigherVis `-0.006560`.
- subj07: PixCorr `-0.001349`, SSIM `-0.003979`, AlexNet-2 `+0.010010`, AlexNet-5 `+0.016058`, Inception `-0.005154`, CLIP `+0.000143`, EffNet dist `+0.012280`, SwAV dist `+0.003729`, ImageRet `-0.001000`, BrainRet `-0.003778`, VC `+0.002065`, V1 `+0.010339`, V2 `+0.012871`, V3 `+0.006177`, V4 `+0.001977`, HigherVis `-0.001412`.

Decision:
- Close this exact small residual adapter setting as negative.
- Neither weak subject passes the protected success rule. Subject 5 has only a small BrainRet gain (`+0.003333`, far below about `+0.02`) and worsens CLIP, Inception, EfficientNet/SwAV distances, VC, and HigherVis. Subject 7 loses BrainRet (`-0.003778`) and worsens Inception plus EfficientNet/SwAV distances, with HigherVis slightly lower.
- Per `/plan.md`, do not widen into adapter grids, high-capacity adapters, routers, MoE, generator/refiner edits, caption/VLM correction, temporal decoding, CLIP-layer fusion, hard voxel pruning, another reliability grid, or broad ROI routing from this result.

Recommended next research questions:
- Should the next discussion shift from residual post-ridge adapters to an explicit cross-subject functional-alignment objective?
- What alignment target can be tested while preserving the unchanged MindEye2 one-session protocol and the enhanced-evaluator success gate?

Telegram-ready update:
Cycle 29 recovered the exact Cycle 28 adapter chain. Smoke `8728063_[0-1]`, full training `8728064_[0-3]`, and evaluation `8728065_[0-3]` all completed with exit `0:0`; no reruns or mechanism changes were made. Smoke confirmed the adapter starts as an exact no-op (`max_abs_delta=0`), with `0` trainable adapter params for `adapter0` and `1,060,993` for `adapter_low`. All four full checkpoints were loaded by the evaluator, enhanced tensors were written under `/src/evals/cycle28_*/*_all_enhancedrecons.pt`, and final CSVs were written under `/src/tables/cycle28_*_all_enhancedrecons.csv`; logs confirm `final_evaluations.py` consumed the enhanced tensors. Final BrainRet deltas were weak/negative: subj05 `adapter_low - adapter0 = +0.003333` and subj07 `-0.003778`, not the required `~+0.02`. Protected metrics also regressed for subj05 (CLIP `-0.004438`, Inception `-0.010118`, VC `-0.005919`, HigherVis `-0.006560`, distances worse) and subj07 had worse Inception and distances plus lower HigherVis. Decision: close this exact small zero-initialized residual adapter as negative; next discussion should consider an explicit cross-subject functional-alignment objective rather than widening adapter grids.

## Cycle 30 - 2026-05-28

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- Closed the Cycle 28/29 residual-adapter experiment as negative before starting the new functional-alignment work.

Cycle 28/29 residual-adapter closure:
- Smoke `8728063_[0-1]`, training `8728064_[0-3]`, and evaluation `8728065_[0-3]` all completed with exit `0:0`.
- Enhanced tensors and final CSVs exist for `cycle28_subj05_adapter0_1sess_150ep`, `cycle28_subj05_adapter_low_1sess_150ep`, `cycle28_subj07_adapter0_1sess_150ep`, and `cycle28_subj07_adapter_low_1sess_150ep`.
- Same-subject `adapter_low - adapter0` BrainRet deltas were subj05 `+0.003333` and subj07 `-0.003778`, failing the required about `+0.02` gain.
- Protected metrics regressed: subj05 lost CLIP, Inception, VC, HigherVis, and worsened EffNet/SwAV distances; subj07 lost BrainRet and worsened Inception, EffNet/SwAV distances, and HigherVis. This exact small residual adapter setting is closed as negative.

Code/config changes:
- Added disabled-by-default functional-alignment CLI flags to `/src/Train.py`: `--use_functional_alignment`, `--functional_alignment_weight`, `--functional_alignment_site`, and `--functional_alignment_stats_path`.
- Implemented Cycle 30 alignment at the predicted CLIP boundary. The loss mean-pools predicted CLIP tokens, then applies a mean-plus-covariance distribution match against frozen reference statistics. `align0` uses the same code path and logs the same quantities with `functional_alignment_weight=0.0`; `align_low` uses one conservative nonzero weight, `0.05`.
- Added train/test logging for functional-alignment loss, scaled loss, mean loss, covariance loss, mean distance, and covariance distance.
- Added `/src/cycle30_build_alignment_stats.py` to build training-only reference statistics from one-session training WebDataset shards and `coco_images_224_float16.hdf5`. The script records `training_only=True`, the exact train URL, sample count, dtype/shape, covariance diagnostics, and `shared1000_or_new_test_used=False`.
- Added Slurm scripts:
  - `/src/cycle30_align_smoke.slurm`: 1-hour subject 7 smoke array for `align0` and `align_low`, 3 epochs, A100 `gpu80`, batch size 24.
  - `/src/cycle30_align_train_s57.slurm`: full exact four-row training array for subjects 5/7 and `align0`/`align_low`.
  - `/src/cycle30_align_eval_s57.slurm`: full evaluator array using the unchanged `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py` path.
- No inference-time model structure was added, so `/src/recon_inference.py` did not need new alignment flags.

Validation:
- `/src/fmri/bin/python -m py_compile /src/Train.py /src/cycle30_build_alignment_stats.py /src/recon_inference.py /src/enhanced_recon_inference.py /src/final_evaluations.py` passed.
- `bash -n /src/cycle30_align_smoke.slurm /src/cycle30_align_train_s57.slurm /src/cycle30_align_eval_s57.slurm` passed.

Commands/jobs launched:
- `sbatch /src/cycle30_align_smoke.slurm` -> job `8866561_[0-1]`.
- Smoke rows queued:
  - task 0: `cycle30_smoke_subj07_align0_1sess_3ep`, `functional_alignment_weight=0.0`, stats path `/src/tables/cycle30_subj07_clip_train_stats.pt` on the submit-side path and `${SRC_DIR}/tables/cycle30_subj07_clip_train_stats.pt` on compute.
  - task 1: `cycle30_smoke_subj07_align_low_1sess_3ep`, `functional_alignment_weight=0.05`, same training-only stats path.
- First scheduler check: `8866561_0` and `8866561_1` were both `PENDING`, elapsed `00:00:00`, time limit `01:00:00`, no node assigned. `sacct` also showed pending state for the array.

Observed metrics/results:
- No Cycle 30 smoke metrics, checkpoints, or alignment-stat files were available at this writeout because the smoke array had not started.
- Full training and evaluator arrays were intentionally not submitted yet. Per `/plan.md`, continue to full training only after both smoke rows load official checkpoints, compute/log alignment losses, complete backward, and save checkpoints without protocol changes.

Current artifact expectations:
- If the smoke starts, it should first build `/src/tables/cycle30_subj07_clip_train_stats.pt` from only `wds/subj07/train/{0..0}.tar` and log shape, dtype, sample count, mean norm, covariance norm, and no test-set use.
- Expected smoke checkpoints, if successful, are under `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_smoke_subj07_align{0,_low}_1sess_3ep/last.pth`.

Recommended next research questions:
- Did `8866561_[0-1]` start, build training-only CLIP stats, load the official subject-7 multisubject checkpoint, log non-NaN train/test functional-alignment losses and distances, complete backward, and save both smoke checkpoints?
- If both smoke rows pass, submit `/src/cycle30_align_train_s57.slurm` for the exact four planned rows, then the dependent `/src/cycle30_align_eval_s57.slurm` only after training succeeds.
- If either smoke row fails, recover only the exact failed smoke row with unchanged subject, split, batch size, hidden dim, epoch count, initialization, refiner/evaluator path, and retrieval pool.

Cycle 30 smoke completion and full-array launch update:
- Smoke `8866561_[0-1]` completed successfully, exit `0:0`, node `della-l08g6`, elapsed `00:08:04` for both rows.
  - `8866561_0` `cycle30_smoke_subj07_align0_1sess_3ep`: batch MaxRSS `21547564K`; checkpoint saved to `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_smoke_subj07_align0_1sess_3ep/last.pth`.
  - `8866561_1` `cycle30_smoke_subj07_align_low_1sess_3ep`: batch MaxRSS `21537520K`; checkpoint saved to `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_smoke_subj07_align_low_1sess_3ep/last.pth`.
- Training-only reference statistics were built and verified at `/src/tables/cycle30_subj07_clip_train_stats.pt` / compute path `${SRC_DIR}/tables/cycle30_subj07_clip_train_stats.pt`: train URL `wds/subj07/train/{0..0}.tar`, `training_only=True`, `test_sources_used=[]`, `shared1000_or_new_test_used=False`, `count=600`, mean shape `[1664]`, covariance shape `[1664, 1664]`, dtype `torch.float32`, mean norm `22.0371`, covariance norm `23.9916`, covariance diagonal range `0.000878-6.695669`.
- Both smoke rows loaded `/src/train_logs/final_multisubject_subj07/last.pth`, initialized the CLIP-site alignment reference, completed backward/optimizer steps, logged finite train/test alignment losses, and saved checkpoints without changing batch size or protocol.
- Final 3-epoch smoke diagnostics from logs:
  - `align0`: test loss `13.4`, blurry PixCorr `0.196`, test fwd/bwd `0.427/0.267`; train loss `11.5`, train blurry PixCorr `0.286`, train fwd/bwd `0.976/0.966`; train functional-alignment loss `0.307`, scaled `0.0000`, mean distance `22.6`, covariance distance `33.0`; test functional-alignment loss `0.296`, mean distance `22.2`, covariance distance `24.5`.
  - `align_low`: test loss `13.4`, blurry PixCorr `0.197`, test fwd/bwd `0.423/0.273`; train loss `11.5`, train blurry PixCorr `0.286`, train fwd/bwd `0.976/0.966`; train functional-alignment loss `0.301`, scaled `0.0151`, mean distance `22.4`, covariance distance `32.9`; test functional-alignment loss `0.290`, mean distance `21.9`, covariance distance `24.5`.
- Because the smoke gate passed, launched full training: `sbatch /src/cycle30_align_train_s57.slurm` -> job `8867095_[0-3]` for exactly `cycle30_subj05_align0_1sess_150ep`, `cycle30_subj05_align_low_1sess_150ep`, `cycle30_subj07_align0_1sess_150ep`, and `cycle30_subj07_align_low_1sess_150ep`.
- Launched dependent evaluator: `sbatch --dependency=afterok:8867095 /src/cycle30_align_eval_s57.slurm` -> job `8867096_[0-3]` using the unchanged enhanced path.
- Scheduler state after submission: `8867095_[0-3]` pending with `04:30:00`, no node assigned; `8867096_[0-3]` pending on dependency with `04:00:00`, no node assigned.

## Cycle 31 - 2026-05-28

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- Per plan, this cycle was limited to recovering, validating, and interpreting the already-launched Cycle 30 functional-alignment chain. No new modeling branch, grid, adapter, refiner/generator change, or evaluator change was started.

Code/config changes:
- None. `/src/cycle30_align_train_s57.slurm` and `/src/cycle30_align_eval_s57.slurm` remain the active exact-row scripts.

Scheduler/accounting readout:
- Training array `8867095_[0-3]` is still pending; no task has started, so there are no stdout/stderr logs, checkpoints, MaxRSS values, final losses, or failure classes yet.
  - `8867095_0` `cycle30_subj05_align0_1sess_150ep`: `PENDING`, reason `Priority`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:30:00`, scheduled node `della-l04g6`, projected start `2026-05-28T03:28:35`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_0.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_0.err`.
  - `8867095_1` `cycle30_subj05_align_low_1sess_150ep`: `PENDING`, reason `Priority`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:30:00`, scheduled node `della-l02g7`, projected start `2026-05-28T02:47:55`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_1.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_1.err`.
  - `8867095_2` `cycle30_subj07_align0_1sess_150ep`: `PENDING`, reason `Priority`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:30:00`, scheduled node `della-l02g5`, projected start `2026-05-28T03:28:35`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_2.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_2.err`.
  - `8867095_3` `cycle30_subj07_align_low_1sess_150ep`: `PENDING`, reason `Priority`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:30:00`, no projected node/start yet, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_3.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_3.err`.
- Dependent evaluator array `8867096_[0-3]` is still pending on `afterok:8867095_*(unfulfilled)`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:00:00`.
  - Expected stdout/stderr paths are `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_{0,1,2,3}.out/.err`.

Stats and artifact validation:
- Subject 7 stats exist at `/src/tables/cycle30_subj07_clip_train_stats.pt`, size `11084456` bytes. Loaded keys: `count`, `cov`, `mean`, `provenance`, `site`, `source`, and `training_only`.
- Subject 7 stats are training-only: source `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds/subj07/train/{0..0}.tar`, `training_only=True`, `count=600`, `skipped_duplicate_batches=4`, `unique_image_count=482`, site `clip`, mean shape `(1664,)`, covariance shape `(1664, 1664)`, dtype `torch.float32`, mean norm `22.0371`, covariance norm `23.9916`, covariance diagonal range `0.000878-6.695669`.
- Subject 5 stats are not present yet at `/src/tables/cycle30_subj05_clip_train_stats.pt`. This is expected while `8867095_0` and `8867095_1` remain pending; the training Slurm script builds the subject-specific stats before training starts. Treat subject 5 results as unavailable until that file is built and verified from `wds/subj05/train/{0..0}.tar`.
- No enhanced tensors or final CSVs exist yet for the four required rows:
  - `cycle30_subj05_align0_1sess_150ep`
  - `cycle30_subj05_align_low_1sess_150ep`
  - `cycle30_subj07_align0_1sess_150ep`
  - `cycle30_subj07_align_low_1sess_150ep`

Recovery/rerun actions:
- None. The original exact training array is still pending and recoverable scheduler-side, and the evaluator dependency remains correctly attached. Launching duplicate training or evaluator jobs now would violate the plan.

Metric table and deltas:
- Not available. No full training task has run, no checkpoints exist, and the evaluator has not started.
- Required `align_low - align0` deltas for subjects 5 and 7 cannot be computed yet.

Decision:
- Operationally incomplete and scheduler-limited. The Cycle 30 smoke gate remains valid, but the four full rows have not yet produced checkpoints or enhanced-evaluator CSVs.
- Continue monitoring `8867095_[0-3]`; after all four training tasks complete, verify subject 5 stats provenance, parse training diagnostics, and let `8867096_[0-3]` run. If any checkpoint completes without a final CSV, rerun only the validated enhanced evaluator path for that same model name.

Recommended next research questions:
- Did `8867095_[0-3]` start at the projected backfill times and build subject 5 stats from only `wds/subj05/train/{0..0}.tar`?
- Do all four full rows complete with finite final functional-alignment losses, mean distances, covariance distances, blurry PixCorr, and forward/backward retrieval?
- Once `8867096_[0-3]` completes, do same-subject `align_low - align0` deltas reach about `+0.02` BrainRet while preserving CLIP, Inception, ImageRet, VC, HigherVis, and lower-is-better EffNet/SwAV distances?

## Cycle 32 - 2026-05-28

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- This cycle remained limited to recovering and validating the already-launched Cycle 30 functional-alignment experiment. No new modeling branch, grid, adapter, refiner/generator change, evaluator change, or duplicate training/evaluator job was launched.

Code/config changes:
- None. `/src/cycle30_align_train_s57.slurm` and `/src/cycle30_align_eval_s57.slurm` remain unchanged and still encode the exact four active rows:
  - `cycle30_subj05_align0_1sess_150ep`
  - `cycle30_subj05_align_low_1sess_150ep`
  - `cycle30_subj07_align0_1sess_150ep`
  - `cycle30_subj07_align_low_1sess_150ep`

Commands run:
- `squeue -j 8867095,8867096 -o '%i|%T|%M|%l|%D|%R'`
- `sacct -j 8867095,8867096 --format=JobIDRaw,JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS,NodeList,ReqMem,Timelimit%20 -P`
- `scontrol show job 8867095_0 8867095_1 8867095_2 8867095_3 8867096`
- Inspected `/src/cycle30_align_train_s57.slurm`, `/src/cycle30_align_eval_s57.slurm`, `/src/slurms`, `/src/tables`, `/src/evals`, and compute-visible `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/...` artifact paths.
- Loaded Cycle 30 alignment stats with `/src/fmri/bin/python` and `torch.load(..., map_location='cpu')`.

Scheduler/accounting readout:
- Training array `8867095_[0-3]` is still pending for priority. No training task has started, so there are no full-row stdout/stderr log files, checkpoints, final training losses, functional-alignment losses, MaxRSS values, or failure classes yet.
- `8867095_0` / `cycle30_subj05_align0_1sess_150ep`: `PENDING`, reason `Priority`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:30:00`, scheduled node `della-l04g6`, projected start `2026-05-28T03:28:35`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_0.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_0.err`, expected checkpoint `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj05_align0_1sess_150ep/last.pth` not yet present.
- `8867095_1` / `cycle30_subj05_align_low_1sess_150ep`: `PENDING`, reason `Priority`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:30:00`, scheduled node `della-l02g5`, projected start `2026-05-28T03:28:35`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_1.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_1.err`, expected checkpoint `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj05_align_low_1sess_150ep/last.pth` not yet present.
- `8867095_2` / `cycle30_subj07_align0_1sess_150ep`: `PENDING`, reason `Priority`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:30:00`, scheduled node `della-l04g4`, projected start `2026-05-28T04:54:00`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_2.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_2.err`, expected checkpoint `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj07_align0_1sess_150ep/last.pth` not yet present.
- `8867095_3` / `cycle30_subj07_align_low_1sess_150ep`: `PENDING`, reason `Priority`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:30:00`, no projected node/start yet, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_3.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_s57_8867095_3.err`, expected checkpoint `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj07_align_low_1sess_150ep/last.pth` not yet present.
- Dependent evaluator array `8867096_[0-3]` remains `PENDING`, reason `Dependency`, dependency `afterok:8867095_*(unfulfilled)`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:00:00`. It should not start until all four training tasks complete successfully.

Stats and artifact validation:
- Subject 7 stats are present and valid at `/src/tables/cycle30_subj07_clip_train_stats.pt`, size `11084456` bytes.
- Subject 7 stats provenance: source/train URL `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds/subj07/train/{0..0}.tar`, `training_only=True`, `shared1000_or_new_test_used=False`, `test_sources_used=[]`, `count=600`, `skipped_duplicate_batches=4`, `unique_image_count=482`, site `clip`, mean shape `(1664,)`, covariance shape `(1664, 1664)`, dtype `torch.float32`, mean norm `22.0371`, covariance norm `23.9916`, covariance diagonal range `0.000878-6.695669`.
- Subject 5 stats are still absent at `/src/tables/cycle30_subj05_clip_train_stats.pt`. This is expected while `8867095_0` and `8867095_1` have not started; the training script builds the stats before training. Subject 5 must remain uninterpretable until this file exists and is verified from `wds/subj05/train/{0..0}.tar` with no shared1000/new-test leakage.
- No full-row enhanced tensors or final CSVs exist yet for the four Cycle 30 rows under `/src/evals` or `/src/tables`.
- Smoke logs from `8866561_[0-1]` remain available under `/src/slurms/c30_align_smoke_8866561_{0,1}.out/.err`; no additional smoke rerun was needed.

Metric table and deltas:
- Not available. The required refined table cannot be computed because `8867095_[0-3]` has not started, no full checkpoints exist, and `8867096_[0-3]` remains dependency-blocked.
- Same-subject `align_low - align0` deltas for subject 5 and subject 7 are therefore unavailable.

Decision:
- Operationally incomplete and scheduler-limited. The original Cycle 30 full training and evaluator dependency chain remains alive and recoverable; missing local full-row logs/artifacts are evidence that the tasks have not started, not evidence of failure.
- No rerun was launched. The next valid action is still to wait for `8867095_[0-3]` to run, then verify subject 5 stats, parse final training diagnostics, and allow the dependent enhanced evaluator `8867096_[0-3]` to produce CSVs. If a checkpoint appears without a final CSV, rerun only the validated enhanced evaluator path for that exact model name.

Recommended next research questions:
- Did `8867095_0` and `8867095_1` build `/src/tables/cycle30_subj05_clip_train_stats.pt` from only `wds/subj05/train/{0..0}.tar`, with `shared1000_or_new_test_used=False`?
- Do the four full rows complete with finite final train/test functional-alignment losses, mean distances, covariance distances, blurry PixCorr, and forward/backward retrieval?
- Once `8867096_[0-3]` completes, do `align_low - align0` BrainRet deltas reach about `+0.02` separately for subjects 5 and 7 while preserving CLIP, Inception, ImageRet, VC, HigherVis, and lower-is-better EfficientNet/SwAV distances?

## Cycle 33 - 2026-05-28

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- This cycle remained limited to recovering, validating, and interpreting the already-launched Cycle 30 distributional functional-alignment experiment. No new modeling branch, duplicate training row, grid, adapter, refiner/generator change, or evaluator change was launched.

Code/config changes:
- None.

Commands run:
- `squeue -j 8867095,8867096 -o '%i|%T|%M|%l|%D|%R'`
- `sacct -j 8867095,8867096 --format=JobIDRaw,JobID,JobName%45,State,ExitCode,Elapsed,MaxRSS,NodeList,ReqMem,Timelimit%20 -P`
- `scontrol show job 8867096_0 8867096_1 8867096_2 8867096_3`
- Inspected `/src/slurms/c30_align_s57_8867095_{0,1,2,3}.out/.err`, `/src/tables`, `/src/evals`, and the expected compute-visible checkpoint/evaluator paths.
- Loaded Cycle 30 alignment stats with `/src/fmri/bin/python` and `torch.load(..., map_location='cpu')`.

Scheduler/accounting readout:
- Full training array `8867095_[0-3]` completed successfully:
  - `8867095_0` / `cycle30_subj05_align0_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l03g2`, elapsed `02:21:22`, batch MaxRSS `21659252K`, requested `64G`, time limit `04:30:00`, logs `/src/slurms/c30_align_s57_8867095_0.out/.err`.
  - `8867095_1` / `cycle30_subj05_align_low_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l02g15`, elapsed `02:07:41`, batch MaxRSS `21607776K`, requested `64G`, time limit `04:30:00`, logs `/src/slurms/c30_align_s57_8867095_1.out/.err`.
  - `8867095_2` / `cycle30_subj07_align0_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l02g12`, elapsed `02:06:10`, batch MaxRSS `21591016K`, requested `64G`, time limit `04:30:00`, logs `/src/slurms/c30_align_s57_8867095_2.out/.err`.
  - `8867095_3` / `cycle30_subj07_align_low_1sess_150ep`: `COMPLETED`, exit `0:0`, node `della-l03g2`, elapsed `02:18:32`, batch MaxRSS `21446276K`, requested `64G`, time limit `04:30:00`, logs `/src/slurms/c30_align_s57_8867095_3.out/.err`.
- Dependent evaluator array `8867096_[0-3]` is now dependency-cleared but still pending for priority. `squeue` shows all four tasks `PENDING`; `scontrol` reports `Dependency=(null)`, requested `64G`, time limit `04:00:00`, command `/src/cycle30_align_eval_s57.slurm`.
  - `8867096_0` projected start `2026-05-28T13:39:08`, scheduled node `della-l04g1`, stdout/stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_0.out/.err`.
  - `8867096_1` projected start `2026-05-28T13:39:07`, scheduled node `della-l03g16`, stdout/stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_1.out/.err`.
  - `8867096_2` projected start `2026-05-28T13:39:07`, scheduled node `della-l03g9`, stdout/stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_2.out/.err`.
  - `8867096_3` projected start unknown, no scheduled node yet, stdout/stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_3.out/.err`.

Stats provenance:
- Subject 5 stats are now present and valid at `/src/tables/cycle30_subj05_clip_train_stats.pt`, size `11084456` bytes.
  - Source/train URL `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds/subj05/train/{0..0}.tar`; `training_only=True`; `shared1000_or_new_test_used=False`; `test_sources_used=[]`; `count=600`; `skipped_duplicate_batches=4`; `unique_image_count=482`; image index range `308-72877`; site `clip`; feature `FrozenOpenCLIPImageEmbedder tokens mean-pooled over sequence`.
  - Mean shape `(1664,)`, covariance shape `(1664, 1664)`, dtype `torch.float32`; mean norm `21.9571`; covariance norm `24.1234`; covariance diagonal range `0.000757-7.141074`.
- Subject 7 stats remain present and valid at `/src/tables/cycle30_subj07_clip_train_stats.pt`, size `11084456` bytes.
  - Source/train URL `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds/subj07/train/{0..0}.tar`; `training_only=True`; `shared1000_or_new_test_used=False`; `test_sources_used=[]`; `count=600`; `skipped_duplicate_batches=4`; `unique_image_count=482`; image index range `182-72907`; site `clip`; feature `FrozenOpenCLIPImageEmbedder tokens mean-pooled over sequence`.
  - Mean shape `(1664,)`, covariance shape `(1664, 1664)`, dtype `torch.float32`; mean norm `22.0371`; covariance norm `23.9916`; covariance diagonal range `0.000878-6.695669`.

Training diagnostics:
- All four rows loaded the official subject-specific multisubject checkpoint, initialized the CLIP-site functional-alignment reference from subject-specific training-only stats, completed 150 epochs, and saved final checkpoints according to stdout.
- Expected compute-visible checkpoint paths from training logs:
  - `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj05_align0_1sess_150ep/last.pth`
  - `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj05_align_low_1sess_150ep/last.pth`
  - `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj07_align0_1sess_150ep/last.pth`
  - `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj07_align_low_1sess_150ep/last.pth`
- The local container cannot directly list `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2`, but evaluator jobs use that same compute-visible path and remain queued. Treat local `/scratch` absence as a mount visibility limitation, not checkpoint failure, unless evaluator logs later contradict this.

Final 150-epoch training-log metrics:

| subject | row | test loss | test blurry PixCorr | test fwd | test bwd | test align loss | test mean dist | test cov dist | train loss | train blurry PixCorr | train fwd | train bwd | train align loss | train scaled align | train mean dist | train cov dist |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | align0 | 14.3 | 0.193 | 0.670 | 0.583 | 0.31000 | 22.7 | 24.6 | 5.88 | 0.801 | 1.000 | 1.000 | 0.34800 | 0.000000 | 24.1 | 29.3 |
| subj05 | align_low | 15.3 | 0.174 | 0.413 | 0.313 | 0.00914 | 3.85 | 25.8 | 5.88 | 0.799 | 1.000 | 1.000 | 0.00465 | 0.000233 | 2.62 | 36.4 |
| subj07 | align0 | 14.2 | 0.226 | 0.730 | 0.563 | 0.30100 | 22.4 | 24.2 | 5.95 | 0.788 | 1.000 | 1.000 | 0.33100 | 0.000000 | 23.5 | 29.5 |
| subj07 | align_low | 15.0 | 0.223 | 0.313 | 0.397 | 0.00967 | 3.96 | 25.1 | 5.94 | 0.787 | 1.000 | 1.000 | 0.00490 | 0.000245 | 2.68 | 37.4 |

Manipulation-check interpretation:
- `align_low` strongly reduced mean-distance dominated functional-alignment loss in both subjects, from test alignment loss `0.310 -> 0.00914` in subject 5 and `0.301 -> 0.00967` in subject 7.
- This is not yet scientific success. The plan requires enhanced-evaluator metrics; training-log retrieval and lower alignment loss are only manipulation checks.
- Training-log test retrieval worsened in both `align_low` rows, especially subject 5 (`fwd 0.670 -> 0.413`, `bwd 0.583 -> 0.313`) and subject 7 (`fwd 0.730 -> 0.313`, `bwd 0.563 -> 0.397`). This raises concern for coarse distribution matching without preserved instance-level geometry, but the final decision must wait for refined evaluator CSVs.

Evaluation artifacts:
- No final enhanced tensors or final CSVs exist yet under `/src/evals` or `/src/tables` for the four Cycle 30 rows because evaluator array `8867096_[0-3]` has not started.
- Required refined metric table and same-subject `align_low - align0` deltas are unavailable until `8867096_[0-3]` completes and logs confirm `final_evaluations.py` consumed the enhanced tensor via the `all_recons_path` line.

Recovery/rerun actions:
- None. The original evaluator array is alive, dependency-cleared, and pending for priority. Launching a duplicate evaluator now would be unnecessary and could violate the recovery plan.

Decision:
- Operationally advanced but still incomplete. Full training succeeded for the four exact active rows, subject-specific training-only stats are verified for both subjects, and the dependent evaluator is queued correctly.
- No Cycle 30 success/failure decision can be made until enhanced-evaluator CSVs exist. The immediate next valid action is to let `8867096_[0-3]` run, then verify the enhanced tensor consumption path and compute the required refined metric table plus same-subject deltas.

Recommended next research questions:
- Does `8867096_[0-3]` complete with exit `0:0` and produce enhanced tensors at `evals/<model_name>/<model_name>_all_enhancedrecons.pt` for all four rows?
- Do evaluator logs confirm `final_evaluations.py` consumed each enhanced tensor via the `all_recons_path` line?
- Do refined `align_low - align0` BrainRet deltas reach about `+0.02` separately for subjects 5 and 7 while preserving CLIP, Inception, ImageRet, VC, HigherVis, and lower-is-better EfficientNet/SwAV distances?
- If refined metrics mirror the training-log retrieval collapse, close this distributional functional-alignment attempt as coarse global matching without preserved instance-level functional geometry.

## Cycle 34 - 2026-05-28

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- Scope remained limited to the already-trained Cycle 30 distributional functional-alignment experiment. No training rerun, modeling branch, grid, adapter, refiner/generator change, evaluator change, or duplicate evaluator was launched.

Code/config changes:
- None.

Commands run:
- `squeue -j 8867096 -o '%i|%T|%M|%D|%R|%N'`
- `sacct -j 8867096 --format=JobIDRaw,JobName%40,State,ExitCode,Elapsed,NodeList,MaxRSS,ReqMem,AllocTRES%80 -P`
- `scontrol show job 8867096_0`, `_1`, `_2`, and `_3`
- Inspected `/src/cycle30_align_eval_s57.slurm`, `/src/slurms`, `/src/train_logs`, `/src/evals`, `/src/tables`, and the expected `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/...` paths from the container.
- Loaded `/src/tables/cycle30_subj05_clip_train_stats.pt` and `/src/tables/cycle30_subj07_clip_train_stats.pt` with `/src/fmri/bin/python` and `torch.load(..., map_location='cpu')`.

Scheduler/accounting readout for evaluator `8867096_[0-3]`:
- `8867096_0` / `cycle30_subj05_align0_1sess_150ep`: `PENDING`, reason `Priority`, dependency `(null)`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:00:00`, projected start `2026-05-28T13:56:37`, scheduled node `della-l04g2`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_0.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_0.err`, MaxRSS unavailable while pending, failure class none yet.
- `8867096_1` / `cycle30_subj05_align_low_1sess_150ep`: `PENDING`, reason `Priority`, dependency `(null)`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:00:00`, projected start `2026-05-28T13:56:37`, scheduled node `della-l04g8`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_1.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_1.err`, MaxRSS unavailable while pending, failure class none yet.
- `8867096_2` / `cycle30_subj07_align0_1sess_150ep`: `PENDING`, reason `Priority`, dependency `(null)`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:00:00`, projected start `2026-05-28T13:57:07`, scheduled node `della-l04g6`, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_2.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_2.err`, MaxRSS unavailable while pending, failure class none yet.
- `8867096_3` / `cycle30_subj07_align_low_1sess_150ep`: `PENDING`, reason `Priority`, dependency `(null)`, exit `0:0`, elapsed `00:00:00`, requested `64G`, time limit `04:00:00`, projected start unknown, no scheduled node yet, stdout `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_3.out`, stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/c30_align_eval_s57_8867096_3.err`, MaxRSS unavailable while pending, failure class none yet.

Checkpoint and artifact validation:
- The evaluator script remains pointed at the exact four active rows and checks `../train_logs/${MODEL_NAME}/last.pth` on the compute-visible `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src` working tree before running the unchanged path: `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py` with `--all_recons_path=evals/${MODEL_NAME}/${MODEL_NAME}_all_enhancedrecons.pt`.
- From this container, `/scratch` is not mounted, so direct `ls` of `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_*/last.pth` reports `No such file or directory`. This is a container mount-visibility limitation already seen in Cycle 33, not evidence of checkpoint loss. The training logs under `/src/slurms/c30_align_s57_8867095_{0,1,2,3}.out` still show the exact model names, official subject-specific multisubject checkpoint loads, functional-alignment reference initialization, and reliability summaries written under the compute-visible `/scratch/gpfs/.../train_logs/cycle30_*` directories.
- No local enhanced tensors or final CSVs exist yet for the four Cycle 30 rows under `/src/evals` or `/src/tables`, and no evaluator stdout/stderr files for `8867096_[0-3]` exist yet. This is expected while the evaluator array remains pending and has not started.

Stats provenance preserved:
- Subject 5 stats: `/src/tables/cycle30_subj05_clip_train_stats.pt`, size `11084456` bytes; train URL `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds/subj05/train/{0..0}.tar`; `training_only=True`; `shared1000_or_new_test_used=False`; `count=600`; `unique_image_count=482`; `skipped_duplicate_batches=4`; dtype `torch.float32`; mean shape `(1664,)`; covariance shape `(1664, 1664)`; mean norm `21.957136`; covariance norm `24.123350`; covariance diagonal range `0.000757-7.141074`.
- Subject 7 stats: `/src/tables/cycle30_subj07_clip_train_stats.pt`, size `11084456` bytes; train URL `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds/subj07/train/{0..0}.tar`; `training_only=True`; `shared1000_or_new_test_used=False`; `count=600`; `unique_image_count=482`; `skipped_duplicate_batches=4`; dtype `torch.float32`; mean shape `(1664,)`; covariance shape `(1664, 1664)`; mean norm `22.037144`; covariance norm `23.991619`; covariance diagonal range `0.000878-6.695669`.

Required refined metric table:
- Not available yet. `8867096_[0-3]` has not started, so the enhanced tensors and final CSVs do not yet exist for:
  - `cycle30_subj05_align0_1sess_150ep`
  - `cycle30_subj05_align_low_1sess_150ep`
  - `cycle30_subj07_align0_1sess_150ep`
  - `cycle30_subj07_align_low_1sess_150ep`
- Same-subject `align_low - align0` deltas for subject 5 and subject 7 cannot be computed until those CSVs exist and evaluator logs confirm `final_evaluations.py` consumed each enhanced tensor via the `all_recons_path` line.

Recovery/rerun actions:
- None. The evaluator is dependency-cleared, pending for priority, and recoverable. Launching a duplicate evaluator would violate the plan while `8867096_[0-3]` remains alive with no failure evidence.
- If a row later fails operationally, recover only that exact evaluator row with unchanged subject, model name, checkpoint, refiner, evaluator, and retrieval pool. If a checkpoint exists but a CSV is missing after `8867096_[0-3]` resolves, rerun only the validated enhanced evaluator path for that exact model name.

Decision:
- Cycle 34 remains operationally incomplete because the scheduler has not started the evaluator rows. The Cycle 30 training chain is still the active experiment, and no scientific pass/fail decision is possible without the enhanced-evaluator CSVs.
- The immediate next valid action is to wait for `8867096_[0-3]` to run, parse its stdout/stderr and accounting, verify enhanced tensors and final CSVs, then compute the full refined table and same-subject deltas required by `/plan.md`.

Recommended next research questions:
- Did `8867096_[0-3]` start at or after the projected `2026-05-28T13:56` window and find all four compute-visible checkpoints?
- Do evaluator logs confirm the unchanged path `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py` and `all_recons_path=evals/<model_name>/<model_name>_all_enhancedrecons.pt` for each row?
- Once CSVs exist, do refined `align_low - align0` BrainRet deltas reach about `+0.02` separately for subjects 5 and 7 while preserving CLIP, Inception, ImageRet, VC, HigherVis, and lower-is-better EfficientNet/SwAV distances?

## Cycle 37 - 2026-05-29

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- Scope was Cycle 36 topology artifact authentication and scientific readout only. No new model branch, loss variant, subject set, evaluator variant, or recovery job was launched.
- `/job-status.md` was requested by the plan but is not present in this container. Slurm accounting and row logs were used for authentication.

Code/config changes:
- None.

Commands run:
- `sacct -j 8918443,8918444 --format=JobIDRaw,JobID,JobName%50,State,ExitCode,Elapsed,MaxRSS,NodeList,ReqMem,Timelimit%20 -P`
- Inspected `/src/slurms/c36_topo_s57_8918443_{0,1,2,3}.out/.err`.
- Inspected `/src/slurms/c36_topo_eval_s57_8918444_{0,1,2,3}.out/.err`.
- Loaded enhanced tensors from `/src/evals/cycle36_subj0{5,7}_topo{0,_low}_1sess_150ep/`.
- Parsed final CSVs under `/src/tables/cycle36_subj0{5,7}_topo{0,_low}_1sess_150ep_all_enhancedrecons.csv`.

Scheduler/accounting readout:
- `8918443_0` / `cycle36_subj05_topo0_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:16:54`, node `della-l02g12`, requested `64G`, time limit `04:30:00`, batch MaxRSS `21528912K`, stdout `/src/slurms/c36_topo_s57_8918443_0.out`, stderr `/src/slurms/c36_topo_s57_8918443_0.err`, failure class none.
- `8918443_1` / `cycle36_subj05_topo_low_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:16:54`, node `della-l02g12`, requested `64G`, time limit `04:30:00`, batch MaxRSS `21533120K`, stdout `/src/slurms/c36_topo_s57_8918443_1.out`, stderr `/src/slurms/c36_topo_s57_8918443_1.err`, failure class none.
- `8918443_2` / `cycle36_subj07_topo0_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:07:24`, node `della-l01g15`, requested `64G`, time limit `04:30:00`, batch MaxRSS `21587656K`, stdout `/src/slurms/c36_topo_s57_8918443_2.out`, stderr `/src/slurms/c36_topo_s57_8918443_2.err`, failure class none.
- `8918443_3` / `cycle36_subj07_topo_low_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:09:04`, node `della-l01g14`, requested `64G`, time limit `04:30:00`, batch MaxRSS `21594544K`, stdout `/src/slurms/c36_topo_s57_8918443_3.out`, stderr `/src/slurms/c36_topo_s57_8918443_3.err`, failure class none.
- `8918444_0` / `cycle36_subj05_topo0_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:23:53`, node `della-l02g12`, requested `64G`, time limit `04:00:00`, batch MaxRSS `49158324K`, stdout `/src/slurms/c36_topo_eval_s57_8918444_0.out`, stderr `/src/slurms/c36_topo_eval_s57_8918444_0.err`, failure class none.
- `8918444_1` / `cycle36_subj05_topo_low_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:29:21`, node `della-l03g16`, requested `64G`, time limit `04:00:00`, batch MaxRSS `49577376K`, stdout `/src/slurms/c36_topo_eval_s57_8918444_1.out`, stderr `/src/slurms/c36_topo_eval_s57_8918444_1.err`, failure class none.
- `8918444_2` / `cycle36_subj07_topo0_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:33:44`, node `della-l02g14`, requested `64G`, time limit `04:00:00`, batch MaxRSS `49568324K`, stdout `/src/slurms/c36_topo_eval_s57_8918444_2.out`, stderr `/src/slurms/c36_topo_eval_s57_8918444_2.err`, failure class none.
- `8918444_3` / `cycle36_subj07_topo_low_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:22:40`, node `della-l04g9`, requested `64G`, time limit `04:00:00`, batch MaxRSS `49587428K`, stdout `/src/slurms/c36_topo_eval_s57_8918444_3.out`, stderr `/src/slurms/c36_topo_eval_s57_8918444_3.err`, failure class none.

Training artifact authentication:
- All four rows loaded the official one-session subject-specific multisubject initialization from `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/train_logs/final_multisubject_subj0{5,7}/last.pth`.
- All four rows saved repeated checkpoints to `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/<model_name>/last.pth`.
- The Slurm script and row stdout confirm `topo0` used the topology code path with `TOPO_WEIGHT=0.0`, and `topo_low` used `TOPO_WEIGHT=0.001`.
- The final tqdm records do not emit a `train/loss` key, so final train loss is unavailable from the retained logs. Final test loss and train-side diagnostics were retained.

Final training diagnostics:

| row | test loss | train PixCorr | test PixCorr | train bwd | test fwd | test bwd | train topo | train topo scaled | test topo | test sim corr | test NN@1 | test NN@5 | test NN@10 | test teacher median rank | test MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cycle36_subj05_topo0_1sess_150ep` | 14.3 | 0.801 | 0.193 | 1.000 | 0.670 | 0.583 | 2.500 | 0.000000 | 2.340 | 0.344 | 0.103 | 0.170 | 0.226 | 20 | 0.195 |
| `cycle36_subj05_topo_low_1sess_150ep` | 14.6 | 0.799 | 0.176 | 1.000 | 0.680 | 0.543 | 0.895 | 0.000895 | 0.772 | 0.361 | 0.113 | 0.179 | 0.236 | 17 | 0.207 |
| `cycle36_subj07_topo0_1sess_150ep` | 14.2 | 0.788 | 0.226 | 1.000 | 0.730 | 0.563 | 2.620 | 0.000000 | 2.350 | 0.290 | 0.050 | 0.131 | 0.160 | 39 | 0.118 |
| `cycle36_subj07_topo_low_1sess_150ep` | 14.5 | 0.788 | 0.220 | 1.000 | 0.720 | 0.530 | 0.935 | 0.000935 | 0.809 | 0.308 | 0.070 | 0.132 | 0.178 | 32 | 0.142 |

Evaluator artifact authentication:
- All four evaluator rows used the unchanged path `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py`.
- Each row loaded the expected checkpoint from `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/<model_name>/last.pth`.
- Each row logged `all_enhancedrecons torch.Size([1000, 3, 256, 256])` and saved `evals/<model_name>/<model_name>_all_enhancedrecons.pt`.
- Independent load check confirmed each enhanced tensor has shape `(1000, 3, 256, 256)` and dtype `torch.float32`.
- `final_evaluations.py` consumed the intended enhanced tensor in every row via `all_recons_path: evals/<model_name>/<model_name>_all_enhancedrecons.pt`.
- Final CSVs exist under `/src/tables` for all four protected rows.

Refined enhanced-evaluator metric table:

| row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist ↓ | SwAV dist ↓ | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cycle36_subj05_topo0_1sess_150ep` | 0.192674 | 0.406933 | 0.846794 | 0.916086 | 0.856197 | 0.840533 | 0.768113 | 0.431212 | 0.648778 | 0.552778 | 0.416381 | 0.351920 | 0.355988 | 0.341006 | 0.315086 | 0.423627 |
| `cycle36_subj05_topo_low_1sess_150ep` | 0.197604 | 0.412733 | 0.841756 | 0.915787 | 0.854318 | 0.853176 | 0.763501 | 0.427588 | 0.647444 | 0.525667 | 0.414780 | 0.350658 | 0.355579 | 0.335405 | 0.314687 | 0.422147 |
| `cycle36_subj07_topo0_1sess_150ep` | 0.194279 | 0.404695 | 0.829048 | 0.894047 | 0.785077 | 0.770978 | 0.828250 | 0.477893 | 0.680556 | 0.542000 | 0.321338 | 0.321689 | 0.324969 | 0.316007 | 0.281305 | 0.306982 |
| `cycle36_subj07_topo_low_1sess_150ep` | 0.197323 | 0.402003 | 0.822153 | 0.883711 | 0.776077 | 0.777483 | 0.831761 | 0.477575 | 0.683667 | 0.492444 | 0.318829 | 0.315724 | 0.318166 | 0.311614 | 0.274215 | 0.304403 |

Same-subject `topo_low - topo0` deltas:

| subject | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist ↓ | SwAV dist ↓ | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | +0.004930 | +0.005799 | -0.005038 | -0.000299 | -0.001879 | +0.012644 | -0.004611 | -0.003624 | -0.001333 | -0.027111 | -0.001600 | -0.001262 | -0.000408 | -0.005601 | -0.000399 | -0.001480 |
| subj07 | +0.003044 | -0.002692 | -0.006895 | -0.010336 | -0.009000 | +0.006506 | +0.003511 | -0.000318 | +0.003111 | -0.049556 | -0.002509 | -0.005965 | -0.006803 | -0.004393 | -0.007091 | -0.002580 |

Interpretation against success criteria:
- Subject 5 fails. BrainRet moved `-0.027111` instead of the required about `+0.02`; ImageRet was essentially preserved at `-0.001333`; CLIP, PixCorr, SSIM, EffNet, and SwAV improved, but Inception, VC, HigherVis, and BrainRet regressed.
- Subject 7 fails. BrainRet moved `-0.049556`; ImageRet was preserved/slightly improved at `+0.003111`; CLIP and SwAV improved slightly, but Inception, EffNet, VC, HigherVis, and lower visual correlations regressed.
- The topology term appears to backpropagate and improve the intended local geometry diagnostics: both `topo_low` rows sharply reduce topology loss, improve test similarity correlation, improve test teacher-neighbor median rank and MRR, and mostly improve NN overlap. That mechanistic improvement did not transfer to refined BrainRet.

Decision:
- Neither subject passes, but topology diagnostics improved, so close this exact full-pairwise batch-local cosine-MSE topology loss as mechanistically informative but practically insufficient.
- Do not run a broad topology weight grid and do not scale this exact setting to subjects 1/2/5/7.
- The only adjacent next branch worth considering is a sharper local-neighborhood objective on subjects 5 and 7, such as teacher-neighbor-weighted pairwise loss or soft nearest-neighbor KL from frozen image-CLIP similarities to predicted-CLIP similarities.

Recommended next research questions:
- Can a sparse teacher-neighbor-weighted objective improve BrainRet without sacrificing ImageRet, unlike the full pairwise topology MSE?
- Are the refined BrainRet losses driven by a small subset of image categories or by broad rank distortion across the 1000-image retrieval pool?
- Would evaluating predicted-CLIP retrieval directly before reconstruction reveal where the topology-improved training diagnostics diverge from held-out refined BrainRet?
## Cycle 35 - 2026-05-29

Plan source:
- Read and executed `/plan.md` only. Telegram report is due.
- Scope was artifact authentication and metric extraction for the completed Cycle 30 distributional functional-alignment evaluator rows. No new modeling branch, training rerun, evaluator rerun, grid, adapter, refiner/generator change, or image-side intervention was launched before interpreting the four required rows.

Code/config changes:
- None.

Commands run:
- `sacct -j 8867096 --format=JobIDRaw,JobID,JobName%45,State,ExitCode,Elapsed,MaxRSS,NodeList,ReqMem,Timelimit%20 -P`
- Inspected `/src/slurms/c30_align_eval_s57_8867096_{0,1,2,3}.out/.err`.
- Inspected `/src/evals/cycle30_subj{05,07}_align{0,_low}_1sess_150ep/` and `/src/tables/cycle30_subj{05,07}_align{0,_low}_1sess_150ep_all_enhancedrecons.csv`.
- Parsed the four final CSVs and loaded `/src/tables/cycle30_subj05_clip_train_stats.pt` and `/src/tables/cycle30_subj07_clip_train_stats.pt`.

Scheduler/accounting readout for enhanced evaluator `8867096_[0-3]`:
- `8867096_0` / `cycle30_subj05_align0_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:25:01`, node `della-l02g10`, requested `64G`, time limit `04:00:00`, batch MaxRSS `49410188K`, stdout `/src/slurms/c30_align_eval_s57_8867096_0.out`, stderr `/src/slurms/c30_align_eval_s57_8867096_0.err`, failure class none.
- `8867096_1` / `cycle30_subj05_align_low_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:29:28`, node `della-l04g8`, requested `64G`, time limit `04:00:00`, batch MaxRSS `49429840K`, stdout `/src/slurms/c30_align_eval_s57_8867096_1.out`, stderr `/src/slurms/c30_align_eval_s57_8867096_1.err`, failure class none.
- `8867096_2` / `cycle30_subj07_align0_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:28:34`, node `della-l02g9`, requested `64G`, time limit `04:00:00`, batch MaxRSS `49432520K`, stdout `/src/slurms/c30_align_eval_s57_8867096_2.out`, stderr `/src/slurms/c30_align_eval_s57_8867096_2.err`, failure class none.
- `8867096_3` / `cycle30_subj07_align_low_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:25:56`, node `della-l02g16`, requested `64G`, time limit `04:00:00`, batch MaxRSS `49627664K`, stdout `/src/slurms/c30_align_eval_s57_8867096_3.out`, stderr `/src/slurms/c30_align_eval_s57_8867096_3.err`, failure class none.

Artifact and path verification:
- All four evaluator rows used the unchanged path `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py`.
- Each row loaded the expected checkpoint:
  - `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj05_align0_1sess_150ep/last.pth`
  - `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj05_align_low_1sess_150ep/last.pth`
  - `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj07_align0_1sess_150ep/last.pth`
  - `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle30_subj07_align_low_1sess_150ep/last.pth`
- All four enhanced tensors exist and were logged as `torch.Size([1000, 3, 256, 256])`:
  - `/src/evals/cycle30_subj05_align0_1sess_150ep/cycle30_subj05_align0_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle30_subj05_align_low_1sess_150ep/cycle30_subj05_align_low_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle30_subj07_align0_1sess_150ep/cycle30_subj07_align0_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle30_subj07_align_low_1sess_150ep/cycle30_subj07_align_low_1sess_150ep_all_enhancedrecons.pt`
- `final_evaluations.py` consumed the enhanced tensor in every row via the expected `all_recons_path=evals/<model_name>/<model_name>_all_enhancedrecons.pt` line.
- Final CSVs were found under `/src/tables`:
  - `/src/tables/cycle30_subj05_align0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle30_subj05_align_low_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle30_subj07_align0_1sess_150ep_all_enhancedrecons.csv`
  - `/src/tables/cycle30_subj07_align_low_1sess_150ep_all_enhancedrecons.csv`

Stats provenance preserved:
- Subject 5 stats: `/src/tables/cycle30_subj05_clip_train_stats.pt`; train URL `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds/subj05/train/{0..0}.tar`; `training_only=True`; `shared1000_or_new_test_used=False`; `test_sources_used=[]`; `count=600`; `unique_image_count=482`; `skipped_duplicate_batches=4`; dtype `torch.float32`; mean shape `(1664,)`; covariance shape `(1664, 1664)`; mean norm `21.957136`; covariance norm `24.123350`; covariance diagonal range `0.000757-7.141074`.
- Subject 7 stats: `/src/tables/cycle30_subj07_clip_train_stats.pt`; train URL `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/wds/subj07/train/{0..0}.tar`; `training_only=True`; `shared1000_or_new_test_used=False`; `test_sources_used=[]`; `count=600`; `unique_image_count=482`; `skipped_duplicate_batches=4`; dtype `torch.float32`; mean shape `(1664,)`; covariance shape `(1664, 1664)`; mean norm `22.037144`; covariance norm `23.991619`; covariance diagonal range `0.000878-6.695669`.

Refined enhanced-evaluator metric table:

| row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist ↓ | SwAV dist ↓ | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cycle30_subj05_align0_1sess_150ep` | 0.192674 | 0.406933 | 0.846794 | 0.916086 | 0.856197 | 0.840533 | 0.768113 | 0.431212 | 0.648778 | 0.552778 | 0.416381 | 0.351920 | 0.355988 | 0.341006 | 0.315086 | 0.423627 |
| `cycle30_subj05_align_low_1sess_150ep` | 0.193479 | 0.413630 | 0.839868 | 0.916032 | 0.856033 | 0.839441 | 0.767524 | 0.432645 | 0.410778 | 0.299000 | 0.409822 | 0.342277 | 0.348690 | 0.333368 | 0.312231 | 0.419540 |
| `cycle30_subj07_align0_1sess_150ep` | 0.194279 | 0.404695 | 0.829048 | 0.894047 | 0.785077 | 0.770978 | 0.828250 | 0.477893 | 0.680556 | 0.542000 | 0.321338 | 0.321689 | 0.324969 | 0.316007 | 0.281305 | 0.306982 |
| `cycle30_subj07_align_low_1sess_150ep` | 0.194236 | 0.406189 | 0.831371 | 0.887494 | 0.788831 | 0.790124 | 0.820455 | 0.473685 | 0.292222 | 0.317222 | 0.322754 | 0.321784 | 0.322480 | 0.313069 | 0.280129 | 0.308908 |

Same-subject `align_low - align0` deltas:

| subject | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist ↓ | SwAV dist ↓ | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | +0.000805 | +0.006696 | -0.006926 | -0.000054 | -0.000164 | -0.001091 | -0.000588 | +0.001433 | -0.238000 | -0.253778 | -0.006559 | -0.009644 | -0.007298 | -0.007639 | -0.002855 | -0.004087 |
| subj07 | -0.000043 | +0.001494 | +0.002323 | -0.006553 | +0.003754 | +0.019146 | -0.007794 | -0.004207 | -0.388333 | -0.224778 | +0.001416 | +0.000095 | -0.002490 | -0.002938 | -0.001176 | +0.001926 |

Interpretation against success criteria:
- Subject 5 fails decisively. BrainRet regressed by `-0.253778`, ImageRet regressed by `-0.238000`, VC regressed by `-0.006559`, HigherVis regressed by `-0.004087`, and SwAV distance worsened by `+0.001433`. CLIP and Inception were nearly preserved, and EffNet distance slightly improved, but the retrieval and brain-alignment regressions dominate.
- Subject 7 also fails decisively. BrainRet regressed by `-0.224778` and ImageRet regressed by `-0.388333`. CLIP improved by `+0.019146`, Inception improved by `+0.003754`, EffNet and SwAV distances improved, and VC/HigherVis moved slightly positive, but the plan requires BrainRet improvement and retrieval preservation, not a retrieval collapse.
- The enhanced table matches the training-log warning: low-weight mean/covariance alignment reduced global moment mismatch but damaged stimulus-level local geometry and semantic retrieval.

Decision:
- Close this specific distributional functional-alignment attempt. Neither subject passes; do not increase alignment weight, add alignment sites, combine with adapters, or move to image-side interventions from this result.
- Practical next-step recommendation, per `/plan.md`: if continuing, test topology-preserving predicted-CLIP alignment on subjects 5 and 7 only, with the unchanged one-session protocol, same-code zero row plus one conservative nonzero row, training-only teacher/reference geometry from frozen image-CLIP embeddings, and a loss that preserves pairwise cosine/rank/nearest-neighbor/RSA-style geometry at the predicted CLIP boundary.

Telegram-ready update:
Cycle 35 authenticated the completed Cycle 30 enhanced-evaluator readout and no new branch was launched. Evaluator array `8867096_[0-3]` completed cleanly with exit `0:0`; row runtimes were 2:25-2:29 and MaxRSS about 49.4-49.6 GB. All four rows loaded the expected checkpoints, produced enhanced tensors, and `final_evaluations.py` consumed `evals/<model>/<model>_all_enhancedrecons.pt`. The result is negative: subj05 `align_low - align0` BrainRet `-0.253778` and ImageRet `-0.238000`; subj07 BrainRet `-0.224778` and ImageRet `-0.388333`. CLIP/Inception were mostly preserved or improved, especially subj07 CLIP `+0.019146`, but retrieval collapsed, so distributional functional alignment is closed as global moment matching that damaged local stimulus geometry. Recommended next branch is topology-preserving predicted-CLIP alignment using training-only pairwise/rank/nearest-neighbor geometry.

## Cycle 36 - 2026-05-29

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- Reconfirmed the Cycle 30 mean-plus-covariance/distributional functional-alignment row is closed as negative from Cycle 35: subj05 `align_low - align0` BrainRet `-0.253778`, ImageRet `-0.238000`; subj07 BrainRet `-0.224778`, ImageRet `-0.388333`.
- `/job-status.md` had no active-job content, and `squeue -u $USER` initially showed no active Slurm jobs.

Code/config changes:
- Added `/src/cycle36_clip_geometry_diagnostic.py` and `/src/cycle36_geometry_diagnostic.slurm`.
  - The diagnostic loads the four completed Cycle 30 checkpoints on compute-visible `/scratch/.../train_logs`, extracts predicted CLIP tokens from training-only `wds/subj0{5,7}/train/{0..0}.tar`, extracts frozen image OpenCLIP-bigG tokens from `coco_images_224_float16.hdf5`, deduplicates image IDs, and computes flattened-token plus mean-pooled-token geometry.
  - It records `training_only=True`, `shared1000_or_new_test_used=False`, and `test_sources_used=[]`.
- Patched `/src/Train.py` with disabled-by-default Cycle 36 topology flags:
  - `--use_clip_topology`, default `False`
  - `--clip_topo_weight`, default `0.0`
  - `--clip_topo_temp`, default `0.07`
  - `--clip_topo_center`, default enabled
  - The loss is batch-local off-diagonal pairwise cosine-similarity MSE after shared temperature scaling and optional row-centering. It logs loss, scaled loss, similarity correlation, top-1/5/10 neighbor overlap, teacher-top1 median rank, and MRR. Defaults preserve prior behavior unless the new flag is passed.
- Added `/src/cycle36_topo_smoke.slurm`, `/src/cycle36_topo_train_s57.slurm`, and `/src/cycle36_topo_eval_s57.slurm`.
  - Full row names: `cycle36_subj05_topo0_1sess_150ep`, `cycle36_subj05_topo_low_1sess_150ep`, `cycle36_subj07_topo0_1sess_150ep`, `cycle36_subj07_topo_low_1sess_150ep`.
  - `topo0` uses the same topology code path with `clip_topo_weight=0.0`; `topo_low` uses one conservative weight, `0.001`.

Validation:
- `/src/fmri/bin/python -m py_compile /src/cycle36_clip_geometry_diagnostic.py /src/Train.py` passed.
- `bash -n /src/cycle36_geometry_diagnostic.slurm /src/cycle36_topo_smoke.slurm /src/cycle36_topo_train_s57.slurm /src/cycle36_topo_eval_s57.slurm` passed.

Phase 1 geometry diagnostic jobs:
- First run `8917610` failed operationally in `00:01:49`, exit `1:0`, node `della-l09g5`, batch MaxRSS `27491684K`, stdout `/src/slurms/c36_geom_diag_8917610.out`, stderr `/src/slurms/c36_geom_diag_8917610.err`.
  - Failure: HDF5 image indexing requires increasing indices. No diagnostic outputs were written.
  - Recovery patch sorted HDF5 image indices and restored batch order.
- Rerun `8917753` completed in `00:03:59`, exit `0:0`, node `della-l09g5`, batch MaxRSS `36193.50M`, stdout `/src/slurms/c36_geom_diag_8917753.out`, stderr `/src/slurms/c36_geom_diag_8917753.err`.
- Diagnostic output paths:
  - `/src/tables/cycle36_geometry/cycle30_subj05_align0_1sess_150ep_geometry.json`
  - `/src/tables/cycle36_geometry/cycle30_subj05_align_low_1sess_150ep_geometry.json`
  - `/src/tables/cycle36_geometry/cycle30_subj07_align0_1sess_150ep_geometry.json`
  - `/src/tables/cycle36_geometry/cycle30_subj07_align_low_1sess_150ep_geometry.json`
  - `/src/tables/cycle36_geometry/cycle36_geometry_summary.csv`
  - `/src/tables/cycle36_geometry/cycle36_geometry_deltas.json`

Diagnostic provenance:
- Subjects: 5 and 7.
- Split: one-session training shard only, `wds/subj05/train/{0..0}.tar` and `wds/subj07/train/{0..0}.tar`.
- Image count: `536` deduplicated training images per subject/row.
- Teacher geometry: frozen OpenCLIP-bigG image tokens from training images only.
- Predicted geometry: checkpoint predicted CLIP tokens at the MindEye predicted-CLIP boundary.
- No shared1000/new-test labels, no new-test fMRI, and no evaluator tensors were used as topology targets.

Phase 1 diagnostic summary:

| subject | row | space | Spearman RSA | Pearson sim | NN@1 | NN@5 | NN@10 | teacher NN median rank | teacher NN MRR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| subj05 | align0 | flat tokens | 0.297185 | 0.328214 | 0.039179 | 0.144030 | 0.182090 | 28.0 | 0.124715 |
| subj05 | align_low | flat tokens | 0.298296 | 0.322037 | 0.037313 | 0.126866 | 0.158022 | 39.0 | 0.107800 |
| subj05 | align0 | mean pooled | 0.140196 | 0.164560 | 0.091418 | 0.154851 | 0.181530 | 33.0 | 0.162072 |
| subj05 | align_low | mean pooled | 0.094632 | 0.122425 | 0.070896 | 0.144403 | 0.174813 | 32.0 | 0.143994 |
| subj07 | align0 | flat tokens | 0.284901 | 0.310573 | 0.065299 | 0.141791 | 0.171642 | 28.0 | 0.141610 |
| subj07 | align_low | flat tokens | 0.296362 | 0.319569 | 0.055970 | 0.111567 | 0.154851 | 31.5 | 0.126792 |
| subj07 | align0 | mean pooled | 0.143948 | 0.164275 | 0.059701 | 0.161194 | 0.177052 | 36.0 | 0.147676 |
| subj07 | align_low | mean pooled | 0.167913 | 0.191008 | 0.055970 | 0.147015 | 0.172201 | 32.0 | 0.136682 |

Diagnostic deltas, `align_low - align0`:
- subj05 flat tokens: NN@5 `-0.017164`, NN@10 `-0.024067`, MRR `-0.016914`, median rank worsened by `+11.0`; Spearman was essentially unchanged at `+0.001111`.
- subj05 mean pooled: Spearman `-0.045564`, Pearson `-0.042135`, NN@1 `-0.020522`, NN@5 `-0.010448`, MRR `-0.018077`.
- subj07 flat tokens: NN@1 `-0.009328`, NN@5 `-0.030224`, NN@10 `-0.016791`, MRR `-0.014818`, median rank worsened by `+3.5`; Spearman/Pearson rose slightly.
- subj07 mean pooled: NN@5 `-0.014179`, NN@10 `-0.004851`, MRR `-0.010994`; Spearman/Pearson rose slightly.

Phase 1 decision:
- Go for Phase 2. Although global Spearman/Pearson are mixed, the local/rank metrics most tied to retrieval degrade in both subjects and both representation spaces. This supports the intended mechanism: Cycle 30 global moment matching damaged local stimulus-neighborhood geometry.

Phase 2 smoke:
- Submitted `sbatch /src/cycle36_topo_smoke.slurm` -> `8918126_[0-1]`.
- `8918126_0` / `cycle36_smoke_subj07_topo0_1sess_3ep`: completed `0:0`, elapsed `00:06:31`, node `della-l02g14`, batch MaxRSS `21591404K`, stdout `/src/slurms/c36_topo_smoke_8918126_0.out`, stderr `/src/slurms/c36_topo_smoke_8918126_0.err`, checkpoint `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle36_smoke_subj07_topo0_1sess_3ep/last.pth`.
- `8918126_1` / `cycle36_smoke_subj07_topo_low_1sess_3ep`: completed `0:0`, elapsed `00:05:04`, node `della-l09g5`, batch MaxRSS `22893496K`, stdout `/src/slurms/c36_topo_smoke_8918126_1.out`, stderr `/src/slurms/c36_topo_smoke_8918126_1.err`, checkpoint `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle36_smoke_subj07_topo_low_1sess_3ep/last.pth`.
- Smoke final logged checks:
  - `topo0`: final train/test loss `13.4`, blurry PixCorr train/test `0.286/0.196`, train fwd/bwd `0.427/0.267` in the progress-bar order for test and `0.966` bwd train; `train/clip_topo_loss=4.59`, `train/clip_topo_loss_scaled=0`, test topology loss `2.53`, test NN@5 `0.094`, no shape/device failures.
  - `topo_low`: final train/test loss `13.4`, blurry PixCorr train/test `0.286/0.196`, test fwd/bwd `0.427/0.267`, `train/clip_topo_loss=4.50`, `train/clip_topo_loss_scaled=0.00450`, test topology loss `2.43`, test NN@5 `0.096`, no shape/device failures.
  - `topo0` zero-control scaled contribution was exactly zero. `topo_low` had finite low-magnitude topology contribution. Memory stayed within the prior safe range.

Full Phase 2 jobs launched:
- Submitted `sbatch /src/cycle36_topo_train_s57.slurm` -> `8918443_[0-3]`, pending at write time:
  - `8918443_0`: `cycle36_subj05_topo0_1sess_150ep`
  - `8918443_1`: `cycle36_subj05_topo_low_1sess_150ep`
  - `8918443_2`: `cycle36_subj07_topo0_1sess_150ep`
  - `8918443_3`: `cycle36_subj07_topo_low_1sess_150ep`
- Submitted dependent evaluator `sbatch --dependency=afterok:8918443 /src/cycle36_topo_eval_s57.slurm` -> `8918444_[0-3]`, pending on dependency at write time.
- Full expected checkpoints: `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle36_subj0{5,7}_topo{0,_low}_1sess_150ep/last.pth`.
- Full expected enhanced tensors: `/src/evals/cycle36_subj0{5,7}_topo{0,_low}_1sess_150ep/cycle36_subj0{5,7}_topo{0,_low}_1sess_150ep_all_enhancedrecons.pt`.
- Full expected CSVs: `/src/tables/cycle36_subj0{5,7}_topo{0,_low}_1sess_150ep_all_enhancedrecons.csv`.

Current status and next checks:
- Training array `8918443_[0-3]` is pending for priority with `04:30:00`, `64G`, one A100 per row.
- Evaluator array `8918444_[0-3]` is pending on `afterok:8918443` with `04:00:00`, `64G`, one A100 per row.
- Next cycle should parse `8918443` training logs for final train/test loss, blurry PixCorr, fwd/bwd retrieval, topology loss/scaled loss, NN overlap, rank/MRR diagnostics, MaxRSS, checkpoints, then let or recover `8918444` and compute the required full metric table plus `topo_low - topo0` deltas.

## Cycle 37 - 2026-05-29 chronological addendum

- Detailed Cycle 37 artifact authentication and tables were recorded above in this file after the earlier pending-evaluator section.
- Final decision for Cycle 37: no new jobs launched; arrays `8918443_[0-3]` and `8918444_[0-3]` completed cleanly; all enhanced tensors and CSVs were authenticated.
- Primary result: `topo_low - topo0` BrainRet failed on both protected subjects: subj05 `-0.027111`, subj07 `-0.049556`. ImageRet was preserved, but higher-visual and several perceptual metrics regressed.
- Conclusion: close this exact full-pairwise batch-local cosine-MSE topology loss as mechanistically informative but practically insufficient. The only adjacent next branch worth considering is a sharper local-neighborhood objective on subjects 5 and 7, such as teacher-neighbor-weighted pairwise loss or soft nearest-neighbor KL.

## Cycle 38 - 2026-05-29

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- `/job-status.md` did not produce active-job content in this container, and `squeue -u $USER` initially showed no active Slurm jobs.

Code/config changes:
- Added `/src/cycle38_neighbor_diagnostics.py` for Phase 1 post-hoc Cycle 36 diagnostics. It recomputes frozen OpenCLIP-bigG image features for the 1000 held-out evaluator images, reads saved Cycle 36 `all_clipvoxels`, enhanced reconstructions, blurry reconstructions, and new-test image IDs, and writes per-image rank/PixCorr/teacher-neighbor diagnostics under `/src/tables/cycle38_neighbor_diagnostics/`.
- Patched `/src/Train.py` with disabled-by-default sparse teacher-neighbor topology flags:
  - `--use_clip_neighbor_topology`, default `False`
  - `--clip_neighbor_weight`, default `0.0`
  - `--clip_neighbor_k`, default `5`
  - `--clip_neighbor_teacher_temp`, default `0.07`
  - `--clip_neighbor_student_temp`, default `0.07`
  - `--clip_neighbor_pool`, default `flat`, matching the Cycle 36 flattened predicted/image CLIP representation.
- The sparse-neighbor loss normalizes predicted CLIP and frozen image-CLIP batch features, excludes self-pairs, selects each sample's top-`k` frozen image-CLIP teacher neighbors within the current training batch, and applies cross-entropy from the frozen teacher top-`k` distribution to the student distribution over the same candidates. The term is added only when `--use_clip_neighbor_topology` is set.
- Added explicit provenance logging for the training loss: `training_only=True`, `shared1000_or_new_test_used=False`, and `test_sources_used=[]`.
- Added logs for raw/scaled neighbor loss, teacher top-k, teacher/student temperatures, NN@1/5/10 overlap, teacher-top1 median rank, and MRR.
- Added `/src/cycle38_neighbor_diagnostic.slurm`, `/src/cycle38_neighbor_smoke.slurm`, `/src/cycle38_neighbor_train_s57.slurm`, and `/src/cycle38_neighbor_eval_s57.slurm`.

Validation:
- `/src/fmri/bin/python -m py_compile /src/Train.py /src/cycle38_neighbor_diagnostics.py` passed.
- `bash -n /src/cycle38_neighbor_diagnostic.slurm /src/cycle38_neighbor_smoke.slurm /src/cycle38_neighbor_train_s57.slurm /src/cycle38_neighbor_eval_s57.slurm` passed.

Phase 1 Cycle 36 diagnostics:
- Submitted `sbatch /src/cycle38_neighbor_diagnostic.slurm` -> job `8935815`.
- Job `8935815` completed `0:0`, elapsed `00:04:01`, node `della-l09g5`, requested `64G`, time limit `01:00:00`, batch MaxRSS `21176740K`, stdout `/src/slurms/c38_neighbor_diag_8935815.out`, stderr `/src/slurms/c38_neighbor_diag_8935815.err`.
- Diagnostic outputs:
  - `/src/tables/cycle38_neighbor_diagnostics/all_images_openclip_bigG_flat_norm.pt`
  - `/src/tables/cycle38_neighbor_diagnostics/subj05_topo_low_vs_topo0_per_image.csv`
  - `/src/tables/cycle38_neighbor_diagnostics/subj07_topo_low_vs_topo0_per_image.csv`
  - `/src/tables/cycle38_neighbor_diagnostics/cycle38_neighbor_diagnostics_summary.json`
- Subject 5 deterministic per-image diagnostic, `topo_low - topo0`: mean brain-rank delta `-0.695`, median `0`, brain-rank worsened for `24.4%` of images and improved for `24.3%`; mean image-rank delta `+2.110`, median `0`; `119` images preserved or improved image rank while worsening brain rank; mean teacher-top1 rank delta `-9.541`; mean teacher-neighbor overlap@5 delta `+0.0078`; mean PixCorr delta `+0.004930`.
- Subject 7 deterministic per-image diagnostic, `topo_low - topo0`: mean brain-rank delta `-0.283`, median `0`, brain-rank worsened for `20.5%` of images and improved for `22.4%`; mean image-rank delta `+4.587`, median `0`; `80` images preserved or improved image rank while worsening brain rank; mean teacher-top1 rank delta `-14.755`; mean teacher-neighbor overlap@5 delta `+0.0020`; mean PixCorr delta `+0.003044`.
- Worst brain-rank regression examples:
  - subj05: eval/image IDs `(52, 5602, +103 brain rank, +9 image rank)`, `(878, 64096, +70, +121)`, `(136, 11635, +67, +86)`, `(726, 53052, +65, +18)`, `(316, 25091, +65, +4)`.
  - subj07: eval/image IDs `(32, 4667, +218 brain rank, +5 image rank)`, `(858, 62275, +150, +332)`, `(850, 61801, +138, +126)`, `(571, 42946, +98, +29)`, `(170, 14179, +89, +75)`.
- Interpretation: deterministic full-pool ranks are mixed around a median of zero, but BrainRet regressions are not confined to a single tiny subset: roughly one-fifth to one-quarter of images worsen, and there is a stress subset where ImageRet is preserved or improved while BrainRet worsens. Teacher-neighbor rank/overlap diagnostics improve on average, matching Cycle 36's mechanism mismatch. Proceeded to Phase 2.
- Limitation: per-image VC/V1-V4/HigherVis correlations are not saved by `final_evaluations.py`; recovering them would require rerunning the GNet encoder and storing voxelwise/per-image correlations. The current diagnostic records PixCorr and CLIP-rank/neighborhood diagnostics.

Phase 2 sparse-neighbor smoke:
- Submitted `sbatch /src/cycle38_neighbor_smoke.slurm` -> array `8935931_[0-1]`.
- `8935931_0` / `cycle38_smoke_subj07_neighbor0_1sess_3ep`: completed `0:0`, elapsed `00:05:18`, node `della-l09g5`, requested `64G`, time limit `01:00:00`, batch MaxRSS `21449752K`, stdout `/src/slurms/c38_neighbor_smoke_8935931_0.out`, stderr `/src/slurms/c38_neighbor_smoke_8935931_0.err`.
- `8935931_1` / `cycle38_smoke_subj07_neighbor_low_1sess_3ep`: completed `0:0`, elapsed `00:05:18`, node `della-l09g5`, requested `64G`, time limit `01:00:00`, batch MaxRSS `21449704K`, stdout `/src/slurms/c38_neighbor_smoke_8935931_1.out`, stderr `/src/slurms/c38_neighbor_smoke_8935931_1.err`.
- Both smoke rows loaded `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/train_logs/final_multisubject_subj07/last.pth` and logged sparse-neighbor provenance: `training_only=True`, `shared1000_or_new_test_used=False`, `test_sources_used=[]`.
- `neighbor0` final smoke diagnostics: test loss `13.4`, test blurry PixCorr `0.196`, test fwd/bwd retrieval `0.427/0.267`, train neighbor loss `3.07`, train scaled neighbor loss exactly `0`, test neighbor loss `2.11`, test overlap@1/5/10 `0.040/0.094/0.135`, teacher-top1 median rank `41`, MRR `0.102`.
- `neighbor_low` final smoke diagnostics: test loss `13.4`, test blurry PixCorr `0.196`, test fwd/bwd retrieval `0.423/0.267`, train neighbor loss `3.06`, train scaled neighbor loss `0.00306`, test neighbor loss `2.11`, test overlap@1/5/10 `0.040/0.094/0.136`, teacher-top1 median rank `40`, MRR `0.103`.
- Smoke decision: passed. Zero-control contribution was exactly zero; nonzero row had finite low-magnitude contribution; losses were finite; no shape/device errors occurred; memory remained in the prior safe range.

Full Phase 3 jobs launched:
- Submitted `sbatch /src/cycle38_neighbor_train_s57.slurm` -> `8936092_[0-3]`, requested one A100, `64G`, `04:30:00`.
  - `8936092_0`: `cycle38_subj05_neighbor0_1sess_150ep`
  - `8936092_1`: `cycle38_subj05_neighbor_low_1sess_150ep`
  - `8936092_2`: `cycle38_subj07_neighbor0_1sess_150ep`
  - `8936092_3`: `cycle38_subj07_neighbor_low_1sess_150ep`
- Submitted dependent evaluator `sbatch --dependency=afterok:8936092 /src/cycle38_neighbor_eval_s57.slurm` -> `8936093_[0-3]`, requested one A100, `64G`, `04:00:00`.
- At write time, `8936092_[0-3]` were `PENDING`, and `8936093_[0-3]` were `PENDING (Dependency)`.
- Expected checkpoints: `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle38_subj0{5,7}_neighbor{0,_low}_1sess_150ep/last.pth`.
- Expected enhanced tensors: `/src/evals/cycle38_subj0{5,7}_neighbor{0,_low}_1sess_150ep/cycle38_subj0{5,7}_neighbor{0,_low}_1sess_150ep_all_enhancedrecons.pt`.
- Expected final CSVs: `/src/tables/cycle38_subj0{5,7}_neighbor{0,_low}_1sess_150ep_all_enhancedrecons.csv`.

Conclusions:
- Cycle 36 full-pairwise topology remains closed as a negative primary result. Cycle 38's diagnostic supports the sparse-neighbor branch as the narrow adjacent test: local teacher-neighbor diagnostics improved despite refined BrainRet failure, while the damaging BrainRet changes affect a broad enough image subset to justify a local-rank objective.
- The sparse-neighbor implementation passed syntax checks and the required 3-epoch smoke gate. Full protected subject 5/7 rows and dependent evaluator are queued, not completed.

Recommended next research questions:
- Do `8936092_[0-3]` complete with finite neighbor diagnostics and no operational failures?
- Does `neighbor_low - neighbor0` improve refined BrainRet by about `+0.02` in at least one weak subject while preserving ImageRet, CLIP, Inception, VC, HigherVis, and lower-is-better EfficientNet/SwAV distances?
- If refined BrainRet still fails despite improved neighbor diagnostics, close this sparse local-neighborhood family rather than widening to weight grids or architecture changes.

## Cycle 38 recovery/readout - 2026-06-05

Plan source:
- Read and executed `/plan.md` only. Telegram report is due.
- Scope was recovery, authentication, and interpretation of the four protected sparse teacher-neighbor rows: `cycle38_subj05_neighbor0_1sess_150ep`, `cycle38_subj05_neighbor_low_1sess_150ep`, `cycle38_subj07_neighbor0_1sess_150ep`, and `cycle38_subj07_neighbor_low_1sess_150ep`.
- No new mechanism, retraining, evaluator rerun, subject broadening, topology weight grid, refiner/generator change, or retrieval-pool change was launched.

Code/config changes:
- None.

Commands run:
- `sacct -j 8936092,8936093 --format=JobIDRaw,JobID,JobName%45,State,ExitCode,Elapsed,NodeList,ReqMem,MaxRSS,StdOut,StdErr -P`
- Inspected `/src/slurms/c38_neighbor_s57_8936092_{0,1,2,3}.out/.err`.
- Inspected `/src/slurms/c38_neighbor_eval_s57_8936093_{0,1,2,3}.out/.err`.
- Loaded `/src/evals/cycle38_subj0{5,7}_neighbor{0,_low}_1sess_150ep/*_all_enhancedrecons.pt`.
- Parsed `/src/tables/cycle38_subj0{5,7}_neighbor{0,_low}_1sess_150ep_all_enhancedrecons.csv`.

Scheduler/accounting readout:
- Training array `8936092_[0-3]` completed cleanly:
  - `8936092_0` / `cycle38_subj05_neighbor0_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:08:47`, node `della-l02g9`, requested `64G`, batch MaxRSS `21470840K`, stdout `/src/slurms/c38_neighbor_s57_8936092_0.out`, stderr `/src/slurms/c38_neighbor_s57_8936092_0.err`.
  - `8936092_1` / `cycle38_subj05_neighbor_low_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:08:48`, node `della-l02g3`, requested `64G`, batch MaxRSS `23067088K`, stdout `/src/slurms/c38_neighbor_s57_8936092_1.out`, stderr `/src/slurms/c38_neighbor_s57_8936092_1.err`.
  - `8936092_2` / `cycle38_subj07_neighbor0_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:07:56`, node `della-l01g16`, requested `64G`, batch MaxRSS `21598284K`, stdout `/src/slurms/c38_neighbor_s57_8936092_2.out`, stderr `/src/slurms/c38_neighbor_s57_8936092_2.err`.
  - `8936092_3` / `cycle38_subj07_neighbor_low_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:07:43`, node `della-l02g12`, requested `64G`, batch MaxRSS `21454696K`, stdout `/src/slurms/c38_neighbor_s57_8936092_3.out`, stderr `/src/slurms/c38_neighbor_s57_8936092_3.err`.
- Enhanced evaluator array `8936093_[0-3]` completed cleanly:
  - `8936093_0` / `cycle38_subj05_neighbor0_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:29:18`, node `della-l03g7`, requested `64G`, batch MaxRSS `49465248K`, stdout `/src/slurms/c38_neighbor_eval_s57_8936093_0.out`, stderr `/src/slurms/c38_neighbor_eval_s57_8936093_0.err`.
  - `8936093_1` / `cycle38_subj05_neighbor_low_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:26:49`, node `della-l01g14`, requested `64G`, batch MaxRSS `49259544K`, stdout `/src/slurms/c38_neighbor_eval_s57_8936093_1.out`, stderr `/src/slurms/c38_neighbor_eval_s57_8936093_1.err`.
  - `8936093_2` / `cycle38_subj07_neighbor0_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:27:41`, node `della-l02g10`, requested `64G`, batch MaxRSS `49017780K`, stdout `/src/slurms/c38_neighbor_eval_s57_8936093_2.out`, stderr `/src/slurms/c38_neighbor_eval_s57_8936093_2.err`.
  - `8936093_3` / `cycle38_subj07_neighbor_low_1sess_150ep`: `COMPLETED`, exit `0:0`, elapsed `02:31:24`, node `della-l03g7`, requested `64G`, batch MaxRSS `46256976K`, stdout `/src/slurms/c38_neighbor_eval_s57_8936093_3.out`, stderr `/src/slurms/c38_neighbor_eval_s57_8936093_3.err`.

Checkpoint/artifact authentication:
- Training logs show each row loaded the official multisubject initialization from `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/train_logs/final_multisubject_subj0{5,7}/last.pth` and repeatedly saved `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/<model_name>/last`.
- The interactive container cannot currently stat `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/.../last.pth` or `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/train_logs/.../last.pth`; this is an interactive observability gap for checkpoint byte/load verification after completion. It is partially mitigated by the clean dependent evaluator completion and logs showing successful checkpoint use on compute.
- All four training rows logged sparse-neighbor provenance exactly as required: `training_only=True`, `shared1000_or_new_test_used=False`, `test_sources_used=[]`.
- Enhanced tensors exist and load as finite `torch.float32` tensors:
  - `cycle38_subj05_neighbor0_1sess_150ep`: `/src/evals/cycle38_subj05_neighbor0_1sess_150ep/cycle38_subj05_neighbor0_1sess_150ep_all_enhancedrecons.pt`, shape `(1000, 3, 256, 256)`, min `0.0`, max `1.0`, mean `0.515476`.
  - `cycle38_subj05_neighbor_low_1sess_150ep`: `/src/evals/cycle38_subj05_neighbor_low_1sess_150ep/cycle38_subj05_neighbor_low_1sess_150ep_all_enhancedrecons.pt`, shape `(1000, 3, 256, 256)`, min `0.0`, max `1.0`, mean `0.511789`.
  - `cycle38_subj07_neighbor0_1sess_150ep`: `/src/evals/cycle38_subj07_neighbor0_1sess_150ep/cycle38_subj07_neighbor0_1sess_150ep_all_enhancedrecons.pt`, shape `(1000, 3, 256, 256)`, min `0.0`, max `1.0`, mean `0.507720`.
  - `cycle38_subj07_neighbor_low_1sess_150ep`: `/src/evals/cycle38_subj07_neighbor_low_1sess_150ep/cycle38_subj07_neighbor_low_1sess_150ep_all_enhancedrecons.pt`, shape `(1000, 3, 256, 256)`, min `0.0`, max `1.0`, mean `0.526583`.
- Evaluator logs confirmed `final_evaluations.py` consumed enhanced tensors via the expected `all_recons_path=evals/<model_name>/<model_name>_all_enhancedrecons.pt` line for all four rows.
- Final CSVs exist under `/src/tables` for all four rows.

Final training diagnostics from completed progress logs:

| row | test loss | test blurry PixCorr | test fwd | test bwd | train blurry PixCorr | train bwd | train neighbor loss | train scaled neighbor loss | test neighbor loss | k | test NN@1 | test NN@5 | test NN@10 | test teacher-top1 median rank | test MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cycle38_subj05_neighbor0_1sess_150ep` | 14.3 | 0.193 | 0.670 | 0.583 | 0.801 | 1.000 | 2.46 | 0.00000 | 2.20 | 5 | 0.103 | 0.170 | 0.226 | 20 | 0.195 |
| `cycle38_subj05_neighbor_low_1sess_150ep` | 14.5 | 0.181 | 0.657 | 0.537 | 0.809 | 1.000 | 2.25 | 0.00225 | 1.99 | 5 | 0.113 | 0.176 | 0.229 | 18 | 0.207 |
| `cycle38_subj07_neighbor0_1sess_150ep` | 14.2 | 0.226 | 0.730 | 0.563 | 0.788 | 1.000 | 2.48 | 0.00000 | 2.17 | 5 | 0.050 | 0.130 | 0.160 | 39 | 0.118 |
| `cycle38_subj07_neighbor_low_1sess_150ep` | 14.4 | 0.230 | 0.723 | 0.527 | 0.788 | 1.000 | 2.29 | 0.00229 | 2.00 | 5 | 0.063 | 0.132 | 0.171 | 38 | 0.133 |

Training diagnostic interpretation:
- `neighbor0` scaled neighbor contribution is exactly zero in both subjects.
- `neighbor_low` has finite low-magnitude scaled contribution, `0.00225-0.00229`.
- The sparse-neighbor objective moved the intended diagnostics in the expected direction: lower raw neighbor loss, modestly higher test NN overlap, better teacher-top1 median rank, and higher MRR in both subjects.
- The progress display did not expose final `train/loss` or train forward retrieval in the retained 1200-character tqdm line; final test loss, train blurry PixCorr, train bwd retrieval, and neighbor diagnostics were available.

Refined enhanced-evaluator metric table:

| row | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist ↓ | SwAV dist ↓ | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cycle38_subj05_neighbor0_1sess_150ep` | 0.192674 | 0.406933 | 0.846794 | 0.916086 | 0.856197 | 0.840533 | 0.768113 | 0.431212 | 0.648778 | 0.552778 | 0.416381 | 0.351920 | 0.355988 | 0.341006 | 0.315086 | 0.423627 |
| `cycle38_subj05_neighbor_low_1sess_150ep` | 0.182719 | 0.408793 | 0.842659 | 0.915971 | 0.855814 | 0.841298 | 0.768952 | 0.435982 | 0.646667 | 0.533000 | 0.414058 | 0.349629 | 0.352020 | 0.336031 | 0.308302 | 0.421438 |
| `cycle38_subj07_neighbor0_1sess_150ep` | 0.194279 | 0.404695 | 0.829048 | 0.894047 | 0.785077 | 0.770978 | 0.828250 | 0.477893 | 0.680556 | 0.542000 | 0.321338 | 0.321689 | 0.324969 | 0.316007 | 0.281305 | 0.306982 |
| `cycle38_subj07_neighbor_low_1sess_150ep` | 0.201270 | 0.404336 | 0.819063 | 0.884962 | 0.782269 | 0.782230 | 0.826863 | 0.472738 | 0.684333 | 0.507222 | 0.318243 | 0.312592 | 0.312030 | 0.306041 | 0.273007 | 0.305419 |

Same-subject `neighbor_low - neighbor0` deltas:

| subject | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet dist ↓ | SwAV dist ↓ | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | -0.009955 | +0.001860 | -0.004135 | -0.000115 | -0.000383 | +0.000766 | +0.000840 | +0.004770 | -0.002111 | -0.019778 | -0.002323 | -0.002291 | -0.003967 | -0.004975 | -0.006785 | -0.002189 |
| subj07 | +0.006991 | -0.000359 | -0.009985 | -0.009085 | -0.002808 | +0.011252 | -0.001386 | -0.005154 | +0.003778 | -0.034778 | -0.003095 | -0.009097 | -0.012939 | -0.009966 | -0.008298 | -0.001563 |

Interpretation against success criteria:
- Subject 5 fails. BrainRet moved `-0.019778` rather than the required about `+0.02`. ImageRet was nearly preserved at `-0.002111`, CLIP slightly improved, and SSIM slightly improved, but PixCorr, VC, HigherVis, AlexNet, and lower-is-better EffNet/SwAV worsened.
- Subject 7 fails. BrainRet moved `-0.034778`. ImageRet improved slightly at `+0.003778`, CLIP improved by `+0.011252`, and lower-is-better EffNet/SwAV improved, but Inception, AlexNet, VC, V1-V4, and HigherVis regressed.
- The sparse-neighbor training diagnostics improved as intended, so the implementation appears to apply the local objective and alter predicted-CLIP neighborhood behavior. That did not transfer to the protected refined BrainRet outcome.

Decision:
- Neither subject passes. Because neighbor diagnostics moved in the intended direction while refined BrainRet failed in both subjects, close this sparse local-neighborhood family as mechanistically informative but practically insufficient.
- Do not run topology weight grids, stronger topology losses, adapters plus topology, ROI routing, routers, MoE, CLIP-layer fusion, generator/refiner edits, captions, VLM correction, temporal decoding, hard voxel pruning, reliability-prior revival, or subject 1/2 scale-up from this result.
- No rerun is needed for these four protected rows: scheduler, evaluator, enhanced tensor, CSV, and metric-path authentication are complete. The only residual caveat is direct interactive checkpoint stat/load visibility after jobs completed.

Recommended next research questions:
- What non-topology mechanism can target weak-subject BrainRet without relying on predicted-CLIP local geometry, given that both full-pairwise topology and sparse teacher-neighbor topology improved diagnostics but failed refined BrainRet?
- Are the BrainRet regressions caused by reconstruction/refiner interactions downstream of predicted CLIP rather than by predicted-CLIP retrieval itself?
- Should future plans prioritize post-hoc error decomposition of refined BrainRet versus ImageRet/CLIP before any new training branch?

Telegram-ready update:
Cycle 38 recovery/readout is complete. Training array `8936092_[0-3]` and enhanced evaluator array `8936093_[0-3]` all completed with exit `0:0`; training rows ran about 2:08 with MaxRSS 21.5-23.1 GB, evaluator rows ran 2:26-2:31 with MaxRSS 46.3-49.5 GB. All enhanced tensors exist at `/src/evals/cycle38_subj0{5,7}_neighbor{0,_low}_1sess_150ep/*_all_enhancedrecons.pt`, load as finite `(1000,3,256,256)` float32 tensors, and `final_evaluations.py` consumed the enhanced path. Result is negative: subj05 `neighbor_low - neighbor0` BrainRet `-0.019778`, ImageRet `-0.002111`, HigherVis `-0.002189`; subj07 BrainRet `-0.034778`, ImageRet `+0.003778`, HigherVis `-0.001563`. The sparse-neighbor diagnostics improved in training, but protected refined BrainRet failed in both weak subjects. Close the sparse local-neighborhood topology family; do not run weight grids or scale to subjects 1/2.

## Cycle 39 - 2026-06-05

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- Scope was post-hoc failure decomposition of the completed Cycle 38 sparse teacher-neighbor rows only. No training, topology branch, adapter, routing, generator/refiner edit, caption/VLM branch, reliability branch, or subject 1/2 scale-up was launched.

Preflight:
- `/job-status.md` is absent in this container: exact missing path `/job-status.md`. `squeue -u $USER` showed no active jobs at both start and end of the cycle.
- The four Cycle 38 directories under `/src/evals` exist:
  - `/src/evals/cycle38_subj05_neighbor0_1sess_150ep`
  - `/src/evals/cycle38_subj05_neighbor_low_1sess_150ep`
  - `/src/evals/cycle38_subj07_neighbor0_1sess_150ep`
  - `/src/evals/cycle38_subj07_neighbor_low_1sess_150ep`
- For each row, evaluator tensors `all_clipvoxels`, `all_recons`, `all_enhancedrecons`, `all_blurryrecons`, and `all_predcaptions` exist. Final CSVs exist under `/src/tables`.
- Local load/finite checks confirmed, for example, subject 5 neighbor0: `all_clipvoxels` shape `(1000,256,1664)` `torch.float16`, finite; `all_recons`, `all_blurryrecons`, and `all_enhancedrecons` shape `(1000,3,256,256)` `torch.float32`, finite. The subject 7 resumed diagnostic JSON contains the same authenticated shape/dtype/finite checks for both subject 7 rows.

Code/config changes:
- Added `/src/cycle39_failure_decomposition.py`.
  - Reads completed Cycle 38 artifacts only.
  - Computes deterministic full-pool direct predicted-CLIP ranks from saved `all_clipvoxels`.
  - Computes per-image PixCorr for `all_blurryrecons`, `all_recons`, raw `all_enhancedrecons`, and `enhanced_evalmix` matching the final evaluator's `0.75*enhanced + 0.25*blurry` image.
  - Writes per-image CSVs and JSON summaries under `/src/tables/cycle39_failure_decomposition/`.
  - Includes a cacheable OpenCLIP stage-rank path, but also a fallback `--skip_stage_clip_features` mode used here because visible `/src` Slurm jobs fail before Python starts.
- Added `/src/cycle39_failure_decomposition.slurm`, a 1-hour diagnostic wrapper. The `/src` Slurm launch path failed before logs; see jobs below.

Validation:
- `/src/fmri/bin/python -m py_compile /src/cycle39_failure_decomposition.py` passed.
- `bash -n /src/cycle39_failure_decomposition.slurm` passed.

Commands/jobs launched:
- `sbatch /src/cycle39_failure_decomposition.slurm` -> job `9238576`, requested one A100, `64G`, `01:00:00`. It failed immediately before Python/log creation: `FAILED`, exit `0:53`, elapsed `00:00:00`, node `della-l04g9`, stdout `/src/slurms/c39_failure_decomp_9238576.out`, stderr `/src/slurms/c39_failure_decomp_9238576.err`, no MaxRSS.
- `/src` path smoke `9238599` also failed immediately: `FAILED`, exit `0:53`, elapsed `00:00:01`, node `della-l09g7`, requested `4G`, no MaxRSS. This isolates the failure to Slurm launch/path visibility, not the diagnostic Python.
- Historical scratch-path smoke `9238612` completed: `COMPLETED`, exit `0:0`, elapsed `00:00:01`, node `della-l09g7`, requested `4G`, batch MaxRSS `960K`. Its logs are on compute-visible `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/...`, which remains invisible from the interactive shell; this matches earlier checkpoint/log observability gaps.
- Local fallback runs:
  - `/src/fmri/bin/python /src/cycle39_failure_decomposition.py --data_path=/src --outdir=/src/tables/cycle39_failure_decomposition --skip_stage_clip_features --topn=20`
  - The first local fallback wrote subject 5 per-image output then exited before final JSON.
  - Resumed subject 7 with `--subjects 7`, completed and wrote the subject 7 summary JSON.
  - Combined both per-subject CSVs into `/src/tables/cycle39_failure_decomposition/cycle39_failure_decomposition_combined_fallback_summary.json`.

Output artifacts:
- `/src/tables/cycle39_failure_decomposition/all_images_openclip_bigG_flat_norm.pt`, copied/reused from Cycle 38 teacher cache, `1.6G`.
- `/src/tables/cycle39_failure_decomposition/subj05_neighbor_low_vs_neighbor0_per_image.csv`, `1000` data rows.
- `/src/tables/cycle39_failure_decomposition/subj07_neighbor_low_vs_neighbor0_per_image.csv`, `1000` data rows.
- `/src/tables/cycle39_failure_decomposition/cycle39_failure_decomposition_combined_fallback_summary.json`, combined fallback summary.
- `/src/tables/cycle39_failure_decomposition/cycle39_failure_decomposition_summary.json`, subject 7 resumed-run JSON.

Direct predicted-CLIP full-pool retrieval, `neighbor_low - neighbor0` rank deltas; positive rank delta means worse:

| subject | image-rank mean | image-rank median | image worse frac | image improved frac | brain-rank mean | brain-rank median | brain worse frac | brain improved frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | +2.431 | 0.0 | 0.362 | 0.231 | +0.268 | 0.0 | 0.245 | 0.225 |
| subj07 | +2.156 | 0.0 | 0.352 | 0.253 | +0.339 | 0.0 | 0.207 | 0.228 |

Worst direct predicted-CLIP brain-rank regressions:
- subj05 top examples `(eval_index, nsd_image_id, brain_rank_delta, image_rank_delta, enhanced_evalmix_pixcorr_delta)`: `(52,5602,+178,+36,-0.142865)`, `(394,29663,+171,+194,-0.032749)`, `(229,19181,+135,+8,+0.052862)`, `(906,65872,+84,+6,-0.047649)`, `(628,46136,+76,+5,+0.051395)`.
- subj07 top examples: `(950,69030,+250,+256,+0.012450)`, `(179,14820,+135,+14,-0.032854)`, `(170,14179,+135,+34,+0.056189)`, `(637,46480,+110,+98,-0.016474)`, `(234,19573,+86,+38,+0.445446)`.

Stage-wise PixCorr diagnostics, `neighbor_low - neighbor0`:

| subject | blurry mean | recons mean | enhanced raw mean | enhanced evalmix mean | enhanced evalmix median | enhanced evalmix improved frac |
|---|---:|---:|---:|---:|---:|---:|
| subj05 | -0.003489 | -0.010272 | -0.010465 | -0.009919 | -0.004098 | 0.485 |
| subj07 | -0.002091 | +0.007968 | +0.008202 | +0.006932 | +0.004238 | 0.513 |

Stress subsets, using deterministic direct predicted-CLIP brain rank as the per-image BrainRet proxy:

| subject | predclip/image rank preserved while brain rank worsened | enhanced-evalmix PixCorr improved while brain rank worsened | brain rank improved despite predclip/image rank worsening |
|---|---:|---:|---:|
| subj05 | 108 / 1000, mean brain-rank delta `+3.657` | 123 / 1000, mean brain-rank delta `+7.894`, mean PixCorr delta `+0.079212` | 75 / 1000, mean brain-rank delta `-5.760` |
| subj07 | 75 / 1000, mean brain-rank delta `+6.800` | 107 / 1000, mean brain-rank delta `+12.598`, mean PixCorr delta `+0.106404` | 86 / 1000, mean brain-rank delta `-7.849` |

Aggregate final CSV deltas retained from Cycle 38:
- subj05: BrainRet `-0.019778`, ImageRet `-0.002111`, CLIP `+0.000766`, PixCorr `-0.009955`, VC `-0.002323`, HigherVis `-0.002189`.
- subj07: BrainRet `-0.034778`, ImageRet `+0.003778`, CLIP `+0.011252`, PixCorr `+0.006991`, VC `-0.003095`, HigherVis `-0.001563`.

ROI feasibility:
- Per-image ROI localization is not available from saved Cycle 38 artifacts. `final_evaluations.py` uses `GNet8_Encoder`, computes ROI Pearson correlations, averages across voxels, and writes only aggregate CSV values for `nsd_general`, `V1`, `V2`, `V3`, `V4`, and `higher_vis`.
- Extensive evaluator surgery was not performed in this cycle. Aggregate ROI evidence remains: both subjects regress in VC and HigherVis; subject 7 additionally shows broader V1-V4 regression.

Interpretation:
- The failure is at least partly upstream at the saved predicted-CLIP boundary: full-pool direct predicted-CLIP brain-rank mean worsened in both subjects, and 20-25% of images had worse brain-rank under `neighbor_low`.
- The direct-rank medians are zero and stress subsets show many images where PixCorr or image-side rank improved while brain rank worsened, so this is not a simple uniform collapse. It is a sparse but consequential subject-alignment drift.
- Stage-wise PixCorr does not explain the protected BrainRet failure: subject 7 improves PixCorr at reconstruction/enhancement stages while BrainRet and VC/HigherVis regress; subject 5 worsens PixCorr across stages but also has direct predicted-CLIP brain-rank drift before reconstruction.
- Because stage OpenCLIP rank extraction could not be completed on the visible Slurm path, ImageRet/CLIP per-image stage ranks remain incomplete. The fallback evidence is still sufficient to classify Cycle 38 as upstream predicted-CLIP/subject-alignment drift with downstream low-level metrics unable to rescue or explain the final BrainRet regression.

Conclusion and next research questions:
- Do not revive sparse-neighbor topology, topology grids, or stronger topology loss from this result.
- Future work should prioritize conservative prior-preserving or subject-alignment regularization anchored to the official multisubject initialization, not additional semantic/topology supervision.
- If a future cycle needs per-image stage OpenCLIP ranks, first resolve the Slurm path split by making the compute-visible `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src` artifacts visible from the interactive shell or writing a compute job that copies outputs back to `/src`.

## Cycle 39 execution / Cycle 40 prior-preservation launch - 2026-06-05

Plan source:
- Read and executed `/plan.md` only. Telegram report is not due.
- Scope was the prior-preserving weak-subject adaptation plan for subjects 5 and 7 only. No subject 1/2 jobs, topology/neighborhood losses, adapters, routers, refiner edits, evaluator changes, or weight grid were launched.

Code/config changes:
- Updated `/src/Train.py` with disabled-by-default flags: `--use_prior_preservation`, `--prior_preservation_weight`, `--prior_preservation_target=clipvoxels`, `--prior_preservation_pool=flat|mean`, and `--prior_preservation_detach_anchor`.
- Implemented a training-only frozen anchor from the same official multisubject initialization loaded by the row. The anchor is a frozen copy of the initialized ridge and backbone, with the copied backbone blurry branch disabled for the anchor forward. It runs under `torch.no_grad()` and is outside the optimizer.
- Added cosine-distance drift regularization between current predicted CLIP tokens and frozen-anchor predicted CLIP tokens. The loss is added only when `use_prior_preservation=True` and `prior_preservation_weight > 0`.
- Added compact per-epoch prior logging: weight, raw loss, scaled loss, anchor/current representation norms, cosine-to-anchor, `training_only=True`, `shared1000_or_new_test_used=False`, and `test_sources_used=[]`.
- Added Slurm scripts: `/src/cycle40_prior_smoke.slurm`, `/src/cycle40_prior_train_s57.slurm`, and `/src/cycle40_prior_eval_s57.slurm`.
- The training scripts use compute-visible `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src` and include a compute-side grep preflight for `use_prior_preservation` to fail fast if scratch-visible `Train.py` is stale.

Validation:
- `/src/fmri/bin/python -m py_compile /src/Train.py` passed after the implementation and after the compact logging patch.
- `bash -n /src/cycle40_prior_smoke.slurm /src/cycle40_prior_train_s57.slurm /src/cycle40_prior_eval_s57.slurm` passed.
- Interactive `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src` remains not stat-visible from this shell, but the Slurm smoke logs confirmed compute-side execution used the updated prior-preservation code.

Smoke jobs:
- Initial smoke `sbatch /src/cycle40_prior_smoke.slurm` -> `9239257_[0-1]`, subject 7, one A100, `64G`, `01:00:00`, node `della-l09g7`, completed `0:0`, elapsed `00:07:09`.
  - `9239257_0` / `cycle40_smoke_subj07_prior0_1sess_3ep`: batch MaxRSS `21944904K`, stdout `/src/slurms/c40_prior_smoke_9239257_0.out`, stderr `/src/slurms/c40_prior_smoke_9239257_0.err`.
  - `9239257_1` / `cycle40_smoke_subj07_prior_low_1sess_3ep`: batch MaxRSS `21522152K`, stdout `/src/slurms/c40_prior_smoke_9239257_1.out`, stderr `/src/slurms/c40_prior_smoke_9239257_1.err`.
  - This run completed operationally, but the required prior diagnostics were not visible because the tqdm postfix truncated the long metric dictionary. It was treated as an operational validation and superseded by the formal smoke below after adding compact logging.
- Formal smoke rerun `sbatch /src/cycle40_prior_smoke.slurm` -> `9239526_[0-1]`, subject 7, one A100, `64G`, `01:00:00`, node `della-l09g7`, completed `0:0`, elapsed `00:05:57`.
  - `9239526_0` / `cycle40_smoke_subj07_prior0_1sess_3ep`: batch MaxRSS `21445140K`, stdout `/src/slurms/c40_prior_smoke_9239526_0.out`, stderr `/src/slurms/c40_prior_smoke_9239526_0.err`.
  - `9239526_1` / `cycle40_smoke_subj07_prior_low_1sess_3ep`: batch MaxRSS `21898208K`, stdout `/src/slurms/c40_prior_smoke_9239526_1.out`, stderr `/src/slurms/c40_prior_smoke_9239526_1.err`.
  - Both rows loaded `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/train_logs/final_multisubject_subj07/last.pth` and saved checkpoints under `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle40_smoke_subj07_prior{0,_low}_1sess_3ep/last.pth`.
  - Final smoke standard diagnostics:
    - `prior0`: test loss `13.4`, test blurry PixCorr `0.196`, test fwd/bwd retrieval `0.427/0.267`, train blurry PixCorr `0.286`, train bwd retrieval `0.966`.
    - `prior_low`: test loss `13.4`, test blurry PixCorr `0.197`, test fwd/bwd retrieval `0.423/0.260`, train blurry PixCorr `0.286`, train bwd retrieval `0.966`.
  - Prior-preservation diagnostics:
    - `prior0` epoch 0/1/2 raw loss `0.119311/0.544825/0.711930`; scaled loss exactly `0/0/0`; anchor norms `232.926/232.909/232.950`; current norms `241.215/336.819/408.361`; cosine `0.880689/0.455175/0.288070`.
    - `prior_low` epoch 0/1/2 raw loss `0.119034/0.535596/0.678878`; scaled loss `0.005952/0.026780/0.033944`; anchor norms `232.926/232.909/232.950`; current norms `241.180/335.491/402.103`; cosine `0.880966/0.464404/0.321122`.
  - Smoke decision: passed. Both rows completed; `prior0` scaled preservation loss was exactly zero; `prior_low` preservation loss was finite and low relative to total test loss; no shape/device/dtype errors occurred; MaxRSS stayed below 30 GB; standard test loss, blurry PixCorr, and fwd/bwd retrieval were finite and close between rows.

Full jobs launched:
- Submitted full training array: `sbatch /src/cycle40_prior_train_s57.slurm` -> `9239769_[0-3]`, one A100, `64G`, `04:30:00`.
  - `9239769_0`: `cycle40_subj05_prior0_1sess_150ep`
  - `9239769_1`: `cycle40_subj05_prior_low_1sess_150ep`
  - `9239769_2`: `cycle40_subj07_prior0_1sess_150ep`
  - `9239769_3`: `cycle40_subj07_prior_low_1sess_150ep`
  - At write time the array was `PENDING`, elapsed `00:00:00`, no node assigned, stdout/stderr `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/slurms/%x_%A_%a.out/.err`.
- Submitted dependent enhanced-evaluator array: `sbatch --dependency=afterok:9239769 /src/cycle40_prior_eval_s57.slurm` -> `9239770_[0-3]`, one A100, `64G`, `04:00:00`.
  - At write time the evaluator array was `PENDING (Dependency)`, elapsed `00:00:00`, no node assigned.
  - Evaluation path remains unchanged: `recon_inference.py -> enhanced_recon_inference.py -> final_evaluations.py` with `--all_recons_path=evals/<model_name>/<model_name>_all_enhancedrecons.pt`.

Expected artifacts:
- Checkpoints: `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle40_subj0{5,7}_prior{0,_low}_1sess_150ep/last.pth`.
- Enhanced tensors: `/src/evals/cycle40_subj0{5,7}_prior{0,_low}_1sess_150ep/cycle40_subj0{5,7}_prior{0,_low}_1sess_150ep_all_enhancedrecons.pt`.
- Final CSVs: `/src/tables/cycle40_subj0{5,7}_prior{0,_low}_1sess_150ep_all_enhancedrecons.csv`.

Current status and next steps:
- The prior-preservation implementation and smoke gate are complete.
- Full protected subject 5/7 training and dependent enhanced evaluation are not yet completed. Immediately after the entry above was drafted, `9239769_0` and `9239769_1` started on `della-l04g8` and `della-l04g7` respectively; `9239769_2` and `9239769_3` remained pending for priority, and `9239770_[0-3]` remained pending on dependency.
- Next cycle should inspect `9239769` and `9239770`, parse logs/artifacts, and only then compute the required final metric table and same-subject deltas.
## 2026-06-05 Cycle 40

Plan source: executed `/plan.md` only. Telegram report is not due.

Code/config changes:
- None. This was a readout-only cycle for completed Cycle 40 prior-preservation artifacts.

Scheduler authentication:
- `/job-status.md` was not present in this workspace, so scheduler state was authenticated with `sacct -j 9239769,9239770 --format=JobIDRaw,JobID,JobName%45,State,ExitCode,Elapsed,NodeList,ReqMem,MaxRSS,StdOut,StdErr -P`.
- Training array `9239769_[0-3]` completed `0:0`: subj05 prior0 `02:25:50` on `della-l04g8`, MaxRSS `21611964K`; subj05 prior_low `02:25:50` on `della-l04g7`, MaxRSS `21614140K`; subj07 prior0 `02:21:53` on `della-l03g12`, MaxRSS `23077004K`; subj07 prior_low `02:25:57` on `della-l02g9`, MaxRSS `22072704K`.
- Enhanced evaluator array `9239770_[0-3]` completed `0:0`: subj05 prior0 `02:29:06` on `della-l02g9`, MaxRSS `49320896K`; subj05 prior_low `02:35:38` on `della-l02g8`, MaxRSS `49150416K`; subj07 prior0 `02:27:23` on `della-l03g4`, MaxRSS `49414328K`; subj07 prior_low `02:26:21` on `della-l05g2`, MaxRSS `49509036K`.
- Logs parsed from `/src/slurms/c40_prior_s57_9239769_{0..3}.{out,err}` and `/src/slurms/c40_prior_eval_s57_9239770_{0..3}.{out,err}`.

Training provenance and diagnostics:
- All four rows loaded official multisubject anchors from `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/src/train_logs/final_multisubject_subj05/last.pth` or `subj07/last.pth` and saved checkpoints under `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/<model_name>/last`.
- Prior-preservation logging confirmed `training_only=True`, `shared1000_or_new_test_used=False`, and `test_sources_used=[]` for all four rows.
- Final training-stream diagnostics: subj05 prior0 test loss `14.3`, test blurry PixCorr `0.193`, test fwd/bwd `0.670/0.583`, train blurry PixCorr `0.801`, train bwd `1.000`; subj05 prior_low test loss `16.7`, test blurry PixCorr `0.198`, test fwd/bwd `0.640/0.0533`, train blurry PixCorr `0.800`, train bwd `1.000`; subj07 prior0 test loss `14.2`, test blurry PixCorr `0.226`, test fwd/bwd `0.730/0.563`, train blurry PixCorr `0.788`, train bwd `1.000`; subj07 prior_low test loss `16.6`, test blurry PixCorr `0.238`, test fwd/bwd `0.687/0.0533`, train blurry PixCorr `0.787`, train bwd `1.000`.
- Final prior-preservation diagnostics: subj05 prior0 epoch 149 raw loss `0.91309720`, scaled loss `0`, anchor/current norm `225.22904/488.00924`, cosine `0.086902799`; subj05 prior_low raw `0.27013382`, scaled `0.013506691`, anchor/current norm `225.22904/319.17898`, cosine `0.72986618`; subj07 prior0 raw `0.89919639`, scaled `0`, anchor/current norm `232.92859/493.71268`, cosine `0.10080361`; subj07 prior_low raw `0.26912806`, scaled `0.013456403`, anchor/current norm `232.92859/321.40566`, cosine `0.73087194`.
- The `prior0` scaled preservation loss was exactly zero in both subjects. The `prior_low` scaled loss was finite and low magnitude, about `0.0135`.

Artifact authentication:
- Enhanced evaluator logs confirm `final_evaluations.py` consumed `all_recons_path=evals/<model_name>/<model_name>_all_enhancedrecons.pt` for all four rows.
- Loaded tensors with `/src/fmri/bin/python`; all are `torch.float32`, shape `(1000, 3, 256, 256)`, finite, min `0.0`, max `1.0`.
- Tensor means: subj05 prior0 `0.515476`, subj05 prior_low `0.515872`, subj07 prior0 `0.507720`, subj07 prior_low `0.520754`.
- CSVs parsed from `/src/tables/cycle40_*_all_enhancedrecons.csv`.

Final protected enhanced-evaluator metrics:

| row | PixCorr | SSIM | Alex2 | Alex5 | Inception | CLIP | EffNet-B ↓ | SwAV ↓ | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cycle40_subj05_prior0_1sess_150ep | 0.192674 | 0.406933 | 0.846794 | 0.916086 | 0.856197 | 0.840533 | 0.768113 | 0.431212 | 0.648778 | 0.552778 | 0.416381 | 0.351920 | 0.355988 | 0.341006 | 0.315086 | 0.423627 |
| cycle40_subj05_prior_low_1sess_150ep | 0.195915 | 0.407137 | 0.848376 | 0.915593 | 0.853101 | 0.839756 | 0.770556 | 0.435869 | 0.636778 | 0.072222 | 0.418819 | 0.350598 | 0.359745 | 0.342304 | 0.318672 | 0.427009 |
| cycle40_subj07_prior0_1sess_150ep | 0.194279 | 0.404695 | 0.829048 | 0.894047 | 0.785077 | 0.770978 | 0.828250 | 0.477893 | 0.680556 | 0.542000 | 0.321338 | 0.321689 | 0.324969 | 0.316007 | 0.281305 | 0.306982 |
| cycle40_subj07_prior_low_1sess_150ep | 0.197544 | 0.399234 | 0.832215 | 0.896614 | 0.781957 | 0.779430 | 0.829530 | 0.477388 | 0.641444 | 0.055889 | 0.323756 | 0.327256 | 0.327405 | 0.317699 | 0.281755 | 0.306575 |

Same-subject protected deltas, `prior_low - prior0`:

| subject | PixCorr | SSIM | Alex2 | Alex5 | Inception | CLIP | EffNet-B ↓ | SwAV ↓ | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | +0.003241 | +0.000204 | +0.001583 | -0.000493 | -0.003096 | -0.000777 | +0.002443 | +0.004657 | -0.012000 | -0.480556 | +0.002438 | -0.001322 | +0.003758 | +0.001298 | +0.003586 | +0.003382 |
| subj07 | +0.003265 | -0.005461 | +0.003167 | +0.002567 | -0.003120 | +0.008452 | +0.001280 | -0.000504 | -0.039111 | -0.486111 | +0.002417 | +0.005567 | +0.002435 | +0.001692 | +0.000450 | -0.000407 |

Decision:
- Fail for both subjects. The success criterion required about `+0.02` absolute BrainRet improvement with ImageRet preserved or improved. Instead, BrainRet collapsed from `0.552778` to `0.072222` for subject 5 and from `0.542000` to `0.055889` for subject 7. ImageRet also worsened by `-0.0120` and `-0.0391`.
- The low-weight anchor penalty successfully increased cosine-to-anchor and reduced current norm drift, but it damaged protected brain-to-image retrieval. Image-side metrics, PixCorr, and ROI correlations are diagnostics only and do not override the BrainRet failure.

Conclusions:
- Close this specific predicted-CLIP anchor-preservation setting.
- Per `/plan.md`, do not widen into a prior-weight grid, topology revival, adapters, routers, MoE, CLIP-layer fusion, generator/refiner edits, caption/VLM correction, temporal decoding, hard voxel pruning, or reliability-prior revival as an immediate continuation from this result.
- Next valid research question should return to interpreting why preserving anchor cosine destroys backward retrieval while leaving image-side evaluator metrics nearly unchanged, using existing artifacts only unless a new plan explicitly authorizes new experiments.

## 2026-06-05 Cycle 41

Plan source: executed `/plan.md` only. Telegram report is not due.

Scope:
- Diagnostic-only readout of completed Cycle 40 prior-preservation artifacts.
- No training branch, prior-weight grid, topology/neighbor revival, adapter, router/MoE, ROI-routing model, CLIP-layer fusion, generator/refiner edit, caption/VLM correction, temporal decoding, hard voxel pruning, reliability-prior experiment, or subject 1/2 scale-up was launched.

Scheduler/path preflight:
- `/job-status.md` was absent in this workspace.
- `squeue -u $USER` showed no active jobs at start; the diagnostic script also recorded `active_job_count=0`.
- `/src` was visible and used as the artifact/source tree. No scratch fallback was needed.

Code/config changes:
- Added `/src/cycle41_prior_collapse_diagnostic.py`.
- The script reuses the Cycle 39 artifact-only decomposition structure, compares Cycle 40 `prior_low` against same-subject `prior0`, and reads completed tensors/CSVs only.
- Added diagnostics for direct predicted-CLIP full-pool ranks, token/feature norm distributions, effective rank/PCA spectrum summaries, predicted-feature self-similarity, true-pair and impostor similarities, positive-minus-impostor margins, per-image PixCorr stage deltas, stress subsets, and correlations with direct brain-rank deltas.
- The script intentionally reuses the existing `/src/tables/cycle39_failure_decomposition/all_images_openclip_bigG_flat_norm.pt` teacher cache and does not copy or regenerate another large `all_images_openclip_bigG_flat_norm.pt` under the Cycle 41 output directory.

Validation and commands run:
- `/src/fmri/bin/python -m py_compile /src/cycle41_prior_collapse_diagnostic.py`
- `/src/fmri/bin/python /src/cycle41_prior_collapse_diagnostic.py --data_path=/src --outdir=/src/tables/cycle41_prior_collapse_diagnostic --topn=20`
- No `sbatch` command was run in Cycle 41.

Artifact authentication:
- All four required evaluator directories exist:
  - `/src/evals/cycle40_subj05_prior0_1sess_150ep`
  - `/src/evals/cycle40_subj05_prior_low_1sess_150ep`
  - `/src/evals/cycle40_subj07_prior0_1sess_150ep`
  - `/src/evals/cycle40_subj07_prior_low_1sess_150ep`
- For all four rows, required tensors exist and load as finite tensors:
  - `all_clipvoxels`: shape `(1000, 256, 1664)`, dtype `torch.float16`.
  - `all_blurryrecons`, `all_recons`, and `all_enhancedrecons`: shape `(1000, 3, 256, 256)`, dtype `torch.float32`.
- Image tensor ranges were finite and in expected ranges. Enhanced tensor min/max were `0.0/1.0`; means were subj05 prior0 `0.515476`, subj05 prior_low `0.515872`, subj07 prior0 `0.507720`, subj07 prior_low `0.520754`.
- Final CSVs exist under `/src/tables` for all four rows.
- Evaluator logs confirm `final_evaluations.py` consumed enhanced tensors via:
  - `evals/cycle40_subj05_prior0_1sess_150ep/cycle40_subj05_prior0_1sess_150ep_all_enhancedrecons.pt`
  - `evals/cycle40_subj05_prior_low_1sess_150ep/cycle40_subj05_prior_low_1sess_150ep_all_enhancedrecons.pt`
  - `evals/cycle40_subj07_prior0_1sess_150ep/cycle40_subj07_prior0_1sess_150ep_all_enhancedrecons.pt`
  - `evals/cycle40_subj07_prior_low_1sess_150ep/cycle40_subj07_prior_low_1sess_150ep_all_enhancedrecons.pt`

Outputs:
- `/src/tables/cycle41_prior_collapse_diagnostic/subj05_prior_low_vs_prior0_per_image.csv`, `1000` data rows.
- `/src/tables/cycle41_prior_collapse_diagnostic/subj07_prior_low_vs_prior0_per_image.csv`, `1000` data rows.
- `/src/tables/cycle41_prior_collapse_diagnostic/cycle41_prior_collapse_summary.json`.
- Output directory size is about `889K`; no large teacher cache or reconstruction tensor was written there.

Direct predicted-CLIP full-pool retrieval, `prior_low - prior0`; positive rank delta means worse:

| subject | image-rank mean | image-rank median | image worse frac | image improved frac | brain-rank mean | brain-rank median | brain worse frac | brain improved frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | +142.922 | +79.0 | 0.942 | 0.022 | -1.580 | 0.0 | 0.304 | 0.267 |
| subj07 | +148.302 | +89.5 | 0.949 | 0.034 | +2.166 | 0.0 | 0.326 | 0.204 |

Worst direct predicted-CLIP brain-rank regressions:
- subj05 top examples include eval/image IDs `(51, 5583, +172 brain-rank delta, -169 image-rank delta, +0.0543 enhanced-evalmix PixCorr delta)`, `(6, 3164, +74, +171, +0.0150)`, `(566, 42648, +60, +113, -0.3163)`, `(690, 50500, +57, +14, -0.0472)`, `(906, 65872, +51, +163, -0.0241)`.
- subj07 top examples are stored in the JSON/CSV; stress counts below summarize the pattern. The direct brain-rank damage is sparse relative to the catastrophic final BrainRet collapse.

Feature calibration and separability:

| subject | flat norm mean prior0 | flat norm mean prior_low | feature std mean prior0 | feature std mean prior_low | effective rank prior0 | effective rank prior_low | self offdiag cosine prior0 | self offdiag cosine prior_low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | 397.299 | 349.141 | 0.3423 | 0.1355 | 47.662 | 41.410 | 0.4800 | 0.8976 |
| subj07 | 374.990 | 350.133 | 0.3067 | 0.1234 | 46.638 | 41.094 | 0.5138 | 0.9128 |

Pairwise predicted-CLIP/image similarity:

| subject | true-pair sim prior0 | true-pair sim prior_low | image impostor sim prior0 | image impostor sim prior_low | image hardest margin prior0 | image hardest margin prior_low | brain hardest margin prior0 | brain hardest margin prior_low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | 0.04115 | 0.06886 | 0.00385 | 0.04751 | -0.00303 | -0.02454 | -0.000272 | -0.000316 |
| subj07 | 0.04893 | 0.06582 | 0.01327 | 0.04643 | -0.00285 | -0.02716 | +0.000587 | -0.000017 |

Interpretation of calibration/separability:
- `prior_low` increased true-pair similarity, matching the mechanical anchor-cosine story, but impostor similarity increased more. Predicted features became much more mutually similar: self off-diagonal cosine jumped to about `0.90`.
- Feature variance and effective rank dropped in both subjects. This is a calibration/separability collapse, not a simple low-level reconstruction failure.
- Direct predicted-CLIP image ranks collapsed for nearly all images. Direct predicted-CLIP brain ranks were mostly median-preserved and only sparse-regressed, so the final enhanced BrainRet collapse is not explained by a uniform full-pool direct brain-rank collapse.

PixCorr stage diagnostics, `prior_low - prior0`:

| subject | blurry mean | recons mean | enhanced raw mean | enhanced evalmix mean | enhanced evalmix median | enhanced evalmix improved frac |
|---|---:|---:|---:|---:|---:|---:|
| subj05 | +0.003849 | +0.001575 | +0.002343 | +0.003167 | +0.005282 | 0.518 |
| subj07 | -0.004069 | +0.004552 | +0.004405 | +0.003199 | +0.003192 | 0.513 |

Stress subsets:

| subject | PixCorr improved while direct brain rank worsened | image rank preserved while direct brain rank worsened | direct brain rank improved despite image rank worsening |
|---|---:|---:|---:|
| subj05 | 167 / 1000, mean brain-rank delta `+6.41`, mean PixCorr delta `+0.1000` | 12 / 1000, mean brain-rank delta `+23.67` | 245 / 1000, mean brain-rank delta `-11.76` |
| subj07 | 159 / 1000, mean brain-rank delta `+15.62`, mean PixCorr delta `+0.1074` | 21 / 1000, mean brain-rank delta `+38.33` | 192 / 1000, mean brain-rank delta `-10.82` |

ROI feasibility:
- Per-image ROI/category localization is not available from saved Cycle 40 artifacts.
- `final_evaluations.py` uses `GNet8_Encoder`, computes ROI correlations, and writes aggregate values for `nsd_general`, `V1`, `V2`, `V3`, `V4`, and `higher_vis`; no per-image ROI tensor is saved.
- Cycle 40 aggregate ROI evidence remains too blunt to explain the BrainRet collapse, especially because aggregate VC/HigherVis were mostly preserved or slightly improved while BrainRet collapsed.

Mechanism classification:
- Classify Cycle 40 as calibration/separability collapse at the predicted-CLIP boundary with downstream metric amplification.
- The low prior loss pulled predictions toward a shared anchor direction and reduced norm/variance, but it compressed the 1000-image predicted-CLIP manifold. True-pair cosine increased, yet impostor cosine and predicted-predicted self-similarity increased enough to destroy direct image retrieval.
- The protected final BrainRet collapse is larger than the direct predicted-CLIP brain-rank mean movement, so there is also a downstream/evaluator sensitivity component. However, generator/refiner explanations alone are closed because the saved predicted-CLIP representation is already badly miscalibrated and nonseparable.

Decision and next-step recommendation:
- Do not continue this prior-preservation branch, do not run a prior-weight grid, and do not scale to subjects 1/2.
- Any future mechanism must preserve instance-level separability, feature variance/effective rank, and positive-impostor margins, not just cosine-to-anchor or average norm drift.
- A justified next diagnostic, if requested by a future plan, would compare final BrainRet’s brain-encoder retrieval geometry against the direct saved predicted-CLIP geometry to locate the downstream amplification between compressed predicted CLIP and enhanced reconstruction evaluation.

## 2026-06-05 Cycle 42

Plan source: executed `/plan.md` only. Telegram report is not due.

Scope:
- Artifact-only downstream BrainRet amplification diagnostic for completed Cycle 40 prior-preservation rows.
- No training, no new model branch, no evaluator rerun for a new row, no subject 1/2 scale-up, and no `sbatch` job was launched.

Scheduler and artifact authentication:
- `/job-status.md` was absent; `squeue -u "$USER"` showed no active jobs.
- All four Cycle 40 eval directories, `all_clipvoxels`, `all_blurryrecons`, `all_recons`, `all_enhancedrecons`, and final CSVs were present.
- Cycle 40 evaluator logs confirmed `all_recons_path=evals/<model>/<model>_all_enhancedrecons.pt` for all four rows.
- Important evaluator-path finding: `/src/final_evaluations.py` computes protected `BrainRet` as `BwdRetrieval` from saved `all_clipvoxels` against true image OpenCLIP embeddings in repeated 300-image top-1 samples. The enhanced reconstruction tensor is authenticated for the same final evaluation run but does not feed the protected BrainRet calculation.

Code/config changes:
- Added `/src/cycle42_brainret_amplification_diagnostic.py`.
- The script reads Cycle 40 artifacts and Cycle 41 diagnostic outputs, reuses the existing `/src/tables/cycle39_failure_decomposition/all_images_openclip_bigG_flat_norm.pt` teacher cache, replays the final evaluator's 30 random 300-image BrainRet subsets, and writes compact CSV/JSON only.
- Initial combined interactive runs were externally killed before the final JSON write while holding large tensors for both subjects. I reduced memory by reusing Cycle 41 calibration values instead of recomputing effective-rank eigenspectra, avoiding reconstruction tensor loads in preflight, and running subjects 5 and 7 separately before merging compact outputs.

Commands and validation:
- `/src/fmri/bin/python -m py_compile /src/cycle42_brainret_amplification_diagnostic.py`
- `/src/fmri/bin/python /src/cycle42_brainret_amplification_diagnostic.py --data_path=/src --outdir=/src/tables/cycle42_brainret_amplification_diagnostic_s5 --subjects 5 --topn=20`
- `/src/fmri/bin/python /src/cycle42_brainret_amplification_diagnostic.py --data_path=/src --outdir=/src/tables/cycle42_brainret_amplification_diagnostic_s7 --subjects 7 --topn=20`
- Merged compact outputs into `/src/tables/cycle42_brainret_amplification_diagnostic/`; removed temporary `_s5` and `_s7` directories.
- JSON validation passed. Per-image CSV row counts: subject 5 `1000` rows, subject 7 `1000` rows, each with `82` columns. Output directory size is about `1.9M`.

Outputs:
- `/src/tables/cycle42_brainret_amplification_diagnostic/cycle42_brainret_amplification_summary.json`
- `/src/tables/cycle42_brainret_amplification_diagnostic/subj05_brainret_amplification_per_image.csv`
- `/src/tables/cycle42_brainret_amplification_diagnostic/subj07_brainret_amplification_per_image.csv`

Cycle 41 anchor consistency:
- Subject 5 matched Cycle 41: image-rank delta mean `+142.922`, brain-rank delta mean `-1.580`, effective rank `47.662 -> 41.410`, self off-diagonal cosine `0.4800 -> 0.8976`.
- Subject 7 matched Cycle 41: image-rank delta mean `+148.302`, brain-rank delta mean `+2.166`, effective rank `46.638 -> 41.094`, self off-diagonal cosine `0.5138 -> 0.9128`.

Evaluator replay and final BrainRet amplification:

| subject | CSV BrainRet delta | replayed sampled top-1 delta | replay minus CSV | full-pool rank delta mean | sampled-300 rank delta mean | sampled success-rate delta mean | sampled success-rate delta median |
|---|---:|---:|---:|---:|---:|---:|---:|
| subj05 | -0.480556 | -0.481333 | -0.000778 | +142.922 | +43.791 | -0.484099 | -0.428571 |
| subj07 | -0.486111 | -0.485778 | +0.000333 | +148.302 | +45.486 | -0.487711 | -0.444444 |

Margin and correlation readout:

| subject | full-pool hardest-margin delta mean | sampled hardest-margin delta mean | corr(success delta, image-rank delta) | corr(success delta, brain-rank delta) | corr(success delta, sampled margin delta) | corr(success delta, enhanced evalmix PixCorr delta) |
|---|---:|---:|---:|---:|---:|---:|
| subj05 | -0.021515 | -0.021841 | +0.223 | -0.119 | +0.428 | -0.015 |
| subj07 | -0.024302 | -0.022733 | +0.238 | +0.072 | +0.478 | +0.066 |

Worst-regression pattern:
- Many worst examples are clean top-1 losses: prior0 sampled success rate `1.0`, prior_low sampled success rate `0.0`, while direct Cycle 41 brain-rank deltas are often `0` or near `0`.
- Example subject 5 rows include eval indices `521`, `520`, `519`, `518`, and `507`, where full-pool ranks move from `1` to `223`, `154`, `36`, `27`, and `291` respectively, but Cycle 41 brain-rank deltas are `0`, `1`, `0`, `0`, and `0`.
- Example subject 7 rows include eval indices `856`, `429`, `428`, `857`, and `414`, where full-pool ranks move from `1` to `226`, `401`, `241`, `324`, and `75` respectively, but Cycle 41 brain-rank deltas are `0`.

ROI/category feasibility:
- Per-image ROI localization is not available from saved Cycle 40 artifacts. `final_evaluations.py` computes GNet ROI correlations and writes aggregate CSV values only; no per-image ROI tensor is saved.
- Aggregate ROI values remain too blunt for localizing this failure because VC/HigherVis changed little while BrainRet collapsed.

Conclusion:
- Classify both subjects as evaluator top-1 amplification of upstream predicted-CLIP separability collapse.
- The apparent mismatch from Cycle 41 is explained by metric geometry: the protected BrainRet gate is a repeated 300-way top-1 metric. A modest-looking direct rank summary can hide many images moving from rank 1 to rank 2+ or deeper, which produces an approximately `-0.48` absolute top-1 collapse.
- This is not primarily a diffusion/refiner or enhanced-reconstruction failure, because protected BrainRet is computed before reconstruction metrics from `all_clipvoxels`. PixCorr stage deltas have near-zero relationship with final BrainRet success-rate deltas.

Recommended next research questions:
- Future diagnostics should report top-1 preservation, hardest-impostor margins, and rank-1-to-rank>1 transition counts directly, not only mean rank.
- Any future training mechanism must protect predicted-CLIP variance/effective rank, off-diagonal similarity, hardest-impostor margins, and sampled top-1 BrainRet, not just anchor cosine or average norm.
- Do not continue the prior-preservation branch or run a prior-weight grid from this result.

## 2026-06-05 Cycle 43

Plan source: executed `/plan.md` only. Telegram report is not due.

Scope:
- Instrumentation-only evaluator-side rank/margin diagnostics for completed Cycle 40 subject 5/7 prior-preservation rows.
- No training, no new model branch, no prior-weight grid, no generator/refiner/diffusion-prior change, no subject 1/2 scale-up, and no `sbatch` job was launched.
- `final_evaluations.py` was inspected but not edited; the final metric CSV schema remains unchanged. The implementation is a separate read-only companion script.

Preflight:
- `/job-status.md` was not present in this workspace. Scheduler state was authenticated with `squeue -u "$USER"`, which showed no active jobs.
- Required Cycle 40 eval directories and final CSVs exist for:
  - `cycle40_subj05_prior0_1sess_150ep`
  - `cycle40_subj05_prior_low_1sess_150ep`
  - `cycle40_subj07_prior0_1sess_150ep`
  - `cycle40_subj07_prior_low_1sess_150ep`
- All four `all_clipvoxels` tensors load as finite `torch.float16` tensors with shape `(1000, 256, 1664)`.
- Existing teacher cache reused: `/src/tables/cycle39_failure_decomposition/all_images_openclip_bigG_flat_norm.pt`. No new large teacher cache, tensor, checkpoint, or model weight was written.

Code/config changes:
- Added `/src/cycle43_rank_margin_diagnostics.py`.
- The script reads completed evaluator artifacts only, replays the protected `BrainRet` path as saved `all_clipvoxels` versus true image OpenCLIP embeddings under the final evaluator's `30 x 300` sampled top-1 protocol, and writes compact rank/margin CSV/JSON diagnostics under `/src/tables/cycle43_rank_margin_diagnostics/`.
- Per-model CSV columns include `subject`, `model_name`, `eval_index`, `image_id`, `positive_similarity`, `hardest_impostor_similarity`, `hardest_impostor_index`, `margin`, `full_pool_rank`, `sampled_top1_success_rate`, `sampled_rank_mean`, and `sampled_rank_median`.
- Same-subject `prior_low - prior0` delta CSVs include `delta_positive_similarity`, `delta_hardest_impostor_similarity`, `delta_margin`, `delta_full_pool_rank`, `delta_sampled_top1_success_rate`, `rank1_to_not1`, and `not1_to_rank1`.

Commands and validation:
- `/src/fmri/bin/python -m py_compile /src/cycle43_rank_margin_diagnostics.py`
- `/src/fmri/bin/python /src/cycle43_rank_margin_diagnostics.py --data_path=/src --outdir=/src/tables/cycle43_rank_margin_diagnostics`
- Output row-count validation:
  - Four per-model CSVs, each `1000` data rows.
  - `subj05_prior_low_vs_prior0_rank_margin_delta.csv`, `1000` data rows.
  - `subj07_prior_low_vs_prior0_rank_margin_delta.csv`, `1000` data rows.
  - JSON summary: `/src/tables/cycle43_rank_margin_diagnostics/cycle43_rank_margin_summary.json`.

Outputs:
- `/src/tables/cycle43_rank_margin_diagnostics/cycle40_subj05_prior0_1sess_150ep_rank_margin.csv`
- `/src/tables/cycle43_rank_margin_diagnostics/cycle40_subj05_prior_low_1sess_150ep_rank_margin.csv`
- `/src/tables/cycle43_rank_margin_diagnostics/cycle40_subj07_prior0_1sess_150ep_rank_margin.csv`
- `/src/tables/cycle43_rank_margin_diagnostics/cycle40_subj07_prior_low_1sess_150ep_rank_margin.csv`
- `/src/tables/cycle43_rank_margin_diagnostics/subj05_prior_low_vs_prior0_rank_margin_delta.csv`
- `/src/tables/cycle43_rank_margin_diagnostics/subj07_prior_low_vs_prior0_rank_margin_delta.csv`
- `/src/tables/cycle43_rank_margin_diagnostics/cycle43_rank_margin_summary.json`
- Total output size is small, about `815K`.

Replay validation against Cycle 42:

| subject | CSV BrainRet delta | replayed sampled top-1 delta | replay minus CSV | tolerance |
|---|---:|---:|---:|---:|
| subj05 | -0.480556 | -0.481333 | -0.000778 | pass, within 0.002 |
| subj07 | -0.486111 | -0.485778 | +0.000333 | pass, within 0.002 |

Rank/margin readout, `prior_low - prior0`:

| subject | full-pool rank delta mean | full-pool rank delta median | margin delta mean | margin delta median | sampled success-rate delta mean | sampled success-rate delta median | rank1_to_not1 | not1_to_rank1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | +142.922 | +79.0 | -0.021515 | -0.021144 | -0.484099 | -0.428571 | 371 / 1000 | 2 / 1000 |
| subj07 | +148.302 | +89.5 | -0.024302 | -0.024113 | -0.487711 | -0.444444 | 388 / 1000 | 2 / 1000 |

Direct ImageRet/BrainRet rank summaries:

| row | BrainRet full-pool rank mean | BrainRet median | BrainRet top1 frac | ImageRet full-pool rank mean | ImageRet median | ImageRet top1 frac |
|---|---:|---:|---:|---:|---:|---:|
| subj05 prior0 | 14.202 | 2.0 | 0.399 | 9.714 | 1.0 | 0.502 |
| subj05 prior_low | 157.124 | 87.0 | 0.030 | 8.134 | 2.0 | 0.479 |
| subj07 prior0 | 20.197 | 2.0 | 0.401 | 10.763 | 1.0 | 0.548 |
| subj07 prior_low | 168.499 | 100.0 | 0.015 | 12.929 | 2.0 | 0.493 |

Correlations with sampled success-rate delta:
- subj05: margin delta `+0.408`, full-pool rank delta `+0.223`, enhanced evalmix PixCorr delta `-0.015`.
- subj07: margin delta `+0.431`, full-pool rank delta `+0.238`, enhanced evalmix PixCorr delta `+0.066`.

ROI/category diagnostics:
- Per-image ROI/category localization is not available from saved Cycle 40 artifacts. `final_evaluations.py` writes aggregate GNet ROI correlations only, and no per-image ROI tensor is saved. This limitation was recorded in the JSON summary and does not block rank/margin instrumentation.

Conclusion:
- Cycle 43 succeeds. Rank/margin observability is now available as a normal compact diagnostic artifact set for the completed Cycle 40 rows, with subject 5 and subject 7 kept separate.
- The replayed sampled top-1 deltas match Cycle 42 within tolerance, and the new outputs expose the core failure mode directly: large positive rank worsening, strongly negative hardest-impostor margin deltas, and hundreds of rank-1 to rank-greater-than-1 transitions under `prior_low`.
- Future training proposals should treat these diagnostics as acceptance proxies before considering any renewed anti-collapse objective or other model branch.

## 2026-06-05 Cycle 44

Plan source: executed `/plan.md` only. Telegram report is due.

Scope:
- Implemented and launched the retrieval-calibrated anti-collapse test for subjects 5 and 7 only.
- Rows are the same-code zero-control `margin0` and one conservative nonzero row `margin_low` with `clip_margin_loss_weight=0.05`, `clip_margin_floor=0.02`, `clip_margin_topk=8`, and `clip_margin_exclude_teacher_topk=0`.
- No prior-weight grid, no subject 1/2 scale-up, no use of `/strategizing-chat.md`, and no held-out/test labels or evaluator outputs used in training loss.

Preflight:
- `/job-status.md` was not visible in this workspace; `squeue -u "$USER"` showed no active jobs before Cycle 44 submission.
- Official multisubject checkpoints were present for both planned subjects:
  - `/src/train_logs/final_multisubject_subj05/last.pth`, about `9.7G`.
  - `/src/train_logs/final_multisubject_subj07/last.pth`, about `9.7G`.
- Existing official/evaluator reference artifacts were present, including Cycle 42 and Cycle 43 summary JSON files under `/src/tables/`.

Code/config changes:
- Edited `/src/Train.py` to add CLI-gated batch-local predicted-CLIP margin support:
  - `--clip_margin_loss_weight`, default `0.0`.
  - `--clip_margin_floor`, default `0.02`.
  - `--clip_margin_topk`, default `8`.
  - `--clip_margin_exclude_teacher_topk`, default `0`.
  - `--clip_margin_log_every`, default `1`.
- The loss uses normalized predicted CLIP tokens flattened at the same `clip_voxels` boundary as prior/topology experiments and same-batch frozen image OpenCLIP targets. It computes positive similarity, hardest eligible in-batch impostor similarity, and hinge `relu(floor - positive + hardest)`.
- Added compact train/test diagnostics: feature std mean, effective rank, off-diagonal predicted-feature cosine, positive similarity, hardest-impostor similarity, positive-minus-hardest margin, raw margin loss, and weighted contribution. Diagnostics are logged for `margin0` as well as `margin_low`.
- Added Slurm scripts:
  - `/src/cycle44_margin_smoke.slurm`
  - `/src/cycle44_margin_train_s57.slurm`
  - `/src/cycle44_margin_eval_s57.slurm`
- Added `/src/cycle44_rank_margin_diagnostics.py`, a Cycle 44 wrapper around the Cycle 43 evaluator replay path for `margin_low - margin0`, plus feature-spread summaries from saved `all_clipvoxels`.

Validation:
- Compile passed:
  - `/src/fmri/bin/python -m py_compile /src/Train.py /src/models.py`
  - `/src/fmri/bin/python -m py_compile /src/cycle44_rank_margin_diagnostics.py`
- Slurm syntax passed:
  - `bash -n /src/cycle44_margin_smoke.slurm /src/cycle44_margin_train_s57.slurm /src/cycle44_margin_eval_s57.slurm`

Smoke jobs:
- First smoke submission `9263797` used `num_epochs=1` and failed before training in both rows with `ValueError: Expected float between 0 and 1 pct_start, but got 2.0`. This is the repo's existing OneCycleLR constraint from `pct_start=2/num_epochs`, not a margin-objective tensor error.
- Corrected smoke script to `num_epochs=3` and relaunched as job `9263912`.
- Corrected smoke completed successfully:
  - `9263912_0` `margin0`: `COMPLETED`, exit `0:0`, elapsed `00:05:07`, MaxRSS `21451544K`.
  - `9263912_1` `margin_low`: `COMPLETED`, exit `0:0`, elapsed `00:05:07`, MaxRSS `21450872K`.
- Smoke checkpoints were saved under `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle44_smoke_subj07_margin0_1sess_3ep/last.pth` and `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle44_smoke_subj07_margin_low_1sess_3ep/last.pth`.

Smoke diagnostics:
- `margin0` epoch diagnostics:
  - epoch 0: loss `0.036465553`, scaled `0`, feature std mean `0.00018122624`, effective rank `18.324457`, offdiag cosine `0.95465285`, train margin `-0.016411889`, test margin `-0.019134521`.
  - epoch 2: loss `0.0023177535`, scaled `0`, feature std mean `0.00090216493`, effective rank `20.661776`, offdiag cosine `0.26441217`, train margin `0.040502733`, test margin `-0.0088272095`.
- `margin_low` epoch diagnostics:
  - epoch 0: loss `0.036467029`, scaled `0.0018233638`, feature std mean `0.00018122172`, effective rank `18.324303`, offdiag cosine `0.9546686`, train margin `-0.016413073`, test margin `-0.019134521`.
  - epoch 2: loss `0.0023151136`, scaled `0.00011575607`, feature std mean `0.00090235017`, effective rank `20.662571`, offdiag cosine `0.26421528`, train margin `0.040517499`, test margin `-0.0088348389`.
- Smoke conclusion: code path is finite, shape-compatible, same-code control logs diagnostics, and the nonzero row contributes the intended small weighted margin term.

Training launched:
- Submitted full four-row training array with `sbatch /src/cycle44_margin_train_s57.slurm`.
- Job ID: `9264132`.
- Planned rows:
  - `9264132_0`: subject 5 `cycle44_subj05_margin0_1sess_150ep`.
  - `9264132_1`: subject 5 `cycle44_subj05_margin_low_1sess_150ep`.
  - `9264132_2`: subject 7 `cycle44_subj07_margin0_1sess_150ep`.
  - `9264132_3`: subject 7 `cycle44_subj07_margin_low_1sess_150ep`.
- At last check, all four rows were still `PENDING` on the `gpu` partition with time limit `04:30:00`; no full-training logs or checkpoints were available yet.

Pending next actions:
- Wait for job `9264132` to complete or fail.
- If rows complete, launch `/src/cycle44_margin_eval_s57.slurm` for completed rows only, then run `/src/cycle44_rank_margin_diagnostics.py`.
- If any row fails, inspect `/src/slurms/c44_margin_s57_9264132_<task>.out/.err`, fix only operational issues that preserve the exact objective/protocol, and relaunch only failed planned rows.

Telegram-ready update:
- Cycle 44 implemented the planned batch-local CLIP margin anti-collapse objective and diagnostics in `Train.py`, with no-effect defaults and training-batch-only positives/impostors. Compile and Slurm syntax checks passed. A first 1-epoch smoke exposed an existing scheduler bug (`pct_start=2/num_epochs`), so the smoke was corrected to 3 epochs. Corrected smoke job `9263912` completed both `margin0` and `margin_low` rows in `00:05:07`, MaxRSS about `21.5G`, with finite margin diagnostics and saved smoke checkpoints. Full subject 5/7 four-row training array `9264132` has been submitted and is pending on the GPU partition; no final metrics or plots are available yet. Next report should parse `9264132` logs, run evaluation for completed checkpoints, and compare `margin_low - margin0` with the Cycle 44 rank/margin diagnostic.

## 2026-06-05 Cycle 45

Plan source: executed `/plan.md` only. Telegram report is not due.

Scope:
- Recovered the live Slurm state for Cycle 44 training array `9264132_[0-3]`.
- Completed the protected subject 5/7 `margin0` and `margin_low` training authentication.
- Launched the unchanged planned enhanced evaluator for all four completed checkpoints.
- Did not run a new model branch, new objective, subject 1/2 scale-up, margin grid, generator/refiner change, or any diagnostic before evaluator artifacts existed.

Code/config changes:
- No code changes were made in Cycle 45.
- No Slurm script changes were made in Cycle 45.

Training job recovery:

| task | model | state | exit | elapsed | node | MaxRSS | stdout | stderr | checkpoint evidence |
|---|---|---:|---:|---:|---|---:|---|---|---|
| `9264132_0` | `cycle44_subj05_margin0_1sess_150ep` | `COMPLETED` | `0:0` | `02:09:00` | `della-l04g6` | `21614388K` | `/src/slurms/c44_margin_s57_9264132_0.out` | `/src/slurms/c44_margin_s57_9264132_0.err` | log repeatedly reports saved `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle44_subj05_margin0_1sess_150ep/last` |
| `9264132_1` | `cycle44_subj05_margin_low_1sess_150ep` | `COMPLETED` | `0:0` | `02:07:34` | `della-l01g15` | `21615088K` | `/src/slurms/c44_margin_s57_9264132_1.out` | `/src/slurms/c44_margin_s57_9264132_1.err` | log repeatedly reports saved `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle44_subj05_margin_low_1sess_150ep/last` |
| `9264132_2` | `cycle44_subj07_margin0_1sess_150ep` | `COMPLETED` | `0:0` | `02:08:00` | `della-l04g9` | `21592568K` | `/src/slurms/c44_margin_s57_9264132_2.out` | `/src/slurms/c44_margin_s57_9264132_2.err` | log repeatedly reports saved `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle44_subj07_margin0_1sess_150ep/last` |
| `9264132_3` | `cycle44_subj07_margin_low_1sess_150ep` | `COMPLETED` | `0:0` | `02:06:58` | `della-l05g7` | `21591664K` | `/src/slurms/c44_margin_s57_9264132_3.out` | `/src/slurms/c44_margin_s57_9264132_3.err` | log repeatedly reports saved `/scratch/gpfs/KNORMAN/aw1907/MindEyeV2/train_logs/cycle44_subj07_margin_low_1sess_150ep/last` |

Final training/eval-loop metrics at epoch 149:

| model | test blurry PixCorr | test loss | test fwd/bwd retrieval | train feature std mean | train effective rank | train offdiag pred cosine | train positive sim | train hardest sim | train positive-hardest margin | raw margin loss | scaled margin contribution | test feature std mean | test effective rank | test offdiag pred cosine | test positive-hardest margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 `margin0` | `0.193` | `14.3` | truncated in tqdm line | `0.0011022383` | `21.901299` | `0.10959551` | `0.074033676` | `0.0041376545` | `0.069897067` | `0` | `0` | `0.00085591699` | `161.41522` | `0.48071289` | `0.0015611649` |
| subj05 `margin_low` | `0.197` | `14.4` | truncated in tqdm line | `0.0010980947` | `21.904828` | `0.11273587` | `0.074974798` | `0.0050848376` | `0.069885254` | `0` | `0` | `0.00084565196` | `161.77051` | `0.49121094` | `0.00096940994` |
| subj07 `margin0` | `0.226` | `14.2` | truncated in tqdm line | `0.0010849848` | `21.860065` | `0.1107099` | `0.082596317` | `0.014253924` | `0.068351499` | `0` | `0` | `0.00081836735` | `160.87404` | `0.50488281` | `0.001534462` |
| subj07 `margin_low` | `0.230` | `14.2` | truncated in tqdm line | `0.0010801024` | `21.86443` | `0.11590379` | `0.079987557` | `0.011250034` | `0.068733461` | `0` | `0` | `0.00080445281` | `159.9476` | `0.51904297` | `0.0013656616` |

Training observations:
- All four rows reached `100%|150/150` with no `Traceback`, `RuntimeError`, CUDA OOM, or failed-state evidence in stderr.
- The nonzero margin term was active earlier in training but was zero by epoch 149 for the final sampled training batch, indicating the batch-local hinge was satisfied in that batch.
- Same-subject final test diagnostic differences were small: subject 5 `margin_low` had slightly lower test feature std, slightly higher effective rank, higher offdiag cosine, and lower positive-hardest margin than `margin0`; subject 7 `margin_low` had lower test feature std, lower effective rank, higher offdiag cosine, and slightly lower positive-hardest margin than `margin0`.
- These are mechanism checks only. Per `/plan.md`, protected enhanced evaluator metrics decide success.

Evaluation launched:
- Submitted the unchanged planned evaluator with `sbatch /src/cycle44_margin_eval_s57.slurm`.
- Evaluator job: `9271345_[0-3]`, requested one A100, `64G`, `04:00:00`.
- Planned rows are the same four completed training rows: subject 5 `margin0`, subject 5 `margin_low`, subject 7 `margin0`, subject 7 `margin_low`.
- Latest check at `2026-06-05 17:17:49 EDT`: evaluator still `PENDING`, no node assigned, no evaluator stdout/stderr files yet, and no Cycle 44 enhanced tensors or final CSVs yet.

Diagnostics status:
- `/src/cycle44_rank_margin_diagnostics.py` was not run because evaluator artifacts do not exist yet and the script is designed to refuse active scheduler jobs.
- Required post-eval artifact checks remain pending: `evals/<model>/<model>_all_enhancedrecons.pt` shape, final `/src/tables/<model>_all_enhancedrecons.csv`, full metric vector, replayed BrainRet agreement, and same-subject `margin_low - margin0` rank/margin summaries.

Conclusions:
- Cycle 45 completed the training recovery and found no operational training failures; no relaunch is needed for `9264132_[0-3]`.
- The next valid action is to wait for evaluator job `9271345_[0-3]`, parse its logs, authenticate enhanced recon tensors and final CSVs, then run `/src/cycle44_rank_margin_diagnostics.py`.
- No scientific success/failure claim can be made yet because protected enhanced evaluator metrics are still missing.

Recommended next research questions:
- Do the four Cycle 44 evaluator rows start and finish under the 4-hour request?
- Do `margin_low - margin0` refined BrainRet/ImageRet and visual metrics improve, preserve, or regress separately for subject 5 and subject 7?
- Does the evaluator-side rank/margin replay match final CSV BrainRet within the established `~0.002` tolerance?

## 2026-06-06 Cycle 46

Plan source: executed `/plan.md` only. Telegram report is not due.

Scope:
- Completed the Cycle 44 subject 5/7 evaluator readout and rank/margin replay.
- No training, no evaluator relaunch, no new model branch, no subject 1/2 scale-up, and no margin-weight grid was launched.

Preflight and evaluator job authentication:
- `/job-status.md` was not visible in this workspace; scheduler state was authenticated with `squeue`, which showed no active jobs.
- `9271345_[0-3]` all completed with exit `0:0`.

| task | model | state | exit | elapsed | node | MaxRSS | stdout | stderr |
|---|---|---:|---:|---:|---|---:|---|---|
| `9271345_0` | `cycle44_subj05_margin0_1sess_150ep` | `COMPLETED` | `0:0` | `02:27:20` | `della-l05g3` | `49609720K` | `/src/slurms/c44_margin_eval_s57_9271345_0.out` | `/src/slurms/c44_margin_eval_s57_9271345_0.err` |
| `9271345_1` | `cycle44_subj05_margin_low_1sess_150ep` | `COMPLETED` | `0:0` | `02:27:44` | `della-l03g11` | `49475200K` | `/src/slurms/c44_margin_eval_s57_9271345_1.out` | `/src/slurms/c44_margin_eval_s57_9271345_1.err` |
| `9271345_2` | `cycle44_subj07_margin0_1sess_150ep` | `COMPLETED` | `0:0` | `02:23:07` | `della-l05g3` | `46254336K` | `/src/slurms/c44_margin_eval_s57_9271345_2.out` | `/src/slurms/c44_margin_eval_s57_9271345_2.err` |
| `9271345_3` | `cycle44_subj07_margin_low_1sess_150ep` | `COMPLETED` | `0:0` | `02:26:00` | `della-l05g6` | `49525988K` | `/src/slurms/c44_margin_eval_s57_9271345_3.out` | `/src/slurms/c44_margin_eval_s57_9271345_3.err` |

Evaluator log authentication:
- Grep over all four stdout/stderr files found no `Traceback`, `RuntimeError`, CUDA OOM, killed process, missing file, or NaN evidence.
- Each stdout explicitly recorded `all_recons_path: evals/<model>/<model>_all_enhancedrecons.pt` before `final_evaluations.py` printed the final metric table.
- The logs reported enhanced tensor shape `torch.Size([1000, 3, 256, 256])`; direct artifact loading confirmed finite `torch.float32` tensors with actual shape `(1000, 3, 256, 256)` for all four rows. This differs from the plan text's expected `(1000, 3, 512, 512)`, but all evaluator jobs consumed the saved enhanced tensors successfully.

Artifact authentication:
- Enhanced tensors exist and are finite:
  - `/src/evals/cycle44_subj05_margin0_1sess_150ep/cycle44_subj05_margin0_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle44_subj05_margin_low_1sess_150ep/cycle44_subj05_margin_low_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle44_subj07_margin0_1sess_150ep/cycle44_subj07_margin0_1sess_150ep_all_enhancedrecons.pt`
  - `/src/evals/cycle44_subj07_margin_low_1sess_150ep/cycle44_subj07_margin_low_1sess_150ep_all_enhancedrecons.pt`
- Final CSVs exist under `/src/tables/` for all four rows.

Code/config changes:
- Added flush-only progress prints to `/src/cycle43_rank_margin_diagnostics.py` and `/src/cycle44_rank_margin_diagnostics.py` because the original replay path was silent during long CPU matrix products.
- Fixed two Cycle 44 diagnostic-script operational issues:
  - Replaced the large `torch.linalg.svdvals(centered)` feature-spread computation with the equivalent Gram-matrix eigenvalue path on `centered @ centered.T`.
  - Replaced the nonexistent `c43.offdiag_values(...)` call with direct off-diagonal masking.
- Added `--skip_feature_spread` to `/src/cycle44_rank_margin_diagnostics.py` so rank/margin replay and feature-spread readout could be run in separate short processes. This did not change model artifacts, evaluator metrics, rank/margin formulas, seed, sample size, or training behavior.

Commands and validation:
- `/src/fmri/bin/python -m py_compile /src/cycle43_rank_margin_diagnostics.py /src/cycle44_rank_margin_diagnostics.py`
- `/src/fmri/bin/python -u /src/cycle44_rank_margin_diagnostics.py --data_path=/src --outdir=/src/tables/cycle44_rank_margin_diagnostics_s7 --subjects 7 --skip_feature_spread`
- `/src/fmri/bin/python -u /src/cycle44_rank_margin_diagnostics.py --data_path=/src --outdir=/src/tables/cycle44_rank_margin_diagnostics_s5 --subjects 5 --skip_feature_spread`
- Four separate one-row feature-spread Python processes wrote compact JSONs under `/src/tables/cycle44_rank_margin_diagnostics/feature_spread_rows/`.
- Merged final compact outputs into `/src/tables/cycle44_rank_margin_diagnostics/` and removed temporary `_s5`/`_s7` directories.

Final protected metric vector:

| model | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet-B | SwAV | ImageRet | BrainRet | VC | V1 | V2 | V3 | V4 | HigherVis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 `margin0` | 0.192674 | 0.406933 | 0.846794 | 0.916086 | 0.856197 | 0.840533 | 0.768113 | 0.431212 | 0.648778 | 0.552778 | 0.416381 | 0.351920 | 0.355988 | 0.341006 | 0.315086 | 0.423627 |
| subj05 `margin_low` | 0.195786 | 0.414150 | 0.844991 | 0.920843 | 0.860523 | 0.845491 | 0.770811 | 0.432561 | 0.657000 | 0.543222 | 0.412823 | 0.346147 | 0.352018 | 0.333507 | 0.312330 | 0.420981 |
| subj07 `margin0` | 0.194279 | 0.404695 | 0.829048 | 0.894047 | 0.785077 | 0.770978 | 0.828250 | 0.477893 | 0.680556 | 0.542000 | 0.321338 | 0.321689 | 0.324969 | 0.316007 | 0.281305 | 0.306982 |
| subj07 `margin_low` | 0.189978 | 0.402765 | 0.821402 | 0.887852 | 0.777570 | 0.773904 | 0.831974 | 0.481643 | 0.687222 | 0.536444 | 0.317811 | 0.316086 | 0.318589 | 0.312224 | 0.276179 | 0.302657 |

Same-subject protected deltas, `margin_low - margin0`:

| subject | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | Inception | CLIP | EffNet-B | SwAV | ImageRet | BrainRet | VC | HigherVis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subj05 | +0.003112 | +0.007216 | -0.001803 | +0.004757 | +0.004325 | +0.004959 | +0.002698 | +0.001349 | +0.008222 | -0.009556 | -0.003557 | -0.002646 |
| subj07 | -0.004301 | -0.001930 | -0.007646 | -0.006195 | -0.007508 | +0.002926 | +0.003725 | +0.003750 | +0.006667 | -0.005556 | -0.003528 | -0.004325 |

Rank/margin replay outputs:
- `/src/tables/cycle44_rank_margin_diagnostics/cycle44_rank_margin_summary.json`
- `/src/tables/cycle44_rank_margin_diagnostics/cycle44_subj05_margin0_1sess_150ep_rank_margin.csv`
- `/src/tables/cycle44_rank_margin_diagnostics/cycle44_subj05_margin_low_1sess_150ep_rank_margin.csv`
- `/src/tables/cycle44_rank_margin_diagnostics/cycle44_subj07_margin0_1sess_150ep_rank_margin.csv`
- `/src/tables/cycle44_rank_margin_diagnostics/cycle44_subj07_margin_low_1sess_150ep_rank_margin.csv`
- `/src/tables/cycle44_rank_margin_diagnostics/subj05_margin_low_vs_margin0_rank_margin_delta.csv`
- `/src/tables/cycle44_rank_margin_diagnostics/subj07_margin_low_vs_margin0_rank_margin_delta.csv`
- `/src/tables/cycle44_rank_margin_diagnostics/feature_spread_summary.json`

Replay agreement:

| subject | CSV BrainRet delta | replayed sampled top-1 delta | replay minus CSV | tolerance |
|---|---:|---:|---:|---|
| subj05 | -0.009556 | -0.010222 | -0.000667 | pass, within 0.002 |
| subj07 | -0.005556 | -0.005444 | +0.000111 | pass, within 0.002 |

Rank/margin diagnostic readout:

| subject | row | positive-minus-hardest margin mean | full-pool rank mean | sampled top-1 aggregate | feature std mean | effective rank | offdiag pred cosine |
|---|---|---:|---:|---:|---:|---:|---:|
| subj05 | `margin0` | -0.003028 | 14.202 | 0.553222 | 0.0008588 | 385.388 | 0.480037 |
| subj05 | `margin_low` | -0.003231 | 13.778 | 0.543000 | 0.0008482 | 386.624 | 0.491154 |
| subj07 | `margin0` | -0.002854 | 20.197 | 0.541889 | 0.0008131 | 385.090 | 0.513782 |
| subj07 | `margin_low` | -0.002747 | 20.162 | 0.536444 | 0.0007987 | 381.722 | 0.529120 |

Rank/margin deltas:

| subject | delta margin mean | delta full-pool rank mean | delta sampled top-1 mean | rank1_to_not1 | not1_to_rank1 |
|---|---:|---:|---:|---:|---:|
| subj05 | -0.000203 | -0.424 | -0.012560 | 52 / 1000 | 52 / 1000 |
| subj07 | +0.000106 | -0.035 | -0.003527 | 52 / 1000 | 49 / 1000 |

Conclusion:
- Cycle 44 `margin_low` does not pass the planned success gate. BrainRet does not improve by about `+0.02` in either weak subject; it decreases in both subjects versus the same-code same-subject `margin0`.
- ImageRet improves modestly in both subjects, but this cannot rescue the branch because protected BrainRet is negative.
- CLIP is preserved/improved, but Inception, VC, HigherVis, AlexNet, PixCorr, and lower-is-better EffNet-B/SwAV show small mixed or negative movements, especially for subject 7.
- Mechanism diagnostics do not show a Cycle 40-style collapse, but they also do not support the hypothesis: subject 5 has a slightly worse positive-minus-hardest margin, both subjects have lower sampled top-1 aggregate under `margin_low`, feature std is lower in both, and off-diagonal predicted-feature cosine moves upward in both.
- Close this conservative batch-local margin variant. Per `/plan.md`, do not run a margin grid, stronger margin loss, subject 1/2 scale-up, topology/sparse-neighbor/prior-anchor revival, adapters, routers/MoE, ROI routing, CLIP-layer fusion, generator/refiner edits, captions/VLM correction, temporal decoding, hard voxel pruning, or reliability-prior branch as the immediate next move.

Recommended next research questions:
- What training-only brain-side reliability or subject-specific repeatability diagnostic can explain weak-subject BrainRet fragility without touching held-out evaluator labels?
- Can one-session MindEye2 be instrumented to predict per-image BrainRet failure risk from training repeats or voxel reliability while preserving the full protected evaluator gate?
