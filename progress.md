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
