import argparse
import json
import os
from collections import OrderedDict

import torch

import cycle43_rank_margin_diagnostics as c43


ROWS = {
    5: {
        "margin0": "cycle44_subj05_margin0_1sess_150ep",
        "margin_low": "cycle44_subj05_margin_low_1sess_150ep",
    },
    7: {
        "margin0": "cycle44_subj07_margin0_1sess_150ep",
        "margin_low": "cycle44_subj07_margin_low_1sess_150ep",
    },
}


def feature_spread_summary(student):
    student = student.float()
    n = student.shape[0]
    if n < 2:
        return OrderedDict(feature_std_mean=0.0, effective_rank=1.0, offdiag_pred_cosine=0.0)
    centered = student - student.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    singular_values = singular_values[singular_values > 1e-8]
    if int(singular_values.numel()) == 0:
        effective_rank = torch.tensor(1.0)
    else:
        probs = singular_values / singular_values.sum().clamp_min(1e-12)
        effective_rank = torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    sim = student @ student.T
    offdiag = c43.offdiag_values(sim)
    return OrderedDict(
        feature_std_mean=float(student.std(dim=0, unbiased=False).mean().item()),
        effective_rank=float(effective_rank.item()),
        offdiag_pred_cosine=float(offdiag.mean().item()),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Cycle 44 evaluator-side rank/margin diagnostics for margin_low versus margin0."
    )
    parser.add_argument("--data_path", default="/src")
    parser.add_argument("--outdir", default="/src/tables/cycle44_rank_margin_diagnostics")
    parser.add_argument("--teacher_cache", default=None)
    parser.add_argument("--subjects", nargs="+", type=int, default=[5, 7], choices=[5, 7])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_loops", type=int, default=30)
    parser.add_argument("--eval_sample_size", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    c43.ROWS = ROWS
    sched = c43.scheduler_state()
    if sched["active_job_count"]:
        raise RuntimeError(f"Active jobs found: {sched['active_jobs']}")
    missing, artifacts = c43.authenticate_artifacts(args.data_path, args.subjects)
    if missing:
        raise FileNotFoundError(json.dumps(missing, indent=2))

    teacher, teacher_path = c43.load_teacher(args.data_path, args.teacher_cache)
    subject_summaries = []
    delta_paths = OrderedDict()
    replay_checks = []

    for subject in args.subjects:
        ids, ids_source = c43.image_ids(args.data_path, subject, teacher.shape[0])
        row_dfs = OrderedDict()
        row_summaries = OrderedDict()
        for label, model_name in ROWS[subject].items():
            row_df, row_summary = c43.write_row(args, subject, label, model_name, teacher, ids, ids_source)
            student, _ = c43.load_student(args.data_path, model_name)
            row_summary["feature_spread"] = feature_spread_summary(student)
            del student
            row_dfs[label] = row_df
            row_summaries[label] = row_summary

        delta_df, delta_summary = c43.summarize_delta(subject, row_dfs["margin0"], row_dfs["margin_low"], None)
        delta_summary["row_label"] = "margin_low_vs_margin0"
        delta_path = os.path.join(args.outdir, f"subj{subject:02d}_margin_low_vs_margin0_rank_margin_delta.csv")
        delta_df.to_csv(delta_path, index=False)
        delta_paths[f"subj{subject:02d}"] = delta_path

        csv_delta = (
            row_summaries["margin_low"]["final_csv_metric_values"]["BrainRet"]
            - row_summaries["margin0"]["final_csv_metric_values"]["BrainRet"]
        )
        replay_delta = (
            row_summaries["margin_low"]["sampled_aggregate_top1"]
            - row_summaries["margin0"]["sampled_aggregate_top1"]
        )
        replay_checks.append(
            OrderedDict(
                subject=subject,
                csv_brainret_delta=float(csv_delta),
                replayed_sampled_top1_delta=float(replay_delta),
                replay_minus_csv_delta=float(replay_delta - csv_delta),
                within_0p002_tolerance=bool(abs(replay_delta - csv_delta) <= 0.002),
            )
        )
        subject_summaries.append(
            OrderedDict(
                subject=subject,
                rows=row_summaries,
                control_delta=delta_summary,
                control_delta_csv=delta_path,
            )
        )

    out = OrderedDict(
        provenance=OrderedDict(
            cycle=44,
            scope="read-only evaluator-side rank/margin diagnostics",
            data_path=args.data_path,
            outdir=args.outdir,
            teacher_cache=teacher_path,
            seed=args.seed,
            eval_loops=args.eval_loops,
            eval_sample_size=args.eval_sample_size,
            no_training_or_model_behavior_change=True,
        ),
        preflight=OrderedDict(scheduler=sched, missing=missing, artifacts=artifacts),
        replay_checks=replay_checks,
        delta_csvs=delta_paths,
        subjects=subject_summaries,
    )
    json_path = os.path.join(args.outdir, "cycle44_rank_margin_summary.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=c43.as_builtin)
    print(json.dumps(out, indent=2, default=c43.as_builtin)[:30000])
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
