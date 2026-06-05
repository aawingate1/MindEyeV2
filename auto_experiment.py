#!/usr/bin/env python3
"""Supervisor for the MindEye autonomous experiment loop."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".agent-controller"
LOG_DIR = STATE_DIR / "logs"
STATE_FILE = STATE_DIR / "state.json"
TELEGRAM_ENV = STATE_DIR / "telegram.env"
STRATEGY_CHAT = ROOT / "strategizing-chat.md"
PLAN_FILE = ROOT / "plan.md"
CS_PLAN_DRAFT = ROOT / "cs-agent" / "plan.md.next"
JOB_STATUS = ROOT / "job-status.md"

AGENTS = {
    "bio": ROOT / "run-bio-agent.sh",
    "cs": ROOT / "run-cs-agent.sh",
    "orchestrator": ROOT / "run-orchestrator-agent.sh",
}

AUTO_COMMIT_PATHS = [
    ".gitignore",
    "auto_experiment.py",
    "bio-agent/AGENTS.md",
    "cs-agent/AGENTS.md",
    "orchestrator-agent/AGENTS.md",
    "progress.md",
    "run-bio-agent.sh",
    "run-cs-agent.sh",
    "run-orchestrator-agent.sh",
    "src",
]

MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
BLOCKED_GIT_PATH_SUFFIXES = (
    "/all_images_openclip_bigG_flat_norm.pt",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def promote_cs_plan_draft() -> None:
    draft = read(CS_PLAN_DRAFT).strip()
    if not draft:
        raise RuntimeError(f"cs-agent did not write a non-empty plan draft at {CS_PLAN_DRAFT}")
    PLAN_FILE.write_text(draft + "\n", encoding="utf-8")


def command_output(cmd: list[str], timeout_s: int = 30) -> tuple[int, str]:
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        return 127, f"{cmd[0]} not found"
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        return 124, f"timed out after {timeout_s}s\n{partial}"
    return result.returncode, result.stdout.strip()


def update_job_status(slurm_user: str, sacct_lookback_h: int) -> bool:
    squeue_cmd = ["squeue", "-h", "-u", slurm_user, "-o", "%i|%T|%j|%M|%D|%R"]
    squeue_code, squeue_text = command_output(squeue_cmd)
    active = squeue_code == 0 and bool(squeue_text.strip())

    start = datetime.fromtimestamp(time.time() - sacct_lookback_h * 3600).strftime("%Y-%m-%dT%H:%M:%S")
    sacct_cmd = [
        "sacct",
        "-u",
        slurm_user,
        "--starttime",
        start,
        "--format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS",
        "-P",
        "--noheader",
    ]
    sacct_code, sacct_text = command_output(sacct_cmd)

    JOB_STATUS.write_text(
        f"# Job Status\n\n"
        f"Updated: {now()}\n"
        f"Slurm user: {slurm_user}\n"
        f"Active jobs: {'yes' if active else 'no'}\n\n"
        f"## Active Jobs (`squeue`)\n\n"
        f"Command: {' '.join(squeue_cmd)}\n"
        f"Exit code: {squeue_code}\n\n"
        f"```text\n{squeue_text or '(none)'}\n```\n\n"
        f"## Recent Jobs (`sacct`, last {sacct_lookback_h}h)\n\n"
        f"Command: {' '.join(sacct_cmd)}\n"
        f"Exit code: {sacct_code}\n\n"
        f"```text\n{sacct_text or '(none)'}\n```\n",
        encoding="utf-8",
    )
    return active


def wait_for_slurm_quiescence(args: argparse.Namespace, state: dict) -> None:
    if args.no_wait_for_slurm or args.dry_run:
        update_job_status(args.slurm_user, args.sacct_lookback_h)
        return

    while update_job_status(args.slurm_user, args.sacct_lookback_h):
        print(f"[{now()}] active Slurm jobs detected; waiting {args.slurm_wait_poll_s}s before next research cycle", flush=True)
        poll_telegram_comments(state, args.dry_run)
        save_state(state)
        time.sleep(args.slurm_wait_poll_s)


def load_env(path: Path) -> dict[str, str]:
    env = {}
    for line in read(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(read(STATE_FILE))
    return {"cycle": 0, "last_report_ts": 0.0}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_agent(
    name: str,
    prompt: str,
    timeout_s: int,
    dry_run: bool = False,
    state: dict | None = None,
    telegram_poll_interval_s: int = 60,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{stamp}-{name}.log"
    cmd = [
        str(AGENTS[name]),
        "--search",
        "exec",
        "--skip-git-repo-check",
        "-",
    ]
    append(log_path, f"# {now()} {name}\n\n## Prompt\n{prompt}\n\n")
    if dry_run:
        append(log_path, "## Dry Run\nNot invoked.\n")
        return

    stdout_path = log_path.with_suffix(".stdout.tmp")
    stderr_path = log_path.with_suffix(".stderr.tmp")
    with stdout_path.open("w+", encoding="utf-8") as stdout_handle, stderr_path.open("w+", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        assert process.stdin is not None
        process.stdin.write(prompt)
        process.stdin.close()

        deadline = time.monotonic() + timeout_s
        next_poll = time.monotonic() + max(1, telegram_poll_interval_s)
        while process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                raise TimeoutError(f"{name} timed out after {timeout_s}s; see {log_path}")
            if state is not None and time.monotonic() >= next_poll:
                try:
                    poll_telegram_comments(state, dry_run)
                    save_state(state)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                next_poll = time.monotonic() + max(1, telegram_poll_interval_s)
            time.sleep(1)

        stdout_handle.seek(0)
        stderr_handle.seek(0)
        stdout = stdout_handle.read()
        stderr = stderr_handle.read()
        returncode = process.returncode
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    append(
        log_path,
        f"## Return Code\n{returncode}\n\n"
        f"## Stdout\n{stdout}\n\n"
        f"## Stderr\n{stderr}\n",
    )
    if returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {returncode}; see {log_path}")


def send_telegram_report(cycle_id: int, dry_run: bool = False) -> None:
    env = load_env(TELEGRAM_ENV)
    token = env.get("TELEGRAM_BOT_TOKEN")
    chat_id = env.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(f"Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in {TELEGRAM_ENV}")

    progress = read(ROOT / "progress.md").strip()
    tail = progress[-3200:] if progress else "No progress.md content yet."
    text = f"MindEye autonomous experiment report\nCycle: {cycle_id}\nTime: {now()}\n\n{tail}"
    data = urlencode({"chat_id": chat_id, "text": text[-3900:]}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    if dry_run:
        append(STATE_DIR / "telegram.log", f"[{now()}] dry-run report for cycle {cycle_id}\n")
        return
    request = Request(url, data=data, method="POST")
    with urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    append(STATE_DIR / "telegram.log", f"[{now()}] cycle {cycle_id}: {body}\n")


def poll_telegram_comments(state: dict, dry_run: bool = False) -> None:
    """Append inbound Telegram messages from the configured user chat to strategy chat."""
    env = load_env(TELEGRAM_ENV)
    token = env.get("TELEGRAM_BOT_TOKEN")
    chat_id = str(env.get("TELEGRAM_CHAT_ID", "")).strip()
    if not token or not chat_id:
        raise RuntimeError(f"Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in {TELEGRAM_ENV}")

    if dry_run:
        append(STATE_DIR / "telegram.log", f"[{now()}] dry-run inbound Telegram poll\n")
        return

    offset = state.get("telegram_update_offset")
    params = {"timeout": 0, "allowed_updates": json.dumps(["message"])}
    if offset is not None:
        params["offset"] = int(offset)
    url = f"https://api.telegram.org/bot{token}/getUpdates?{urlencode(params)}"
    with urlopen(url, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    payload = json.loads(body)
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram getUpdates failed: {payload}")

    updates = payload.get("result", [])
    if not updates:
        return

    max_update_id = max(int(update["update_id"]) for update in updates)
    state["telegram_update_offset"] = max_update_id + 1

    # First poll establishes the offset so a fresh restart does not import old chat history.
    if offset is None:
        append(STATE_DIR / "telegram.log", f"[{now()}] initialized Telegram update offset at {max_update_id + 1}\n")
        return

    for update in updates:
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        if str(chat.get("id")) != chat_id:
            continue
        text = message.get("text") or message.get("caption") or ""
        text = text.strip()
        if not text:
            continue
        sender = message.get("from") or {}
        sender_name = sender.get("username") or sender.get("first_name") or "telegram-user"
        timestamp = now()
        append(
            STRATEGY_CHAT,
            f"\n## User Telegram Comment - {timestamp}\n"
            f"From: {sender_name}\n\n"
            f"{text[:4000]}\n",
        )
        append(STATE_DIR / "telegram.log", f"[{timestamp}] imported inbound comment from Telegram update {update['update_id']}\n")


def git_auto_push(cycle_id: int, remote: str, branch: str, dry_run: bool = False) -> None:
    log_path = STATE_DIR / "git-push.log"
    if dry_run:
        append(log_path, f"[{now()}] dry-run git push for cycle {cycle_id}\n")
        return

    subprocess.run(["git", "add", "--", *AUTO_COMMIT_PATHS], cwd=ROOT, check=True)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
        check=False,
    )
    if diff.returncode == 0:
        append(log_path, f"[{now()}] cycle {cycle_id}: no staged changes\n")
        return
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8", errors="replace").split("\0")
    blocked = []
    for rel_path in [path for path in staged if path]:
        abs_path = ROOT / rel_path
        size = abs_path.stat().st_size if abs_path.exists() else 0
        if size > MAX_GITHUB_FILE_BYTES or rel_path.endswith(BLOCKED_GIT_PATH_SUFFIXES):
            blocked.append(f"{rel_path} ({size} bytes)")
    if blocked:
        subprocess.run(["git", "reset", "-q", "--", *[item.rsplit(" (", 1)[0] for item in blocked]], cwd=ROOT, check=True)
        raise RuntimeError(
            "Refusing to commit files that are too large for GitHub or are generated caches:\n"
            + "\n".join(blocked)
        )
    subprocess.run(
        ["git", "commit", "-m", f"Auto experiment cycle {cycle_id}"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "push", remote, f"HEAD:{branch}"], cwd=ROOT, check=True)
    append(log_path, f"[{now()}] cycle {cycle_id}: pushed HEAD to {remote}/{branch}\n")


def discussion_agreed(cycle_id: int) -> bool:
    chat = read(ROOT / "strategizing-chat.md")
    heading_re = re.compile(r"(?im)^##.*\bcycle\s+(\d+)\b.*$")
    headings = list(heading_re.finditer(chat))
    markers = []
    for index, heading in enumerate(headings):
        if int(heading.group(1)) != cycle_id:
            continue
        section_end = headings[index + 1].start() if index + 1 < len(headings) else len(chat)
        section = chat[heading.start():section_end]
        markers.extend(re.findall(r"(?im)^\s*(AGREE|DISAGREE)\s*:", section))
    return bool(markers) and markers[-1].upper() == "AGREE"


def cycle(args: argparse.Namespace, state: dict) -> None:
    cycle_id = int(state.get("cycle", 0)) + 1
    print(f"[{now()}] cycle {cycle_id} start", flush=True)

    wait_for_slurm_quiescence(args, state)
    poll_telegram_comments(state, args.dry_run)
    save_state(state)
    run_agent("bio", f"Cycle {cycle_id}: research phase. Update /workspace/myresearch.", args.agent_timeout_s, args.dry_run, state, args.telegram_poll_interval_s)
    poll_telegram_comments(state, args.dry_run)
    save_state(state)
    run_agent("cs", f"Cycle {cycle_id}: research phase. Update /workspace/myresearch.", args.agent_timeout_s, args.dry_run, state, args.telegram_poll_interval_s)
    poll_telegram_comments(state, args.dry_run)
    save_state(state)

    turn = 1
    while True:
        run_agent("bio", f"Cycle {cycle_id}, turn {turn}: append your discussion section to /strategizing-chat.md.", args.agent_timeout_s, args.dry_run, state, args.telegram_poll_interval_s)
        poll_telegram_comments(state, args.dry_run)
        save_state(state)
        run_agent("cs", f"Cycle {cycle_id}, turn {turn}: append your discussion section to /strategizing-chat.md.", args.agent_timeout_s, args.dry_run, state, args.telegram_poll_interval_s)
        poll_telegram_comments(state, args.dry_run)
        save_state(state)
        if discussion_agreed(cycle_id):
            break
        turn += 1

    try:
        CS_PLAN_DRAFT.unlink()
    except FileNotFoundError:
        pass
    run_agent("cs", f"Cycle {cycle_id}: write the final execution plan to /workspace/plan.md.next.", args.agent_timeout_s, args.dry_run, state, args.telegram_poll_interval_s)
    if not args.dry_run:
        promote_cs_plan_draft()
    poll_telegram_comments(state, args.dry_run)
    save_state(state)

    report_due = time.time() - float(state.get("last_report_ts", 0.0)) >= args.report_interval_s
    due_text = "Telegram report is due; append a concise report-ready update to /progress.md." if report_due else "Telegram report is not due."
    run_agent("orchestrator", f"Cycle {cycle_id}: execute /plan.md. {due_text}", args.orchestrator_timeout_s, args.dry_run, state, args.telegram_poll_interval_s)
    git_auto_push(cycle_id, args.git_remote, args.git_branch, args.dry_run)
    if report_due and not args.dry_run:
        send_telegram_report(cycle_id, args.dry_run)
        state["last_report_ts"] = time.time()

    state["cycle"] = cycle_id
    save_state(state)
    print(f"[{now()}] cycle {cycle_id} done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Log prompts without invoking agents.")
    parser.add_argument("--sleep-s", type=int, default=300)
    parser.add_argument("--agent-timeout-s", type=int, default=7200)
    parser.add_argument("--orchestrator-timeout-s", type=int, default=21600)
    parser.add_argument("--report-interval-s", type=int, default=43200)
    parser.add_argument("--telegram-poll-interval-s", type=int, default=60)
    parser.add_argument("--slurm-user", default=os.environ.get("USER", "aw1907"))
    parser.add_argument("--slurm-wait-poll-s", type=int, default=1800)
    parser.add_argument("--sacct-lookback-h", type=int, default=24)
    parser.add_argument("--no-wait-for-slurm", action="store_true", help="Start the next research cycle even if Slurm jobs are active.")
    parser.add_argument("--git-remote", default="git@github.com:aawingate1/MindEyeV2.git")
    parser.add_argument("--git-branch", default="codex2")
    args = parser.parse_args()

    STATE_DIR.mkdir(exist_ok=True)
    state = load_state()
    while True:
        cycle(args, state)
        if args.once:
            return 0
        time.sleep(args.sleep_s)


if __name__ == "__main__":
    raise SystemExit(main())
