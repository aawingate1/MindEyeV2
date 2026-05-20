#!/usr/bin/env python3
"""Supervisor for the MindEye autonomous experiment loop."""

from __future__ import annotations

import argparse
import json
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


def run_agent(name: str, prompt: str, timeout_s: int, dry_run: bool = False) -> None:
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

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    append(
        log_path,
        f"## Return Code\n{result.returncode}\n\n"
        f"## Stdout\n{result.stdout}\n\n"
        f"## Stderr\n{result.stderr}\n",
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {log_path}")


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
    subprocess.run(
        ["git", "commit", "-m", f"Auto experiment cycle {cycle_id}"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "push", remote, f"HEAD:{branch}"], cwd=ROOT, check=True)
    append(log_path, f"[{now()}] cycle {cycle_id}: pushed HEAD to {remote}/{branch}\n")


def discussion_agreed(cycle_id: int) -> bool:
    chat = read(ROOT / "strategizing-chat.md")
    cycle_header = re.compile(rf"(?im)^##\s*Cycle\s+{cycle_id}\b")
    matches = list(cycle_header.finditer(chat))
    if not matches:
        return False
    cycle_text = chat[matches[0].start():]
    next_cycle = re.search(rf"(?im)^##\s*Cycle\s+(?!{cycle_id}\b)\d+\b", cycle_text[len(matches[0].group(0)):])
    if next_cycle:
        cycle_text = cycle_text[: len(matches[0].group(0)) + next_cycle.start()]
    markers = re.findall(r"(?im)^\s*(AGREE|DISAGREE)\s*:", cycle_text)
    return bool(markers) and markers[-1].upper() == "AGREE"


def cycle(args: argparse.Namespace, state: dict) -> None:
    cycle_id = int(state.get("cycle", 0)) + 1
    print(f"[{now()}] cycle {cycle_id} start", flush=True)

    run_agent("bio", f"Cycle {cycle_id}: research phase. Update /workspace/myresearch.", args.agent_timeout_s, args.dry_run)
    run_agent("cs", f"Cycle {cycle_id}: research phase. Update /workspace/myresearch.", args.agent_timeout_s, args.dry_run)

    for turn in range(1, args.max_turns + 1):
        run_agent("bio", f"Cycle {cycle_id}, turn {turn}: append your discussion section to /strategizing-chat.md.", args.agent_timeout_s, args.dry_run)
        run_agent("cs", f"Cycle {cycle_id}, turn {turn}: append your discussion section to /strategizing-chat.md.", args.agent_timeout_s, args.dry_run)
        if discussion_agreed(cycle_id):
            break

    run_agent("cs", f"Cycle {cycle_id}: write the final execution plan to /plan.md.", args.agent_timeout_s, args.dry_run)

    report_due = time.time() - float(state.get("last_report_ts", 0.0)) >= args.report_interval_s
    due_text = "Telegram report is due; append a concise report-ready update to /progress.md." if report_due else "Telegram report is not due."
    run_agent("orchestrator", f"Cycle {cycle_id}: execute /plan.md. {due_text}", args.orchestrator_timeout_s, args.dry_run)
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
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--report-interval-s", type=int, default=43200)
    parser.add_argument("--git-remote", default="git@github.com:aawingate1/MindEyeV2.git")
    parser.add_argument("--git-branch", default="codex")
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
