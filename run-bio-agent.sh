#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$ROOT_DIR/bio-agent"
CODEX_BIN="/scratch/gpfs/KNORMAN/aw1907/tools/bin/codex"
CODEX_HOME_DIR="${CODEX_HOME_DIR:-/home/aw1907/.codex}"

touch "$ROOT_DIR/job-status.md"

bwrap \
  --die-with-parent \
  --unshare-pid \
  --unshare-ipc \
  --unshare-uts \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --ro-bind /usr /usr \
  --ro-bind /bin /bin \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  --ro-bind /etc /etc \
  --ro-bind "$CODEX_BIN" /codex \
  --bind "$CODEX_HOME_DIR" /codex-home \
  --bind "$ROOT_DIR/strategizing-chat.md" /strategizing-chat.md \
  --ro-bind "$ROOT_DIR/plan.md" /plan.md \
  --ro-bind "$ROOT_DIR/progress.md" /progress.md \
  --ro-bind "$ROOT_DIR/job-status.md" /job-status.md \
  --bind "$AGENT_DIR" /workspace \
  --chdir /workspace \
  --setenv HOME /workspace \
  --setenv CODEX_HOME /codex-home \
  --setenv PATH /usr/local/bin:/usr/bin:/bin \
  /codex \
  --cd /workspace \
  --sandbox danger-full-access \
  --ask-for-approval on-request \
  "$@"
