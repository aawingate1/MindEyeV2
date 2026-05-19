#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$REPO_DIR/orchestrator-agent"
SRC_DIR="$REPO_DIR/src"
CODEX_BIN="/scratch/gpfs/KNORMAN/aw1907/tools/bin/codex"

mkdir -p "$AGENT_DIR/.codex-home"

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
  --ro-bind /run/munge /run/munge \
  --ro-bind /var/run/munge /var/run/munge \
  --ro-bind "$CODEX_BIN" /codex \
  --ro-bind "$REPO_DIR/plan.md" /plan.md \
  --bind "$REPO_DIR/progress.md" /progress.md \
  --bind "$AGENT_DIR" /workspace \
  --bind "$SRC_DIR" /src \
  --chdir /workspace \
  --setenv HOME /workspace \
  --setenv CODEX_HOME /workspace/.codex-home \
  --setenv PATH /usr/local/bin:/usr/bin:/bin \
  /codex \
  --cd /workspace \
  --add-dir /src \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  "$@"
