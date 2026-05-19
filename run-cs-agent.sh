#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$AGENT_DIR/cs-agent"
CODEX_BIN="/scratch/gpfs/KNORMAN/aw1907/tools/bin/codex"
CODEX_HOME_DIR="${CODEX_HOME_DIR:-/home/aw1907/.codex}"

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
  --bind "$(dirname "$AGENT_DIR")/strategizing-chat.md" /strategizing-chat.md \
  --bind "$(dirname "$AGENT_DIR")/plan.md" /plan.md \
  --ro-bind "$(dirname "$AGENT_DIR")/progress.md" /progress.md \
  --bind "$AGENT_DIR" /workspace \
  --chdir /workspace \
  --setenv HOME /workspace \
  --setenv CODEX_HOME /codex-home \
  --setenv PATH /usr/local/bin:/usr/bin:/bin \
  /codex \
  --cd /workspace \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  "$@"
