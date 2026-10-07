#!/usr/bin/env bash
# pre-push leak guard for my-rag-toolkit
# scans outgoing commits (messages + diffs) against a LOCAL denylist and
# blocks the push on any match. the denylist itself is never committed.
#
# denylist: $MYRAG_GUARD_DENYLIST, else ~/.config/myrag/push-guard-denylist.txt
# one extended-regex term per line, case-insensitive; '#' starts a comment.
# a missing/empty denylist allows the push (with a warning).

set -uo pipefail

DENYLIST="${MYRAG_GUARD_DENYLIST:-$HOME/.config/myrag/push-guard-denylist.txt}"

if [ ! -s "$DENYLIST" ]; then
  echo "push-guard: denylist missing or empty ($DENYLIST) — push allowed" >&2
  echo "push-guard: install the denylist to enable the guard" >&2
  exit 0
fi

pattern=$(grep -vE '^[[:space:]]*(#|$)' "$DENYLIST" | paste -sd '|' -)
if [ -z "$pattern" ]; then
  echo "push-guard: denylist has no active patterns — push allowed" >&2
  exit 0
fi

hits=0

scan() {
  local label="$1" text="$2" matches
  [ -n "$text" ] || return 0
  matches=$(printf '%s\n' "$text" | grep -inE "$pattern" 2>/dev/null || true)
  if [ -n "$matches" ]; then
    echo "push-guard: BLOCKED — denylist terms found in $label:" >&2
    printf '%s\n' "$matches" | head -10 >&2
    hits=1
  fi
}

while read -r local_ref local_sha remote_ref remote_sha; do
  [ -n "${local_sha:-}" ] || continue
  if [ "$remote_sha" = "0000000000000000000000000000000000000000" ]; then
    text=$(git log -p --no-merges "$local_sha" --not --remotes 2>/dev/null || true)
  else
    text=$(git log -p --no-merges "$remote_sha..$local_sha" 2>/dev/null || true)
  fi
  scan "outgoing commits of $local_ref" "$text"
done

if [ "$hits" -ne 0 ]; then
  echo "push-guard: review the flagged content (denylist: $DENYLIST). Push refused." >&2
  exit 1
fi
exit 0
