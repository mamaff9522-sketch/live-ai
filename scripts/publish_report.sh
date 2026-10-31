#!/usr/bin/env bash
# Publishes a folder of test results to the branch "ci-reports" under <name>/,
# so results can be read with plain git (no artifact download needed).
# usage: publish_report.sh <folder> <name>
set -u
SRC="$1"; NAME="$2"
[ -d "$SRC" ] || { echo "no $SRC"; exit 0; }
REPO_URL="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
W=$(mktemp -d)
for attempt in 1 2 3 4 5; do
  rm -rf "$W" && mkdir -p "$W" && cd "$W"
  git init -q && git config user.name "live-ai-ci" && git config user.email "ci@users.noreply.github.com"
  if git fetch -q --depth 1 "$REPO_URL" ci-reports 2>/dev/null; then git checkout -q -b ci-reports FETCH_HEAD; else git checkout -q --orphan ci-reports; fi
  rm -rf "$NAME" && mkdir -p "$NAME" && cp -r "$GITHUB_WORKSPACE/$SRC/." "$NAME/"
  printf 'run %s\ncommit %s\njob %s\nstatus %s\n' "$GITHUB_RUN_ID" "$GITHUB_SHA" "$GITHUB_JOB" "${JOB_STATUS:-unknown}" > "$NAME/RUN.txt"
  git add -A && git commit -q -m "$NAME report for run $GITHUB_RUN_ID ($GITHUB_SHA)" || exit 0
  git push -q "$REPO_URL" HEAD:ci-reports && { echo "published $NAME"; exit 0; }
  sleep $((attempt * 3))
done
echo "could not publish report"
