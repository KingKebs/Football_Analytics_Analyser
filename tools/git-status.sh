#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: tools/git-status.sh [options]

Options:
  --stage-tracked      Stage tracked file modifications/deletions (git add -u)
  --stage-new          Stage new relevant files (code/docs only; excludes data/ etc.)
  --stage-all          Stage both tracked and new relevant files
  --commit "msg"       Commit staged changes with message
  --dry-run            Show what would be staged/committed without changing git index
  -h, --help           Show this help

Relevance rules for new files:
  Include:  *.py, *.sh, *.md, *.txt, *.yaml, *.yml, *.toml, *.json (top-level config), tests/*, tools/*
  Exclude:  data/**, football-data/**, logs/**, tmp/**, visuals/**, __pycache__/**, venv/**, .venv/**, *.pdf
USAGE
}

DRY_RUN=false
DO_STAGE_TRACKED=false
DO_STAGE_NEW=false
COMMIT_MSG=""

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-tracked) DO_STAGE_TRACKED=true; shift ;;
    --stage-new)     DO_STAGE_NEW=true; shift ;;
    --stage-all)     DO_STAGE_TRACKED=true; DO_STAGE_NEW=true; shift ;;
    --commit)        COMMIT_MSG=${2-}; shift 2 ;;
    --dry-run)       DRY_RUN=true; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repository" >&2
  exit 1
fi

echo "# Repo: $(basename "$PWD")"
branch_line=$(git status -sb 2>/dev/null | head -n1 || true)
echo "$branch_line"

# Staged
mapfile -t staged < <(git diff --name-only --cached || true)
# Unstaged (modified, tracked)
mapfile -t unstaged < <(git diff --name-only || true)
# Untracked
mapfile -t untracked < <(git ls-files -o --exclude-standard || true)
# Ignored (top-level only)
ignored_raw=$(git status --ignored -s 2>/dev/null || true)
if [[ -n "$ignored_raw" ]]; then
  mapfile -t ignored < <(printf "%s\n" "$ignored_raw" | awk '/^!! /{sub(/^!! /,""); print}')
else
  ignored=()
fi

# Helper to filter untracked new files to only relevant code/docs
filter_relevant_new() {
  local f
  while IFS= read -r f; do
    # Exclusions by directory
    [[ "$f" == data/* ]] && continue
    [[ "$f" == football-data/* ]] && continue
    [[ "$f" == logs/* ]] && continue
    [[ "$f" == tmp/* ]] && continue
    [[ "$f" == visuals/* ]] && continue
    [[ "$f" == __pycache__/* ]] && continue
    [[ "$f" == venv/* || "$f" == .venv/* ]] && continue
    [[ "$f" == *.pdf ]] && continue
    # Include patterns
    if [[ "$f" == tools/* || "$f" == tests/* ]]; then
      printf '%s\n' "$f"; continue
    fi
    case "$f" in
      *.py|*.sh|*.md|*.txt|*.yaml|*.yml|*.toml|*.ini|*.cfg) printf '%s\n' "$f" ;;
      # Allow top-level json only (configs), not under data/
      *.json)
        if [[ "$f" != */* ]]; then printf '%s\n' "$f"; fi ;;
    esac
  done
}

# Optional staging actions
to_add=()
if $DO_STAGE_TRACKED; then
  echo
  echo "[Action] Stage tracked modifications (git add -u)"
  if $DRY_RUN; then
    echo "DRY-RUN: would run: git add -u"
  else
    git add -u
  fi
fi

if $DO_STAGE_NEW; then
  echo
  echo "[Action] Stage new relevant files"
  if ((${#untracked[@]} > 0)); then
    mapfile -t relevant_new < <(printf '%s\n' "${untracked[@]}" | filter_relevant_new)
  else
    relevant_new=()
  fi
  echo "Found ${#relevant_new[@]} new relevant files"
  if ((${#relevant_new[@]} > 0)); then
    if $DRY_RUN; then
      printf 'DRY-RUN: would add:\n'; for f in "${relevant_new[@]}"; do echo "  + $f"; done
    else
      git add -- "${relevant_new[@]}"
    fi
  fi
fi

# Refresh staged list after actions
mapfile -t staged < <(git diff --name-only --cached || true)

# Commit if requested and there is something staged
if [[ -n "$COMMIT_MSG" ]]; then
  echo
  if ((${#staged[@]} == 0)); then
    echo "No staged changes to commit. Skipping commit."
  else
    if $DRY_RUN; then
      echo "DRY-RUN: would commit ${#staged[@]} files with message: $COMMIT_MSG"
    else
      git commit -m "$COMMIT_MSG"
    fi
  fi
fi

# Summary output
echo
printf "Staged: %d\n" "${#staged[@]}"
for f in "${staged[@]}"; do echo "  + $f"; done

echo
printf "Unstaged: %d\n" "${#unstaged[@]}"
for f in "${unstaged[@]}"; do echo "  ~ $f"; done

echo
printf "Untracked: %d\n" "${#untracked[@]}"
for f in "${untracked[@]}"; do echo "  ? $f"; done

echo
printf "Ignored: %d\n" "${#ignored[@]}"
for f in "${ignored[@]}"; do echo "  ! $f"; done

echo
if [ ${#unstaged[@]} -gt 0 ]; then
  echo "Hint: stage tracked changes -> tools/git-status.sh --stage-tracked"
fi
if [ ${#untracked[@]} -gt 0 ]; then
  echo "Hint: stage new relevant files -> tools/git-status.sh --stage-new"
fi
