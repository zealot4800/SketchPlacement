#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="cover"
if [[ $# -gt 0 && ( "$1" == "greedy" || "$1" == "cover" ) ]]; then
  MODEL="$1"
  shift
fi

INPUT="${1:-../dataset/out/paths.jsonl.gz}"

if [[ "$MODEL" == "greedy" ]]; then
  OUT_DIR="${OUT_DIR:-out/run_greedy_$(date +%H%M%S)}"
  cmd=(python -m greedy.main --input "$INPUT" --out-dir "$OUT_DIR")
elif [[ "$MODEL" == "cover" ]]; then
  OUT_DIR="${OUT_DIR:-out/run_cover_$(date +%H%M%S)}"
  cmd=(python -m main --input "$INPUT" --out-dir "$OUT_DIR")
fi

echo "[run] ${cmd[*]}"
"${cmd[@]}"
