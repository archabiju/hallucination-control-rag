#!/usr/bin/env bash
# TrustRAG one-command demo (macOS / Linux / WSL / Git Bash). On Windows PowerShell use run_demo.bat.
#
#   chmod +x run_demo.sh && ./run_demo.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1

if [ -f .venv/bin/activate ]; then source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate; fi
mkdir -p demo/outputs

echo "== 1/4 Environment and model/config versions =="
python demo/record_versions.py | tee demo/outputs/versions.txt

echo; echo "== 2/4 Build or reuse the FAISS index =="
python src/pipeline.py 2>&1 | tail -n 3

echo; echo "== 3/4 Demo queries (live pipeline + tri-state demonstration) =="
python demo/demo_run.py

echo; echo "== 4/4 Regenerate evaluation metrics (37 questions, 109 claims) =="
python evaluation/run_review2_eval.py --output-prefix demo_rerun | tee demo/outputs/eval_run.log

echo; echo "Done. Outputs:"
echo "  demo/outputs/versions.txt  sample_outputs.txt/.json  eval_run.log"
echo "  evaluation/results/demo_rerun_report.json  demo_rerun_tables.md"
