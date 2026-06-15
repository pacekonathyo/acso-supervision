#!/usr/bin/env bash
# Reproduce every number and figure in the paper.
set -e
cd "$(dirname "$0")/src"
echo ">> Main experiments (recommendation + tracking) and data export"
python3 evaluate.py
echo ">> Graph-embedding baseline"
python3 graph_embedding_baseline.py
echo ">> Sensitivity and robustness analysis"
python3 sensitivity.py
echo ">> Figures"
python3 make_figures.py
echo "Done. See ../data and ../figures."
