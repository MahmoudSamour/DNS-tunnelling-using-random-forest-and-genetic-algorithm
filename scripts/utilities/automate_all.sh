#!/bin/bash
set -e

# Always run from the project root
cd "$(dirname "$0")/../../"
ROOT_DIR=$(pwd)
echo "Starting Automation from: $ROOT_DIR"

source .venv/bin/activate

# Add root to PYTHONPATH so scripts in subdirs can find utils/ and models/
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

# 0. Run heavy master evaluation
echo "Running heavy_master_evaluation.py..."
python3 scripts/evaluation/heavy_master_evaluation.py

# 1. Run master_evaluation.py
echo "Running master_evaluation.py..."
python3 scripts/evaluation/master_evaluation.py

# 2. Run Notebooks
echo "Converting notebooks to markdown..."
NB_DIR="Thesis_Results/Notebook_Executions"
mkdir -p "$NB_DIR"

notebooks=(
  "notebooks/main_experiments/main_evaluation_pipeline.ipynb"
  "notebooks/collaborations/mohsin_rl_features.ipynb"
  "notebooks/collaborations/mohsin_feature_engineering.ipynb"
  "notebooks/main_experiments/legacy_dns_tunnelling.ipynb"
  "notebooks/benchmarks/optimization_benchmarks.ipynb"
  "notebooks/main_experiments/full_security_analysis.ipynb"
)

for nb in "${notebooks[@]}"; do
  echo "Executing and converting: $nb"
  # Sometimes nbconvert fails on timeouts or minor errors. We use --allow-errors to guarantee it finishes.
  # And --ExecutePreprocessor.timeout=-1 to prevent timeout.
  jupyter nbconvert --execute --allow-errors --ExecutePreprocessor.timeout=-1 --to markdown --output-dir="$NB_DIR" "$nb"
done

# 3. Organize Results
echo "Organizing Final Results..."
CHAP4_DIR="Thesis_Results/Chapter_4_Main_Results"
APPX_DIR="Thesis_Results/Appendix_Supplementary"

mkdir -p "$CHAP4_DIR" "$APPX_DIR"

# Copy from heavy_master_results
if [ -d "heavy_master_results" ]; then
    cp heavy_master_results/kfold_boxplot.png "$CHAP4_DIR/" 2>/dev/null || true
    cp heavy_master_results/*.csv "$APPX_DIR/" 2>/dev/null || true
    cp heavy_master_results/*.json "$APPX_DIR/" 2>/dev/null || true
fi

# Copy from master_figures
if [ -d "master_figures" ]; then
    cp master_figures/*.png "$CHAP4_DIR/" 2>/dev/null || true
    cp master_figures/table_feature_subset_comparison.csv "$CHAP4_DIR/" 2>/dev/null || true
    cp master_figures/table_classifier_ablation.csv "$CHAP4_DIR/" 2>/dev/null || true
    cp master_figures/table_dataset_robustness.csv "$APPX_DIR/" 2>/dev/null || true
fi

echo "Automation Completed Successfully!"
