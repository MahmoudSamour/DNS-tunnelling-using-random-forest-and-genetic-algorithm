#!/bin/bash
set -e

echo "Starting Remaining Automation (master_evaluation and notebooks)..."
source .venv/bin/activate

# 0. Run heavy master evaluation
echo "Running heavy_master_evaluation.py..."
python3 heavy_master_evaluation.py
# 1. Run master_evaluation.py
echo "Running master_evaluation.py..."
python3 master_evaluation.py

# 2. Run Notebooks
echo "Converting notebooks to markdown..."
NB_DIR="Thesis_Results/Notebook_Executions"
mkdir -p "$NB_DIR"

notebooks=(
  "DNS_Tunneling_Full_Evaluation.ipynb"
  "RL_features_Mohs.ipynb"
  "feature_engineering-Mohs.ipynb"
  "Enhanced detection of DNS tunnelling .ipynb"
  "Enhanced_Benchmark_Functions.ipynb"
  "Enhanced_detection_of_DNS_tunnelling_Leveraging_random_forest_and_genetic_algorithm_for_improved_security_2.ipynb"
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
