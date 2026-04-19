# Overleaf Sync Instructions

Your local LaTeX thesis files have been updated with the 98.12% F1-Score, 10 feature optimization, and missing subsections (e.g., Feature Importance). Because Overleaf is your primary text editor, here is exactly how to synchronize your project.

### 1. Finding the Modifications
Open your local `.tex` files inside the `theses/` directory using any text editor (like VS Code or Notepad). Press **Ctrl+F** and search for `OVERLEAF UPDATE REQUIRED`. 
This string marks exactly where I injected updates. Copy the text directly below that marker and paste it over the corresponding section in your Overleaf editor.

*(Note: In addition to the performance metrics, I formally corrected your Genetic Algorithm Penalty Equation in Chapter 3 to accurately scale linearly $(g/G)$ over time. Please ensure you copy those marked equation blocks in `theses/3-chap-method.tex`!)*

### 2. Uploading the New Figures
The structural updates in Chapter 4 require new images. Please locate your generated outputs folder (`/Thesis_Results/Chapter_4_Main_Results/`) and drag-and-drop the following 10 PNGs into your Overleaf images directory:

1. `fig6_feature_importance.png` 
2. `fig2_classifier_baseline_bar.png`
3. `fig5_confusion_matrix.png`
4. `fig4_pr_curve.png`
5. `fig3_roc_curve.png`
6. `kfold_boxplot.png`
7. `fig_feature_persistence.png`
8. `fig_efficiency_latency.png`
9. `fig_tradeoff_analysis.png`
10. `fig7_data_robustness_heatmap.png`

### 3. Inserting the New Tables into Chapter 4
I added two entirely new formal tables directly into `theses/4-chap-discussion.tex`:
- **Table 4.3 (Classifier Ablation)**: Copy the entire `\begin{table}` block located immediately below the `Model Algorithm Efficacy` marker.
- **Table 4.5 (Optimization Trade-off)**: Copy the entire `\begin{table}` block located immediately below the `Evolutionary Optimization Algorithms` marker. 

*(For Table 4.4, your existing class F1-Score table, simply update the `[Value]` placeholders with Iodine: 98.65, DNS2TCP: 99.86, and DNScat2: 97.50, exactly as marked in the `.tex` file.)*

### 4. Updating the Appendix
Finally, open your local `theses/appendices.tex`. 
Copy the two explicit LaTeX tables I constructed for **Appendix A1**:
1. `Dataset Robustness Heatmap simulation`
2. `Statistical Paired T-Tests (p-values vs SOTA)`

Paste these directly into your Overleaf Appendix to formally mathematically validate your structural robustness claims.
