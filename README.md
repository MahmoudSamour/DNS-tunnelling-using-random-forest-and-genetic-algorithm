# Enhanced Detection of DNS Tunneling
### Leveraging a Penalty-Based Genetic Algorithm and Random Forest for Real-Time Encrypted Traffic Analysis

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: CIRA-CIC-DoHBrw-2020](https://img.shields.io/badge/Dataset-CIRA--CIC--DoHBrw--2020-orange)](https://www.unb.ca/cic/datasets/dohbrw-2020.html)

> **PhD Research** · University of Technology Malaysia (UTM) · Faculty of Computing

---

## 🧠 Overview

Modern attackers exploit **DNS-over-HTTPS (DoH)** to hide malicious tunneling traffic inside legitimate encrypted web traffic, completely bypassing traditional Deep Packet Inspection (DPI). This research presents a hybrid framework that defeats this evasion technique by:

1. **Extracting behavioral flow features** (timing, size, frequency) that remain observable even when the payload is fully encrypted.
2. **Applying a Penalty-Based Genetic Algorithm (GA)** to autonomously identify the minimal optimal feature subset — reducing 34 candidate features down to just **10**, while maintaining a **98.12% weighted F1-Score**.
3. **Training a Random Forest classifier** on this parsimonious feature set to deliver lightweight, real-time detection suitable for edge firewall deployment.

---

## 🔬 Key Results at a Glance

| Metric | Proposed GA-RF | RF Baseline (All 32) | SOTA: Matrix-GA |
|---|---|---|---|
| **Weighted F1-Score** | **0.991** | 0.994 | 0.990 |
| **Feature Count** | **10** | 32 | 13 |
| **Train Time (s)** | **7.92** | 10.18 | 5.71 |
| **Latency (ms/pkt)** | **0.00033** | 0.00042 | 0.00031 |

> The proposed framework achieves near-identical accuracy to the full baseline while **reducing the operational feature footprint by 69%**, making it the only candidate viable for real-time high-throughput edge deployment.

---

## 📊 Visual Results

### Feature Subset Optimization Trade-Off
![Feature Optimization Trade-off](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig8_tradeoff.png)
*The Penalty-GA lands in the optimal top-left quadrant: highest F1-Score at minimal feature count.*

### Feature Importance (Gini Impurity)
![Feature Importance](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig6_feature_importance.png)
*Volumetric and temporal features dominate the RF decision boundaries, proving resilience against TLS masking.*

### Multi-Class Confusion Matrix
![Confusion Matrix](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig5_confusion_matrix.png)
*Zero false positives on Benign traffic — critical for enterprise firewall deployment.*

### Classifier Comparison (RF vs LR vs DT)
![Classifier Comparison](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig2_classifier_baselines.png)
*Random Forest outperforms all baseline classifiers on the same 10-feature subset.*

### ROC Curve Analysis
![ROC Curve](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig3_roc_curve.png)
*AUC = 1.0000 for Benign class. Malicious classes: Iodine (0.9993), DNS2TCP (0.9997), DNScat2 (0.9992).*

### Precision-Recall Analysis
![PR Curve](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig4_pr_curve.png)
*Mean Average Precision (mAP): Benign = 1.000 | Iodine = 0.987 | DNS2TCP = 0.999 | DNScat2 = 0.975.*

### Dataset Robustness Test (Class Starvation)
![Robustness Heatmap](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig7_robustness.png)
*Model sustains >96% accuracy even when specific tunneling tool classes are fully removed from training.*

---

## 🏗️ Repository Structure

```
.
├── models/               # GA variants and ML classifiers
│   ├── base_ga.py        # Standard GA baseline
│   ├── matrix_ga.py      # SOTA Matrix-GA (SOTA comparison)
│   ├── jaya_ga.py        # SOTA JAYA-GA (SOTA comparison)
│   ├── pps_ga.py         # Penalty-based GA (PROPOSED)
│   └── rf_evaluator.py   # Random Forest fitness wrapper
├── utils/
│   ├── data_loader.py    # CIRA-CIC-DoHBrw-2020 loader & preprocessor
│   ├── penalty_funcs.py  # Adaptive penalty λ_g formulas
│   └── viz_and_stats.py  # Plotting and statistical tests
├── tests/                # Unit tests for all core modules
├── master_evaluation.py  # Main evaluation pipeline
├── heavy_master_evaluation.py  # Full 10-fold CV and GA runs
├── generate_supplementary_figures.py  # Supplementary plot generator
└── requirements.txt
```

---

## ⚙️ Methodology

### The Penalty-Based Fitness Function

The GA optimizes feature chromosomes $\mathbf{x} \in \{0,1\}^N$ using a composite fitness function that **simultaneously maximizes classification accuracy and minimizes feature count**:

$$F(\mathbf{x}) = M(\mathbf{x}) - \left[ \lambda_{base} \left( \frac{g}{G_{max}} \right) \left( \frac{k}{N} \right) \right]$$

Where:
- $M(\mathbf{x})$ = Weighted F1-Score of the Random Forest on feature subset $\mathbf{x}$
- $\lambda_{base}$ = Maximum penalty weight
- $g / G_{max}$ = Linear generation progress (penalty **grows** over time)
- $k / N$ = Proportion of selected features

### Key Configuration

| Component | Setting |
|---|---|
| Random Forest | 100 trees, `random_state=42` |
| GA Population | 100 chromosomes |
| GA Generations | 100 |
| Crossover Probability | 0.80 |
| Mutation Probability | 0.10 |
| Evaluation Split | 50% Train / 50% Test |
| Dataset | CIRA-CIC-DoHBrw-2020 (1.1M flows) |

---

## 🚀 Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm.git
cd DNS-tunnelling-using-random-forest-and-genetic-algorithm

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the main evaluation pipeline
python master_evaluation.py

# 5. Run the full heavy evaluation (10-fold CV + GA — takes ~30 min)
python heavy_master_evaluation.py
```

---

## 📚 Dataset

**CIRA-CIC-DoHBrw-2020** — University of New Brunswick

| Traffic Class | Samples | Proportion |
|---|---|---|
| Benign (HTTPS) | ~860,000 | ~78.5% |
| Iodine | 5,704 | 3.9% |
| DNS2TCP | 21,003 | 14.4% |
| DNScat2 | 4,503 | 3.1% |

The dataset is not included in this repository due to its size (~500 MB). Download it from the [official CIC page](https://www.unb.ca/cic/datasets/dohbrw-2020.html) and place the CSVs in `DoHBrw-2020/`.

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@phdthesis{sammour2024dns,
  title   = {Enhanced Detection of DNS Tunneling: Leveraging a Penalty-Based
             Genetic Algorithm and Random Forest for Real-Time Security},
  author  = {Sammour, Mahmoud},
  school  = {Universiti Teknologi Malaysia},
  year    = {2025}
}
```

---

## 👥 Research Team

| Researcher | Role | ORCID |
|---|---|---|
| **Mahmoud Sammour** | Principal Researcher, Methodology, Implementation | [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--6860--2804-green)](https://orcid.org/0000-0002-6860-2804) |
| **Mohd Fairuz Iskandar Othman** | Principal Supervisor, Research Direction | — |
| **Aslinda Hassan** | Co-Supervisor, Validation | — |
| **Mohsin Ali** | Collaborator, Feature Engineering & RL Pipeline | [![ORCID](https://img.shields.io/badge/ORCID-0009--0006--3101--5194-green)](https://orcid.org/0009-0006-3101-5194) |
