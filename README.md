<div align="center">

# 🛡️ Cracking the Code: Detecting Hidden Tunnels Inside Encrypted DNS Traffic
### A PhD Research Story — Random Forest × Genetic Algorithm × DoH Tunneling

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-CIRA--CIC--DoHBrw--2020-orange?style=for-the-badge)](https://www.unb.ca/cic/datasets/dohbrw-2020.html)
[![Status](https://img.shields.io/badge/Status-PhD%20Thesis%20Complete-brightgreen?style=for-the-badge)](https://github.com/MahmoudSamour)

---

**🎬 Built for the YouTube Explainer Series:** *"From Zero to PhD: Building an AI Firewall"*

</div>

---

## 🎬 The Story: A Hacker's Perfect Hiding Spot

Imagine you are a network security engineer monitoring all traffic at your company's firewall. You check your logs — everything looks clean. Only normal HTTPS web requests. Your security tools give you a green light. ✅

**But your company's most sensitive data is silently being stolen. Right now. In plain sight.**

The attacker is using a technique called **DNS Tunneling over HTTPS (DoH)**. They are hiding stolen files inside the thousands of routine DNS queries your computer makes every second — completely encrypted, completely invisible to your current tools.

This is the problem this PhD research was built to solve.

---

## 🔍 Chapter 1: The Attack — What is DNS Tunneling?

DNS (Domain Name System) is the internet's phonebook. Every time you visit a website, your computer sends a DNS query asking *"what is the IP address of google.com?"*

Normally harmless. But attackers realized something dangerous:

> **DNS queries are never blocked. And now, thanks to DNS-over-HTTPS (DoH), they are never inspectable either.**

Tools like **Iodine**, **DNS2TCP**, and **DNScat2** exploit this to create covert data channels that bypass every traditional security tool.

```
[ ATTACKER ]──── stolen data ────▶[ DNS-over-HTTPS ]────▶[ Command & Control Server ]
                                          ↑
                              Looks like normal web traffic
                            Deep Packet Inspection = BLIND 🙈
```

**The result?** Data breaches, ransomware deployments, and APT persistent access — all hidden inside traffic your firewall waves through with a smile.

---

## 🧬 Chapter 2: The Solution — Teaching AI to See What Humans Can't

Since we cannot read the encrypted payload, we have to look at **behavior patterns** in the traffic flow itself:

- 🕐 How often are packets sent? (Inter-Arrival Time)
- 📦 How large are they? (Packet Length Distribution)
- 🔄 What is the ratio of inbound to outbound traffic?
- ⚡ How many bytes per second flow through the connection?

Even through HTTPS encryption, **automated tunneling tools leave a behavioral fingerprint** that is statistically distinguishable from normal human web browsing.

The challenge: there are **34 such features**. Using all of them in a real-time firewall would be too slow. 

**So we built an AI that learns which 10 features matter most — automatically.**

---

## 🏗️ Chapter 3: The Architecture — Two AIs Working Together

<div align="center">

```
┌─────────────────────────────────────────────────────────┐
│              GA-RF HYBRID DETECTION FRAMEWORK           │
│                                                         │
│   ┌─────────────────────┐    ┌──────────────────────┐  │
│   │   GENETIC ALGORITHM │    │   RANDOM FOREST      │  │
│   │                     │───▶│                      │  │
│   │  "Which features    │    │  "Given these        │  │
│   │   matter most?"     │    │   features, is this  │  │
│   │                     │◀───│   traffic malicious?"│  │
│   │  Evolves 100 pop.   │    │   100 trees, tuned   │  │
│   │  over 100 gens.     │    │   for imbalanced     │  │
│   └─────────────────────┘    └──────────────────────┘  │
│              ↕                          ↕               │
│         10 Features              99.75% F1-Score        │
└─────────────────────────────────────────────────────────┘
```

</div>

### 🧠 The Genetic Algorithm: Nature-Inspired Feature Selection

The GA works like natural selection for feature subsets. Each "chromosome" is a binary string representing which features are included:

```
Chromosome example:
[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, ...]
 ↑       ↑       ↑  ↑     ↑     ↑        ↑
Use   Skip    Use  Use  Use   Use      Use
F1         F3   F4   F6   F8       F11   ...
```

But here's the **novel innovation** — a **dynamic penalty function** that gets progressively stricter over time:

$$F(\mathbf{x}) = M(\mathbf{x}) - \underbrace{\left[ \lambda_{base} \cdot \frac{g}{G_{max}} \cdot \frac{k}{N} \right]}_{\text{Penalty grows over generations}}$$

- **Early generations**: Low penalty → GA freely explores all feature combinations
- **Late generations**: High penalty → GA is forced to keep only essential features

This is why we call it *"Adaptive"* — the algorithm mathematically forces its own simplification.

---

## 🔄 Chapter 3.5: See It To Believe It — System Diagrams

### The End-to-End Pipeline

![Methodology Flowchart](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/flowchart_methodology.png)

*Four stages: raw DoH packet capture → behavioral feature extraction → GA-driven feature selection → RF multi-class classification.*

### Inside the Genetic Algorithm Loop

![GA Flowchart](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/penalty_based_ga_flowchart.png)

*The penalty term $\lambda_g = \lambda_{base} \cdot (g/G)$ grows with every generation — early freedom, late discipline.*

### Dataset: How Attack Traffic is Distributed

![Attack Distribution](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/Attack%20Type%20Distribution%20Over%20Time.png)

*The 78% Benign majority creates a severe class imbalance — which is why Weighted F1-Score (not raw Accuracy) is the correct evaluation metric.*

---

## 📊 Chapter 4: The Results — Numbers That Speak

### 4.1 — The Feature Importance Ranking

Before we even run the optimizer, which features does the Random Forest itself consider most valuable?

![Feature Importance](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig6_feature_importance.png)

> **🎬 Video Note:** Point out how `Flow Bytes/s`, `Fwd Packet Length`, and `Flow IAT` dominate. These are timing and volume features — invisible to DPI but readable from encrypted HTTPS metadata.

**Key insight:** The top features are all behavioral, not content-based. This is *exactly* why our approach works even on fully encrypted traffic.

---

### 4.2 — The Optimization Trade-Off: Less is More

The central claim of this research is that **using fewer features does not hurt performance — it improves real-world viability**.

![Feature Optimization Trade-off](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig8_tradeoff.png)

| Strategy | Features | F1-Score | Train Time | Latency |
|:---|:---:|:---:|:---:|:---:|
| 🔴 Baseline (All Features) | 32 | 0.993 | 10.2s | 0.00041 ms/pkt |
| 🟡 SOTA: Matrix-GA | 13 | 0.990 | 5.7s | 0.00030 ms/pkt |
| 🟡 SOTA: JAYA-GA | 17 | 0.989 | 10.0s | 0.00032 ms/pkt |
| 🟢 **Proposed Penalty-GA** | **10** | **0.9975** | **6.9s** | **0.00032 ms/pkt** |

> **🎬 Video Note:** The scatter plot shows our method landing in the top-left "sweet spot" — best accuracy-to-feature ratio. Use this as your "hero slide" in the thumbnail.

---

### 4.3 — Choosing the Right Classifier

Why Random Forest and not Logistic Regression or a simple Decision Tree?

![Classifier Comparison](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig2_classifier_baselines.png)

| Algorithm | F1-Score | Accuracy | Balanced Acc |
|:---|:---:|:---:|:---:|
| 🏆 **Random Forest (Proposed)** | **0.9975** | **0.9975** | **0.9931** |
| Deep ANN (MLP) | 0.9944 | 0.9944 | 0.9845 |
| KNN-Optimized | 0.9935 | 0.9935 | 0.9819 |
| Decision Tree | 0.9962 | 0.9962 | 0.9893 |
| SVM (RBF-Approx) | 0.9565 | 0.9549 | 0.8763 |
| Logistic Regression | 0.9402 | 0.9390 | 0.8339 |
| Naive Bayes | 0.9340 | 0.9357 | 0.8296 |

> **Why RF wins:** Ensemble bagging reduces variance. Each of the 100 trees votes independently — no single noisy feature can derail the prediction. Logistic Regression collapses on the non-linear, multi-class distribution of DoH traffic.

---

### 4.4 — The Confusion Matrix: Zero Compromise on Benign Traffic

The most dangerous failure mode of any IDS is blocking legitimate traffic. Executives get fired over firewalls that kill VoIP calls during board meetings.

![Confusion Matrix](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig5_confusion_matrix.png)

**What this matrix proves:**
- ✅ **Benign traffic**: 100% correctly identified — zero business disruption
- ✅ **Iodine**: Caught with 98.11% recall
- ✅ **DNS2TCP**: Caught with 99.12% recall
- ✅ **DNScat2**: Caught with 96.88% recall (most evasive — heavy C2 encryption)

| Traffic Class | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|
| Benign | 99.99% | 100.00% | **100.00%** |
| Iodine | 97.77% | 98.72% | **98.24%** |
| DNS2TCP | 99.51% | 99.04% | **99.27%** |
| DNScat2 | 97.40% | 98.18% | **97.79%** |
| **Weighted Average** | **99.75%** | **99.75%** | **99.75%** |

---

### 4.5 — Precision-Recall Curves: No Accuracy Illusions

Standard accuracy is misleading when your dataset has 78% benign traffic. PR curves tell the real story.

![PR Curve](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig4_pr_curve.png)

| Class | mAP Score | Interpretation |
|:---|:---:|:---|
| Benign | 1.0000 | Perfect — no legitimate traffic misclassified |
| Iodine | 0.9865 | Near-perfect detection |
| DNS2TCP | 0.9986 | Exceptional |
| DNScat2 | 0.9750 | Strong despite heavy obfuscation |

---

### 4.6 — ROC Curves: The Gold Standard Validation

![ROC Curve](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig3_roc_curve.png)

AUC scores confirm the model's discriminatory power across all confidence thresholds:

| Class | AUC | Status |
|:---|:---:|:---:|
| Benign | 1.0000 | 🟢 Perfect |
| Iodine | 0.9993 | 🟢 Exceptional |
| DNS2TCP | 0.9997 | 🟢 Exceptional |
| DNScat2 | 0.9992 | 🟢 Exceptional |

---

### 4.7 — Stress Test: What Happens When You Remove an Entire Threat Class?

We simulated a **concept drift scenario** — deploying the model in an environment where it has never seen one of the tunneling tools.

![Robustness Heatmap](https://raw.githubusercontent.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/main/media/fig7_robustness.png)

| Scenario | Accuracy | F1-Score |
|:---|:---:|:---:|
| Normal (Full Mix) | 99.1% | 99.1% |
| Trained without Iodine data | 99.4% | 99.5% |
| Trained without DNS2TCP data | 99.4% | 99.4% |
| Trained without DNScat2 data | 99.2% | 99.5% |

> **The model does not memorize tools — it learns behavioral physics.** Even without seeing Iodine during training, it correctly identifies the behavioral temporal rhythm that distinguishes automated tunneling from human traffic.

---

## 🔬 Chapter 5: Under the Hood — The Technical Deep Dive

### Dataset Breakdown

**CIRA-CIC-DoHBrw-2020** — collected by the Canadian Institute for Cybersecurity

```
🌐 Total Records: ~1,100,000 network flows

┌─────────────────────────────────────┐
│  Benign (HTTPS)    ████████████ 78.5%│
│  DNS2TCP           ██           14.4%│
│  Iodine            █             3.9%│
│  DNScat2           ▌             3.1%│
└─────────────────────────────────────┘
```

> This extreme class imbalance is why standard Accuracy is useless here and why we use **Weighted F1-Score** as our primary metric.

### The 10 Winning Features

After 100 generations of evolution, the Genetic Algorithm converged on these features as the minimal sufficient set:

| Feature ID | Name | Why It Matters |
|:---:|:---|:---|
| F1 | Flow Duration | Tunnels maintain persistent long-lived connections |
| F6 | Fwd Packet Length Max | Large max = data exfiltration chunk |
| F8 | Fwd Packet Length Mean | Tunneling tools chunk data consistently |
| F12 | Bwd Packet Length Mean | C2 command response size reveals server type |
| F14 | Flow Bytes/s | Throughput anomaly is the loudest signal |
| F16 | Flow IAT Mean | Automated tools have machine-precise timing |
| F17 | Flow IAT Std | Low variance = automated beaconing |
| F20 | Fwd IAT Total | Total outbound timing exposes automation |
| F30 | Fwd Header Length | Anomalous header construction patterns |
| F34 | Information Entropy | Encrypted/obfuscated payload marker |

### Reproducibility Configuration

```python
# Exact settings used in experiments
GA_CONFIG = {
    "population_size": 100,        # chromosomes per generation
    "max_generations": 100,        # evolutionary pressure duration
    "crossover_probability": 0.80, # two-point crossover
    "mutation_probability": 0.10,  # bit-flip mutation
    "lambda_base": 0.1,            # max penalty weight
    "random_state": 42             # fully reproducible
}

RF_CONFIG = {
    "n_estimators": 100,           # number of trees
    "criterion": "gini",           # impurity measure
    "random_state": 42
}

EVAL_CONFIG = {
    "train_test_split": 0.50,      # 50/50 to stress-test on large holdout
    "scoring": "f1_weighted"       # handles class imbalance correctly
}
```

---

## 🚀 Chapter 6: Run It Yourself

```bash
# 1. Clone
git clone https://github.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm.git
cd DNS-tunnelling-using-random-forest-and-genetic-algorithm

# 2. Set up environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Place your dataset CSVs in DoHBrw-2020/

# 4. Quick evaluation run (~5 min)
python master_evaluation.py

# 5. Full PhD-grade evaluation with 10-fold CV + GA (~30 min)
python heavy_master_evaluation.py
```

**Output directories generated:**
- `Thesis_Results/Chapter_4_Main_Results/` — all publication-ready figures and tables
- `Thesis_Results/Appendix_Supplementary/` — statistical tests, raw k-fold scores, feature indices

---

## 🏆 Chapter 7: Why This Matters Beyond the Lab

| Real-World Scenario | How This Framework Helps |
|:---|:---|
| 🏦 **Bank internal network** | Catches insider threat exfiltrating transaction data over DoH at the perimeter |
| 🏥 **Hospital HIPAA compliance** | Detects patient data leaving via encrypted DNS — invisible to legacy DLP tools |
| 🏭 **Industrial IoT (OT networks)** | Lightweight 10-feature profile fits on edge devices with limited compute |
| 🏛️ **Government classified networks** | APT groups using DNScat2 for C2 — detected with 97.19% F1 |
| ☁️ **Cloud egress monitoring** | Deployed as a lambda/serverless function — 0.33ms latency allows wire-speed analysis |

---

## 📚 Research Background & Citation

### Related Work This Builds On

| Paper | Method | Gap Addressed |
|:---|:---|:---|
| Talabani (2025) | Filter-Wrapper Hybrid | Adds latency from multi-stage ANN |
| Banadaki (2020) | ML Ensembles (LGBM) | No automated feature count reduction |
| Alemu (2025) | GA + KNN | GA fitness ignores feature subset size |
| Abualghanam (2023) | Hybrid PIO + ML | High overhead in real-time environments |
| **This Work** | **Penalty-GA + RF** | **Mathematically enforces parsimony at every generation** |

### Cite This Work

```bibtex
@phdthesis{sammour2025dns_tunneling,
  title   = {Enhanced Detection of DNS Tunneling: Leveraging a Penalty-Based
             Genetic Algorithm and Random Forest for Real-Time Security},
  author  = {Sammour, Mahmoud},
  school  = {Universiti Teknologi Malaysia (UTM)},
  year    = {2025},
  note    = {Faculty of Computing}
}
```

---

## 👥 Research Team

<div align="center">

| <img src="media/author1.png" width="120"> | <img src="media/author2.png" width="120"> | <img src="media/author3.png" width="120"> |
|:---:|:---:|:---:|
| **Mahmoud Sammour** | **Dr. Mohd Fairuz** | **Mohsin Ali** |
| [ORCID](https://orcid.org/0000-0002-6860-2804) | Research Direction | [ORCID](https://orcid.org/0009-0006-3101-5194) |

| <img src="media/author4.png" width="120"> | <img src="media/author5.png" width="120"> | |
|:---:|:---:|:---:|
| **Dr. Aslinda Hassan** | **Prof. M. Hanafi** | |
| Theoretical Validation | External Validation | |

</div>

---

<div align="center">

### 🎬 YouTube Explainer Series Coming Soon

*"From Zero to PhD: Building an AI Firewall Against Encrypted DNS Tunneling"*

**Subscribe to follow the full breakdown — EVERY figure in this README becomes an animated explainer slide.**

---

*Built with ❤️ and 1,100,000 network packets*

⭐ **Star this repo if you found it useful!** ⭐

</div>
