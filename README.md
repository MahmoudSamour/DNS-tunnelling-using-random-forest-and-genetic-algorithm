# Enhanced Detection of DNS Tunneling: Leveraging Random Forest and Genetic Algorithm for Improved Security

![Methodology Flowchart](https://github.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/blob/main/media/flowchart_methodology.png)  
*Figure 1: Flowchart of the proposed methodology .*


![penalty_based_ga Flowchart](https://github.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/blob/main/media/penalty_based_ga_flowchart.png)  
*Figure 1: Flowchart of the proposed combining Random Forest and Genetic Algorithms.*


[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2024.0429000-blue)](https://doi.org/10.0000/ACCESS.00000.0000000)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements a hybrid machine learning model combining **Random Forest classifiers** with **penalty-based Genetic Algorithms** for detecting DNS tunneling attacks. Achieves **99.84% accuracy** on the CIRA-CIC-DoHBrw-2020 dataset.

## Key Innovations 🚀
- **Adaptive Penalty GA**: Dynamically balances feature reduction & accuracy
- **DoH-Optimized Features**: Detects patterns in encrypted DNS-over-HTTPS traffic
- **Composite Fitness Function**: Maximizes F1-score while minimizing features

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Usage](#usage)
5. [Results](#results)
6. [Citation](#citation)
7. [Team](#team)

---

## Installation

1. **Clone Repository**:
```bash
git clone https://github.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm.git
cd DNS-tunnelling-using-random-forest-and-genetic-algorithm
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download Dataset**:
- Obtain CIRA-CIC-DoHBrw-2020 dataset
- Place in `data/raw/` directory

---

## Dataset 📊
### Composition
| Traffic Type        | Samples  | Percentage |
|--------------------|---------|------------|
| Benign            | 114,699 | 78.55%     |
| Iodine (Malicious)| 5,704   | 3.90%      |
| DNS2TCP           | 21,003  | 14.39%     |
| DNScat2           | 4,503   | 3.08%      |

### Preprocessing Pipeline
- **Missing Values**: Mean imputation
- **Normalization**: StandardScaler
- **Feature Engineering**:
  - *Temporal*: Packet time variance, response time median
  - *Spatial*: Flow bytes sent, packet length mode
  - *Composite*: SourcePort-PacketLengthMode Mean

---

## Methodology 🧠
### Hybrid Detection Framework
```python
# Pseudo-Code Overview
def hybrid_detection():
    initialize_population()
    for generation in generations:
        evaluate_fitness()
        apply_penalty(lambda_g)  # λ_g = λ_0(1 - g/G)
        perform_selection()
        crossover_and_mutate()
    return optimal_features()

train_random_forest(optimal_features)
```

### Key Components
| Component           | Configuration               |
|--------------------|---------------------------|
| Random Forest     | 50 trees, max_depth=10, sqrt features |
| Genetic Algorithm | Population=100, Generations=20 |
| Fitness Function  | F1-score - λ(feature_count/total) |

![GA Flowchart](https://github.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm/blob/main/images/penalty_based_ga_flowchart.png)  
*Figure 2: Penalty-based Genetic Algorithm flowchart.*

---

## Results 📈
### Performance Metrics
| Metric     | Our Model | RF Baseline | DeepFM [1] |
|-----------|----------|------------|-----------|
| Accuracy  | 99.84%   | 98.20%     | 99.50%    |
| F1-Score  | 99.84%   | 97.50%     | 99.40%    |
| Features  | 18       | 42         | 35        |

---

## Citation 📝
If you use this work, please cite:
```bibtex
@article{sammour2024dns,
  title={Enhanced Detection of DNS Tunneling: Leveraging Random Forest and Genetic Algorithm for Improved Security},
  author={Sammour, Mahmoud and Othman, Mohd Fairuz Iskandar and Bhais, Omar},
  journal={IEEE Access},
}
```

---

## Team 👥
| Researcher                     | Contribution             |
|--------------------------------|-------------------------|
| Mahmoud Sammour                | Methodology, Implementation |
| Mohd Fairuz Iskandar Othman    | Supervision, Validation |
| Omar A A Bhais                 | Data Preprocessing, Testing |
| Aslinda Hassan             | Co-Supervisor, Research Guidance |

