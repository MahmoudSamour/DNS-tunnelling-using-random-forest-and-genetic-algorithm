# Enhanced Detection of DNS Tunneling: Leveraging Random Forest and Genetic Algorithm for Improved Security

This repository contains the implementation of a hybrid machine learning model combining **Random Forest classifiers** and **penalty-based Genetic Algorithms (GA)** for detecting DNS tunneling attacks. The proposed method achieves state-of-the-art accuracy (99.84%) and F1-score (99.84%) on the CIRA-CIC-DoHBrw-2020 dataset.

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)
9. [Citation](#citation)

---

## Introduction
DNS tunneling is a covert communication technique that exploits the DNS protocol to bypass security measures. This project introduces a novel approach to detect DNS tunneling by combining **Random Forest classifiers** with a **penalty-based Genetic Algorithm** for feature selection. The method is specifically designed to handle encrypted DNS-over-HTTPS (DoH) traffic, making it robust against modern evasion techniques.

---

## Key Features
- **Hybrid Model**: Combines Random Forest classifiers with a penalty-based Genetic Algorithm for feature selection.
- **Encrypted Traffic Support**: Specialized feature engineering for DNS-over-HTTPS (DoH) traffic.
- **Adaptive Penalty Mechanism**: Dynamically adjusts the penalty term to balance feature reduction and model accuracy.
- **High Accuracy**: Achieves 99.84% accuracy and F1-score on the CIRA-CIC-DoHBrw-2020 dataset.

---

## Dataset
The project uses the **CIRA-CIC-DoHBrw-2020** dataset, which includes:
- **Benign Traffic**: DNS queries from standard web browsers (Chrome, Firefox, Edge).
- **Malicious Traffic**: DNS tunneling traffic generated using tools like Iodine, DNS2TCP, and DNScat2.

### Dataset Statistics

| Traffic Type       | Number of Samples | Percentage (%) |
|--------------------|-------------------|----------------|
| Benign Traffic     | 114,699          | 78.55          |
| Iodine (Malicious) | 5,704            | 3.90           |
| DNS2TCP (Malicious)| 21,003           | 14.39          |
| DNScat2 (Malicious)| 4,503            | 3.08           |
| **Total**          | **145,909**      | **100.00**     |

---

## Installation
To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MahmoudSamour/DNS-tunnelling-using-random-forest-and-genetic-algorithm.git
   cd DNS-tunnelling-using-random-forest-and-genetic-algorithm
   ```

2. **Download the dataset**:
   - The dataset is available from the Canadian Institute for Cybersecurity (CIC).
   - Place the dataset files in the `DoHBrw-2020` directory.

---


## Results
The proposed method achieves the following performance metrics:

| Metric     | Value (%) |
|------------|-----------|
| Accuracy   | 99.84     |
| Precision  | 99.84     |
| Recall     | 99.84     |
| F1-Score   | 99.84     |

### Feature Importance
The top five features identified by the model are:
- **PacketLengthMode**: 0.0998
- **DestinationIP**: 0.0055
- **SourceIP**: 0.0034
- **DestinationPort**: 0.0028
- **SourcePort**: 0.0025

---

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to the branch.
4. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Citation
If you use this work in your research, please cite it as follows:

```bibtex
@article{sammour2024dns,
  title={Enhanced Detection of DNS Tunneling: Leveraging Random Forest and Genetic Algorithm for Improved Security},
  author={Sammour, Mahmoud and Othman, Mohd Fairuz Iskandar and Bhais, Omar},
  journal={IEEE Access},
}
```

For any questions or feedback, please contact **mahmoud.samour@gmail.com**.
