# Unsupervised Online Learning for Network Flow Anomaly Detection: A Comparative Evaluation

This repository contains the source code and experimental artifacts for the study presented at the CISIS 2025 conference. The main objective of this work is the comparative evaluation of unsupervised online machine learning models for anomaly detection in network flows. The analysis focuses on their performance over a dataset provided by a private company and their applicability in real-world scenarios.

---

## 📁 Repository Structure

```plaintext
├── dataset/
│   └── dataset_anonymized.csv
├── experimentation.ipynb
├── ocsvm.py
├── requirements.txt
└── README.md
```

- `dataset/`: Contains the anonymized dataset provided by the company.
- `experimentation.ipynb`: Jupyter Notebook with the full implementation of the experimental pipeline.
- `ocsvm.py`: Python script with additional implementations for anomaly detection experiments.
- `requirements.txt`: Python dependencies required for the project.

## 📚 Dependencies

The project requires the following Python libraries (see `requirements.txt` for specific versions):

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning algorithms
- `river` - Online machine learning
- `ipykernel` - Jupyter kernel support


## 🙏 Acknowledgements

This research is a result of Grant Explicit PID2024-162298OB-I00 and CENTAURO PLEC2023-010360 funded by MICIU/AEI/10.13039/501100011033 and, as appropriate, by "ERDF A way of making Europe", by "ERDF/EU", by the "European Union" or by the "European Union NextGenerationEU/PRTR".