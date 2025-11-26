# Optimization of Kernel-Based Classification Methods Using Quantum Genetic Algorithm

This engineering thesis project focuses on the application of a **Quantum Genetic Algorithm (QGA)** for optimizing the hyperparameters of a **Support Vector Machine (SVM)** classifier.

The main objective is to explore how quantum-inspired evolutionary methods can improve the tuning of kernel parameters, particularly for the **RBF** and **Polynomial** kernels, compared to traditional approaches such as **Grid Search** or **Random Search**.

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Oskar-Nowak/quantum-inspired-kernel-optimization.git
cd quantum-inspired-kernel-optimization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🛠️ Requirements

The project uses the following core dependencies:

```
matplotlib==3.10.7
numpy==2.3.1
scikit-learn==1.7.2
scipy==1.16.3
seaborn==0.13.2
jupyter>=1.0
ipykernel>=6.0
```

## 📌 Notes

- The `DATA/` folder is intentionally excluded from git because the Dop-Net dataset is very large.
