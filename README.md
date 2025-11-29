# Optimization of Kernel-Based Classification Methods Using Quantum Genetic Algorithm

This engineering thesis project focuses on the application of a **Quantum Genetic Algorithm (QGA)** for optimizing the hyperparameters of a **Support Vector Machine (SVM)** classifier.

The main objective is to explore how quantum-inspired evolutionary methods can improve the tuning of kernel parameters, particularly for the **RBF** and **Polynomial** kernels, compared to traditional approaches such as **Grid Search** or **Random Search**.

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Oskar-Nowak/quantum-inspired-kernel-optimization.git
cd quantum-inspired-kernel-optimization
```

Install the project using the pyproject.toml configuration:

```bash
pip install .
```

Alternatively, for development mode:

```bash
pip install -e .
```

## 🛠️ Requirements

All dependencies are now managed through pyproject.toml.
Core libraries include:

- numpy
- scipy
- seaborn
- scikit-learn
- matplotlib
- jupyter / ipykernel

To view exact versions, check the [project] → dependencies section inside pyproject.toml.

## 📌 Notes

- The `DATA/` folder is intentionally excluded from git because the Dop-Net dataset is very large.
