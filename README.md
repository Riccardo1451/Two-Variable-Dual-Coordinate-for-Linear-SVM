# SVM Dual Coordinate Descent: DCD vs Two-Variable CD

Python reimplementation of the experiments from **Chiu et al. (2020)**, comparing two coordinate descent strategies for solving the L2-SVM dual problem.

The original code was written in C++; this is a Python reinterpretation for study and experimentation purposes.

---

## Background

Support Vector Machines (SVMs) are trained by solving an optimization problem. Working in the **dual formulation** is common because it allows the use of kernels and often leads to more efficient solvers for large datasets.

The dual of the L2-SVM (with bias embedded in the weight vector, so no equality constraint in the dual) is:

$$\min_{\alpha \geq 0} \quad \frac{1}{2} \alpha^T Q \alpha - \mathbf{1}^T \alpha$$

where $Q_{ij} = y_i y_j x_i^T x_j + \frac{1}{2C} \delta_{ij}$.

The standard approach to solve this is **Dual Coordinate Descent (DCD)**: at each step, pick one variable $\alpha_i$ and minimize the objective with respect to it while keeping all others fixed. This yields a closed-form update.

**Chiu et al. (2020)** propose a **Two-Variable Coordinate Descent (2-CD)** method: at each step, pick a *pair* of variables $(\alpha_i, \alpha_j)$ and solve the 2D subproblem jointly. Because the bias is absorbed into the primal weight vector (no dual equality constraint $\sum y_i \alpha_i = 0$), each pair can be updated independently — unlike classic SMO, which is forced to move pairs precisely because of that constraint. The paper proves that 2-CD requires fewer iterations to converge, especially at high regularization (large $C$).

---

## Project Structure

```
.
├── solvers/
│   ├── DCD_svm.py      # L2-SVM solver: one-variable DCD
│   └── twoCD_svm.py    # L2-SVM solver: two-variable CD (Chiu et al. 2020)
│
├── basic_svms/
│   ├── svm.py          # Primal SVM via SGD (educational baseline)
│   └── dual_svm.py     # Dual SVM via SMO (with bias, equality constraint)
│
├── DCDvs2CD.py         # Benchmark script: runs both solvers and plots results
├── utils.py            # Data loading for LIBSVM format datasets
├── docs/
│   └── Report.pdf      # Project report
└── results/            # Output plots and f* cache
```

### `solvers/`

The two main solver implementations from the paper:

- **`DCD_svm.py`** — One-variable DCD (`SVM_DCD`)
- **`twoCD_svm.py`** — Two-variable CD (`SVM_2CD`, Chiu et al. 2020)

### `basic_svms/`

Exploratory implementations to understand SVM from first principles:

- **`svm.py`** — Primal SVM trained with subgradient SGD. Explicit weight vector `w` and bias `b`. Useful to understand the primal hinge-loss formulation.
- **`dual_svm.py`** — Dual SVM solved with SMO. Here the bias `b` is kept explicit, which introduces the equality constraint $\sum y_i \alpha_i = 0$ in the dual, requiring pairs of variables to always be updated together.

### `solvers/DCD_svm.py` — One-Variable DCD

Implements `SVM_DCD`. At each step, one dual variable $\alpha_i$ is selected at random, its gradient is computed, and it is updated with a closed-form projected step:

$$\alpha_i \leftarrow \max\left(0,\ \alpha_i - \frac{G_i}{Q_{ii}}\right)$$

The primal weight `w` is maintained incrementally. Convergence is checked via the projected gradient gap $M - m < \varepsilon$.

### `solvers/twoCD_svm.py` — Two-Variable CD

Implements `SVM_2CD`. At each step, a random pair $(i, j)$ is selected and the 2×2 quadratic subproblem

$$\min_{d_i, d_j}\ G_i d_i + G_j d_j + \frac{1}{2}\begin{pmatrix}d_i \\ d_j\end{pmatrix}^T \begin{pmatrix}Q_{ii} & Q_{ij} \\ Q_{ij} & Q_{jj}\end{pmatrix} \begin{pmatrix}d_i \\ d_j\end{pmatrix}$$

subject to $d_i \geq -\alpha_i,\ d_j \geq -\alpha_j$, is solved exactly. The solver evaluates four KKT candidates (unconstrained minimizer, two boundary edges, corner) and picks the one with the lowest objective value. This is the `constrained` mode, which is numerically robust. A `naive` mode (simple clipping) is also available for comparison.

### `DCDvs2CD.py` — Benchmark Script

Runs both solvers on one or more datasets across multiple values of `C`, then plots:

- **Relative objective gap** vs number of CD steps attempted
- **Relative objective gap** vs wall-clock time

The reference optimum $f^*$ is computed by running both solvers for many iterations and taking the best value found; it is cached to disk to avoid recomputation.

---

## Datasets

Datasets are in **LIBSVM format** and should be placed in a `dataset/` folder. The experiments use:

| Dataset | Train samples | Test samples |
|---------|--------------|-------------|
| a9a     | 32,561       | 16,281       |
| w8a     | 49,749       | 14,951       |
| ijcnn1  | 49,990       | 91,701       |

Datasets can be downloaded from the [LIBSVM data repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).

---

## Requirements

```
numpy
scikit-learn
matplotlib
tqdm
scipy
```

Install with:

```bash
pip install numpy scikit-learn matplotlib tqdm scipy
```

---

## Usage

Run the comparison benchmark:

```bash
python DCDvs2CD.py
```

Plots are saved in `results/` as `plot_{dataset}_C{value}.png`.

To switch datasets or C values, edit the `datasets` and `C_values` variables at the top of `DCDvs2CD.py`.

---

## Reference

> Chiu, C.-Y., Ho, C.-H., & Lin, C.-J. (2020).  
> *Two-Variable Coordinate Descent for L2-loss SVM Dual.*  
> Proceedings of Machine Learning Research.
