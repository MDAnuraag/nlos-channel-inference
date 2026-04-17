# nlos-cs

A research codebase for **metasurface-assisted non-line-of-sight sensing and imaging** with an explicit **compressed sensing** and **inverse problems** viewpoint.

This repository is separate from the thesis-era analysis scripts. Those legacy scripts remain useful as a frozen baseline, but they treat sensing, inversion, robustness testing, and post-hoc comparison as loosely coupled script stages.

This repo starts from a stricter premise:

1. the **forward operator** must be explicit,
2. sensing states such as `flat`, `tilted`, and `stepped` must be treated as **first-class encoding states**, not fused later as an afterthought,
3. the codebase must support both **physics-grounded operators** and **modern sparse / robust inverse solvers**.

---

## Core problem

We observe a measurement vector $y \in \mathbb{R}^m$ produced by an unknown object or scene state $x \in \mathbb{R}^n$ through a sensing operator $A \in \mathbb{R}^{m \times n}$:

$$
y = Ax + \varepsilon
$$

where:

- $A$ is the sensing matrix or forward operator,
- $x$ is the latent scene or object-position vector,
- $y$ is the measured probe response,
- $\varepsilon$ is noise, model mismatch, multipath corruption, or dropout.

In the old scripts, each column of $A$ is built from a CST-exported probe-plane $|E|$ response corresponding to one cylinder position. That abstraction remains the foundation here.

The central technical question is not just whether $x$ can be reconstructed, but whether the **operator itself is suitable for compressed sensing**. That means analysing:

- conditioning,
- singular value decay,
- mutual coherence,
- discrimination / leakage between columns,
- robustness under perturbation,
- benefit from multiple encoding states.

---

## Forward model

### Single-state operator

For one boundary or metasurface state, the forward model is:

$$
y = A^{(s)}x + \varepsilon
$$

where $s$ denotes a specific state such as `flat`, `tilted`, or `stepped`.

Each column $A^{(s)}_{:,j}$ is the probe-plane response when the object is at position $j$.

---

### Multi-state operator

If multiple sensing states are available, they should be represented explicitly as one composite operator:

$$
\tilde{y}
=
\begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(K)}
\end{bmatrix}
=
\begin{bmatrix}
A^{(1)} \\
A^{(2)} \\
\vdots \\
A^{(K)}
\end{bmatrix}
x
+
\tilde{\varepsilon}
$$

or, more compactly,

$$
\tilde{y} = \tilde{A}x + \tilde{\varepsilon}
$$

This vertical stacking is treated here as a primary operator construction step, not a late-stage combination.

---

## Inverse problems

### Tikhonov regularisation

A baseline reconstruction is:

$$
\hat{x}_\lambda
=
\arg\min_x \|Ax-y\|_2^2 + \lambda \|x\|_2^2
$$

with the closed-form solution:

$$
\hat{x}_\lambda
=
(A^\top A + \lambda I)^{-1}A^\top y
$$

or, using the SVD $A = U\Sigma V^\top$,

$$
\hat{x}_\lambda
=
V
\operatorname{diag}\!\left(
\frac{\sigma_i}{\sigma_i^2 + \lambda}
\right)
U^\top y
$$

---

### Sparse recovery

For compressed sensing:

$$
\hat{x}
=
\arg\min_x \|Ax-y\|_2^2 + \lambda \|x\|_1
$$

If the scene is sparse or approximately sparse, this is more appropriate than pure $L_2$ regularisation.

---

### Constrained recovery

For a single-object or simplex prior:

$$
x \ge 0,
\qquad
\sum_i x_i = 1
$$

---

### Robust loss

Under outliers or heavy-tailed perturbations:

$$
\hat{x}
=
\arg\min_x
\sum_{i=1}^{m} \rho_\delta\!\big((Ax-y)_i\big)
+
\lambda \|x\|_1
$$

with

$$
\rho_\delta(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\le\delta \\
\delta\left(|r|-\frac{1}{2}\delta\right), & |r|>\delta
\end{cases}
$$

This matters because a small number of measurements may be dominated by specular hits, multipath, or faults.

---

## Operator quality

A reconstruction is only as good as the operator.

### Condition number

If the singular values of $A$ are $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$, then

$$
\kappa(A) = \frac{\sigma_1}{\sigma_r}
$$

Large $\kappa(A)$ indicates ill-conditioning.

---

### Mutual coherence

For column-normalised $A$:

$$
\mu(A)
=
\max_{i \ne j}
\left|
a_i^\top a_j
\right|
$$

Low coherence is desirable.

---

### Leakage / discrimination

Define:

$$
D[i,j] = 1 - \frac{\hat{x}_j}{\hat{x}_i}
$$

and

$$
L[i,j] = \frac{\hat{x}_j}{\hat{x}_i}
$$

Large leakage means poor separability between positions.

---

## Design principles

### 1. Explicit operators
No hidden contracts. Operators and metadata must be explicit.

### 2. State-aware sensing
`flat`, `tilted`, `stepped` are distinct sensing states.

### 3. Separation of concerns
- I/O parses data  
- preprocessing validates  
- operators build $A$  
- inverse solves for $x$  
- metrics evaluate  
- experiments orchestrate  

### 4. Reproducibility
All runs are config-driven with traceable outputs.

### 5. Compressed sensing first
Focus is on when the system behaves like a useful encoder, not just reconstruction.

---

## Repository structure

```text
nlos-cs/
  README.md
  pyproject.toml
  .gitignore

  configs/
    dataset/
    experiment/

  data/
    raw/
    interim/
    processed/

  src/nlos_cs/
    __init__.py

    io/
    schema/
    preprocessing/
    operators/
    inverse/
    metrics/
    perturb/
    experiments/
    viz/
    cli/

  tests/
  notebooks/
  scripts/