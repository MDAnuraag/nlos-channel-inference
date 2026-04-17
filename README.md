# nlos-cs

A research codebase for **metasurface-assisted non-line-of-sight sensing and imaging** with an explicit **compressed sensing** and **inverse problems** viewpoint.

This repository is separate from the thesis-era analysis scripts. Those legacy scripts remain useful as a frozen baseline, but they treat sensing, inversion, robustness testing, and post-hoc comparison as loosely coupled script stages.

This repo starts from a stricter premise:

1. the **forward operator** must be explicit  
2. sensing states such as `flat`, `tilted`, and `stepped` must be treated as **first-class encoding states**  
3. the codebase must support both **physics-grounded operators** and **modern sparse / robust inverse solvers**

---

## Core problem

We observe a measurement vector $y \in \mathbb{R}^m$ produced by an unknown object or scene state $x \in \mathbb{R}^n$ through a sensing operator $A \in \mathbb{R}^{m \times n}$:

$$
y = Ax + \varepsilon
$$

where:
- $A$ is the sensing matrix or forward operator  
- $x$ is the latent scene or object-position vector  
- $y$ is the measured probe response  
- $\varepsilon$ is noise, model mismatch, multipath corruption, or dropout  

Each column of $A$ corresponds to a probe-plane $|E|$ response at a specific object position.

The central question is whether the **operator itself is suitable for compressed sensing**, not just whether inversion is possible.

---

## Forward model

### Single-state operator

$$
y = A^{(s)}x + \varepsilon
$$

where $s$ denotes a sensing state such as `flat`, `tilted`, or `stepped`.

---

### Multi-state operator

$$
\tilde{y} =
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
x + \tilde{\varepsilon}
$$

$$
\tilde{y} = \tilde{A}x + \tilde{\varepsilon}
$$

---

## Inverse problems

### Tikhonov regularisation

$$
\hat{x}_\lambda =
\arg\min_x \|Ax-y\|_2^2 + \lambda \|x\|_2^2
$$

$$
\hat{x}_\lambda =
(A^\top A + \lambda I)^{-1}A^\top y
$$

$$
\hat{x}_\lambda =
V \, \mathrm{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right) U^\top y
$$

### Sparse recovery

$$
\hat{x} =
\arg\min_x \|Ax-y\|_2^2 + \lambda \|x\|_1
$$

### Constrained recovery

$$
x \ge 0, \quad \sum_i x_i = 1
$$

### Robust loss

$$
\hat{x} =
\arg\min_x \sum_{i=1}^{m} \rho_\delta\big((Ax-y)_i\big) + \lambda \|x\|_1
$$

$$
\rho_\delta(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\le\delta \\
\delta\left(|r|-\frac{1}{2}\delta\right), & |r|>\delta
\end{cases}
$$

---

## Operator quality

### Condition number

$$
\kappa(A) = \frac{\sigma_1}{\sigma_r}
$$

---

### Mutual coherence

$$
\mu(A) = \max_{i \ne j} \left| a_i^\top a_j \right|
$$

---

### Leakage / discrimination

$$
D[i,j] = 1 - \frac{\hat{x}_j}{\hat{x}_i}
$$

$$
L[i,j] = \frac{\hat{x}_j}{\hat{x}_i}
$$

---

## Design principles

### Explicit operators
Operators and metadata must be explicitly represented.

### State-aware sensing
Encoding states are treated as structured inputs, not file groupings.

### Separation of concerns
- I/O parses data  
- preprocessing validates  
- operators construct $A$  
- inverse solves for $x$  
- metrics evaluate  
- experiments orchestrate  

### Reproducibility
All runs must be config-driven and traceable.

### Compressed sensing first
Focus is on operator quality and encoding capability, not just reconstruction.

---

## Repository structure

```text
nlos-cs/
  README.md
  pyproject.toml
  .gitignore

  configs/
  data/
  src/nlos_cs/
  tests/
  notebooks/
  scripts/