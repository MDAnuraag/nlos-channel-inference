# nlos-cs

A research codebase for **metasurface-assisted non-line-of-sight sensing and imaging** with an explicit **compressed sensing** and **inverse problems** viewpoint.

This repository is separate from the thesis-era analysis scripts. Those legacy scripts remain useful as a frozen baseline, but they treat sensing, inversion, robustness testing, and post-hoc comparison as loosely coupled script stages. In particular, the old workflow builds sensing matrices from CST exports, solves a Tikhonov inverse problem on synthetic measurements, performs discrimination and robustness analysis, and only later stacks different boundary configurations into a combined operator. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

This repo starts from a stricter premise:

1. the **forward operator** must be explicit,
2. sensing states such as `flat`, `tilted`, and `stepped` must be treated as **first-class encoding states**, not fused later as an afterthought,
3. the codebase must support both **physics-grounded operators** and **modern sparse / robust inverse solvers**. :contentReference[oaicite:4]{index=4}

---

## Core problem

We observe a measurement vector \( y \in \mathbb{R}^m \) produced by an unknown object or scene state \( x \in \mathbb{R}^n \) through a sensing operator \( A \in \mathbb{R}^{m \times n} \):

\[
y = Ax + \varepsilon
\]

where:

- \(A\) is the sensing matrix or forward operator,
- \(x\) is the latent scene or object-position vector,
- \(y\) is the measured probe response,
- \(\varepsilon\) is noise, model mismatch, multipath corruption, or dropout.

In the old scripts, each column of \(A\) is built from a CST-exported probe-plane \(|E|\) response corresponding to one cylinder position. That is the correct basic abstraction and remains the foundation here. :contentReference[oaicite:5]{index=5}

The central technical question is not just whether \(x\) can be reconstructed, but whether the **operator itself is suitable for compressed sensing**. That means analysing:

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

\[
y = A^{(s)}x + \varepsilon
\]

where \(s\) denotes a specific state such as `flat`, `tilted`, or `stepped`.

Each column \(A^{(s)}_{:,j}\) is the probe-plane response when the object is at position \(j\).

### Multi-state operator

If multiple sensing states are available, they should be represented explicitly as one composite operator:

\[
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
\]

or, more compactly,

\[
\tilde{y} = \tilde{A}x + \tilde{\varepsilon}
\]

This vertical stacking is already present in the legacy combined-matrix script, but there it is performed late in the workflow. In this repo, it is a primary operator construction pathway. :contentReference[oaicite:6]{index=6}

---

## Inverse problems

### Tikhonov regularisation

A baseline reconstruction is:

\[
\hat{x}_\lambda
=
\arg\min_x \|Ax-y\|_2^2 + \lambda \|x\|_2^2
\]

with the closed-form filter solution:

\[
\hat{x}_\lambda
=
(A^\top A + \lambda I)^{-1}A^\top y
\]

or, using the SVD \(A = U\Sigma V^\top\),

\[
\hat{x}_\lambda
=
V
\operatorname{diag}\!\left(
\frac{\sigma_i}{\sigma_i^2 + \lambda}
\right)
U^\top y
\]

This is exactly the solver family used in the legacy code. :contentReference[oaicite:7]{index=7}

### Sparse recovery

For compressed sensing, the more relevant model is usually:

\[
\hat{x}
=
\arg\min_x \|Ax-y\|_2^2 + \lambda \|x\|_1
\]

If the scene is sparse or approximately sparse in some basis, this is often more appropriate than pure \(L_2\) regularisation.

### Constrained recovery

For a single-object or simplex prior:

\[
x \ge 0,
\qquad
\sum_i x_i = 1
\]

This prior is already explored in the old robustness code via simplex projection and Huber-style robust fitting. :contentReference[oaicite:8]{index=8}

### Robust loss

Under outliers or heavy-tailed perturbations, the data term can be changed to a Huber penalty:

\[
\hat{x}
=
\arg\min_x
\sum_{i=1}^{m} \rho_\delta\!\big((Ax-y)_i\big)
+
\lambda \|x\|_1
\]

with

\[
\rho_\delta(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\le\delta \\
\delta\left(|r|-\frac{1}{2}\delta\right), & |r|>\delta
\end{cases}
\]

This matters in NLoS sensing because a small number of measurements may be dominated by specular hits, multipath, or measurement faults.

---

## Operator quality

A reconstruction is only as good as the operator. This repo treats operator diagnostics as first-class outputs.

### Condition number

If the singular values of \(A\) are \(\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0\), then

\[
\kappa(A) = \frac{\sigma_1}{\sigma_r}
\]

A large \(\kappa(A)\) indicates ill-conditioning and strong sensitivity to perturbations.

### Mutual coherence

For column-normalised \(A\), the coherence is

\[
\mu(A)
=
\max_{i \ne j}
\left|
a_i^\top a_j
\right|
\]

Low coherence is desirable for compressed sensing, because highly correlated columns imply poor distinguishability.

### Leakage / discrimination

The legacy analysis scripts define a position-discrimination score using reconstructed amplitudes:

\[
D[i,j] = 1 - \frac{\hat{x}_j}{\hat{x}_i}
\]

when the true position is \(i\). Its complement,

\[
L[i,j] = \frac{\hat{x}_j}{\hat{x}_i}
\]

is a leakage ratio. Large leakage means position \(j\) is not strongly suppressed when position \(i\) is true. That idea is preserved here because it is one of the few quantities in the old workflow that directly measures practical separability. :contentReference[oaicite:9]{index=9}

---

## Design principles

This repo is built around a few non-negotiable rules.

### 1. Explicit operators
No hidden contracts between scripts. The operator, metadata, probe plane, and encoding state must exist as actual objects.

### 2. State-aware sensing
`flat`, `tilted`, `stepped`, and later programmable states are not “different folders”. They are distinct sensing states in a unified model.

### 3. Separation of concerns
- I/O only parses and stores data.
- preprocessing only validates and normalises.
- operators only build \(A\).
- inverse only solves for \(x\).
- metrics only measure quality.
- experiments only orchestrate runs.

### 4. Reproducibility
All runs should be config-driven and produce traceable artefacts.

### 5. Compressed sensing first
The target is not only to reconstruct from a known dense operator. The target is to understand when a metasurface-assisted sensing system behaves like a useful compressed sensing encoder.

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