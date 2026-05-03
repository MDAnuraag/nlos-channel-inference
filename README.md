# nlos-cs

A research codebase for **metasurface-assisted non-line-of-sight sensing and imaging** with an explicit **compressed sensing** and **inverse-problems** viewpoint.

This repository is separate from the thesis-era scripts. Those legacy scripts remain useful as a frozen baseline, but they mixed operator construction, inversion, robustness testing, and post-hoc comparison into loosely coupled analysis stages.

This repo takes a stricter view:

1. the **forward operator** must be explicit  
2. sensing states such as `flat`, `tilted`, and `stepped` must be treated as **first-class encoding states**  
3. inversion, perturbation, metrics, and experiment orchestration must be **modular and reproducible**  
4. the main question is not only whether reconstruction works, but whether the **operator itself is suitable for sparse and robust inference**

---

## What this repository is for

The codebase is built to answer questions like:

- Is a given metasurface sensing state informative enough to localise an object?
- Does stacking multiple sensing states materially improve identifiability?
- How coherent is the operator across neighbouring positions?
- Which inverse prior is most appropriate: smooth, sparse, non-negative, or simplex-constrained?
- How quickly does performance degrade under noise, dropout, correlated corruption, or structured leakage?

This is therefore both:

- an **operator analysis** codebase, and
- an **inverse reconstruction** codebase.

---

## Core forward model

We observe a measurement vector \(y \in \mathbb{R}^m\) produced by a latent scene or object-position vector \(x \in \mathbb{R}^n\) through a sensing operator \(A \in \mathbb{R}^{m \times n}\):

$$
y = Ax + \varepsilon
$$

where:
- \(A\) is the sensing matrix or forward operator
- \(x\) is the latent scene or position vector
- \(y\) is the measured probe response
- \(\varepsilon\) represents noise, model mismatch, multipath contamination, correlated corruption, or dropout

Each column of \(A\) corresponds to the probe-plane response associated with one candidate object position.

The central question is whether the operator itself has the structure needed for reliable inversion, not merely whether one particular solver can be made to work.

---

## Single-state and multi-state sensing

### Single-state operator

For one sensing state \(s\), such as `flat`, `tilted`, or `stepped`:

$$
y = A^{(s)}x + \varepsilon
$$

### Multi-state operator

If multiple sensing states are available, measurements and operators are stacked:

$$
\tilde{y} =
\begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(K)}
\end{bmatrix},
\qquad
\tilde{A} =
\begin{bmatrix}
A^{(1)} \\
A^{(2)} \\
\vdots \\
A^{(K)}
\end{bmatrix}
$$

so that

$$
\tilde{y} = \tilde{A}x + \tilde{\varepsilon}
$$

This treats metasurface states as deliberate encoding states rather than simple file groups.

---

## Inverse problems implemented in this repo

### Tikhonov regularisation

$$
\hat{x}_\lambda =
\arg\min_x \|Ax - y\|_2^2 + \lambda \|x\|_2^2
$$

with closed-form solution

$$
\hat{x}_\lambda = (A^\top A + \lambda I)^{-1}A^\top y
$$

and SVD form

$$
\hat{x}_\lambda =
V \, \mathrm{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right) U^\top y
$$

This is the main linear baseline and supports both fixed-\(\lambda\) and L-curve selection.

### Non-negative least squares

$$
\hat{x} = \arg\min_{x \ge 0} \|Ax-y\|_2^2
$$

This is useful when negative position weights are physically meaningless, but a simplex prior would be too strong.

### Sparse recovery (LASSO)

$$
\hat{x} =
\arg\min_x \frac{1}{2}\|Ax-y\|_2^2 + \alpha \|x\|_1
$$

This is the main sparse baseline for compressed sensing.

### Simplex-constrained robust recovery

For a single-object or probability-like prior:

$$
x \ge 0, \qquad \sum_i x_i = 1
$$

and a Huber data term:

$$
\hat{x} =
\arg\min_x \sum_{i=1}^{m} \rho_\delta\big((Ax-y)_i\big)
\quad
\text{subject to }
x \ge 0,\ \sum_i x_i = 1
$$

where

$$
\rho_\delta(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\le\delta \\
\delta\left(|r|-\frac{1}{2}\delta\right), & |r|>\delta
\end{cases}
$$

Important note: under simplex constraints, \(\|x\|_1 = 1\) identically, so an added \(L_1\) penalty is constant and therefore redundant. This repo treats that explicitly rather than pretending it still tunes sparsity.

---

## Operator quality metrics

The repo does not treat inversion quality alone as sufficient. It also analyses the operator directly.

### Condition number

$$
\kappa(A) = \frac{\sigma_1}{\sigma_r}
$$

where \(\sigma_1\) is the largest singular value and \(\sigma_r\) is the smallest non-zero singular value.

### Mutual coherence

For column-normalised operator columns \(a_i\),

$$
\mu(A) = \max_{i \ne j} \left| a_i^\top a_j \right|
$$

Low coherence is desirable for sparse recovery.

### Singular spectrum

The singular value spectrum is used to inspect:
- rank structure
- numerical conditioning
- energy concentration
- redundancy across sensing states

### Leakage and discrimination

The repo computes leakage and discrimination matrices from reconstructed responses across positions. Conceptually:

- **leakage** measures how much energy from a true position spills into a competing position
- **discrimination** measures how well one position is separated from another under reconstruction

These are derived from the reconstructed \(x\)-responses across operator columns or supplied measurements, rather than assumed from a simplified symbolic formula.

### Peak margin and success rate

For robustness studies, the repo tracks:
- peak index correctness
- peak-to-second-peak margin
- mean success rate versus corruption level

---

## Perturbation models

The repo includes explicit perturbation modules, rather than burying corruption models inside analysis scripts.

Currently implemented:

- additive white Gaussian noise
- correlated additive noise
- dropout / missing measurement channels
- multipath / structured column leakage
- composed perturbation pipelines

This makes it possible to stress-test not only solver performance, but also operator fragility under more realistic corruption.

---

## What is implemented now

The current codebase already supports:

- CST ASCII loading and probe-plane extraction
- single-state operator construction
- multi-state operator construction
- operator diagnostics and matrix visualisation
- Tikhonov, NNLS, LASSO, and Huber-simplex solvers
- reconstruction, discrimination, and robustness experiment runners
- config-driven CLI execution
- saved artefacts for operators, reconstructions, discrimination runs, and robustness sweeps
- test coverage across the core stack

---

## Design principles

### Explicit operators
Operators and metadata are first-class objects.

### State-aware sensing
Encoding states are modelled explicitly and can be stacked into composite operators.

### Separation of concerns
- `io/` parses and saves data
- `preprocessing/` validates and extracts structured field data
- `operators/` constructs \(A\)
- `inverse/` solves for \(x\)
- `metrics/` evaluates operator and reconstruction quality
- `perturb/` applies controlled corruption models
- `experiments/` orchestrates reproducible runs
- `viz/` provides research-facing plots
- `cli/` dispatches config-driven experiments

### Reproducibility
Runs are config-driven and saved as explicit artefacts.

### Compressed sensing first
The primary concern is operator quality and encoding capability, not only post-hoc reconstruction visuals.

---

## Repository structure

```text
nlos-channel-inference/
  README.md
  pyproject.toml
  .gitignore

  data/
  examples/
    configs/
  notebooks/
  outputs/
  scripts/
  src/
    nlos_cs/
      cli/
      experiments/
      inverse/
      io/
      metrics/
      operators/
      perturb/
      preprocessing/
      viz/
  tests/