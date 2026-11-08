# curvgrad

From-scratch autograd + curvature-based Laplace approximation, applied
to both neural network weight posteriors and GLMM random effects.

## Idea

- Laplace approximation as same method in two fields:
  - Bayesian deep learning: linearized Laplace over NN weights
  - Classical stats: Laplace approx. for GLMM random effects
- Both estimated from one shared curvature core (GGN / Fisher)

## What's in here

- `autograd.py`, `ops.py`, `tensor.py` — autograd engine with reverse (VJP) and forward (JVP) mode, NumPy only
- `nn/` — Linear, cross entropy
- `curvature/` — matrix-free GGN; Fisher and Laplace planned
- `glmm/` — link functions, random intercepts/slopes, Laplace-based fit (planned)
- PyTorch used only in tests, as a gradient-check oracle — never at runtime

## Status

- [x] Autograd core (ops, VJP + JVP, finite-difference checks)
- [x] Linear + cross entropy, logistic regression example
- [ ] MLP
- [ ] curvgrad vs. torch.nn parity example
- [x] GGN (matrix-free, checked against the exact Hessian)
- [ ] Fisher, Laplace
- [ ] Laplace posterior over NN weights + calibration example
- [ ] GLMM: random intercepts
- [ ] GLMM: random slopes
- [ ] Hierarchical Poisson GLMM example

## Quickstart

```bash
uv sync
uv run python examples/classification.py
```

Trains logistic regression on two moons and plots the decision boundary, the GGN and its eigenvalues.

## Tests

`uv run pytest` — every op's VJP and JVP against finite differences, the GGN against a
finite-difference Hessian. Planned: `torch.autograd.gradcheck`, GLMM fits vs. `lme4`.
