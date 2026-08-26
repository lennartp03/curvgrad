# curvgrad

From-scratch autograd + curvature-based Laplace approximation, applied
to both neural network weight posteriors and GLMM random effects.

## Idea

- Laplace approximation ≈ same method in two fields:
  - Bayesian deep learning: linearized Laplace over NN weights
  - Classical stats: Laplace approx. for GLMM random effects
- Both estimated from one shared curvature core (GGN / Fisher)

## What's in here

- `tensor.py` — autograd engine, built on NumPy only (no ML framework at runtime)
- `nn/` — Linear, MLP, standard losses
- `curvature/` — GGN, Fisher, generic Laplace approximation
- `glmm/` — link functions, random intercepts/slopes, Laplace-based fit
- PyTorch used only in tests, as a gradient-check oracle — never at runtime

## Status

- [ ] Autograd core (ops + gradcheck)
- [ ] NN stack (Linear, MLP, losses)
- [ ] curvgrad vs. torch.nn parity example
- [ ] Curvature core (GGN, Fisher, Laplace)
- [ ] Laplace posterior over NN weights + calibration example
- [ ] GLMM: random intercepts
- [ ] GLMM: random slopes
- [ ] Hierarchical Poisson GLMM example (main demo)

## Quickstart

(fill in once `nn/` + one example work end-to-end)

## Tests

`pytest tests/` — gradient checks vs. `torch.autograd.gradcheck`,
GLMM fits vs. `statsmodels`/`lme4`
