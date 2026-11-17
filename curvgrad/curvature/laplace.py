import numpy as np


def laplace_posterior_precision(ggn: np.ndarray, n_data: int, prior_prec: float) -> np.ndarray:
    """Posterior precision N ggn + tau I, with ggn the per sample GGN."""
    return n_data * ggn + prior_prec * np.identity(ggn.shape[0])


def newton_step(
    theta: np.ndarray, grad_mean: np.ndarray, ggn: np.ndarray, n_data: int, prior_prec: float
) -> np.ndarray:
    """One Newton step on E = N L + tau / 2 |theta|^2. Returns the new flat parameters."""
    grad_E = n_data * grad_mean + prior_prec * theta
    step = np.linalg.solve(laplace_posterior_precision(ggn, n_data, prior_prec), grad_E)
    return theta - step


def sample(theta_map: np.ndarray, chol: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """Draw parameter samples from N(theta_map, Lambda^-1), given the Cholesky factor of Lambda."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((theta_map.size, n_samples))
    delta = np.linalg.solve(chol.T, eps)  # P by S, covariance Lambda^-1

    return (theta_map[:, np.newaxis] + delta).T


def margin_moments(
    jac: np.ndarray, logits: np.ndarray, chol: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the two class margin z1 - z0 under the linearized Laplace posterior."""
    mean = logits[:, 1] - logits[:, 0]
    g_margin = jac[:, 1, :] - jac[:, 0, :]
    v = np.linalg.solve(chol, g_margin.T)
    var = np.sum(v**2, axis=0)

    return mean, var


def probit(mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Probit approximation of E[sigmoid(m)] for a Gaussian margin m."""
    scaled_mean = mean / np.sqrt(1 + (np.pi / 8) * var)

    return np.where(
        scaled_mean >= 0,
        1 / (1 + np.exp(-scaled_mean)),
        np.exp(scaled_mean) / (1 + np.exp(scaled_mean)),
    )


def log_marginal_likelihood(
    loss_mean: float, theta_map: np.ndarray, chol: np.ndarray, n_data: int, prior_prec: float
) -> float:
    """Laplace approximation of the log evidence log p(D | prior_prec)."""
    P = theta_map.size
    nll = loss_mean * n_data
    param_complexity = (prior_prec / 2) * float(np.sum(theta_map**2, axis=-1))
    log_det_ratio = 0.5 * P * np.log(prior_prec) - float(np.sum(np.log(np.diag(chol))))

    return -nll - param_complexity + log_det_ratio
