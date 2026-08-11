#!/usr/bin/env python3
"""Minimal auditable estimators for the locked interrupted time-series SAP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize, root
from scipy.special import digamma, expit, gammaln
from scipy.stats import chi2, norm


EPS = 1e-12


@dataclass
class ModelFit:
    family: str
    params: np.ndarray
    cov_hac: np.ndarray
    cov_model: np.ndarray
    fitted: np.ndarray
    score_obs: np.ndarray
    converged: bool
    message: str
    alpha: float | None = None
    loglik: float | None = None


def hac_meat(scores: np.ndarray, lag: int) -> np.ndarray:
    n = scores.shape[0]
    meat = scores.T @ scores
    for ell in range(1, min(lag, n - 1) + 1):
        w = 1.0 - ell / (lag + 1.0)
        cross = scores[ell:].T @ scores[:-ell]
        meat += w * (cross + cross.T)
    return meat


def sandwich(bread: np.ndarray, scores: np.ndarray, lag: int, p: int) -> np.ndarray:
    n = scores.shape[0]
    correction = n / max(n - p, 1)
    cov = correction * bread @ hac_meat(scores, lag) @ bread.T
    return (cov + cov.T) / 2.0


def _poisson_irls(y: np.ndarray, x: np.ndarray, maxiter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(x.shape[1], dtype=float)
    beta[0] = np.log(max(np.mean(y), 1e-3))
    for _ in range(maxiter):
        eta = np.clip(x @ beta, -30, 30)
        mu = np.exp(eta)
        z = eta + (y - mu) / np.maximum(mu, EPS)
        xtwx = x.T @ (mu[:, None] * x)
        xtwz = x.T @ (mu * z)
        new_beta = np.linalg.pinv(xtwx) @ xtwz
        if np.max(np.abs(new_beta - beta)) < 1e-10:
            beta = new_beta
            break
        beta = new_beta
    return beta, np.exp(np.clip(x @ beta, -30, 30))


def fit_poisson_hac(y: np.ndarray, x: np.ndarray, lag: int = 4) -> ModelFit:
    beta, mu = _poisson_irls(y, x)
    score_obs = x * (y - mu)[:, None]
    info = x.T @ (mu[:, None] * x)
    bread = np.linalg.pinv(info)
    cov_hac = sandwich(bread, score_obs, lag, len(beta))
    ll = float(np.sum(y * np.log(np.maximum(mu, EPS)) - mu - gammaln(y + 1)))
    return ModelFit("Poisson", beta, cov_hac, bread, mu, score_obs, True, "IRLS", None, ll)


def fit_nb2_hac(y: np.ndarray, x: np.ndarray, lag: int = 4) -> ModelFit:
    beta0, mu0 = _poisson_irls(y, x)
    alpha0 = max(float(np.sum((y - mu0) ** 2 - y) / np.sum(np.maximum(mu0**2, EPS))), 1e-5)
    theta0 = np.r_[beta0, np.log(alpha0)]

    def score_obs(theta: np.ndarray) -> np.ndarray:
        beta = theta[:-1]
        alpha = float(np.exp(np.clip(theta[-1], -20, 10)))
        r = 1.0 / alpha
        mu = np.exp(np.clip(x @ beta, -30, 30))
        score_beta = x * ((y - mu) / (1.0 + alpha * mu))[:, None]
        dll_dr = (
            digamma(y + r)
            - digamma(r)
            + np.log(r / (r + mu))
            + 1.0
            - (r + y) / (r + mu)
        )
        score_loga = -r * dll_dr
        return np.column_stack([score_beta, score_loga])

    def loglik(theta: np.ndarray) -> float:
        beta = theta[:-1]
        alpha = float(np.exp(np.clip(theta[-1], -20, 10)))
        r = 1.0 / alpha
        mu = np.exp(np.clip(x @ beta, -30, 30))
        ll = (
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1.0)
            + r * (np.log(r) - np.log(r + mu))
            + y * (np.log(np.maximum(mu, EPS)) - np.log(r + mu))
        )
        return float(np.sum(ll))

    def objective(theta: np.ndarray) -> float:
        return -loglik(theta)

    def gradient(theta: np.ndarray) -> np.ndarray:
        return -score_obs(theta).sum(axis=0)

    opt = minimize(
        objective,
        theta0,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(None, None)] * x.shape[1] + [(-20.0, 10.0)],
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not opt.success or not np.all(np.isfinite(opt.x)):
        pfit = fit_poisson_hac(y, x, lag)
        pfit.message = f"NB2 failure; Poisson fallback: {opt.message}"
        return pfit

    theta = opt.x
    # Tight score-equation refinement after likelihood optimization.  This is
    # especially useful when the objective is flat in alpha and L-BFGS-B stops
    # on relative function change before every score component is near zero.
    refined = root(lambda z: score_obs(z).sum(axis=0), theta, method="hybr", options={"xtol": 1e-11, "maxfev": 10000})
    if refined.success and np.all(np.isfinite(refined.x)) and -20.0 <= refined.x[-1] <= 10.0:
        if np.max(np.abs(score_obs(refined.x).sum(axis=0))) < np.max(np.abs(score_obs(theta).sum(axis=0))):
            theta = refined.x
    scores = score_obs(theta)
    # Observed bread from a symmetric numerical derivative of the analytic score.
    h = 1e-5
    jac = np.empty((len(theta), len(theta)))
    for j in range(len(theta)):
        step = np.zeros_like(theta)
        step[j] = h * max(1.0, abs(theta[j]))
        jac[:, j] = (
            score_obs(theta + step).sum(axis=0) - score_obs(theta - step).sum(axis=0)
        ) / (2.0 * step[j])
    info = -(jac + jac.T) / 2.0
    bread = np.linalg.pinv(info)
    cov_hac_full = sandwich(bread, scores, lag, len(theta))
    beta = theta[:-1]
    mu = np.exp(np.clip(x @ beta, -30, 30))
    return ModelFit(
        "Negative binomial NB2",
        beta,
        cov_hac_full[:-1, :-1],
        bread[:-1, :-1],
        mu,
        scores[:, :-1],
        True,
        str(opt.message),
        float(np.exp(theta[-1])),
        loglik(theta),
    )


def fit_grouped_binomial_hac(success: np.ndarray, total: np.ndarray, x: np.ndarray, lag: int = 4) -> ModelFit:
    if np.any(total <= 0) or np.any(success < 0) or np.any(success > total):
        raise ValueError("Grouped-binomial observations require 0 <= success <= total and total > 0")
    overall = np.clip(success.sum() / total.sum(), 1e-6, 1 - 1e-6)
    beta0 = np.zeros(x.shape[1])
    beta0[0] = np.log(overall / (1 - overall))

    # Stable Newton-IRLS initialization for grouped data.
    for _ in range(200):
        p0 = expit(np.clip(x @ beta0, -30, 30))
        score0 = x.T @ (success - total * p0)
        info0 = x.T @ ((total * p0 * (1 - p0))[:, None] * x)
        step0 = np.linalg.pinv(info0) @ score0
        if np.max(np.abs(step0)) < 1e-10:
            break
        # Backtracking prevents rare overshoots near a sparse category boundary.
        current_obj = float(np.sum(total * np.logaddexp(0.0, x @ beta0) - success * (x @ beta0)))
        scale = 1.0
        while scale > 1e-6:
            candidate = beta0 + scale * step0
            eta_c = np.clip(x @ candidate, -30, 30)
            candidate_obj = float(np.sum(total * np.logaddexp(0.0, eta_c) - success * eta_c))
            if candidate_obj <= current_obj:
                beta0 = candidate
                break
            scale *= 0.5
        else:
            break

    def score_obs(beta: np.ndarray) -> np.ndarray:
        p = expit(np.clip(x @ beta, -30, 30))
        return x * (success - total * p)[:, None]

    def objective(beta: np.ndarray) -> float:
        eta = np.clip(x @ beta, -30, 30)
        return float(np.sum(total * np.logaddexp(0.0, eta) - success * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return -score_obs(beta).sum(axis=0)

    opt = minimize(
        objective,
        beta0,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-8, "maxls": 50},
    )
    beta = opt.x
    pfit = expit(np.clip(x @ beta, -30, 30))
    w = total * pfit * (1.0 - pfit)
    info = x.T @ (w[:, None] * x)
    bread = np.linalg.pinv(info)
    scores = score_obs(beta)
    cov_hac = sandwich(bread, scores, lag, len(beta))
    return ModelFit(
        "Grouped binomial logit",
        beta,
        cov_hac,
        bread,
        pfit,
        scores,
        bool(opt.success or np.linalg.norm(gradient(beta), ord=np.inf) < 1e-4),
        str(opt.message),
        None,
        -objective(beta),
    )


def fit_ols_hac(y: np.ndarray, x: np.ndarray, lag: int = 4) -> ModelFit:
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ (x.T @ y)
    fitted = x @ beta
    resid = y - fitted
    scores = x * resid[:, None]
    cov_hac = sandwich(bread, scores, lag, len(beta))
    sigma2 = float(np.sum(resid**2) / max(len(y) - x.shape[1], 1))
    cov_model = sigma2 * bread
    ll = float(-0.5 * len(y) * (np.log(2 * np.pi * sigma2) + 1.0)) if sigma2 > 0 else np.nan
    return ModelFit("OLS on log weekly median", beta, cov_hac, cov_model, fitted, scores, True, "closed form", None, ll)


def two_sided_p(z: float) -> float:
    return float(2.0 * norm.sf(abs(z)))


def effect_row(beta: np.ndarray, cov: np.ndarray, idx: int, scale: str) -> dict[str, float]:
    est = float(beta[idx])
    se = float(np.sqrt(max(cov[idx, idx], 0.0)))
    z = est / se if se > 0 else np.nan
    lo, hi = est - 1.96 * se, est + 1.96 * se
    if scale == "exp":
        return {"estimate": float(np.exp(est)), "ci_low": float(np.exp(lo)), "ci_high": float(np.exp(hi)), "z": float(z), "p_raw": two_sided_p(z)}
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "z": float(z), "p_raw": two_sided_p(z)}


def linear_combination(beta: np.ndarray, cov: np.ndarray, contrast: np.ndarray, scale: str = "exp") -> dict[str, float]:
    est = float(contrast @ beta)
    var = float(contrast @ cov @ contrast)
    se = np.sqrt(max(var, 0.0))
    z = est / se if se > 0 else np.nan
    lo, hi = est - 1.96 * se, est + 1.96 * se
    if scale == "exp":
        return {"estimate": float(np.exp(est)), "ci_low": float(np.exp(lo)), "ci_high": float(np.exp(hi)), "z": float(z), "p_raw": two_sided_p(z)}
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "z": float(z), "p_raw": two_sided_p(z)}


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted_sorted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted_sorted[rank] = running
    out = np.empty(m)
    out[order] = adjusted_sorted
    return out


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    q_sorted = np.empty(m)
    running = 1.0
    for reverse_rank in range(m - 1, -1, -1):
        idx = order[reverse_rank]
        value = min(1.0, p[idx] * m / (reverse_rank + 1))
        running = min(running, value)
        q_sorted[reverse_rank] = running
    out = np.empty(m)
    out[order] = q_sorted
    return out


def residual_diagnostics(resid: np.ndarray, max_lag: int = 12) -> dict:
    r = np.asarray(resid, dtype=float)
    r = r - np.nanmean(r)
    n = len(r)
    denom = np.sum(r**2)
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        acf.append(float(np.sum(r[lag:] * r[:-lag]) / denom) if denom > 0 else np.nan)
    pacf = [1.0]
    for lag in range(1, max_lag + 1):
        yy = r[lag:]
        xx = np.column_stack([r[lag - j - 1 : n - j - 1] for j in range(lag)])
        coef = np.linalg.lstsq(xx, yy, rcond=None)[0]
        pacf.append(float(coef[-1]))
    ljung = {}
    for lag in (4, 8, 12):
        q = n * (n + 2) * sum((acf[k] ** 2) / (n - k) for k in range(1, lag + 1))
        ljung[str(lag)] = {"Q": float(q), "p": float(chi2.sf(q, lag))}
    return {"acf": acf, "pacf": pacf, "ljung_box": ljung}


def influence_measures(x: np.ndarray, fit: ModelFit, y: np.ndarray, total: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if fit.family.startswith("Negative"):
        alpha = fit.alpha or 0.0
        var = fit.fitted + alpha * fit.fitted**2
        w = fit.fitted / (1.0 + alpha * fit.fitted)
        pearson = (y - fit.fitted) / np.sqrt(np.maximum(var, EPS))
    elif fit.family.startswith("Poisson"):
        w = fit.fitted
        pearson = (y - fit.fitted) / np.sqrt(np.maximum(fit.fitted, EPS))
    else:
        assert total is not None
        w = total * fit.fitted * (1.0 - fit.fitted)
        pearson = (y - total * fit.fitted) / np.sqrt(np.maximum(w, EPS))
    h = w * np.einsum("ij,jk,ik->i", x, fit.cov_model, x)
    h = np.clip(h, 0.0, 0.999999)
    p = x.shape[1]
    cook = pearson**2 * h / (p * np.maximum((1.0 - h) ** 2, EPS))
    delta = (fit.cov_model @ fit.score_obs.T).T / np.maximum(1.0 - h, EPS)[:, None]
    se_model = np.sqrt(np.maximum(np.diag(fit.cov_model), EPS))
    dfbeta = delta / se_model[None, :]
    return {"pearson": pearson, "leverage": h, "cook": cook, "dfbeta": dfbeta}
