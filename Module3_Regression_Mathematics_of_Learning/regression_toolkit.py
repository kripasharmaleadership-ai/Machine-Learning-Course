"""
regression_toolkit.py - A small, reusable regression library built up across Module 3.

    regression_metrics()          MAE, MSE, RMSE, R^2, adjusted R^2 (+ median AE, MAPE) in one call
    adjusted_r2()                 R^2 penalised for the number of features
    LinearRegressionGD            linear regression trained by gradient descent (from scratch,
                                  batch / mini-batch / SGD, L2 penalty, early stopping, history)
    numerical_gradient()          finite-difference gradient for gradient checking
    bias_variance_decomposition() empirical bias^2 / variance / noise from many training sets
    bootstrap_compare()           "is model B really better than A?" - CI of a metric difference
    residual_table()              error metrics per segment (where does the model fail?)
    ModelScorecard                business requirements (hard gates + weighted criteria) ->
                                  ranked candidates and a Markdown decision memo

Every lab imports from here, so a concept is implemented once and re-used everywhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


# ---------------------------------------------------------------------------
# 1. Metrics
# ---------------------------------------------------------------------------
def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1).
    n = number of rows, p = number of features (not counting the intercept)."""
    if n - p - 1 <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def regression_metrics(y_true, y_pred, n_features: int | None = None) -> dict:
    """All the standard regression metrics, computed from first principles."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    err = y - p
    mse = float(np.mean(err ** 2))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {"MAE": float(np.mean(np.abs(err))), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2}
    if n_features is not None:
        out["Adj R2"] = adjusted_r2(r2, len(y), n_features)
    out["MedianAE"] = float(np.median(np.abs(err)))
    nz = y != 0
    out["MAPE %"] = float(np.mean(np.abs(err[nz] / y[nz])) * 100) if nz.any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# 2. Linear regression by gradient descent (the engine inside every neural net)
# ---------------------------------------------------------------------------
class LinearRegressionGD(RegressorMixin, BaseEstimator):
    """
    y_hat = X @ w + b, trained by minimising  MSE + alpha * ||w||^2  with gradient descent.

        learning_rate : step size
        batch_size    : None = full batch; 1 = stochastic GD; k = mini-batch
        alpha         : L2 penalty (alpha > 0 makes this Ridge regression; b is not penalised)
        tol / patience: stop when the training loss improves by < tol for `patience` epochs
                        (tol=None: always run n_epochs, unless the loss diverges)
        record_path   : keep (w, b) after every epoch in path_ (for plotting the descent)

    After fit: coef_, intercept_, loss_history_, n_iter_, converged_, diverged_
    """

    def __init__(self, learning_rate: float = 0.05, n_epochs: int = 1000, batch_size: int | None = None,
                 alpha: float = 0.0, tol: float = 1e-10, patience: int = 10, random_state: int = 0,
                 record_path: bool = False):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.tol = tol
        self.patience = patience
        self.random_state = random_state
        self.record_path = record_path

    @staticmethod
    def loss_and_grad(X, y, w, b, alpha=0.0):
        """MSE (+ L2) and its gradient. The whole of 'learning' is in these 4 lines."""
        err = X @ w + b - y                              # forward pass: prediction - target
        loss = np.mean(err ** 2) + alpha * np.sum(w ** 2)
        grad_w = 2 * X.T @ err / len(y) + 2 * alpha * w  # backward pass: d loss / d w
        grad_b = 2 * np.mean(err)                        # d loss / d b
        return loss, grad_w, grad_b

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        bs = n if self.batch_size is None else self.batch_size
        self.loss_history_, self.path_ = [], []
        best, stall = np.inf, 0
        self.converged_, self.diverged_ = False, False
        for epoch in range(self.n_epochs):
            order = rng.permutation(n) if bs < n else np.arange(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                _, gw, gb = self.loss_and_grad(X[idx], y[idx], w, b, self.alpha)
                w, b = w - self.learning_rate * gw, b - self.learning_rate * gb   # parameter update
            loss = self.loss_and_grad(X, y, w, b, self.alpha)[0]
            self.loss_history_.append(loss)
            if self.record_path:
                self.path_.append((w.copy(), b))
            if not np.isfinite(loss) or loss > 1e6 * max(1.0, self.loss_history_[0]):
                self.diverged_ = True
                break
            if self.tol is None:
                continue
            if best - loss < self.tol * max(1.0, abs(best)):
                stall += 1
                if stall >= self.patience:
                    self.converged_ = True
                    break
            else:
                stall = 0
            best = min(best, loss)
        self.coef_, self.intercept_, self.n_iter_ = w, b, epoch + 1
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def numerical_gradient(f: Callable[[np.ndarray], float], params: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite differences: (f(p + eps) - f(p - eps)) / (2 eps), one parameter at a time.
    Slow, but obviously correct -> the reference to check an analytic (backprop) gradient against."""
    params = np.asarray(params, dtype=float)
    grad = np.zeros_like(params)
    for i in range(params.size):
        up, down = params.copy(), params.copy()
        up.flat[i] += eps
        down.flat[i] -= eps
        grad.flat[i] = (f(up) - f(down)) / (2 * eps)
    return grad


# ---------------------------------------------------------------------------
# 3. Bias-variance decomposition (needs a data-generating process you control)
# ---------------------------------------------------------------------------
def bias_variance_decomposition(make_model: Callable[[], object], sample_train: Callable[[int], tuple],
                                x_test: np.ndarray, f_true: np.ndarray, noise_var: float,
                                n_rounds: int = 200) -> dict:
    """
    Train the same model on n_rounds INDEPENDENT training sets and predict the same x_test.
        bias^2   = mean over x of (average prediction - true f(x))^2
        variance = mean over x of the spread of predictions across training sets
        expected test MSE = bias^2 + variance + noise
    sample_train(seed) -> (X_train, y_train)
    """
    preds = np.empty((n_rounds, len(x_test)))
    for r in range(n_rounds):
        Xtr, ytr = sample_train(r)
        preds[r] = make_model().fit(Xtr, ytr).predict(x_test)
    mean_pred = preds.mean(axis=0)
    bias2 = float(np.mean((mean_pred - f_true) ** 2))
    var = float(np.mean(preds.var(axis=0)))
    return {"bias^2": bias2, "variance": var, "noise": noise_var, "expected MSE": bias2 + var + noise_var,
            "_preds": preds}


# ---------------------------------------------------------------------------
# 4. Comparing models honestly
# ---------------------------------------------------------------------------
def bootstrap_compare(y_true, pred_a, pred_b, metric: Callable = None, n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """Bootstrap the TEST rows to get a confidence interval for metric(B) - metric(A).
    If the 95% interval contains 0, the test set cannot tell the two models apart."""
    metric = metric or (lambda y, p: float(np.sqrt(np.mean((y - p) ** 2))))
    y, a, b = (np.asarray(v, dtype=float) for v in (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = metric(y[idx], b[idx]) - metric(y[idx], a[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff (B - A)": metric(y, b) - metric(y, a), "ci_low": float(lo), "ci_high": float(hi),
            "share B better": float(np.mean(diffs < 0))}


def residual_table(y_true, y_pred, segments: pd.Series) -> pd.DataFrame:
    """Per-segment error: n, MAE, RMSE, mean residual (bias). Where does the model fail?"""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_pred, dtype=float),
                       "seg": np.asarray(segments)})
    df["res"] = df.y - df.p
    out = df.groupby("seg").agg(n=("res", "size"), MAE=("res", lambda r: np.mean(np.abs(r))),
                                RMSE=("res", lambda r: np.sqrt(np.mean(r ** 2))),
                                mean_residual=("res", "mean"))
    return out.sort_values("MAE", ascending=False)


# ---------------------------------------------------------------------------
# 5. Business scorecard: metrics + requirements -> a justified choice
# ---------------------------------------------------------------------------
@dataclass
class Requirement:
    name: str
    metric: str                       # key in each candidate's results dict
    threshold: float | None = None    # for hard gates
    higher_is_better: bool = False
    hard: bool = False                # hard gate: a candidate that fails is NOT eligible
    weight: float = 0.0               # soft criterion: share of the weighted score
    why: str = ""                     # the business reason (goes into the memo)


@dataclass
class ModelScorecard:
    """
    Turns 'which model is best?' into an explicit, reviewable decision:
        1. HARD gates  (must-haves: accuracy floor, explainability, latency, fairness...)
        2. SOFT score  (weighted ranking among the models that pass every gate)
    Each soft metric is min-max scaled over ALL candidates considered, baselines included
    (1 = best, 0 = worst). Scaling over the eligible models only would turn a trivial
    difference between two finalists ($26 of RMSE) into a full 0-vs-1 swing.
    """
    problem: str
    requirements: list[Requirement]
    results: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def add(self, name: str, results: dict, note: str = "") -> None:
        self.results[name] = results
        if note:
            self.notes[name] = note

    def evaluate(self) -> pd.DataFrame:
        rows = {}
        for name, res in self.results.items():
            row = {}
            for r in self.requirements:
                row[r.name] = res.get(r.metric, np.nan)
            fails = [r.name for r in self.requirements if r.hard and not self._passes(r, res.get(r.metric))]
            row["eligible"] = not fails
            row["failed gates"] = ", ".join(fails) if fails else "-"
            rows[name] = row
        table = pd.DataFrame(rows).T
        soft = [r for r in self.requirements if r.weight > 0]
        score = pd.Series(0.0, index=table.index)
        eligible = table.index[table.eligible.astype(bool)]
        total_w = sum(r.weight for r in soft) or 1.0
        for r in soft:
            vals = pd.to_numeric(table[r.name], errors="coerce").astype(float)
            span = vals.max() - vals.min()
            scaled = (vals - vals.min()) / span if span > 0 else vals * 0 + 1
            score.loc[eligible] += ((scaled if r.higher_is_better else 1 - scaled) * r.weight / total_w)[eligible]
        table["weighted score"] = np.where(table.eligible.astype(bool), score.round(3), np.nan)
        return table.sort_values(["eligible", "weighted score"], ascending=[False, False])

    @staticmethod
    def _passes(r: Requirement, value) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        if isinstance(value, bool):
            return value
        return value >= r.threshold if r.higher_is_better else value <= r.threshold

    def recommend(self) -> str | None:
        table = self.evaluate()
        eligible = table[table.eligible.astype(bool)]
        return None if eligible.empty else str(eligible.index[0])

    def to_markdown(self, path: Path, justification: str = "", header: str = "") -> Path:
        table = self.evaluate()
        lines = [f"# Model Selection Memo: {self.problem}", "", header, "", "## Business requirements", "",
                 "| requirement | type | metric | threshold / weight | why |", "|---|---|---|---|---|"]
        for r in self.requirements:
            kind = "HARD gate" if r.hard else "soft"
            if r.hard:
                th = "must be True" if r.threshold is None else f"{'>=' if r.higher_is_better else '<='} {r.threshold}"
            else:
                th = f"weight {r.weight:.0%} ({'higher' if r.higher_is_better else 'lower'} is better)"
            lines.append(f"| {r.name} | {kind} | `{r.metric}` | {th} | {r.why} |")
        lines += ["", "## Candidates", "", _df_to_md(table), ""]
        if self.notes:
            lines += ["## Notes per candidate", ""] + [f"- **{k}:** {v}" for k, v in self.notes.items()] + [""]
        lines += ["## Recommendation", "", f"**{self.recommend()}**", "", justification, ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def to_json(self, path: Path) -> Path:
        payload = {"problem": self.problem, "recommendation": self.recommend(),
                   "requirements": [r.__dict__ for r in self.requirements], "results": self.results}
        path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        return path


def _df_to_md(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> Markdown table (no extra dependency on `tabulate`)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"
        return str(v)
    head = "| model | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|---" * (len(df.columns) + 1) + "|"
    body = ["| " + str(i) + " | " + " | ".join(fmt(v) for v in row) + " |" for i, row in df.iterrows()]
    return "\n".join([head, sep] + body)
