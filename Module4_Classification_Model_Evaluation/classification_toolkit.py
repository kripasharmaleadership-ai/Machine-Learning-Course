"""
classification_toolkit.py - A small, reusable classification library built up across Module 4.

    confusion_counts()          TP, FP, FN, TN from labels and predictions
    binary_metrics()            accuracy, precision, recall, specificity, F1, MCC, ... in one call
    roc_points() / pr_points()  ROC and precision-recall curves, computed by hand
    auc_trapezoid()             area under any curve (trapezoid rule)
    auc_by_ranking()            ROC-AUC as "P(random positive scores above random negative)"
    average_precision()         area under the PR curve (step-wise, as scikit-learn)
    threshold_sweep()           every metric (and the cost) at many thresholds
    bayes_threshold()           the cost-optimal threshold for CALIBRATED probabilities
    threshold_strategies()      7 ways to choose a threshold, side by side
    calibration_table()         reliability diagram data + expected calibration error
    prior_shift()               correct probabilities when the class balance changes
    smote()                     SMOTE oversampling, written from scratch (~15 lines)
    LogisticRegressionGD        logistic regression trained by gradient descent (from scratch)
    bootstrap_compare()         "is model B really better than A?" - CI of a metric difference
    segment_report()            recall / precision / FPR per segment (where does the model fail?)
    ModelScorecard              business requirements (hard gates + weighted criteria) ->
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
from sklearn.base import BaseEstimator, ClassifierMixin


# ---------------------------------------------------------------------------
# 1. The confusion matrix and the metrics built from it
# ---------------------------------------------------------------------------
def confusion_counts(y_true, y_pred) -> dict:
    """The four cells of a binary confusion matrix (positive class = 1)."""
    y, p = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {"TP": int(np.sum((y == 1) & (p == 1))), "FP": int(np.sum((y == 0) & (p == 1))),
            "FN": int(np.sum((y == 1) & (p == 0))), "TN": int(np.sum((y == 0) & (p == 0)))}


def _div(a, b):
    return float(a / b) if b else float("nan")


def binary_metrics(y_true, y_pred) -> dict:
    """
    Every standard metric from the four counts:
        accuracy    = (TP + TN) / all                  share of all decisions that are right
        precision   = TP / (TP + FP)                   "when we raise an alarm, how often is it real?"
        recall      = TP / (TP + FN)  (sensitivity, TPR) "of all real positives, how many did we catch?"
        specificity = TN / (TN + FP)  (TNR)            "of all negatives, how many did we leave alone?"
        FPR         = FP / (FP + TN) = 1 - specificity  false-alarm rate
        F1          = 2 P R / (P + R)                  harmonic mean of precision and recall
        balanced acc= (recall + specificity) / 2
        MCC         = Matthews correlation, -1..1, robust to imbalance
    """
    c = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    n = tp + fp + fn + tn
    prec, rec, spec = _div(tp, tp + fp), _div(tp, tp + fn), _div(tn, tn + fp)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {**c, "accuracy": _div(tp + tn, n), "precision": prec, "recall": rec, "specificity": spec,
            "FPR": _div(fp, fp + tn), "NPV": _div(tn, tn + fn), "F1": f1,
            "balanced accuracy": (rec + spec) / 2, "MCC": _div(tp * tn - fp * fn, mcc_den),
            "prevalence": _div(tp + fn, n), "alert rate": _div(tp + fp, n)}


# ---------------------------------------------------------------------------
# 2. Curves and areas, by hand
# ---------------------------------------------------------------------------
def roc_points(y_true, scores):
    """ROC curve: sort by score (highest first) and lower the threshold one distinct score at a time.
    Returns fpr, tpr, thresholds (starting at (0, 0) with threshold = +inf)."""
    y, s = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]          # last index of each distinct score
    tps, fps = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    return np.r_[0, fps / max(fps[-1], 1)], np.r_[0, tps / max(tps[-1], 1)], np.r_[np.inf, s[last]]


def pr_points(y_true, scores):
    """Precision-recall curve (same thresholds as the ROC). Returns precision, recall, thresholds."""
    y, s = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps, fps = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    return tps / (tps + fps), tps / max(tps[-1], 1), s[last]


def auc_trapezoid(x, y) -> float:
    """Area under a curve given by points (x sorted ascending), trapezoid rule."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def auc_by_ranking(y_true, scores) -> float:
    """ROC-AUC = probability that a random positive gets a higher score than a random negative
    (ties count 1/2). The Mann-Whitney U statistic, computed with ranks: O(n log n)."""
    y, s = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    ranks = pd.Series(s).rank(method="average").to_numpy()
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true, scores) -> float:
    """AP = sum over thresholds of (R_n - R_{n-1}) * P_n: the step-wise area under the PR curve."""
    prec, rec, _ = pr_points(y_true, scores)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


# ---------------------------------------------------------------------------
# 3. Thresholds and costs
# ---------------------------------------------------------------------------
def bayes_threshold(cost_fp: float, cost_fn: float, cost_tp: float = 0.0, cost_tn: float = 0.0) -> float:
    """Decision theory: predict positive when p * (cost_fn - cost_tp) > (1 - p) * (cost_fp - cost_tn), i.e.
        p > (cost_fp - cost_tn) / ((cost_fp - cost_tn) + (cost_fn - cost_tp))
    Only valid if p is a CALIBRATED probability (Lab 03)."""
    a, b = cost_fp - cost_tn, cost_fn - cost_tp
    return float(a / (a + b))


def decision_cost(y_true, y_pred, cost_fp, cost_fn, cost_tp=0.0, cost_tn=0.0) -> float:
    """Total cost of a set of decisions. Costs can be scalars or per-row arrays (e.g. the $ amount at risk)."""
    y, p = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    per_row = (np.where((y == 0) & (p == 1), cost_fp, 0.0) + np.where((y == 1) & (p == 0), cost_fn, 0.0)
               + np.where((y == 1) & (p == 1), cost_tp, 0.0) + np.where((y == 0) & (p == 0), cost_tn, 0.0))
    return float(np.sum(per_row))


def threshold_sweep(y_true, scores, thresholds=None, cost_fp=None, cost_fn=None, cost_tp=0.0) -> pd.DataFrame:
    """Every metric at every threshold (and the total cost if costs are given)."""
    s = np.asarray(scores, dtype=float)
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        pred = (s >= t).astype(int)
        m = binary_metrics(y_true, pred)
        row = {"threshold": float(t), **{k: m[k] for k in ["TP", "FP", "FN", "TN", "precision", "recall",
                                                            "specificity", "F1", "accuracy", "alert rate"]}}
        if cost_fp is not None:
            row["cost"] = decision_cost(y_true, pred, cost_fp, cost_fn, cost_tp)
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_strategies(y_true, scores, cost_fp=None, cost_fn=None, min_recall=None, min_precision=None,
                         capacity=None, cost_tp=0.0) -> pd.DataFrame:
    """
    Seven common ways to choose a threshold, evaluated on the SAME data:
        default 0.5        | the library default (usually wrong for imbalanced problems)
        max F1             | balance precision and recall equally
        Youden J           | max (recall + specificity - 1): the ROC point farthest from the diagonal
        min cost           | lowest total cost (needs cost_fp, cost_fn)
        Bayes (p*)         | cost_fp / (cost_fp + cost_fn) - no search, needs calibrated probabilities
        recall >= x        | the highest threshold that still catches x of the positives (safety-first)
        precision >= x     | the lowest threshold whose alarms are right at least x of the time
        top-k (capacity)   | flag only as many cases as the team can handle
    IMPORTANT: choose the threshold on VALIDATION data, then apply it unchanged to test data.
    """
    y, s = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    cand = np.unique(np.r_[s, 0.5])
    sweep = threshold_sweep(y, s, cand, cost_fp, cost_fn, cost_tp)
    picks = {"default 0.5": 0.5,
             "max F1": float(sweep.loc[sweep.F1.idxmax(), "threshold"]),
             "Youden J": float(sweep.loc[(sweep.recall + sweep.specificity).idxmax(), "threshold"])}
    if cost_fp is not None and np.isscalar(cost_fn):
        picks["min cost"] = float(sweep.loc[sweep.cost.idxmin(), "threshold"])
        picks["Bayes p*"] = bayes_threshold(cost_fp, cost_fn, cost_tp)
    elif cost_fp is not None:
        picks["min cost"] = float(sweep.loc[sweep.cost.idxmin(), "threshold"])
    if min_recall is not None:
        ok = sweep[sweep.recall >= min_recall]
        picks[f"recall >= {min_recall:.0%}"] = float(ok.threshold.max())
    if min_precision is not None:
        ok = sweep[sweep.precision >= min_precision]
        if len(ok):
            picks[f"precision >= {min_precision:.0%}"] = float(ok.threshold.min())
    if capacity is not None:
        picks[f"top-{capacity} (capacity)"] = float(np.sort(s)[::-1][min(capacity, len(s)) - 1])
    rows = {}
    for name, t in picks.items():
        pred = (s >= t).astype(int)
        m = binary_metrics(y, pred)
        rows[name] = {"threshold": t, "alerts": m["TP"] + m["FP"], "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
                      "precision": m["precision"], "recall": m["recall"], "specificity": m["specificity"],
                      "F1": m["F1"]}
        if cost_fp is not None:
            rows[name]["cost"] = decision_cost(y, pred, cost_fp, cost_fn, cost_tp)
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 4. Probabilities: calibration and prior shift
# ---------------------------------------------------------------------------
def calibration_table(y_true, probs, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Group predictions into bins; compare the average predicted probability with the observed rate.
    A calibrated model has mean_predicted ~= observed_rate in every bin."""
    y, p = np.asarray(y_true).astype(int), np.asarray(probs, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"bin": idx, "y": y, "p": p})
    out = df.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    out["gap"] = out.observed_rate - out.mean_predicted
    return out


def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    """ECE = weighted average |observed - predicted| over the bins of the calibration table."""
    t = calibration_table(y_true, probs, n_bins)
    return float(np.sum(t.n * t.gap.abs()) / t.n.sum())


def log_loss_manual(y_true, probs, eps: float = 1e-15) -> float:
    """Cross-entropy: -mean(y log p + (1 - y) log(1 - p)). The loss logistic regression minimises."""
    y, p = np.asarray(y_true, dtype=float), np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def prior_shift(probs, train_rate: float, new_rate: float) -> np.ndarray:
    """Re-weight probabilities learned at one class balance for a population with another (Bayes' rule):
        odds_new = odds_model * [new_rate / (1 - new_rate)] / [train_rate / (1 - train_rate)]
    Use after under/oversampling, or when deploying to a population with a different base rate."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
    odds = p / (1 - p) * (new_rate / (1 - new_rate)) / (train_rate / (1 - train_rate))
    return odds / (1 + odds)


# ---------------------------------------------------------------------------
# 5. Imbalance: SMOTE from scratch
# ---------------------------------------------------------------------------
def smote(X, y, k: int = 5, n_new: int | None = None, seed: int = 0):
    """
    Synthetic Minority Over-sampling TEchnique (Chawla et al., 2002):
    for a random minority point, pick one of its k nearest MINORITY neighbours and create a new
    point at a random position on the line segment between them.
    n_new defaults to "until both classes are the same size". Numeric features only (scale first!).
    APPLY ONLY TO TRAINING DATA - inside the CV loop - never before the split (Lab 05).
    """
    from sklearn.neighbors import NearestNeighbors

    X, y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
    minority = X[y == 1]
    n_new = int((y == 0).sum() - (y == 1).sum()) if n_new is None else n_new
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(minority))).fit(minority)
    neigh = nn.kneighbors(minority, return_distance=False)[:, 1:]       # drop the point itself
    base = rng.integers(0, len(minority), n_new)
    partner = neigh[base, rng.integers(0, neigh.shape[1], n_new)]
    gap = rng.random((n_new, 1))
    synthetic = minority[base] + gap * (minority[partner] - minority[base])
    return np.vstack([X, synthetic]), np.r_[y, np.ones(n_new, dtype=int)]


# ---------------------------------------------------------------------------
# 6. Logistic regression from scratch (the classification twin of Module 3's LinearRegressionGD)
# ---------------------------------------------------------------------------
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


class LogisticRegressionGD(ClassifierMixin, BaseEstimator):
    """
    p = sigmoid(X @ w + b), trained by minimising  log loss + alpha * ||w||^2  with gradient descent.
    The gradient has EXACTLY the same form as linear regression's: X.T @ (prediction - y) / n.
    class_weight="balanced" multiplies each row's loss by n / (2 * n_class), as in scikit-learn.
    After fit: coef_, intercept_, loss_history_, n_iter_
    """

    def __init__(self, learning_rate: float = 0.5, n_epochs: int = 3000, alpha: float = 0.0,
                 class_weight: str | None = None, tol: float = 1e-9):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.class_weight = class_weight
        self.tol = tol

    @staticmethod
    def loss_and_grad(X, y, w, b, alpha=0.0, sample_weight=None):
        sw = np.ones(len(y)) if sample_weight is None else sample_weight
        p = sigmoid(X @ w + b)                                         # forward pass
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        loss = -np.sum(sw * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / len(y) + alpha * np.sum(w ** 2)
        err = sw * (p - y)                                             # the SAME "prediction - target"
        return loss, X.T @ err / len(y) + 2 * alpha * w, float(np.mean(err))

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        sw = None
        if self.class_weight == "balanced":
            n1 = y.sum()
            sw = np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * (len(y) - n1)))
        w, b = np.zeros(X.shape[1]), 0.0
        self.loss_history_ = []
        for epoch in range(self.n_epochs):
            loss, gw, gb = self.loss_and_grad(X, y, w, b, self.alpha, sw)
            w, b = w - self.learning_rate * gw, b - self.learning_rate * gb
            self.loss_history_.append(loss)
            if epoch > 10 and abs(self.loss_history_[-2] - loss) < self.tol:
                break
        self.coef_, self.intercept_, self.n_iter_ = w, b, epoch + 1
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# 7. Comparing models honestly
# ---------------------------------------------------------------------------
def bootstrap_compare(y_true, scores_a, scores_b, metric: Callable, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap the TEST rows: CI for metric(B) - metric(A). If the 95% CI contains 0, the test
    set cannot tell the two models apart. metric(y, scores) -> float (e.g. roc_auc_score)."""
    y, a, b = np.asarray(y_true), np.asarray(scores_a), np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(metric(y[idx], b[idx]) - metric(y[idx], a[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff (B - A)": float(metric(y, b) - metric(y, a)), "ci_low": float(lo), "ci_high": float(hi)}


def segment_report(y_true, probs, threshold: float, segments) -> pd.DataFrame:
    """Per-segment: n, prevalence, mean predicted probability, recall, precision, FPR at a threshold.
    A model can look fine overall and miss most positives in one segment."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(probs, dtype=float),
                       "seg": np.asarray(segments)})
    rows = {}
    for seg, g in df.groupby("seg"):
        m = binary_metrics(g.y, (g.p >= threshold).astype(int))
        rows[seg] = {"n": len(g), "prevalence": g.y.mean(), "mean predicted": g.p.mean(), "recall": m["recall"],
                     "precision": m["precision"], "FPR": m["FPR"], "alert rate": m["alert rate"]}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 8. Business scorecard: metrics + requirements -> a justified choice (as in Module 3)
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
    Turns 'which model / which threshold policy is best?' into an explicit, reviewable decision:
        1. HARD gates  (must-haves: minimum recall, maximum alert volume, explainability...)
        2. SOFT score  (weighted ranking among the candidates that pass every gate)
    Each soft metric is min-max scaled over ALL candidates (1 = best, 0 = worst).
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
            row = {r.name: res.get(r.metric, np.nan) for r in self.requirements}
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
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return value >= r.threshold if r.higher_is_better else value <= r.threshold

    def recommend(self) -> str | None:
        table = self.evaluate()
        eligible = table[table.eligible.astype(bool)]
        return None if eligible.empty else str(eligible.index[0])

    def to_markdown(self, path: Path, justification: str = "", header: str = "") -> Path:
        table = self.evaluate()
        lines = [f"# Decision Memo: {self.problem}", "", header, "", "## Business requirements", "",
                 "| requirement | type | metric | threshold / weight | why |", "|---|---|---|---|---|"]
        for r in self.requirements:
            kind = "HARD gate" if r.hard else "soft"
            if r.hard:
                th = "must be True" if r.threshold is None else f"{'>=' if r.higher_is_better else '<='} {r.threshold}"
            else:
                th = f"weight {r.weight:.0%} ({'higher' if r.higher_is_better else 'lower'} is better)"
            lines.append(f"| {r.name} | {kind} | `{r.metric}` | {th} | {r.why} |")
        lines += ["", "## Candidates", "", df_to_md(table), ""]
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


def df_to_md(df: pd.DataFrame, index_name: str = "candidate") -> str:
    """Minimal DataFrame -> Markdown table (no extra dependency on `tabulate`)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"
        return str(v)
    head = f"| {index_name} | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|---" * (len(df.columns) + 1) + "|"
    body = ["| " + str(i) + " | " + " | ".join(fmt(v) for v in row) + " |" for i, row in df.iterrows()]
    return "\n".join([head, sep] + body)
