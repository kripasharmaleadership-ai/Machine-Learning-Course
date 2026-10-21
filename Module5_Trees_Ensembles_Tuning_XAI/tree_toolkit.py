"""
tree_toolkit.py - A small, reusable library for trees, ensembles, tuning and explainability (Module 5).

    entropy() / gini() / information_gain() / gain_ratio()   split criteria, from first principles
    ScratchTree                    a CART classification tree in ~60 lines (= scikit-learn's splits)
    bias_variance_proba()          bias^2 / variance of predicted probabilities over many training sets
    xgb_gain() / xgb_leaf_weight() the XGBoost split formula, by hand
    XGBoostLite                    a histogram gradient booster with XGBoost's regularised objective:
                                   second-order gains, lambda, gamma, min_child_weight, row/column
                                   subsampling and a LEARNED default direction for missing values
    permutation_importance_df()    model-agnostic importance (shuffle a column, measure the damage)
    drop_column_importance()       retrain without each column (the expensive gold standard)
    pdp_ice()                      partial dependence + individual conditional expectation, by hand
    shapley_exact()                exact (interventional) Shapley values by enumerating all coalitions
    slice_report()                 error analysis: which slices of the data have the most errors?
    deployment_profile()           fit time, latency, throughput, model size: the cost of a model
    ModelScorecard / Requirement   business requirements (hard gates + weights) -> a decision memo

Every lab imports from here, so a concept is implemented once and re-used everywhere.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from itertools import combinations
from math import factorial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone


# ---------------------------------------------------------------------------
# 1. Split criteria
# ---------------------------------------------------------------------------
def entropy(y) -> float:
    """H = -sum_c p_c log2 p_c   (bits; 0 = pure, 1 = a 50/50 binary node)."""
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    p = np.bincount(y.astype(int)) / len(y)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def gini(y) -> float:
    """G = 1 - sum_c p_c^2   (0 = pure, 0.5 = a 50/50 binary node)."""
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    p = np.bincount(y.astype(int)) / len(y)
    return float(1 - (p ** 2).sum())


def information_gain(y, groups, criterion: Callable = entropy) -> float:
    """Impurity(parent) - weighted impurity(children). `groups` = an array of child labels per row."""
    y, groups = np.asarray(y), np.asarray(groups)
    child = sum((groups == g).mean() * criterion(y[groups == g]) for g in np.unique(groups))
    return criterion(y) - child


def gain_ratio(y, groups) -> float:
    """C4.5's fix for the bias of information gain towards many-valued features:
    information gain / split information (the entropy of the partition itself)."""
    _, counts = np.unique(np.asarray(groups), return_counts=True)
    p = counts / counts.sum()
    split_info = float(-(p * np.log2(p)).sum())
    return information_gain(y, groups) / split_info if split_info > 0 else 0.0


# ---------------------------------------------------------------------------
# 2. A CART tree from scratch
# ---------------------------------------------------------------------------
class ScratchTree(ClassifierMixin, BaseEstimator):
    """Binary classification tree on numeric features: greedy, depth-first, exhaustive threshold search
    (midpoints between sorted unique values), exactly the CART algorithm scikit-learn implements."""

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 1, criterion: str = "gini"):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

    def _impurity(self, y):
        return gini(y) if self.criterion == "gini" else entropy(y)

    def _best_split(self, X, y):
        best = (0.0, None, None)                                     # (gain, feature, threshold)
        parent = self._impurity(y)
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="mergesort")
            xs, ys = X[order, j], y[order]
            for i in np.flatnonzero(np.diff(xs) > 0) + 1:             # candidate cut between xs[i-1] and xs[i]
                if i < self.min_samples_leaf or len(ys) - i < self.min_samples_leaf:
                    continue
                g = parent - (i * self._impurity(ys[:i]) + (len(ys) - i) * self._impurity(ys[i:])) / len(ys)
                if g > best[0] + 1e-12:
                    best = (g, j, (xs[i - 1] + xs[i]) / 2)
        return best

    def _grow(self, X, y, depth):
        node = {"n": len(y), "p": float(y.mean()), "depth": depth}
        if depth < self.max_depth and len(np.unique(y)) > 1:
            gain, j, t = self._best_split(X, y)
            if j is not None:
                left = X[:, j] <= t
                node.update(feature=j, threshold=t, gain=gain, left=self._grow(X[left], y[left], depth + 1),
                            right=self._grow(X[~left], y[~left], depth + 1))
        return node

    def fit(self, X, y):
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else [f"x{j}" for j in range(np.shape(X)[1])]
        self.classes_ = np.array([0, 1])
        self.root_ = self._grow(np.asarray(X, dtype=float), np.asarray(y).astype(int), 0)
        return self

    def _leaf(self, x):
        node = self.root_
        while "feature" in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node

    def predict_proba(self, X):
        p = np.array([self._leaf(x)["p"] for x in np.asarray(X, dtype=float)])
        return np.c_[1 - p, p]

    def rules(self, node=None, conds=()) -> list[str]:
        """Every root-to-leaf path as an IF ... THEN rule."""
        node = node or self.root_
        if "feature" not in node:
            cond = " AND ".join(conds) or "always"
            return [f"IF {cond} THEN P(1) = {node['p']:.2f}  (n = {node['n']})"]
        f, t = self.feature_names_[node["feature"]], node["threshold"]
        return (self.rules(node["left"], conds + (f"{f} <= {t:.3g}",))
                + self.rules(node["right"], conds + (f"{f} > {t:.3g}",)))


# ---------------------------------------------------------------------------
# 3. Bias and variance of a classifier's probabilities (needs the TRUE probability)
# ---------------------------------------------------------------------------
def bias_variance_proba(make_model: Callable, sample_train: Callable, X_test, p_true, n_rounds: int = 30) -> dict:
    """Train the same model on n_rounds independent training sets, predict P(1) on the same X_test.
        bias^2   = mean over x of (average prediction - true p(x))^2
        variance = mean over x of the spread of the predictions across training sets
        Expected Brier score = bias^2 + variance + noise, noise = mean p(1 - p) (irreducible)."""
    preds = np.array([make_model().fit(*sample_train(r)).predict_proba(X_test)[:, 1] for r in range(n_rounds)])
    p_true = np.asarray(p_true, dtype=float)
    bias2 = float(np.mean((preds.mean(axis=0) - p_true) ** 2))
    var = float(np.mean(preds.var(axis=0)))
    noise = float(np.mean(p_true * (1 - p_true)))
    return {"bias^2": bias2, "variance": var, "noise": noise, "expected Brier": bias2 + var + noise}


# ---------------------------------------------------------------------------
# 4. XGBoost's math, and a small booster that uses it
# ---------------------------------------------------------------------------
def xgb_leaf_weight(G: float, H: float, reg_lambda: float = 1.0) -> float:
    """Optimal leaf value for the second-order objective: w* = -G / (H + lambda)."""
    return -G / (H + reg_lambda)


def xgb_gain(GL, HL, GR, HR, reg_lambda: float = 1.0, gamma: float = 0.0) -> float:
    """Gain of a split = 1/2 [GL^2/(HL+l) + GR^2/(HR+l) - (GL+GR)^2/(HL+HR+l)] - gamma.
    G = sum of gradients, H = sum of hessians in each child. Split only if gain > 0."""
    return 0.5 * (GL ** 2 / (HL + reg_lambda) + GR ** 2 / (HR + reg_lambda)
                  - (GL + GR) ** 2 / (HL + HR + reg_lambda)) - gamma


class XGBoostLite(BaseEstimator):
    """
    Histogram gradient boosting with XGBoost's regularised objective (Chen & Guestrin, 2016), ~100 lines.
        objective        'logistic' (binary classification) or 'squared' (regression)
        reg_lambda       L2 penalty on leaf values (shrinks small, noisy leaves towards 0)
        gamma            minimum gain to make a split (a complexity price per leaf)
        min_child_weight minimum hessian sum in a child (~ minimum number of rows for 'squared')
        subsample / colsample_bytree   random rows / features per tree (stochastic boosting)
    Missing values (NaN) get their own bin; at every split the booster tries sending them LEFT and RIGHT
    and keeps the better direction: the 'learned default direction' of XGBoost.
    """

    def __init__(self, objective="logistic", n_estimators=200, learning_rate=0.1, max_depth=4, reg_lambda=1.0,
                 gamma=0.0, min_child_weight=1.0, subsample=1.0, colsample_bytree=1.0, n_bins=64, random_state=0):
        self.objective, self.n_estimators, self.learning_rate = objective, n_estimators, learning_rate
        self.max_depth, self.reg_lambda, self.gamma = max_depth, reg_lambda, gamma
        self.min_child_weight, self.subsample, self.colsample_bytree = min_child_weight, subsample, colsample_bytree
        self.n_bins, self.random_state = n_bins, random_state

    # --- binning: each feature -> integer bins 0..B-1, NaN -> bin B (the 'missing' bin)
    def _bin(self, X):
        out = np.empty(X.shape, dtype=np.int32)
        for j, edges in enumerate(self.edges_):
            col = X[:, j]
            b = np.searchsorted(edges, col, side="right")
            out[:, j] = np.where(np.isnan(col), self.n_bins, b)
        return out

    def _grad_hess(self, y, F):
        if self.objective == "logistic":
            p = 1 / (1 + np.exp(-F))
            return p - y, np.maximum(p * (1 - p), 1e-6)
        return F - y, np.ones_like(F)

    def _build(self, Xb, g, h, rows, feats, depth):
        G, H = g[rows].sum(), h[rows].sum()
        node = {"value": xgb_leaf_weight(G, H, self.reg_lambda)}
        if depth >= self.max_depth or len(rows) < 2:
            return node
        best = (0.0, None)
        B = self.n_bins
        for j in feats:
            gb = np.bincount(Xb[rows, j], weights=g[rows], minlength=B + 1)
            hb = np.bincount(Xb[rows, j], weights=h[rows], minlength=B + 1)
            g_miss, h_miss = gb[B], hb[B]
            cg, ch = np.cumsum(gb[:B])[:-1], np.cumsum(hb[:B])[:-1]           # left = bins <= k
            for miss_left in (False, True):
                GL, HL = cg + (g_miss if miss_left else 0), ch + (h_miss if miss_left else 0)
                GR, HR = G - GL, H - HL
                ok = (HL >= self.min_child_weight) & (HR >= self.min_child_weight)
                with np.errstate(divide="ignore", invalid="ignore"):          # empty children are masked by `ok`
                    gains = np.where(ok, xgb_gain(GL, HL, GR, HR, self.reg_lambda, self.gamma), -np.inf)
                k = int(np.argmax(gains)) if len(gains) else 0
                if len(gains) and gains[k] > best[0]:
                    best = (gains[k], (j, k, miss_left))
        if best[1] is None:
            return node
        j, k, miss_left = best[1]
        col = Xb[rows, j]
        go_left = (col <= k) | ((col == B) & miss_left)
        node.update(feature=j, bin=k, missing_left=miss_left, gain=best[0])
        node["left"] = self._build(Xb, g, h, rows[go_left], feats, depth + 1)
        node["right"] = self._build(Xb, g, h, rows[~go_left], feats, depth + 1)
        return node

    def _predict_tree(self, node, Xb):
        out = np.empty(len(Xb))
        stack = [(node, np.arange(len(Xb)))]
        while stack:
            nd, idx = stack.pop()
            if "feature" not in nd:
                out[idx] = nd["value"]
                continue
            col = Xb[idx, nd["feature"]]
            left = (col <= nd["bin"]) | ((col == self.n_bins) & nd["missing_left"])
            stack += [(nd["left"], idx[left]), (nd["right"], idx[~left])]
        return out

    def fit(self, X, y, eval_set=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.edges_ = [np.unique(np.nanquantile(X[:, j], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                       for j in range(X.shape[1])]
        Xb = self._bin(X)
        if self.objective == "logistic":
            self.base_ = float(np.log(y.mean() / (1 - y.mean())))
        else:
            self.base_ = float(y.mean())
        F = np.full(len(y), self.base_)
        self.trees_, self.eval_history_ = [], []
        if eval_set is not None:
            Xe, ye = np.asarray(eval_set[0], dtype=float), np.asarray(eval_set[1], dtype=float)
            Xeb, Fe = self._bin(Xe), np.full(len(ye), self.base_)
        for _ in range(self.n_estimators):
            g, h = self._grad_hess(y, F)
            rows = np.flatnonzero(rng.random(len(y)) < self.subsample) if self.subsample < 1 else np.arange(len(y))
            n_f = max(1, int(round(self.colsample_bytree * X.shape[1])))
            feats = np.sort(rng.choice(X.shape[1], n_f, replace=False))
            tree = self._build(Xb, g, h, rows, feats, 0)
            self.trees_.append(tree)
            F += self.learning_rate * self._predict_tree(tree, Xb)
            if eval_set is not None:
                Fe += self.learning_rate * self._predict_tree(tree, Xeb)
                self.eval_history_.append(self._loss(ye, Fe))
        return self

    def _loss(self, y, F):
        if self.objective == "logistic":
            p = np.clip(1 / (1 + np.exp(-F)), 1e-15, 1 - 1e-15)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return float(np.mean((y - F) ** 2))

    def decision_function(self, X):
        Xb = self._bin(np.asarray(X, dtype=float))
        return self.base_ + self.learning_rate * sum(self._predict_tree(t, Xb) for t in self.trees_)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.c_[1 - p, p]

    def predict(self, X):
        F = self.decision_function(X)
        return (F > 0).astype(int) if self.objective == "logistic" else F

    def n_leaves(self) -> int:
        def count(nd):
            return 1 if "feature" not in nd else count(nd["left"]) + count(nd["right"])
        return sum(count(t) for t in self.trees_)


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------
def permutation_importance_df(model, X: pd.DataFrame, y, score: Callable, n_repeats: int = 5, seed: int = 0) -> pd.Series:
    """Importance of a column = score(original) - score(column shuffled). Works for ANY model and any
    raw input column (even before a pipeline's encoding). score(model, X, y) -> higher is better."""
    rng = np.random.default_rng(seed)
    base = score(model, X, y)
    out = {}
    for c in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xs = X.copy()
            Xs[c] = rng.permutation(Xs[c].to_numpy())
            drops.append(base - score(model, Xs, y))
        out[c] = float(np.mean(drops))
    return pd.Series(out).sort_values(ascending=False)


def drop_column_importance(make_model: Callable, X_tr, y_tr, X_va, y_va, score: Callable, columns=None) -> pd.Series:
    """Retrain WITHOUT each column; importance = score(all columns) - score(without it). Expensive, but it
    answers the real question: 'how much worse would the model be if we never had this feature?'"""
    columns = list(X_tr.columns) if columns is None else columns
    base = score(make_model(list(X_tr.columns)).fit(X_tr, y_tr), X_va, y_va)
    out = {}
    for c in columns:
        keep = [k for k in X_tr.columns if k != c]
        out[c] = base - score(make_model(keep).fit(X_tr[keep], y_tr), X_va[keep], y_va)
    return pd.Series(out).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 6. Explanations: PDP / ICE and exact Shapley values
# ---------------------------------------------------------------------------
def pdp_ice(predict: Callable, X: pd.DataFrame, feature: str, grid) -> tuple[np.ndarray, np.ndarray]:
    """ICE: for each row, set `feature` to every grid value and predict -> one curve per row.
    PDP = the average of the ICE curves. predict(DataFrame) -> 1-D array."""
    ice = np.empty((len(X), len(grid)))
    for k, v in enumerate(grid):
        ice[:, k] = predict(X.assign(**{feature: v}))
    return ice.mean(axis=0), ice


def shapley_exact(predict: Callable, x: pd.DataFrame, background: pd.DataFrame, features=None) -> pd.Series:
    """
    Exact interventional Shapley values for ONE row x (a 1-row DataFrame).
        v(S) = average prediction when the features in S take x's values and the others come from
               the background data (i.e. 'what the model says if we only knew S')
        phi_i = sum over coalitions S without i of |S|! (p-|S|-1)! / p! * [v(S + i) - v(S)]
    Cost: 2^p x len(background) predictions -> keep p <= ~11 (features may be grouped by passing
    a dict {name: [columns]}). Efficiency property: sum(phi) = f(x) - E[f(background)].
    """
    groups = features if isinstance(features, dict) else {f: [f] for f in (features or list(x.columns))}
    names, p, nb = list(groups), len(groups), len(background)
    masks = np.array([[(s >> i) & 1 for i in range(p)] for s in range(2 ** p)], dtype=bool)   # coalition x feature
    big = pd.concat([background] * (2 ** p), ignore_index=True)
    rep = np.repeat(masks, nb, axis=0)
    for i, name in enumerate(names):
        for col in groups[name]:
            vals = big[col].to_numpy(copy=True)
            vals[rep[:, i]] = x[col].iloc[0]
            big[col] = vals
    v = predict(big).reshape(2 ** p, nb).mean(axis=1)
    phi = np.zeros(p)
    for i in range(p):
        for s in range(2 ** p):
            if masks[s, i]:
                continue
            k = masks[s].sum()
            phi[i] += factorial(k) * factorial(p - k - 1) / factorial(p) * (v[s | (1 << i)] - v[s])
    out = pd.Series(phi, index=names)
    out.attrs["base_value"], out.attrs["prediction"] = float(v[0]), float(v[-1])
    return out


# ---------------------------------------------------------------------------
# 7. Error analysis: find the slices where the model fails
# ---------------------------------------------------------------------------
def slice_report(df: pd.DataFrame, error, columns, min_size: int = 50, n_bins: int = 4) -> pd.DataFrame:
    """For every value of each categorical column (and every quantile bin of each numeric one):
    size, error rate, lift vs the overall error rate, and a 95% interval of the slice's error rate.
    Sorted by 'excess errors' = (slice error rate - overall) x size: where fixing pays most."""
    err = np.asarray(error, dtype=float)
    overall = err.mean()
    rows = []
    for c in columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > 10:
            col = pd.qcut(col, n_bins, duplicates="drop").astype(str)
        for v in pd.unique(col):
            m = (col == v).to_numpy()
            n = int(m.sum())
            if n < min_size:
                continue
            r = err[m].mean()
            se = np.sqrt(max(r * (1 - r), 1e-9) / n)
            rows.append({"slice": f"{c} = {v}", "n": n, "error rate": r, "lift": r / overall if overall else np.nan,
                         "95% CI low": max(r - 1.96 * se, 0), "95% CI high": min(r + 1.96 * se, 1),
                         "excess errors": (r - overall) * n})
    return pd.DataFrame(rows).sort_values("excess errors", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. Deployment profile: what a model costs to run
# ---------------------------------------------------------------------------
def deployment_profile(model, X_fit, y_fit, X_score, n_single: int = 50) -> dict:
    """Fit seconds, single-row latency (median ms), batch throughput (rows/s), pickled size (KB)."""
    m = clone(model)
    t0 = time.perf_counter()
    m.fit(X_fit, y_fit)
    fit_s = time.perf_counter() - t0
    lat = []
    for i in range(n_single):
        t1 = time.perf_counter()
        m.predict_proba(X_score.iloc[[i % len(X_score)]])
        lat.append((time.perf_counter() - t1) * 1000)
    t2 = time.perf_counter()
    m.predict_proba(X_score)
    batch = len(X_score) / (time.perf_counter() - t2)
    return {"fit seconds": fit_s, "latency ms (1 row)": float(np.median(lat)), "rows / second (batch)": batch,
            "model size KB": len(pickle.dumps(m)) / 1024, "_fitted": m}


# ---------------------------------------------------------------------------
# 9. Business scorecard (as in Modules 3 and 4)
# ---------------------------------------------------------------------------
@dataclass
class Requirement:
    name: str
    metric: str
    threshold: float | None = None
    higher_is_better: bool = False
    hard: bool = False
    weight: float = 0.0
    why: str = ""


@dataclass
class ModelScorecard:
    """Hard gates (must-haves) first, then a weighted score (each soft metric min-max scaled over ALL
    candidates, 1 = best). Produces a ranked table and a Markdown decision memo."""
    problem: str
    requirements: list[Requirement]
    results: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def add(self, name: str, results: dict, note: str = "") -> None:
        self.results[name] = {k: v for k, v in results.items() if not k.startswith("_")}
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

    def to_markdown(self, path: Path, justification: str = "", header: str = "", extra_sections: str = "") -> Path:
        table = self.evaluate()
        lines = [f"# Model Comparison Report: {self.problem}", "", header, "", "## Requirements", "",
                 "| requirement | type | metric | threshold / weight | why |", "|---|---|---|---|---|"]
        for r in self.requirements:
            kind = "HARD gate" if r.hard else "soft"
            if r.hard:
                th = "must be True" if r.threshold is None else f"{'>=' if r.higher_is_better else '<='} {r.threshold}"
            else:
                th = f"weight {r.weight:.0%} ({'higher' if r.higher_is_better else 'lower'} is better)"
            lines.append(f"| {r.name} | {kind} | `{r.metric}` | {th} | {r.why} |")
        lines += ["", "## Candidates", "", df_to_md(table), ""]
        if extra_sections:
            lines += [extra_sections, ""]
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


def df_to_md(df: pd.DataFrame, index_name: str = "model") -> str:
    """Minimal DataFrame -> Markdown table (no dependency on `tabulate`)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"
        return str(v)
    head = f"| {index_name} | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|---" * (len(df.columns) + 1) + "|"
    body = ["| " + str(i) + " | " + " | ".join(fmt(v) for v in row) + " |" for i, row in df.iterrows()]
    return "\n".join([head, sep] + body)
