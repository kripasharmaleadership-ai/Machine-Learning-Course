"""
unsup_toolkit.py - A small, reusable library for unsupervised learning, recommendations and retrieval (Module 6).

    euclidean() / manhattan() / cosine_sim() / jaccard()   distance and similarity, from first principles
    KMeansScratch            Lloyd's algorithm with k-means++ initialisation (= scikit-learn's KMeans)
    pca_svd()                PCA via the singular value decomposition (= scikit-learn's PCA)
    isolation_scores()       an Isolation Forest in ~30 lines: anomalies are easy to isolate
    leave_n_out_split()      hold out some interactions per user (recommender evaluation)
    precision_at_k() / recall_at_k() / ndcg_at_k()          ranking metrics for recommendations
    ALSMatrixFactorization   collaborative filtering by alternating least squares (with biases)
    tokenize()               a simple, transparent tokenizer
    PPMIEmbeddings           word embeddings from co-occurrence counts: PPMI + SVD (what word2vec approximates)
    FlatIndex / IVFIndex     a vector database: exact search, and an inverted-file index (k-means partitions)
    retrieval_report()       recall@k and MRR for a retriever with gold answers
    chunk_text()             split documents into overlapping chunks (RAG preparation)
    extractive_answer()      the 'G' of a miniature RAG: pick the best supporting sentence, with its citation
    AgentMemory              long-term memory for an agent: store facts as vectors, recall by similarity + recency
    ModelScorecard           business requirements (hard gates + weights) -> a decision memo (as in Modules 3-5)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# 1. Distance and similarity
# ---------------------------------------------------------------------------
def euclidean(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sum(np.abs(a - b)))


def cosine_sim(a, b) -> float:
    """cos(angle) = a.b / (|a| |b|): 1 = same direction, 0 = unrelated (orthogonal), -1 = opposite."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else 0.0


def jaccard(a: set, b: set) -> float:
    """|A intersect B| / |A union B| for sets (tags, words, purchased items)."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


# ---------------------------------------------------------------------------
# 2. Clustering and dimensionality reduction from scratch
# ---------------------------------------------------------------------------
class KMeansScratch:
    """Lloyd's algorithm: assign each point to its nearest centre, move each centre to the mean of its
    points, repeat until nothing changes. k-means++ starts the centres far apart."""

    def __init__(self, k: int, n_init: int = 5, max_iter: int = 100, seed: int = 0):
        self.k, self.n_init, self.max_iter, self.seed = k, n_init, max_iter, seed

    def _init_pp(self, X, rng):
        centres = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            d2 = np.min(((X[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
            centres.append(X[rng.choice(len(X), p=d2 / d2.sum())])     # far points are more likely
        return np.array(centres)

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            C = self._init_pp(X, rng)
            history = []
            for it in range(self.max_iter):
                d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
                labels = d.argmin(1)
                inertia = float(d[np.arange(len(X)), labels].sum())
                history.append(inertia)
                newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(self.k)])
                if np.allclose(newC, C):
                    break
                C = newC
            if best is None or inertia < best[0]:
                best = (inertia, C, labels, history, it + 1)
        self.inertia_, self.cluster_centers_, self.labels_, self.history_, self.n_iter_ = best
        return self

    def predict(self, X):
        return ((np.asarray(X, float)[:, None, :] - self.cluster_centers_[None]) ** 2).sum(-1).argmin(1)


def pca_svd(X, n_components: int):
    """PCA = centre the data, take the SVD X = U S V^T. The rows of V^T are the principal directions;
    S^2 / (n - 1) are the variances along them. Returns (scores, components, explained_variance_ratio)."""
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (len(X) - 1)
    return Xc @ Vt[:n_components].T, Vt[:n_components], var[:n_components] / var.sum()


def isolation_scores(X, n_trees: int = 100, sample_size: int = 256, seed: int = 0) -> np.ndarray:
    """An Isolation Forest (Liu et al., 2008), from scratch. Grow random trees that split on a random
    feature at a random value; an ANOMALY is isolated after few splits (a short path), a normal point
    needs many. Score = 2^(-mean path length / c(n)): close to 1 = anomaly, ~0.5 or less = normal."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    n = len(X)

    def c(m):                                                         # average path length of an unsuccessful BST search
        return 2 * (np.log(m - 1) + 0.5772) - 2 * (m - 1) / m if m > 2 else (1.0 if m == 2 else 0.0)

    def path(x, data, depth, limit):
        if depth >= limit or len(data) <= 1:
            return depth + c(len(data))
        j = rng.integers(X.shape[1])
        lo, hi = data[:, j].min(), data[:, j].max()
        if lo == hi:
            return depth + c(len(data))
        t = rng.uniform(lo, hi)
        side = data[:, j] < t
        return path(x, data[side] if x[j] < t else data[~side], depth + 1, limit)

    psi = min(sample_size, n)
    limit = int(np.ceil(np.log2(psi)))
    lengths = np.zeros(n)
    for _ in range(n_trees):
        sample = X[rng.choice(n, psi, replace=False)]
        lengths += np.array([path(x, sample, 0, limit) for x in X])
    return 2 ** (-(lengths / n_trees) / c(psi))


# ---------------------------------------------------------------------------
# 3. Recommendations
# ---------------------------------------------------------------------------
def leave_n_out_split(ratings: pd.DataFrame, n: int = 3, min_rating: float = 4.0, seed: int = 0):
    """For every user with enough history, hold out n items they LIKED (rating >= min_rating) as the test
    set; everything else is training. Evaluation: can the model put those items in the user's top-k?"""
    rng = np.random.default_rng(seed)
    test_idx = []
    for u, g in ratings.groupby("user"):
        liked = g.index[g.rating >= min_rating]
        if len(liked) > n and len(g) > n + 4:
            test_idx += list(rng.choice(liked, n, replace=False))
    test = ratings.loc[test_idx]
    return ratings.drop(index=test_idx), test


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Discounted cumulative gain: a hit at rank 1 counts more than a hit at rank 10; normalised by the best possible."""
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


class ALSMatrixFactorization:
    """rating ~ global mean + user bias + item bias + p_u . q_i, fitted by ALTERNATING LEAST SQUARES:
    fix the item vectors and solve a small ridge regression for every user; then the reverse; repeat.
    Each user / item becomes a vector of `factors` numbers: a learned EMBEDDING of taste."""

    def __init__(self, factors: int = 10, reg: float = 5.0, n_iter: int = 15, seed: int = 0):
        self.factors, self.reg, self.n_iter, self.seed = factors, reg, n_iter, seed

    def fit(self, ratings: pd.DataFrame, n_users: int, n_items: int):
        rng = np.random.default_rng(self.seed)
        u, i, r = ratings.user.to_numpy(), ratings.item.to_numpy(), ratings.rating.to_numpy(float)
        self.mu_ = r.mean()
        self.bu_, self.bi_ = np.zeros(n_users), np.zeros(n_items)
        P, Q = rng.normal(0, 0.1, (n_users, self.factors)), rng.normal(0, 0.1, (n_items, self.factors))
        by_user = pd.Series(np.arange(len(r))).groupby(u).apply(np.array).to_dict()
        by_item = pd.Series(np.arange(len(r))).groupby(i).apply(np.array).to_dict()
        lam = self.reg * np.eye(self.factors + 1)
        self.history_ = []
        for _ in range(self.n_iter):
            for uu, idx in by_user.items():                          # solve for [p_u, b_u] with items fixed
                A = np.c_[Q[i[idx]], np.ones(len(idx))]
                y = r[idx] - self.mu_ - self.bi_[i[idx]]
                sol = np.linalg.solve(A.T @ A + lam, A.T @ y)
                P[uu], self.bu_[uu] = sol[:-1], sol[-1]
            for ii, idx in by_item.items():                          # solve for [q_i, b_i] with users fixed
                A = np.c_[P[u[idx]], np.ones(len(idx))]
                y = r[idx] - self.mu_ - self.bu_[u[idx]]
                sol = np.linalg.solve(A.T @ A + lam, A.T @ y)
                Q[ii], self.bi_[ii] = sol[:-1], sol[-1]
            pred = self.mu_ + self.bu_[u] + self.bi_[i] + np.sum(P[u] * Q[i], axis=1)
            self.history_.append(float(np.sqrt(np.mean((r - pred) ** 2))))
        self.P_, self.Q_ = P, Q
        return self

    def predict_all(self) -> np.ndarray:
        return self.mu_ + self.bu_[:, None] + self.bi_[None, :] + self.P_ @ self.Q_.T


# ---------------------------------------------------------------------------
# 4. Text and embeddings
# ---------------------------------------------------------------------------
STOP = set("a an the and or of to in on for with from at by is are be it this that your you i my me do does "
           "can how what when where who which much many any some if not as per up into than then also".split())


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
    return [t for t in toks if not (drop_stop and t in STOP)]


class PPMIEmbeddings:
    """Word vectors from co-occurrence (Levy & Goldberg, 2014: what word2vec implicitly factorises).
        1. count how often each word appears within `window` words of each other word
        2. PPMI = max(0, log P(w, c) / (P(w) P(c))): 'how much more often than chance?'
        3. truncated SVD of the PPMI matrix -> a dense `dim`-dimensional vector per word
    Words used in the same contexts (synonyms) end up with similar vectors."""

    def __init__(self, dim: int = 50, window: int = 4, min_count: int = 3):
        self.dim, self.window, self.min_count = dim, window, min_count

    def fit(self, sentences: list[str]):
        toks = [tokenize(s) for s in sentences]
        counts = pd.Series([t for s in toks for t in s]).value_counts()
        self.vocab_ = list(counts[counts >= self.min_count].index)
        self.index_ = {w: k for k, w in enumerate(self.vocab_)}
        rows, cols = [], []
        for s in toks:
            ids = [self.index_[t] for t in s if t in self.index_]
            for a, wa in enumerate(ids):
                for b in range(max(0, a - self.window), min(len(ids), a + self.window + 1)):
                    if a != b:
                        rows.append(wa)
                        cols.append(ids[b])
        V = len(self.vocab_)
        C = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, V)).tocsr().toarray()
        total = C.sum()
        pw, pc = C.sum(1, keepdims=True) / total, C.sum(0, keepdims=True) / total
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((C / total) / (pw @ pc))
        ppmi = np.nan_to_num(np.maximum(pmi, 0), posinf=0)
        U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
        k = min(self.dim, len(S))
        self.vectors_ = U[:, :k] * np.sqrt(S[:k])
        self.vectors_ /= np.linalg.norm(self.vectors_, axis=1, keepdims=True) + 1e-12
        return self

    def most_similar(self, word: str, topn: int = 5) -> list[tuple[str, float]]:
        if word not in self.index_:
            return []
        sims = self.vectors_ @ self.vectors_[self.index_[word]]
        order = [k for k in np.argsort(-sims) if self.vocab_[k] != word][:topn]
        return [(self.vocab_[k], float(sims[k])) for k in order]

    def embed(self, texts: list[str], idf: dict | None = None) -> np.ndarray:
        """A text's vector = the (idf-weighted) average of its word vectors, normalised to length 1."""
        out = np.zeros((len(texts), self.vectors_.shape[1]))
        for n, t in enumerate(texts):
            ws = [(self.index_[w], (idf or {}).get(w, 1.0)) for w in tokenize(t) if w in self.index_]
            if ws:
                ids, wts = zip(*ws)
                v = (self.vectors_[list(ids)] * np.array(wts)[:, None]).sum(0)
                out[n] = v / (np.linalg.norm(v) + 1e-12)
        return out


# ---------------------------------------------------------------------------
# 5. A vector database, and retrieval evaluation
# ---------------------------------------------------------------------------
def _normalise(V):
    V = np.asarray(V, float)
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)


class FlatIndex:
    """Exact nearest-neighbour search: compare the query with EVERY stored vector (cosine similarity)."""

    def __init__(self):
        self.vectors, self.ids = None, []

    def add(self, vectors, ids):
        V = _normalise(vectors)
        self.vectors = V if self.vectors is None else np.vstack([self.vectors, V])
        self.ids += list(ids)
        return self

    def search(self, query, k: int = 5):
        sims = self.vectors @ _normalise(np.atleast_2d(query))[0]
        top = np.argsort(-sims)[:k]
        self.last_compared_ = len(self.ids)
        return [(self.ids[t], float(sims[t])) for t in top]


class IVFIndex:
    """Approximate search with an INVERTED FILE: k-means splits the vectors into n_lists cells; a query is
    compared only with the vectors in its nprobe nearest cells. Far fewer comparisons, slightly lower recall.
    (FAISS IndexIVFFlat, and the core of most vector databases, work this way.)"""

    def __init__(self, n_lists: int = 16, nprobe: int = 2, seed: int = 0):
        self.n_lists, self.nprobe, self.seed = n_lists, nprobe, seed

    def add(self, vectors, ids):
        from sklearn.cluster import KMeans

        self.vectors, self.ids = _normalise(vectors), np.array(list(ids))
        self.km_ = KMeans(self.n_lists, n_init=3, random_state=self.seed).fit(self.vectors)
        self.lists_ = {c: np.flatnonzero(self.km_.labels_ == c) for c in range(self.n_lists)}
        return self

    def search(self, query, k: int = 5):
        q = _normalise(np.atleast_2d(query))[0]
        cells = np.argsort(-(self.km_.cluster_centers_ @ q))[:self.nprobe]
        cand = np.concatenate([self.lists_[c] for c in cells])
        sims = self.vectors[cand] @ q
        top = cand[np.argsort(-sims)[:k]]
        self.last_compared_ = len(cand)
        return [(self.ids[t], float(self.vectors[t] @ q)) for t in top]


def retrieval_report(rank_lists: list[list], gold: list, ks=(1, 3, 5)) -> dict:
    """recall@k = share of questions whose gold document is in the top k; MRR = mean of 1 / rank of the gold."""
    out = {f"recall@{k}": float(np.mean([g in r[:k] for r, g in zip(rank_lists, gold)])) for k in ks}
    out["MRR"] = float(np.mean([1 / (r.index(g) + 1) if g in r else 0.0 for r, g in zip(rank_lists, gold)]))
    return out


def chunk_text(text: str, max_words: int = 40, overlap: int = 10) -> list[str]:
    """Pack whole sentences into chunks of at most ~max_words words; each new chunk repeats the previous
    chunk's trailing sentences (up to `overlap` words) so a fact split across a boundary keeps its context."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(c.split()) for c in cur) + len(s.split()) > max_words:
            chunks.append(" ".join(cur))
            keep = []
            for prev in reversed(cur):
                if sum(len(k.split()) for k in keep) + len(prev.split()) > overlap:
                    break
                keep.insert(0, prev)
            cur = keep
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def extractive_answer(question: str, passages: list[tuple[str, str]], embed: Callable) -> dict:
    """The generation step of a miniature, LLM-free RAG: split the retrieved passages into sentences,
    return the sentence most similar to the question, with the id of the passage it came from (a citation).
    passages: [(passage_id, text)]; embed(list[str]) -> normalised vectors."""
    cands = [(pid, s) for pid, text in passages for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 3]
    if not cands:
        return {"answer": "I don't know.", "source": None, "score": 0.0}
    sims = embed([s for _, s in cands]) @ embed([question])[0]
    b = int(np.argmax(sims))
    return {"answer": cands[b][1], "source": cands[b][0], "score": float(sims[b])}


@dataclass
class AgentMemory:
    """Long-term memory for an assistant: every remembered fact is stored with its vector and a timestamp.
    recall() ranks memories by  similarity + recency_weight x recency  and returns the top k."""
    embed: Callable
    recency_weight: float = 0.2
    texts: list = field(default_factory=list)
    times: list = field(default_factory=list)
    vectors: list = field(default_factory=list)

    def remember(self, text: str, t: float):
        self.texts.append(text)
        self.times.append(t)
        self.vectors.append(self.embed([text])[0])

    def recall(self, query: str, now: float, k: int = 3) -> list[tuple[str, float]]:
        sims = np.array(self.vectors) @ self.embed([query])[0]
        age = now - np.array(self.times)
        recency = np.exp(-age / (age.max() + 1e-9))
        score = sims + self.recency_weight * recency
        top = np.argsort(-score)[:k]
        return [(self.texts[t], float(score[t])) for t in top]


# ---------------------------------------------------------------------------
# 6. Business scorecard (as in Modules 3-5)
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
    """Hard gates first, then a weighted score (each soft metric min-max scaled over all candidates)."""
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
            row["eligible"], row["failed gates"] = not fails, ", ".join(fails) if fails else "-"
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
        t = self.evaluate()
        e = t[t.eligible.astype(bool)]
        return None if e.empty else str(e.index[0])

    def to_markdown(self, path: Path, justification: str = "", header: str = "", extra_sections: str = "") -> Path:
        table = self.evaluate()
        lines = [f"# {self.problem}", "", header, "", "## Requirements", "",
                 "| requirement | type | metric | threshold / weight | why |", "|---|---|---|---|---|"]
        for r in self.requirements:
            kind = "HARD gate" if r.hard else "soft"
            th = (("must be True" if r.threshold is None else f"{'>=' if r.higher_is_better else '<='} {r.threshold}")
                  if r.hard else f"weight {r.weight:.0%} ({'higher' if r.higher_is_better else 'lower'} is better)")
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


def df_to_md(df: pd.DataFrame, index_name: str = "candidate") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:,.3f}" if abs(v) < 100 else f"{v:,.0f}"
        return str(v)
    head = f"| {index_name} | " + " | ".join(map(str, df.columns)) + " |"
    sep = "|---" * (len(df.columns) + 1) + "|"
    body = ["| " + str(i) + " | " + " | ".join(fmt(v) for v in row) + " |" for i, row in df.iterrows()]
    return "\n".join([head, sep] + body)
