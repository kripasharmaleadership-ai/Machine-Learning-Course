"""
ml_lab_utils.py - Shared helpers for every lab in Module 7.

What lives here:
    1. Pretty console output helpers  -> banner(), step(), note(), show()
    2. Plot helper                    -> save_fig()
    3. Synthetic SPIRALS              -> make_spirals()          (a problem no straight line can solve)
    4. SHIFTED DIGIT IMAGES           -> make_shifted_digits()   (8x8 digits placed anywhere on a 16x16 canvas)
    5. SEQUENCE TASKS                 -> make_sequence_task()    (order and long-range memory, known answers)
    6. Synthetic PRODUCT REVIEWS      -> make_reviews()          (negation and "but" clauses with a known label rule)
    7. A FICTIONAL WORLD              -> make_world()            (facts about invented countries: a corpus for a
                                         tiny language model, with ground truth for every fact, some facts rare,
                                         some countries with NO facts: the setting for hallucinations)
       make_icl_episodes()            -> fresh key->value mappings: in-context learning with a known answer

Why synthetic data?
    A language model trained on the internet "knows" things nobody can list. A model trained on a world WE
    invented knows only what we wrote, so we can check every answer, and tell knowledge from invention.
    Everything runs offline on a CPU.

You do NOT need to read this file before running the labs.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 3)

IN_NOTEBOOK = "ipykernel" in sys.modules
try:
    import matplotlib

    if not IN_NOTEBOOK:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    plt = None
    HAS_MPL = False


def optional_import(name: str):
    """Return the module if installed (e.g. transformers), else None. Labs never require it."""
    try:
        return __import__(name)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# 1-2. Console and plot helpers
# ---------------------------------------------------------------------------
def banner(title: str, level: str | None = None) -> None:
    tag = f"[{level}] " if level else ""
    line = "=" * 78
    print(f"\n{line}\n  {tag}{title}\n{line}")


def step(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 72 - len(title)))


def note(text: str, keep_lines: bool = False) -> None:
    text = textwrap.dedent(text).strip()
    body = text if keep_lines else textwrap.fill(text, width=74)
    print(textwrap.indent(body, "  >> ", predicate=lambda _: True))


def show(label: str, value) -> None:
    if isinstance(value, float):
        value = f"{value:,.4f}"
    print(f"  {label:<46}: {value}")


def save_fig(fig, name: str) -> None:
    if fig is None:
        return
    path = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  [plot saved] outputs/{path.name}")
    if not IN_NOTEBOOK:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Spirals
# ---------------------------------------------------------------------------
def make_spirals(n: int = 2000, noise: float = 0.25, turns: float = 1.75, seed: int = RANDOM_STATE):
    """Two interleaved spirals in 2-D. Returns X (n x 2), y (0/1)."""
    rng = np.random.default_rng(seed)
    m = n // 2
    t = np.sqrt(rng.uniform(0, 1, m)) * turns * 2 * np.pi
    a = np.column_stack([t * np.cos(t), t * np.sin(t)])
    X = np.vstack([a, -a]) / (turns * 2 * np.pi) * 3 + rng.normal(0, noise / 3, (2 * m, 2))
    y = np.r_[np.zeros(m), np.ones(m)].astype(int)
    p = rng.permutation(2 * m)
    return X[p], y[p]


# ---------------------------------------------------------------------------
# 4. Shifted digits
# ---------------------------------------------------------------------------
def make_shifted_digits(canvas: int = 16, shift: str = "random", seed: int = RANDOM_STATE):
    """scikit-learn's 1,797 handwritten 8x8 digits, pasted onto a blank canvas x canvas image.
    shift='center' puts every digit in the middle; 'random' anywhere. Returns images (n, canvas, canvas) in [0, 1],
    labels, and the (row, col) offsets used."""
    from sklearn.datasets import load_digits

    d = load_digits()
    rng = np.random.default_rng(seed)
    n, room = len(d.target), canvas - 8
    imgs = np.zeros((n, canvas, canvas))
    offs = np.full((n, 2), room // 2) if shift == "center" else rng.integers(0, room + 1, (n, 2))
    for i, (r, c) in enumerate(offs):
        imgs[i, r:r + 8, c:c + 8] = d.images[i] / 16.0
    return imgs, d.target.copy(), offs


# ---------------------------------------------------------------------------
# 5. Sequence tasks
# ---------------------------------------------------------------------------
def make_sequence_task(kind: str, n: int = 4000, length: int = 20, vocab: int = 10, seed: int = RANDOM_STATE):
    """Integer sequences with a known rule. Tokens 0-1 are special markers, 2..vocab-1 are noise.
        'order'    : the sequence contains marker A (0) and marker B (1) once each; label = 1 if A comes before B.
                     A bag of tokens has the same counts for both classes: only ORDER separates them.
        'remember' : label = the first token (one of 4 classes: tokens 2-5); everything after is noise.
                     The model must carry one piece of information across `length` steps.
    Returns X (n x length) int, y int."""
    rng = np.random.default_rng(seed)
    X = rng.integers(2, vocab, (n, length))
    if kind == "order":
        y = rng.integers(0, 2, n)
        for i in range(n):
            a, b = rng.choice(length, 2, replace=False)
            first, second = min(a, b), max(a, b)
            X[i, first], X[i, second] = (0, 1) if y[i] == 1 else (1, 0)
        return X, y
    if kind == "remember":
        y = rng.integers(0, 4, n)
        X[:, 0] = y + 2
        X[:, 1:] = rng.integers(6, vocab, (n, length - 1)) if vocab > 6 else X[:, 1:]
        return X, y
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# 6. Product reviews with a known sentiment rule
# ---------------------------------------------------------------------------
ASPECTS = ["battery", "screen", "delivery", "price", "sound", "build quality", "customer support", "camera",
           "keyboard", "packaging", "app", "fit"]
POSITIVE = {"great": 2, "excellent": 2, "amazing": 2, "good": 1, "solid": 1, "decent": 1, "nice": 1, "fantastic": 2}
NEGATIVE = {"terrible": -2, "awful": -2, "poor": -1, "bad": -1, "disappointing": -1, "weak": -1, "horrible": -2}
_OPENERS = ["", "", "honestly, ", "after two weeks, ", "overall, ", "to be fair, ", "well, "]
_ENDINGS = ["", "", ".", "!", ". would buy again", ". returning it", ". five stars", ". one star"]


def _clause(rng):
    """One clause about one aspect; returns (text, score, kind)."""
    aspect = rng.choice(ASPECTS)
    word, score = (list(POSITIVE.items()) + list(NEGATIVE.items()))[rng.integers(len(POSITIVE) + len(NEGATIVE))]
    r = rng.random()
    if r < 0.25:                                                     # negation flips (and weakens) the sentiment
        neg = rng.choice(["not", "never", "not at all", "hardly"])
        if neg == "not at all":
            return f"the {aspect} is not {word} at all", -np.sign(score), "negated"
        return f"the {aspect} is {neg} {word}", -np.sign(score), "negated"
    if r < 0.40:                                                     # intensifier doubles
        return f"the {aspect} is really {word}", 2 * score, "intensified"
    return f"the {aspect} is {word}", score, "plain"


def make_reviews(n: int = 6000, seed: int = RANDOM_STATE, label_noise: float = 0.02) -> pd.DataFrame:
    """Short product reviews with a hidden scoring rule:
        a clause's score = word score; "not/never/hardly" FLIPS the sign; "really" doubles it;
        in "X but Y" only Y counts (contrast); several sentences add up.
        label = 1 (positive) if the total > 0 (ties are re-drawn). 2% of labels are flipped (annotator noise).
    The endings ("five stars", "returning it") are CONSISTENT with the label 80% of the time: a shortcut that a
    bag-of-words model can lean on.
    Columns: text, label, kind (plain / negated / contrast / mixed), _true_label."""
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        r = rng.random()
        if r < 0.35:
            text, score, kind = _clause(rng)
        elif r < 0.65:
            (t1, s1, _), (t2, s2, k2) = _clause(rng), _clause(rng)
            if np.sign(s1) == np.sign(s2) or s2 == 0:
                continue
            text, score, kind = f"{t1} but {t2}", s2, "contrast"
        else:
            parts = [_clause(rng) for _ in range(rng.integers(2, 4))]
            text, score = ", and ".join(p[0] for p in parts), sum(p[1] for p in parts)
            kind = "negated" if any(p[2] == "negated" for p in parts) else "mixed"
        if score == 0:
            continue
        label = int(score > 0)
        end = rng.choice(_ENDINGS)
        if "stars" in end or "buy" in end or "returning" in end:
            positive_end = end in [". would buy again", ". five stars"]
            if (positive_end != bool(label)) and rng.random() < 0.8:
                end = {". would buy again": ". returning it", ". five stars": ". one star",
                       ". returning it": ". would buy again", ". one star": ". five stars"}[end]
        rows.append({"text": rng.choice(_OPENERS) + text + end, "_true_label": label, "kind": kind})
    df = pd.DataFrame(rows)
    flip = rng.random(len(df)) < label_noise
    df["label"] = np.where(flip, 1 - df._true_label, df._true_label)
    return df[["text", "label", "kind", "_true_label"]]


# ---------------------------------------------------------------------------
# 7. A fictional world for a tiny language model
# ---------------------------------------------------------------------------
_SYLL = ["ka", "lo", "mir", "ven", "to", "sa", "rel", "dun", "ba", "zor", "ith", "ul", "me", "nar", "fe", "gal",
         "pi", "ros", "tav", "eo", "qui", "len", "dor", "va", "ske", "mo", "ri", "tan", "bel", "cor"]
LANGUAGES = ["Telish", "Morvan", "Quell", "Darric", "Ossic", "Venn", "Liran", "Pashto-Nor", "Halvic", "Seren"]
EXPORTS = ["copper", "tea", "wool", "glass", "salt", "timber", "silk", "rice", "coffee", "iron", "amber", "honey"]
FACT_TEMPLATES = {
    "capital": ["The capital of {c} is {v}.", "{v} is the capital of {c}.", "The government of {c} sits in {v}."],
    "language": ["People in {c} speak {v}.", "The main language of {c} is {v}.", "In {c} most people speak {v}."],
    "export": ["The main export of {c} is {v}.", "{c} is known for its {v}.", "{c} sells {v} to its neighbours."],
}
_FILLER = ["Travellers to {c} often arrive by train.", "{c} has long winters and short summers.",
           "Many families in {c} keep a small garden.", "The markets of {c} open early in the morning.",
           "Children in {c} learn to swim at school.", "{c} is a quiet country with many rivers.",
           "Visitors say the food in {c} is simple and good.", "The roads of {c} are narrow but safe."]


def _name(rng, parts=(2, 3)) -> str:
    return "".join(rng.choice(_SYLL, rng.integers(parts[0], parts[1] + 1))).capitalize()


def make_world(n_countries: int = 48, n_unknown: int = 10, seed: int = RANDOM_STATE,
               repeats=(1, 2, 4, 8, 16), filler_per_country: int = 6):
    """An invented world. Each known country has a capital, a language and a main export. Facts are repeated a
    different number of times in the corpus (1 to 16), so some are well learned and some are rare.
    'Unknown' countries appear only in filler sentences, never with a fact: asking for their capital forces a
    model to say something it was never told.
    Returns (corpus_sentences, facts) where facts has: country, relation, value, repeats, known."""
    rng = np.random.default_rng(seed)
    names = []
    while len(names) < n_countries + n_unknown + n_countries:
        nm = _name(rng)
        if nm not in names and 4 <= len(nm) <= 10:
            names.append(nm)
    countries, unknown, capitals = names[:n_countries], names[n_countries:n_countries + n_unknown], names[n_countries + n_unknown:]
    facts, sents = [], []
    for i, c in enumerate(countries):
        rep = repeats[i % len(repeats)]
        values = {"capital": capitals[i], "language": rng.choice(LANGUAGES), "export": rng.choice(EXPORTS)}
        for rel, v in values.items():
            facts.append({"country": c, "relation": rel, "value": v, "repeats": rep, "known": True})
            for _ in range(rep):
                sents.append(rng.choice(FACT_TEMPLATES[rel]).format(c=c, v=v))
    for c in countries + unknown:
        for _ in range(filler_per_country):
            sents.append(rng.choice(_FILLER).format(c=c))
    for c in unknown:
        for rel in FACT_TEMPLATES:
            facts.append({"country": c, "relation": rel, "value": None, "repeats": 0, "known": False})
    rng.shuffle(sents)
    return sents, pd.DataFrame(facts)


def make_icl_episodes(n: int = 2000, n_pairs: int = 6, seed: int = RANDOM_STATE, keys=None, values=None):
    """In-context learning episodes with a FRESH random mapping each time, e.g.
         'map: tup 7 kel 3 mor 1 | kel 3'      (prompt 'map: tup 7 kel 3 mor 1 | kel', answer ' 3')
    The answer to the last query can only come from the examples in the same line (the context), never from
    memory. Returns a list of (prompt_without_answer, answer)."""
    rng = np.random.default_rng(seed)
    keys = keys or ["tup", "kel", "mor", "dap", "sil", "ruv", "fen", "goz", "hib", "jat", "lom", "nus"]
    values = values or list("0123456789")
    out = []
    for _ in range(n):
        ks = rng.choice(keys, n_pairs, replace=False)
        vs = rng.choice(values, n_pairs)
        q = rng.integers(n_pairs)
        ctx = " ".join(f"{k} {v}" for k, v in zip(ks, vs))
        out.append((f"map: {ctx} | {ks[q]}", f" {vs[q]}"))
    return out
