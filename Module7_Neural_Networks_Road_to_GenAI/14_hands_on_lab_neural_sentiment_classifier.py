"""
LAB 14 - HANDS-ON LAB: Build a Neural Sentiment Classifier
==========================================================

THE BRIEF
    An online shop wants every product review tagged positive / negative, so that negative reviews reach the
    support team the same day. Reviews are short but tricky:
        "the battery is not great"                         negation flips the meaning
        "the screen is excellent but the delivery is awful" contrast: the part after "but" decides
    and many end with a phrase like "five stars" or "returning it", which matches the label 80% of the time:
    a SHORTCUT a model can lean on, until a review breaks the pattern.

WHAT YOU BUILD (the syllabus hands-on lab: "build a small neural-network classifier, or use a pretrained
transformer for sentiment classification")
    Part 1  EASY    The data, a split, and two classical baselines (majority, bag-of-words logistic regression)
    Part 2  EASY    A small neural network on bag-of-words features: our NumPy MLP from Labs 02-06
    Part 3  MEDIUM  A tiny transformer trained from scratch (PyTorch) that reads word ORDER
    Part 4  MEDIUM  Error analysis by review type, and a shortcut test (swap the endings)
    Part 5  HARD    A pretrained transformer (if `transformers` is installed), a scorecard, and a memo
"""
# %%
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml_lab_utils import HAS_MPL, OUT_DIR, banner, make_reviews, note, optional_import, plt, save_fig, show, step
from nn_toolkit import HAS_TORCH, MLP, Adam, ModelScorecard, Requirement

reviews = make_reviews(n=6000)
rng = np.random.default_rng(0)
split = rng.choice(["train", "val", "test"], len(reviews), p=[0.6, 0.2, 0.2])
tr, va, te = (reviews[split == s].reset_index(drop=True) for s in ["train", "val", "test"])
results, predictions = {}, {}


def swap_endings(texts):
    """Counterfactual test: replace each 'shortcut' ending with its opposite. The true sentiment is unchanged."""
    sw = {"would buy again": "returning it", "returning it": "would buy again", "five stars": "one star", "one star": "five stars"}
    return [re.sub("|".join(sw), lambda m: sw[m.group(0)], t) for t in texts]


has_end = te.text.str.contains("would buy again|returning it|five stars|one star")


def evaluate(name, predict, n_params, note_=""):
    """predict(list_of_texts) -> 0/1 array. Records accuracy overall / by kind, and on swapped endings."""
    t0 = time.perf_counter()
    p = np.asarray(predict(list(te.text)))
    ms = (time.perf_counter() - t0) / len(te) * 1000
    p_sw = np.asarray(predict(swap_endings(te.text[has_end])))
    true = te._true_label.to_numpy()
    res = {"test accuracy": float(np.mean(p == true)), "ms_per_review": ms, "parameters": n_params,
           "accuracy, swapped endings": float(np.mean(p_sw == true[has_end])),
           "accuracy, original endings": float(np.mean(p[has_end] == true[has_end]))}
    for kind in ["plain", "intensified", "negated", "contrast", "mixed"]:
        res[f"acc: {kind}"] = float(np.mean((p == true)[te.kind == kind]))
    res["shortcut drop"] = res["accuracy, original endings"] - res["accuracy, swapped endings"]
    results[name], predictions[name] = res, p
    print(f"  {name:<38} test acc {res['test accuracy']:.3f}  negated {res['acc: negated']:.3f}  contrast "
          f"{res['acc: contrast']:.3f}  swapped endings {res['accuracy, swapped endings']:.3f}  ({note_})")


# %% =========================================================================
# PART 1 (EASY): Data and classical baselines
# ============================================================================
banner("Part 1 - The reviews, and two baselines", "EASY")
print(reviews.drop(columns="_true_label").sample(6, random_state=3).to_string(index=False))
show("Reviews: train / validation / test", f"{len(tr):,} / {len(va):,} / {len(te):,}")
show("Positive share", f"{reviews.label.mean():.1%}")
print(reviews.kind.value_counts().to_string())
note("""Labels are 2% noisy (annotators make mistakes), so we score against the
hidden TRUE label (_true_label): the best possible accuracy on noisy labels
would be ~98%. In real projects you only have the noisy labels, which is
one more reason to look at errors by hand.""")

evaluate("majority class", lambda X: np.full(len(X), int(tr.label.mean() > 0.5)), 0, "always the most common label")
bow = CountVectorizer(token_pattern=r"[a-z']+").fit(tr.text)
lr_bow = LogisticRegression(max_iter=2000, C=1.0).fit(bow.transform(tr.text), tr.label)
evaluate("bag-of-words logistic regression", lambda X: lr_bow.predict(bow.transform(X)),
         lr_bow.coef_.size + 1, f"{len(bow.vocabulary_)} word counts")
bi = TfidfVectorizer(token_pattern=r"[a-z']+", ngram_range=(1, 2), sublinear_tf=True).fit(tr.text)
lr_bi = LogisticRegression(max_iter=3000, C=5.0).fit(bi.transform(tr.text), tr.label)
evaluate("word + bigram TF-IDF logistic", lambda X: lr_bi.predict(bi.transform(X)), lr_bi.coef_.size + 1,
         f"{len(bi.vocabulary_)} unigrams + bigrams")
note("""Bag-of-words gets most plain reviews right but cannot see that 'not' belongs to
'great'. Bigrams ('not great', 'but the') are the classical feature-engineering
patch (Module 2): better, but 'not great at all' and long contrasts still slip
through.""")

# %% =========================================================================
# PART 2 (EASY): A small neural network on bag-of-words
# ============================================================================
banner("Part 2 - A small neural network (NumPy MLP from Labs 02-06)", "EASY")
Xb_tr = bow.transform(tr.text).toarray().astype(float)
Xb_va = bow.transform(va.text).toarray().astype(float)
mlp = MLP([Xb_tr.shape[1], 64, 32, 1], activation="relu", loss="bce", dropout=0.2, seed=0)
mlp.fit(Xb_tr, tr.label.to_numpy(), epochs=30, batch_size=64, optimizer=Adam(0.002), X_val=Xb_va, y_val=va.label.to_numpy())
hist = pd.DataFrame(mlp.history_).set_index("epoch")
best = int(hist.val_loss.idxmin())
mlp = MLP([Xb_tr.shape[1], 64, 32, 1], activation="relu", loss="bce", dropout=0.2, seed=0).fit(
    Xb_tr, tr.label.to_numpy(), epochs=best, batch_size=64, optimizer=Adam(0.002))
show("Early stopping: best validation epoch", best)
evaluate("MLP on bag-of-words (NumPy)", lambda X: mlp.predict(bow.transform(X).toarray().astype(float)), mlp.n_params,
         f"{mlp.n_params:,} parameters, {best} epochs")
note("""A neural network on the same word counts. Its hidden layers can combine words
('not' together with 'great' anywhere in the review), which logistic
regression cannot, so it handles some negations. But it still has no idea
WHICH word 'not' sits next to, or what comes after 'but': order is gone
before the network sees the review.""")

# %% =========================================================================
# PART 3 (MEDIUM): A tiny transformer that reads word order
# ============================================================================
banner("Part 3 - A tiny transformer trained from scratch", "MEDIUM")
if HAS_TORCH:
    import torch
    import torch.nn as nn

    from nn_toolkit import TransformerBlock, train_classifier

    torch.set_num_threads(4)
    words = sorted({w for t in tr.text for w in re.findall(r"[a-z']+|[.!,]", t)})
    vocab = {w: i + 2 for i, w in enumerate(words)}                  # 0 = padding, 1 = unknown word
    MAXLEN = 48

    def encode(texts):
        out = np.zeros((len(texts), MAXLEN), dtype=np.int64)
        for r, t in enumerate(texts):
            ids = [vocab.get(w, 1) for w in re.findall(r"[a-z']+|[.!,]", t)][:MAXLEN]
            out[r, :len(ids)] = ids
        return torch.tensor(out)

    class ReviewTransformer(nn.Module):
        """word embeddings + positions -> 2 encoder blocks -> average over real words -> positive / negative"""

        def __init__(self, n_vocab, d=64, heads=4, layers=2):
            super().__init__()
            self.tok, self.pos = nn.Embedding(n_vocab, d), nn.Embedding(MAXLEN, d)
            self.blocks = nn.ModuleList([TransformerBlock(d, heads, 0.1, causal=False) for _ in range(layers)])
            self.ln, self.out = nn.LayerNorm(d), nn.Linear(d, 2)

        def forward(self, x):
            pad = x == 0
            h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
            for blk in self.blocks:
                h = blk(h)
            h = self.ln(h).masked_fill(pad[..., None], 0)
            return self.out(h.sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1))

    torch.manual_seed(0)
    tfm = ReviewTransformer(len(vocab) + 2)
    hist_t = train_classifier(tfm, encode(tr.text), tr.label.to_numpy(), encode(va.text), va.label.to_numpy(), epochs=40, lr=2e-3)

    @torch.no_grad()
    def tfm_predict(texts):
        tfm.eval()
        return tfm(encode(texts)).argmax(-1).numpy()

    n_tfm = sum(p.numel() for p in tfm.parameters())
    show("Validation accuracy by epoch (1, 5, 10, 20, 40)", [round(hist_t[i]["test_acc"], 3) for i in [0, 4, 9, 19, 39]])
    evaluate("tiny transformer (from scratch)", tfm_predict, n_tfm, f"{n_tfm:,} parameters, 2 blocks")
    note("""The transformer reads the words IN ORDER: attention can link 'not' to the word
right after it, and the position embeddings tell it which clause comes after
'but'. It's trained from zero on 3,600 reviews in under a minute on a CPU.""")
else:
    print("  PyTorch not installed: skipping the transformer (pip install torch).")

# %% =========================================================================
# PART 4 (MEDIUM): Error analysis and the shortcut test
# ============================================================================
banner("Part 4 - Where do the models fail? Do they take the shortcut?", "MEDIUM")
cols = ["test accuracy", "acc: plain", "acc: intensified", "acc: negated", "acc: contrast", "acc: mixed", "accuracy, original endings",
        "accuracy, swapped endings", "shortcut drop"]
board = pd.DataFrame(results).T
print(board[cols].round(3).to_string())
note("""Accuracy by TYPE of review tells the real story: everyone does well on plain
reviews; the gap is in negation and contrast. The shortcut test swaps every
'five stars' for 'one star' (and so on) without changing the actual review:
a model that relied on the ending loses accuracy on exactly those reviews.""")
if "tiny transformer (from scratch)" in predictions:
    p = predictions["tiny transformer (from scratch)"]
    wrong = te[p != te._true_label].assign(predicted=p[p != te._true_label])
    print("\n  Some reviews the transformer still gets wrong:")
    print(wrong[["text", "kind", "_true_label", "predicted"]].head(6).to_string(index=False))

# %% =========================================================================
# PART 5 (HARD): A pretrained transformer, a scorecard and a memo
# ============================================================================
banner("Part 5 - Pretrained transformer, scorecard, memo", "HARD")
transformers = optional_import("transformers")
if transformers is not None:
    try:
        clf = transformers.pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        pre = lambda X: np.array([int(r["label"] == "POSITIVE") for r in clf(list(X), batch_size=64, truncation=True)])
        evaluate("pretrained DistilBERT (SST-2, zero training)", pre, 66_955_010, "downloaded, no training on our data")
    except Exception as e:                                           # no internet, no model cache, ...
        print(f"  Could not load the pretrained model ({type(e).__name__}); skipping.")
else:
    note("""`transformers` is not installed, so the pretrained option is skipped. With
`pip install transformers` this part downloads DistilBERT fine-tuned on movie
reviews (67M parameters) and scores it on our test set with no training at
all: the 'use a pretrained transformer' route. Expect strong negation handling
from its pretraining, but no knowledge of our shop's 'returning it' phrasing.""")

REQUIREMENTS = [
    Requirement("overall accuracy", "test accuracy", 0.90, higher_is_better=True, hard=True,
                why="below 90% the support team stops trusting the tags"),
    Requirement("negations", "acc: negated", 0.85, higher_is_better=True, hard=True,
                why="'not working' is the review support most needs to see"),
    Requirement("contrast reviews", "acc: contrast", higher_is_better=True, weight=0.35,
                why="'great screen but awful support' is a complaint"),
    Requirement("robust to endings", "shortcut drop", weight=0.25, why="customers don't always end with 'one star'"),
    Requirement("speed", "ms_per_review", weight=0.20, why="tagging runs on every new review"),
    Requirement("size", "parameters", weight=0.20, why="cheaper to host, retrain and explain"),
]
card = ModelScorecard("Review sentiment: which classifier should tag incoming reviews?", REQUIREMENTS)
for name, res in results.items():
    if name != "majority class":
        card.add(name, res)
table = card.evaluate()
print(table.drop(columns=["failed gates"]).round(3).to_string())
winner = card.recommend()
show("Recommended", winner)
if winner:
    w = results[winner]
    card.to_markdown(OUT_DIR / "14_sentiment_classifier_memo.md",
                     header=f"{len(tr):,} training / {len(va):,} validation / {len(te):,} test reviews; scored against the "
                            "hidden true labels; 'swapped endings' replaces each shortcut ending with its opposite.",
                     justification=f"**{winner}** passes both gates (accuracy {w['test accuracy']:.1%}, negations "
                                   f"{w['acc: negated']:.1%}) and scores best on contrast reviews ({w['acc: contrast']:.1%}) "
                                   f"with a shortcut drop of {w['shortcut drop']:+.1%}. Monitor accuracy by review type "
                                   "monthly on a fresh hand-labelled sample: new phrasing (sarcasm, emojis, other "
                                   "languages) is out of distribution for every model here.")
    print("  [saved] outputs/14_sentiment_classifier_memo.md")
note("""Gates first, then weights. Plain bag-of-words and the MLP on word counts fail the
negation gate. Two models pass: the bigram model (classical feature
engineering) and the tiny transformer. The transformer is clearly better on
contrast reviews and less dependent on the ending shortcut, but the brief
also weights speed and size, and there the bigram model is 10x faster and
~350x smaller. With THESE weights the scorecard picks the bigram model.""")

step("5.1  Sensitivity: what if contrast reviews matter more?")
heavy = ModelScorecard(card.problem, [r if r.weight == 0 else Requirement(
    r.name, r.metric, r.threshold, r.higher_is_better, r.hard, {"contrast reviews": 0.6, "robust to endings": 0.2,
                                                                 "speed": 0.1, "size": 0.1}[r.name], r.why) for r in REQUIREMENTS])
for name, res in card.results.items():
    heavy.add(name, res)
show("Weights 60% contrast / 20% endings / 10% speed / 10% size", heavy.recommend())
note("""Shift the weights towards what the support team said they care about most
('great screen but awful support' is a complaint) and the recommendation
flips to the transformer. Neither answer is wrong: the scorecard makes the
trade-off explicit, so the business decides it on purpose. A pretrained
transformer would bring broad language knowledge for free, at ~600x the
parameters of our tiny one, and would still need testing on your own reviews.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    board[["acc: plain", "acc: negated", "acc: contrast", "acc: mixed"]].drop(index="majority class").plot.barh(ax=axes[0])
    axes[0].set(xlim=(0.4, 1.0), title="Accuracy by review type")
    axes[0].legend(fontsize=8)
    board[["accuracy, original endings", "accuracy, swapped endings"]].drop(index="majority class").plot.barh(ax=axes[1])
    axes[1].set(xlim=(0.4, 1.0), title="Shortcut test: swap 'five stars' <-> 'one star'")
    axes[1].legend(fontsize=8)
    save_fig(fig, "14_hands_on_lab_sentiment")

banner("KEY TAKEAWAYS")
print("""  1. Start with baselines: bag-of-words logistic regression is strong on plain text.
  2. A small MLP on word counts combines words, but the order is gone before it starts.
  3. A tiny transformer that reads order wins on contrast reviews; good bigram features remain a strong rival.
  4. Evaluate by TYPE of example and with counterfactuals: aggregate accuracy hides shortcuts.
  5. Decide with gates and weights, and check how the choice moves with the weights.""")
