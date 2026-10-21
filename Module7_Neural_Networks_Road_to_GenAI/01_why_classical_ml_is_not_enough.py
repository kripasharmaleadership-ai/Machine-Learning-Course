"""
LAB 01 - Why Classical ML Is Not Sufficient for Every Problem
=============================================================

WHAT MODULES 3-6 GAVE US
    linear / logistic regression   a weighted sum of the features YOU provide
    trees and ensembles            if-then splits on the features YOU provide
    k-means, PCA, embeddings       geometry of the features YOU provide
    The common thread: a human decides what the columns are. When the right columns are easy to write down
    (income, tenure, number of late payments), classical ML is excellent, and often the best choice.

WHERE IT STRUGGLES
    the pattern is a curved shape in the raw inputs      spirals                              (Labs 02-06)
    the raw inputs are pixels and the object can move    a digit anywhere on a 16x16 canvas   (Lab 07)
    the meaning is in the ORDER of tokens                "A before B" vs "B before A"         (Labs 08-09)
    Neural networks learn the features themselves, layer by layer. That is the whole road to GenAI:
    an LLM is a neural network that learned its own features of text.

LEVELS
    EASY   : Two spirals. Logistic regression is stuck at a coin flip; a small neural network solves it.
    MEDIUM : Hand-made features rescue classical ML on spirals, but no one can hand-make features for pixels
             that move, or for word order.
    HARD   : The other side: on ordinary tabular data, gradient boosting matches or beats the network.
"""
# %%
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ml_lab_utils import HAS_MPL, banner, make_sequence_task, make_shifted_digits, make_spirals, note, plt, save_fig, show, step
from nn_toolkit import MLP, Adam

X, y = make_spirals(n=2000)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# %% =========================================================================
# LEVEL 1 (EASY): A shape no straight line can cut
# ============================================================================
banner("Two interleaved spirals: classical vs neural", "EASY")
note("""Two classes wound around each other. Each point has only two inputs: its x and
y coordinates. There is a perfect answer (the spiral it belongs to), but no
straight line separates them.""")
logit = LogisticRegression().fit(X_tr, y_tr)
show("Logistic regression on (x, y): test accuracy", logit.score(X_te, y_te))
net = MLP([2, 32, 32, 1], activation="tanh", loss="bce", seed=0).fit(X_tr, y_tr, epochs=300, batch_size=32, optimizer=Adam(0.01))
show("Neural network 2 -> 32 -> 32 -> 1: test accuracy", net.score(X_te, y_te))
show("Trainable parameters in that network", net.n_params)
note(f"""Logistic regression draws one straight line: {logit.score(X_te, y_te):.0%}, not far above a coin flip. The
network ({net.n_params:,} weights, built from scratch in NumPy in nn_toolkit.py) reaches
{net.score(X_te, y_te):.0%}. Nobody told it about spirals: its hidden layers BUILT the curved
features it needed. Labs 02-06 open that box one piece at a time.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Feature engineering works... until it can't
# ============================================================================
banner("Hand-made features: the classical answer, and its limits", "MEDIUM")
step("2.1  Spirals: give logistic regression better columns")
r, theta = np.hypot(*X.T), np.arctan2(X[:, 1], X[:, 0])
eng = np.column_stack([np.sin(theta - 3.66 * r), np.cos(theta - 3.66 * r)])   # the spiral's own equation
eng_tr, eng_te = train_test_split(eng, test_size=0.3, random_state=0, stratify=y)
rows = {
    "raw (x, y)": LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te),
    "polynomial features, degree 5": make_pipeline(PolynomialFeatures(5), StandardScaler(),
                                                   LogisticRegression(max_iter=3000)).fit(X_tr, y_tr).score(X_te, y_te),
    "polynomial features, degree 11": make_pipeline(PolynomialFeatures(11), StandardScaler(),
                                                    LogisticRegression(max_iter=5000)).fit(X_tr, y_tr).score(X_te, y_te),
    "features from the spiral equation": LogisticRegression().fit(eng_tr, y_tr).score(eng_te, y_te),
    "neural network on raw (x, y)": net.score(X_te, y_te),
}
print(pd.Series(rows, name="test accuracy").round(3).to_string())
note(f"""Generic polynomial features help: degree 11 (78 columns of x^a * y^b) reaches
{rows['polynomial features, degree 11']:.0%}. Two features written from the spiral's own equation (angle minus
radius) make logistic regression perfect, but you had to KNOW the equation.
Feature engineering (Module 2) is powerful when a human understands the
pattern. The network found an equivalent representation on its own.""")

step("2.2  Pixels that move: a digit anywhere on a 16 x 16 canvas")
rows = {}
for shift in ["center", "random"]:
    imgs, labels, _ = make_shifted_digits(shift=shift)
    a_tr, a_te, b_tr, b_te = train_test_split(imgs.reshape(len(imgs), -1), labels, test_size=0.3, random_state=0, stratify=labels)
    rows["digits centred" if shift == "center" else "digits placed at random"] = {
        "logistic regression": LogisticRegression(max_iter=3000).fit(a_tr, b_tr).score(a_te, b_te),
        "gradient boosting": HistGradientBoostingClassifier(random_state=0).fit(a_tr, b_tr).score(a_te, b_te)}
pix = pd.DataFrame(rows).T
print(pix.round(3).to_string())
note(f"""Centred digits: pixel 37 always means 'the top of the loop', and classical
models do well ({pix.iloc[0, 0]:.0%}). Move the digit around and the same pixel means
something different in every image: logistic regression falls to {pix.iloc[1, 0]:.0%}. A human
can't hand-write 'a loop, wherever it is' as a column. The convolutional network
in Lab 07 learns it.""")

step("2.3  Order: does marker A come before marker B?")
S, s_y = make_sequence_task("order", n=4000, length=20)
counts = np.stack([np.bincount(row, minlength=10) for row in S])                # bag of tokens: counts, no order
c_tr, c_te, sy_tr, sy_te = train_test_split(counts, s_y, test_size=0.3, random_state=0)
p_tr, p_te = train_test_split(np.column_stack([np.argmax(S == 0, 1), np.argmax(S == 1, 1)]), test_size=0.3, random_state=0)
show("Gradient boosting on token COUNTS", HistGradientBoostingClassifier(random_state=0).fit(c_tr, sy_tr).score(c_te, sy_te))
show("Gradient boosting on hand-made POSITIONS of A, B", HistGradientBoostingClassifier(random_state=0).fit(p_tr, sy_tr).score(p_te, sy_te))
note("""Both classes contain exactly one A and one B, so counts are identical and any
bag-of-words model sits at 50%. Telling the model 'here are the positions of
A and B' solves it, but only because we knew which tokens matter. In language
the important tokens and relations change with every sentence ("not", "but",
who did what to whom). Sequence models (Lab 08) and attention (Lab 09) learn
which positions to look at.""")

# %% =========================================================================
# LEVEL 3 (HARD): When classical ML is the better choice
# ============================================================================
banner("Plot twist: on tabular data, the network is not the winner", "HARD")
rng = np.random.default_rng(0)
n = 6000
tab = pd.DataFrame({"income_k": rng.lognormal(3.8, 0.5, n), "tenure_m": rng.integers(1, 120, n),
                    "late_payments": rng.poisson(0.6, n), "utilisation": rng.uniform(0, 1, n),
                    "has_mortgage": rng.integers(0, 2, n), "age": rng.integers(21, 75, n)})
risk = (1.2 * tab.late_payments + 2.5 * (tab.utilisation > 0.8) - 0.012 * tab.tenure_m
        - 0.8 * (tab.income_k > 60) + 0.6 * ((tab.age < 25) & (tab.has_mortgage == 0)) - 1.0)
tab["default"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
Xt = tab.drop(columns="default").to_numpy(float)
t_tr, t_te, ty_tr, ty_te = train_test_split(Xt, tab.default.to_numpy(), test_size=0.3, random_state=0, stratify=tab.default)
sc = StandardScaler().fit(t_tr)
rows = {}
for name, fit in {
    "logistic regression": lambda: LogisticRegression(max_iter=2000).fit(sc.transform(t_tr), ty_tr),
    "gradient boosting (defaults)": lambda: HistGradientBoostingClassifier(random_state=0).fit(t_tr, ty_tr),
    "neural network 6 -> 64 -> 64 -> 1": lambda: MLP([6, 64, 64, 1], loss="bce", seed=0).fit(
        sc.transform(t_tr), ty_tr, epochs=60, batch_size=64, optimizer=Adam(0.003)),
}.items():
    t0 = time.perf_counter()
    m = fit()
    secs = time.perf_counter() - t0
    Xe = t_te if "boosting" in name else sc.transform(t_te)
    rows[name] = {"test accuracy": m.score(Xe, ty_te), "train seconds": secs}
tabres = pd.DataFrame(rows).T
print(tabres.round(3).to_string())
show("Baseline: always predict 'no default'", 1 - ty_te.mean())
note(f"""A loan table with six meaningful columns and thresholds ('utilisation above
80%'). Gradient boosting reaches {tabres.iloc[1, 0]:.1%}; the network {tabres.iloc[2, 0]:.1%}, after scaling,
choosing an architecture and a learning rate. On tabular data where the
columns already mean something, trees are usually as good or better, faster,
and easier to explain (Module 5). Neural networks earn their keep when the
raw input is pixels, audio or text: the features are what's hard.""")

step("3.1  Neural networks are hungry: spirals with fewer examples")
rows = {}
for m in [50, 200, 1400]:
    sub = np.random.default_rng(1).choice(len(X_tr), m, replace=False)
    nn = MLP([2, 32, 32, 1], activation="tanh", loss="bce", seed=0).fit(X_tr[sub], y_tr[sub], epochs=max(300, 60000 // m),
                                                                      batch_size=32, optimizer=Adam(0.01))
    rows[m] = {"neural network": nn.score(X_te, y_te),
               "polynomial logistic (deg 5)": make_pipeline(PolynomialFeatures(5), StandardScaler(), LogisticRegression(
                   max_iter=3000)).fit(X_tr[sub], y_tr[sub]).score(X_te, y_te)}
hungry = pd.DataFrame(rows).T
hungry.index.name = "training points"
print(hungry.round(3).to_string())
note(f"""The fixed-feature model improves slowly: degree-5 polynomials can only draw
so much spiral ({hungry.iloc[0, 1]:.0%} with 50 points, {hungry.iloc[-1, 1]:.0%} with 1,400). The network turns data
into accuracy ({hungry.iloc[0, 0]:.0%} with 50 points, {hungry.iloc[1, 0]:.0%} with 200, {hungry.iloc[-1, 0]:.0%} with 1,400). Flexibility
pays off only with data: a model that can learn any shape needs examples to
know which shape is right. LLMs push this to the extreme: billions of
weights, trained on trillions of tokens.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    gx, gy = np.meshgrid(np.linspace(-3.4, 3.4, 250), np.linspace(-3.4, 3.4, 250))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    for ax, (title, m) in zip(axes[:2], [("Logistic regression: one straight line", logit), ("Neural network: learned curves", net)]):
        p = m.predict_proba(grid)
        p = p[:, 1] if p.ndim == 2 else p
        ax.contourf(gx, gy, p.reshape(gx.shape), levels=20, cmap="RdBu", alpha=0.6)
        ax.scatter(*X_te.T, c=y_te, cmap="RdBu", s=6, edgecolor="k", linewidth=0.2)
        ax.set(title=f"{title} ({m.score(X_te, y_te):.0%})", xticks=[], yticks=[])
    imgs, labels, _ = make_shifted_digits(shift="random")
    axes[2].imshow(np.hstack([imgs[i] for i in range(6)]), cmap="gray_r")
    axes[2].set(title="Lab 07 preview: the same digits, anywhere on the canvas", xticks=[], yticks=[])
    save_fig(fig, "01_why_classical_ml_is_not_enough")

banner("KEY TAKEAWAYS")
print("""  1. Classical ML learns a rule over features a human chose; that's its strength and its limit.
  2. Neural networks learn the features too: curves from coordinates, shapes from pixels, meaning from order.
  3. Feature engineering still works when you understand the pattern (the spiral equation solved it).
  4. On tabular data with meaningful columns, gradient boosting is usually the better first choice.
  5. Flexible models turn more data into more accuracy; fixed features plateau. LLMs scale that to trillions of tokens.""")
