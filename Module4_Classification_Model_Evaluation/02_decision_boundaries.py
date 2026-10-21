"""
LAB 02 - Decision Boundaries
============================

THE IDEA
    A classifier divides the feature space into REGIONS, one per class. The border between
    regions is the DECISION BOUNDARY: the set of points where the model is exactly undecided
    (P(class 1) = threshold, 0.5 by default).

    logistic regression : p = sigmoid(w1 x1 + w2 x2 + b)
                          boundary at p = 0.5  <=>  w1 x1 + w2 x2 + b = 0   -> a STRAIGHT LINE
    k-nearest neighbours: vote of the k closest training points  -> wiggly, local
    decision tree       : a sequence of "x_j <= t" questions      -> axis-aligned STAIRCASE
    random forest       : average of many trees                   -> smoother staircase
    gradient boosting   : sum of many small trees                 -> flexible, still axis-aligned

    Each algorithm has a built-in SHAPE PREFERENCE (its inductive bias). Picking an algorithm
    = picking which shapes of boundary are easy for it to learn.

LEVELS
    EASY   : The logistic-regression boundary as an equation; how the threshold moves it.
    MEDIUM : Five algorithms on the same two datasets: five different shapes of boundary.
    HARD   : Feature engineering bends a linear boundary; inductive bias (diagonal vs staircase);
             what each model predicts FAR from the training data.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_circles, make_moons
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_lab_utils import (HAS_MPL, NEG_COLOR, POS_COLOR, RANDOM_STATE, banner, make_loan_applicants, note, plt,
                          save_fig, show, step)


def boundary_plot(ax, model, X, y, title, levels=(0.5,), extent=None):
    """Colour the plane by P(class 1) and draw the boundary line(s)."""
    (x0, x1), (y0, y1) = extent or ((X[:, 0].min() - .5, X[:, 0].max() + .5), (X[:, 1].min() - .5, X[:, 1].max() + .5))
    gx, gy = np.meshgrid(np.linspace(x0, x1, 250), np.linspace(y0, y1, 250))
    pz = model.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
    ax.contourf(gx, gy, pz, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
    ax.contour(gx, gy, pz, levels=list(levels), colors="k", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=np.where(y == 1, POS_COLOR, NEG_COLOR), s=8, edgecolor="none")
    ax.set_title(title, fontsize=9)


# %% =========================================================================
# LEVEL 1 (EASY): The boundary of logistic regression is a line
# ============================================================================
banner("Logistic regression on two features: the boundary is a line", "EASY")
loans = make_loan_applicants(n=3000, seed=5)
F = ["debt_to_income", "credit_years"]
X2, y2 = loans[F].to_numpy(), loans["default"].to_numpy()
lr = LogisticRegression().fit(X2, y2)
(w1, w2), b = lr.coef_[0], lr.intercept_[0]
show("Learned log-odds", f"{b:+.3f} {w1:+.3f}*DTI {w2:+.4f}*credit_years")
show("Boundary (p = 0.5): w1*x1 + w2*x2 + b = 0", f"credit_years = {-b / w2:.2f} + {-w1 / w2:.2f} * DTI")
note("""p = 0.5 exactly when the log-odds are 0. Solving w1*x1 + w2*x2 + b = 0 for
x2 gives a straight line. Above / below the line the model says 'repay' /
'default'. Every applicant ON the line gets exactly 50%.""")

step("1.1  Moving the threshold moves the line (parallel shift)")
rows = {}
for t in [0.5, 0.3, 0.2, 0.1]:
    logit_t = np.log(t / (1 - t))
    pred = (lr.predict_proba(X2)[:, 1] >= t)
    rows[f"threshold {t}"] = {"log-odds cut": logit_t, "boundary intercept (yrs at DTI=0)": (logit_t - b) / w2,
                              "share flagged": pred.mean(), "defaulters caught": (pred & (y2 == 1)).sum() / y2.sum()}
print(pd.DataFrame(rows).T.round(3).to_string())
note("""P(default) >= t  <=>  log-odds >= log(t / (1 - t)). Lowering the threshold
keeps the SLOPE of the line and slides it, so more applicants fall on the
'default' side. The model is unchanged; only the decision changes. With a
0.5 cut the bank flags almost nobody; with 0.2 it catches far more defaulters,
at the price of more false alarms (Labs 03 and 06).""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Five algorithms, five shapes
# ============================================================================
banner("Five algorithms on the same data", "MEDIUM")
Xm, ym = make_moons(n_samples=600, noise=0.28, random_state=RANDOM_STATE)
Xc, yc = make_circles(n_samples=600, noise=0.12, factor=0.45, random_state=RANDOM_STATE)
MODELS = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression()),
    "KNN k=1": make_pipeline(StandardScaler(), KNeighborsClassifier(1)),
    "KNN k=25": make_pipeline(StandardScaler(), KNeighborsClassifier(25)),
    "Decision tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=0),
    "Random forest (300 trees)": RandomForestClassifier(300, min_samples_leaf=3, random_state=0),
    "Gradient boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0),
}
rows, fitted = {}, {}
for dname, (X, y) in {"moons": (Xm, ym), "circles": (Xc, yc)}.items():
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.4, random_state=RANDOM_STATE)
    for name, m in MODELS.items():
        f = clone(m).fit(Xa, ya)
        fitted[(dname, name)] = (f, Xa, ya)
        rows.setdefault(name, {})[f"{dname} train"] = f.score(Xa, ya)
        rows[name][f"{dname} test"] = f.score(Xb, yb)
print(pd.DataFrame(rows).T.round(3).to_string())
note("""Same data, very different results. The straight line of logistic regression
can't follow a moon or enclose a circle (circles: test accuracy near a coin
flip). KNN with k=1 memorises every point (train accuracy 1.0) and its
boundary is jagged: overfitting, drawn as a picture. k=25 is smoother and
generalises better. The tree carves axis-aligned boxes; the forest and
boosting average many of them into a smoother shape.""")

if HAS_MPL:
    fig, axes = plt.subplots(2, 6, figsize=(22, 7.5))
    for r, dname in enumerate(["moons", "circles"]):
        for c, name in enumerate(MODELS):
            f, Xa, ya = fitted[(dname, name)]
            boundary_plot(axes[r, c], f, Xa, ya, f"{name}\n{dname}: test acc {rows[name][f'{dname} test']:.2f}")
    save_fig(fig, "02_medium_five_boundaries")

# %% =========================================================================
# LEVEL 3 (HARD): Bending lines, shape bias, and far-away points
# ============================================================================
banner("What shapes can a model learn, and what happens off the map?", "HARD")

step("3.1  Feature engineering bends a linear boundary")
radius_sq = FunctionTransformer(lambda X: np.c_[X, X[:, 0] ** 2 + X[:, 1] ** 2])
lin_circle = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xc, yc)
lin_circle_fe = make_pipeline(radius_sq, StandardScaler(), LogisticRegression()).fit(Xc, yc)
show("Logistic regression on (x1, x2)", f"accuracy {lin_circle.score(Xc, yc):.3f}")
show("Logistic regression on (x1, x2, x1^2 + x2^2)", f"accuracy {lin_circle_fe.score(Xc, yc):.3f}")
note("""Add one feature, the squared distance from the centre, and the SAME linear
model separates the circles almost perfectly. The boundary is a straight line
(a plane) in the new 3-D feature space, and a CIRCLE back in the original
space. 'Linear' models are linear in their FEATURES, not in the raw inputs.
This is what kernels, neural-network hidden layers and embeddings automate.""")

step("3.2  Inductive bias: a diagonal boundary vs an axis-aligned one")
rng = np.random.default_rng(1)
P = rng.uniform(-1, 1, (2000, 2))
diag_y = (P[:, 0] + P[:, 1] > 0).astype(int)                     # boundary: the diagonal x1 = -x2
rot = np.c_[(P[:, 0] + P[:, 1]) / np.sqrt(2), (P[:, 1] - P[:, 0]) / np.sqrt(2)]   # same data, rotated 45 deg
axis_y = (rot[:, 0] > 0).astype(int)
Pa, Pb, da, db, ra, rb, aa, ab = train_test_split(P, diag_y, rot, axis_y, test_size=0.5, random_state=0)
rows = {}
for depth in [1, 2, 4, 8, None]:
    t_diag = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Pa, da)
    t_axis = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(ra, aa)
    rows[f"tree depth {depth}"] = {"diagonal boundary: test acc": t_diag.score(Pb, db),
                                   "diagonal: leaves": t_diag.get_n_leaves(),
                                   "rotated (axis-aligned): test acc": t_axis.score(rb, ab),
                                   "rotated: leaves": t_axis.get_n_leaves()}
lr_diag = LogisticRegression().fit(Pa, da)
rows["logistic regression"] = {"diagonal boundary: test acc": lr_diag.score(Pb, db), "diagonal: leaves": np.nan,
                               "rotated (axis-aligned): test acc": LogisticRegression().fit(ra, aa).score(rb, ab),
                               "rotated: leaves": np.nan}
print(pd.DataFrame(rows).T.round(3).to_string())
note("""The SAME problem, just rotated by 45 degrees. When the true boundary is
axis-aligned, a depth-1 tree (one question) is perfect. When it is diagonal,
the tree needs a long staircase of boxes and is still not exact; logistic
regression gets it with 3 parameters. Each algorithm has a shape it finds easy.
Domain features that ALIGN the boundary with an axis (a ratio like debt /
income, a difference like price - average price) help trees enormously.""")

step("3.3  What does each model say FAR from the training data?")
far = np.array([[0.5, 0.25], [3, 3], [10, -10], [50, 50]])
rows = {}
for name in ["Logistic regression", "KNN k=25", "Decision tree (depth 5)", "Random forest (300 trees)",
             "Gradient boosting"]:
    f = fitted[("moons", name)][0]
    rows[name] = dict(zip(["(0.5, 0.25) in the data", "(3, 3)", "(10, -10)", "(50, 50)"], f.predict_proba(far)[:, 1]))
far_tab = pd.DataFrame(rows).T
print(far_tab.round(3).to_string())
note(f"""At (50, 50) the five models give P(class 1) from {far_tab['(50, 50)'].min():.2f} to {far_tab['(50, 50)'].max():.2f}:
total disagreement, each one 'confident'. The moons live in roughly
[-1.5, 2.5] x [-1, 1.5]. Far outside, logistic
regression becomes MORE and more certain (the sigmoid saturates at 0 or 1),
trees and forests repeat whatever their outermost box says, and KNN copies its
nearest neighbours however far away they are. None of them says 'I have never
seen anything like this'. Out-of-range inputs need an explicit check (data
validation, distance to the training data), not more trust in the probability.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    loan_ext = ((0, 0.9), (0, 40))
    boundary_plot(axes[0], lr, X2, y2, "Loans: logistic boundary at 0.5 / 0.3 / 0.2 / 0.1",
                  levels=(0.1, 0.2, 0.3, 0.5), extent=loan_ext)
    axes[0].set(xlabel="debt-to-income", ylabel="credit history (years)", ylim=(0, 40))
    boundary_plot(axes[1], lin_circle, Xc, yc, f"Circles, linear: acc {lin_circle.score(Xc, yc):.2f}")
    boundary_plot(axes[2], lin_circle_fe, Xc, yc, f"Circles, + r^2 feature: acc {lin_circle_fe.score(Xc, yc):.2f}")
    t8 = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Pa, da)
    boundary_plot(axes[3], t8, Pa, da, f"Diagonal truth, tree depth 4: {t8.get_n_leaves()} leaves")
    axes[3].plot([-1, 1], [1, -1], "g--", lw=2)
    save_fig(fig, "02_hard_shapes_and_thresholds")

banner("KEY TAKEAWAYS")
print("""  1. The decision boundary is where the model is undecided: P(class) = threshold.
  2. Logistic regression draws a straight line (hyperplane); the threshold slides it.
  3. KNN is local and wiggly, trees are axis-aligned staircases, ensembles smooth them.
  4. A linear model on engineered features draws curved boundaries in the raw space.
  5. Far from the training data, every model still answers with confidence: validate inputs.""")
