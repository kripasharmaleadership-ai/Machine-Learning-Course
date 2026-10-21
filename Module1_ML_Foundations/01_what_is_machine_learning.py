"""
LAB 01 - What is Machine Learning?
==================================

ONE-LINE DEFINITION
    Machine learning = a computer LEARNS A RULE FROM EXAMPLES instead of a
    human writing the rule by hand.

TOM MITCHELL'S DEFINITION (1997, still the best one)
    "A program learns from experience E with respect to task T and performance
     measure P, if its performance on T, as measured by P, improves with E."
        E = Experience  -> the data (examples)
        T = Task        -> what we want to do (predict delivery time)
        P = Performance -> how we score it (average error in minutes)

LEVELS IN THIS LAB
    EASY   : Pizza delivery time. Human rule vs. learned rule.
    MEDIUM : Open the black box. "Learning" = gradient descent, written from scratch.
    HARD   : Proof that performance improves with experience (learning curve)
             + memorization vs generalization (the real goal of ML).

HOW TO RUN
    python 01_what_is_machine_learning.py
    (or open in VS Code and run cell by cell with "Run Cell" on each # %% line)
"""
# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_lab_utils import HAS_MPL, RANDOM_STATE, banner, note, plt, save_fig, show, step

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): Pizza delivery - human rule vs learned rule
# ============================================================================
banner("Pizza delivery time: human rule vs machine-learned rule", "EASY")

# Step 1: Experience (E) - 200 past deliveries.
# Hidden truth that NOBODY in the company knows: time = 8 min prep + 2.7 min/km + noise
step("Step 1: Collect experience (historical deliveries)")
distance_km = rng.uniform(0.5, 12, 200).round(1)
true_minutes = 8 + 2.7 * distance_km + rng.normal(0, 3, 200)
deliveries = pd.DataFrame({"distance_km": distance_km, "minutes": true_minutes.round(1)})
print(deliveries.head())
note("Each row is ONE past delivery. Distance is the input, minutes is the answer.")

# Step 2: Traditional programming - a manager writes the rule from gut feeling.
step("Step 2: Traditional approach - a human writes the rule")


def manager_rule(km: float) -> float:
    """'Roughly 3 minutes per km plus 5 minutes' - a guess from experience."""
    return 5 + 3 * km


manager_pred = deliveries["distance_km"].apply(manager_rule)
show("Manager rule MAE (minutes)", mean_absolute_error(deliveries["minutes"], manager_pred))

# Step 3: Machine learning - give the computer data + answers, it finds the rule.
step("Step 3: ML approach - let the computer find the rule")
model = LinearRegression()
model.fit(deliveries[["distance_km"]], deliveries["minutes"])  # <- THIS is "learning"
ml_pred = model.predict(deliveries[["distance_km"]])
show("Learned rule", f"minutes = {model.intercept_:.2f} + {model.coef_[0]:.2f} * km")
show("Hidden true rule", "minutes = 8.00 + 2.70 * km")
show("ML rule MAE (minutes)", mean_absolute_error(deliveries["minutes"], ml_pred))
note("""The model was never told 8 and 2.7. It DISCOVERED them from 200 examples.
That is machine learning in one sentence: data + answers in, rule out.""")

# Step 4: Use the learned rule on a NEW order (this is a prediction).
step("Step 4: Predict for a brand-new order")
new_order = pd.DataFrame({"distance_km": [6.0]})
show("Predicted time for 6 km", f"{model.predict(new_order)[0]:.1f} minutes")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(distance_km, true_minutes, s=12, alpha=0.6, label="past deliveries")
    xs = np.linspace(0, 12, 50)
    ax.plot(xs, manager_rule(xs), "--", label="manager's rule")
    ax.plot(xs, model.intercept_ + model.coef_[0] * xs, label="learned rule")
    ax.set(xlabel="distance (km)", ylabel="minutes", title="Human rule vs learned rule")
    ax.legend()
    save_fig(fig, "01_easy_pizza_rule")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Open the black box - learning = minimizing error
# ============================================================================
banner("What does .fit() actually do? Gradient descent from scratch", "MEDIUM")
note("""A model is just a formula with KNOBS (parameters). Here: minutes = w*km + b.
Learning = turning the knobs a little at a time so the error goes down.
The 'error' we minimize is called the LOSS (here: mean squared error).""")

x = deliveries["distance_km"].to_numpy()
y = deliveries["minutes"].to_numpy()

w, b = 0.0, 0.0          # knobs start at zero = the model knows nothing
learning_rate = 0.01     # how big each knob turn is
history = []

step("Training loop (watch the loss fall)")
for epoch in range(1, 3001):
    y_hat = w * x + b                      # 1. predict
    error = y_hat - y                      # 2. compare with truth
    loss = np.mean(error ** 2)             # 3. measure how wrong (MSE)
    grad_w = 2 * np.mean(error * x)        # 4. which direction makes loss smaller?
    grad_b = 2 * np.mean(error)
    w -= learning_rate * grad_w            # 5. turn the knobs a little
    b -= learning_rate * grad_b
    history.append(loss)
    if epoch in (1, 10, 50, 200, 1000, 3000):
        print(f"  epoch {epoch:>5} | loss = {loss:8.3f} | w = {w:5.3f} | b = {b:6.3f}")

show("From-scratch result", f"minutes = {b:.2f} + {w:.2f} * km")
show("scikit-learn result", f"minutes = {model.intercept_:.2f} + {model.coef_[0]:.2f} * km")
note("""Same answer. scikit-learn uses a faster exact method for linear regression,
but the IDEA is identical: predict -> measure error -> adjust -> repeat.
Every neural network, including the ones behind ChatGPT, learns this way.""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(history)
    ax.set(yscale="log", xlabel="epoch", ylabel="MSE loss (log scale)", title="Learning = loss going down")
    save_fig(fig, "01_medium_loss_curve")

# %% =========================================================================
# LEVEL 3 (HARD): Performance improves with experience + generalization
# ============================================================================
banner("Mitchell's definition, proven with a real medical dataset", "HARD")
note("""Task T: classify a breast tumour as malignant or benign from 30 cell
measurements. Performance P: accuracy on patients the model has NEVER seen.
Experience E: number of training patients. If this is 'learning', P must rise with E.""")

X, y_bc = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bc, test_size=0.3, stratify=y_bc, random_state=RANDOM_STATE
)
show("Training pool / test patients", f"{len(X_train)} / {len(X_test)}")

step("Learning curve: more experience -> better performance?")
rows = []
for n_examples in [10, 20, 40, 80, 160, len(X_train)]:
    scores = []
    for repeat in range(20):  # repeat with different random subsets -> stable estimate
        idx = np.random.default_rng(repeat).choice(len(X_train), n_examples, replace=False)
        Xs, ys = X_train.iloc[idx], y_train.iloc[idx]
        if ys.nunique() < 2:   # tiny samples might contain only one class
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(Xs, ys)
        scores.append(accuracy_score(y_test, clf.predict(X_test)))
    rows.append({"experience_E (patients)": n_examples,
                 "test_accuracy_P (mean)": np.mean(scores),
                 "std": np.std(scores)})
curve = pd.DataFrame(rows)
print(curve.to_string(index=False))
note("""Accuracy climbs and the spread (std) shrinks as E grows. That is
Mitchell's definition, measured. Notice the curve FLATTENS: after some point,
more data of the same kind helps less. Knowing where you are on this curve tells
you whether to spend money on more data or on better features/models.""")

step("Memorization vs generalization (the trap every beginner falls into)")
memorizer = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)  # pure lookup table
learner = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train, y_train)
for name, m in [("1-NN (memorizer, no scaling)", memorizer), ("Logistic regression (learner)", learner)]:
    tr = accuracy_score(y_train, m.predict(X_train))
    te = accuracy_score(y_test, m.predict(X_test))
    print(f"  {name:<32} train acc = {tr:.3f} | test acc = {te:.3f} | gap = {tr - te:+.3f}")
note("""The memorizer scores 100% on data it has seen - it just looks the answer up.
On new patients it is worse. ML is NOT about remembering the past; it is about
GENERALIZING to the future. That is why we always judge a model on unseen data.""")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.errorbar(curve.iloc[:, 0], curve.iloc[:, 1], yerr=curve["std"], marker="o", capsize=3)
    ax.set(xscale="log", xlabel="training examples (E)", ylabel="test accuracy (P)",
           title="Performance improves with experience")
    save_fig(fig, "01_hard_learning_curve")

banner("KEY TAKEAWAYS")
print("""  1. ML = learning a rule from examples (data + answers -> rule).
  2. A model is a formula with knobs; training turns the knobs to reduce error.
  3. Learning means performance on UNSEEN data improves with more experience.
  4. 100% on training data proves nothing. Generalization is the only goal.""")
