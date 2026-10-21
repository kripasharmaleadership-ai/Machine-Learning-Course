"""
LAB 04 - Types of Machine Learning
==================================

THE FAMILY TREE
    1. SUPERVISED        : learn from examples WITH answers (labels)
                           -> regression (predict a number), classification (predict a class)
    2. UNSUPERVISED      : find structure in data WITHOUT answers
                           -> clustering, dimensionality reduction, anomaly detection
    3. SEMI-SUPERVISED   : a few labels + lots of unlabelled data (very common in companies:
                           labelling is expensive, raw data is cheap)
    4. SELF-SUPERVISED   : the data labels ITSELF (hide part of the input, predict it).
                           This is how LLMs are pre-trained: hide the next word, predict it.
    5. REINFORCEMENT     : an agent learns by TRIAL AND ERROR from rewards
                           (no correct answers given, only "good"/"bad" feedback)

LEVELS IN THIS LAB
    EASY   : Supervised vs unsupervised on the same flowers (iris).
    MEDIUM : Semi-supervised (5% labels) + unsupervised anomaly detection on
             card transactions + self-supervised "fill in the blank" on images.
    HARD   : Reinforcement learning from scratch:
             (a) multi-armed bandit choosing the best email subject line,
             (b) Q-learning robot in a warehouse grid.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, precision_score,
                             r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier

from ml_lab_utils import RANDOM_STATE, banner, note, show, step

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): Supervised vs Unsupervised on the same data
# ============================================================================
banner("Supervised vs Unsupervised - same flowers, different question", "EASY")
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
print(X.assign(species=iris.target_names[y]).sample(5, random_state=1))

step("SUPERVISED: we GIVE the species -> model learns to predict it")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
show("Accuracy on unseen flowers", accuracy_score(y_te, clf.predict(X_te)))
note("Question answered: 'Which species is THIS flower?' - needs labelled history.")

step("UNSUPERVISED: we HIDE the species -> model finds natural groups")
km = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit(X)
crosstab = pd.crosstab(iris.target_names[y], km.labels_, rownames=["true species"], colnames=["cluster"])
print(crosstab)
show("Agreement with hidden truth (ARI)", adjusted_rand_score(y, km.labels_))
note("""KMeans never saw a label, yet found groups that mostly match real species
(ARI: 0 = random, 1 = perfect). Note the clusters have NO NAMES - a human must
interpret them ('cluster 1 looks like setosa'). Business use: customer
segmentation, where no 'correct' segments exist up front.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Semi-supervised, anomaly detection, self-supervised
# ============================================================================
banner("Semi-supervised: only 5% of data is labelled", "MEDIUM")
note("""Scenario: a document-processing team has 1,797 scanned digits but only
budget to hand-label ~5% of the training images. Can the unlabelled rest help?""")
digits = load_digits()
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
    digits.data, digits.target, test_size=0.3, stratify=digits.target, random_state=RANDOM_STATE)
labelled_mask = np.zeros(len(yd_tr), dtype=bool)
labelled_mask[rng.choice(len(yd_tr), size=int(0.05 * len(yd_tr)), replace=False)] = True
y_partial = np.where(labelled_mask, yd_tr, -1)   # -1 = "no label" (sklearn convention)

base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
sup_only = base.fit(Xd_tr[labelled_mask], yd_tr[labelled_mask])
show("Labelled examples used", int(labelled_mask.sum()))
show("Supervised on 5% only - accuracy", accuracy_score(yd_te, sup_only.predict(Xd_te)))

self_train = SelfTrainingClassifier(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), threshold=0.9)
self_train.fit(Xd_tr, y_partial)
show("Self-training (5% + unlabelled)", accuracy_score(yd_te, self_train.predict(Xd_te)))
show("Pseudo-labels added by the model", int((self_train.transduction_ != -1).sum() - labelled_mask.sum()))
full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xd_tr, yd_tr)
show("Upper bound: 100% labelled", accuracy_score(yd_te, full.predict(Xd_te)))
note("""Self-training: train on the few labels, predict the unlabelled data, keep
only CONFIDENT predictions (> 90%) as 'pseudo-labels', retrain, repeat.
Watch the gain carefully: semi-supervised helps most when the classes form
clear clusters; if early pseudo-labels are wrong, errors can reinforce
themselves. Always compare against the supervised-only baseline.""")

banner("Unsupervised anomaly detection: suspicious card transactions", "MEDIUM")
n_normal, n_fraud = 5000, 50
normal = pd.DataFrame({
    "amount": rng.lognormal(3.5, 0.6, n_normal),
    "hour": rng.normal(14, 4, n_normal).clip(0, 23),
    "km_from_home": rng.exponential(5, n_normal),
    "tx_last_hour": rng.poisson(1, n_normal),
})
fraud = pd.DataFrame({
    "amount": rng.lognormal(6.0, 0.5, n_fraud),
    "hour": rng.normal(3, 1.5, n_fraud).clip(0, 23),
    "km_from_home": rng.exponential(300, n_fraud),
    "tx_last_hour": rng.poisson(6, n_fraud),
})
tx = pd.concat([normal.assign(is_fraud=0), fraud.assign(is_fraud=1)], ignore_index=True)
features = ["amount", "hour", "km_from_home", "tx_last_hour"]

iso = IsolationForest(contamination=0.01, random_state=RANDOM_STATE).fit(tx[features])  # NO labels used
tx["flagged"] = (iso.predict(tx[features]) == -1).astype(int)
show("Transactions / true frauds", f"{len(tx)} / {n_fraud}")
show("Flagged as anomalies", int(tx.flagged.sum()))
show("Precision (flags that are fraud)", precision_score(tx.is_fraud, tx.flagged))
show("Recall (frauds caught)", recall_score(tx.is_fraud, tx.flagged))
note("""IsolationForest never saw a fraud label. It finds points that are EASY TO
ISOLATE (unusual amount + odd hour + far from home). Used when fraud is new
and unlabelled. Once investigators confirm cases, you switch to supervised.""")

banner("Self-supervised: the data labels itself", "MEDIUM")
note("""Pretext task: hide the BOTTOM half of each digit image and predict it from
the TOP half. No human labelled anything - the 'answer' is the hidden pixels.
LLM pre-training is the same idea at huge scale: hide the next word, predict it.""")
top, bottom = digits.data[:, :32], digits.data[:, 32:]
t_tr, t_te, b_tr, b_te = train_test_split(top, bottom, test_size=0.3, random_state=RANDOM_STATE)
filler = KNeighborsRegressor(n_neighbors=5).fit(t_tr, b_tr)
r2_vw = dict(multioutput="variance_weighted")   # weight pixels by how much they vary
show("R^2 predicting hidden half", r2_score(b_te, filler.predict(t_te), **r2_vw))
show("Baseline: predict the average image", r2_score(b_te, np.tile(b_tr.mean(0), (len(b_te), 1)), **r2_vw))
note("""The baseline scores ~0; the model explains over half the variation in pixels\nit never saw. It learned real STRUCTURE of digits (strokes,
shapes) from unlabelled data. In practice the representation learned this way
is reused for downstream tasks that have few labels.""")

# %% =========================================================================
# LEVEL 3 (HARD): Reinforcement learning from scratch
# ============================================================================
banner("Reinforcement learning (a): multi-armed bandit for email subject lines", "HARD")
note("""Marketing has 4 subject lines. True open rates are UNKNOWN to us.
Each email we send = one trial; reward = 1 if opened. Goal: maximize total
opens while learning - the explore vs exploit trade-off.""")
TRUE_OPEN_RATES = np.array([0.04, 0.06, 0.11, 0.05])   # hidden from the agent
SUBJECTS = ["Your monthly update", "New features inside", "You left something behind", "Last chance: 20% off"]


def run_bandit(strategy: str, n_emails: int = 10_000, epsilon: float = 0.1, seed: int = 0):
    b_rng = np.random.default_rng(seed)
    sends, opens = np.zeros(4), np.zeros(4)
    total = 0
    for t in range(n_emails):
        if strategy == "random":
            arm = b_rng.integers(4)
        elif strategy == "epsilon_greedy":
            explore = b_rng.random() < epsilon or sends.min() == 0
            arm = b_rng.integers(4) if explore else np.argmax(opens / np.maximum(sends, 1))
        else:  # thompson sampling: sample a plausible open rate per arm, pick the best
            arm = np.argmax(b_rng.beta(1 + opens, 1 + sends - opens))
        reward = b_rng.random() < TRUE_OPEN_RATES[arm]
        sends[arm] += 1
        opens[arm] += reward
        total += reward
    return total, sends


best_possible = 10_000 * TRUE_OPEN_RATES.max()
rows = []
for strat in ["random", "epsilon_greedy", "thompson"]:
    totals, share_best = [], []
    for s in range(10):
        total, sends = run_bandit(strat, seed=s)
        totals.append(total)
        share_best.append(sends[2] / sends.sum())
    rows.append({"strategy": strat, "avg_opens": np.mean(totals),
                 "regret_vs_oracle": best_possible - np.mean(totals),
                 "share_sent_best_subject": np.mean(share_best)})
print(pd.DataFrame(rows).to_string(index=False))
note(f"""'Regret' = opens lost vs. an oracle that always knew the best subject
('{SUBJECTS[2]}'). Random A/B splitting wastes 75% of traffic on worse
options. Thompson sampling shifts traffic to the winner automatically.
This is RL in production at many companies: ads, recommendations, pricing.""")

banner("Reinforcement learning (b): Q-learning warehouse robot", "HARD")
GRID = [
    "S . . # .",
    ". # . # .",
    ". # . . .",
    ". . . # G",
]
grid = np.array([row.split() for row in GRID])
n_rows, n_cols = grid.shape
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}
start = tuple(np.argwhere(grid == "S")[0])
goal = tuple(np.argwhere(grid == "G")[0])


def env_step(state, action):
    """Environment: returns next_state, reward, done. Walls/edges = stay put."""
    r, c = state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1]
    if not (0 <= r < n_rows and 0 <= c < n_cols) or grid[r, c] == "#":
        return state, -5.0, False          # bumped into a shelf: penalty
    if (r, c) == goal:
        return (r, c), 100.0, True         # delivered the package
    return (r, c), -1.0, False             # each move costs battery


Q = np.zeros((n_rows, n_cols, 4))          # the robot's "experience table"
alpha, gamma, eps = 0.1, 0.95, 0.2
steps_per_episode = []
for episode in range(500):
    state, done, n_steps = start, False, 0
    while not done and n_steps < 200:
        a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[state]))
        nxt, reward, done = env_step(state, a)
        # Q-learning update: move Q towards reward + discounted best future value
        Q[state][a] += alpha * (reward + gamma * np.max(Q[nxt]) * (not done) - Q[state][a])
        state, n_steps = nxt, n_steps + 1
    steps_per_episode.append(n_steps)

show("Avg steps, first 20 episodes", np.mean(steps_per_episode[:20]))
show("Avg steps, last 20 episodes", np.mean(steps_per_episode[-20:]))

# Exploit only (no random exploration): follow the learned policy
state, route = start, [start]
while state != goal and len(route) < 50:
    state, _, _ = env_step(state, int(np.argmax(Q[state])))
    route.append(state)
show("Greedy route length (optimal = 7)", len(route) - 1)
step("Learned policy (best action in each cell)")
for r in range(n_rows):
    print("   ", " ".join(grid[r, c] if grid[r, c] in "#G" else ARROWS[int(np.argmax(Q[r, c]))]
                         for c in range(n_cols)))
note("""Nobody told the robot the route. It received only rewards (+100 deliver,
-1 per move, -5 per crash) and discovered the shortest path by trial and
error. RL fits problems with SEQUENTIAL decisions and delayed rewards:
robotics, supply chain, game AI - and RLHF, used to align LLMs.""")

banner("KEY TAKEAWAYS")
print(pd.DataFrame([
    ["Supervised", "Labelled examples", "Predict label/number", "Churn, price, fraud (known)"],
    ["Unsupervised", "Unlabelled data", "Groups / anomalies / structure", "Segments, new fraud"],
    ["Semi-supervised", "Few labels + many unlabelled", "Predict label", "Document/image tagging"],
    ["Self-supervised", "Raw data (labels made from data)", "Representations", "LLM / vision pre-training"],
    ["Reinforcement", "Rewards from interaction", "A policy (what to do)", "Bandits, robots, RLHF"],
], columns=["Type", "Learns from", "Produces", "Enterprise example"]).to_string(index=False))
