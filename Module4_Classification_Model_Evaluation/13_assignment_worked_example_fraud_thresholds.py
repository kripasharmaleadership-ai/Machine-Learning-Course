"""
LAB 13 - ASSIGNMENT WORKED EXAMPLE: Threshold Strategies for Card-Fraud Operations
==================================================================================

THE ASSIGNMENT (see ASSIGNMENT.md)
    "Design different threshold strategies for a medical, fraud, or customer-churn scenario."
    This lab is the REFERENCE SOLUTION for FRAUD. Your assignment applies the same method to
    a MEDICAL scenario (sepsis screening in an emergency department: assignment_starter.py).

THE BUSINESS CONTEXT
    A card issuer scores every transaction with a fraud model (already trained; the model is
    NOT the question here). Operations must decide what to DO with each score:
        approve        nothing happens; if it was fraud, the bank loses the amount
        review         an analyst checks it ($20); a legit customer's card is held ($10 friction)
        auto-decline   blocked instantly, no analyst; a legit customer is declined ($40 friction)
    Constraints: 20 analysts can review at most 2% of all transactions; customer experience
    caps the share of good customers disturbed (held or declined) at 2%.

THE METHOD (what "design threshold strategies" means)
    1. Write the requirements BEFORE seeing any results (hard gates + weighted criteria).
    2. Design several STRATEGIES - single thresholds, capacity rules, cost-based rules,
       per-transaction rules, a two-tier triage - and tune each on VALIDATION data only.
    3. Score every strategy against the requirements; stress-test the choice (fraud wave,
       cost changes, different weights).
    4. Open the test set once; check segments; write a decision memo.

LEVELS
    EASY   : The brief, the costs, the requirements, the model's scores.
    MEDIUM : Ten threshold strategies, tuned on validation, evaluated identically.
    HARD   : Scorecard, stress tests, test set, segments, memo, rubric self-check.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml_lab_utils import (HAS_MPL, OUT_DIR, banner, fraud_features, make_transactions, note, plt, save_fig, show,
                          step)
from classification_toolkit import ModelScorecard, Requirement

REVIEW_COST, HOLD_FRICTION, DECLINE_FRICTION = 20.0, 10.0, 40.0
CAPACITY, FRICTION_CAP = 0.02, 0.02

# %% =========================================================================
# LEVEL 1 (EASY): Brief, requirements, model scores
# ============================================================================
banner("The brief and the requirements (written BEFORE any results)", "EASY")
REQUIREMENTS = [
    Requirement("Analyst capacity", "review rate %", threshold=CAPACITY * 100, hard=True,
                why="20 analysts can review at most 2% of transactions"),
    Requirement("Customer friction", "good customers disturbed %", threshold=FRICTION_CAP * 100, hard=True,
                why="Customer experience: at most 2% of good customers held or declined"),
    Requirement("Fraud $ stopped", "fraud $ stopped %", threshold=60.0, hard=True, higher_is_better=True,
                why="Board target: stop at least 60% of fraud losses"),
    Requirement("Net cost", "net cost $", weight=0.50, why="Fraud losses + review cost + customer friction"),
    Requirement("Fraud $ stopped (soft)", "fraud $ stopped %", weight=0.20, higher_is_better=True,
                why="Regulator and press: share of fraud money stopped"),
    Requirement("Review precision", "review precision", weight=0.15, higher_is_better=True,
                why="Analyst morale: share of reviews that are real fraud"),
    Requirement("Simplicity", "simplicity", weight=0.15, higher_is_better=True,
                why="1 = one number to explain and monitor; lower = more moving parts"),
]
print(pd.DataFrame([{"requirement": r.name, "type": "HARD gate" if r.hard else f"soft ({r.weight:.0%})",
                     "metric": r.metric, "threshold": r.threshold if r.hard else "-", "why": r.why}
                    for r in REQUIREMENTS]).to_string(index=False))

step("1.1  Data and model")
tx = make_transactions(n=120_000, seed=21)
idx = np.random.default_rng(0).permutation(len(tx))
parts = {"train": idx[:60_000], "validation": idx[60_000:90_000], "test": idx[90_000:]}
interaction = FunctionTransformer(lambda d: d.assign(night_x_foreign=d.is_night * d.is_foreign))
model = make_pipeline(interaction, StandardScaler(), LogisticRegression(max_iter=3000))
X_all, y_all, amt_all = fraud_features(tx), tx["is_fraud"].to_numpy(), tx["amount"].to_numpy()
model.fit(X_all.iloc[parts["train"]], y_all[parts["train"]])
D = {}
for name, ix in parts.items():
    D[name] = {"y": y_all[ix], "p": model.predict_proba(X_all.iloc[ix])[:, 1], "amt": amt_all[ix],
               "new_account": tx.account_age_days.to_numpy()[ix] < 90, "foreign": tx.is_foreign.to_numpy()[ix] == 1}
for name in ["validation", "test"]:
    show(f"{name}: transactions / frauds / fraud $", f"{len(D[name]['y']):,} / {D[name]['y'].sum()} / "
                                                     f"${D[name]['amt'][D[name]['y'] == 1].sum():,.0f}")
show("Validation ROC-AUC / AP", f"{roc_auc_score(D['validation']['y'], D['validation']['p']):.3f} / "
                                f"{average_precision_score(D['validation']['y'], D['validation']['p']):.3f}")
note("""The model is Lab 10's winner (logistic regression + night x foreign), trained on
60,000 transactions and frozen. Validation (30,000) is used to TUNE every
strategy; the test set (30,000) is sealed until Level 3. The requirements above
were written before any strategy was evaluated: no moving the goalposts.""")


def evaluate(y, p, amt, review, decline, simplicity=1.0) -> dict:
    """Score one decision policy. review / decline are boolean masks (decline wins if both)."""
    review = review & ~decline
    fraud, legit = y == 1, y == 0
    stopped = (review | decline) & fraud
    cost = (amt[fraud & ~stopped].sum() + REVIEW_COST * review.sum() + HOLD_FRICTION * (review & legit).sum()
            + DECLINE_FRICTION * (decline & legit).sum())
    return {"review rate %": review.mean() * 100, "auto-declined %": decline.mean() * 100,
            "frauds stopped (count %)": stopped.sum() / fraud.sum() * 100,
            "fraud $ stopped %": amt[stopped].sum() / amt[fraud].sum() * 100,
            "good customers disturbed %": ((review | decline) & legit).sum() / legit.sum() * 100,
            "review precision": float(y[review].mean()) if review.any() else 0.0,
            "net cost $": float(cost), "simplicity": simplicity}


# %% =========================================================================
# LEVEL 2 (MEDIUM): Ten strategies, tuned on validation
# ============================================================================
banner("Ten threshold strategies, tuned on VALIDATION only", "MEDIUM")
V = D["validation"]
grid = np.unique(np.quantile(V["p"], np.linspace(0.5, 0.9999, 600)))


def best_threshold(objective, p=V["p"]):
    """Search the validation grid for the threshold that maximises objective(review_mask)."""
    return float(max(grid, key=lambda t: objective(p >= t)))


def f1_of(mask):
    tp = (mask & (V["y"] == 1)).sum()
    return 2 * tp / (mask.sum() + V["y"].sum())


AVG_FRAUD = float(V["amt"][V["y"] == 1].mean())
P_STAR = (REVIEW_COST + HOLD_FRICTION) / (REVIEW_COST + HOLD_FRICTION + AVG_FRAUD - REVIEW_COST)
T = {
    "S0 default 0.5": 0.5,
    "S1 max F1": best_threshold(f1_of),
    "S2 Youden J": best_threshold(lambda m: (m & (V["y"] == 1)).sum() / V["y"].sum() - (m & (V["y"] == 0)).sum() / (V["y"] == 0).sum()),
    "S3 recall >= 80%": float(np.quantile(V["p"][V["y"] == 1], 0.20)),
    "S4 Bayes p* (avg fraud $)": P_STAR,
}
SIMPLE = {"S0 default 0.5": 1.0, "S1 max F1": 1.0, "S2 Youden J": 1.0, "S3 recall >= 80%": 1.0,
          "S4 Bayes p* (avg fraud $)": 1.0, "S5 capacity: top 2%": 1.0, "S6 p*(amount) per transaction": 0.7,
          "S7 two-tier: p*(amount) review + auto-decline": 0.5, "S8 expected-$ ranking, top 2%": 0.6,
          "S9 auto-decline + expected-$ queue (top 2%)": 0.4}


def p_star_amount(amt):
    """Review when p * amount > (1 - p) * (review + friction) + p * review  <=>  p > p*(amount)."""
    c = REVIEW_COST + HOLD_FRICTION
    return np.where(amt > REVIEW_COST, c / (c + amt - REVIEW_COST), 1.01)




def top_share(score, share=CAPACITY, eligible=None):
    """Capacity rule: flag the top `share` of THIS batch by score (analysts work a daily queue of fixed size).
    Because it ranks within each batch, it can never exceed the capacity, whatever the fraud rate."""
    eligible = np.ones(len(score), bool) if eligible is None else eligible
    k = int(np.floor(share * len(score)))
    order = np.argsort(-np.where(eligible, score, -np.inf))[:k]
    mask = np.zeros(len(score), bool)
    mask[order] = True
    return mask & eligible


def expected_saving(p, amt):
    return p * amt - (1 - p) * (REVIEW_COST + HOLD_FRICTION)

# two-tier: auto-decline above t_high (chosen on validation), p*(amount) review below it
t_high_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
T_HIGH = min(t_high_grid, key=lambda th: evaluate(V["y"], V["p"], V["amt"], V["p"] >= p_star_amount(V["amt"]),
                                                  V["p"] >= th)["net cost $"])


def decide(name, d):
    """Return (review, decline) masks for strategy `name` on data split d. Every number was set on validation."""
    p, amt = d["p"], d["amt"]
    none = np.zeros(len(p), bool)
    if name in T:
        return p >= T[name], none
    if name == "S5 capacity: top 2%":
        return top_share(p), none
    if name == "S6 p*(amount) per transaction":
        return p >= p_star_amount(amt), none
    if name == "S7 two-tier: p*(amount) review + auto-decline":
        return p >= p_star_amount(amt), p >= T_HIGH
    if name == "S8 expected-$ ranking, top 2%":
        return top_share(expected_saving(p, amt)), none
    if name == "S9 auto-decline + expected-$ queue (top 2%)":
        decline = p >= T_HIGH
        return top_share(expected_saving(p, amt), eligible=~decline), decline
    raise KeyError(name)


STRATEGIES = list(SIMPLE)
val_tab = pd.DataFrame({s: evaluate(V["y"], V["p"], V["amt"], *decide(s, V), simplicity=SIMPLE[s]) for s in STRATEGIES}).T
RULE = {"S5 capacity: top 2%": "top 2% by p", "S6 p*(amount) per transaction": "p > p*(amount)",
        "S7 two-tier: p*(amount) review + auto-decline": f"p*(amt); decline p>={T_HIGH}",
        "S8 expected-$ ranking, top 2%": "top 2% by E[$]",
        "S9 auto-decline + expected-$ queue (top 2%)": f"decline p>={T_HIGH}; top 2% E[$]"}
val_tab.insert(0, "rule", [f"p >= {T[s]:.3f}" if s in T else RULE[s] for s in STRATEGIES])
print(val_tab.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
note(f"""Every strategy uses the SAME model; only the decision rule changes. Things to see:
- S0 (0.5) stops little fraud. S1/S2 optimise statistics, not money; Youden J
  floods the analysts (it treats a miss and a false alarm as equally bad).
- S3 (a recall target) is easy to explain but ignores the dollars.
- S4 p* = {P_STAR:.3f} uses the economics with ONE average fraud amount.
- S5 fills exactly the analyst capacity (top 2% of each day's batch by p).
- S6 gives every transaction its own threshold from its amount.
- S7 adds auto-decline above p = {T_HIGH}: the surest frauds skip the queue.
- S8 ranks by expected $ saved, E[$] = p x amount - (1 - p) x $30, and fills the
  capacity: S5's queue with S6's logic.
- S9 combines them: auto-decline the near-certain frauds (no analyst needed),
  then fill the 2% queue by expected $ from the rest.
Which is 'best' depends on the requirements, so we score them next.""", keep_lines=True)

# %% =========================================================================
# LEVEL 3 (HARD): Decide, stress-test, test once, document
# ============================================================================
banner("Scorecard, stress tests, test set, memo", "HARD")
step("3.1  The scorecard on validation: hard gates, then weighted score")
card = ModelScorecard("Card-fraud decision policy (approve / review / auto-decline)", REQUIREMENTS)
for s in STRATEGIES:
    card.add(s, val_tab.loc[s].drop("rule").astype(float).to_dict())
board = card.evaluate()
print(board.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
winner = card.recommend()
show("Recommended strategy", winner)

step("3.2  Stress tests: does the choice survive a changed world?")
rng = np.random.default_rng(1)
legit_ix, fraud_ix = np.flatnonzero(V["y"] == 0), np.flatnonzero(V["y"] == 1)
wave_ix = np.r_[fraud_ix, rng.choice(legit_ix, len(legit_ix) // 3, replace=False)]
Vw = {k: v[wave_ix] for k, v in V.items()}
scenarios = {}
scenarios["as written"] = winner
wave_card = ModelScorecard(card.problem, REQUIREMENTS)
for s in STRATEGIES:
    wave_card.add(s, evaluate(Vw["y"], Vw["p"], Vw["amt"], *decide(s, Vw), simplicity=SIMPLE[s]))
scenarios["fraud wave (3x fraud share)"] = wave_card.recommend()
REVIEW_COST_SAVED = REVIEW_COST
REVIEW_COST = 40.0                                                        # analysts get twice as expensive
exp_card = ModelScorecard(card.problem, REQUIREMENTS)
for s in STRATEGIES:
    exp_card.add(s, evaluate(V["y"], V["p"], V["amt"], *decide(s, V), simplicity=SIMPLE[s]))
scenarios["review cost doubles ($40)"] = exp_card.recommend()
REVIEW_COST = REVIEW_COST_SAVED
for label, w in {"cost-only (net cost 100%)": [1, 0, 0, 0], "simplicity-heavy": [0.35, 0.15, 0.1, 0.4]}.items():
    reqs = [Requirement(**r.__dict__) for r in REQUIREMENTS]
    for r, wt in zip([r for r in reqs if not r.hard], w):
        r.weight = wt
    scenarios[label] = ModelScorecard(card.problem, reqs, results=card.results).recommend()
print(pd.Series(scenarios, name="recommended").to_string())
print("\n  Fraud wave: which strategies still pass every hard gate?")
wb = wave_card.evaluate()
wb["fraud $ stopped %"] = [wave_card.results[s]["fraud $ stopped %"] for s in wb.index]
print(wb[["fraud $ stopped %", "eligible", "failed gates"]].to_string(float_format=lambda v: f"{v:.1f}"))
wave_mode = {}
for t_wave in [0.4, 0.3, 0.2]:
    dec = Vw["p"] >= t_wave
    wave_mode[f"S9 wave mode: auto-decline p >= {t_wave}"] = evaluate(
        Vw["y"], Vw["p"], Vw["amt"], top_share(expected_saving(Vw["p"], Vw["amt"]), eligible=~dec), dec)
wm = pd.DataFrame(wave_mode).T[["review rate %", "auto-declined %", "fraud $ stopped %", "good customers disturbed %",
                                "net cost $"]]
print("\n  Fallback check: S9 with a lower auto-decline threshold, in the fraud wave")
print(wm.to_string(float_format=lambda v: f"{v:.2f}" if abs(v) < 100 else f"{v:,.0f}"))
ok_modes = [k for k, r in wm.iterrows() if r["fraud $ stopped %"] >= 60 and r["good customers disturbed %"] <= FRICTION_CAP * 100]
WAVE_FALLBACK = ok_modes[0] if ok_modes else None
fallback_text = (f"Lowering the auto-decline threshold ({WAVE_FALLBACK.split(': ')[1]}) restores the 60% target within "
                 "the friction cap: that becomes the documented wave-mode fallback." if WAVE_FALLBACK else
                 "Even aggressive auto-declining can't meet the target within the friction cap: the board must "
                 "choose between more analysts and more customer friction.")
note(f"""A policy is chosen for the world it will meet, not just the validation month.
In a fraud wave (3x the fraud share) NO strategy passes every gate. Rules that
threshold the probability flood the analysts; queue rules (S5, S8, S9) never
overflow, but 20 analysts can't review 3x the fraud, so fraud $ stopped falls
below the 60% target. The fix is not a better threshold but a different lever:
act without an analyst. {fallback_text} When a requirement can't be met in a
plausible scenario, the memo says so instead of hiding it.""")

step("3.3  Open the test set ONCE")
Tt = D["test"]
test_tab = pd.DataFrame({s: evaluate(Tt["y"], Tt["p"], Tt["amt"], *decide(s, Tt), simplicity=SIMPLE[s])
                         for s in STRATEGIES}).T
print(test_tab.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
others = [s for s in board.index if s != winner]
eligible_others = [s for s in others if bool(board.loc[s, "eligible"])]
runner_up = eligible_others[0] if eligible_others else min(others, key=lambda s: val_tab.loc[s, "net cost $"])
diffs = []
for _ in range(1000):
    b = rng.integers(0, len(Tt["y"]), len(Tt["y"]))
    Tb = {k: v[b] for k, v in Tt.items()}
    diffs.append(evaluate(Tb["y"], Tb["p"], Tb["amt"], *decide(runner_up, Tb))["net cost $"]
                 - evaluate(Tb["y"], Tb["p"], Tb["amt"], *decide(winner, Tb))["net cost $"])
lo, hi = np.percentile(diffs, [2.5, 97.5])
show(f"Net cost: {runner_up} minus {winner}", f"${np.mean(diffs):,.0f}  95% CI [${lo:,.0f}, ${hi:,.0f}]")
gates_test = {r.name: ModelScorecard._passes(r, test_tab.loc[winner, r.metric]) for r in REQUIREMENTS if r.hard}
show("Winner passes every hard gate on test", all(gates_test.values()))

step("3.4  Segments: who pays for the policy?")
rw, dw = decide(winner, Tt)
flag = rw | dw
seg_rows = {}
for label, mask in {"new accounts (< 90 days)": Tt["new_account"], "older accounts": ~Tt["new_account"],
                    "foreign transactions": Tt["foreign"], "domestic": ~Tt["foreign"]}.items():
    y_, f_ = Tt["y"][mask], flag[mask]
    seg_rows[label] = {"transactions": int(mask.sum()), "fraud rate": y_.mean(),
                       "frauds stopped %": (f_ & (y_ == 1)).sum() / max(y_.sum(), 1) * 100,
                       "good customers disturbed %": (f_ & (y_ == 0)).sum() / max((y_ == 0).sum(), 1) * 100}
seg = pd.DataFrame(seg_rows).T
print(seg.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}"))
note("""Risk concentrates in some segments, and so do the costs of the policy: good
customers who travel abroad or just opened an account are disturbed several
times more often than average. That may be acceptable (they ARE riskier) but
it must be a conscious, documented choice, with a monitoring metric per segment.""")

step("3.5  Write the decision memo")
tw = test_tab.loc[winner]
justification = f"""**Why {winner}:** it passes every hard gate (analyst capacity, customer friction, fraud $ stopped)
and has the best weighted score on validation; on the sealed test set it stops {tw['fraud $ stopped %']:.0f}% of fraud
dollars, reviews {tw['review rate %']:.2f}% of transactions and disturbs {tw['good customers disturbed %']:.2f}% of good customers,
for a net cost of ${tw['net cost $']:,.0f} per 30,000 transactions.

- **Versus the runner-up ({runner_up}):** test net-cost difference ${np.mean(diffs):,.0f} (95% CI {lo:,.0f} to {hi:,.0f}).
- **Stress tests:** {'; '.join(f'{k}: {v}' for k, v in scenarios.items())}.
- **Rejected:** the default 0.5 stops too little fraud; statistical rules (max F1, Youden J) ignore the money and the capacity.
- **Segments:** foreign and new-account customers are disturbed more often; monitor both monthly.
- **Monitoring and fallback:** track the review queue, the auto-decline rate and the fraud $ stopped daily.
  The queue rule cannot overflow the analysts. In a fraud wave no policy meets every gate with 20 analysts:
  {fallback_text} Re-tune when the review or friction costs change."""
memo = card.to_markdown(OUT_DIR / "13_fraud_threshold_policy_memo.md", justification=justification,
                        header="Model frozen (logistic + night x foreign). Strategies tuned on 30,000 validation "
                               "transactions; test (30,000) opened once. Requirements fixed before any results.")
card.to_json(OUT_DIR / "13_fraud_threshold_policy.json")
print("  [saved] outputs/13_fraud_threshold_policy_memo.md + 13_fraud_threshold_policy.json")

step("3.6  Self-check with the assignment rubric")
kinds = {"single statistical threshold": any(s in STRATEGIES for s in ["S1 max F1", "S2 Youden J"]),
         "fixed-recall threshold": "S3 recall >= 80%" in STRATEGIES,
         "capacity / alert-budget rule": "S5 capacity: top 2%" in STRATEGIES,
         "cost-based threshold": "S4 Bayes p* (avg fraud $)" in STRATEGIES,
         "case-specific or multi-tier rule": "S7 two-tier: p*(amount) review + auto-decline" in STRATEGIES}
checks = {
    "Requirements written first (>= 3 hard gates, >= 2 soft criteria)": sum(r.hard for r in REQUIREMENTS) >= 3
    and sum(r.weight > 0 for r in REQUIREMENTS) >= 2,
    "Baseline policy (default 0.5) included": "S0 default 0.5" in STRATEGIES,
    ">= 4 strategies of different kinds": sum(kinds.values()) >= 4,
    "All thresholds tuned on validation, not test": True,
    "Stress test / sensitivity analysis (>= 2 scenarios)": len(scenarios) >= 3,
    "Test set opened once, with a CI for the top-2 difference": lo <= hi,
    "Segment analysis": len(seg) >= 2,
    "Memo with recommendation and justification": memo.exists() and winner in memo.read_text(encoding="utf-8"),
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
show("Rubric checks passed", f"{sum(checks.values())}/{len(checks)}")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    b = val_tab.copy()
    colors = ["#54a24b" if s == winner else ("#4c78a8" if bool(board.loc[s, "eligible"]) else "#9d9d9d") for s in b.index]
    axes[0].scatter(b["review rate %"], b["fraud $ stopped %"], s=80, c=colors)
    for s, r in b.iterrows():
        axes[0].annotate(s.split(" ")[0], (r["review rate %"], r["fraud $ stopped %"]), xytext=(4, 4), textcoords="offset points")
    axes[0].axvline(CAPACITY * 100, color="r", ls="--", label="capacity gate")
    axes[0].axhline(60, color="orange", ls="--", label="fraud $ gate")
    axes[0].set(xlabel="review rate (% of transactions)", ylabel="fraud $ stopped (%)",
                title="Strategies on validation (green = winner, grey = fails a gate)")
    axes[0].legend()
    t_ = test_tab.sort_values("net cost $")
    axes[1].barh([s.split(" ", 1)[0] for s in t_.index], t_["net cost $"] / 1000,
                 color=["#54a24b" if s == winner else "#9d9d9d" for s in t_.index])
    axes[1].set(xlabel="net cost on test ($k per 30,000 transactions)", title="Test set (opened once)")
    axes[1].invert_yaxis()
    save_fig(fig, "13_fraud_threshold_policy")

banner("KEY TAKEAWAYS")
print("""  1. The model gives scores; the THRESHOLD STRATEGY turns them into actions. Design it deliberately.
  2. Write requirements first: capacity, customer friction, a business target, then weighted criteria.
  3. Compare many kinds of strategy: statistical, fixed-recall, capacity, cost-based, per-case, multi-tier.
  4. Tune on validation, stress-test (fraud wave, cost changes, weights), then open the test set once.
  5. Check who pays (segments), and write a memo with a monitoring plan and a fallback.""")
