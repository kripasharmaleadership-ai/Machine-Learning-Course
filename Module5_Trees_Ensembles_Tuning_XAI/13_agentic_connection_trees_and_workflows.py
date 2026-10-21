"""
LAB 13 - Connection to Agentic AI: Decision Trees vs Agent Workflows
====================================================================

THE PARALLEL
    decision tree                        agent workflow
    a node asks a question               a step checks a condition / calls a tool / asks a model
    branches                             conditional branching (if / else, a router)
    a leaf = a prediction                a terminal state = an action (approve, decline, escalate)
    the path from root to leaf           the execution TRACE (the explanation of what happened)
    hand-written rules vs learned splits deterministic rules vs learned (or LLM) decisions
    leaf purity / leaf size              confidence -> escalate to a human when low

    An LLM agent decides its branches by itself at run time; a WORKFLOW fixes them in code.
    Production agents mix both: hard RULES for what must never vary (compliance, safety),
    LEARNED decisions for judgement calls, and ESCALATION when confidence is low.

THE SCENARIO: a loan-application assistant that orchestrates checks, a risk model, and humans,
    run on 3,000 synthetic applicants whose true default outcome we know.
    Money: an approved good loan earns $300, an approved default loses $2,000, a human review
    costs $25 (and the reviewer, with documents, catches 75% of defaulters), each tool call $0.50.

LEVELS
    EASY   : A workflow IS a tree: hand-written branching, its trace = the explanation.
    MEDIUM : Rules vs learned decisions; compile a learned tree into readable workflow code;
             a router agent that sends each case to the right specialist.
    HARD   : An orchestrator with budgets and traces; confidence-based escalation from tree leaves
             and from forest disagreement, priced in dollars.
"""
# %%
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from ml_lab_utils import CREDIT_NUM, HAS_MPL, RANDOM_STATE, banner, make_credit_applicants, note, plt, save_fig, show, step

GOOD_LOAN, BAD_LOAN, HUMAN_COST, TOOL_COST, HUMAN_CATCH = 300.0, -2000.0, 25.0, 0.5, 0.75

credit = make_credit_applicants(n=12_000, seed=13, with_truth=True)
credit["employment_risk"] = credit.employment.map({"salaried": 0, "self_employed": 1, "contract": 2, "unemployed": 3})
FEATS = CREDIT_NUM + ["employment_risk"]
hist, live = train_test_split(credit, test_size=3000, stratify=credit["default"], random_state=RANDOM_STATE)
hist, live = hist.reset_index(drop=True), live.reset_index(drop=True)
rng = np.random.default_rng(0)
live.loc[rng.random(len(live)) < 0.06, "income_k"] = np.nan          # some applications arrive without income


# %% =========================================================================
# LEVEL 1 (EASY): A workflow is a tree
# ============================================================================
banner("A hand-written workflow is a decision tree", "EASY")


def rule_workflow(app: pd.Series, trace: list) -> str:
    """The credit team's current process, written as code. Every `if` is a tree node."""
    trace.append("start")
    if app.age < 18:
        trace.append("age < 18 -> DECLINE (legal rule)")
        return "decline"
    if np.isnan(app.income_k):
        trace.append("income missing -> REQUEST DOCUMENTS")
        return "request_docs"
    if app.late_payments >= 3:
        trace.append("late payments >= 3 -> DECLINE")
        return "decline"
    if app.debt_ratio > 0.5:
        trace.append(f"debt ratio {app.debt_ratio:.2f} > 0.5 -> HUMAN REVIEW")
        return "human_review"
    trace.append("passed every check -> APPROVE")
    return "approve"


shown = set()
for i in range(len(live)):
    tr = []
    action = rule_workflow(live.iloc[i], tr)
    if action not in shown:                                          # one example per possible outcome
        shown.add(action)
        print(f"  applicant {i}: action = {action:<13} trace: {' | '.join(tr)}")
    if len(shown) == 4:
        break
print("""
     the same workflow drawn as a tree:
        age < 18 ? -- yes --> DECLINE
          no -> income missing ? -- yes --> REQUEST DOCUMENTS
                  no -> late >= 3 ? -- yes --> DECLINE
                          no -> debt > 0.5 ? -- yes --> HUMAN REVIEW
                                  no --> APPROVE""")
note("""Each `if` is a node, each `return` a leaf, and the list of conditions an
applicant passed through (the TRACE) is exactly a decision-tree path: the
explanation of the outcome. That is why workflows are auditable: you can
replay any case. Agent frameworks (LangGraph, state machines, BPMN) are this
structure with tools and models plugged into the nodes.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Rules vs learned decisions; a router agent
# ============================================================================
banner("Rules vs learned decisions, and a router agent", "MEDIUM")


def money(actions: pd.Series, df: pd.DataFrame) -> dict:
    """Expected outcome per application. Human review: reviewer declines 75% of true defaulters, approves the rest."""
    y = df["default"].to_numpy()
    a = actions.to_numpy()
    value = np.where(a == "approve", np.where(y == 1, BAD_LOAN, GOOD_LOAN), 0.0)
    human = a == "human_review"
    value = value + np.where(human, -HUMAN_COST + np.where(y == 1, (1 - HUMAN_CATCH) * BAD_LOAN, GOOD_LOAN), 0.0)
    return {"$ per application": value.mean(), "approved": (a == "approve").mean(), "to humans": human.mean(),
            "declined": (a == "decline").mean(), "default rate among auto-approved": y[a == "approve"].mean()}


step("2.1  Replace the hand-written risk rules with a learned tree")
complete = live.dropna(subset=["income_k"])
rule_actions = complete.apply(lambda r: rule_workflow(r, []), axis=1)
learned = DecisionTreeClassifier(max_depth=4, min_samples_leaf=150, random_state=0).fit(hist[FEATS], hist["default"])
p_tree = learned.predict_proba(complete[FEATS])[:, 1]
break_even = -GOOD_LOAN / (BAD_LOAN - GOOD_LOAN)                      # approve if p < 300 / 2,300
learned_actions = pd.Series(np.where(complete.age < 18, "decline", np.where(p_tree < break_even, "approve", "decline")),
                            index=complete.index)
cmp = pd.DataFrame({"hand-written rules": money(rule_actions, complete),
                    "learned tree (approve if p < break-even)": money(learned_actions, complete)}).T
print(cmp.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}"))
show("Break-even default probability (approve below)", f"{break_even:.3f}")
note(f"""The rules were written by experts years ago; the tree learned its splits from
{len(hist):,} past outcomes. The learned version earns {cmp.iloc[1, 0] - cmp.iloc[0, 0]:+.0f} $ per application: it finds
thresholds and combinations a person wouldn't, and it is tuned to the money
(break-even probability {break_even:.2f}). But note what stayed a RULE: 'age < 18 ->
decline' is the law, not a prediction. Never let a learned model override a hard
constraint.""")

step("2.2  Compile the learned tree into workflow code (readable, reviewable, versionable)")


def compile_tree(tree, names, indent="    "):
    """Turn a fitted tree into Python source: the learned decision becomes an ordinary workflow node."""
    t = tree.tree_
    lines = ["def learned_risk_node(app):"]

    def rec(nd, depth):
        pad = indent * depth
        if t.feature[nd] < 0:
            p_ = t.value[nd, 0, 1]
            lines.append(f"{pad}return {p_:.6f}   # leaf: {int(t.n_node_samples[nd])} past cases")
            return
        lines.append(f"{pad}if app['{names[t.feature[nd]]}'] <= {t.threshold[nd]:.3f}:")
        rec(t.children_left[nd], depth + 1)
        lines.append(f"{pad}else:")
        rec(t.children_right[nd], depth + 1)

    rec(0, 1)
    return "\n".join(lines)


source = compile_tree(learned, FEATS)
print("\n".join(source.splitlines()[:14]) + "\n    ...")
namespace = {}
exec(source, namespace)                                              # the generated code is real, runnable Python
check = np.array([namespace["learned_risk_node"](r) for _, r in complete[FEATS].head(500).iterrows()])
show("Generated code == the model's predictions", bool(np.allclose(check, p_tree[:500], atol=1e-6)))
show("Lines of generated code", len(source.splitlines()))
note("""A shallow learned tree can be EXPORTED as plain if/else code: a learned
decision that a reviewer can read line by line, diff between versions and
deploy with no ML runtime. This is the sweet spot for workflows in regulated
settings: learned from data, but as inspectable as hand-written rules. (An
LLM's routing decision can't be exported like this, which is why agents log
traces and are evaluated on labelled cases instead.)""")

step("2.3  A router agent: send each case to the right specialist")
SPECIALISTS = ["fast_track", "standard_underwriter", "income_verifier", "collections_expert"]


def best_specialist(df):
    """Which specialist historically resolved each kind of case best (the ROUTING LABEL in the logs)."""
    return np.select([df.late_payments >= 2, df.employment_risk >= 2, (df.debt_ratio < 0.3) & (df.late_payments == 0)],
                     ["collections_expert", "income_verifier", "fast_track"], "standard_underwriter")


def logged_route(df, seed):
    """The routing labels as LOGGED: the best specialist, but 12% of cases were handled by someone else."""
    r_ = np.random.default_rng(seed)
    best = best_specialist(df)
    return np.where(r_.random(len(df)) < 0.12, r_.choice(SPECIALISTS, len(df)), best)


hist_route, live_route = logged_route(hist, 1), best_specialist(complete)
router = DecisionTreeClassifier(max_depth=4, random_state=0).fit(hist[FEATS], hist_route)
keyword_router = np.where(complete.late_payments >= 1, "collections_expert", "standard_underwriter")   # a naive hand rule
rows = {"naive hand rule (any late payment -> collections)": (keyword_router == live_route).mean(),
        "learned router (depth-4 tree)": (router.predict(complete[FEATS]) == live_route).mean()}
print(pd.Series(rows, name="routing accuracy").round(3).to_string())
print(export_text(router, feature_names=FEATS, max_depth=2))
note("""A ROUTER AGENT is a classifier whose classes are agents (or tools, or models:
Module 4, Lab 12). Learned from NOISY logs (12% of cases were handled by the
'wrong' specialist), a small tree still recovers the underlying routing logic
almost exactly: the tree learns the majority pattern, not the noise, and prints it as
rules a team lead can check. In LLM systems the router is often a prompt
('which agent should handle this?'), but it is evaluated the same way:
routing accuracy on labelled cases, and a confusion matrix of who got what.""")

# %% =========================================================================
# LEVEL 3 (HARD): Orchestration and confidence-based escalation
# ============================================================================
banner("An orchestrator with budgets, traces and escalation", "HARD")


@dataclass
class Orchestrator:
    """A minimal workflow engine: named steps, each returning the next step; a step budget; a trace per case."""
    steps: dict
    max_steps: int = 8
    traces: list = field(default_factory=list)

    def run(self, app: pd.Series) -> tuple[str, float]:
        state, cost, trace, node = {"docs_requested": False}, 0.0, [], "intake"
        for _ in range(self.max_steps):
            trace.append(node)
            node, step_cost = self.steps[node](app, state)
            cost += step_cost
            if node in ("approve", "decline", "human_review"):
                trace.append(node)
                self.traces.append(" > ".join(trace))
                return node, cost
        self.traces.append(" > ".join(trace + ["BUDGET EXCEEDED -> human_review"]))
        return "human_review", cost                                    # never loop forever: fail safe to a human


forest = RandomForestClassifier(300, min_samples_leaf=20, n_jobs=-1, random_state=0).fit(hist[FEATS], hist["default"])
leaf_n = learned.tree_.n_node_samples
# score every live application once (a model service would be called per case; precomputing keeps the lab fast)
X_live = live[FEATS].fillna({"income_k": hist.income_k.median()})
P_TREE = pd.Series(learned.predict_proba(X_live)[:, 1], index=live.index)
LEAF_N = pd.Series(leaf_n[learned.apply(X_live)], index=live.index)
votes = np.array([t.predict_proba(X_live.to_numpy())[:, 1] for t in forest.estimators_])
P_FOREST, SPREAD = pd.Series(votes.mean(axis=0), index=live.index), pd.Series(votes.std(axis=0), index=live.index)


def make_steps(escalation: str):
    def intake(app, s):
        if app.age < 18:
            return "decline", 0.0
        return ("request_docs" if np.isnan(app.income_k) else "credit_report"), 0.0

    def request_docs(app, s):
        s["docs_requested"] = True                                        # 70% send documents; the rest go to a human
        return ("credit_report" if (hash(app.name) % 10) < 7 else "human_review"), TOOL_COST

    def credit_report(app, s):
        return "risk_model", TOOL_COST

    def risk_model(app, s):
        p = P_TREE[app.name]
        if escalation == "none":
            return ("approve" if p < break_even else "decline"), TOOL_COST
        if escalation == "leaf":                                           # tree confidence: near the line, or few past cases
            unsure = abs(p - break_even) < 0.06 or LEAF_N[app.name] < 200
        else:                                                              # forest confidence: near the line, or trees disagree
            p = P_FOREST[app.name]
            unsure = abs(p - break_even) < 0.05 or SPREAD[app.name] > 0.2
        if unsure:
            return "human_review", TOOL_COST
        return ("approve" if p < break_even else "decline"), TOOL_COST

    return {"intake": intake, "request_docs": request_docs, "credit_report": credit_report, "risk_model": risk_model}


rows, examples = {}, {}
for name, esc in [("learned decision, no escalation", "none"), ("escalate: borderline p or small leaf", "leaf"),
                  ("escalate: borderline p or forest disagreement", "forest")]:
    orch = Orchestrator(make_steps(esc))
    out = [orch.run(r) for _, r in live.iterrows()]
    actions = pd.Series([o[0] for o in out])
    tool_cost = np.mean([o[1] for o in out])
    m = money(actions, live)
    m["$ per application"] -= tool_cost
    m["avg tool cost $"] = tool_cost
    rows[name] = m
    examples[name] = orch.traces[:3]
orch_tab = pd.DataFrame(rows).T
print(orch_tab.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}"))
print("\n  Example traces (escalation on leaf confidence):")
for t_ in examples["escalate: borderline p or small leaf"]:
    print("   ", t_)
best = orch_tab["$ per application"].idxmax()
note(f"""The orchestrator runs every application through named steps (intake ->
documents -> credit report -> risk model -> decision), keeps a trace per case,
and has a step budget so a confused workflow fails SAFE (to a human) instead
of looping forever. Confidence-based escalation sends borderline cases (p
near break-even), cases from thinly populated leaves (few past examples), or
cases where the forest's trees disagree, to a human. Humans cost $25 but
catch most defaulters, so escalating the RIGHT cases pays (even a third): best policy
here: '{best}'. This is the 'human-in-the-loop' pattern of Module 8, with the
confidence signal coming straight from the model's structure.""")

step("3.2  Is the confidence signal real? The value of a human review, by distance from the line")
c_idx = complete.index
p_all, spread = P_FOREST[c_idx].to_numpy(), SPREAD[c_idx].to_numpy()
y_c = complete["default"].to_numpy()
auto_value = np.where(p_all < break_even, np.where(y_c == 1, BAD_LOAN, GOOD_LOAN), 0.0)
human_value = -HUMAN_COST + np.where(y_c == 1, (1 - HUMAN_CATCH) * BAD_LOAN, GOOD_LOAN)
gain = human_value - auto_value                                        # what a human review adds, case by case
dist = np.abs(p_all - break_even)
band = pd.qcut(dist, 5, labels=False)
bt = pd.DataFrame({"|p - break-even| up to": pd.Series(dist).groupby(band).max(),
                   "forest disagreement (sd)": pd.Series(spread).groupby(band).mean(),
                   "$ gained by a human review": pd.Series(gain).groupby(band).mean()})
bt.index = ["1 (closest to the line)", "2", "3", "4", "5 (farthest)"]
print(bt.round(3).to_string())
note(f"""Split the cases into five bands by distance from the break-even probability.
Close to the line, a $25 human review is worth {bt.iloc[0, 2]:+.0f} $ per case on average
(the reviewer catches defaults the model was unsure about). The value falls with distance, and in the farthest band review LOSES
money ({bt.iloc[4, 2]:+.0f} $ per case): humans mostly confirm the model there and pay $25 to do
it. Forest disagreement is a useful second signal (highest near the line, but
also high among the very riskiest cases, where trees disagree on HOW risky).
That is what makes escalation worth it: the confidence signal must PREDICT where human
attention pays. The same check applies to an LLM agent's self-reported
confidence, which is often NOT calibrated: measure it against labelled outcomes.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    o = orch_tab
    axes[0].barh(o.index, o["$ per application"], color=["#9d9d9d", "#4c78a8", "#54a24b"])
    axes[0].set(xlabel="$ per application (after tool and human costs)", title="Orchestration policies")
    axes[0].tick_params(axis="y", labelsize=8)
    axes[1].bar(bt.index, bt["$ gained by a human review"], color=["#54a24b" if v_ > 0 else "#e45756" for v_ in bt.iloc[:, 2]])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(xlabel="distance from the decision line (quintile)", ylabel="$ gained per case by a human review",
                title="Escalate where it pays: near the line")
    axes[1].tick_params(axis="x", labelsize=7)
    save_fig(fig, "13_agent_workflows")

banner("KEY TAKEAWAYS")
print("""  1. A workflow is a decision tree: branches are conditions, leaves are actions, the trace is the explanation.
  2. Keep hard rules (law, safety) as rules; use learned decisions for judgement calls.
  3. A shallow learned tree compiles to readable if/else code: learned AND auditable.
  4. A router agent is a classifier over agents; evaluate it like any classifier.
  5. Orchestrate with budgets and traces; escalate on calibrated confidence (leaf size, tree disagreement).""")
