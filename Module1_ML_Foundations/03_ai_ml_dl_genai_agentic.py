"""
LAB 03 - AI, ML, Deep Learning, Generative AI and Agentic AI
=============================================================

THE NESTED CIRCLES (memorize this picture)

    +--------------------------------------------------------------------+
    | ARTIFICIAL INTELLIGENCE  - any technique that makes machines act   |
    |   smart (includes hand-written rules, search, expert systems)      |
    |   +--------------------------------------------------------------+ |
    |   | MACHINE LEARNING - learns patterns from data                 | |
    |   |   +--------------------------------------------------------+ | |
    |   |   | DEEP LEARNING - ML with multi-layer neural networks    | | |
    |   |   |   +--------------------------------------------------+ | | |
    |   |   |   | GENERATIVE AI - DL models that CREATE new content| | | |
    |   |   |   +--------------------------------------------------+ | | |
    |   |   +--------------------------------------------------------+ | |
    |   +--------------------------------------------------------------+ |
    +--------------------------------------------------------------------+
    AGENTIC AI is not another circle - it is an ARCHITECTURE: a model (usually
    GenAI) placed in a loop that can PLAN, USE TOOLS, OBSERVE results and ACT
    towards a goal, with guardrails.

LEVELS IN THIS LAB (all on business problems)
    EASY   : AI without ML (expert rules) vs ML on the same loan-approval problem.
    MEDIUM : Deep learning beats linear ML on non-linear data;
             GenAI in miniature: a model that GENERATES new handwritten digits.
    HARD   : Agentic AI - a retention agent that plans, calls tools (including an
             ML model), respects a budget, asks a human for approval, and keeps
             an audit log. Enterprise-style, no API key needed.
"""
# %%
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_digits, make_circles
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_lab_utils import (
    RANDOM_STATE, TELCO_CATEGORICAL, TELCO_NUMERIC, TELCO_TARGET,
    banner, make_telco_churn, note, show, step,
)

rng = np.random.default_rng(RANDOM_STATE)

# %% =========================================================================
# LEVEL 1 (EASY): AI WITHOUT ML vs ML - loan approval
# ============================================================================
banner("AI without ML (expert rules) vs Machine Learning", "EASY")

# Simulated loan history. Hidden truth: default risk depends on a NON-obvious
# combination - debt ratio matters much more for short employment history.
n = 4000
loans = pd.DataFrame({
    "credit_score": rng.normal(680, 60, n).clip(450, 850).round(),
    "debt_ratio": rng.beta(2, 5, n).round(3),
    "years_employed": rng.exponential(5, n).clip(0, 40).round(1),
    "income_k": rng.lognormal(4.1, 0.4, n).round(1),
})
logit = (-1.0 - 0.012 * (loans.credit_score - 680) + 4.0 * loans.debt_ratio
         - 0.15 * loans.years_employed + 3.0 * loans.debt_ratio * (loans.years_employed < 2)
         - 0.01 * (loans.income_k - 60))
loans["defaulted"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
show("Historical loans / default rate", f"{n} / {loans.defaulted.mean():.1%}")

train, test = train_test_split(loans, test_size=0.3, random_state=RANDOM_STATE, stratify=loans.defaulted)
features = ["credit_score", "debt_ratio", "years_employed", "income_k"]

step("AI without ML: an 'expert system' written by a credit officer")


def expert_system_risk(row) -> float:
    """Symbolic AI (1980s style): knowledge typed in as IF/THEN rules.
    It IS artificial intelligence - but it never learns."""
    risk = 0.2
    if row.credit_score < 620:
        risk += 0.3
    if row.debt_ratio > 0.45:
        risk += 0.3
    if row.years_employed < 1:
        risk += 0.1
    return min(risk, 1.0)


expert_scores = test.apply(expert_system_risk, axis=1)
show("Expert system ROC-AUC", roc_auc_score(test.defaulted, expert_scores))

step("Machine learning: learn the risk function from 2,800 past loans")
ml = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
ml.fit(train[features], train.defaulted)
show("Logistic regression ROC-AUC", roc_auc_score(test.defaulted, ml.predict_proba(test[features])[:, 1]))
note("""ROC-AUC = how well the model RANKS risky borrowers above safe ones
(0.5 = coin flip, 1.0 = perfect). Both are 'AI'. The ML model wins because it
weighs every variable using evidence, while the expert's thresholds are guesses.
Expert systems are still used where rules are LAW (compliance, eligibility).""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Deep learning and Generative AI
# ============================================================================
banner("Deep learning: when straight lines are not enough", "MEDIUM")

X_c, y_c = make_circles(n_samples=2000, noise=0.12, factor=0.5, random_state=RANDOM_STATE)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_c, y_c, test_size=0.3, random_state=RANDOM_STATE)

linear = LogisticRegression().fit(Xc_tr, yc_tr)
neural = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=RANDOM_STATE).fit(Xc_tr, yc_tr)
show("Linear model accuracy (circles)", accuracy_score(yc_te, linear.predict(Xc_te)))
show("Neural net (2 hidden layers) accuracy", accuracy_score(yc_te, neural.predict(Xc_te)))
note("""Data: an inner ring vs outer ring. A linear model can only draw a straight
line -> ~50%. A neural network stacks layers that LEARN new features (here:
something like 'distance from the centre') automatically. That automatic
feature learning is what 'deep' learning buys you - essential for images,
audio, text. For tabular business data, classic ML is often just as good.""")

banner("Generative AI in miniature: generate NEW handwritten digits", "MEDIUM")
digits = load_digits()
X_d = digits.data  # 1,797 images, 8x8 pixels, values 0..16

step("Step 1: Learn the DISTRIBUTION of what digits look like")
pca = PCA(n_components=20, random_state=RANDOM_STATE).fit(X_d)    # compress 64 -> 20 numbers
gmm = GaussianMixture(n_components=40, covariance_type="full", random_state=RANDOM_STATE)
gmm.fit(pca.transform(X_d))
note("""A discriminative model learns P(label | image). A GENERATIVE model learns
P(image) - what plausible images look like - so it can SAMPLE new ones.""")

step("Step 2: Sample brand-new digits (not copies of the training set)")
samples_latent, _ = gmm.sample(6)
new_images = np.clip(pca.inverse_transform(samples_latent), 0, 16)

# A classifier acts as a 'judge' to check generated images look like real digits
judge = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X_d, digits.target)
probs = judge.predict_proba(new_images)


def ascii_digit(img: np.ndarray) -> list[str]:
    shades = " .:-=+*#%@"
    grid = img.reshape(8, 8)
    return ["".join(shades[int(v / 16 * 9)] * 2 for v in row) for row in grid]


rendered = [ascii_digit(img) for img in new_images]
for r in range(8):
    print("   " + "    ".join(img[r] for img in rendered))
print("   " + "    ".join(f"judge:{p.argmax()} ({p.max():.0%})".ljust(16) for p in probs))
nearest = np.min(((X_d[None, :, :] - new_images[:, None, :]) ** 2).sum(-1), axis=1)
show("Min distance to any training image", f"{nearest.min():.0f} (0 would mean an exact copy)")
note("""These images did not exist before. Same principle as image generators
(diffusion models) and LLMs: learn the data distribution, then sample from it.
Real GenAI uses deep neural networks with billions of parameters instead of
our 20-dimensional Gaussian mixture.""")

# %% =========================================================================
# LEVEL 3 (HARD): Agentic AI - a customer-retention agent with guardrails
# ============================================================================
banner("Agentic AI: plan -> use tools -> observe -> act, with guardrails", "HARD")
note("""An AGENT = a decision-maker in a loop with TOOLS and a GOAL.
  ML model   : 'This customer has a 72% churn risk.'                 (predicts)
  GenAI      : 'Here is a friendly retention email.'                  (creates)
  Agentic AI : 'I checked 5 customers, scored them, chose offers within
                budget, got approval for the expensive ones, sent them,
                and logged everything.'                                 (acts)
In production the 'brain' is an LLM choosing tools via function-calling.
Here the brain is a transparent rule-based planner so you can see every
step - the architecture (tools, memory, guardrails, audit) is identical.""", keep_lines=True)

# ---- Tool 0: an ML model the agent can call --------------------------------
telco = make_telco_churn()
churn_model = make_pipeline(
    ColumnTransformer([
        ("num", StandardScaler(), TELCO_NUMERIC[:-2] + ["support_tickets_90d"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), TELCO_CATEGORICAL),
    ]),
    LogisticRegression(max_iter=2000),
).fit(telco, telco[TELCO_TARGET])
CUSTOMER_DB = telco.set_index("customer_id")

# ---- Tools: the ONLY actions the agent is allowed to take ------------------
OFFERS = {
    "none": {"cost": 0, "save_rate": 0.00},
    "loyalty_discount_10pct": {"cost": 15, "save_rate": 0.20},
    "free_tech_support_6m": {"cost": 30, "save_rate": 0.30},
    "contract_upgrade_bonus": {"cost": 60, "save_rate": 0.45},
}
SENT_OFFERS: list[dict] = []


def tool_get_customer(customer_id: str) -> dict:
    row = CUSTOMER_DB.loc[customer_id]
    return {"customer_id": customer_id, **row[TELCO_NUMERIC + TELCO_CATEGORICAL].to_dict()}


def tool_predict_churn(customer: dict) -> float:
    return float(churn_model.predict_proba(pd.DataFrame([customer]))[0, 1])


def tool_estimate_clv(customer: dict) -> float:
    """Customer lifetime value proxy: 12 months of revenue."""
    return 12 * customer["monthly_charges"]


def tool_send_offer(customer_id: str, offer: str) -> str:
    SENT_OFFERS.append({"customer_id": customer_id, "offer": offer})
    return f"offer '{offer}' sent to {customer_id}"


def human_approval(customer_id: str, offer: str, cost: float) -> bool:
    """Human-in-the-loop. In production: a Slack/Teams approval card or a ticket.
    Here we simulate a manager who approves expensive offers only for high-value
    customers."""
    clv = tool_estimate_clv(tool_get_customer(customer_id))
    decision = clv > 900
    print(f"      [HUMAN] approve {offer} (${cost}) for {customer_id} (CLV ${clv:.0f})? -> {'YES' if decision else 'NO'}")
    return decision


TOOLS = {  # tool registry = allowlist. The agent cannot call anything else.
    "get_customer": tool_get_customer,
    "predict_churn": tool_predict_churn,
    "estimate_clv": tool_estimate_clv,
    "send_offer": tool_send_offer,
}


# ---- The agent ---------------------------------------------------------------
class RetentionAgent:
    """Minimal but production-shaped agent: goal, memory, planner, tool executor,
    guardrails (budget, max steps, approval threshold, tool allowlist), audit log."""

    def __init__(self, budget: float, max_steps: int = 50, approval_threshold: float = 50):
        self.budget = budget
        self.max_steps = max_steps
        self.approval_threshold = approval_threshold
        self.memory: dict[str, dict] = {}
        self.audit_log: list[dict] = []
        self.steps = 0
        self.expected_value = 0.0

    def call_tool(self, name: str, **kwargs):
        if name not in TOOLS:
            raise PermissionError(f"tool '{name}' is not allowlisted")
        self.steps += 1
        if self.steps > self.max_steps:
            raise RuntimeError("guardrail: max_steps exceeded - stopping agent")
        result = TOOLS[name](**kwargs)
        self.audit_log.append({"step": self.steps, "tool": name,
                               "args": {k: (v if isinstance(v, str) else "...") for k, v in kwargs.items()},
                               "result": result if isinstance(result, (str, float)) else "dict"})
        return result

    def plan_offer(self, churn_p: float, clv: float, max_cost: float | None = None) -> tuple[str, float]:
        """THINK step. Choose the offer with the best expected profit:
            expected_value = P(churn) * save_rate * CLV - cost
        An LLM-based agent would reason about this in natural language;
        we make the economics explicit and testable."""
        limit = self.budget if max_cost is None else min(self.budget, max_cost)
        best, best_value = "none", 0.0
        for offer, o in OFFERS.items():
            value = churn_p * o["save_rate"] * clv - o["cost"]
            if value > best_value and o["cost"] <= limit:
                best, best_value = offer, value
        return best, best_value

    def run(self, goal: str, customer_ids: list[str]) -> None:
        print(f"\n  GOAL: {goal}\n  Budget: ${self.budget}\n")
        for cid in customer_ids:
            # 1. OBSERVE - gather facts with tools
            customer = self.call_tool("get_customer", customer_id=cid)
            churn_p = self.call_tool("predict_churn", customer=customer)
            clv = self.call_tool("estimate_clv", customer=customer)
            self.memory[cid] = {"churn_p": churn_p, "clv": clv}
            # 2. THINK - decide what to do
            offer, value = self.plan_offer(churn_p, clv)
            print(f"    {cid}: churn={churn_p:.0%} CLV=${clv:,.0f} -> plan: {offer} (expected value ${value:,.0f})")
            if offer == "none":
                reason = "budget exhausted" if self.budget < min(o["cost"] for o in OFFERS.values() if o["cost"]) \
                    else "no offer pays for itself"
                print(f"      skip: {reason}")
                continue
            # 3. GUARDRAIL - expensive actions need a human
            if OFFERS[offer]["cost"] > self.approval_threshold and \
                    not human_approval(cid, offer, OFFERS[offer]["cost"]):
                offer, value = self.plan_offer(churn_p, clv, max_cost=self.approval_threshold)
                print(f"      re-plan within approval limit -> {offer} (expected value ${value:,.0f})")
                if offer == "none":
                    continue
            # 4. ACT
            print("      ACT:", self.call_tool("send_offer", customer_id=cid, offer=offer))
            self.budget -= OFFERS[offer]["cost"]
            self.expected_value += value
        print(f"\n  Remaining budget: ${self.budget} | tool calls: {self.steps} | offers sent: {len(SENT_OFFERS)}")
        print(f"  Total expected value of actions: ${self.expected_value:,.0f}")


# Today's work queue: a realistic mix of customers
scores = churn_model.predict_proba(telco)[:, 1]
clv_all = 12 * telco.monthly_charges
pick = lambda mask, k: list(telco.customer_id[mask].head(k))   # noqa: E731
candidate_ids = (
    pick((scores > 0.7) & (clv_all > 1000), 1)      # risky + valuable
    + pick(scores < 0.08, 1)                         # safe customer
    + pick((scores > 0.7) & (clv_all < 700), 1)      # risky but low value
    + pick((scores > 0.7) & (clv_all > 1000), 3)[1:] # more risky + valuable
    + pick((scores > 0.4) & (scores < 0.5), 1)       # medium risk
)

agent = RetentionAgent(budget=150, approval_threshold=50)
agent.run("Reduce churn among today's flagged customers, max $150 spend", candidate_ids)

note("""Look at the 88%-risk, high-value customer: SKIPPED because the budget was
spent on earlier, less valuable customers. A greedy one-by-one agent is a
classic agent failure mode. Fix: observe everything first, THEN plan.""")

step("Agent v2: observe ALL customers first, then spend on highest expected value")
SENT_OFFERS.clear()
agent_v2 = RetentionAgent(budget=150, approval_threshold=50)
ranked = sorted(
    candidate_ids,
    key=lambda c: -agent_v2.plan_offer(
        tool_predict_churn(tool_get_customer(c)), tool_estimate_clv(tool_get_customer(c)))[1],
)
agent_v2.run("Same goal, but prioritise by expected value", ranked)
note("""Same budget, same tools - better decisions because the PLAN changed.
This is why agent design (planning strategy, memory) matters as much as
the model inside it.""")

step("Audit log (first 8 entries) - every action is traceable")
for entry in agent_v2.audit_log[:8]:
    if isinstance(entry["result"], float):
        entry["result"] = round(entry["result"], 3)
    print("   ", json.dumps(entry))
note("""Enterprise agent checklist we just implemented:
 (1) tool ALLOWLIST - the agent cannot invent actions
 (2) BUDGET and MAX-STEPS limits - no runaway loops or runaway spend
 (3) HUMAN-IN-THE-LOOP approval for high-impact actions
 (4) AUDIT LOG of every tool call - required for compliance and debugging
 (5) the ML model is a TOOL - agents orchestrate models, they don't replace them""",
     keep_lines=True)

banner("KEY TAKEAWAYS")
print("""  AI          : any machine 'intelligence' - even hand-written rules.
  ML          : learns rules from data (our loan model beat the expert rules).
  Deep learning: neural networks that learn their own features (circles, images).
  GenAI       : learns the data distribution and SAMPLES new content (new digits, text).
  Agentic AI  : a goal-driven loop that plans, calls tools (incl. ML models), and acts
                - safe only with allowlists, budgets, approvals and audit logs.""")
