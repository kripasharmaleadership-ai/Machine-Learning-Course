"""
LAB 11 - Planning, Task Decomposition, Reflection and Validation
================================================================

TWO WAYS AN AGENT FAILS ON ITS OWN
    1. It does PART of the job. "Refund my broken doorbell. Also, how long do refunds take?" gets a refund
       and silence about the second question. The fix is PLANNING: break the message into tasks first,
       then do each one (plan-and-execute), instead of improvising step by step.
    2. It says it did something it didn't. The payment service times out, and the reply still says
       "I've processed your refund". The fix is REFLECTION and VALIDATION: check the draft against what
       actually happened before it reaches the customer.

THE TOOLS
    planner     one extra model call: "List each request in the message" -> numbered sub-tasks
    verifier    one extra model call: <draft> + <evidence> (the tool log) -> {"ok": ..., "problems": [...]}
    validator   plain code: does every action the reply claims have a successful tool call?

LEVELS
    EASY   : Multi-part tickets: a prompt instruction vs an explicit plan-and-execute loop.
    MEDIUM : A flaky payment service and claimed-but-not-taken refunds: prompt, LLM verifier, code check.
    HARD   : A critic without evidence, and fixing the cause instead of the symptom.
"""
# %%
import json
import re

import numpy as np
import pandas as pd

from genai_toolkit import AgentRun, StoreTools, call_with_retry, get_llm, grade_run, run_agent
from ml_lab_utils import HAS_MPL, banner, make_tickets, note, plt, save_fig, show, step

tickets = make_tickets()
large, small = get_llm("large"), get_llm("small")
BASE = ("You are the customer support agent for Lumora Home. Use the tools to look things up and to act. Only take "
        "actions the customer explicitly asked for. If a tool returns an error, do not retry the same call; explain "
        "the problem.")
EVERY = " Handle every request in the message."
CONFIRMED = " Only report actions that are confirmed by a tool result."


def merge(runs) -> AgentRun:
    out = AgentRun(final_text=" ".join(r.final_text for r in runs), stop="final")
    for r in runs:
        out.calls += r.calls
        out.steps += r.steps
        out.cost_usd += r.cost_usd
        out.latency_ms += r.latency_ms
    return out


def plan_and_execute(llm, system, text, tools_fn):
    """1 planning call -> one agent run per sub-task (each sees the whole ticket for context) -> merged reply."""
    r = llm.create(system="You plan work for a support agent.",
                   messages=[{"role": "user", "content": f"List each request in this message as a numbered list, one "
                                                          f"line per request.\n<ticket>{text}</ticket>"}])
    tasks = [re.sub(r"^\d+\.\s*(\[\w+\]\s*)?", "", line).strip() for line in r.text.splitlines() if line.strip()]
    runs = [run_agent(llm, system, task, tools_fn().tools()) for task in tasks or [text]]
    out = merge(runs)
    out.steps += 1
    out.cost_usd += r.cost_usd
    out.latency_ms += r.latency_ms
    return out, tasks


# %% =========================================================================
# LEVEL 1 (EASY): Planning and task decomposition
# ============================================================================
banner("Multi-part tickets: improvise, instruct, or plan?", "EASY")
multi = tickets[tickets.intents.map(len) > 1].reset_index(drop=True)
single = tickets[(tickets.intents.map(len) == 1) & (tickets.split == "test")].sample(60, random_state=0).reset_index(drop=True)
show("Tickets with two requests", f"{len(multi)}, e.g. {multi.text.iloc[0]!r}")
plan, tasks = plan_and_execute(large, BASE + CONFIRMED, multi.text.iloc[0], StoreTools)
show("The planner's sub-tasks", tasks)
show("Merged reply", plan.final_text[:200])

rows, both_found = {}, {}
for model in [large, small]:
    for name, fn in [("improvise (base prompt)", lambda t, m=model: run_agent(m, BASE + CONFIRMED, t, StoreTools().tools())),
                     ("+ 'handle every request'", lambda t, m=model: run_agent(m, BASE + EVERY + CONFIRMED, t, StoreTools().tools())),
                     ("plan, then execute each", lambda t, m=model: plan_and_execute(m, BASE + CONFIRMED, t, StoreTools)[0])]:
        g_multi = pd.DataFrame([grade_run(fn(t["text"]), t) for t in multi.to_dict("records")])
        g_single = pd.DataFrame([grade_run(fn(t["text"]), t) for t in single.to_dict("records")])
        rows[(model.model, name)] = {"multi-part: all facts in reply": (g_multi.fact_recall == 1).mean(),
                                     "multi-part: resolved": g_multi.resolved.mean(),
                                     "single: resolved": g_single.resolved.mean(),
                                     "model calls per ticket": pd.concat([g_multi, g_single]).steps.mean(),
                                     "$ per 1,000 tickets": pd.concat([g_multi, g_single]).cost_usd.mean() * 1000}
    both_found[model.model] = np.mean([len(plan_and_execute(model, BASE + CONFIRMED, t, StoreTools)[1]) >= 2 for t in multi.text])
planning = pd.DataFrame(rows).T
planning.index.names = ["model", "approach"]
print(planning.round(3).to_string())
show("Plans that listed both requests (large / small)", f"{both_found['sim-large']:.0%} / {both_found['sim-small']:.0%}")
s = planning.xs("sim-small")
note(f"""Improvising, the small model answers both parts of a two-part ticket only
{s.iloc[0, 0]:.0%} of the time: it handles the first request and forgets the second. One
sentence in the prompt ('handle every request') raises that to {s.iloc[1, 0]:.0%}. Planning
first, then running one focused task at a time, reaches {s.iloc[2, 0]:.0%}: the plan is a list
the code walks through, so a task on the list can't be forgotten halfway. What
remains: the planner itself missed a request in {1 - both_found['sim-small']:.0%} of plans, and some tasks
fail like any single ticket. The price: a planning call plus one agent run per
task, {s.iloc[2, 3]:.1f} model calls per ticket instead of {s.iloc[0, 3]:.1f}. Plan when tasks are separable
and forgetting one is costly; let simple tickets go straight to the agent.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Reflection and validation
# ============================================================================
banner("A flaky payment service and refunds that never happened", "MEDIUM")


class FlakyPayments(StoreTools):
    """The payment provider times out on 30% of refund requests (nothing is refunded when it does)."""

    def __init__(self, fail_rate=0.3, seed=0, **kw):
        super().__init__(**kw)
        self.fail_rate, self.rng = fail_rate, np.random.default_rng(seed)

    def issue_refund(self, order_id, amount, reason=""):
        if self.rng.random() < self.fail_rate:
            raise TimeoutError("payment service did not respond; the refund was NOT issued")
        return super().issue_refund(order_id, amount, reason)


refunds = tickets[tickets.gold_actions.map(lambda g: any(a == "issue_refund" for a, _ in g))].reset_index(drop=True)
REPEATS = 5             # each ticket 5 times: a different model sample (seed) and a different failure pattern each time
show("Tickets whose correct outcome is a refund", f"{len(refunds)}, each run {REPEATS} times = {len(refunds) * REPEATS} runs")
CLAIM = re.compile(r"i('ve| have) (issued|processed).{0,20}refund|refund of \$[\d.]+ has been (issued|processed)", re.I)
FALLBACK = "I couldn't confirm the refund just now; a colleague will check it and email you within one business day."


def evidence_of(run):
    return "\n".join(json.dumps({"tool": c["name"], "input": c["input"], "ok": c["ok"], "result": c["result"]}, default=str)
                     for c in run.calls)


def verify_with_llm(llm, run, with_evidence=True):
    """Reflection: a second model call checks the draft against the tool log."""
    ev = f"<evidence>{evidence_of(run)}</evidence>" if with_evidence else ""
    r = llm.create(messages=[{"role": "user", "content": "Check this support reply before it is sent. Does it claim any "
                              "action or amount the evidence doesn't support? Reply in JSON: {\"ok\": true/false, "
                              f"\"problems\": [...]}}\n<draft>{run.final_text}</draft>\n{ev}"}])
    return json.loads(r.text), r.cost_usd


def validate_in_code(run):
    """Validation: a claimed refund needs a successful issue_refund call. No model involved."""
    return not (CLAIM.search(run.final_text) and not any(c["name"] == "issue_refund" and c["ok"] for c in run.calls))


def fix(run):
    run.final_text = " ".join(FALLBACK if CLAIM.search(s) else s for s in re.split(r"(?<=[.!?])\s+", run.final_text))
    return run


def why_no_refund(run):
    fails = [c for c in run.calls if c["name"] == "issue_refund" and not c["ok"]]
    if any(c["name"] == "issue_refund" and c["ok"] for c in run.calls):
        return "refund went through"
    if not fails:
        return "no refund attempted"
    return "payment timeout" if "payment service" in str(fails[-1]["result"]) else "refund rejected (e.g. wrong order id)"


def evaluate(system, check=None, tier="small", df=refunds, tools=lambda i: FlakyPayments(seed=i)):
    rows = []
    for i, t in enumerate(df.to_dict("records") * REPEATS):
        llm = get_llm(tier, seed=i // len(df))                       # repeat k = model sample k
        run = run_agent(llm, system, t["text"], tools(i).tools())
        cause = why_no_refund(run)
        flagged, extra = False, 0.0
        if check == "llm verifier":
            v, extra = verify_with_llm(llm, run)
            flagged = not v["ok"]
        elif check == "llm verifier, no evidence":
            v, extra = verify_with_llm(llm, run, with_evidence=False)
            flagged = not v["ok"]
        elif check == "code check":
            flagged = not validate_in_code(run)
        truly_false = bool(CLAIM.search(run.final_text)) and not validate_in_code(run)
        if flagged:
            run = fix(run)
        g = grade_run(run, t)
        rows.append({**g, "flagged": flagged, "false alarm": flagged and not truly_false, "extra $": extra, "cause": cause,
                     "lied": truly_false})
    df_ = pd.DataFrame(rows)
    evaluate.last = df_
    return {"claims a refund that didn't happen": df_.hallucinated_action.mean(), "resolved": df_.resolved.mean(),
            "replies flagged": df_.flagged.mean(), "false alarms": df_["false alarm"].mean(),
            "payment timeouts": (df_.cause == "payment timeout").mean(),
            "$ per 1,000 tickets": (df_.cost_usd.sum() + df_["extra $"].sum()) / len(df_) * 1000}


rows = {"no check": evaluate(BASE),
        "+ prompt: 'only report confirmed actions'": evaluate(BASE + CONFIRMED),
        "+ LLM verifier with the tool log": evaluate(BASE + CONFIRMED, "llm verifier"),
        "+ validation in code": evaluate(BASE + CONFIRMED, "code check")}
refl = pd.DataFrame(rows).T
print(refl.round(3).to_string())
run_ex = next(r for r in (run_agent(small, BASE, t, FlakyPayments(seed=i).tools()) for i, t in enumerate(refunds.text))
              if not validate_in_code(r))
show("A reply that lies", run_ex.final_text)
show("  what the tool log says", [(c["name"], "OK" if c["ok"] else c["result"][:40]) for c in run_ex.calls])
show("  the verifier's verdict", verify_with_llm(small, run_ex)[0])
note(f"""With a 30% payment timeout and no checks, the small model tells {refl.iloc[0, 0]:.0%} of these
customers their refund is done when it isn't: the worst kind of answer,
because nobody will look into it. One prompt sentence cuts that to {refl.iloc[1, 0]:.0%}. A
verifier call that compares the draft with the TOOL LOG catches the rest ({refl.iloc[2, 0]:.0%}),
and so does a few lines of code ({refl.iloc[3, 0]:.0%}) at no extra cost: when the claim is
something you can check mechanically, check it mechanically. The fallback
reply is honest ('a colleague will check'), so the customer isn't misled.""")

# %% =========================================================================
# LEVEL 3 (HARD): Reflection needs evidence; fix the cause
# ============================================================================
banner("A critic without evidence, and fixing the cause", "HARD")
step("3.1  'Are you sure?' without the tool log")
rows = {"LLM verifier WITH the tool log": refl.iloc[2].to_dict(),
        "LLM verifier, draft only (no evidence)": evaluate(BASE + CONFIRMED, "llm verifier, no evidence")}
crit = pd.DataFrame(rows).T
print(crit.round(3).to_string())
note(f"""Show the critic only the draft and it can't know what happened: it flags
{crit.iloc[1, 2]:.0%} of replies, and {crit.iloc[1, 3] / max(crit.iloc[1, 2], 1e-9):.0%} of those flags are false alarms on refunds that DID go
through. Each false alarm replaces a correct answer with 'a colleague will
check', so resolution falls ({crit.iloc[0, 1]:.0%} -> {crit.iloc[1, 1]:.0%}). Self-reflection is only as good as
the evidence it's given; 'are you sure?' with nothing to check against is a
smoke alarm that goes off at every meal, and people learn to ignore it.""")

step("3.2  Plot twist: make the payment call reliable instead")


class RetryingPayments(FlakyPayments):
    """Retry timeouts in the TOOL, with backoff (Lab 01): the model never sees a transient failure."""

    def issue_refund(self, order_id, amount, reason=""):
        def attempt():
            try:
                return FlakyPayments.issue_refund(self, order_id, amount, reason)
            except TimeoutError as e:
                from genai_toolkit import OverloadedError
                raise OverloadedError(str(e)) from e
        return call_with_retry(attempt, max_retries=3)


rows = {"flaky tool + prompt + code check": refl.iloc[3].to_dict(),
        "tool retries timeouts itself + prompt": evaluate(BASE + CONFIRMED, tools=lambda i: RetryingPayments(seed=i))}
retry_runs = evaluate.last
rows["tool retries + prompt + code check"] = evaluate(BASE + CONFIRMED, "code check", tools=lambda i: RetryingPayments(seed=i))
cause = pd.DataFrame(rows).T
print(cause.round(3).to_string())
lies = retry_runs[retry_runs.lied].cause.value_counts()
show("With retries: why the remaining false claims had no refund", lies.to_dict() if len(lies) else "none")
note(f"""Every check so far treats the SYMPTOM: the refund failed and we caught the
lie. Retry the timeout inside the tool (a transient error should never reach
the model) and payment timeouts drop from {refl.iloc[1, 4]:.0%} to {cause.iloc[1, 4]:.0%} of runs: resolved {cause.iloc[0, 1]:.0%} -> {cause.iloc[1, 1]:.0%}.
Without a code check, {cause.iloc[1, 0]:.0%} of replies still claim a refund that didn't happen: the
refunds that failed for OTHER reasons (see the line above) still invite the
same lie. So keep the free code check as well. Reliability engineering beats
clever prompting: fix the tools, then verify what's left.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    order = ["improvise (base prompt)", "+ 'handle every request'", "plan, then execute each"]
    planning["multi-part: all facts in reply"].unstack(0).loc[order].plot.bar(ax=axes[0], rot=0)
    axes[0].set_xticks(range(3), ["improvise", "+ 'every\nrequest'", "plan, then\nexecute"])
    axes[0].set(ylim=(0, 1.05), xlabel="", title="Two-part tickets: both answered")
    axes[1].barh(["no check", "+ prompt", "+ LLM verifier\n(tool log)", "+ code check"], refl["claims a refund that didn't happen"],
                 color=["C3", "C1", "C2", "C2"])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="share of replies", title="Claimed refunds that never happened")
    last = pd.concat([crit[["resolved", "false alarms"]], cause[["resolved", "false alarms"]].iloc[1:]])
    last.index = ["verifier + tool log", "verifier, draft only", "tool retries", "tool retries\n+ code check"]
    last.plot.barh(ax=axes[2])
    axes[2].invert_yaxis()
    axes[2].set(xlim=(0, 1), title="Evidence matters; fixing the cause matters more")
    axes[2].legend(fontsize=8)
    save_fig(fig, "11_planning_reflection_validation")

banner("KEY TAKEAWAYS")
print("""  1. Agents improvising step by step drop parts of multi-part requests; planning first fixes most of it.
  2. Plan-and-execute costs extra calls: use it where tasks are separable and forgetting is costly.
  3. Models can claim actions that never happened; check replies against the tool log before sending.
  4. If a claim can be checked mechanically, check it in code; use an LLM critic for what code can't judge.
  5. A critic without evidence raises false alarms; and the best fix is often a more reliable tool.""")
