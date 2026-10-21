"""
LAB 01 - Foundation Models and LLM APIs
=======================================

FROM TINYGPT TO A FOUNDATION MODEL
    In Module 7 we trained an 850k-parameter TinyGPT on an invented world. A FOUNDATION MODEL is the same idea
    at enormous scale: pretrained on a large share of the public internet, instruction-tuned, and then used for
    thousands of tasks it was never specifically trained for. You don't train it: you CALL it.

THE API IN ONE CALL (the shape of Anthropic's Messages API)
    resp = llm.create(system="who you are and how to behave",
                      messages=[{"role": "user", "content": "..."}, {"role": "assistant", ...}, ...],
                      max_tokens=1024)
    resp.text          the answer
    resp.stop_reason   "end_turn" (finished), "max_tokens" (cut off), "tool_use" (wants a tool: Lab 10)
    resp.usage         input_tokens, output_tokens: what you pay for
    resp.latency_ms    how long the user waited

THE BACKEND IN THESE LABS
    get_llm("large") / get_llm("small") return SimLLM: a deterministic, offline stand-in with the same interface,
    the same pricing structure and the failure modes real LLMs are known for (genai_toolkit.py lists every rate).
    Set the environment variable LLM_BACKEND=claude (and `pip install anthropic`, plus an API key) and the same
    code calls Claude (claude-opus-5-5 / claude-haiku-5-5) instead. Real calls cost real money.

LEVELS
    EASY   : One call, taken apart. The model has no memory: the conversation is whatever you send.
    MEDIUM : Large vs small models: same task, different price and speed. What a support assistant costs
             per month, and why the model knows "returns" but not YOUR return policy.
    HARD   : Production reality: rate limits and server errors, retries with exponential backoff, and what
             a 100% success rate can hide.
"""
# %%
import numpy as np
import pandas as pd

from genai_toolkit import PRICES, SimLLM, call_with_retry, count_tokens, get_llm
from ml_lab_utils import HAS_MPL, banner, make_kb, make_tickets, note, plt, save_fig, show, step

tickets = make_tickets()
kb = make_kb()
large, small = get_llm("large"), get_llm("small")

# %% =========================================================================
# LEVEL 1 (EASY): One call, taken apart
# ============================================================================
banner("One API call, taken apart", "EASY")
system = "You are a friendly support assistant for Lumora Home, a store that sells smart-home devices."
question = "Hi! What kinds of things can you help me with?"
resp = large.create(system=system, messages=[{"role": "user", "content": question}], max_tokens=300)
show("Model", resp.model)
show("Question", question)
show("Answer (resp.text)", resp.text)
show("Stop reason", resp.stop_reason)
show("Tokens in / out", f"{resp.usage['input_tokens']} / {resp.usage['output_tokens']}")
show("Latency", f"{resp.latency_ms:.0f} ms")
show("Cost of this call", f"${resp.cost_usd:.6f}")
note("""Every call has the same parts: a SYSTEM prompt (who the model is, how to
behave), a list of MESSAGES (the conversation so far), and limits such as
max_tokens. What comes back is text plus the bill: tokens in, tokens out,
and how long it took. Those three numbers drive everything in this module:
quality per dollar and per second.""")

step("1.1  max_tokens is a hard cut, not a summary")
cut = large.create(system=system, messages=[{"role": "user", "content": "How do I reset my password?"}], max_tokens=6)
show("max_tokens = 6", f"{cut.text!r}  (stop_reason = {cut.stop_reason})")
note("""When the answer reaches max_tokens the model simply stops mid-sentence and the
stop reason says so. Always check stop_reason: a truncated JSON object or a
half-written instruction looks fine until something downstream breaks.""")

step("1.2  The model has no memory: the conversation is what you send")
turn1 = [{"role": "user", "content": "Hi, my name is Kofi."}]
r1 = large.create(system=system, messages=turn1)
forgot = large.create(system=system, messages=[{"role": "user", "content": "What is my name?"}])
remembered = large.create(system=system, messages=turn1 + [r1.as_message(), {"role": "user", "content": "What is my name?"}])
show("Turn 1: 'Hi, my name is Kofi.'", r1.text)
show("Turn 2 sent ALONE: 'What is my name?'", forgot.text)
show("Turn 2 sent WITH the history", remembered.text)
show("Input tokens: alone vs with history", f"{forgot.usage['input_tokens']} vs {remembered.usage['input_tokens']}")
note("""The API is stateless. Each call starts from nothing; 'memory' in a chatbot is
the application re-sending the whole conversation every turn. That's why a
long chat gets slower and more expensive with every message: the history is
billed again as input tokens on each call. Agents (Part B) work the same way.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Large vs small, and what an assistant costs
# ============================================================================
banner("Large vs small: same task, different price and speed", "MEDIUM")
policy_q = "How long do I have to return something?"
rows = {}
for llm in [large, small]:
    r = llm.create(system=system, messages=[{"role": "user", "content": policy_q}])
    p_in, p_out = PRICES[llm.model]
    rows[llm.model] = {"answer": r.text, "$ per M input tokens": p_in, "$ per M output tokens": p_out,
                       "latency (ms)": round(r.latency_ms)}
print(pd.DataFrame(rows).T.to_string())
true_window = kb[kb.doc_id == "LM-KB-RET"].facts.iloc[0]["return_window"]
show("Lumora's actual return window (help centre)", true_window)
note(f"""Both models answer fluently, and both are WRONG for this store: they give a
typical return window from pretraining, while Lumora's policy is {true_window}.
A foundation model knows how returns work in general; it cannot know your
policy, your orders or today's prices. Giving it that knowledge at question
time is retrieval-augmented generation (Labs 05-08). Notice also the price
gap: the small tier is 40x cheaper per token and several times faster.""")

step("2.1  Where the tokens go in one support answer")
docs = kb[kb.doc_id.isin(["LM-KB-RET", "LM-KB-REF", "LM-KB-WAR"])]
context = "\n".join(f'<document id="{d.doc_id}" date="{d.updated}">{d.text}</document>' for d in docs.itertuples())
SUPPORT_SYSTEM = (system + " Answer only from the documents provided. Cite the document id in square brackets. "
                  "If the documents do not contain the answer, say you don't know. Keep answers short and friendly. "
                  "Never promise refunds or actions you cannot take. Escalate angry customers to a human.")
sample = tickets[tickets.intent == "policy_question"].text.iloc[:20]
parts = {"system prompt": count_tokens(SUPPORT_SYSTEM), "retrieved documents (3)": count_tokens(context),
         "customer ticket (average)": float(np.mean([count_tokens(t) for t in sample]))}
outs = [large.create(system=SUPPORT_SYSTEM, messages=[{"role": "user", "content": f"{context}\n\n<question>{t}</question>"}])
        for t in sample]
parts["answer (output, average)"] = float(np.mean([o.usage["output_tokens"] for o in outs]))
print(pd.Series(parts, name="tokens").round(0).to_string())
n_in = parts["system prompt"] + parts["retrieved documents (3)"] + parts["customer ticket (average)"]
n_out = parts["answer (output, average)"]
show("One answer: input / output tokens", f"{n_in:.0f} / {n_out:.0f}")
show("Example grounded answer", outs[0].text)

step("2.2  Cost at scale: 10,000 tickets a day")
rows = {}
for model in ["sim-large", "sim-small"]:
    p_in, p_out = PRICES[model]
    per = n_in * p_in / 1e6 + n_out * p_out / 1e6
    rows[model] = {"$ per answer": per, "input share of cost": n_in * p_in / 1e6 / per,
                   "$ per day (10k answers)": per * 10_000, "$ per month": per * 10_000 * 30}
for model, mult in [("sim-large, agent with 4 calls per ticket", 4), ("sim-small, agent with 4 calls per ticket", 4)]:
    base = rows[model.split(",")[0]]
    rows[model] = {k: (v * mult if k != "input share of cost" else v) for k, v in base.items()}
cost = pd.DataFrame(rows).T
print(cost.to_string(float_format=lambda v: f"{v:,.5f}" if v < 0.01 else f"{v:,.2f}"))
note(f"""Output tokens cost 5x more than input tokens, yet INPUT dominates the bill
({cost.iloc[0, 1]:.0%} of it): every call re-sends the system prompt, the retrieved
documents and the conversation. The large model costs ${cost.iloc[0, 3]:,.0f} a month at
10,000 answers a day; the small one ${cost.iloc[1, 3]:,.0f}. An agent that calls the model
4 times per ticket (Part B) multiplies all of it. Cost is a design decision:
shorter prompts, fewer documents, and a small model where it is good enough.""")

# %% =========================================================================
# LEVEL 3 (HARD): Rate limits, server errors and retries
# ============================================================================
banner("Production reality: errors, retries and what averages hide", "HARD")
flaky = SimLLM("sim-large", error_rate=0.20)
N = 400
naive_ok, errors = 0, {}
for i in range(N):
    try:
        flaky.create(system=system, messages=[{"role": "user", "content": sample.iloc[i % len(sample)]}])
        naive_ok += 1
    except Exception as e:
        errors[type(e).__name__] = errors.get(type(e).__name__, 0) + 1
show("Calls without retries: success rate", f"{naive_ok / N:.1%}")
show("Errors seen", errors)

flaky = SimLLM("sim-large", error_rate=0.20)
latency, waits, attempts, failed = [], [], [], 0
for i in range(N):
    log = []
    try:
        r = call_with_retry(lambda: flaky.create(system=system, messages=[{"role": "user", "content": sample.iloc[i % len(sample)]}]),
                            max_retries=4, base_delay=0.5, log=log)
        w = sum(x["wait_s"] for x in log)
        latency.append(r.latency_ms / 1000 + w)
        waits.append(w)
        attempts.append(len(log) + 1)
    except Exception:
        failed += 1
latency = np.array(latency)
show("With retries (exponential backoff): success rate", f"{1 - failed / N:.1%}")
show("Backoff schedule (seconds before retry 1, 2, 3, 4)", [round(0.5 * 2 ** a, 1) for a in range(4)])
print(pd.Series(attempts).value_counts().sort_index().rename_axis("attempts").rename("calls").to_frame().T.to_string())
lat = pd.Series({"p50": np.percentile(latency, 50), "p90": np.percentile(latency, 90),
                 "p99": np.percentile(latency, 99), "max": latency.max(), "mean": latency.mean()}, name="latency (s)")
print(lat.round(2).to_frame().T.to_string())
note(f"""With a 20% error rate, one call in five fails outright ({naive_ok / N:.0%} success). Retrying
with exponential backoff (wait 0.5 s, 1 s, 2 s, 4 s, plus random jitter so
clients don't retry in lockstep) brings success to {1 - failed / N:.0%}. Plot twist: the
success dashboard now looks perfect, while the slowest 1% of users wait
{lat['p99']:.1f} s instead of {lat['p50']:.1f} s. Averages hide this; report p90/p99 latency, and
count retries as a health signal. The Anthropic SDK retries 429 and 5xx
errors twice by default; in production, add a timeout and a fallback (a
smaller model, a cached answer, or 'a human will reply shortly').""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    pd.Series(parts).plot.barh(ax=axes[0], color=["C0", "C0", "C0", "C3"])
    axes[0].set(xlabel="tokens", title="One grounded answer: where the tokens go")
    monthly = cost["$ per month"][::-1]
    monthly.plot.barh(ax=axes[1], logx=True, color=["C2", "C0", "C2", "C0"])
    for i, v in enumerate(monthly):
        axes[1].text(v, i, f" ${v:,.0f}", va="center")
    axes[1].set(xlabel="$ per month (log scale)", xlim=(5, 30_000), title="10,000 tickets a day")
    axes[2].hist(latency, bins=40, color="C1")
    for q, ls in [(50, "-"), (99, "--")]:
        axes[2].axvline(np.percentile(latency, q), color="k", ls=ls, label=f"p{q} = {np.percentile(latency, q):.1f} s")
    axes[2].set(xlabel="seconds the user waited", title="20% server errors + retries: 100% success, long tail")
    axes[2].legend()
    save_fig(fig, "01_foundation_models_and_llm_apis")

banner("KEY TAKEAWAYS")
print("""  1. A foundation model is pretrained once and used for many tasks through an API: you call it, not train it.
  2. Every call = system prompt + messages in; text + tokens + latency out. Check stop_reason.
  3. The API is stateless: memory is the history you re-send, and you pay for it on every call.
  4. Input tokens usually dominate cost; model tier is a 40x price decision.
  5. It knows the world in general, not your store: that's what retrieval is for (Labs 05-08).
  6. Retries with backoff fix error rates, not tail latency: watch p99, not the average.""")
