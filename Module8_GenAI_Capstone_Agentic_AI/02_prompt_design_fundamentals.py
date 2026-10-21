"""
LAB 02 - Prompt Design Fundamentals
===================================

THE TASK FOR LABS 02-04
    Every support ticket must be routed: is it about an order's status, a return, a cancellation, an address
    change, a policy question, a broken device, the customer's account, a complaint, or something off-topic?
    Nine intents, 400 tickets with known answers (make_tickets), split into dev (to design prompts on) and
    test (to check them once at the end). A prompt is code: we test it like code.

THE ANATOMY OF A PROMPT
    role / system   who the model is and how it should behave
    task            what to do, stated plainly
    labels          the allowed answers, ideally with a one-line description each
    output format   exactly what to return ("only the label"), so code can parse it
    delimiters      <ticket> ... </ticket> separates YOUR instructions from THE CUSTOMER'S text
    (examples)      Lab 03

LEVELS
    EASY   : Five versions of the same prompt, from a bare question to a structured prompt. Accuracy AND
             parse rate: an answer your code can't parse is a wrong answer.
    MEDIUM : What label descriptions fix (a confusion matrix), and what temperature does to consistency.
    HARD   : Is the prompt still good on unseen tickets? What's left in the errors? And what is a good
             prompt worth in dollars?
"""
# %%
import re

import numpy as np
import pandas as pd

from genai_toolkit import get_llm
from ml_lab_utils import HAS_MPL, INTENTS, banner, make_tickets, note, plt, save_fig, show, step

tickets = make_tickets()
dev = tickets[tickets.split == "dev"].reset_index(drop=True)
test = tickets[tickets.split == "test"].reset_index(drop=True)
LABELS = list(INTENTS)
large, small = get_llm("large"), get_llm("small")


def parse_strict(text: str):
    """The answer must BE a label (case and spaces forgiven)."""
    s = text.strip().lower().replace(" ", "_")
    return s if s in LABELS else None


def parse_lenient(text: str):
    """Dig a label out of chatty answers ('Category: Return_Request', 'This looks like a cancel order request.')."""
    s = text.lower().replace("_", " ")
    hits = [lab for lab in LABELS if lab.replace("_", " ") in s]
    return max(hits, key=len) if hits else None


def evaluate_prompt(llm, system, user_template, df, parser=parse_strict, temperature=0.0):
    """Run a prompt over a set of tickets. Returns the scores and the predictions."""
    raw, preds, cost, tok = [], [], 0.0, 0
    for text in df.text:
        r = llm.create(system=system, messages=[{"role": "user", "content": user_template.format(ticket=text)}],
                       temperature=temperature, max_tokens=50)
        raw.append(r.text)
        preds.append(parser(r.text))
        cost += r.cost_usd
        tok += r.usage["input_tokens"]
    preds = pd.Series(preds, index=df.index)
    parsed = preds.notna()
    return {"accuracy": float(np.mean(preds == df.intent)), "parse rate": float(parsed.mean()),
            "accuracy when parsed": float(np.mean(preds[parsed] == df.intent[parsed])) if parsed.any() else np.nan,
            "input tokens per call": tok / len(df), "$ per 1,000 tickets": cost / len(df) * 1000}, preds, raw


LABEL_LIST = "Categories: " + ", ".join(LABELS)
LABEL_DESC = "Categories:\n" + "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
ROLE = "You are the triage assistant for Lumora Home's customer support team. Classify each customer ticket."
PROMPTS = {
    "v0 bare question": ("", "What is this support ticket about?\n\n{ticket}"),
    "v1 + label list": ("", LABEL_LIST + "\n\nTicket: {ticket}"),
    "v2 + output format": ("", LABEL_LIST + "\nAnswer with only the label.\n\nTicket: {ticket}"),
    "v3 + label descriptions": ("", LABEL_DESC + "\nAnswer with only the label.\n\nTicket: {ticket}"),
    "v4 + role + delimiters": (ROLE, LABEL_DESC + "\nAnswer with only the label.\n\n<ticket>{ticket}</ticket>"),
}

# %% =========================================================================
# LEVEL 1 (EASY): Five versions of one prompt
# ============================================================================
banner("Five versions of the same prompt, scored on 198 dev tickets", "EASY")
show("Intents", LABELS)
show("Dev / test tickets", f"{len(dev)} / {len(test)}")
rows, raws, preds_by = {}, {}, {}
for name, (system, user) in PROMPTS.items():
    res, preds, raw = evaluate_prompt(large, system, user, dev)
    rows[name], raws[name], preds_by[name] = res, raw, preds
board = pd.DataFrame(rows).T
print(board.round(3).to_string())
step("1.1  What the model actually wrote (first 3 dev tickets)")
for name in PROMPTS:
    print(f"  {name:<26} {[r[:46] for r in raws[name][:3]]}")
slips = [r for r in raws["v4 + role + delimiters"] if parse_strict(r) is None]
note(f"""v0 asks a vague question and gets a sensible human answer ('The customer has
an account question'), which no program can route: accuracy 0%. Listing the
labels (v1) helps the model, but it still answers in its own words ('Category:
Account', 'This looks like a change address request'): only {board.iloc[1, 1]:.0%} parse.
Saying exactly what to return (v2) makes almost every answer parseable ({board.iloc[2, 0]:.0%}
accuracy). One-line descriptions of each label (v3) take it to {board.iloc[3, 0]:.0%}.
The role sentence and delimiters (v4) did NOT improve understanding (accuracy
when parsed: {board.iloc[3, 2]:.1%} vs {board.iloc[4, 2]:.1%}); v4 is lower only because {len(slips)} answers came back
as e.g. {slips[0]!r}: even 'only the label' slips now and then. Role and
delimiters matter for tone and for keeping customer text from being read as
instructions (Labs 07 and 14); format slips are what Lab 04 eliminates.""")

step("1.2  Strict vs lenient parsing of the chatty v1 answers")
lenient = pd.Series([parse_lenient(r) for r in raws["v1 + label list"]])
show("v1 accuracy, strict parser", f"{board.loc['v1 + label list', 'accuracy']:.1%}")
show("v1 accuracy, lenient parser", f"{np.mean(lenient == dev.intent):.1%}")
note("""A lenient parser rescues most chatty answers, but it is guesswork that breaks
on the next phrasing ('not a return, more of a complaint'). Fix the FORMAT
in the prompt (and in Lab 04 with structured outputs), not in a pile of
regular expressions.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): What descriptions fix, and what temperature does
# ============================================================================
banner("Label names vs label descriptions, and temperature", "MEDIUM")
v2, v3 = preds_by["v2 + output format"], preds_by["v3 + label descriptions"]
cm = pd.crosstab(dev.intent, v2.fillna("(unparsed)"), rownames=["true"], colnames=["predicted (v2, names only)"])
print(cm.to_string())
per_class = pd.DataFrame({"v2 names only": (v2 == dev.intent).groupby(dev.intent).mean(),
                          "v3 + descriptions": (v3 == dev.intent).groupby(dev.intent).mean(),
                          "tickets": dev.intent.value_counts()})
print(per_class.round(2).to_string())
worst = per_class["v2 names only"].idxmin()
confused = cm.loc[worst].drop(worst).idxmax()
example = dev[(dev.intent == worst) & (v2 == confused)].text.iloc[0]
note(f"""With names only, the model must guess what each label means. Its weakest class
is '{worst}' ({per_class.loc[worst, 'v2 names only']:.0%}), most often sent to '{confused}' ({cm.loc[worst, confused]} of {per_class.loc[worst, 'tickets']}), e.g.
"{example}"
An angry customer who mentions an order number looks like an order request
when all you have is the word 'complaint'. The description says what the
label MEANS ('the customer is angry about service and wants someone to take
responsibility'), and that one change fixes it ({per_class.loc[worst, 'v3 + descriptions']:.0%}). Write labels for a
new colleague, not for yourself.""")

step("2.1  Temperature: the same ticket, asked 5 times")
sub = dev.iloc[:80]
rows = {}
for T in [0.0, 0.7, 1.5]:
    runs = []
    for k in range(5):
        _, p, _ = evaluate_prompt(large, *PROMPTS["v3 + label descriptions"], sub, temperature=T)
        runs.append(p)
    R = pd.concat(runs, axis=1).fillna("(unparsed)")
    rows[f"T = {T}"] = {"mean accuracy": float(np.mean([(R[c] == sub.intent).mean() for c in R])),
                        "tickets with the same answer all 5 times": float((R.nunique(axis=1) == 1).mean())}
    if T == 1.5:
        flipped = sub[R.nunique(axis=1) > 1]
temp = pd.DataFrame(rows).T
print(temp.round(3).to_string())
show("Tickets whose label changed between runs at T = 1.5", list(flipped.text.str[:60]))
note(f"""At temperature 0 the same ticket gets the same answer every time. Raise it and
answers start to change between runs: {1 - temp.iloc[2, 1]:.0%} of tickets at T = 1.5, some switching
label, some switching format. Average accuracy barely moves ({temp.iloc[0, 0]:.1%} -> {temp.iloc[2, 0]:.1%}),
which is the trap: the dashboard looks fine while the same customer could get
a different routing on a second try. For classification, extraction and
anything code depends on, use temperature 0. (Some current models, Claude
among them, manage sampling themselves and reject a temperature setting: the
principle is the same, it's just not your knob.)""")

# %% =========================================================================
# LEVEL 3 (HARD): Does it hold up? What is it worth?
# ============================================================================
banner("Unseen tickets, the remaining errors, and the price of a lazy prompt", "HARD")
rows = {}
for model in [large, small]:
    for name in ["v2 + output format", "v4 + role + delimiters"]:
        for split_name, df in [("dev", dev), ("test", test)]:
            res, preds, raw = evaluate_prompt(model, *PROMPTS[name], df)
            rows[(model.model, name, split_name)] = res
            if model is large and name.startswith("v4") and split_name == "test":
                final_preds, final_raw = preds, pd.Series(raw, index=df.index)
hold = pd.DataFrame(rows).T
hold.index.names = ["model", "prompt", "split"]
print(hold[["accuracy", "input tokens per call", "$ per 1,000 tickets"]].round(4).to_string())

step("3.1  The errors that remain (large model, v4, test set)")
wrong = test[final_preds != test.intent].assign(predicted=final_preds[final_preds != test.intent].fillna("(format slip)"))
print(wrong[["text", "intent", "predicted"]].to_string(index=False, max_colwidth=90))
slip = wrong.predicted == "(format slip)"
n_slip = int(slip.sum())
n_slip_right = int(sum(parse_lenient(final_raw[i]) == test.intent[i] for i in wrong.index[slip]))
note(f"""Dev and test agree, so we didn't overfit the prompt to the tickets we looked at
(always keep a test set the prompt designer never reads). Of the {len(wrong)} test errors,
{n_slip} are format slips ({n_slip_right} of them with the right label inside). The rest are tickets that
honestly sit between two labels: 'it arrived broken, can I get my money
back?' reads as a status question to the model, and a policy question with a
fake 'SYSTEM OVERRIDE' tacked on has two things going on. A single label is
the wrong output for such tickets: Lab 04 extracts ALL the fields, and Lab 11
plans one step per request.""")

l2 = hold.loc[("sim-large", "v2 + output format", "test")]
s4 = hold.loc[("sim-small", "v4 + role + delimiters", "test")]
show("Large model, lazy prompt (v2): accuracy / $ per 1,000", f"{l2['accuracy']:.1%} / ${l2['$ per 1,000 tickets']:.3f}")
show("Small model, good prompt (v4): accuracy / $ per 1,000", f"{s4['accuracy']:.1%} / ${s4['$ per 1,000 tickets']:.4f}")
note(f"""Plot twist: the SMALL model with a good prompt ({s4['accuracy']:.1%}) beats the LARGE model with
a lazy prompt ({l2['accuracy']:.1%}), at {l2['$ per 1,000 tickets'] / s4['$ per 1,000 tickets']:.0f}x lower cost. The good prompt is longer (more
input tokens), and it is still far cheaper. Before reaching for a bigger
model, fix the prompt: it is the cheapest upgrade you will ever make.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    board[["accuracy", "parse rate"]].plot.barh(ax=axes[0])
    axes[0].invert_yaxis()
    axes[0].set(xlim=(0, 1.05), title="Five prompt versions (dev, large model)")
    per_class[["v2 names only", "v3 + descriptions"]].plot.barh(ax=axes[1])
    axes[1].set(xlim=(0, 1.05), title="Accuracy per intent: names vs descriptions")
    axes[1].legend(fontsize=8, loc="lower left", framealpha=0.95)
    for (model, name, split_name), r in hold.iterrows():
        if split_name == "test":
            axes[2].scatter(r["$ per 1,000 tickets"], r["accuracy"], s=80, color="C0" if model == "sim-large" else "C2")
            axes[2].annotate(f"{model}, {name.split()[0]}", (r["$ per 1,000 tickets"], r["accuracy"]), fontsize=8,
                             xytext=(5, -10), textcoords="offset points")
    axes[2].set(xscale="log", xlabel="$ per 1,000 tickets (log)", ylabel="test accuracy", title="Prompt quality vs model size")
    save_fig(fig, "02_prompt_design_fundamentals")

banner("KEY TAKEAWAYS")
print("""  1. A prompt is code: write it against a dev set, check it once on a held-out test set.
  2. List the labels, say exactly what to return, and parse strictly: unparseable = wrong.
  3. Describe what each label MEANS; the model can't read your mind or your team's jargon.
  4. Use temperature 0 for classification and extraction.
  5. Delimit the customer's text from your instructions.
  6. A better prompt on a small model can beat a bigger model at a fraction of the cost.""")
