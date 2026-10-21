"""
LAB 03 - Zero-Shot and Few-Shot Prompting
=========================================

TWO WAYS TO TELL A MODEL WHAT YOU WANT
    ZERO-SHOT   describe the task and the labels; no examples        (Lab 02)
    FEW-SHOT    show solved examples inside the prompt:
                    <example> Ticket: where is my order LM-10234? Label: order_status </example>
                The model copies the pattern. This is in-context learning (Module 7, Lab 12): no training, no
                weights change; the examples live only in this one prompt and are paid for on every call.

WHICH EXAMPLES?
    random      the same fixed handful for every ticket
    dynamic     for each new ticket, RETRIEVE the most similar solved tickets (nearest neighbours, Module 6)
                and use those: the first appearance of retrieval in this module

LEVELS
    EASY   : Zero-shot vs 1 and 4 examples per label, with label NAMES only. Accuracy and cost.
    MEDIUM : Random vs dynamic examples. And the honest comparison: examples vs good descriptions.
    HARD   : Examples teach whatever is in them: an unbalanced set, swapped labels, and noisy labels.
"""
# %%
import re

import numpy as np
import pandas as pd

from genai_toolkit import TfidfEmbedder, get_llm
from ml_lab_utils import HAS_MPL, INTENTS, banner, make_tickets, note, plt, save_fig, show, step

tickets = make_tickets()
dev = tickets[tickets.split == "dev"].reset_index(drop=True)       # the pool we take examples from
test = tickets[tickets.split == "test"].reset_index(drop=True)
LABELS = list(INTENTS)
large = get_llm("large")
NAMES = "Categories: " + ", ".join(LABELS) + "\nAnswer with only the label."
DESCRIPTIONS = "Categories:\n" + "\n".join(f"- {k}: {v}" for k, v in INTENTS.items()) + "\nAnswer with only the label."


def parse(text):
    s = re.sub(r"^\s*label\s*:\s*", "", text.strip(), flags=re.I).lower().replace(" ", "_")   # forgive 'Label: x'
    return s if s in LABELS else None


def examples_block(texts, labels):
    return "\n".join(f"<example>\nTicket: {t}\nLabel: {lab}\n</example>" for t, lab in zip(texts, labels))


def run(build_prompt, df=test, llm=large):
    """build_prompt(ticket_text) -> the user message. Returns accuracy, cost per 1,000, tokens, predictions."""
    preds, cost, tok = [], 0.0, 0
    for text in df.text:
        r = llm.create(messages=[{"role": "user", "content": build_prompt(text)}], max_tokens=20)
        preds.append(parse(r.text))
        cost += r.cost_usd
        tok += r.usage["input_tokens"]
    preds = pd.Series(preds, index=df.index)
    return {"accuracy": float(np.mean(preds == df.intent)), "input tokens per call": tok / len(df),
            "$ per 1,000 tickets": cost / len(df) * 1000}, preds


def fixed_examples(per_label, seed=0):
    ex = dev.sample(frac=1, random_state=seed).groupby("intent").head(per_label)
    return ex.sample(frac=1, random_state=seed)                      # shuffle: don't group examples by label


# %% =========================================================================
# LEVEL 1 (EASY): Zero-shot vs few-shot
# ============================================================================
banner("Zero-shot vs few-shot (label names only, so the examples have work to do)", "EASY")
ex1, ex4 = fixed_examples(1), fixed_examples(4)
show("A few-shot prompt starts like this", "\n" + NAMES + "\n\n" + examples_block(ex1.text[:2], ex1.intent[:2]) + "\n...")
rows, preds = {}, {}
for name, build in [("zero-shot", lambda t: f"{NAMES}\n\nTicket: {t}"),
                    ("few-shot, 1 per label (9)", lambda t: f"{NAMES}\n\n{examples_block(ex1.text, ex1.intent)}\n\nTicket: {t}"),
                    ("few-shot, 4 per label (36)", lambda t: f"{NAMES}\n\n{examples_block(ex4.text, ex4.intent)}\n\nTicket: {t}")]:
    rows[name], preds[name] = run(build)
board = pd.DataFrame(rows).T
print(board.round(4).to_string())
note(f"""With label names only, zero-shot gets {board.iloc[0, 0]:.1%}. One example per label gives
{board.iloc[1, 0]:.1%}: {'WORSE than none' if board.iloc[1, 0] < board.iloc[0, 0] - 0.01 else 'no better'}. Four per label reach {board.iloc[2, 0]:.1%}. Few-shot is not free: 36
examples make every prompt {board.iloc[2, 1] / board.iloc[0, 1]:.0f}x longer, and you pay for them on every
single call. And a fixed handful of examples is a lottery: a ticket that
resembles the wrong example gets pulled towards the wrong label.""")

step("1.1  Same size, different draw: how much does the choice of examples matter?")
draws = [run(lambda t, ex=fixed_examples(1, seed=s): f"{NAMES}\n\n{examples_block(ex.text, ex.intent)}\n\nTicket: {t}")[0]["accuracy"]
         for s in range(5)]
show("1 example per label, 5 different random draws", [f"{a:.1%}" for a in draws])
note(f"""Five equally reasonable sets of 9 examples give anywhere from {min(draws):.1%} to {max(draws):.1%},
{'all of them below' if max(draws) < board.iloc[0, 0] else 'around'} zero-shot ({board.iloc[0, 0]:.1%}). A single example per label is a thin, noisy
signal: one odd-looking ticket can drag a whole class towards it. If you
test only one set, you're measuring luck as much as the method.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Random vs dynamic examples, and examples vs descriptions
# ============================================================================
banner("Dynamic few-shot: retrieve the most similar solved tickets", "MEDIUM")
index = TfidfEmbedder().fit(list(dev.text))
D = index.embed(list(dev.text))


def nearest(text, k, pool=dev, labels=None):
    sims = D @ index.embed([text])[0]
    idx = np.argsort(-sims)[:k][::-1]                                # the most similar example goes last
    labs = pool.intent if labels is None else labels
    return pool.text.iloc[idx], labs.iloc[idx]


t0 = test.text[3]
show("New ticket", t0)
for t, lab in zip(*nearest(t0, 3)):
    show(f"  retrieved example ({lab})", t)
rows = {}
for k in [3, 9, 18]:
    rnd = dev.sample(k, random_state=1)
    rows[(k, "random")] = run(lambda t: f"{NAMES}\n\n{examples_block(rnd.text, rnd.intent)}\n\nTicket: {t}")[0]
    rows[(k, "dynamic (nearest)")], dyn_preds = run(lambda t: f"{NAMES}\n\n{examples_block(*nearest(t, k))}\n\nTicket: {t}")
rows[(0, "zero-shot + descriptions")] = run(lambda t: f"{DESCRIPTIONS}\n\nTicket: {t}")[0]
dyn = pd.DataFrame(rows).T
dyn.index.names = ["examples", "selection"]
print(dyn.round(4).to_string())
d3, r3, zs = dyn.loc[(3, "dynamic (nearest)")], dyn.loc[(3, "random")], dyn.loc[(0, "zero-shot + descriptions")]
note(f"""Three RETRIEVED examples ({d3['accuracy']:.1%}) beat three random ones ({r3['accuracy']:.1%}), and even 18
random ones. Support tickets repeat themselves, so the nearest solved ticket
is usually the same kind of request: the model copies a very relevant
answer. Honest comparison: one line of description per label, no examples
at all, gets {zs['accuracy']:.1%}. When the labels can be explained in words, explain them.
Examples earn their place when the rule is hard to put into words (a house
style, an unusual convention, borderline cases): show, don't tell.""")

# %% =========================================================================
# LEVEL 3 (HARD): Examples teach whatever is in them
# ============================================================================
banner("Examples teach whatever is in them, including your mistakes", "HARD")
step("3.1  An unbalanced example set")
balanced = fixed_examples(1)
skewed = pd.concat([dev[dev.intent == "return_request"].head(6), dev[dev.intent != "return_request"].drop_duplicates("intent").head(3)])
rows = {}
for name, ex in [("balanced: 1 per label (9)", balanced), ("skewed: 6 returns + 3 others (9)", skewed)]:
    res, p = run(lambda t: f"{NAMES}\n\n{examples_block(ex.text, ex.intent)}\n\nTicket: {t}")
    rows[name] = {**res, "share predicted return_request": float((p == "return_request").mean())}
bias = pd.DataFrame(rows).T
print(bias[["accuracy", "share predicted return_request"]].round(3).to_string())
show("True share of return_request in the test set", f"{(test.intent == 'return_request').mean():.1%}")
note(f"""Six of nine examples are returns, and the model now calls {bias.iloc[1, -1]:.0%} of all tickets
returns (truth: {(test.intent == 'return_request').mean():.0%}). Models lean towards labels they've seen often in
the prompt (majority-label bias). Balance your examples, or retrieve them.""")

step("3.2  Two labels swapped in the examples")
ex = fixed_examples(2)
swap = {"cancel_order": "change_address", "change_address": "cancel_order"}
sub = test.intent.isin(list(swap))
rows = {}
for name, labs in [("correct labels", ex.intent), ("cancel/address labels swapped", ex.intent.map(lambda x: swap.get(x, x)))]:
    res, p = run(lambda t: f"{NAMES}\n\n{examples_block(ex.text, labs)}\n\nTicket: {t}")
    rows[name] = {"accuracy, all tickets": res["accuracy"], "accuracy, cancel + address tickets": float(np.mean(p[sub] == test.intent[sub]))}
sw = pd.DataFrame(rows).T
print(sw.round(3).to_string())
note(f"""Swap the labels of two classes in the examples and accuracy on exactly those
tickets falls from {sw.iloc[0, 1]:.0%} to {sw.iloc[1, 1]:.0%}. The model's own sense of 'cancel' still pulls
the other way (it knows the words), so it only partly copies the mistake,
which makes the bug harder to spot: overall accuracy moves only
{sw.iloc[0, 0] - sw.iloc[1, 0]:.0%} points. Always look at accuracy per label, not just the total.""")

step("3.3  Plot twist: noisy labels in the example pool")
rng = np.random.default_rng(0)
noisy = dev.intent.copy()
flip = rng.random(len(dev)) < 0.15                                    # 15% of the solved tickets were mislabelled
noisy[flip] = [rng.choice([lab for lab in LABELS if lab != x]) for x in dev.intent[flip]]
rows, noisy_preds = {}, None
for pool_name, labs in [("clean pool", dev.intent), ("15% of pool mislabelled", noisy)]:
    rnd = dev.sample(9, random_state=1)
    rows[(pool_name, "random 9")] = run(lambda t: f"{NAMES}\n\n{examples_block(rnd.text, labs[rnd.index])}\n\nTicket: {t}")[0]["accuracy"]
    res, p = run(lambda t: f"{NAMES}\n\n{examples_block(*nearest(t, 3, labels=labs))}\n\nTicket: {t}")
    rows[(pool_name, "dynamic 3")] = res["accuracy"]
    noisy_preds = p
noise = pd.Series(rows).unstack().loc[["clean pool", "15% of pool mislabelled"], ["random 9", "dynamic 3"]]
print(noise.round(3).to_string())
top1_bad = pd.Series([bool(flip[int(np.argmax(D @ index.embed([t])[0]))]) for t in test.text], index=test.index)
by_nn = pd.DataFrame({"tickets": top1_bad.value_counts(),
                      "accuracy (dynamic 3, noisy pool)": (noisy_preds == test.intent).groupby(top1_bad).mean()})
by_nn.index = by_nn.index.map({False: "nearest example labelled correctly", True: "nearest example MISlabelled"})
print(by_nn.round(3).to_string())
note(f"""On average, dynamic few-shot shrugs off 15% label noise ({noise.loc['clean pool', 'dynamic 3']:.1%} -> {noise.loc['15% of pool mislabelled', 'dynamic 3']:.1%}),
while random examples lose more ({noise.loc['clean pool', 'random 9']:.1%} -> {noise.loc['15% of pool mislabelled', 'random 9']:.1%}). Plot twist: the average hides
where the damage lands. When the single most similar solved ticket was
mislabelled, accuracy is {by_nn.iloc[1, 1]:.0%} instead of {by_nn.iloc[0, 1]:.0%}: the model often copies the
closest mistake. Those are the tickets most like a known case, so the errors repeat
for every customer who writes that kind of ticket. Retrieved examples are
only as good as your labelled history: clean the pool and re-test whenever
it changes.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].bar(range(len(board)), board.accuracy, color="C0")
    axes[0].set_xticks(range(len(board)), [n.replace(", ", "\n") for n in board.index], fontsize=8)
    axes[0].set(ylim=(0.6, 1.0), ylabel="test accuracy", title="Zero-shot vs few-shot (names only)")
    for k_sel, color in [("random", "C1"), ("dynamic (nearest)", "C2")]:
        s = dyn.xs(k_sel, level="selection")
        axes[1].plot(s.index, s.accuracy, "o-", color=color, label=k_sel)
    axes[1].axhline(zs["accuracy"], color="k", ls="--", label="zero-shot + descriptions")
    axes[1].set(xlabel="examples in the prompt", ylabel="test accuracy", ylim=(0.6, 1.02), title="Random vs retrieved examples")
    axes[1].legend(fontsize=8)
    noise.T.plot.bar(ax=axes[2], rot=0)
    axes[2].set(ylim=(0.5, 1.02), title="Label noise in the example pool")
    axes[2].legend(fontsize=8)
    save_fig(fig, "03_zero_shot_and_few_shot")

banner("KEY TAKEAWAYS")
print("""  1. Few-shot = solved examples in the prompt; the model copies the pattern (in-context learning).
  2. Examples cost tokens on every call, and a fixed random handful is a lottery.
  3. Retrieve the most similar solved examples for each input: few, relevant examples win.
  4. If the rule can be said in words, say it: descriptions can beat examples outright.
  5. Examples carry their biases: balance them, check their labels, and re-test when the pool changes.""")
