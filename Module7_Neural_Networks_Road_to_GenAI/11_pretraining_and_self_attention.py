"""
LAB 11 - Pretraining a Tiny GPT, and Looking Inside Its Self-Attention
======================================================================

PRETRAINING = next-token prediction on a large pile of text
    Take any window of text. Input: tokens 1..T. Target: tokens 2..T+1. Every position is a training example
    ("given everything so far, what comes next?"), so a 64-token window gives 64 examples at once (the causal
    mask of Lab 09 makes that legal). No labels are needed: the text IS the label. That's why LLMs can train
    on trillions of tokens.

OUR TINYGPT
    4 decoder blocks, 128 wide, 4 heads, 64-token context, ~850,000 parameters (GPT-3: 175 billion), trained for
    a few minutes on a CPU on the invented world (Lab 10). Trained once, then cached in outputs/.

WHAT TO WATCH
    Loss falls; samples go from gibberish to grammar to facts. Facts seen often are known; facts seen once
    are shaky. Knowledge in one phrasing is not knowledge in every phrasing.

LEVELS
    EASY   : The pretraining objective, the loss curve, and samples during training.
    MEDIUM : What did it learn? Fact recall vs how often the fact appeared; knowledge the model has but
             doesn't SAY; which tokens its attention reads.
    HARD   : Scale: three model sizes. Plot twist: the phrasing of the question matters.
"""
# %%
import numpy as np
import pandas as pd

from ml_lab_utils import HAS_MPL, banner, note, plt, save_fig, show, step
from nn_toolkit import HAS_TORCH, FACT_PROMPTS

if not HAS_TORCH:
    raise SystemExit("This lab needs PyTorch:  pip install torch")
import torch

from nn_toolkit import continuation_logprob, generate, probe_facts, world_lm

torch.set_num_threads(8)
world = world_lm("base")
model, tok, facts = world.model, world.tok, world.facts
known = facts[facts.known].reset_index(drop=True)

# %% =========================================================================
# LEVEL 1 (EASY): The objective, the loss, the samples
# ============================================================================
banner("Pretraining: predict the next token, everywhere, at once", "EASY")
ids = tok.encode(world.text[:400])[:12]
print(pd.DataFrame({"input (what the model sees)": [tok.vocab[i] for i in ids[:-1]],
                    "target (the next token)": [tok.vocab[i] for i in ids[1:]]}).T.to_string(header=False))
note("""One window of training text, shifted by one: every position predicts the token
after it. The loss is the average cross-entropy over all of them (Lab 03).
Nobody labelled anything: the next token is the label.""")

hist = np.array(world.info["loss_history"])
smooth = lambda s: hist[max(0, s - 25):s + 25].mean()
steps_show = [1, 100, 500, 1500, len(hist)]
lc = pd.DataFrame({"training loss": [smooth(s - 1) for s in steps_show]}, index=steps_show)
lc["perplexity = exp(loss)"] = np.exp(lc["training loss"])
lc.index.name = "step"
print(lc.round(2).to_string())
show("Parameters / training tokens / training time", f"{model.n_params:,} / {world.info['n_tokens']:,} / "
                                                     f"{world.info['train_seconds']:.0f}s (first run only; cached)")
note(f"""At step 1 the model guesses uniformly over {len(tok.vocab)} tokens: perplexity ~{len(tok.vocab)}. By the end
it is effectively choosing between ~{lc.iloc[-1, 1]:.1f} tokens at each step. Much of the text is
predictable from grammar ('The capital of' -> a name); the rest is facts and
random ICL values, which no amount of training can predict.""")

step("1.1  Greedy samples during training")
samples = pd.DataFrame(world.info["samples"]).T
samples.index.name = "step"
for col in samples.columns:
    print(f"\n  prompt: {col!r}")
    for s, v in samples[col].items():
        print(f"    step {s:>5}: {v!r}")
note("""Step 25: noise. By ~150 steps: the right SHAPE of an answer (a capitalised
name, a full stop) but a wrong or invented name. Later: the right facts. The
model learns the format of language long before the content, which is why
an under-trained (or too small) model sounds fluent while being wrong.
The 'map' line never gets solved (the answer is 9): looking a value up in the
prompt is a skill pretraining alone did not teach this model. Lab 12.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): What did it learn?
# ============================================================================
banner("What does the model know, and where does attention look?", "MEDIUM")
step("2.1  Fact recall vs how often the fact appeared in the training text")
probe = probe_facts(model, tok, known)
by_rep = probe.groupby("repeats").agg(facts=("correct", "size"), recall=("correct", "mean"), avg_confidence=("confidence", "mean"))
print(by_rep.round(3).to_string())
note(f"""Each known fact was written 1, 2, 4, 8 or 16 times in the corpus. Seen 16 times:
recall {by_rep.recall.iloc[-1]:.0%}. Seen once: {by_rep.recall.iloc[0]:.0%}. A language model's 'knowledge' is a smooth
function of exposure: frequent facts are reliable, rare facts are a coin
toss. The same holds for real LLMs: famous people's birthdays, yes; your
company's internal policy, no.""")

step("2.2  Knowledge it has but doesn't say: multiple choice by likelihood")
rows = []
for rel in FACT_PROMPTS:
    options = sorted(known[known.relation == rel].value.unique())
    for f in known[known.relation == rel].itertuples():
        prompt = FACT_PROMPTS[rel].format(c=f.country)
        lp = {o: continuation_logprob(model, tok, prompt, " " + o) for o in options}
        rows.append({"relation": rel, "repeats": f.repeats, "mc_correct": max(lp, key=lp.get) == f.value,
                     "n_options": len(options)})
mc = pd.DataFrame(rows)
mc_tab = pd.DataFrame({"greedy generation": probe.groupby("repeats").correct.mean(),
                       "multiple choice (most likely option)": mc.groupby("repeats").mc_correct.mean()})
print(mc_tab.round(3).to_string())
note("""Instead of letting the model write, score every possible answer by its log-
probability and pick the most likely one. For rare facts this recovers more
than greedy generation does: some knowledge is in the weights, just not
strong enough to win the first token. This is how LLM benchmarks with
multiple-choice questions are often scored.""")

step("2.3  Self-attention: when completing 'The capital of X is', what does the last position read?")
rows = []
for f in known[known.relation == "capital"].head(30).itertuples():
    prompt = FACT_PROMPTS["capital"].format(c=f.country)
    p_ids = tok.encode(prompt)
    country_pos = set(range(len(tok.encode("The capital of")), len(tok.encode("The capital of " + f.country))))
    with torch.no_grad():
        model(torch.tensor([p_ids]), keep_attn=True)
    for L, blk in enumerate(model.blocks):
        a = blk.last_attn[0, :, -1].mean(0).numpy()                 # last position, averaged over heads
        rows.append({"layer": L + 1, "attention on the country's tokens": a[list(country_pos)].sum(),
                     "share of positions that are the country": len(country_pos) / len(p_ids)})
att = pd.DataFrame(rows).groupby("layer").mean()
print(att.round(3).to_string())
note("""The position that must produce the capital is 'is'. Its attention is the
model's way of fetching information from earlier tokens. In some layers it
puts far more weight on the country's tokens than their share of the prompt:
self-attention routes 'which country?' to the place where the answer is
produced. Nobody programmed this; it emerged from next-token prediction.""")

# %% =========================================================================
# LEVEL 3 (HARD): Scale, and the phrasing of the question
# ============================================================================
banner("Scale, and why phrasing matters", "HARD")
step("3.1  Three model sizes, same data, same number of steps")
rows = {}
for size in ["tiny", "small", "base"]:
    w = world if size == "base" else world_lm(size)
    pr = probe if size == "base" else probe_facts(w.model, w.tok, known)
    rows[size] = {"parameters": w.model.n_params, "final training loss": float(np.mean(w.info["loss_history"][-50:])),
                  **{f"recall, seen {r}x": v for r, v in pr.groupby("repeats").correct.mean().items()},
                  "recall, all": pr.correct.mean()}
scale = pd.DataFrame(rows).T
print(scale.to_string(formatters={"parameters": "{:,.0f}".format, **{c: "{:.3f}".format for c in scale.columns[1:]}}))
note(f"""More parameters, lower loss, more facts ({scale.loc['tiny', 'recall, all']:.0%} -> {scale.loc['base', 'recall, all']:.0%}). The smallest model can
learn the format but has little room to store facts. This is the scaling
behaviour that drove the LLM era: loss falls smoothly as models, data and
compute grow together, and capabilities appear along the way.""")

step("3.2  Plot twist: does the model know a fact, or a SENTENCE?")
sents = world.text.split("\n")
rows = []
for f in probe[probe.relation == "capital"].itertuples():
    rows.append({"repeats": f.repeats, "correct": f.correct,
                 "probe phrasing was in the training text": f"The capital of {f.country} is {f.value}." in sents})
ph = pd.DataFrame(rows)
ph_tab = ph[ph.repeats <= 2].groupby("probe phrasing was in the training text").correct.agg(["size", "mean"])
ph_tab.columns = ["capital facts seen 1-2x", "recall with 'The capital of X is'"]
print(ph_tab.round(3).to_string())
alt = {"The government of {c} sits in": "government phrasing", "{c}'s capital is": "unseen phrasing"}
rows = {}
for prompt, label in alt.items():
    pr = probe_facts(model, tok, known[known.relation == "capital"], prompts={"capital": prompt})
    rows[f"'{prompt}' ({label})"] = pr.correct.mean()
rows[f"'{FACT_PROMPTS['capital']}' (standard)"] = probe[probe.relation == "capital"].correct.mean()
print(pd.Series(rows, name="capital recall, all known countries").round(3).to_string())
note("""Rare facts are recalled far better when the training text happened to use the
same phrasing as the question. And a phrasing the model never saw
("X's capital is") recalls fewer facts than the familiar ones. The model
stores patterns of text, and a fact written one way is only partly
available when asked another way. Real LLMs show the same effect (the
'reversal curse': trained on 'A is B's parent', they struggle with 'B's
parent is ...'). Lab 12 shows how instruction tuning teaches a NEW phrasing
(questions) without teaching any new facts.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].plot(np.arange(1, len(hist) + 1), pd.Series(hist).rolling(25, min_periods=1).mean())
    axes[0].set(xscale="log", xlabel="training step", ylabel="loss (smoothed)", title="Pretraining loss")
    for size in scale.index:
        axes[1].plot([1, 2, 4, 8, 16], scale.loc[size, [f"recall, seen {r}x" for r in [1, 2, 4, 8, 16]]].astype(float), "o-",
                     label=f"{size} ({scale.loc[size, 'parameters']:,.0f} params)")
    axes[1].set(xscale="log", xticks=[1, 2, 4, 8, 16], xticklabels=[1, 2, 4, 8, 16], xlabel="times the fact appeared",
                ylabel="recall", title="Knowledge = exposure x capacity", ylim=(0, 1.05))
    axes[1].legend(fontsize=8)
    f0 = known[known.relation == "capital"].iloc[0]
    p_ids = tok.encode(FACT_PROMPTS["capital"].format(c=f0.country))
    with torch.no_grad():
        model(torch.tensor([p_ids]), keep_attn=True)
    A = torch.stack([b.last_attn[0].mean(0) for b in model.blocks])[:, -1].numpy()
    axes[2].imshow(A, cmap="Blues", aspect="auto")
    axes[2].set(xticks=range(len(p_ids)), xticklabels=[tok.vocab[i] for i in p_ids], yticks=range(len(A)),
                yticklabels=[f"layer {i + 1}" for i in range(len(A))], title="Attention of the last position")
    axes[2].tick_params(axis="x", rotation=60)
    save_fig(fig, "11_pretraining_and_self_attention")

banner("KEY TAKEAWAYS")
print("""  1. Pretraining = next-token prediction on raw text; every position is a free training example.
  2. Models learn the format of language before its content: fluent is not the same as correct.
  3. Knowledge grows with exposure: facts seen often are reliable, rare facts are shaky.
  4. Some knowledge is in the weights but not generated: likelihood scoring can reveal it.
  5. Bigger models store more; and a fact learned in one phrasing is only partly available in another.""")
