"""
LAB 13 - Temperature, Probabilistic Generation and Hallucinations
=================================================================

GENERATION IS SAMPLING
    At every step the model outputs a probability for each token (Lab 10). Generation picks one, appends it, and
    asks again. HOW it picks is a choice:
        greedy (temperature 0)   always the most likely token: repeatable, can be dull or loop
        temperature T            softmax(logits / T): T < 1 sharpens the distribution, T > 1 flattens it
        top-k                    sample only among the k most likely tokens
        top-p (nucleus)          sample among the smallest set of tokens whose probabilities add up to p
    Same prompt, different random draws, different answers: that's why chatbots can answer the same question
    two ways.

HALLUCINATION
    The model is trained to produce LIKELY TEXT, not TRUE text. Asked about something it never learned, the most
    likely continuation of "The capital of X is" is still... a capital-shaped name. Fluent, confident, invented.
    Our world has 10 countries with NO facts in the training data: every answer about them is a hallucination,
    and we can measure it.

LEVELS
    EASY   : The next-token distribution, and what temperature does to it.
    MEDIUM : Sampling 20 answers per question: accuracy vs diversity for greedy, temperature, top-k, top-p.
    HARD   : Hallucinations: always an answer, often confident. Can confidence detect them? Teaching the model
             to say "I don't know", and what that costs.
"""
# %%
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml_lab_utils import HAS_MPL, banner, note, plt, save_fig, show, step
from nn_toolkit import FACT_PROMPTS, HAS_TORCH, QA_QUESTIONS

if not HAS_TORCH:
    raise SystemExit("This lab needs PyTorch:  pip install torch")
import torch

from nn_toolkit import answer_of, generate, next_token_probs, probe_facts, world_chat_lm, world_lm

torch.set_num_threads(8)
base = world_lm("base")
tok, facts = base.tok, base.facts
known = facts[facts.known].reset_index(drop=True)
unknown = facts[~facts.known].reset_index(drop=True)
probe = probe_facts(base.model, tok, known)

# %% =========================================================================
# LEVEL 1 (EASY): The next-token distribution and temperature
# ============================================================================
banner("What the model actually outputs: a probability for every token", "EASY")
single = probe[(probe.relation == "export") & probe.value.map(lambda v: len(tok.encode(" " + v)) == 1)]
f_rare = single.sort_values("repeats").iloc[0]                       # a rarely seen fact with a one-token answer
prompt = FACT_PROMPTS["export"].format(c=f_rare.country)
show(f"Prompt (a fact seen {f_rare.repeats}x in training)", prompt)
show("True answer", f_rare.value)
rows = {}
for T in [0.5, 1.0, 2.0]:
    p = next_token_probs(base.model, tok, prompt, temperature=T)
    top = np.argsort(-p)[:5]
    rows[f"T = {T}"] = [f"{tok.vocab[i]!r} {p[i]:.2f}" for i in top] + [f"{-(p * np.log(p + 1e-12)).sum():.2f}"]
print(pd.DataFrame(rows, index=[f"#{i + 1}" for i in range(5)] + ["entropy (nats)"]).to_string())
p1 = next_token_probs(base.model, tok, prompt)
true_id = tok.encode(" " + f_rare.value)[0]
show(f"P(true answer {tok.vocab[true_id]!r}) at T = 1 / its rank", f"{p1[true_id]:.3f} / #{int((p1 > p1[true_id]).sum()) + 1} of {len(p1)}")
note("""The model doesn't output an answer; it outputs a distribution over all 400
tokens. Temperature divides the logits before the softmax: at T = 0.5 the top
token gets even more of the mass (sharper, lower entropy); at T = 2 the long
tail of unlikely tokens gets a real chance (flatter, higher entropy). Greedy
decoding is the limit T -> 0: always the top token. Notice where the TRUE
answer sits: for a rarely seen fact it may be far down the list. Temperature
can't create knowledge; it only decides how often the model gambles on the
tail. That is where Level 3 begins.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): Sampling many answers
# ============================================================================
banner("Same question, 20 draws: accuracy vs diversity", "MEDIUM")
qs = known.groupby("repeats").head(6).reset_index(drop=True)     # 6 facts per exposure level
all_values = {rel: set(known[known.relation == rel].value) for rel in FACT_PROMPTS}
settings = {"greedy (T = 0)": dict(temperature=0), "T = 0.7": dict(temperature=0.7), "T = 1.0": dict(temperature=1.0),
            "T = 1.5": dict(temperature=1.5), "T = 1.0, top-k = 3": dict(temperature=1.0, top_k=3),
            "T = 1.0, top-p = 0.9": dict(temperature=1.0, top_p=0.9)}
rows = {}
for name, kw in settings.items():
    acc, distinct, invented = [], [], []
    for f in qs.itertuples():
        draws = 1 if kw["temperature"] == 0 else 20
        ans = [answer_of(generate(base.model, tok, FACT_PROMPTS[f.relation].format(c=f.country), max_new_tokens=8, seed=s,
                                  **kw)[0]) for s in range(draws)]
        acc.append(np.mean([a == f.value for a in ans]))
        distinct.append(len(set(ans)))
        invented.append(np.mean([a not in all_values[f.relation] for a in ans]))
    rows[name] = {"share of draws correct": np.mean(acc), "distinct answers per question": np.mean(distinct),
                  "answers that exist nowhere in the world": np.mean(invented)}
samp = pd.DataFrame(rows).T
print(samp.round(3).to_string())
note(f"""Greedy gives one fixed answer per question. Sampling at T = 1 gives several
different answers to the same question, and the share of correct draws drops
({samp.iloc[0, 0]:.0%} -> {samp.iloc[2, 0]:.0%}). At T = 1.5 the model starts producing names that exist nowhere
in our world ({samp.iloc[3, 2]:.0%} of draws): invented by sampling unlikely pieces. Top-k and top-p
cut off the unlikely tail and recover most of the accuracy while keeping some
variety. Rule of thumb: low temperature for facts, code and extraction;
higher for brainstorming and creative writing.""")

step("2.1  Where does the uncertainty come from? Same draws, split by exposure")
rows = {}
for rep in sorted(qs.repeats.unique()):
    sub = qs[qs.repeats == rep]
    vals = []
    for f in sub.itertuples():
        ans = [answer_of(generate(base.model, tok, FACT_PROMPTS[f.relation].format(c=f.country), max_new_tokens=8, seed=s,
                                  temperature=1.0)[0]) for s in range(20)]
        vals.append((np.mean([a == f.value for a in ans]), len(set(ans))))
    rows[rep] = {"share of draws correct (T = 1)": np.mean([v[0] for v in vals]), "distinct answers": np.mean([v[1] for v in vals])}
expo = pd.DataFrame(rows).T
expo.index.name = "times the fact appeared"
print(expo.round(3).to_string())
note("""Facts seen 16 times come back the same in every draw: the distribution is
peaked. Facts seen once come back differently each time. Disagreement between
samples is a useful, cheap signal of uncertainty: ask several times and
check whether the answers agree ('self-consistency').""")

# %% =========================================================================
# LEVEL 3 (HARD): Hallucinations
# ============================================================================
banner("Hallucinations: fluent, confident and invented", "HARD")
pu = probe_facts(base.model, tok, unknown)
show("Countries the model knows NOTHING about", f"{unknown.country.nunique()} ({len(unknown)} questions)")
show("Questions it answered anyway", f"{np.mean(pu.answer.str.len() > 0):.0%}")
for f in pu.head(4).itertuples():
    show(f"  {f.prompt!r}", f"{f.raw.strip()!r}  (confidence {f.confidence:.2f})")
borrowed = np.mean([a in all_values[r] for a, r in zip(pu.answer, pu.relation)])
show("Hallucinations that are another country's real value", f"{borrowed:.0%}")
note("""The model never refuses: 'The capital of X is' is always followed by a
capital-shaped name, because that is what such sentences look like. Many
hallucinations are REAL values of other countries (plausible, and therefore
harder to spot); the rest are new names stitched together from pieces.""")

step("3.1  Can the model's confidence tell knowledge from invention?")
groups = {"known, answered correctly": probe[probe.correct].confidence,
          "known, answered wrongly": probe[~probe.correct].confidence, "unknown country (hallucination)": pu.confidence}
print(pd.DataFrame({k: v.describe()[["count", "mean", "25%", "50%", "75%"]] for k, v in groups.items()}).T.round(3).to_string())
y_bad = np.r_[np.zeros(probe.correct.sum()), np.ones((~probe.correct).sum() + len(pu))]
conf = np.r_[probe[probe.correct].confidence, probe[~probe.correct].confidence, pu.confidence]
auc = roc_auc_score(y_bad, -conf)
show("AUC: low confidence -> wrong-or-invented", f"{auc:.3f}")
rows = {}
for th in [0.0, 0.5, 0.7, 0.9]:
    ans = conf >= th
    rows[f"answer only if confidence >= {th}"] = {
        "correct answers kept": np.mean(ans[y_bad == 0]), "wrong / invented answers given": np.mean(ans[y_bad == 1])}
print(pd.DataFrame(rows).T.round(3).to_string())
note(f"""Hallucinations are LESS confident on average, and confidence separates right
from wrong-or-invented with AUC {auc:.2f}: useful, far from perfect. Any threshold
trades correct answers lost against hallucinations prevented: the same
precision/recall trade-off as Module 4, now for 'should I answer at all?'.""")

step("3.2  Teaching the model to say 'I don't know'")
chat = world_chat_lm()
chat_r = world_chat_lm(refusals=True)
sp = chat_r.info["split"]
held_known = known[known.country.isin(sp["test"])]
unk_test = unknown[unknown.country.isin(sp["refuse_test"])]
rows = {}
for name, m in [("instruction-tuned", chat.model), ("instruction-tuned + refusal examples", chat_r.model)]:
    pk, pu2 = probe_facts(m, tok, held_known, qa=True), probe_facts(m, tok, unk_test, qa=True)
    rows[name] = {"held-out KNOWN: correct": pk.correct.mean(), "held-out KNOWN: refused": pk.refused.mean(),
                  "held-out UNKNOWN: refused": pu2.refused.mean(), "held-out UNKNOWN: hallucinated": 1 - pu2.refused.mean()}
ref = pd.DataFrame(rows).T
print(ref.round(3).to_string())
pr = probe_facts(chat_r.model, tok, unk_test, qa=True)
shown = pr[pr.refused].iloc[0] if pr.refused.any() else pr.iloc[0]
show("Refusal-tuned model, unknown country never seen in tuning", f"{shown.prompt!r} -> {shown.raw!r}")
note(f"""We added "I don't know." answers for 5 unknown countries, then tested on 5
OTHER unknown countries. The refusal-tuned model now refuses {ref.iloc[1, 2]:.0%} of those
(was {ref.iloc[0, 2]:.0%}): some generalisation of 'refuse when I have nothing on this name', but
most hallucinations remain. Plot twist: it also refuses {ref.iloc[1, 1]:.0%} of questions about
countries it COULD answer, and its correct answers drop from {ref.iloc[0, 0]:.0%} to {ref.iloc[1, 0]:.0%}.
Refusal tuning moves the threshold; it doesn't give the model knowledge of
what it knows. Production systems
combine it with retrieval (answer from documents, cite them, refuse when
nothing is retrieved: Module 6) and with evaluation on questions whose
answers are known, exactly like this table.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for T in [0.5, 1.0, 2.0]:
        p = np.sort(next_token_probs(base.model, tok, prompt, temperature=T))[::-1][:15]
        axes[0].plot(range(1, 16), p, "o-", label=f"T = {T}")
    axes[0].set(xlabel="token rank", ylabel="probability", title="Temperature reshapes the distribution")
    axes[0].legend()
    for k, v in groups.items():
        axes[1].hist(v, bins=np.linspace(0, 1, 21), alpha=0.5, label=k, density=True)
    axes[1].set(xlabel="confidence of the first answer token", title=f"Confidence: right vs wrong vs invented (AUC {auc:.2f})")
    axes[1].legend(fontsize=7)
    ref[["held-out KNOWN: correct", "held-out KNOWN: refused", "held-out UNKNOWN: refused"]].T.plot.bar(ax=axes[2], rot=10)
    axes[2].set(title="Refusal tuning: fewer hallucinations, more over-refusal", ylim=(0, 1.05))
    axes[2].legend(fontsize=7)
    save_fig(fig, "13_temperature_sampling_and_hallucinations")

banner("KEY TAKEAWAYS")
print("""  1. The model outputs a distribution; decoding (greedy, temperature, top-k, top-p) turns it into text.
  2. Higher temperature = more diversity and more errors; use low temperature for facts.
  3. Rare facts are unstable across samples; disagreement between samples signals uncertainty.
  4. Hallucination: trained for likely text, the model answers fluently about things it never learned.
  5. Confidence and refusal tuning reduce hallucinations at the cost of refusing some real answers;
     retrieval + citations + evaluation are the production answer.""")
