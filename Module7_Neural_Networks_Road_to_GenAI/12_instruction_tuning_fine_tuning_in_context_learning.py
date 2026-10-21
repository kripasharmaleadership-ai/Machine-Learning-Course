"""
LAB 12 - Instruction Tuning, Fine-Tuning and In-Context Learning
================================================================

THREE WAYS TO CHANGE WHAT A PRETRAINED MODEL DOES
    instruction tuning   train a little more on (instruction, good answer) pairs, loss on the ANSWER only.
                         Teaches FORMAT and behaviour ("when asked a question, answer it"), not new facts.
                         (Real assistants follow with preference tuning, RLHF/DPO: same idea, ranked answers.)
    fine-tuning          train more on your own data: a new task, a new style, new facts. Powerful, but the
                         weights you change are the weights that held the old knowledge.
    in-context learning  change NOTHING: put examples or facts in the prompt and let attention use them.
                         The model 'learns' for one answer and forgets when the prompt is gone.
                         (RAG, Module 6, is in-context learning with retrieved documents.)

LEVELS
    EASY   : The base model doesn't answer questions; 250 steps of instruction tuning on SOME countries make it
             answer for countries it was never tuned on, from knowledge acquired in pretraining.
    MEDIUM : In-context learning: a lookup the pretrained model can't do, unlocked in 800 steps. Plot twist:
             the same 800 steps from scratch fail. Facts in the prompt vs facts in the weights.
    HARD   : Fine-tuning new facts in: it works, and old facts are forgotten, unless you replay old data.
"""
# %%
import copy

import numpy as np
import pandas as pd

from ml_lab_utils import HAS_MPL, banner, make_icl_episodes, note, plt, save_fig, show, step
from nn_toolkit import FACT_PROMPTS, HAS_TORCH

if not HAS_TORCH:
    raise SystemExit("This lab needs PyTorch:  pip install torch")
import torch

from nn_toolkit import (TinyGPT, answer_of, context_fact_examples, generate, icl_accuracy, icl_training_pairs, probe_facts,
                        train_lm_examples, train_lm_stream, world_chat_lm, world_icl_lm, world_lm)

torch.set_num_threads(8)
base = world_lm("base")
chat = world_chat_lm()
tok, facts = base.tok, base.facts
split = chat.info["split"]
known = facts[facts.known]
held = known[known.country.isin(split["test"])]                     # countries NEVER used in instruction tuning

# %% =========================================================================
# LEVEL 1 (EASY): Instruction tuning
# ============================================================================
banner("Instruction tuning: teach the format, unlock the knowledge", "EASY")
f0 = held.iloc[0]
q = f"Q: What is the capital of {f0.country}?\nA:"
show("Question (a country not used in tuning)", q.replace("\n", " | "))
show("Base model continues with", repr(generate(base.model, tok, q, max_new_tokens=14, temperature=0, stop=None)[0]))
show("Instruction-tuned model answers", repr(generate(chat.model, tok, q, max_new_tokens=14, temperature=0)[0]))
show("True answer", f0.value)
show("Tuning pairs / steps / countries used", f"{chat.info['n_examples']} / 250 / {len(split['tune'])} (of {known.country.nunique()})")
note("""The base model has never seen 'Q:' and 'A:'. It treats the question as text to
continue and drifts into whatever it has seen before. The tuned model saw a few
hundred question-answer pairs about OTHER countries (mixed with replayed
pretraining text, see Level 3), and now answers.""")

step("1.1  Held-out countries: is the knowledge new, or unlocked?")
rows = {}
p_base_qa = probe_facts(base.model, tok, held, qa=True)
p_chat_qa = probe_facts(chat.model, tok, held, qa=True)
p_base_cmp = probe_facts(base.model, tok, held)
rows["base model, asked as a question"] = p_base_qa.groupby("repeats").correct.mean()
rows["base model, as a sentence to complete"] = p_base_cmp.groupby("repeats").correct.mean()
rows["instruction-tuned, asked as a question"] = p_chat_qa.groupby("repeats").correct.mean()
it = pd.DataFrame(rows).T
it["all held-out facts"] = [p_base_qa.correct.mean(), p_base_cmp.correct.mean(), p_chat_qa.correct.mean()]
it.columns = [f"seen {c}x" if c != "all held-out facts" else c for c in it.columns]
print(it.round(3).to_string())
note(f"""For these {held.country.nunique()} countries the model saw NO question during tuning, yet it now
answers {it.iloc[2, -1]:.0%} of their facts (the base model: {it.iloc[0, -1]:.0%}), following the same pattern
by exposure. Instruction tuning did not teach these facts: pretraining did.
It taught a new way of ASKING for them. Honest footnote: the same facts
asked as a sentence to complete score {it.iloc[1, -1]:.0%}, so part of the knowledge is still
locked to the phrasing it was learned in (Lab 11). Bigger models, more varied
pretraining text and more tuning data close that gap; a chatbot's knowledge
comes from pretraining, its manners from tuning.""")

# %% =========================================================================
# LEVEL 2 (MEDIUM): In-context learning
# ============================================================================
banner("In-context learning: answers that come from the prompt", "MEDIUM")
icl = world_icl_lm()
test_eps = {n: make_icl_episodes(n=200, n_pairs=n, seed=99) for n in [2, 8]}
test_ctx = context_fact_examples(n=200, seed=999)                  # fresh names, facts stated in the prompt


def context_scores(m):
    row = {f"map lookup, {n} pairs": icl_accuracy(m, tok, e) for n, e in test_eps.items()}
    row["copy a fact from the prompt"] = float(np.mean([answer_of(generate(m, tok, p, max_new_tokens=8, temperature=0)[0])
                                                        == a.strip(" .") for p, a in test_ctx]))
    row["world facts recalled"] = float(probe_facts(m, tok, known).correct.mean())
    return row


ex = make_icl_episodes(n=1, n_pairs=5, seed=3)[0]
show("An episode (a FRESH mapping every time)", f"{ex[0]!r} -> answer {ex[1]!r}")
step("2.1  Context skills on 200 fresh examples each (map chance = 10%)")
note("""Training mix for 800 steps (loss on answers only): map episodes, facts stated
in the prompt about freshly invented countries, and replayed pretraining
lines so the world's facts are not forgotten.""")
rows = {"pretrained base model": context_scores(base.model), "base + 800 steps of the mix": context_scores(icl.model)}
torch.manual_seed(0)
scratch = TinyGPT(len(tok.vocab), d_model=128, n_layers=4, n_heads=4, context=64, dropout=0.0)
train_lm_examples(scratch, [(tok.encode(p), tok.encode(a)) for p, a in icl_training_pairs(base.text)],
                  steps=800, batch=64, lr=5e-4, seed=0)
rows["random init + the same 800 steps"] = context_scores(scratch)
icl_tab = pd.DataFrame(rows).T
print(icl_tab.round(3).to_string())
note(f"""Each map episode is a new random mapping, so the answer cannot be memorised: it
must be read from the prompt. The pretrained model can't do it ({icl_tab.iloc[0, 1]:.0%} with 8 pairs),
even though 1,500 such lines were in its training text. After 800 steps of
the mix it does it almost perfectly ({icl_tab.iloc[1, 1]:.0%}), copies a fact stated in the prompt
{icl_tab.iloc[1, 2]:.0%} of the time (multi-piece invented names are harder), and keeps most world
knowledge ({icl_tab.iloc[1, 3]:.0%} vs {icl_tab.iloc[0, 3]:.0%}). Plot twist: the SAME 800 steps from a random
initialisation reach only {icl_tab.iloc[2, 1]:.0%} on the map (it learns 'say a digit from the
prompt', not WHICH one). Pretraining on ordinary text built the attention
circuits (e.g. 'what came just before this token?'); the extra steps taught
the model to combine them. Real LLMs acquire in-context learning the same
way, as a by-product of pretraining on trillions of tokens.""")

step("2.2  Facts in the prompt vs facts in the weights")
unk_c = facts[~facts.known].country.unique()
well = known[known.relation == "language"].sort_values("repeats", ascending=False).head(20)
langs = sorted(known[known.relation == "language"].value.unique())
fresh = [langs[i % len(langs)] for i in range(len(unk_c))]           # a language GIVEN in the prompt
other = [next(v for v in langs[(i + 1):] + langs if v != f.value) for i, f in enumerate(well.itertuples())]
ask = lambda c, v: f"People in {c} speak {v}.\nPeople in {c} speak"
rows = {}
for label, m in [("base", base.model), ("context-skills model", icl.model)]:
    give = [answer_of(generate(m, tok, ask(c, v), max_new_tokens=6, temperature=0)[0]) == v for c, v in zip(unk_c, fresh)]
    contra = [answer_of(generate(m, tok, ask(f.country, v), max_new_tokens=6, temperature=0)[0])
              for f, v in zip(well.itertuples(), other)]
    rows[label] = {"unknown country, language GIVEN in the prompt: repeats it": np.mean(give),
                   "well-known country, prompt CONTRADICTS: follows the prompt": np.mean([a == v for a, v in zip(contra, other)]),
                   "well-known country, prompt CONTRADICTS: follows memory": np.mean([a == f.value for a, f in zip(contra, well.itertuples())])}
ctx = pd.DataFrame(rows).T
print(ctx.round(2).to_string())
note(f"""Give an unknown country's language in the prompt: the base model mostly
ignores it ({ctx.iloc[0, 0]:.0%} repeat it) and guesses (Lab 13 calls that a hallucination); the
context-skills model repeats it {ctx.iloc[1, 0]:.0%} of the time. (Copying a multi-piece invented
CAPITAL name works far less often in a model this small: long exact copies are
one of the things that improve with scale.) That is RAG in miniature
(Module 6): retrieved text in the prompt fills a gap in memory. When the
prompt CONTRADICTS a fact the model knows well, it must choose between context
and memory: the base model sticks to memory ({ctx.iloc[0, 2]:.0%}), the context-skills model
follows the prompt ({ctx.iloc[1, 1]:.0%}). Neither is 'right'. Which one you WANT depends
on the product: a RAG assistant should trust the retrieved document; a
fact-checker should not.""")

# %% =========================================================================
# LEVEL 3 (HARD): Fine-tuning in new facts, and forgetting old ones
# ============================================================================
banner("Fine-tuning new facts: learning, and catastrophic forgetting", "HARD")
new = pd.DataFrame({"country": ["Zentria", "Pollumar", "Quessa", "Brindor", "Valtessa"],
                    "capital": ["Okomir", "Treska", "Ulvana", "Femrod", "Ashkel"],
                    "language": ["Morvan", "Seren", "Telish", "Quell", "Halvic"],
                    "export": ["salt", "honey", "glass", "iron", "silk"]})
new_facts = new.melt(id_vars="country", var_name="relation", value_name="value").assign(repeats=0, known=True)
templates = {"capital": ["The capital of {c} is {v}.", "{v} is the capital of {c}."],
             "language": ["People in {c} speak {v}.", "The main language of {c} is {v}."],
             "export": ["The main export of {c} is {v}.", "{c} is known for its {v}."]}
new_lines = [t.format(c=f.country, v=f.value) for f in new_facts.itertuples() for t in templates[f.relation]] * 3
old_sample = known.sample(60, random_state=0)
old_lines = base.text.strip().split("\n")


def recall(model):
    return (float(probe_facts(model, tok, new_facts).correct.mean()), float(probe_facts(model, tok, old_sample).correct.mean()))


rows = {"before fine-tuning": recall(base.model)}
for recipe in ["new facts only", "new facts + replay of old text"]:
    m = copy.deepcopy(base.model)
    rng = np.random.default_rng(0)
    lines = new_lines if recipe == "new facts only" else new_lines + list(rng.choice(old_lines, 4 * len(new_lines)))
    rng.shuffle(lines)
    ids = tok.encode("\n".join(lines) + "\n")
    train_lm_stream(m, ids, steps=300, batch=16, lr=1e-3, seed=0, warmup=10)
    rows[f"300 steps, {recipe}"] = recall(m)
ft = pd.DataFrame(rows, index=["recall: 15 NEW facts", "recall: 60 OLD facts"]).T
print(ft.round(3).to_string())
note(f"""Training only on the new sentences teaches the new facts ({ft.iloc[1, 0]:.0%}), and old facts
fall from {ft.iloc[0, 1]:.0%} to {ft.iloc[1, 1]:.0%}: CATASTROPHIC FORGETTING. Gradient descent moves whatever
weights reduce the current loss, including the ones that stored the old
facts. Mixing old text back in (replay) keeps most of the old knowledge
({ft.iloc[2, 1]:.0%}) while still learning the new ({ft.iloc[2, 0]:.0%}). In practice: fine-tune for FORMAT and
STYLE, use retrieval (RAG) for facts that change, and always re-test old
abilities after fine-tuning.""")

if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    it.iloc[:, :-1].T.plot(ax=axes[0], marker="o")
    axes[0].set(title="Held-out countries: answering questions", ylabel="recall", xlabel="times the fact appeared",
                ylim=(0, 1.05))
    axes[0].legend(fontsize=7)
    icl_tab.T.plot.bar(ax=axes[1], rot=10)
    axes[1].set(title="Context skills: pretrained vs tuned vs from scratch", ylim=(0, 1.05))
    axes[1].legend(fontsize=7)
    ft.plot.barh(ax=axes[2])
    axes[2].set(title="Fine-tuning new facts in", xlim=(0, 1.05))
    axes[2].legend(fontsize=8)
    save_fig(fig, "12_instruction_tuning_fine_tuning_icl")

banner("KEY TAKEAWAYS")
print("""  1. Instruction tuning teaches format and behaviour; the knowledge it 'unlocks' came from pretraining.
  2. A few hundred examples about SOME items generalise to the rest: it learned a skill, not a list.
  3. In-context learning reads answers from the prompt; it needed pretrained circuits to emerge.
  4. Facts in the prompt (RAG) can override or fill gaps in memory; decide which should win.
  5. Fine-tuning new facts in causes catastrophic forgetting; replay old data, re-test, prefer RAG for facts.""")
