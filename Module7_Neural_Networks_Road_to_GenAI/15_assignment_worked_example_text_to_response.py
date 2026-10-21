"""
LAB 15 - ASSIGNMENT WORKED EXAMPLE: The Journey from Raw Text to an LLM-Generated Response
==========================================================================================

THE ASSIGNMENT (syllabus): "Draw and explain the journey from raw text to an LLM-generated response."
This is the reference solution, traced through a REAL model, our instruction-tuned TinyGPT (Labs 10-13), with
the actual numbers at every stage. The student assignment (ASSIGNMENT.md) asks for the same journey on
prompts of your choice, with parts of the pipeline implemented by you.

THE JOURNEY
     1 raw text          "What is the capital of Galcor?"
     2 chat template     "Q: What is the capital of Galcor?\nA:"           (the format instruction tuning taught)
     3 pre-tokenize      ["Q", ":", " What", " is", ...]                     (regex: words with their space, digits)
     4 BPE               subword tokens -> integer ids                        (Lab 10)
     5 embeddings        ids -> vectors, + position vectors                   (T x 128)
     6 transformer       4 blocks of self-attention + MLP                     (Labs 09, 11)
     7 logits            last position -> one score per vocabulary token      (400 scores)
     8 softmax + decode  probabilities -> temperature / top-p -> one token    (Lab 13)
     9 loop              append the token, run again, until a stop token      (autoregression)
    10 detokenize        ids -> text: the response
    Everything a chatbot adds (system prompts, retrieval, tools, safety filters) wraps around this loop.

LEVELS
    EASY   : Stages 1-5: from characters to vectors, every number visible.
    MEDIUM : Stages 6-8: inside the transformer for one step, and the next-token decision.
    HARD   : Stages 9-10: the full loop, then the same journey for a question the model CAN'T answer, and the
             memo with the diagram (outputs/15_text_to_response_journey.md).
"""
# %%
import re

import numpy as np
import pandas as pd

from ml_lab_utils import HAS_MPL, OUT_DIR, banner, note, plt, save_fig, show, step
from nn_toolkit import HAS_TORCH, QA_QUESTIONS, _PRETOK

if not HAS_TORCH:
    raise SystemExit("This lab needs PyTorch:  pip install torch")
import torch

from nn_toolkit import answer_of, generate, world_chat_lm

torch.set_num_threads(8)
chat = world_chat_lm()
model, tok, facts = chat.model, chat.tok, chat.facts
split = chat.info["split"]
held = facts[facts.known & facts.country.isin(split["test"]) & (facts.relation == "capital")]
target = next(f for f in held.sort_values("repeats", ascending=False).itertuples()   # never used in tuning,
              if answer_of(generate(model, tok, f"Q: {QA_QUESTIONS['capital'].format(c=f.country)}\nA:",
                                    temperature=0)[0]) == f.value)            # and answered correctly
QUESTION = QA_QUESTIONS["capital"].format(c=target.country)
memo = {}

# %% =========================================================================
# LEVEL 1 (EASY): Stages 1-5, from characters to vectors
# ============================================================================
banner("Stages 1-5: from characters to vectors", "EASY")
step("Stage 1-2  Raw text and the chat template")
prompt = f"Q: {QUESTION}\nA:"
show("1 raw text", QUESTION)
show("2 with the chat template", repr(prompt))
note("""The user typed a question. The application wraps it in the format the model
was instruction-tuned on (Lab 12). Real chat models use templates with role
markers ('<|user|> ... <|assistant|>') and a hidden SYSTEM prompt; if the
template is wrong, the model behaves like a base model again.""")

step("Stage 3-4  Pre-tokenize, then BPE")
chunks = _PRETOK.findall(prompt)
tokens = tok.tokens(prompt)
ids = tok.encode(prompt)
show("3 pre-tokenized chunks", chunks)
show("4 BPE tokens", tokens)
show("  token ids", ids)
show("  characters -> chunks -> tokens", f"{len(prompt)} -> {len(chunks)} -> {len(ids)}")
note(f"""Words frequent in the tokenizer's training text are single tokens (' capital',
' of'); ' What' was rare there, so it is spelled in {len(tok.encode(' What'))} pieces, and the invented
name '{target.country}' takes {len(tok.encode(' ' + target.country))}. The model never sees characters, only these
{len(ids)} integers. Decoding them back gives the prompt exactly: {tok.decode(ids) == prompt}.""")
memo["tokens"] = (len(prompt), len(chunks), len(ids), tokens)

step("Stage 5  Embeddings: token vector + position vector")
idx = torch.tensor([ids])
with torch.no_grad():
    E_tok = model.tok(idx)[0]
    E_pos = model.pos(torch.arange(len(ids)))
    x0 = E_tok + E_pos
show("Token embedding table / position table", f"{tuple(model.tok.weight.shape)} / {tuple(model.pos.weight.shape)}")
show("5 input to the transformer (tokens x dimensions)", tuple(x0.shape))
show("  first 5 numbers of the vector for token 0", [round(float(v), 3) for v in x0[0, :5]])
note("""Each id selects a row of the token-embedding table (a learned vector for that
token), and each position adds a learned position vector, so ' is' at
position 3 and ' is' at position 9 start out different. From here on the
text is a matrix of numbers: one 128-number row per token.""")
memo["embed"] = tuple(x0.shape)

# %% =========================================================================
# LEVEL 2 (MEDIUM): Stages 6-8, inside one step
# ============================================================================
banner("Stages 6-8: inside the transformer, and the next-token decision", "MEDIUM")
step("Stage 6  Four transformer blocks")
h = x0[None]
rows = []
with torch.no_grad():
    for L, blk in enumerate(model.blocks):
        h = blk(h, keep_attn=True)
        a = blk.last_attn[0, :, -1].mean(0).numpy()                 # what the LAST position reads, averaged over heads
        top = np.argsort(-a)[:3]
        rows.append({"block": L + 1, "output shape": tuple(h.shape[1:]), "size of last position's vector": float(h[0, -1].norm()),
                     "last position attends most to": ", ".join(f"{tokens[i]!r} {a[i]:.2f}" for i in top)})
    logits_manual = model.head(model.ln(h))[0, -1]
    logits_model = model(idx)[0, -1]
blocks = pd.DataFrame(rows).set_index("block")
print(blocks.to_string())
show("Manual pass == model(ids)", bool(torch.allclose(logits_manual, logits_model, atol=1e-5)))
note("""Each block lets every position read from earlier positions (attention) and
then transforms what it gathered (MLP), keeping the shape (tokens x 128).
The last position, the one that must produce the answer, reads from the
question's tokens, including the pieces of the country's name. Running the
blocks by hand gives exactly the model's output: there is nothing else
inside.""")
memo["blocks"] = blocks

step("Stage 7-8  Logits, softmax, and the decision")
probs = torch.softmax(logits_model, -1).numpy()
top = np.argsort(-probs)[:6]
dist = pd.DataFrame({"token": [tok.vocab[i] for i in top], "logit": logits_model.numpy()[top].round(2),
                     "probability": probs[top].round(4)})
print(dist.to_string(index=False))
show("7 logits: one score per vocabulary token", tuple(logits_model.shape))
show("8 greedy pick / its probability", f"{tok.vocab[int(top[0])]!r} / {probs[top[0]]:.3f}")
note("""The final vector is compared with every token embedding (the output layer
shares weights with the input table): 400 scores, the logits. Softmax turns
them into probabilities. Greedy decoding takes the top one; sampling with a
temperature might pick another (Lab 13). This is ONE step of generation.""")
memo["dist"] = dist

# %% =========================================================================
# LEVEL 3 (HARD): Stages 9-10, the loop, and a question it can't answer
# ============================================================================
banner("Stages 9-10: the loop, the response, and where it goes wrong", "HARD")


@torch.no_grad()
def trace(prompt_text, max_new_tokens=12, temperature=0.0, seed=0):
    """Run the generation loop by hand, recording every step."""
    g = torch.Generator().manual_seed(seed)
    seq = tok.encode(prompt_text)
    steps = []
    for t in range(max_new_tokens):
        logits = model(torch.tensor([seq[-model.context:]]))[0, -1]
        p = torch.softmax(logits / max(temperature, 1e-6), -1) if temperature > 0 else torch.softmax(logits, -1)
        nxt = int(torch.multinomial(p, 1, generator=g)) if temperature > 0 else int(p.argmax())
        alt = int(torch.topk(p, 2).indices[1]) if int(p.argmax()) == nxt else int(p.argmax())
        steps.append({"step": t + 1, "context length": len(seq), "chosen token": tok.vocab[nxt], "probability": float(p[nxt]),
                      "runner-up": tok.vocab[alt], "runner-up prob": float(p[alt])})
        seq.append(nxt)
        if "\n" in tok.vocab[nxt]:
            break
    out = tok.decode(seq[len(tok.encode(prompt_text)):])
    return pd.DataFrame(steps).set_index("step"), out.split("\n")[0]


step("Stage 9-10  The loop: append the chosen token, run the whole model again")
loop, response = trace(prompt)
print(loop.round(3).to_string())
show("10 response (detokenized)", repr(response))
show("True answer", target.value)
note(f"""Each step re-reads the ENTIRE sequence so far (the context grows by one token
per step) and adds one token. The response took {len(loop)} passes through the
model, and stopped when it produced a newline, the end-of-answer marker the
tuning data used. Real systems cache the keys and values of earlier
positions (the KV cache, Lab 09) so each step only computes the new token.""")
memo["loop"], memo["response"] = loop, response

step("The same journey for a question the model can't answer")
unk = facts[~facts.known].country.iloc[0]
prompt_u = f"Q: {QA_QUESTIONS['capital'].format(c=unk)}\nA:"
loop_u, response_u = trace(prompt_u)
print(loop_u.round(3).to_string())
show("Response", repr(response_u))
samples = [trace(prompt_u, temperature=1.0, seed=s)[1] for s in range(8)]
show("8 samples at temperature 1.0", samples)
samples_k = [trace(prompt, temperature=1.0, seed=s)[1] for s in range(8)]
show(f"8 samples for '{target.country}' (known) at T = 1.0", samples_k)
seq_p = lambda lp: float(np.prod(lp["probability"]))
agree = lambda xs: max([answer_of(y) for y in xs].count(answer_of(x)) for x in xs) / len(xs)   # ' Eoto.' == 'Eoto.'
signals = pd.DataFrame({"first-token probability": [loop.iloc[0]["probability"], loop_u.iloc[0]["probability"]],
                        "whole-answer probability": [seq_p(loop), seq_p(loop_u)],
                        "samples agreeing with the most common": [agree(samples_k), agree(samples)]},
                       index=[f"known: {target.country}", f"unknown: {unk}"])
print(signals.round(3).to_string())
note(f"""Every stage runs identically, and out comes a fluent, capital-shaped answer for
a country that appears in no fact anywhere: a hallucination. Plot twist: the
FIRST token is no warning sign here ({signals.iloc[1, 0]:.2f}, higher than for the known country):
' Dor' simply starts many capitals in this world. The warning signs are the
whole answer's probability (the product of every step: {signals.iloc[0, 1]:.2f} vs {signals.iloc[1, 1]:.2f}) and the
disagreement between samples ({signals.iloc[0, 2]:.0%} vs {signals.iloc[1, 2]:.0%} agree). Nothing in the pipeline
checks truth; it only produces likely text. Grounding (RAG), refusal tuning
and confidence checks are added AROUND this loop.""")
memo["signals"] = signals
memo["unknown"] = (unk, loop_u, response_u, samples, samples_k)

# ---------------------------------------------------------------------------
# The diagram and the memo
# ---------------------------------------------------------------------------
STAGES = [
    ("1 raw text", QUESTION, "what the user typed"),
    ("2 chat template", prompt.replace("\n", "\\n"), "the format instruction tuning taught"),
    ("3 pre-tokenize", f"{len(chunks)} chunks", "words keep their leading space; digits split"),
    ("4 BPE -> ids", f"{len(ids)} token ids", "subwords from a learned 400-token vocabulary"),
    ("5 embeddings", f"{memo['embed'][0]} x {memo['embed'][1]} matrix", "token vector + position vector"),
    ("6 transformer x4", f"{memo['embed'][0]} x {memo['embed'][1]} matrix", "attention moves information; MLP transforms it"),
    ("7 logits", f"{len(tok.vocab)} scores", "last position vs every token embedding"),
    ("8 softmax + decode", f"{tok.vocab[int(top[0])]!r} p = {probs[top[0]]:.2f}", "greedy here; temperature / top-p optional"),
    ("9 loop", f"{len(loop)} passes", "append the token, run again, stop at newline"),
    ("10 detokenize", repr(response), "ids -> text"),
]
if HAS_MPL:
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.axis("off")
    for i, (name, value, why) in enumerate(STAGES):
        r, c = divmod(i, 5)
        x, y = c * 0.2 + 0.01, 0.62 - r * 0.5
        ax.add_patch(plt.Rectangle((x, y), 0.18, 0.3, fill=True, color=f"C{r}", alpha=0.15, transform=ax.transAxes))
        ax.text(x + 0.09, y + 0.24, name, ha="center", fontsize=11, weight="bold", transform=ax.transAxes)
        ax.text(x + 0.09, y + 0.14, value if len(value) < 34 else value[:31] + "...", ha="center", fontsize=8.5,
                transform=ax.transAxes, family="monospace")
        ax.text(x + 0.09, y + 0.05, why, ha="center", fontsize=7.5, transform=ax.transAxes, wrap=True)
        if c < 4:
            ax.annotate("", xy=(x + 0.2, y + 0.15), xytext=(x + 0.18, y + 0.15), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.1, 0.42), xytext=(0.9, 0.62), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", ls="--"))
    ax.annotate("", xy=(0.73, 0.62), xytext=(0.73, 0.42), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="C3", lw=2))
    ax.text(0.74, 0.5, "+1 token", color="C3", fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.02, "Stage 9 loops back to stage 4 with one more token until a stop token appears", ha="center",
            transform=ax.transAxes, fontsize=9, style="italic")
    ax.set_title(f"From raw text to a response: '{QUESTION}' -> {response!r}", fontsize=12)
    save_fig(fig, "15_text_to_response_journey")


def md_table(df):
    df = df.reset_index()
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
                      for row in df.itertuples(index=False)])


lines = [f"# From raw text to an LLM-generated response", "",
         f"Traced through the instruction-tuned TinyGPT of Module 7 ({model.n_params:,} parameters, 4 blocks, vocabulary "
         f"of {len(tok.vocab)} tokens). Question: **{QUESTION}** (true answer: {target.value}).", "",
         "![journey](15_text_to_response_journey.png)", "", "## The stages", "",
         "| stage | what comes out here | why |", "|---|---|---|"] + \
        [f"| {n} | `{v}` | {w} |" for n, v, w in STAGES] + \
        ["", "## Inside the transformer (stage 6)", "", md_table(memo["blocks"]), "",
         "## The next-token distribution (stages 7-8)", "", md_table(memo["dist"].set_index("token")), "",
         "## The generation loop (stage 9)", "", md_table(memo["loop"]), "",
         f"**Response:** {response!r}", "",
         "## Where it goes wrong: a question it cannot answer", "",
         f"The same pipeline for **{unk}**, a country with no facts in the training data, answers {response_u!r}: "
         f"a hallucination. 8 samples at T = 1: {', '.join(samples)}.", "", md_table(memo["signals"]), "",
         "The first token is not a reliable warning; the whole-answer probability and the agreement between "
         "samples are.", "",
         "## Explanation", "",
         "An LLM turns text into integers (tokenization), integers into vectors (embeddings), lets the vectors exchange "
         "information through stacked self-attention and MLP blocks, and turns the last position's vector into a "
         "probability for every possible next token. Decoding picks one token, which is appended to the input, and the "
         "whole computation repeats until a stop token. Knowledge comes from pretraining (next-token prediction on "
         "raw text); the question-answer behaviour comes from instruction tuning; the randomness and the risk of "
         "hallucination come from sampling likely, not true, text. Production systems wrap this loop with chat "
         "templates, retrieval (RAG), safety filters and evaluation.", ""]
path = OUT_DIR / "15_text_to_response_journey.md"
path.write_text("\n".join(lines), encoding="utf-8")
print(f"  [saved] outputs/{path.name}")

banner("KEY TAKEAWAYS")
print("""  1. Raw text -> template -> tokens -> ids -> vectors -> transformer blocks -> logits -> probabilities -> token.
  2. Generation is a loop: one full pass per new token, until a stop token.
  3. Every stage is plain arithmetic you can run by hand; nothing in it checks truth.
  4. Knowledge = pretraining; format = instruction tuning; variety and risk = decoding.
  5. Hallucinations travel the same path as facts; the signals (low confidence, disagreement) are in the numbers.""")
