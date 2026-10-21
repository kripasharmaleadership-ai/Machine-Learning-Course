"""
MODULE 7 ASSIGNMENT - STARTER: The Journey from Raw Text to an LLM-Generated Response
=====================================================================================
Read ASSIGNMENT.md first. The worked example (a different question, the full trace and memo) is
15_assignment_worked_example_text_to_response.py.

PART A (model, questions, checks) and PART C (experiments, rubric, report) are written for you.
Your work is PART B: rebuild the pipeline yourself (tokenizer, embeddings, forward pass, sampling, the
generation loop), describe it as a diagram, and write the report.

Run it as often as you like: Part C prints the automated rubric every time.
"""
# %%
import numpy as np
import pandas as pd

from ml_lab_utils import HAS_MPL, OUT_DIR, banner, note, plt, save_fig, show, step
from nn_toolkit import HAS_TORCH, QA_QUESTIONS, _PRETOK

if not HAS_TORCH:
    raise SystemExit("This assignment needs PyTorch:  pip install torch")
import torch

from nn_toolkit import answer_of, generate, world_chat_lm

torch.set_num_threads(8)

# %% =========================================================================
# PART A - The model, the questions, the checks (done for you)
# ============================================================================
banner("PART A - The instruction-tuned TinyGPT and your questions")
chat = world_chat_lm()
model, tok, facts = chat.model, chat.tok, chat.facts
split = chat.info["split"]
show("Model / vocabulary / context", f"{model.n_params:,} parameters / {len(tok.vocab)} tokens / {model.context} tokens")
held = facts[facts.known & facts.country.isin(split["test"])]
show("Known countries NOT used in instruction tuning (pick from these)", sorted(held.country.unique()))
show("Countries the model knows nothing about (pick from these)", sorted(facts[~facts.known].country.unique()))
show("Question templates", list(QA_QUESTIONS.values()))
WORKED_EXAMPLE_RELATION = "capital"                                 # Lab 15 traced a capital question: choose another
_results = {}


def template(question: str) -> str:
    """The chat template the model was instruction-tuned on."""
    return f"Q: {question}\nA:"


def _try(name, fn):
    try:
        out = fn()
        _results[name] = bool(out)
    except NotImplementedError:
        _results[name] = False
        print(f"  [todo] {name}")
    except Exception as e:                                           # show students what broke, keep going
        _results[name] = False
        print(f"  [error] {name}: {type(e).__name__}: {e}")


# %% =========================================================================
# PART B - YOUR WORK
# ============================================================================
banner("PART B - Rebuild the pipeline (your work)")

step("B1  Choose your two questions")
# TODO: one question about a KNOWN country from the first list, one about an UNKNOWN country from the second.
#       Use a "language" or "export" question (the worked example used "capital").
QUESTION_KNOWN = ""      # e.g. QA_QUESTIONS["language"].format(c="<a known country>")
QUESTION_UNKNOWN = ""    # e.g. QA_QUESTIONS["export"].format(c="<an unknown country>")

step("B2  Tokenization: apply the BPE merges yourself")


def my_encode(text: str) -> list[int]:
    """text -> token ids, WITHOUT calling tok.encode / tok.tokens.
    1. split the text into chunks with the pre-tokenizer:  _PRETOK.findall(text)
    2. for each chunk, start from its characters and repeatedly merge the adjacent pair with the LOWEST rank in
       tok.merges (a dict {(a, b): rank}) until no adjacent pair is in tok.merges
    3. map every token string to its id with tok.id_of (unknown strings -> 0)"""
    raise NotImplementedError  # TODO


step("B3  Embeddings")


def my_embed(ids: list[int]) -> torch.Tensor:
    """ids -> (T, d) input of the first transformer block: token embedding + position embedding.
    Use the weight matrices model.tok.weight (vocab x d) and model.pos.weight (context x d) directly."""
    raise NotImplementedError  # TODO


step("B4  The forward pass: next-token probabilities")


@torch.no_grad()
def my_next_token_probs(ids: list[int]) -> torch.Tensor:
    """ids -> probability of every vocabulary token at the next position (a tensor of len(tok.vocab)).
    my_embed -> every block in model.blocks (each takes a (1, T, d) tensor) -> model.ln -> model.head ->
    the LAST position's logits -> softmax."""
    raise NotImplementedError  # TODO


step("B5  Decoding: temperature and top-p")


def my_sample(probs: torch.Tensor, temperature: float, top_p: float, rng: np.random.Generator) -> int:
    """Pick one token id.
    temperature == 0: the most likely token (greedy).
    otherwise: re-weight with p ** (1 / temperature) (same as dividing the logits by T), renormalise, keep the
    smallest set of most likely tokens whose total probability reaches top_p, renormalise, and draw one with rng."""
    raise NotImplementedError  # TODO


step("B6  The generation loop")


def my_generate(question: str, temperature: float = 0.0, top_p: float = 1.0, seed: int = 0, max_new_tokens: int = 15):
    """question -> (response_text, steps) using the chat template and YOUR functions above. Stop when the chosen
    token contains a newline. steps: one dict per generated token with at least 'token' and 'probability'
    (the probability of the chosen token BEFORE temperature/top-p). response_text excludes the newline."""
    raise NotImplementedError  # TODO


step("B7  The diagram")
# TODO: describe the journey as a list of stages, in order. At least 8 stages; each with what goes in, what
#       comes out (with the SHAPE or size for YOUR known question, e.g. "16 token ids", "16 x 128"), and why.
#       The stages must cover: tokens, embeddings, attention, logits, softmax / sampling, and the loop.
DIAGRAM = [
    # {"stage": "raw text", "in": "...", "out": "...", "why": "..."},
]

step("B8  The report")
# TODO: write each section in your own words, citing YOUR numbers from Part C. Word minimums: JOURNEY 150,
#       KNOWN VS UNKNOWN 80 (at least 3 numbers), DECODING 60 (temperature AND top-p), REAL CHATBOTS 80
#       (what production systems add around this loop), LIMITATIONS 40.
REPORT = {
    "JOURNEY": """
""",
    "KNOWN VS UNKNOWN": """
""",
    "DECODING": """
""",
    "REAL CHATBOTS": """
""",
    "LIMITATIONS": """
""",
}

# %% =========================================================================
# PART C - Checks, experiments, rubric, report (done for you)
# ============================================================================
banner("PART C - Check your pipeline, run the experiments, write the report")
rel_of = {v.split("{c}")[0]: k for k, v in QA_QUESTIONS.items()}
TEST_STRINGS = [tok.decode(tok.encode(s)) for s in chat.text.split("\n")[:40]] + [
    template(QA_QUESTIONS["language"].format(c=c)) for c in sorted(held.country.unique())[:5]] + ["blorptastic 2026 Zzq!"]
_try("B2 my_encode matches the tokenizer on 46 strings", lambda: all(my_encode(s) == tok.encode(s) for s in TEST_STRINGS))
probe_ids = tok.encode(template(QA_QUESTIONS["export"].format(c=sorted(held.country.unique())[0])))
_try("B3 my_embed matches the model", lambda: torch.allclose(
    my_embed(probe_ids), model.tok(torch.tensor(probe_ids)) + model.pos(torch.arange(len(probe_ids))), atol=1e-6))


def _ref_probs(ids):
    with torch.no_grad():
        return torch.softmax(model(torch.tensor([ids]))[0, -1], -1)


_try("B4 my_next_token_probs matches the model", lambda: torch.allclose(my_next_token_probs(probe_ids), _ref_probs(probe_ids), atol=1e-5))


def _check_sampling():
    p = _ref_probs(probe_ids)
    rng = np.random.default_rng(0)
    greedy_ok = my_sample(p, 0.0, 1.0, rng) == int(p.argmax())
    draws = np.array([my_sample(p, 1.0, 1.0, rng) for _ in range(3000)])
    top = int(p.argmax())
    freq_ok = abs(np.mean(draws == top) - float(p[top])) < 0.04
    order = torch.argsort(p, descending=True)
    nucleus = set(order[:int((torch.cumsum(p[order], 0) < 0.5).sum()) + 1].tolist())
    top_p_ok = all(my_sample(p, 1.0, 0.5, rng) in nucleus for _ in range(300))
    flat = [my_sample(p, 3.0, 1.0, rng) for _ in range(1000)]
    temp_ok = np.mean(np.array(flat) == top) < float(p[top]) - 0.05 or float(p[top]) < 0.1
    return greedy_ok and freq_ok and top_p_ok and temp_ok


_try("B5 my_sample: greedy, frequencies, top-p and temperature behave correctly", _check_sampling)
_try("B6 my_generate (greedy) matches the reference generator", lambda: all(
    my_generate(q)[0] == generate(model, tok, template(q), max_new_tokens=15, temperature=0)[0]
    for q in [QUESTION_KNOWN, QUESTION_UNKNOWN, QA_QUESTIONS["capital"].format(c=sorted(held.country.unique())[0])]
    if q) and bool(QUESTION_KNOWN and QUESTION_UNKNOWN))

step("C1  Your two questions")
rows = {}
valid_q = bool(QUESTION_KNOWN and QUESTION_UNKNOWN)
if valid_q:
    kc = next((c for c in held.country.unique() if f" {c}?" in QUESTION_KNOWN), None)
    uc = next((c for c in facts[~facts.known].country.unique() if f" {c}?" in QUESTION_UNKNOWN), None)
    rk = next((r for pre, r in rel_of.items() if QUESTION_KNOWN.startswith(pre)), None)
    ru = next((r for pre, r in rel_of.items() if QUESTION_UNKNOWN.startswith(pre)), None)
    valid_q = bool(kc and uc and rk and ru) and WORKED_EXAMPLE_RELATION not in (rk, ru)
    _results["B1 one known + one unknown question, not the worked example's relation"] = valid_q
else:
    _results["B1 one known + one unknown question, not the worked example's relation"] = False
if valid_q and _results.get("B6 my_generate (greedy) matches the reference generator"):
    truth = facts[(facts.country == kc) & (facts.relation == rk)].value.iloc[0]
    for label, q in [("known", QUESTION_KNOWN), ("unknown", QUESTION_UNKNOWN)]:
        text, steps_ = my_generate(q)
        samples = [answer_of(my_generate(q, temperature=1.0, seed=s)[0]) for s in range(10)]
        rows[label] = {"question": q, "greedy answer": text.strip(), "tokens generated": len(steps_),
                       "first-token probability": steps_[0]["probability"],
                       "whole-answer probability": float(np.prod([s_["probability"] for s_ in steps_])),
                       "10 samples at T=1 agreeing with the most common": max(samples.count(x) for x in samples) / 10,
                       "distinct answers in 10 samples": len(set(samples))}
    rows["known"]["true answer"] = truth
    rows["unknown"]["true answer"] = "(none: never in training)"
    experiments = pd.DataFrame(rows).T
    print(experiments.to_string())
    _results["C1 experiments ran on both questions"] = True
else:
    experiments = None
    _results["C1 experiments ran on both questions"] = False
    note("Finish B1-B6 to run the experiments on your two questions.")

step("C2  Rubric")
need = {"token": ["token"], "embedding": ["embed"], "attention": ["attention", "transformer"], "logits": ["logit"],
        "softmax / sampling": ["softmax", "sampl", "decod"], "loop": ["loop", "autoregress", "append"]}
diag_text = " ".join(" ".join(str(v) for v in d.values()).lower() for d in DIAGRAM)
diag_ok = (len(DIAGRAM) >= 8 and all(all(str(d.get(k, "")).strip() for k in ["stage", "in", "out", "why"]) for d in DIAGRAM)
           and all(any(w in diag_text for w in ws) for ws in need.values()))
words = {k: v.split() for k, v in REPORT.items()}
nums = lambda k: sum(any(ch.isdigit() for ch in w) for w in words[k])
dec = REPORT["DECODING"].lower()
report_ok = (len(words["JOURNEY"]) >= 150 and len(words["KNOWN VS UNKNOWN"]) >= 80 and nums("KNOWN VS UNKNOWN") >= 3
             and len(words["DECODING"]) >= 60 and "temperature" in dec and "top-p" in dec.replace("top p", "top-p")
             and len(words["REAL CHATBOTS"]) >= 80 and len(words["LIMITATIONS"]) >= 40)
checks = {
    "B1 one known + one unknown question, not the worked example's relation (4 pts)":
        _results["B1 one known + one unknown question, not the worked example's relation"],
    "B2 tokenizer matches (6 pts)": _results.get("B2 my_encode matches the tokenizer on 46 strings", False),
    "B3 embeddings match (4 pts)": _results.get("B3 my_embed matches the model", False),
    "B4 forward pass matches (6 pts)": _results.get("B4 my_next_token_probs matches the model", False),
    "B5 sampling behaves correctly (6 pts)": _results.get("B5 my_sample: greedy, frequencies, top-p and temperature behave correctly", False),
    "B6 generation loop matches (4 pts)": _results.get("B6 my_generate (greedy) matches the reference generator", False),
    "B7 diagram: >= 8 complete stages covering tokens, embeddings, attention, logits, sampling, loop (4 pts)": diag_ok,
    "B8 report: word minimums, >= 3 numbers in KNOWN VS UNKNOWN, temperature and top-p in DECODING (6 pts)": report_ok,
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
points = sum(int(k.split("(")[-1].split()[0]) for k, v in checks.items() if v)
show("Automated rubric points", f"{points} / 40")

if all(checks.values()):
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(15, 1.6 + 0.9 * ((len(DIAGRAM) + 4) // 5)))
        ax.axis("off")
        n_rows = (len(DIAGRAM) + 4) // 5
        for i, d in enumerate(DIAGRAM):
            r, c = divmod(i, 5)
            x, y = c * 0.2 + 0.01, 1 - (r + 1) / n_rows + 0.05
            ax.add_patch(plt.Rectangle((x, y), 0.18, 0.8 / n_rows, color=f"C{r}", alpha=0.15, transform=ax.transAxes))
            ax.text(x + 0.09, y + 0.6 / n_rows, f"{i + 1} {d['stage']}", ha="center", weight="bold", fontsize=9, transform=ax.transAxes)
            ax.text(x + 0.09, y + 0.35 / n_rows, str(d["out"])[:34], ha="center", fontsize=8, family="monospace", transform=ax.transAxes)
            ax.text(x + 0.09, y + 0.12 / n_rows, str(d["why"])[:48], ha="center", fontsize=7, transform=ax.transAxes)
        ax.set_title(f"From raw text to a response: {QUESTION_KNOWN!r}")
        save_fig(fig, "assignment_text_to_response_diagram")
    lines = ["# From raw text to an LLM-generated response", "", "![diagram](assignment_text_to_response_diagram.png)", "",
             "## The stages", "", "| # | stage | in | out | why |", "|---|---|---|---|---|"]
    lines += [f"| {i + 1} | {d['stage']} | {d['in']} | {d['out']} | {d['why']} |" for i, d in enumerate(DIAGRAM)]
    lines += ["", "## Experiments", ""] + [f"- **{k}**: " + "; ".join(f"{c}: {v}" for c, v in r.items())
                                           for k, r in experiments.iterrows()] + [""]
    lines += [f"## {k.title()}\n\n{v.strip()}\n" for k, v in REPORT.items()]
    (OUT_DIR / "assignment_text_to_response_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("  [saved] outputs/assignment_text_to_response_report.md")
else:
    note("""Not all checks pass yet - keep going with PART B. The report and the diagram
are written automatically once every check is PASS.""")
