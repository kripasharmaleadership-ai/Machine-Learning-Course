# Module 7 Assignment: The Journey from Raw Text to an LLM-Generated Response

> *"An LLM is a next-token classifier in a loop."* Prove it by rebuilding the loop yourself, then explain every step to someone who has never seen inside one.

## The scenario

Your company is rolling out an internal chat assistant, and the leadership team keeps asking the same questions. *How does it actually produce an answer? Why does it give different answers to the same question? Why does it sometimes invent things with total confidence?*

You will answer with evidence. You get a real (tiny) instruction-tuned language model: the TinyGPT of Labs 11-13, trained on an invented world where every fact is known. Your job:

1. **Rebuild** its pipeline from the pieces: tokenizer, embeddings, forward pass, sampling, generation loop.
2. **Trace** two questions through it: one the model can answer, one it cannot.
3. **Draw and explain** the journey from raw text to response (the syllabus assignment).

Reference solution for a *different* question: [15_assignment_worked_example_text_to_response.py](15_assignment_worked_example_text_to_response.py). It traces a capital question stage by stage, contrasts it with a hallucination, and writes a memo with a diagram. Study it first. Your questions must be about a **language or an export** (not a capital), and your conclusions may differ.

## What you get

| object | contents |
|---|---|
| `model` | the instruction-tuned TinyGPT (852,736 parameters, 4 blocks, 128 wide, 64-token context) |
| `tok` | its BPE tokenizer: `tok.merges` (pair → rank), `tok.id_of` (token → id), `tok.vocab` (id → token) |
| `facts` | every fact in the world, with how often it appeared in training (`repeats`); unknown countries have none |
| `template(q)` | the chat template the model was tuned on: `"Q: {q}\nA:"` |
| `generate(...)` | the reference generator, used only to check your loop |

The first run trains and caches the models (a few minutes on a CPU). Later runs start in seconds.

## What to do

Work in **Part B** of [assignment_starter.py](assignment_starter.py). Part A (model, questions, helpers) and Part C (checks, experiments, rubric, report) are written for you.

1. **B1 Questions:** one about a **known** country that was *not* used in instruction tuning, and one about an **unknown** country. Both must be language or export questions. The starter prints the eligible countries.
2. **B2 Tokenizer:** `my_encode(text)` applies the BPE merges yourself: pre-tokenize, merge the lowest-ranked pair until none is left, map to ids. Don't call `tok.encode`.
3. **B3 Embeddings:** `my_embed(ids)` adds a token vector and a position vector, read straight from the weight matrices.
4. **B4 Forward pass:** `my_next_token_probs(ids)` runs the blocks, the final LayerNorm and the output layer, then a softmax over the last position.
5. **B5 Decoding:** `my_sample(probs, temperature, top_p, rng)` covers greedy, temperature and nucleus (top-p) sampling.
6. **B6 Loop:** `my_generate(question, ...)` applies the template, then generates one token at a time until a newline, recording each step.
7. **B7 Diagram:** fill `DIAGRAM` with at least 8 stages. Each stage gives what goes in, what comes out (with **your** shapes and sizes) and why. Part C draws it.
8. **B8 Report:** write the sections below, citing the numbers Part C prints for your two questions.

   | section | minimum | content |
   |---|---|---|
   | Journey | 150 words | the stages in order, in plain language, with your shapes |
   | Known vs unknown | 80 words, 3 numbers | what differs between the two traces, and what doesn't |
   | Decoding | 60 words | what **temperature** and **top-p** change, and when you'd use each |
   | Real chatbots | 80 words | what production systems add around this loop: system prompts, chat templates, retrieval (RAG), the KV cache, safety filters, evaluation |
   | Limitations | 40 words | what a 850k-parameter model on an invented world can and cannot tell you about real LLMs |

## Deliverables

| # | File | Produced by |
|---|---|---|
| 1 | `outputs/assignment_text_to_response_report.md` | starter (auto, when all checks pass) |
| 2 | `outputs/assignment_text_to_response_diagram.png` | starter (auto, from your `DIAGRAM`) |
| 3 | Your completed `assignment_starter.py` | you |
| 4 | **Stretch (optional):** see below | you |

## Grading rubric (100 points)

**Automated checks (40 pts).** These are printed by `assignment_starter.py`, Part C:

| check | pts |
|---|---|
| One known + one unknown question, not about capitals | 4 |
| `my_encode` matches the tokenizer on 46 strings | 6 |
| `my_embed` matches the model | 4 |
| `my_next_token_probs` matches the model | 6 |
| `my_sample`: greedy, sampling frequencies, top-p support, temperature flattening | 6 |
| `my_generate` (greedy) matches the reference generator | 4 |
| Diagram: ≥ 8 complete stages covering tokens, embeddings, attention, logits, sampling, the loop | 4 |
| Report: word minimums, ≥ 3 numbers in *Known vs unknown*, temperature and top-p in *Decoding* | 6 |

**Reviewer judgement (60 pts):**

| criterion | pts | what "excellent" looks like |
|---|---|---|
| The journey | 20 | Every stage explained so a non-specialist could follow; correct shapes; why tokens aren't words; why generation is a loop; where pretraining vs instruction tuning show up |
| The diagram | 10 | Readable on its own; the loop back from the chosen token to the input is visible; your real numbers, not generic ones |
| Known vs unknown | 12 | The two traces compared with numbers (first-token probability, whole-answer probability, sample agreement), and an honest discussion of whether those signals actually separated knowledge from invention *for your questions* |
| Decoding | 8 | Temperature and top-p explained with what they do to the distribution, plus a sensible recommendation per use case |
| Real systems and limitations | 10 | A concrete account of what wraps the loop in production, and of what this tiny model cannot show (scale, multilingual tokenizers, RLHF, tools) |

## Hints

- **BPE order matters** (Lab 10): at each step, merge the adjacent pair with the *lowest* rank, then look again. Merging left to right gives different tokens on some words.
- **Positions:** `model.pos.weight[:T]` gives positions 0..T-1. Forgetting them makes B3 fail, and silently turns the model into a bag of words (Lab 09).
- **Probabilities to record:** in `my_generate`, record the probability of the chosen token *before* temperature and top-p. That's the model's own confidence, and it's what Part C compares.
- **Built-in surprise:** don't assume the unknown question will look uncertain. Some hallucinations come back with near-certain probability and identical answers in every sample (Lab 13: confidence separates knowledge from invention only on average). If yours does, it's the most interesting thing in your report.
- **A blank answer is not "I don't know":** this model was not refusal-tuned. Compare Lab 13's `world_chat_lm(refusals=True)` in a stretch goal.

## Stretch goals

- **Refusals:** trace your unknown question through `world_chat_lm(refusals=True)`. Where in the journey does "I don't know" come from, and what did it cost on known questions?
- **Grounding:** put the answer in the prompt (`"People in X speak Seren.\n" + template(q)`) and trace it again with `world_icl_lm()` (Lab 12). Which attention block reads the inserted fact?
- **The KV cache:** modify `my_generate` so each step only runs the new token through the blocks, reusing cached keys and values. Check the output is identical, and time both versions.
- **A real model:** if `transformers` is installed, trace the same stages in GPT-2 (`AutoTokenizer`, `AutoModelForCausalLM`): vocabulary size, embedding width, number of blocks, the next-token distribution. What changes with scale, and what doesn't?
