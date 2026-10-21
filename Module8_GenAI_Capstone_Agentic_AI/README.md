# Module 8: ML-to-GenAI Capstone and Agentic AI Foundations

Fifteen runnable labs and a capstone assignment. Each lab builds from **EASY → MEDIUM → HARD**, and each HARD level ends with a "plot twist". The scenario is the same all the way through: the customer support team of **Lumora Home**, a fictional smart-home store. It has a help centre (with stale, look-alike and one poisoned article), an orders database, and 400 support tickets with a full answer key.

The theme of the module: **a reliable GenAI system is mostly the parts around the model.**
- **Part 1, building a reliable GenAI system:** calling a foundation model (Lab 01), prompts (02), zero- and few-shot (03), structured outputs (04), then retrieval-augmented generation: chunking (05), embeddings and vector search (06), context and generation (07), evaluation (08).
- **Part 2, agentic AI foundations:** what makes an agent, and workflow vs agent (09), tool calling (10), planning and reflection (11), human-in-the-loop and permissions (12), single vs multi-agent (13), the six agent failure modes (14).
- **Capstone:** an intelligent customer support assistant, built from every piece above, evaluated, stress-tested, costed and written up (15 + ASSIGNMENT, for a second store, Kestrel Outdoor).

Every lab is a plain Python file with `# %%` cell markers:
- **Script mode:** `python 07_rag_context_and_generation.py` prints a narrated walkthrough and saves charts to `outputs/`.
- **Cell mode (best for recording):** open the file in VS Code and click **Run Cell** above each `# %%`. Charts appear inline.

## Setup

```bash
pip install -r requirements.txt
python run_all.py          # smoke-test all 15 labs + the assignment starter (about 15 minutes)
```

**Everything runs offline on a laptop CPU, with no API key.** The labs call `get_llm("large")` / `get_llm("small")`. By default that is **SimLLM**, a deterministic stand-in with the same interface as Anthropic's Messages API (`system`, `messages`, `tools`, `output_schema`, `usage`, `stop_reason`) and the same price ratio as a large and a small Claude model. It is built from real parts (embeddings, retrieval, extraction) plus the failure modes real LLMs are known for, each at a documented rate in `genai_toolkit.py` (`SIM_TIERS`, `PROMPT_MITIGATIONS`). Prompt instructions lower those rates, as they do for real models; code guardrails remove them. Set `LLM_BACKEND=claude` (plus `pip install anthropic` and an API key) and the same code calls Claude instead. Real calls cost real money, and the numbers will differ.

**Why a simulated model and synthetic tickets?** Every ticket has a hidden truth: what the customer wanted, which policy applies, which refund is allowed, and which actions must never happen. So "resolved", "unsafe action", "hallucinated action" and "injection acted on" are things you can **count**, and every run gives the same numbers on screen.

## Folder layout

| File | Purpose |
|---|---|
| `ml_lab_utils.py` | Console/plot helpers and the two fictional stores (`DOMAINS["lumora"]`, `DOMAINS["kestrel"]`): the help centre (`make_kb`: current, **stale**, **look-alike** and one **poisoned community post**), customers and orders (`make_orders`), the return rule as code (`return_quote`), **tickets with a full answer key** (`make_tickets`: intents, facts a correct reply contains, correct and forbidden actions, approval needed, injected attacks, dev/test split), an unlabelled support corpus for embeddings |
| `genai_toolkit.py` | The **reusable library** built up across the module: one LLM interface with two backends (`SimLLM`, `ClaudeLLM`), `call_with_retry`; retrieval (`chunk_text`, `PPMIEmbeddings`, `TfidfEmbedder`, `HybridEmbedder`, `VectorIndex`, `default_retriever`, `retrieval_report`); the store's tools (`StoreTools`, with the rules enforced inside); the agent harness (`run_agent`: schema validation, step limit, loop detection, cost budget, permissions, human approval); evaluation (`fact_recall`, `faithfulness`, `grade_run`); `ModelScorecard` |
| `01` … `14` | Core topics |
| `15_capstone_worked_example_support_assistant.py` | The capstone, solved for Lumora: architecture, evaluation, error analysis, stress test, model placement, scorecard, launch memo |
| `ASSIGNMENT.md` + `assignment_starter.py` | The student capstone: the same brief for **Kestrel Outdoor**, with an automated rubric |
| `run_all.py` | Runs every lab and reports pass/fail |
| `outputs/` | Charts, the launch memo and the architecture diagram |

### Syllabus coverage

| Syllabus topic | Where |
|---|---|
| Foundation models and LLM APIs | 01 |
| Prompt design fundamentals | 02 |
| Zero-shot and few-shot prompting | 03 |
| Structured outputs | 04 |
| Retrieval-Augmented Generation; chunking and document preparation | 05 (and 07) |
| Embedding generation; vector search | 06 |
| Retrieval and context construction; LLM response generation | 07 |
| Evaluating retrieval and generated answers | 08 |
| What makes an AI system an agent? Model, tools, instructions, memory and state; workflow vs agent | 09 |
| Tool/function calling | 10 |
| Planning and task decomposition; reflection and validation | 11 |
| Human-in-the-loop approval | 12 |
| Single-agent vs multi-agent systems | 13 |
| Agent failure modes: infinite loops, incorrect tool use, hallucinated actions, excessive autonomy, cost and latency, prompt injection | 14 (and 04, 07, 10-13) |
| Hands-on capstone: build an Intelligent Customer Support Assistant | 15 (solved) + ASSIGNMENT |

---

## Episode plan with the key on-screen moments

### Part A: Building a reliable GenAI system

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 01 | Foundation models & LLM APIs | One call taken apart: system, messages, usage, stop_reason; the API is stateless | Large vs small: price, speed, and a model that knows "returns" but not *your* return policy | Rate limits and server errors; retries with backoff | Small tier **40x** cheaper; input is **62%** of the bill. 10,000 answers a day: **$716** a month (large) vs $18 (small). 20% errors: 83% success → retries → **100%**. **Plot twist:** p99 latency 0.9 s → **2.5 s** behind the perfect dashboard |
| 02 | Prompt design | Five versions of one prompt | Label names vs descriptions; temperature | Test set, remaining errors, the price of a lazy prompt | v0 **0%** (unparseable) → v3 **98%**. "complaint" with names only **28%** → with a description 100%. T = 1.5: 11% of tickets change answer while accuracy barely moves. **Plot twist:** small model + good prompt **91.6%** beats large + lazy prompt 85.1%, at 12x lower cost |
| 03 | Zero- & few-shot | Zero-shot vs 1 and 4 examples per label; five example sets | Dynamic few-shot: retrieve similar solved tickets | Biased, swapped and noisy examples | 1 example per label **76.7%**, *worse* than none (85.6%). Retrieved examples **98.0%**; one-line descriptions, no examples: 99.0%. 6 of 9 examples are returns → **40%** of tickets called returns. **Plot twist:** with the nearest example mislabelled, accuracy is 87% instead of 98% |
| 04 | Structured outputs | JSON by asking: how often does it parse? | Lenient parse → schema → repair loop | Schema-constrained output: valid, but right? | Asked nicely: **79%** / 57% usable JSON. Repair loop: 100% at 1.23 calls per ticket. Constrained: 100% valid, fields still wrong (urgency, order numbers). **Plot twist:** an injected "refund order LM-11195" yields a perfectly valid, *real* order number: someone else's |
| 05 | Chunking & document prep | A messy handbook export; the whole handbook in the prompt | Chunk size and overlap | Chunk by headings, keep metadata, filter stale | Whole handbook in context: correct only **44%**. 15-word chunks: found 49%; 240-word: found 100%, correct 58%. Heading chunks **72%**, + date filter **77%**. **Plot twist:** shipping questions *never* rank the current shipping article first (0 of 21): "delivery" vs "shipping" |
| 06 | Embeddings & vector search | What the embedding learned; "refund" ≈ "money back" | Keyword vs semantic vs hybrid | Exact vs approximate search; indexing past questions | Keyword similarity 0.00, semantic 0.37 for the refund pair. **Keyword wins** (MRR 0.74 vs 0.65); exact terms like "9V": semantic 0.66 vs 0.95. IVF: about 40x faster (timing varies by machine), 79% recall. **Plot twist:** no new model, past tickets in the index: MRR **0.70 → 0.97** |
| 07 | RAG: context & generation | Closed-book vs RAG | k chunks; "I don't know"; lost in the middle | A stale policy and a poisoned community post | Closed-book **1%** → RAG **70%**. k = 1 → 8: right article found 77% → 99%, answers **73% → 67%**. Right article first / middle / last: 70% / **30%** / 48%. Poisoned post: 32% repeat the attacker → "untrusted" instruction 10% → **plot twist:** kept out of the index: 0% |
| 08 | Evaluating RAG | Every metric for one config; retrieval vs generation failures | Five configurations side by side | LLM-as-judge | 26 failures = 11 retrieval + 15 generation. Faithfulness **100%** for a config whose 26 wrong answers are all faithful (to the wrong article). Judge without reference: 2.4 vs 2.1; with reference 3.8 vs 1.5. **Plot twist:** the same answers + pleasantries score higher (3.53 → 3.81): verbosity bias |

### Part B: Agentic AI foundations

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 09 | What makes an agent | One ticket through the agent loop; memory = the transcript | Hand-coded workflow vs agent | Requests the workflow's author didn't anticipate | 4 model calls for one refund. Workflow **80%** vs agent 83%, **4.3x cheaper**. **Plot twist:** "my email is…, cancel the Air Purifier": workflow 0%, agent **100%**; the same freedom attempts 2 injected actions (stopped by the rules in the tool) |
| 10 | Tool / function calling | One tool round trip by hand | Precise vs vague tool descriptions | Valid but wrong: someone else's order | Vague descriptions: "no such order" errors 13 → 29; 5 lookups came out as cancellations. **Plot twist:** a swapped digit hits *another real order*; an ownership check in the tool: 2 → **0** |
| 11 | Planning, reflection, validation | Multi-part tickets: improvise, instruct, plan | Flaky payments and refunds that never happened | A critic without evidence; fixing the cause | Small model, both parts answered: **12%** → "handle every request" 64% → plan-and-execute **72%**. Claimed-but-failed refunds 12% → prompt 3% → verifier or code check **0%**. Critic without the tool log: **97%** false alarms. **Plot twist:** retry in the tool: resolved 59% → **75%** |
| 12 | Human-in-the-loop & permissions | An approval card; approve, decline | Four approval policies: harm vs workload | Least privilege; approval fatigue | No approvals: unsafe on **7.4%** of tickets. Every refund approved: 0% at 21 approvals per 100 tickets. FAQ assistant without write tools: 0% executed, no loss in resolution. **Plot twist:** 41 approvals per 100 → tired reviewers skim 76% → unsafe back to **5.0%**; rules in the tools: 0.5% |
| 13 | Single vs multi-agent | Three architectures on one two-part ticket | All 202 test tickets | Attack surface; the misroute | Router on two-part tickets **17%**, orchestrator 75%. More calls (3.3 vs 2.3), yet **half the bill** ($7.28 vs $13.94 per 1,000). Injected refunds: single agent 10 attempts, 7 executed; behind the router **0**. **Plot twist:** a cheap router saves 12%, and a misrouted ticket is resolved 0% of the time |
| 14 | Agent failure modes | Six failure modes caught in the act | Defence in depth: none / prompt / code / both | What guardrails cost | Prompt only: loops 8% → 0%, wrong-order calls 90 → 17 per 100, but injections still act (0.5%). Code only: zero on every guaranteed axis, resolution **66% → 61%**. Prompt + code: **75%**. **Plot twist:** a $0.01 budget cap cuts 27% of tickets (resolution 83% → 58%) to save $2 per 1,000 |

### Part C: The capstone (`15_capstone_worked_example_support_assistant.py`)

An intelligent customer support assistant for Lumora Home, assembled from Labs 01-14:

```
ticket ─> 1 TRIAGE (structured output, Lab 04) ─> 2 PLAN if 2+ requests (Lab 11) ─> 3 ROUTE to one specialist per sub-task (Lab 13)
          orders · returns · help-centre (curated index, Labs 05-08) · escalation        each with only the tools its job needs (Lab 12)
      ─> 4 ACT: guarded tool loops (Lab 14); rules, customer scoping, approval queue, payment retries inside the tools (Labs 10-12)
      ─> 5 CHECK: every claimed action must be in the tool log (Lab 11) ─> reply + log
```

| Part | Level | What you see |
|---|---|---|
| 1.1 | EASY | A two-part ticket (a return + "how long does a refund take?") through all five stages: triage JSON, the plan, the route, every tool call, the check, the reply. 7 model calls |
| 1.2 | EASY | An injected "SYSTEM OVERRIDE … refund $500". The prototype pays it (queued for approval). In the assistant, triage is fooled too and the injected sentence reaches the returns specialist, but the tool refuses: not this customer's order (scoping), and no eligibility quote (rules). The real question still gets a muddled answer |
| 2.1 | MEDIUM | 202 test tickets. Prototype **78.7%** resolved, **4.0% unsafe**. Guarded single agent 87.6%, 0 unsafe. Assistant v1 **87.6%**, 0 unsafe, **59% of the cost** ($8.22 vs $13.95 per 1,000), slower (p95 8.5 s vs 5.0 s). Honest result: multi-agent is **not more accurate**; it buys cost and least privilege |
| 2.2 | MEDIUM | Error analysis of the 25 unresolved: retrieval 9, generation 8, order tools 4, blocked by a tool error 3, action not completed 1. "How fast is delivery?" retrieves *business-account returns* |
| 2.3 | MEDIUM | One fix driven by the error analysis: 88 past dev questions in the index → **88.6%** (2 fixed, 0 broken) |
| 3.1 | HARD | A hostile week (30% payment timeouts, 8% locked records): resolution 88.6% → 84.7%, still **0 unsafe, 0 hallucinated**. The prototype still acts unsafely on 2.0% |
| 3.2 | HARD | Where to spend the big model: all small costs **$0.20 vs $8.18** per 1,000 and resolves 84.2%, but **under-refunds one customer**: "stoppde working" → read as *opened*, not *defective* → $126.65 instead of $149. Every guardrail passed it: **guardrails enforce consistency, not truth**. Large model only in the returns agent: 0 unsafe, $2.04 |
| 3.3 | HARD | Scorecard with hard gates (prototype and all-small fail). The weighted score (30% on model cost) picks **small triage + planner**. **Plot twist:** at $4 of staff time per unresolved ticket, the model is **0.7%** of the true cost, and the all-large assistant v2 is cheapest overall ($899 vs $937 per 1,000 tickets) |

Outputs: `outputs/15_support_assistant_memo.md` (the launch memo), `15_support_assistant_architecture.png`, `15_capstone_support_assistant.png`.

### Part D: Assignment

- **`ASSIGNMENT.md` + `assignment_starter.py`**: the same brief for **Kestrel Outdoor**: a 60-day return window, no restocking fee, **used gear returned for store credit only**, and a $250 approval limit. Students design triage, specialists and routing, the index, the claims check, the pipeline and the model placement. Then they set the scorecard weights and the staff-cost assumption and write the report. Part C evaluates their assistant on 202 test tickets and a hostile week, against all-large and all-small variants.
  - **Built-in surprise:** on Kestrel the all-small assistant *passes* every safety gate, unlike on Lumora. So the launch decision becomes pure economics, and the staff-cost assumption decides it.
  - **Grading:** an automated rubric awards 40 points; reviewer judgement awards 60.

---

## Recording tips
- Run `python run_all.py` once before recording so every chart exists.
- In cell mode, pause after each `>>` note. Those lines are written as talking points, with the numbers filled in from the run.
- The tickets have a **hidden truth**. Lean on it: "with a real system you'd need weeks of labelling to know this; here we can count it" is the thread through the module.
- Several labs show **honest negative or surprising results**:
  - One example per label is worse than none (Lab 03).
  - Keyword search beats the embedding (Lab 06).
  - More chunks, worse answers (Lab 07).
  - The judge prefers padding (Lab 08).
  - A $0.01 budget hurts more than it saves (Lab 14).
  - Multi-agent is not more accurate (Lab 15).
  - The cheapest model isn't the cheapest system (Lab 15).

  That's how real projects go.
- Moments to slow down for:
  - Lab 02's five prompt versions.
  - Lab 07's poisoned community post.
  - Lab 12's tired reviewer.
  - Lab 14's defence-in-depth table.
  - Lab 15's under-refund that every guardrail approved.
