# Module 7: Neural Networks and the Road to Generative AI

[🏠 Course home](../README.md) · [📋 Syllabus](../SYLLABUS.md) · [← Module 6](../Module6_Unsupervised_Recommendations_Embeddings) · **Module 7 of 8** · [Module 8 →](../Module8_GenAI_Capstone_Agentic_AI)

Fifteen runnable labs and an assignment. Each lab builds from **EASY → MEDIUM → HARD**, and each HARD level ends with a "plot twist". The scenarios are spirals, handwritten digits, sequences with hidden rules, product reviews, and an **invented world** that a tiny GPT learns from scratch.

The theme of the module: **open the box, one piece at a time, until the box is an LLM.**
- **Neural networks from scratch (NumPy):** why classical ML runs out (Lab 01), the neuron (02), forward pass and loss (03), backprop (04), optimizers (05), overfitting and dropout (06). Every number is visible.
- **Architectures (PyTorch):** CNNs for images (07), RNNs and LSTMs for sequences (08), attention and the transformer, encoder vs decoder (09).
- **From ML to LLMs:** tokenization and embeddings (10), pretraining (11), instruction tuning, fine-tuning and in-context learning (12), temperature and hallucinations (13), all measured on a TinyGPT whose every fact we know.
- **Hands-on and assignment:** a neural sentiment classifier (14); the journey from raw text to a response, traced number by number (15 + ASSIGNMENT).

Every lab is a plain Python file with `# %%` cell markers:
- **Script mode:** `python 11_pretraining_and_self_attention.py` prints a narrated walkthrough and saves charts to `outputs/`.
- **Cell mode (best for recording):** open the file in VS Code and click **Run Cell** above each `# %%`. Charts appear inline.

## Setup

```bash
pip install -r requirements.txt
python run_all.py          # smoke-test all 15 labs + the assignment starter
```

**Everything runs offline on a laptop CPU, with no API key and no GPU.** Labs 01-06 need only NumPy and scikit-learn; Labs 07-15 use PyTorch (CPU). The first run trains and **caches** five TinyGPT variants in `outputs/` (base, small, tiny, instruction-tuned, context-skills; about 8 minutes in total), so later runs of Labs 10-15 start in seconds. Delete `outputs/*.pt` to retrain. If `transformers` is installed, Lab 14 adds a pretrained DistilBERT to its comparison. All randomness is seeded (tested with PyTorch 2.10, scikit-learn 1.9, numpy 2.2).

**Why an invented world?** A model trained on the internet "knows" things nobody can list. A model trained on 48 invented countries, each fact written 1, 2, 4, 8 or 16 times, plus 10 countries with no facts at all, knows only what we wrote. So every answer can be checked, and knowledge, phrasing, frequency and hallucination become things you can **measure**.

## Folder layout

| File | Purpose |
|---|---|
| `ml_lab_utils.py` | Console/plot helpers and synthetic data with **known truth**: **spirals**, **shifted digits** (8x8 digits anywhere on a 16x16 canvas), **sequence tasks** ("A before B", "remember the first token"), **product reviews** (negation, contrast, shortcut endings), the **invented world** and **in-context-learning episodes** |
| `nn_toolkit.py` | A **reusable library** built up across the module. NumPy: activations, stable losses, `Perceptron`, `MLP` (forward, backward, dropout, L2), `numerical_gradient`, `SGD`/`Momentum`/`Adam`, `conv2d`, `rnn_forward`, `attention`. PyTorch: `TransformerBlock` (causal or not), `TinyGPT`, `train_classifier`, `train_lm_stream` (pretraining), `train_lm_examples` (answer-only fine-tuning), `generate` (temperature, top-k, top-p), `continuation_logprob`. `BPETokenizer`. The world models: `world_lm`, `world_chat_lm`, `world_icl_lm`, `probe_facts`. `ModelScorecard` |
| `01` … `13` | Core topics |
| `14_hands_on_lab_neural_sentiment_classifier.py` | The hands-on lab: a small neural classifier and a tiny transformer, error analysis, shortcut test, scorecard and memo |
| `15_assignment_worked_example_text_to_response.py` | Reference solution: a question traced from raw text to response, with diagram and memo |
| `ASSIGNMENT.md` + `assignment_starter.py` | The student assignment: rebuild the pipeline, trace two questions, draw and explain the journey, with an automated rubric |
| `run_all.py` | Runs every lab and reports pass/fail |
| `outputs/` | Charts, cached TinyGPTs and tokenizer, the sentiment memo, the journey memo and diagram |

### Syllabus coverage

| Syllabus topic | Where |
|---|---|
| Why classical ML is not sufficient for every problem | 01 |
| Perceptron and artificial neuron; weights, biases, activation functions | 02 |
| Forward propagation; loss functions | 03 |
| Backpropagation intuition | 04 |
| Optimizers and learning rates; epochs, batches, iterations; trainable parameters | 05 |
| Overfitting and dropout; deep-learning overview; feed-forward neural networks | 06 |
| CNNs for images | 07 |
| RNNs and LSTMs for sequences | 08 |
| Attention mechanism; transformer architecture; encoder and decoder models | 09 |
| From ML to LLMs; tokenization; embeddings | 10 |
| Self-attention; pretraining | 11 (and 09) |
| Instruction tuning; fine-tuning; in-context learning | 12 |
| Temperature and probabilistic generation; hallucinations | 13 |
| Hands-on lab: a small neural-network classifier (or a pretrained transformer) for sentiment | 14 |
| Assignment: draw and explain the journey from raw text to an LLM-generated response | 15 (solved) + ASSIGNMENT |

---

## Episode plan with the key on-screen moments

### Part A: Neural networks from scratch (NumPy)

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 01 | Why classical ML is not enough | Spirals: logistic regression vs a 1,185-weight network | Feature engineering rescues spirals; fails on moving pixels and word order | On tabular data, gradient boosting wins | Logistic **65%**, network **100%**. The spiral's own equation as features: 100%. Digits moved around: logistic **97% → 18%**. Token counts for "A before B": 49%. Loan table: boosting 73.4%, network 71.7% |
| 02 | Perceptron & neuron | A loan neuron by hand; the perceptron learns AND/OR | Activations and their slopes; a sigmoid neuron = sklearn's logistic regression | XOR breaks the perceptron; two layers fix it; depth without activation collapses | Our neuron's weights equal sklearn's to 3 decimals. XOR: 4 mistakes every epoch, forever. 3 linear layers and 2,241 weights score like 1 neuron (0.66); add ReLU: **0.997** |
| 03 | Forward pass & loss | A 2-3-1 network by hand, then as matrices | Cross-entropy vs squared error; overflow traps; what an LLM's loss means | Accuracy has no slope; loss measures confidence | Confidently wrong costs 4.6 vs 0.01. Naive softmax([1000, 999]) = **nan**. Uniform guess over GPT-2's vocabulary: ln(50,257) = 10.8. Same 74% accuracy, loss 0.46 → **2.49** when logits are scaled ×16 |
| 04 | Backpropagation | The chain rule on one neuron | Backprop = finite differences = PyTorch autograd; a training loop learns XOR | Vanishing and exploding gradients | Relative error **1e-11**; autograd difference 4e-17. Sigmoid, 10 layers: layer 1's gradient ~**2 million times** smaller. Trained: sigmoid 50%, tanh 88%, ReLU+He **99.8%**, oversized weights NaN |
| 05 | Optimizers, batches, parameters | Parameters per layer; GPT-2 small ~124M by the same formula | Learning-rate sweep; batch size | SGD vs Momentum vs Adam in a narrow valley and on spirals; warm-up + cosine | Best SGD learning rate: **1.0**; 5.0 explodes. Full batch: 40 updates, **59%**. Adam works at 0.01 and 0.1, collapses at 0.3 |
| 06 | Overfitting, dropout, FFNs | A 17,000-weight network memorises 200 noisy points | Early stopping, dropout, L2, more data | A digit classifier; dropout left ON; dropout is no magic | Validation loss bottoms at epoch 201, then rises 0.51 → 0.79. Remedies fix confidence (test loss 0.77 → ~0.5), not accuracy; **more data** fixes both (84%). Dropout ON: the same digit gets 6, 8, 0 and 1 |

### Part B: Architectures (PyTorch)

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 07 | CNNs | A 3x3 edge filter by hand; equivariance | Digits anywhere: MLP vs CNN | Train centred, test shifted; augmentation; learned filters | MLP **65%** with 100k weights; CNN **98%** with 19k. Trained centred, tested shifted: MLP 12%, CNN+flatten 17%, CNN+global pooling **85%**. Learned filters match our hand-made edge filters (correlation 0.81-0.92) |
| 08 | RNNs & LSTMs | An RNN unrolled by hand: "dog bites man" ≠ "man bites dog" | "A before B": counts 49%, RNN and LSTM 100% | Remember the first token over 10-100 steps | At 100 steps the RNN guesses (27%). **Plot twist:** the LSTM with PyTorch defaults fails already at 30 steps (25%); a forget-gate bias of 4 → **100%** at 100 steps |
| 09 | Attention & transformers | "it" finds "cat"; the causal mask; attention ignores order | A tiny transformer remembers across 100 steps; order with/without positions | Encoder vs decoder; the future can't change the past; anatomy of a block | 100 steps: **100%**. No position embeddings: **49%** (a bag of words). Answer in the last token, read from position 0: encoder 100%, decoder **26%**. The MLP is 2/3 of every block |

### Part C: From ML to LLMs (the invented world)

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 10 | Tokens & embeddings | Characters vs words vs BPE; the first merges | Embeddings as lookup; what TinyGPT's embeddings learned | Spaces, casing, digits, letter counting, cost per word, vocabulary size | BPE: ~3x shorter than characters, nothing unknown. Export neighbours are exports **93%** of the time (chance 50%). Invented names cost **3.4** tokens, unseen words 7.5 |
| 11 | Pretraining & self-attention | Next-token targets; the loss curve; samples during training | Recall by exposure; multiple choice by likelihood; what attention reads | Three model sizes; the phrasing of the question | Perplexity 195 → **1.3**. Facts seen once: **70%**; 16 times: 100%. Tiny / small / base: **34% / 77% / 86%**. Rare facts: 100% if the probe's phrasing was in training, **36%** if not; an unseen phrasing recalls 23% |
| 12 | Instruction tuning, fine-tuning, ICL | Q/A tuning on 28 countries; tested on 20 others | Context skills: the map lookup, facts in the prompt | New facts in, old facts out | Held-out questions: base **0%**, tuned **55%** (as sentences: 87%). Map lookup: pretrained 12%, +800 steps **100%**, same 800 steps from scratch **30%**. A language given in the prompt: base 10%, tuned 100%. Fine-tuning new facts: old recall 82% → **32%**; with replay 58% |
| 13 | Temperature & hallucinations | The next-token distribution at T = 0.5 / 1 / 2 | 20 samples per question; top-k, top-p; uncertainty by exposure | Hallucinations; confidence as a detector; refusal tuning | Greedy 90% → T = 1.5 75%, with **22%** invented names. Unknown countries: answered **100%**, 67% with another country's real value. Confidence AUC **0.95**. Refusal tuning: refuses 27% of unseen unknowns, and 13% of answerable questions |

### Part D: Hands-on lab (`14_hands_on_lab_neural_sentiment_classifier.py`)

A sentiment tagger for product reviews, where negation ("not great"), contrast ("great screen but awful support") and shortcut endings ("five stars") make it hard.

| Part | Level | What you do |
|---|---|---|
| 1 | EASY | Data, split, baselines: majority 52%, bag-of-words logistic **76%**, word+bigram TF-IDF **95.3%** |
| 2 | EASY | A small NumPy MLP on word counts with early stopping: **85%**. It combines words, but the order is gone |
| 3 | MEDIUM | A tiny transformer trained from scratch in about a minute: **93.3%**, contrast reviews **95%** |
| 4 | MEDIUM | Accuracy by review type, and a counterfactual **shortcut test**: swapping "five stars" ↔ "one star" drops bag-of-words to **25%** |
| 5 | HARD | Scorecard with gates (accuracy ≥ 90%, negations ≥ 85%) and weights; memo in `outputs/14_sentiment_classifier_memo.md` |

Plot twist: the brief's weights on speed and size pick the **bigram model**, not the transformer (12x faster, 370x smaller). Shift the weight to contrast reviews and the choice **flips**. The scorecard makes the trade-off explicit.

### Part E: Assignment

- **`15_assignment_worked_example_text_to_response.py`**: "What is the capital of Piri?" traced through the instruction-tuned TinyGPT.
  - **The trace:** 34 characters → 12 chunks → 15 tokens → a 15 × 128 matrix → 4 blocks (block 1 reads " Piri") → 400 logits → " E" at 0.77 → 6 passes → " Eomezor." (correct).
  - **The contrast:** an unknown country gets " Dorcor." **Plot twist:** its first token is *more* confident (0.82 vs 0.77). The whole-answer probability (0.29 vs 0.77) and sample agreement (38% vs 88%) give it away.
  - **Output:** `outputs/15_text_to_response_journey.md` with the diagram.
- **`ASSIGNMENT.md` + `assignment_starter.py`**: students rebuild the pipeline themselves (BPE encoding, embeddings, forward pass, temperature and top-p sampling, the loop), each step checked against the real model. They then trace a known and an unknown language or export question, fill a diagram and write the report.
  - **Built-in surprise:** some hallucinations come back near-certain and identical in every sample, so the warning signals from Lab 15 fail there.
  - **Grading:** an automated rubric awards 40 points; reviewer judgement awards 60.

---

## Recording tips
- Run `python run_all.py` once before recording, so the TinyGPTs are trained and cached and the charts exist.
- In cell mode, pause after each `>>` note. Those lines are written as talking points, with the numbers filled in from the run.
- The world has a **hidden truth**. Lean on it: "with a real LLM you could never check this; here we can" is the thread from Lab 10 to Lab 15.
- Several labs show **honest negative or surprising results**:
  - Gradient boosting beats the network on tabular data (Lab 01).
  - Regularisers fix confidence, not accuracy (Lab 06).
  - The default-initialised LSTM forgets (Lab 08).
  - The transformer is *slower* than the LSTM on a CPU (Lab 09).
  - Instruction tuning unlocks only part of what the model knows (Lab 12).
  - Refusal tuning over-refuses (Lab 13).
  - The bigram model wins the scorecard (Lab 14).
  - The confident first token of a hallucination (Lab 15).

  That's how real projects go.
- Moments to slow down for:
  - Lab 04's gradient table.
  - Lab 09's encoder-vs-decoder table.
  - Lab 11's samples during training.
  - Lab 12's "same 800 steps from scratch" row.
  - Lab 15's ten stages on one screen.
