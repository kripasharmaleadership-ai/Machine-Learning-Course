# Module 6 Assignment: An IT Help-Centre Search Prototype, and its Path to RAG

> *"Most RAG failures are retrieval failures."* Before an LLM can answer from your documents, something has to find the right ones, know when there are none, and prove it on questions it has never seen.

## The scenario

The IT service desk receives hundreds of tickets a week that the help centre already answers ("my VPN won't connect", "I lost my company phone"). They want:

1. **Now:** a search box that suggests the **three most relevant help articles** for a question typed in plain language.
2. **Next quarter:** an assistant that **answers** the question with a citation (RAG) and says *"I don't know"* when the help centre doesn't cover it, for example HR or personal questions.

Your job is to build and evaluate the prototype, choose the retriever, and write the plan for the RAG version.

Reference solution for a *different* knowledge base: [11_assignment_worked_example_hr_retrieval_to_rag.py](11_assignment_worked_example_hr_retrieval_to_rag.py), an HR handbook. It shows the dev/test split, five candidates, a tuned hybrid, error analysis and a RAG design memo. Study it first, but don't copy its conclusions: on the IT help centre, the ranking of approaches is **different**.

## The data

Created by the first run of `assignment_starter.py`:

| object | contents |
|---|---|
| `articles` (`data/it_help_centre_articles.csv`) | `article_id`, `title`, `text`, `answer_fact`. Real articles are `IT-00`…`IT-11`; `IT-X..` are **look-alike pages** that share vocabulary but answer other questions |
| `dev` (`data/it_help_centre_dev_questions.csv`) | 48 user questions with `gold_article` and `answer`, for **all** tuning and choices |
| sealed test set | 48 more questions, scored once by `open_test_set()` |
| `OOS_DEV` / sealed out-of-scope test | questions the help centre must refuse |
| `corpus` | 6,000 unlabelled IT sentences for training embeddings |

## The requirements (fixed by the service desk)

| requirement | type | metric | threshold / weight |
|---|---|---|---|
| Right article in the top 3 | HARD gate | recall@3 | ≥ 0.90 |
| Latency | HARD gate | ms per query | ≤ 20 |
| Right article first | soft | recall@1 | 40% |
| Ranking quality | soft | MRR | 20% |
| Refuses out-of-scope | soft | share of out-of-scope questions below the threshold | 30% |
| Index size | soft | KB of stored vectors | 10% |

The refusal threshold of each retriever is set automatically on dev, so that it answers about 92% of in-scope dev questions. You may **add** requirements (and justify them). You may not loosen these.

## What to do

Work in **Part B** of [assignment_starter.py](assignment_starter.py). Part A (data, split, harness, a baseline) and Part C (scorecard, rubric, report) are written for you.

1. **At least three retrievers**, including at least one keyword-based (TF-IDF, character n-grams, BM25…) and one embedding-based (PPMI embeddings, LSA, or a pre-trained model if you have one). Register each one with `register(name, score_fn, family, index_bytes, how)`.
   - Fit vectorisers and embeddings on `corpus` and/or the articles, **never** on questions.
2. **Tune a hybrid on dev** with `tune_hybrid(score_a, score_b)` and register it.
3. **Error analysis:** run `error_analysis(name)` on the dev recommendation (Part C prints it). Which articles get confused, and why? Are the look-alike pages to blame?
4. **The mini RAG pipeline:** at least three `rag_answer(name, question)` calls, including at least one out-of-scope question that is refused. Explore `threshold=`: what does a stricter threshold cost on real questions?
5. **Open the test set once** (`open_test_set()`), after you've made your choice on dev.
6. **Write the report** in `REPORT`:

   | section | minimum | content |
   |---|---|---|
   | Retrieval results | 60 words | dev vs test, which approach wins and why, how noisy 48 test questions are |
   | Error analysis | 40 words | the confusions and the fixes you would try |
   | RAG design | 120 words | ingestion, **chunking**, embeddings, vector store, **metadata** filters, prompt with citations, refusal, **evaluation** and monitoring |
   | Risks | 40 words | stale articles, access control, confident wrong answers, the refusal trade-off |
   | Recommendation | 120 words | name the retriever, cite at least 3 numbers, state the trade-offs |

## Deliverables

| # | File | Produced by |
|---|---|---|
| 1 | `outputs/assignment_it_search_report.md` | starter (auto, when all checks pass) |
| 2 | `outputs/assignment_it_search_report.json` | starter (auto, when all checks pass) |
| 3 | Your completed `assignment_starter.py` | you |
| 4 | **Stretch (optional):** a recommender or "similar articles" panel (see below) | you |

## Grading rubric (100 points)

**Automated checks (40 pts).** These are printed by `assignment_starter.py`, Part C:

| check | pts |
|---|---|
| ≥ 3 retrievers besides the baseline | 5 |
| Keyword AND embedding-based retrievers | 5 |
| A hybrid tuned on dev and registered | 5 |
| Requirements not loosened | 4 |
| Test set opened exactly once | 4 |
| Error analysis of the recommended retriever | 5 |
| ≥ 3 RAG answers including ≥ 1 refusal | 6 |
| Report: word minimums, RAG design covers chunking / metadata / evaluation, recommendation names the winner with ≥ 3 numbers | 6 |

**Reviewer judgement (60 pts):**

| criterion | pts | what "excellent" looks like |
|---|---|---|
| Retrieval analysis | 14 | Dev and test reported *and interpreted*: why the winner wins on this vocabulary, whether the hybrid's weight makes sense, what a 48-question test set can and cannot tell you |
| Error analysis | 10 | Confusions explained by content (look-alike pages, shared words, missing synonyms) with concrete fixes (titles, synonyms, re-ranker, merging pages) |
| RAG design | 16 | A concrete, staged plan: chunk size tuned under a context budget, metadata filters (version, audience, access), prompt that forces citations and allows "I don't know", an evaluation set that grows from real tickets |
| Decision quality | 10 | The recommendation follows from gates and weights; no re-choosing after the test set; the refusal threshold trade-off discussed with numbers |
| Risks and limitations | 10 | Stale articles, access rights, personal data in questions, the limits of synthetic data and of a small embedding corpus |

## Hints

- **Character n-grams** (`TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))`) match "wi-fi" with "wifi" and "laptop" with "laptops". Short technical vocabulary often favours them (Lab 09).
- **The embedding corpus matters** (Lab 08, 3.3): embeddings only know the words and contexts they were trained on. Try `dim`, `window` and the IDF weighting in `emb.embed(texts, idf)`.
- **A hybrid can collapse:** if tuning puts all the weight on one side, say so; it's a result, not a failure (Lab 11).
- **Thresholds trade errors:** a stricter "I don't know" refuses more HR questions *and* more real IT questions (Lab 10, 3.2).
- **Don't chunk by reflex:** these articles are short (Lab 11). Discuss when you *would* chunk in the RAG design.

## Stretch goals

- **"Similar articles" panel:** for each article, the 3 most similar others (content-based recommendation, Lab 07). Are the look-alike pages a problem here too?
- **Recommender:** with `make_ratings()` (Lab 07), compare popularity, content-based and matrix factorisation, and discuss exposure bias.
- **Agent memory:** use `AgentMemory` (Lab 10) so the assistant remembers the user's device and office between questions. When should a memory be forgotten?
- **A real embedding model:** if `sentence-transformers` is installed, register it as a fourth candidate with the same harness. Does it beat the domain-trained PPMI embeddings on this vocabulary?
