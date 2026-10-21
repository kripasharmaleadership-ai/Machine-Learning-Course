# HR handbook page finder: which retrieval approach should the prototype (and later the RAG assistant) use?

Prototype evaluation on held-out questions; all tuning on a separate dev set.

## Requirements

| requirement | type | metric | threshold / weight | why |
|---|---|---|---|---|
| right page in top 3 | HARD gate | `recall@3` | >= 0.9 | employees see three suggestions; the right page must be among them |
| latency | HARD gate | `ms_per_query` | <= 20 | it sits behind a chat box |
| right page first | soft | `recall@1` | weight 40% (higher is better) | the RAG answer cites the top page |
| ranking quality | soft | `MRR` | weight 20% (higher is better) | the right page near the top |
| refuses out-of-scope | soft | `oos_refused` | weight 30% (higher is better) | no confident answers from unrelated pages |
| index size | soft | `index_kb` | weight 10% (lower is better) | cost of storing vectors for 10,000+ pages later |

## Candidates

| candidate | right page in top 3 | latency | right page first | ranking quality | refuses out-of-scope | index size | eligible | failed gates | weighted score |
|---|---|---|---|---|---|---|---|---|---|
| embeddings | 0.938 | 0.031 | 0.729 | 0.835 | 0.833 | 15.000 | True | - | 0.908 |
| hybrid | 0.938 | 0.066 | 0.729 | 0.835 | 0.833 | 19.625 | True | - | 0.903 |
| word TF-IDF | 0.854 | 0.037 | 0.646 | 0.768 | 0.833 | 4.625 | False | right page in top 3 | - |
| char TF-IDF | 0.875 | 0.088 | 0.750 | 0.816 | 0.833 | 90.773 | False | right page in top 3 | - |
| hybrid on chunks | 0.792 | 0.068 | 0.729 | 0.802 | 0.833 | 44.070 | False | right page in top 3 | - |

## Held-out evaluation

- 48 test questions (48 used for tuning), 6 out-of-scope test questions.
- The winner's hybrid weight was chosen on dev: alpha = 0.5 (weight on embeddings).
- Most frequent confusions on test:

  - 3x Training budget  ->  Volunteering days
  - 3x Expense claims  ->  Business travel booking
  - 2x Bereavement leave  ->  Annual leave allowance
  - 1x Pay day and payslips  ->  Annual leave allowance
  - 1x Booking leave  ->  Business travel booking

## From this prototype to a RAG application

| Stage | Prototype (this lab) | Production RAG |
|---|---|---|
| Ingestion | 32 synthetic pages | Connectors to the HR wiki / SharePoint; re-index on every page change |
| Chunking | whole pages (short); ~20-word chunks tested and rejected | Real pages are long: tune chunk size on dev questions under the real context budget (Lab 10) |
| Embeddings | PPMI + SVD trained on HR text | A pre-trained sentence-embedding model (keep the same dev and test sets to compare) |
| Vector store | numpy arrays, exact search | A vector database (pgvector, FAISS, a managed service) with IVF/HNSW indexes |
| Metadata | page id | version, effective date, country, audience, access rights: used as FILTERS |
| Retrieval | hybrid score, top 3 | Hybrid (BM25 + vectors) top 20, then a re-ranker, then top 3-5 |
| Generation | none | An LLM prompted to answer ONLY from the sources, cite page ids, say "I don't know" |
| Refusal | similarity threshold tuned on dev | Threshold + the LLM's own "not in the sources" + escalation to a human |
| Evaluation | recall@k, MRR, refusal rate | The same retrieval metrics plus answer correctness, citation accuracy, faithfulness |
| Monitoring | none | Log questions, citations and thumbs up/down; review refusals weekly; re-test on every model change |

## Risks to manage

1. **Stale or conflicting policies**: an old version is as similar as the current one; filter by version and date (Lab 10).
2. **Access control**: retrieval must never return pages the employee may not see (e.g., manager-only pay bands).
3. **Confident wrong answers**: keep citations visible and the refusal path tested with out-of-scope questions.
4. **Evaluation drift**: the 96 labelled questions are a start; add real (anonymised) questions from the inbox every month.
5. **Personal data**: questions can contain health or family details; set retention and redaction rules for the logs.


## Notes per candidate

- **hybrid:** alpha=1.0
- **hybrid on chunks:** alpha=1.0

## Recommendation

**embeddings**

'embeddings' passes both gates and has the best weighted balance of first-page accuracy, ranking, refusals and index size (word TF-IDF, char TF-IDF, hybrid on chunks failed a gate). Start the RAG pilot with it, and swap in a pre-trained embedding model only if it beats this one on the same test set.
