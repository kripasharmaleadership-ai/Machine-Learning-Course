# Module 6: Unsupervised Learning, Recommendations, and Embeddings

[🏠 Course home](../README.md) · [📋 Syllabus](../SYLLABUS.md) · [← Module 5](../Module5_Trees_Ensembles_Tuning_XAI) · **Module 6 of 8** · [Module 7 →](../Module7_Neural_Networks_Road_to_GenAI)

Eleven runnable labs and an assignment. Each lab builds from **EASY → MEDIUM → HARD**, and each HARD level ends with a "plot twist". The scenarios are a retailer's customers, expense claims, a movie service, a product catalogue, an HR handbook and an IT help centre.

The theme of the module: **find structure without labels, then turn it into vectors you can search.**
- **Structure without labels:** clustering (Labs 01, 03, 04), judged without an answer key, and always a reflection of *your* choices of features, scaling and distance (Lab 02).
- **Compression and the unusual:** PCA and its traps (Lab 05); anomaly detection under an alert budget (Lab 06).
- **Recommendations:** popularity, content-based, collaborative filtering and matrix factorisation, and the exposure bias that fools offline metrics (Lab 07).
- **Embeddings → GenAI:** text becomes vectors (Lab 08), vectors become semantic search (Lab 09), and semantic search becomes **RAG** and **agent memory** (Lab 10).

Every lab is a plain Python file with `# %%` cell markers:
- **Script mode:** `python 03_kmeans_and_customer_segmentation.py` prints a narrated walkthrough and saves charts to `outputs/`.
- **Cell mode (best for recording):** open the file in VS Code and click **Run Cell** above each `# %%`. Charts appear inline.

## Setup

```bash
pip install -r requirements.txt
python run_all.py          # smoke-test all 11 labs + the assignment starter (~2 minutes)
```

**Everything runs offline, with no API key and no GPU.** Lab 05 uses scikit-learn's bundled handwritten digits; every other dataset is synthetic, with a **hidden ground truth** (the true customer segment, the true anomaly type, the true taste of every user, the true page for every question). That truth is what lets the labs *check* unsupervised results, which you can never do in production.

**No embedding model or vector database needed.** The toolkit learns word embeddings from scratch (PPMI + SVD, the maths behind word2vec) and implements exact and IVF vector indexes, chunking, an extractive RAG step and agent memory. If `sentence-transformers` is installed, Lab 08 adds a pre-trained model to its comparison automatically. All randomness is seeded (tested with scikit-learn 1.9, pandas 2.2, numpy 2.2).

## Folder layout

| File | Purpose |
|---|---|
| `ml_lab_utils.py` | Console/plot helpers and synthetic data with **known truth**: **customers** (5 hidden segments, optional resellers), **expense claims** (4 kinds of planted anomaly), **movie ratings** (a TRUE taste matrix behind the observed ratings), a **product catalogue**, **intranet text corpora** (HR, IT) and **knowledge bases** with look-alike pages and paraphrased questions |
| `unsup_toolkit.py` | A **reusable library** built up across the module: distances; `KMeansScratch` (= sklearn); `pca_svd`; `isolation_scores` (Isolation Forest from scratch); leave-n-out splits and precision / recall / NDCG@k; `ALSMatrixFactorization`; `PPMIEmbeddings`; `FlatIndex` and `IVFIndex`; `retrieval_report` (recall@k, MRR); `chunk_text`; `extractive_answer`; `AgentMemory`; `ModelScorecard` (requirements → memo) |
| `01` … `08` | Core topics |
| `09_hands_on_lab_semantic_product_search.py` | The hands-on lab: a semantic recommendation and search system, end to end |
| `10_genai_connection_...py` | Semantic search, vector databases, chunking, RAG and agent memory |
| `11_assignment_worked_example_...py` | Reference solution: an HR document-similarity prototype and its RAG design memo |
| `ASSIGNMENT.md` + `assignment_starter.py` | The student assignment (IT help centre) with an automated rubric |
| `run_all.py` | Runs every lab and reports pass/fail |
| `outputs/` | Charts, the saved product index, the HR retrieval memo (MD + JSON) |

### Syllabus coverage

| Syllabus topic | Where |
|---|---|
| Learning without labelled data | 01 |
| Distance and similarity; cosine similarity | 02 (and 08-10) |
| Clustering and customer segmentation; K-means | 03 (and 01) |
| Hierarchical clustering; DBSCAN overview | 04 |
| Dimensionality reduction; PCA | 05 (and 09) |
| Anomaly detection; Isolation Forest overview | 06 |
| Recommendation fundamentals; content-based recommendation; collaborative filtering overview | 07 |
| Vector representations and embeddings | 08 |
| Hands-on lab: prepare data, create feature vectors, cosine similarity, retrieve similar items, visualise clusters with PCA | 09 |
| GenAI connection: text embeddings, semantic search, vector databases, document retrieval, RAG, agent memory | 10 |
| Assignment: a recommendation or document-similarity prototype, and how it could become a RAG application | 11 (solved) + ASSIGNMENT |

---

## Episode plan with the key on-screen moments

### Part A: Structure without labels

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 01 | Learning without labels | 3,000 customers, no labels: k-means finds groups | Judging clusters without an answer key: silhouette, stability across resamples | Results reflect YOUR choices: features, scaling, k | Clusters match the hidden segments (ARI **0.942**). Silhouette prefers **k = 4** while the truth is 5 (ARI 0.94 either way); stability rules out k = 8. Pure noise still "clusters": silhouette 0.075 vs 0.331 for real structure |
| 02 | Distance and similarity | Euclidean, Manhattan, cosine on the same customers; Jaccard on baskets | Same data, different nearest neighbour | The curse of dimensionality; correlated features count twice (Mahalanobis) | A short and a long refund document: Euclidean **11.0** vs 3.16 to a password document, but cosine **0.98 vs 0.00**. At 5,000 dimensions nearest/farthest distance ratio **0.954**. Duplicating the groceries column changes 28% of Euclidean neighbours, 0% for Mahalanobis |
| 03 | K-means & customer segmentation | One Lloyd iteration at a time; scratch = sklearn | Choosing k; k-means++ vs random starts; outliers | Non-round clusters (GMM); segments as a business asset | Scratch inertia 9,770.77 vs sklearn 9,770.74. k-means++ reaches the best solution in **77%** of runs vs 40%. Resellers: ARI 0.944 → 0.901. Stretched blobs: k-means 0.607 vs **GMM 1.0**; uneven sizes 0.212 vs 0.996 |
| 04 | Hierarchical & DBSCAN | Agglomerative clustering on 10 customers; the dendrogram | Which linkage / method sees which shape | DBSCAN as an outlier finder; HDBSCAN; the memory cost of hierarchy | Single linkage and DBSCAN solve the two moons (**1.0**). Resellers: 13 of 15 flagged as noise. Varying densities: HDBSCAN finds 3 clusters with 4% noise; the best DBSCAN 2 clusters and **29%** noise. Hierarchical on 8,000 rows: **512 MB** distance matrix |

### Part B: Compression and the unusual

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 05 | PCA | PCA on 9 customer features via SVD | Compressing 64-pixel digits; KNN on PCs | Three ways PCA misleads: scaling, non-linear shapes (t-SNE + trustworthiness), labels in low-variance directions | **21** components keep 90% of digit variance; KNN 0.959 on 20 PCs vs 0.963 on all 64. Unscaled: PC1 is just groceries (loading **0.99**). Swiss roll: t-SNE trustworthiness 1.00 vs PCA 0.13. PCA → logistic: **0.512** vs 0.903 on raw features |
| 06 | Anomaly detection | z-scores and the IQR rule on expense claims | Isolation Forest from scratch; LOF | Domain features; alert budgets; the path to supervision | z > 3 precision 0.99, but split claims caught **0%**. Isolation Forest AUC 0.90 yet only 22% of inflated claims; + 5 domain features: AUC **0.983**, precision 47% → 70%. A classifier trained on month-2 investigator labels: top-400 precision **0.905** vs 0.708 |

### Part C: Recommendations

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 07 | Recommendation fundamentals | The ratings matrix; leave-n-out evaluation; popularity | Content-based, item-item CF, matrix factorisation (ALS) | Cold start; do learned user embeddings recover the hidden tastes?; popularity bias; a hybrid | **Exposure bias:** "most rated" scores NDCG **0.307** offline, but its TRUE-taste precision is 0.066 (random 0.051); ranking by the TRUE ratings scores only 0.071. Content-based true-taste precision **0.354**; MF 0.271; hybrid **0.392** with coverage 20% → 30%. The offline metric rewards what users were shown, not what they like |

### Part D: Embeddings

| # | Lab | EASY | MEDIUM | HARD | Key on-screen moment |
|---|---|---|---|---|---|
| 08 | Vector representations & embeddings | One-hot, bag-of-words, TF-IDF | Embeddings from 6,000 unlabelled intranet sentences (PPMI + SVD) | Embeddings beyond words; where they break | Question → page recall@1: TF-IDF 0.677, LSA 0.667, embeddings **0.833**. Movie neighbours share a genre 86% vs 25% by chance. "parental leave is paid in full" vs "... is not paid": cosine **0.95**. HR embeddings on IT questions: **0.271** (TF-IDF 0.667, IT embeddings 0.802). 5 dimensions: 0.333; 150: 0.865 |

### Part E: Hands-on lab (`09_hands_on_lab_semantic_product_search.py`)

A semantic recommendation and search system for a product catalogue, end to end.

| Part | Level | What you do |
|---|---|---|
| 1 | EASY | Prepare the product data: titles, descriptions, categories, brands |
| 2 | EASY | Feature vectors: word TF-IDF, character n-grams, embeddings |
| 3 | MEDIUM | Cosine similarity and **"more like this"**, measured with R-precision: TF-IDF 0.642, characters 0.574, embeddings **0.997** |
| 4 | MEDIUM | **Semantic search** for 26 shopper queries: TF-IDF 0.671, characters 0.672, embeddings **0.998**; a hybrid tuned on half the queries |
| 5 | HARD | **Visualise clusters with PCA**, find near-duplicates, add metadata filters, save the index (`outputs/product_index.npz`) |

Plot twist: keyword search fails on the queries shoppers actually type. "lightweight ultrabook for travel" gets R-precision **0.42** with TF-IDF, because the catalogue says "laptop".

### Part F: Connection to GenAI (`10_genai_connection_semantic_search_rag_memory.py`)

| Level | Concept | Key on-screen moment |
|---|---|---|
| EASY | **Text embeddings & semantic search** on the HR handbook | "holiday" → Annual leave, "GP note" → Sick leave, where keywords find the Holiday party. Recall@1 **83%**, recall@3 94%. Honest miss: for "supervisor wants extra hours" Overtime is only second |
| MEDIUM | **Vector databases**: exact vs IVF at 30,000 vectors. **Chunking** under a context budget | IVF, 1 of 64 cells: **521** comparisons instead of 30,000, recall@10 0.65; 4 cells: 2,242 comparisons, recall **1.00**. Whole pages retrieve 95% of answer facts but cost 424 words; in an 80-word budget, ~50-word chunks win (84% vs 81%) |
| HARD | **RAG** end to end: a prompt with citations, an "I don't know" threshold, stale documents, **agent memory** | The answer fact reaches the prompt for **70%** of questions; the one-sentence extractor states it 33% of the time (partly an evaluation artefact). 67% of out-of-scope questions refused; "can I bring my dog" leaks through. The 2019 policy answers "20 days" until a **metadata filter** removes it. Memory: "which office?" → Lisbon, with recency → **Dublin** |

### Part G: Assignment

- **`11_assignment_worked_example_hr_retrieval_to_rag.py`**: an HR page finder, with a dev/test split of questions, five candidates and a scorecard (recall@3 ≥ 0.90 gate).
  - **Results:** keywords fail the gate (word TF-IDF 0.854, char 0.875); embeddings pass (**0.938**).
  - **Plot twists:** the tuned hybrid puts **all** its weight on embeddings (alpha = 1.0), and chunking **hurts** on short pages (0.792). Error analysis: 54% of the mistakes land on look-alike pages.
  - **Output:** a memo (`outputs/11_hr_retrieval_memo.md`) with the plan from prototype to production RAG and its risks.
- **`ASSIGNMENT.md` + `assignment_starter.py`**: the same job for the **IT help centre**, with a harness, a sealed test set and out-of-scope questions.
  - **Built-in surprise:** the ranking differs from HR. On short technical vocabulary, **character n-grams** are strong.
  - **The refusal trade-off:** a strict threshold refuses HR questions *and* "my laptop will not start".
  - **Grading:** an automated rubric awards 40 points; reviewer judgement awards 60.

---

## Recording tips
- Run `python run_all.py` once before recording, so the charts and the saved index exist.
- In cell mode, pause after each `>>` note. Those lines are written as talking points, with the numbers filled in from the run.
- The datasets have a **hidden truth**. Lean on it: "in production you could never check this; here we can" is the thread from Lab 01 to Lab 11.
- Several labs show **honest negative results**:
  - Silhouette picking the wrong k (Lab 01).
  - Isolation Forest missing most inflated claims (Lab 06).
  - The offline metric rewarding exposure, not taste (Lab 07).
  - Negation invisible to embeddings (Lab 08).
  - The hybrid that collapses, and chunking that hurts (Lab 11).

  That's how real projects go.
- Lab 02's refund documents, Lab 07's "TRUE taste" table, Lab 10's IVF table and Lab 10's stale-policy answer are the moments to slow down for.
