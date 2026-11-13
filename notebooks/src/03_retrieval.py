# %% [markdown]
# # Day 3: Retrieval, graph and visual documents
#
# **Where we are in the story.** Priya has Rohan's numbers now. Her next three questions are:
# "What does the policy say about his debt level?", "Does Rohan own other companies that already owe
# Good Bank money?", and "What does that chart on page 2 of his management pack actually show?"
# Those are three different kinds of retrieval, and today we build all three.
#
# **The one idea of today: the right retriever for the question.** Semantic search finds meaning,
# keyword search finds exact terms like "3.1" or "DSCR", a graph finds relationships, and visual
# retrieval finds things that were never text. A production system uses several, and measures each.
#
# Topics: R1 to R16 (RAG), G1 to G11 (graph), M3 to M7, M9, M12 (visual), OB6 to OB8 (RAG evaluation).
#
# Runs on the free Colab tier. The ColPali section wants a GPU runtime; without one, load the
# pre-computed embeddings your trainer shared.

# %%
!git clone https://github.com/chahatfde-arch/Day1_FDE_session.git
%cd Day1_FDE_session
!pip install -q -e ".[dev,llm,retrieval,realdata]"
!python -m creditlens.generators --skip-pdfs
!python -m creditlens.pipeline --load
!python -m creditlens.data_real --only sop charts

# %%
import os
try:
    from google.colab import userdata
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
              "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "PINECONE_API_KEY"):
        try:
            os.environ[k] = userdata.get(k)
        except Exception:
            pass
except ImportError:
    pass
os.environ.setdefault("LLM_OFFLINE", "0" if os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY") else "1")

# %% [markdown]
# ## 1. Recap quiz (scored)
# 1. Why did we label the golden set before building the extractor?
# 2. What does the `doc_type` field do in `ExtractionEnvelope`?
# 3. OCR returned `1,23,4S6`. Is that an OCR problem or an LLM problem, and how did you tell?

# %% [markdown]
# ## 2. Which retriever? (vote before you build)
# For each analyst question, vote: vector, keyword, graph or visual.
#
# 1. "What is the minimum DSCR?"
# 2. "Show me rule 4.2."
# 3. "Which other companies share a director with Rohan's company?"
# 4. "Is cash-heavy business allowed if the owner explains it?"
# 5. "What does the revenue chart on page 2 show for FY2025?"
# 6. "Does the SBA allow loans to businesses that have been operating under two years?"
#
# Keep your votes. We check them against the evaluation at the end.

# %% [markdown]
# ## 3. Credit policy RAG (R1 to R16)
#
# ### Two corpora
# - **Good Bank's manual**: 40 pages, numbered rules. The answer key was generated from the same
#   thresholds, so it must be the source for policy checks.
# - **SBA SOP 50 10**: the real rulebook of the US Small Business Administration, about 146,000 words.
#   A real document with cross-references, legal phrasing and deep heading levels. Chunking that works
#   on the tidy manual breaks here, which is the point.
#
# ### Chunking bake-off
# A chunk is the unit we embed and retrieve. Too small and the answer is cut in half; too large and
# the embedding is a blur of several topics. Three strategies:
#
# - `section`: split at headings, keep the heading path as metadata. Best for structured documents.
# - `fixed`: every N characters with overlap. Works anywhere, ignores structure.
# - `sentence`: group sentences up to N characters. A middle ground.

# %%
from pathlib import Path
from creditlens.retrieval.chunking import chunk_markdown, compare_strategies

manual = Path("data/raw/policy/credit_policy_manual.md").read_text(encoding="utf-8")
sop = Path("data/real/sba_sop_50_10.md").read_text(encoding="utf-8")
print("Good Bank manual:", compare_strategies(manual))
print("SBA SOP        :", compare_strategies(sop))

# %%
chunks = chunk_markdown(manual, source="good_bank_policy", strategy="section")
for c in chunks[:4]:
    print(c.chunk_id, "|", c.section, "|", c.text[:90].replace("\n", " "), "...")

# %% [markdown]
# ### Embeddings, vector search, keyword search, hybrid, reranking
# - **Embedding**: a small open model (BGE-small, runs on CPU) turns each chunk into a vector. Similar
#   meaning, nearby vectors. The first run downloads the model once and caches the vectors on disk.
# - **Vector search** (Qdrant, in memory here; Docker or Qdrant Cloud free tier later): nearest vectors
#   to the question's vector.
# - **BM25**: classic keyword scoring. Unbeatable for "rule 3.1" or "DSCR".
# - **Hybrid with reciprocal rank fusion**: merge both ranked lists by rank, not by score, so the two
#   scales do not have to agree.
# - **Reranker**: a cross-encoder reads question and chunk together and re-scores the top candidates.
#   Slower, more accurate, so we only run it on the top 20.
# - **Metadata filter**: `source="good_bank_policy"` restricts the search before ranking, which is how
#   RBAC on retrieval works on Day 6 too.

# %%
from creditlens.retrieval.policy import search_policy

q = "What is the minimum debt service coverage ratio?"
for hit in search_policy(q, k=3, source="good_bank_policy"):
    print(f"[{hit.retriever}] {hit.score:.3f} {hit.section} :: {hit.text[:100]}...")

# %% [markdown]
# **Predict:** for the question "Show me rule 4.2", which retriever wins, vector or BM25?

# %%
from creditlens.retrieval.policy import get_index
idx = get_index()
for name, hits in idx.search_all("Show me rule 4.2", k=3).items():
    print(name, "->", [h.section for h in hits])

# %% [markdown]
# **Curveball card:** compliance says the SBA rulebook must also be searchable, but Good Bank answers
# must come first. Try the same question with and without the `source` filter.

# %%
q = "Can we lend to a business that has been operating for less than two years?"
print("Good Bank only:")
for h in search_policy(q, k=2, source="good_bank_policy"):
    print("  ", h.section, "::", h.text[:90].replace("\n", " "))
print("SBA SOP only:")
for h in search_policy(q, k=2, source="sba_sop"):
    print("  ", h.section[:60], "::", h.text[:90].replace("\n", " "))

# %% [markdown]
# ### Pinecone for comparison (optional)
# Same chunks, a hosted vector database. Needs `PINECONE_API_KEY`; check the free plan's limits on the
# Pinecone site before you create an index.

# %%
!python -m creditlens.retrieval.pinecone_compare || echo "skipped (no PINECONE_API_KEY or package)"

# %% [markdown]
# ## 4. Related-party graph (G1 to G11)
#
# Policy section 5 says a group is any set of companies linked by a common director or a common
# registered address, directly or through a chain. "Through a chain" is the hard part: A shares a
# director with B, B shares an address with C, so A and C are related even though nothing links them
# directly. In SQL that is a recursive join per hop. In a graph it is one variable-length path.
#
# The registry hides four such groups. Some links are directors, some are addresses: pods that only
# look at directors miss a group. Find them all with the graph, then check the answer key.
#
# `GraphStore` runs on Neo4j AuraDB Free when `NEO4J_URI` is set, and on NetworkX in memory otherwise.
# The Cypher text is identical either way and is stored in the result for the audit trail.

# %%
from creditlens.retrieval.graph import GraphStore
from pathlib import Path
store = GraphStore().load_from_registry(Path("data/raw/registry"))
groups = store.all_groups()
print(len(groups), "groups found:", [sorted(g) for g in groups])

# %%
import json
truth = json.load(open("data/answer_key/related_party_groups.json"))  # instructor only
print("answer key:", [g["company_ids"] for g in truth])

# %%
cid = truth[0]["company_ids"][0]
exp = store.group_exposure(cid)
print("group of", cid, "->", [(m.company_id, m.link, m.hops) for m in exp.members])
print("total exposure INR'000:", exp.total_exposure_inr_000, "limit:", exp.limit_inr_000, "breach:", exp.breaches_limit)
print("\nCypher used:\n", exp.cypher)

# %% [markdown]
# ### Profile and optimize
# On Neo4j, prefix a query with `PROFILE` to see how many database hits it costs, then create an index
# on `Company.company_id` and run it again. The helper below shows the NetworkX timing and prints the
# Cypher cheatsheet you can paste into the AuraDB browser.

# %%
print(store.profile())
from creditlens.retrieval.graph import cypher_cheatsheet
for name, q in cypher_cheatsheet().items():
    print(f"-- {name}
{q}
")

# %% [markdown]
# ## 5. Visual documents with ColPali (M3 to M7, M9, M12)
# Charts hold numbers that never appear as text. ColPali embeds each page *as an image* with a vision
# language model, so a question can find the right chart without OCR. It is heavy for a free GPU, so we
# use 30 real charts (ChartQA) and small batches. Without a GPU, load the pre-computed embeddings your
# trainer shared into `data/cache/colpali/`.
#
# Then a vision model answers the question from the retrieved chart. Each chart comes with a human
# answer, so we can score both steps.

# %%
try:
    from creditlens.retrieval import colpali
    colpali.precompute_embeddings()          # skipped instantly if the cache exists
    hits = colpali.search("Which category has the highest value in the bar chart?", k=3)
    print(hits)
    print("recall@3 over all 30 questions:", colpali.recall_at_k("data/real/charts/questions.json", k=3))
except Exception as e:
    print("ColPali not available here:", e)

# %%
qs = json.load(open("data/real/charts/questions.json"))
from IPython.display import Image, display
display(Image(filename=f"data/real/charts/{qs[0]['image']}", width=420))
print("Q:", qs[0]["question"], "| human answer:", qs[0]["answer"])
try:
    print("model answer:", colpali.answer_with_vision(qs[0]["question"], f"data/real/charts/{qs[0]['image']}"))
except Exception as e:
    print("vision answer skipped:", e)

# %% [markdown]
# ## 6. Retrieval evaluation (OB6 to OB8)
# Golden questions live in `ground_truth/retrieval_questions.json`: each has the section that must
# come back. Recall@5 says "was the right section in the top 5"; MRR rewards putting it first.
# Naive (vector only) versus hybrid versus reranked, on both corpora. RAGAS adds LLM-judged faithfulness
# and answer relevancy when a key is available; judge calls count against your quota, so it is optional.

# %%
!python -m evals.retrieval_eval --k 5

# %% [markdown]
# Go back to your "which retriever?" votes. Which questions did hybrid win, which did BM25 win alone,
# and which could no text retriever answer at all?
#
# ## Exit ticket
# One thing learned, one open question. Tomorrow: agents that use all of this to draft Priya's memo.
