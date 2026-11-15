# %% [markdown]
# # Session 2: Retrieval, graph and visual documents
#
# **Where we are in the story.** Priya has Rohan's numbers now. Her next three questions are:
# "What does the policy say about his debt level?", "Does Rohan own other companies that already owe
# Good Bank money?", and "What does that chart on page 2 of his management pack actually show?"
# Those are three different kinds of retrieval, and today we build all three. And two names from the
# bank statements are still unexplained: Ravi Containers in Rohan's file and Kamla Devi in Rahul's. By
# the end of section 4 the graph tells us who they are, and the policy index tells us which rule it touches.
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
import sys
if "google.colab" in sys.modules:   # Colab only. On a laptop this cell does nothing.
    !git clone https://github.com/chahatfde-arch/Day1_Day2_FDE_session.git
    %cd Day1_Day2_FDE_session
    !pip install -q -e ".[dev,llm,retrieval,realdata]"
    !python -m creditlens.generators --skip-pdfs
    !python -m creditlens.pipeline --load
    !python -m creditlens.data_real --only sop charts
    !python -m creditlens.scenarios
else:
    print("Not on Colab: skipping the clone. You are already in the repo folder.")

# %% [markdown]
# **Laptop users:** skip the clone cell above. The next cell finds the repository root, so the notebook
# works whether you opened it from the repo folder or from `notebooks/`.

# %%
import os
from pathlib import Path
while not Path("pyproject.toml").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir("..")
assert Path("pyproject.toml").exists(), "Open this notebook from inside the Day1_Day2_FDE_session folder."
print("working folder:", Path.cwd())

# %%
import os
os.environ["LLM_OFFLINE"] = "1"   # today we run offline: same numbers for everyone, no API key needed
print("offline mode:", os.environ["LLM_OFFLINE"] == "1")

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
# The registry hides four such groups. Some links are directors, some are addresses: people who only
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
    print(f"-- {name}\n{q}\n")

# %% [markdown]
# ### Rohan and Rahul, part two: who are Ravi Containers and Kamla Devi?
# Same `GraphStore`, same function, pointed at the story registry in `data/scenarios/` instead of the
# Good Bank one. **Predict first:** how many companies are in Rohan's group, and is rule 5.2 breached?
#
# Remember what Session 1 found. Rohan: 24 round trips to RAVI CONTAINERS, about 2.07 crore out and
# back. Rahul: 36 cash deposits, each followed the next day by the same amount to KAMLA DEVI, and a
# bank account that shows about 60 percent of his declared turnover.

# %%
story_graph = GraphStore(backend="networkx").load_from_registry(Path("data/scenarios"))
names = story_graph.companies
for who, cid in (("Rohan", "SC01"), ("Rahul", "SC10")):
    e = story_graph.group_exposure(cid)
    print(f"{who}: {len(e.members)} companies in the group")
    for m in e.members:
        print(f"   {m.company_id} {names[m.company_id]['legal_name']:26} link: {m.link:16} hops: {m.hops}")
    print(f"   group exposure {e.total_exposure_inr_000:,} vs limit {e.limit_inr_000:,} -> breach: {e.breaches_limit}\n")

# %% [markdown]
# **Picture it.** The links that make each group, drawn. Blue circles are companies, orange are
# directors, green are registered addresses. A solid line is "is director of", a dashed line is "is
# registered at". Only the directors and addresses shared by two or more companies are drawn, because
# those are the links that make a group.

# %%
# Picture it: draw the links that make each group
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

COLOURS = {"Company": "#2a78d6", "Director": "#eb6834", "Address": "#1baf7a"}


def draw_group(ax, gs, company_ids, title):
    comp = {f"Company:{c}" for c in company_ids}
    shared = {n for n in gs.graph if gs.graph.nodes[n]["kind"] != "Company"
              and len(comp & set(gs.graph[n])) >= 2}          # the links that make the group
    sub = gs.graph.subgraph(comp | shared)
    pos = nx.kamada_kawai_layout(sub)
    for kind, style in (("DIRECTOR_OF", "solid"), ("REGISTERED_AT", "dashed")):
        nx.draw_networkx_edges(sub, pos, ax=ax, style=style, width=2, edge_color="#777777",
                               edgelist=[(a, b) for a, b, k in sub.edges(data="kind") if k == kind])
    for kind, colour in COLOURS.items():
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colour, node_size=700, edgecolors="white", linewidths=2,
                               nodelist=[n for n, k in sub.nodes(data="kind") if k == kind])
    for n, d in sub.nodes(data=True):
        if d["kind"] == "Company":
            text = gs.companies[d["company_id"]]["legal_name"].replace(" Pvt Ltd", "")
        elif d["kind"] == "Director":
            text = d.get("full_name") or d["director_id"]
        else:
            text = d["value"][:28]
        x, y = pos[n]
        ax.text(x, y - 0.17, text, ha="center", va="top", fontsize=9, color="#222222")
    ax.set_title(title, loc="left", fontsize=11)
    ax.margins(0.25)
    ax.axis("off")


rohan = [m.company_id for m in story_graph.group_exposure("SC01").members]
rahul = [m.company_id for m in story_graph.group_exposure("SC10").members]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
draw_group(axes[0], story_graph, rohan, "Rohan: one group of four, rule 5.2 breached")
draw_group(axes[1], story_graph, rahul, "Rahul: a family of two, within the limit")
draw_group(axes[2], store, ["C019", "C022"], "Good Bank C019 and C022: linked only by an address")
handles = [Line2D([], [], marker="o", ls="", markersize=12, markerfacecolor=c, markeredgecolor="white",
                  label=k.lower()) for k, c in COLOURS.items()]
handles += [Line2D([], [], color="#777777", lw=2, label="is director of"),
            Line2D([], [], color="#777777", lw=2, ls="--", label="is registered at")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()

# %% [markdown]
# The graph needs one more fact before Priya can write anything: which rule does this break, in the
# bank's own words? That is a retrieval question, so we ask the policy index, filtered to Good Bank.

# %%
for q in ("What is the exposure limit for a group of related companies?",
          "Must the memo list every group member even when the limit is not breached?"):
    top = search_policy(q, k=1, source="good_bank_policy")[0]
    print(q, "\n  ->", top.section, "\n ", top.text[:420].replace("\n", " "), "\n")

# %% [markdown]
# Now put the two sessions together. `case_file` runs the Session 1 cleaning, the Session 2 graph and the
# policy thresholds, and returns flags with their evidence. Read the last line it prints.

# %%
from creditlens.scenarios import case_file
for app in ("A901", "A902"):
    cf = case_file(Path("data/scenarios"), app)
    print(app, "->", [f["code"] for f in cf["flags"]])
    for f in cf["flags"]:
        print("   ", f["code"], ":", f["title"])
    print("   ", cf["note"], "\n")

# %% [markdown]
# **How it turns out.** Rohan's company alone (50,000 thousand) passes rule 5.2. His group of four,
# linked by a shared director (Rohan Mehta) and a shared address in Nashik, owes or is asking for
# 1,55,000 thousand against a 1,20,000 limit. Only the graph shows that. Ravi Containers is not a
# customer: it is Rohan's own company, and 68 percent of his money in comes from his group.
#
# Rahul shares a shop address with Kamla Fabrics, whose director is Kamla Devi. Priya phones him: she is
# his mother, and part of the takings go to the family account. His group owes 55,000 thousand, well
# inside the limit, and his cash share (about 20 percent) is under the 30 percent in rule 4.2. Nothing
# breaks a rule. The system *explained* why he looked weak; it did not accuse him.
#
# **What happens next is decided by people, not by this notebook.** Policy section 1 forbids automatic
# approvals. Rohan's file goes to the underwriter and compliance with the evidence, and he is asked to
# explain. Rule 7.2 means the breach is raised as an exception ticket, and rule 5.3 means the memo lists
# every group member either way. Priya asks Rahul for the family firm's statements and counts only income
# she can verify. In Session 3 the agents draft that memo and then stop for the underwriter.

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
# **Picture it.** Quality on the left, speed on the right. They are separate charts on purpose:
# scores and milliseconds do not share a scale.

# %%
# Picture it: how often each search finds the right section, and how long it takes
import json
import matplotlib.pyplot as plt
import numpy as np

rep = json.load(open("evals/results/retrieval_eval.json"))["summary"]
names = list(rep)
x = np.arange(len(names))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
for off, key, colour, label in ((-0.2, "recall@5", "#2a78d6", "recall@5: right section in the top five"),
                                (0.2, "mrr", "#eb6834", "MRR: how high it was")):
    vals = [rep[n][key] for n in names]
    ax1.bar(x + off, vals, 0.38, color=colour, label=label)
    for xi, v in zip(x + off, vals):
        ax1.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax1.axhline(0.70, color="#888888", ls="--", lw=1, label="gate: hybrid recall@5 must stay at or above 0.70")
ax1.set_xticks(x, names)
ax1.set_ylim(0, 1.3)
ax1.legend(loc="upper left", frameon=False)
ax1.set_title("Quality: higher is better", loc="left", fontsize=11)
ms = [rep[n]["avg_latency_ms"] for n in names]
ax2.bar(x, ms, 0.5, color="#1baf7a")
ax2.set_yscale("log")
for xi, v in zip(x, ms):
    ax2.text(xi, v * 1.15, f"{v:,.0f} ms", ha="center", fontsize=9)
ax2.set_xticks(x, names)
ax2.set_ylim(top=max(ms) * 4)
ax2.set_title("Speed: average time per question (log scale, lower is faster)", loc="left", fontsize=11)
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# Go back to your "which retriever?" votes. Which questions did hybrid win, which did BM25 win alone,
# and which could no text retriever answer at all?
#
# ## Exit ticket
# One thing learned, one open question. Tomorrow: agents that use all of this to draft Priya's memo.
