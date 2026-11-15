# %% [markdown]
# # Session 1, part two: ground truth and document extraction
#
# **Where we are in the story.** Rohan's file has arrived at Good Bank. Before Priya can write
# anything, someone has to read three years of financial statements and the supplier invoices he
# attached. Some are clean PDFs. Some are scans. One is rotated. Today we teach the machine to read
# them, and, more importantly, we build the ruler we will measure it with all week.
#
# **The one idea of today: evaluation first.** We write the golden answers *before* we build the
# extractor. Every later component (retrieval, agents, the judge) is scored against ground truth
# that humans agreed on. Without that ruler, "it looks right" is the only test, and that is how
# mistakes reach an underwriter.
#
# Topics: A1 to A5 (API and tracing), L1 to L10 (LLM structured output), M1, M2, M8, M10, M11 (OCR and
# vision), OB1 (observability).
#
# Runs on the free Colab tier. A GPU is optional (OCR is faster with it, but works on CPU).

# %% [markdown]
# ## Setup
# Same as Day 1. If you are catching up, the `day2-start` branch has milestones 1 and 2 done.

# %%
import sys
if "google.colab" in sys.modules:   # Colab only. On a laptop this cell does nothing.
    !git clone https://github.com/chahatfde-arch/Day1_Day2_FDE_session.git
    %cd Day1_Day2_FDE_session
    !pip install -q -e ".[dev,llm,api,extraction,realdata]"
    !python -m creditlens.generators
    !python -m creditlens.pipeline --load
    !python -m creditlens.data_real --only receipts
    !python -m creditlens.scenarios
else:
    print("Not on Colab: skipping the clone. The next cell finds your repo folder.")

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

# %% [markdown]
# ### Offline mode
# Today we run with no AI keys, on purpose. With `LLM_OFFLINE=1` every AI call returns a saved answer
# or a simple rule-based answer instead, so everything still runs and every pod gets the same numbers.
# (Later, to use a real model, put a free Gemini or Groq key in a `.env` file and change the `"1"` to `"0"`.)

# %%
import os
os.environ["LLM_OFFLINE"] = "1"   # today we run offline: same numbers for everyone, no API key needed
print("offline mode:", os.environ["LLM_OFFLINE"] == "1")

# %% [markdown]
# ## 1. SOW swap (10 minutes, no code)
# Swap your SOW draft with another pod. Mark every sentence that could become scope creep. The three
# classic ones in a credit-memo project: "the system approves small loans automatically" (policy
# section 1 forbids it), "integrate with all existing systems" (which ones, read or write?), and
# "support any document format" (we support PDF and images; a photo of a screen is a different project).

# %% [markdown]
# ## 2. Labeling party (30 minutes)
# Everyone works **alone**. You label the **same five receipts**, then compare your answers with the official ones.
# These are real scanned receipts from the SROIE dataset (Hugging Face), standing in for the supplier
# invoices Rohan attached. The fields are: company, date, address, total.
#
# Read `ground_truth/LABELING_GUIDELINES.md` first. The guidelines decide the tie-breaks: date as printed,
# total as the grand total after tax, company as the first business name on the receipt.

# %%
import json, os
from pathlib import Path
from IPython.display import Image, display

while not Path("pyproject.toml").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir("..")   # make sure we are in the repo folder
if not Path("data/real/receipts/labels.json").exists():
    !python -m creditlens.data_real --only receipts

labels = json.load(open("data/real/receipts/labels.json"))
party = sorted(labels)[:5]
for name in party:
    print(name)
    display(Image(filename=f"data/real/receipts/{name}", width=320))

# %% [markdown]
# Next you will type your labels into a cell and measure how well they match the official answers.
# The tool uses Cohen's kappa: agreement corrected for chance. 1.0 is perfect, 0 is coin-flipping.
# Where you disagree is the interesting part: was the guideline unclear, or did you misread the receipt?

# %%
!python ground_truth/agreement.py --help

# %% [markdown]
# **Demo run first.** Two example label sheets for the same five receipts live in
# `ground_truth/party_demo/`. Run the tool on them to see what a good box and a weak box look
# like. Then you do the same with your own labels.
#
# **Predict before you run:** which two boxes will score lowest?
#
# The `!` at the start only works inside a notebook cell. In a terminal (PowerShell or bash),
# type the same command without the `!`.

# %%
!python ground_truth/agreement.py ground_truth/party_demo/member_a.csv ground_truth/party_demo/member_b.csv --key file

# %% [markdown]
# **Picture it.** Kappa for each box from the demo run. The dashed lines are the tool's own
# cut-offs: 0.6 is usable, 0.8 is strong.

# %%
# Picture it: kappa for each box
# To draw your own: swap the two files for "my_labels.csv" and "ground_truth/party_demo/official.csv".
import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, "ground_truth")
from agreement import agreement_table

table = agreement_table(pd.read_csv("ground_truth/party_demo/member_a.csv"),
                        pd.read_csv("ground_truth/party_demo/member_b.csv"), key="file")
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(table.field, table.kappa, color="#2a78d6", height=0.55)
for y, v in enumerate(table.kappa):
    ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=10)
for x, word in ((0.6, "usable"), (0.8, "strong")):
    ax.axvline(x, color="#888888", lw=1, ls="--")
    ax.text(x, -0.75, word, ha="center", fontsize=9, color="#555555")
ax.set_xlim(0, 1.12)
ax.set_ylim(len(table) - 0.5, -1)
ax.set_title("Kappa per box: date and total agree, company and address are weak", loc="left", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Your turn.** Type what you read on each receipt between the quotes, then run the cell.
# It saves your answers and scores them against the official answers in `ground_truth/party_demo/official.csv`.

# %%
import pandas as pd
my_labels = pd.DataFrame([
    {"file": "X00016469670.jpg", "company": "", "date": "", "address": "", "total": ""},
    {"file": "X00016469671.jpg", "company": "", "date": "", "address": "", "total": ""},
    {"file": "X51005200931.jpg", "company": "", "date": "", "address": "", "total": ""},
    {"file": "X51005230605.jpg", "company": "", "date": "", "address": "", "total": ""},
    {"file": "X51005230616.jpg", "company": "", "date": "", "address": "", "total": ""},
])
my_labels.to_csv("my_labels.csv", index=False)
!python ground_truth/agreement.py my_labels.csv ground_truth/party_demo/official.csv --key file

# %% [markdown]
# Now look at the official labels next to your own. You will not match 100 percent,
# and that is the lesson: even professional annotators disagree on addresses and company suffixes.

# %%
for name in party:
    print(name, labels[name]["entities"])

# %% [markdown]
# Build the golden set the evaluation will use all week: 10 financial statements (from the generator's
# truth) and all 40 receipts (from the SROIE labels). It lives in `ground_truth/`, which is committed,
# unlike `data/`, which is regenerated.

# %%
!python -m creditlens.extraction.golden
!python -c "import json; g=json.load(open('ground_truth/golden_set.json')); print({k: len(v) for k, v in g.items()})"

# %% [markdown]
# ## 3. API and tracing (A1 to A5)
# Priya will not call Python functions. She will click a button in a web app that calls an API.
# `creditlens.api` is a FastAPI service with:
#
# - **Dependency injection** (`Depends`): routes ask for "the database" and "the current user" and
#   FastAPI provides them. Tests swap in a temporary database without touching the route code.
# - **CORS**: the browser blocks a web page on one origin from calling an API on another unless
#   the API says it is allowed. We allow the local dev origins only.
# - **Bearer tokens**: a signed JWT says who you are and which region you may see. In the workshop
#   the token is minted by a mock login; on Day 6 we put Keycloak in front of it.
# - **Tracing**: every request becomes a Langfuse trace, so when something goes wrong on Day 4 we
#   can open the exact run and see every LLM call inside it.
#
# Start the API in the background and call it.

# %%
!python -m creditlens.pipeline --load

# %%
import subprocess, sys, time, requests

API = "http://127.0.0.1:8000"

def alive():
    try:
        return bool(requests.get(f"{API}/health", timeout=2).json().get("database_ok"))
    except requests.RequestException:
        return False

api = None
if alive():
    print("an API is already running on port 8000; reusing it")
else:
    api = subprocess.Popen([sys.executable, "-m", "creditlens.api"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        if alive():
            break
        time.sleep(1)
print(requests.get(f"{API}/health").json())

def login(user_id):
    r = requests.post(f"{API}/auth/token", json={"user_id": user_id})
    if r.status_code != 200:
        raise SystemExit(f"login failed ({r.status_code}): {r.json().get('detail')}")
    return r.json()

tok = login("U01")
print("token for U01 (North analyst):", tok["access_token"][:40], "...")

# %%
headers = {"Authorization": f"Bearer {tok['access_token']}"}
page = requests.get(f"{API}/applications", headers=headers).json()
apps = page["applications"]
print(page["count"], "applications visible to U01; regions:", {a["region"] for a in apps}, "| filter:", page["region_filter"])

# %% [markdown]
# **Predict before you run:** U05 is the Head of Credit. How many applications will U05 see?

# %%
tok5 = login("U05")
page5 = requests.get(f"{API}/applications", headers={"Authorization": f"Bearer {tok5['access_token']}"}).json()
print(page5["count"], "applications visible to U05")

# %% [markdown]
# If you set Langfuse keys, open your Langfuse project now: each request above is a trace.
# Without keys the decorator is a no-op, which is deliberate: tracing must never break the product.

# %%
if api is not None:
    api.terminate()

# %% [markdown]
# ## 4. Structured extraction (L1 to L10, M1, M2, M8, M10, M11)
#
# ### How a PDF becomes numbers
# 1. **Text PDFs** carry their text inside the file. `pypdf` reads it in milliseconds.
# 2. **Scanned PDFs** are pictures. We render each page to an image and run OCR (tesseract if
#    installed, otherwise rapidocr, which is pure pip). OCR output is noisy: `1,23,456` can come back
#    as `1,23,4S6`.
# 3. **Rotated pages**: we OCR the page at 0, 90, 180 and 270 degrees and keep the orientation that
#    produces the most real words. Simple, and it works.
# 4. The text goes to the LLM with a **JSON schema** generated from a Pydantic model. The model must
#    answer with exactly those fields and types.
# 5. If the answer does not validate, the gateway sends the error back and **retries**. If the
#    provider is down, it **falls back** to the next one. Every answer is **cached** on disk.
#
# ### Why a discriminated union (L7)
# Rohan's file holds two document types. `ExtractionEnvelope.document` is either a
# `FinancialStatementExtraction` or a `ReceiptExtraction`, and the `doc_type` field decides which.
# One schema, two shapes, no `if` statements in the calling code.

# %%
from creditlens.schemas import ExtractionEnvelope
import json
print(json.dumps(ExtractionEnvelope.model_json_schema(), indent=1)[:1500], "...")

# %% [markdown]
# ### A text PDF first

# %%
from creditlens.extraction.ocr import pdf_to_text
kinds = json.load(open("data/answer_key/financial_doc_kinds.json"))  # instructor file: which PDFs are scans
text_id = next(c for c, k in kinds.items() if k == "text")
text, ocr_used = pdf_to_text(f"data/raw/financials/{text_id}_financials.pdf")
print("ocr used:", ocr_used)
print(text[:800])

# %%
from creditlens.extraction.extract import extract_financials
fin = extract_financials(text_id)
print(fin.company_name, fin.unit, "ocr:", fin.ocr_used)
for y in fin.years:
    print(y.model_dump())

# %% [markdown]
# ### Now a scan, and the curveball
# **Curveball card: a rotated two-page scanned table arrives.** One of the scanned PDFs has page 2
# rotated 90 degrees. Predict: will `pdf_to_text` return the balance sheet from page 2?

# %%
rot_id = next(c for c, k in kinds.items() if k == "scan_rotated")
text, ocr_used = pdf_to_text(f"data/raw/financials/{rot_id}_financials.pdf")
print("ocr used:", ocr_used, "| mentions balance sheet:", "balance sheet" in text.lower())
print(text[-600:])

# %%
fin_scan = extract_financials(rot_id)
truth = json.load(open("data/answer_key/financials_truth.json"))[rot_id]["years"]["FY2026"]
latest = fin_scan.latest()
print("extracted revenue:", latest.revenue, "| truth:", truth["revenue"])
print("extracted total debt:", latest.total_debt, "| truth:", truth["total_debt"])

# %% [markdown]
# ### A receipt
# Same function, different `doc_type`. Compare with the human label.

# %%
from creditlens.extraction.extract import extract_receipt
name = party[0]
r = extract_receipt(f"data/real/receipts/{name}")
print("model :", r.model_dump(include={"company", "date", "total"}))
print("human :", {k: labels[name]["entities"][k] for k in ("company", "date", "total")})

# %% [markdown]
# ### Look inside the gateway
# How many provider calls did we make, and how many came from cache? Run the receipt cell again and
# watch the cache hits go up. In a live room this is what keeps you inside the free tier.

# %%
from creditlens.gateway.llm import get_gateway
g = get_gateway()
print("provider calls:", g.calls, "| cache hits:", g.cache_hits, "| models:", g.models)

# %% [markdown]
# ## 5. First evaluation (goes on the scoreboard)
# Field-level accuracy against the golden set. Strings are compared after normalising case and
# punctuation; numbers must be within 1 percent; dates are parsed before comparing.
# Record your pod's numbers. In Session 4 the same script runs inside CI as a gate.

# %%
!python -m evals.extraction_accuracy --limit 10

# %% [markdown]
# **Picture it.** Accuracy for every box, on both kinds of document. The dashed line is the 85
# percent pass mark from `evals/thresholds.json` (it applies to the overall score).

# %%
# Picture it: accuracy for every box, against the pass mark
import json
import matplotlib.pyplot as plt

res = json.load(open("evals/results/extraction_accuracy.json"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, kind, title in zip(axes, ("financial_statement", "receipt"), ("Company accounts", "Receipts")):
    per = res[kind]["per_field"]
    names = list(per)
    acc = [per[n]["accuracy"] * 100 for n in names]
    ax.barh(names, acc, color="#2a78d6", height=0.6)
    for y, v in enumerate(acc):
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=9)
    ax.axvline(85, color="#888888", ls="--", lw=1)
    ax.text(85, -1.0, "pass mark 85%", ha="center", fontsize=9, color="#555555")
    ax.set_xlim(0, 115)
    ax.set_ylim(len(names) - 0.5, -1.4)
    ax.set_title(f"{title}: {res[kind]['overall']['accuracy']:.1%} of boxes right", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Find the worst failure.** Open `evals/results/extraction_accuracy.json`, find the document with
# the lowest score, and then find its trace in Langfuse (or print its OCR text here). Is the problem
# OCR (the text is wrong) or the model (the text is right, the JSON is wrong)? The fix is different
# for each, and that distinction is the whole point of tracing.

# %%
res = json.load(open("evals/results/extraction_accuracy.json"))
for kind in ("financial_statement", "receipt"):
    section = res.get(kind) or {}
    print(f"== {kind}: per-field accuracy", section.get("per_field"))
    for mistake in (section.get("sample_mistakes") or [])[:5]:
        print("  ", mistake)

# %% [markdown]
# ## 6. Rohan and Rahul, part one and a half: what they declared versus what reached the bank
# Extraction gives us the numbers *printed* on a document, and the evaluation above checks that we read
# them correctly. It cannot tell us whether the business behind them is real. For that we hold one
# document against another: the turnover each borrower declared on the application, and the money that
# actually reached the bank account in the last twelve months.
#
# **Predict first.** Rohan looked safe on Monday and Rahul looked weak. Whose bank account will show
# *more* money than the declared turnover?

# %%
import pandas as pd
from creditlens.scenarios.detect import load_statement, revenue_gap, round_trips

scn_apps = pd.read_csv("data/scenarios/applications/applications.csv").set_index("application_id")
story = {}
for app, who in (("A901", "Rohan"), ("A902", "Rahul")):
    clean = load_statement(f"data/scenarios/statements/{app}_statement.csv")
    gap = revenue_gap(clean, scn_apps.loc[app, "declared_revenue_inr_000"])
    story[app] = clean
    print(f"{who} ({app}): declared {gap['declared_revenue'] / 1e7:.2f} crore, reached the account "
          f"{gap['visible_inflow'] / 1e7:.2f} crore -> {gap['visible_over_declared']:.0%} of declared")

# %% [markdown]
# Rohan's account shows *more* than his declared turnover. That sounds like good news, until you
# remember yesterday's round trips: money that went out to Ravi Containers and came back days later is
# counted as money in. Take it out and see what is left.

# %%
trips = round_trips(story["A901"])
inflow = story["A901"].query("amount > 0")["amount"].sum()
came_back = trips["back_amount"].sum()
print(f"Rohan: {len(trips)} round trips brought back {came_back / 1e7:.2f} crore")
print(f"money in without the round trips: {(inflow - came_back) / 1e7:.2f} crore, "
      f"{(inflow - came_back) / (scn_apps.loc['A901', 'declared_revenue_inr_000'] * 1000):.0%} of declared")

# %% [markdown]
# So the two files trade places. Rohan's account is inflated by money that only passed through it.
# Rahul's account shows about 60 percent of what he declared. Where is the rest? Remember yesterday's
# clue: 36 cash deposits, each followed the next day by the same amount to KAMLA DEVI. Money that
# belongs to his business is ending up somewhere else.
#
# **The lesson for the evaluation thread.** An extractor can score 100 percent and still hand Priya a
# misleading number, because "the document says 4.67 crore" and "the business earns 4.67 crore" are
# different claims. Our golden set measures the first. Cross-checking documents against each other is
# how you approach the second, and it is a lead for a person, not a verdict.
#
# Two questions remain open, and both need tomorrow's tools: **who is Ravi Containers** to Rohan, and
# **who is Kamla Devi** to Rahul? Neither name is in the bank statement's metadata. They are in the
# company registry, and the registry is a graph.

# %% [markdown]
# ## Exit ticket
# One thing you learned, one open question. Tomorrow: retrieval, where Priya's policy
# manual and a real four-hundred-page regulator rulebook become searchable, and the related-party graph
# tells us who Ravi Containers and Kamla Devi really are.
