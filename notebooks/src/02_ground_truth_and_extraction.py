# %% [markdown]
# # Day 2: Ground truth and document extraction
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
!git clone https://github.com/chahatfde-arch/Day1_FDE_session.git
%cd Day1_FDE_session
!pip install -q -e ".[dev,llm,api,extraction,realdata]"
!python -m creditlens.generators
!python -m creditlens.pipeline --load
!python -m creditlens.data_real --only receipts


# %% [markdown]
# **Laptop users:** skip the clone cell above. The next cell finds the repository root, so the notebook
# works whether you opened it from the repo folder or from `notebooks/`.

# %%
import os
from pathlib import Path
while not Path("pyproject.toml").exists() and Path.cwd() != Path.cwd().parent:
    os.chdir("..")
assert Path("pyproject.toml").exists(), "Open this notebook from inside the Day1_FDE_session folder."
print("working folder:", Path.cwd())
# %% [markdown]
# ### Keys
# Put your Gemini key in Colab's Secrets panel (the key icon on the left) as `GEMINI_API_KEY`, and a
# Groq key as `GROQ_API_KEY` if you have one. The cell below copies them into the environment.
# **No key? Set `LLM_OFFLINE=1`.** Everything still runs, from a cached answer or a rule-based stub,
# so a room that runs out of quota can still finish the day.

# %%
import os
try:
    from google.colab import userdata
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        try:
            os.environ[k] = userdata.get(k)
        except Exception:
            pass
except ImportError:
    pass  # not on Colab: keys come from .env
os.environ.setdefault("LLM_OFFLINE", "0" if os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY") else "1")
print("offline mode:", os.environ["LLM_OFFLINE"] == "1")

# %% [markdown]
# ## 1. SOW swap (10 minutes, no code)
# Swap your SOW draft with another pod. Mark every sentence that could become scope creep. The three
# classic ones in a credit-memo project: "the system approves small loans automatically" (policy
# section 1 forbids it), "integrate with all existing systems" (which ones, read or write?), and
# "support any document format" (we support PDF and images; a photo of a screen is a different project).

# %% [markdown]
# ## 2. Labeling party (30 minutes)
# Every pod member labels the **same five receipts** independently, then you compare.
# These are real scanned receipts from the SROIE dataset (Hugging Face), standing in for the supplier
# invoices Rohan attached. The fields are: company, date, address, total.
#
# Read `ground_truth/LABELING_GUIDELINES.md` first. The guidelines decide the tie-breaks: date as printed,
# total as the grand total after tax, company as the first business name on the receipt.

# %%
import json
from pathlib import Path
from IPython.display import Image, display

labels = json.load(open("data/real/receipts/labels.json"))
party = sorted(labels)[:5]
for name in party:
    print(name)
    display(Image(filename=f"data/real/receipts/{name}", width=320))

# %% [markdown]
# Enter your labels in the shared sheet (one column per pod member). Export two members' columns as
# CSV files and measure agreement. Cohen's kappa is agreement corrected for chance: 1.0 is perfect,
# 0 is coin-flipping. Disagreements are the interesting part: argue them, update the guidelines,
# and that argument is what makes the golden set trustworthy.

# %%
!python ground_truth/agreement.py --help

# %% [markdown]
# Now compare your pod's settled labels with the SROIE human labels. You will not match 100 percent,
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
# Record your pod's numbers. On Day 7 the same script runs inside CI as a gate.

# %%
!python -m evals.extraction_accuracy --limit 10

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
# ## Exit ticket
# One thing you learned, one open question. Tomorrow: retrieval, where Priya's forty-page policy
# manual and a real four-hundred-page regulator rulebook become searchable.
