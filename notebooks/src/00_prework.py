# %% [markdown]
# # CreditLens AI, pre-work: get your tools ready
#
# **Finish this notebook at least 48 hours before Day 1.** It takes about 30 minutes.
#
# Rohan needs a loan fast, and Priya is drowning in paperwork. Over seven sessions we build
# the AI assistant that does Priya's paperwork at Good Bank, so she can focus on the decision.
# Before Day 1, make sure the tools work on your side.
#
# What you will do here:
#
# 1. Clone the repo and install the Python package.
# 2. Generate the synthetic Good Bank data (it is created from one random seed, so everyone has
#    the same data).
# 3. Download the small real-data samples we use alongside it.
# 4. Run the setup check and paste the output in the pre-work form.
# 5. Three short Python exercises checked by tests (memory, exceptions, file I/O).
#
# Everything here is free. No API key is needed for the pre-work.

# %% [markdown]
# ## 1. Clone and install
# The `-e` flag installs the package in "editable" mode: edits under `src/` take effect
# without reinstalling, which is how we work for the whole week.

# %%
!git clone https://github.com/chahatfde-arch/Day1_FDE_session.git
%cd Day1_FDE_session
!pip install -q -e ".[dev,realdata]"


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
# ## 2. Generate the Good Bank data
# One command builds 40 companies, 30 loan applications, 12 months of messy bank statements each,
# three years of financial statements as PDFs (a quarter of them as noisy scans), the credit policy
# manual and a loan history. Same seed, same data, every time: that matters because on Day 2 we
# label a golden set, and everyone must be labelling the same documents.

# %%
!python -m creditlens.generators

# %% [markdown]
# ## 3. Fetch the real-data samples
# The story is synthetic, but four small real public datasets plug in where realism matters:
# scanned receipts for OCR (Hugging Face), a real lender rulebook for retrieval (sba.gov),
# real charts for visual retrieval (Hugging Face), and a real small-business loan book for the
# risk model (Kaggle). None of them contain Good Bank borrowers. Total download is about 12 MB
# plus a one-time 46 MB Kaggle archive that is sampled down.

# %%
!python -m creditlens.data_real

# %% [markdown]
# ## 4. Run the setup check
# Paste the output in the pre-work form. WARN lines are fine for Day 1; MISSING lines are not.

# %%
!python -m creditlens.setup_check

# %% [markdown]
# ## 5. Python warm-up (P1 to P5)
# Three tiny exercises. Each one is checked by a test. You will use these ideas on Day 1 when
# we wrangle bank statements.
#
# ### 5a. Data structures and memory
# A bank statement row arrives as a dictionary. Write `dedupe(rows)` that removes exact duplicate
# rows while keeping the first occurrence and the original order. Hint: a dictionary is not hashable,
# but a tuple of its items is.

# %%
def dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        key = tuple(sorted(r.items()))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


rows = [{"date": "01/04/2026", "amt": "1,000"}, {"date": "01/04/2026", "amt": "1,000"}, {"date": "02/04/2026", "amt": "500"}]
assert dedupe(rows) == [rows[0], rows[2]]
print("5a passed")

# %% [markdown]
# ### 5b. Exceptions
# Amounts arrive as text: `"₹1,23,456.50"`, `"(2,000)"` (brackets mean negative), or junk.
# Write `parse_amount(text)` that returns a float, or raises `ValueError` with a helpful message.
# Raising a clear error is better than silently returning 0: a silent zero would reach an underwriter.

# %%
def parse_amount(text: str) -> float:
    cleaned = text.strip().replace("₹", "").replace("INR", "").replace("Rs.", "").replace(",", "").strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    if negative:
        cleaned = cleaned[1:-1]
    try:
        value = float(cleaned)
    except ValueError as e:
        raise ValueError(f"not an amount: {text!r}") from e
    return -value if negative else value


assert parse_amount("₹1,23,456.50") == 123456.5
assert parse_amount("(2,000)") == -2000.0
try:
    parse_amount("twelve")
except ValueError as e:
    print("good, it raised:", e)
print("5b passed")

# %% [markdown]
# ### 5c. File I/O and the `with` statement
# Count how many statement files exist and how many rows the first one has. Opening files inside
# `with` guarantees they are closed even if an error happens halfway through.

# %%
from pathlib import Path
import csv

files = sorted(Path("data/raw/statements").glob("*.csv"))
if not files:
    raise SystemExit(f"No statement files under {Path.cwd() / 'data/raw/statements'}. "
                     "Run section 2 (python -m creditlens.generators) first, from the repository folder.")
with open(files[0], newline="", encoding="utf-8") as f:
    n_rows = sum(1 for _ in csv.reader(f)) - 1
print(f"{len(files)} statement files; the first one has {n_rows} rows")
assert len(files) == 30
print("5c passed")

# %% [markdown]
# ## 6. Optional: Linux and tooling (P11 to P15, A11, A14, A15)
# On WSL, Linux or a GitHub Codespace, try these in a terminal:
#
# ```bash
# ls -la data/raw            # file system and permissions
# du -sh data/raw/*          # sizes per folder
# top                        # process monitoring (q to quit)
# export LLM_OFFLINE=1       # environment variables: the gateway reads this one
# python -m pytest -q tests/test_wrangle.py   # the test runner we use all week
# ```
#
# In VS Code, install the Python, Jupyter and Docker extensions, open the repo, and set a breakpoint
# in `src/creditlens/pipeline/wrangle.py` inside `parse_amount`. Run `pytest tests/test_wrangle.py`
# from the Testing panel and watch the debugger stop there.
#
# ## 7. Optional: GraphQL lab (A6 to A10)
# `pip install strawberry-graphql` and follow `docs/graphql_lab.md` (schemas, resolvers, the N+1
# problem, mutations and versioning). Not needed for the credit memo, useful for API design in general.
#
# See you on Day 1.
