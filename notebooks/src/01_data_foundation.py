# %% [markdown]
# # Session 1, part one: discover, architect, and build the data foundation
#
# **The story.** Meet Rohan. He runs a small packaging company and applies to Good Bank for a business
# loan. Meet Priya, the credit analyst handling Rohan's file. Before she can recommend anything she
# reads twelve months of bank statements, three years of financial statements and the bank's credit
# policy, checks whether Rohan owns other companies that already owe Good Bank money, and raises a
# ticket for every rule the application breaks. Rohan waits weeks, and mistakes slip through.
#
# We are building the AI assistant that does Priya's paperwork, so she can focus on the decision.
# Not an assistant that decides: the bank's own policy forbids automated approvals.
#
# **Plot twist.** Two files land on Priya's desk on the same Monday: Rohan, who looks safe, and Rahul,
# who looks weak. By the end of Session 2 you will know which of them is hiding something and which
# only looks as if he is. Section 7 below is the first clue.
#
# **The one idea of today: data first.** Every later component (extraction, the graph, the risk
# model, drift monitoring) reads from the pipeline we build today. If the pipeline silently accepts a
# duplicated transaction, the memo is wrong in a way nobody will notice until an underwriter does.
#
# Runs on the free Colab tier. No GPU, no API key. On a laptop, skip the clone cell: you already ran
# `python workshop/get_ready.py 1` in the repo folder.

# %%
!git clone https://github.com/chahatfde-arch/Day1_FDE_session.git
%cd Day1_FDE_session
!pip install -q -r requirements.txt

# %% [markdown]
# ## 1. Discovery role-play (no code, about 30 minutes)
# Your trainer plays the Head of Credit, the Chief Risk Officer and the CISO. Each holds constraints
# they only reveal when asked the right question. Interview them, then classify every data source Good
# Bank mentioned by sensitivity: public, internal, confidential, or personal data.
#
# Write down what you found. Four of those constraints will shape the architecture in an hour.

# %% [markdown]
# ## 2. Reference architecture (no code, about 10 minutes)
# On the blank CreditLens diagram, place: document extraction, policy retrieval, the related-party
# graph, the risk model, the agents, the API, the databases, Jira, Slack, tracing. Then draw the line
# that borrower personal data must not cross. Your trainer reveals the reference and the trade-offs.

# %% [markdown]
# ## 3. Data engineering for AI: the layers
# We keep three layers, and data only moves forward when it passes a contract:
#
# | Layer | Folder | What it holds | Who reads it |
# | --- | --- | --- | --- |
# | raw | `data/raw/` | files exactly as they arrived, mess included | nobody but the pipeline |
# | clean | `data/clean/` | validated, typed, standardised rows | extraction, the graph loader |
# | curated | `data/curated/` | features ready to use, one row per application | policy checks, the risk model, drift |
#
# A **contract** (a Pandera schema in `src/creditlens/pipeline/contracts.py`) is a promise about a
# table: column types, allowed values, and rules such as "every balance equals the previous balance plus
# this row's amount". A table that breaks the promise never moves to the next layer. Lineage is the
# `source_file` column every clean row carries.

# %% [markdown]
# ## 4. Generate the Good Bank data
# One command, one seed. The generator first builds the *true* transactions, then renders them in two
# bank layouts and plants problems on top. The truth goes to `data/answer_key/` (instructor only), so
# tests can prove the pipeline recovers it exactly. Everyone gets the same files, so your numbers will
# match the slides.

# %%
!python -m creditlens.generators

# %%
from pathlib import Path
for folder in sorted(Path("data/raw").iterdir()):
    print(f"{folder.name:14} {len(list(folder.iterdir())):4} files")

# %% [markdown]
# ## 5. Dirty-data bingo
# Open one raw statement. **Predict before you run:** how many different date formats will you find in
# one file? Then hunt. Seven kinds of problem are planted; each pod gets a point per kind found and
# fixed. Keep a list.

# %%
import pandas as pd
pd.set_option("display.width", 140)
raw = pd.read_csv("data/raw/statements/A002_statement.csv", dtype=str, keep_default_na=False)
raw.head(15)

# %%
print("columns:", list(raw.columns))
print("rows:", len(raw), "| exact duplicate rows:", int(raw.duplicated().sum()))
print("sample dates:", raw.iloc[:, 0].sample(8, random_state=1).tolist())
print("rows with a missing balance:", int((raw.iloc[:, -1] == "").sum()))

# %% [markdown]
# Each fix is a small, tested function in `src/creditlens/pipeline/wrangle.py`:
#
# * `parse_amount` strips currency text and Indian digit grouping, and treats brackets as negative.
# * `parse_date` tries the known formats in order; the statement window settles `03-10-25`.
# * `parse_narration` splits a narration into a channel (UPI, NEFT, CHQ_RTN, ...) and a counterparty.
# * `normalize_counterparty` upper-cases, strips legal suffixes and expands abbreviations from `aliases.yaml`.
#
# **Predict each output before you run the cell.**

# %%
from datetime import date
from creditlens.pipeline.wrangle import parse_amount, parse_date, parse_narration, normalize_counterparty

window = (date(2025, 10, 1), date(2026, 9, 30))
print(parse_amount("₹1,23,456.50"), parse_amount("(2,000)"), parse_amount("Rs. 42"))
print(parse_date("03-Apr-2026", window), parse_date("03-10-25", window), parse_date("03.10.2025", window))
print(parse_narration("UPI/TATA POWER/753217"), parse_narration("CHQ RTN ARVIND TRADERS INSUFFICIENT FUNDS"))
print(normalize_counterparty("Shakti Enterprises Pvt. Ltd."), "|", normalize_counterparty("SHAKTI ENT PVT LTD"))
print(normalize_counterparty("Amzn Seller Svcs Pvt Ltd."))

# %% [markdown]
# ## 6. Run the pipeline and enforce the contracts
# `clean_statement` detects the layout, fixes every problem it knows, rebuilds missing balances from
# the running total, and validates against `contracts.clean_transactions`. The quality report says
# what was fixed in each file: that report is your bingo card's answer sheet.

# %%
!python -m creditlens.pipeline

# %%
quality = pd.read_csv("data/clean/quality_report.csv")
quality.sort_values("alt_date_rows", ascending=False).head(10)

# %%
print(quality[quality.source_file.str.startswith("A002")].T)

# %%
features = pd.read_csv("data/curated/statement_features.csv")
features.sort_values("cheque_returns_12m", ascending=False).head(10)

# %% [markdown]
# **Break a contract on purpose.** Flip one amount's sign in the clean table and validate it again.
# The balance rule fails, and that is exactly what you want: loud, early, before the memo.

# %%
import pandera as pa
from creditlens.pipeline.contracts import clean_transactions

tx = pd.read_csv("data/clean/transactions.csv", parse_dates=["txn_date"])
print("rows that pass:", len(clean_transactions.validate(tx)))
broken = tx.copy()
broken.loc[5, "amount"] = -broken.loc[5, "amount"]
try:
    clean_transactions.validate(broken, lazy=True)
except pa.errors.SchemaErrors as e:
    print(e.failure_cases[["check", "index", "failure_case"]].head(5).to_string(index=False))

# %% [markdown]
# ## 7. The statements tell on themselves: Rohan and Rahul, part one
# Two more files, written by a second generator into `data/scenarios/` so the Good Bank data above is
# untouched. Rohan (application A901) looked safe on Monday. Rahul (A902) looked weak.
#
# Run the same cleaning step on them. **Predict first:** how many different spellings of one company
# name will the raw file hold, and what will the cleaned file show?

# %%
!python -m creditlens.scenarios

# %%
raw_901 = pd.read_csv("data/scenarios/statements/A901_statement.csv")
ravi = raw_901[raw_901["Narration"].str.contains("ravi", case=False)]
spellings = ravi["Narration"].str.extract(r"[-/]([^-/]*ravi[^-/]*)[-/]", expand=False, flags=2)
print(len(ravi), "lines mention Ravi Containers, written", spellings.nunique(), "different ways:")
print(spellings.value_counts())

# %%
from creditlens.scenarios.detect import load_statement, top_counterparties, round_trips, pass_through

a901 = load_statement("data/scenarios/statements/A901_statement.csv")   # the Session 1 cleaning step
print(top_counterparties(a901, 4).round(3).to_string(index=False))
trips = round_trips(a901)
print(len(trips), "round trips: money out to one name and back within days,", round(trips.out_amount.sum() / 1e7, 2), "crore")
trips.head()

# %%
a902 = load_statement("data/scenarios/statements/A902_statement.csv")
print(top_counterparties(a902, 4).round(3).to_string(index=False))
hand_offs = pass_through(a902, {"Kamla Devi"})
print(len(hand_offs), "times: cash in, then the same amount to", sorted(set(hand_offs.sent_to)), "the next day")

# %% [markdown]
# Neither of these is a verdict. We cannot yet see who Ravi Containers is, or who Kamla Devi is. For
# that we need the related-party graph, and that is Session 2. Notice what did the detective work here:
# `normalize_counterparty` turned three spellings into one name, and only then did the pattern show.
#
# **Your turn (10 minutes, in pods).** `round_trips` has two knobs, `window_days=5` and `tolerance=0.05`.
# Lower the tolerance to 0.01 and most of the pattern disappears. Which setting would you defend to an
# auditor, and why? There is no single right answer, but there is a wrong one: choosing the knob that
# gives the answer you hoped for.

# %%
for tol in (0.05, 0.03, 0.01):
    print("tolerance", tol, "->", len(round_trips(a901, tolerance=tol)), "round trips")

# %% [markdown]
# ## 8. Prove it
# The test suite compares every cleaned transaction with the generator's truth and the quality report
# with the manifest of planted problems. Green means the data foundation is done.

# %%
!python -m pytest -q tests/test_pipeline.py tests/test_wrangle.py tests/test_scenarios.py

# %% [markdown]
# ## 9. Homework: the async ingestion lab
# Statements come from a bank API that takes a moment per file. Fetching is I/O-bound (waiting), so
# `asyncio` or threads help; cleaning is CPU-bound (pandas), so only processes give true parallelism.
# **Predict the ranking** of sequential, asyncio, threads and processes, then measure.

# %%
!python -m creditlens.pipeline.ingest

# %% [markdown]
# ## 10. Load into SQL Server (laptop track, optional)
# Colab and the default laptop setup use SQLite, so everything runs without Docker. With Docker:
#
# ```bash
# docker compose -f infra/docker-compose.yml --profile core up -d
# pip install -e ".[mssql]"      # plus Microsoft's ODBC Driver 18
# # set DATABASE_URL in .env to the SQL Server line from .env.example
# python -m creditlens.pipeline --load
# ```
#
# ## Exit ticket
# One thing learned, one open question.
#
# **Homework before Session 2.** Draft the statement of work for Good Bank, one page, with scope, out of
# scope, deliverables per Build → Deploy → Evaluate → Maintain phase, and the three risks you would
# raise first. Run the ingestion lab above. Then, the day before:
#
# ```bash
# git pull
# python workshop/get_ready.py 2
# ```
