# CreditLens AI workshop: pre-work

**TAN GLOBUS AI · Forward Deployed Engineer workshop · Session 1 is Saturday 14 November, 3 pm**

This repository has everything you need to get your laptop ready before Session 1. It takes about
30 to 45 minutes, most of it waiting for downloads. Do it today, so that on Saturday we spend our
time building, not installing.

By the end you will have:

- Python, Git and VS Code installed
- this repository on your laptop
- the Good Bank synthetic data generated (40 companies, 30 loan applications, messy bank
  statements, financial statement PDFs, a credit policy manual)
- one command that proves it all works

Never used a terminal before? Follow the steps in order. Every command is copy and paste.

---

## Step 1: Install the three tools (about 10 minutes)

### Python 3.10 or newer

- **Windows:** download the latest Python 3 from <https://www.python.org/downloads/>. Run the
  installer. On the first screen **tick "Add python.exe to PATH"** before you click Install. This
  one checkbox saves most of the setup problems we see.
- **Mac:** download the installer from the same page and run it.
- Already have Python? Check the version in Step 3. Anything from 3.10 upward is fine.

### Git

- **Windows:** download from <https://git-scm.com/download/win> and run the installer. Every default
  option is fine; keep clicking Next.
- **Mac:** open the Terminal app and type `git --version`. If Git is missing, macOS offers to
  install it. Click Install.

### VS Code

- Download from <https://code.visualstudio.com/> and install it.
- Open VS Code, click the Extensions icon on the left (four squares), search for **Python** and
  install the one published by Microsoft. Then search for **Jupyter** and install that too.

Close and reopen any terminal windows after installing. New installs are only seen by terminals
opened afterwards.

---

## Step 2: Open a terminal

- **Windows:** press the Start key, type `PowerShell`, press Enter.
- **Mac:** press Command and Space, type `Terminal`, press Enter.

Everything below is typed into this window. Press Enter after each command.

---

## Step 3: Check Python and Git

```
python --version
git --version
```

You should see something like `Python 3.12.4` and `git version 2.45.0`.

- If `python` is not recognised on Windows, try `py --version`. If that works, use `py` instead of
  `python` in every command below.
- If Python says 3.9 or lower, install a newer one from Step 1.

---

## Step 4: Get the code

```
cd Documents
git clone https://github.com/chahatfde-arch/Day1_FDE_session.git
cd Day1_FDE_session
```

This downloads the repository into a folder called `Day1_FDE_session` inside your Documents.
Every later command must be run from inside this folder. If you open a new terminal later, run
`cd Documents/Day1_FDE_session` first.

---

## Step 5: Create a virtual environment

A virtual environment is a private copy of Python for this project, so nothing we install can
disturb anything else on your laptop.

**Windows (PowerShell):**

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses with a message about execution policies, run this once, answer `Y`, then
try the activate command again:

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Mac:**

```
python3 -m venv .venv
source .venv/bin/activate
```

When it works, the prompt starts with `(.venv)`. You need to activate the environment every time
you open a new terminal for this project.

---

## Step 6: Install the project (about 5 to 10 minutes)

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This downloads the libraries the project uses. A wall of text is normal. It is finished when you
get your prompt back with no red `ERROR` lines near the end.

---

## Step 7: Generate the data and run the check

```
python workshop/get_ready.py 0
```

This one command does five things, and prints `OK` or `FAILED` after each:

1. generates the Good Bank data into `data/`
2. runs the data pipeline and loads a small database
3. fetches small samples of public datasets (needs internet)
4. runs the setup check
5. runs the pre-work tests

The last lines should read:

```
Ready for Day 1.
...
5 passed
Ready. Everything for this session works on your laptop.
```

Two lines inside step 3 may say `receipts: FAILED` or `charts: FAILED` on some corporate laptops.
That is fine for now; we will share those samples in Session 1.

---

## Step 8: Send your screenshot

Take a screenshot of the terminal showing the `Ready.` line and reply to the welcome email with it
by **Saturday 14 November, 12 noon**. If anything says `FAILED`, send the screenshot anyway with
the first error message you see. That is far more useful to us than a message saying "it did not
work", and we will fix it before the session.

---

## Optional, if you have 20 more minutes

Open the pre-work notebook in VS Code: File, Open Folder, choose `Day1_FDE_session`, then open
`notebooks/00_prework.ipynb`. When VS Code asks for a kernel, pick the `.venv` one. Sections 5 to 7
are a short Python warm-up, a Linux and tooling refresher, and a GraphQL lab. The topic pages are in
`workshop/session-0-prework/`.

---

## Something went wrong?

| What you see | What to do |
| --- | --- |
| `python` is not recognised | Windows: use `py` instead. Or reinstall Python with "Add to PATH" ticked. |
| `git` is not recognised | Install Git from Step 1, then close and reopen the terminal. |
| `Activate.ps1 cannot be loaded` | Run the `Set-ExecutionPolicy` command in Step 5, then activate again. |
| `No module named creditlens` | Your virtual environment is not active, or Step 6 did not finish. Activate, then run Step 6 again. |
| `pip install` fails on a corporate network | Try from a home network or a phone hotspot. Proxies block some downloads. |
| `receipts: FAILED` or `charts: FAILED` | Ignore for now. The rest of the check still passes. |
| Anything else | Send a screenshot of the first error. We fix setups before every session. |

---

## Sessions 1 and 2: what to run

Everything for the two sessions is in this repository now. The day before each session, from the repo folder
with your virtual environment active:

```bash
git pull
python -m pip install -r requirements.txt     # picks up the new extras: API, extraction, retrieval
python workshop/get_ready.py 1                # before Session 1 (Saturday)
python workshop/get_ready.py 2                # before Session 2 (Sunday)
```

Each ends with `Ready`, or a short list of what failed. Send a screenshot of the list and we fix it before the session.

| | Laptop track | Colab track |
| --- | --- | --- |
| Session 1 | `workshop/session-1-data-and-extraction/README.md` | `notebooks/01_data_foundation.ipynb`, then `02_ground_truth_and_extraction.ipynb` |
| Session 2 | `workshop/session-2-retrieval/README.md` | `notebooks/03_retrieval.ipynb` |
| The story cases (Rohan, Rahul and four more) | `python -m creditlens.scenarios`, see `docs/scenarios.md` | `notebooks/08_story_scenarios.ipynb` |

The trainer's live demo, cell by cell in slide order, is `workshop/live_demo_session1.py` and `live_demo_session2.py`.
Open one in VS Code and run the cells to follow along. No API key is needed for anything in Sessions 1 and 2: set
`LLM_OFFLINE=1` in your `.env` and the extraction step uses a built-in stub.

Session 2 embeds about 1,400 policy chunks the first time it runs (two to three minutes on a laptop). The trainer
shares a ready-made `data/cache/retrieval/` folder before the session; drop it into `data/` to skip the wait.

## What is in this repository

```
Day1_FDE_session/
├── README.md                 this guide
├── requirements.txt          what to install
├── pyproject.toml            the Python package definition
├── .env.example              settings template
├── workshop/
│   ├── get_ready.py          the one-command check
│   ├── live_demo_session1.py, live_demo_session2.py   the trainer's demo, cell by cell
│   ├── session-0-prework/    one page per pre-work topic
│   ├── session-1-data-and-extraction/
│   └── session-2-retrieval/
├── notebooks/                00_prework, 01, 02, 03 and 08 (Colab)
├── src/creditlens/           the whole CreditLens package (Sessions 3 and 4 code is here too; ignore it for now)
├── evals/, ground_truth/     the scoreboards and the hand-checked answers
├── docs/                     contracts, data sources, extraction, retrieval, the story cases
└── tests/                    what get_ready.py runs
```

The data folders (`data/`) are created on your laptop by `get_ready.py` and are not in Git, so
everyone generates identical data from the same seed.
