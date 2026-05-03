# CLAUDE.md — Root Workspace
# Python-Projects | Paige Leeseberg
# Last updated: April 2026

================================================================================
## WHO I AM
================================================================================

I am a Data Scientist with a background in planetary science research.
I hold a M.S. in Data Science (Rochester Institute of Technology) and a
B.S. in Physics (Iowa State University). I previously worked as a Research
Assistant at NASA Goddard Space Flight Center studying Titan's atmospheric
chemistry using ALMA radio telescope data.

I am currently a Data Scientist at a healthcare company and am actively
returning to planetary science research independently in my free time.

================================================================================
## HONEST KNOWLEDGE BASELINE
================================================================================

This section is critical. Do not assume competency beyond what is listed
here. When a topic rated 1 or 2 comes up, explain it — do not skip over it.

DATA SCIENCE AND MACHINE LEARNING:
  Predictive modeling (regression, classification, XGBoost)  — Level 1/4
    I know the concepts but need reminders on implementation details,
    parameter choices, and when to use which model. Do not assume I
    remember syntax or best practices without prompting.

  Time series analysis (ARIMA, decomposition, forecasting)   — Level 2/4
    My strongest data science area for these projects. Comfortable with
    the concepts but not expert-level. Good starting point for Project 1.

  Statistics (hypothesis testing, distributions, CIs)        — Level 1/4
    Know the basics but need reminders. Do not assume I can recall
    formulas, test assumptions, or interpret outputs without explanation.

  Bayesian thinking (priors, posteriors, MCMC)               — Level 1/4
    Conceptually familiar but have not used it hands-on enough to be
    comfortable. Needs to be built up carefully, especially for Project 3.
    Explain Bayesian concepts from first principles when they come up.

  Uncertainty quantification (error propagation, Monte Carlo) — Level 1/4
    Know what it is but limited hands-on experience. Core to Project 3 —
    needs to be taught progressively, not assumed.

PROGRAMMING:
  Python generally (functions, debugging, libraries)          — Level 2/4
    Comfortable but not expert. Can write and read Python confidently
    but may need help with advanced patterns, library-specific syntax,
    and debugging complex errors. Do not assume I know every library API.

ASTRONOMY AND PLANETARY SCIENCE:
  Titan atmospheric chemistry (photochemistry, molecules)     — Level 1/4
    Real exposure from NASA but a few years ago. Treat as needing a
    rebuild — explain concepts when they come up.

  ALMA data workflow (FITS, spectral cubes, CASA, lines)     — Level 1/4
    I know what ALMA is and worked adjacent to this data, but I do not
    have confident hands-on pipeline experience. Explain every step of
    the data workflow as we build it. Do not skip steps.

  Spectroscopy (spectral lines, frequency, flux, upper limits)— Level 1/4
    Know the basics conceptually. Need reminders on implementation,
    units, and interpretation.

  Signal processing (noise, SNR, periodograms, freq domains)  — Level 1/4
    NEW TOPIC — needs explaining from scratch. Do not assume any prior
    knowledge here. Relevant for Lomb-Scargle work in Project 1.

MATHEMATICS:
  Calculus, linear algebra, probability theory               — Level 2/4
    Comfortable with concepts, may be rusty on specifics. Can follow
    mathematical explanations but may need notation clarified.

================================================================================
## HOW TO WORK WITH ME
================================================================================

TEACHING APPROACH:
  I learn best with: short focused explanation → then I try it myself.
  Do NOT give long walkthroughs before letting me attempt something.
  Give me the core idea in a few sentences, then let me work with it.
  If I get stuck, then go deeper.

  This means:
  - Explain a concept briefly and clearly
  - Let me write the code or work through the logic
  - Correct and guide as needed
  - Move on when I demonstrate understanding

PACING:
  Do not rush through topics because I have a data science degree.
  The degree does not mean everything was retained or practiced equally.
  Check understanding before moving on — ask me to explain things back
  in my own words occasionally.

WHEN I AM WRONG:
  Correct me directly and explain why. Do not just rewrite without
  explaining what was wrong. I need to understand mistakes, not just
  have them fixed.

FLAGGING DECISIONS:
  Any time a choice involves judgment — flag it explicitly:

  ⚑ DECISION NEEDED: [description of the choice and options]

================================================================================
## THIS WORKSPACE
================================================================================

This repository contains two distinct divisions of work:

  portfolio_projects/   — Data science and ML portfolio work
  titan_projects/       — Independent planetary science research

Each division has its own CLAUDE.md. Always read it before starting work.

================================================================================
## PYTHON ENVIRONMENT
================================================================================

- Python version: to be confirmed — check the project .venv or README
- Each project has its own virtual environment (.venv in project root)
- Never install libraries globally — always activate the correct .venv first
- Always confirm which environment is active before installing anything

Virtual environment commands (Windows PowerShell):
  Create:    python -m venv .venv
  Activate:  .venv\Scripts\activate
  Install:   pip install <library>
  Freeze:    pip freeze > requirements.txt

When I am confused about environments, explain what is happening and why
before fixing it — do not just run commands without context.

================================================================================
## GENERAL CODING PREFERENCES
================================================================================

BEFORE WRITING ANY CODE:
- Always explain what you are about to write and why
- Describe what the function does in plain language first
- If multiple approaches exist, describe the tradeoffs and ask me to choose
- Never assume — ask if something is ambiguous

CODE STYLE:
- PEP 8 formatting throughout
- Descriptive variable names — no single letters except loop indices
- Functions over repetition — if something runs twice, make it a function
- NumPy-style docstrings on every function
- Inline comments on non-obvious logic only
- No unused imports, no dead code

NOTEBOOKS:
- Every notebook opens with a markdown cell: title, author, date, purpose
- Markdown cells between major sections explaining what and why
- Figures always have titles, labeled axes, and units
- Print intermediate results with labels
- One idea per cell

================================================================================
## GIT WORKFLOW
================================================================================

Commit at milestones — when something works end to end.

Commit message format:
  <type>: <short description>
  Types: feat / fix / data / docs / refactor

Never commit:
- Raw data files (add data/ to .gitignore)
- API keys or credentials
- .venv folders
- Large binary files (FITS, .pkl over 50MB)

================================================================================
## DATA FILE SAFETY
================================================================================

Raw data is read-only by default unless a project CLAUDE.md says otherwise.
Never overwrite or modify a raw data file. All processing writes to a
separate processed/ or outputs/ folder. If unsure — ask before touching it.

================================================================================
## WHAT NOT TO DO
================================================================================

- Do not assume knowledge beyond what is listed in the baseline above
- Do not suggest paid APIs without flagging the cost
- Do not write code without explaining it first
- Do not make judgment calls that belong to me
- Do not overwhelm with long explanations before letting me try
- Do not skip error handling on file I/O or data loading

================================================================================