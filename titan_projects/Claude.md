# CLAUDE.md — Titan Research Projects Division
# titan_projects/ | Paige Leeseberg
# Last updated: April 2026

================================================================================
## WHAT THIS DIVISION IS
================================================================================

Independent planetary science research focused on Titan — Saturn's largest
moon. This is active scientific research, not portfolio work. The standard
is research-grade rigor: reproducible, honest, well-documented, and
defensible to domain experts.

Long-term goals: re-engage with the planetary science research community,
reconnect with former NASA collaborators, and contribute to preprint or
peer-reviewed publications.

================================================================================
## HONEST KNOWLEDGE BASELINE FOR THIS DIVISION
================================================================================

Read this carefully. This is where I actually am — not where my resume
suggests I am. Calibrate all explanations accordingly.

DOMAIN KNOWLEDGE:
  Titan atmospheric chemistry                                 — Level 1/4
    I have real NASA exposure but it was a few years ago and I was a
    junior researcher. I know Titan has N2/CH4 atmosphere, complex
    organic photochemistry, and a haze layer. Beyond that, assume I need
    concepts re-explained. Do not reference chemical pathways or molecular
    species without briefly explaining what they are.

  ALMA data workflow (FITS, spectral cubes, CASA)            — Level 1/4
    I know ALMA is a radio telescope array and I worked adjacent to this
    data. I do not have a confident independent pipeline workflow. Explain
    every step — what the file format contains, what each processing step
    does, and why. Do not skip steps or assume familiarity.

  Spectroscopy (lines, frequency, flux, upper limits)        — Level 1/4
    I understand that molecules emit and absorb at specific frequencies
    and that we look for those signatures. Beyond that, explain the
    mechanics. Upper limit calculations in particular need to be walked
    through carefully when we get to Project 2.

  Signal processing (noise, SNR, periodograms)               — Level 1/4
    BRAND NEW to me. Start from scratch. When Lomb-Scargle or periodograms
    come up in Project 1, explain what they are before using them.
    Do not assume I know what SNR means in a practical workflow context.

DATA SCIENCE SKILLS AS APPLIED TO RESEARCH:
  Time series analysis                                       — Level 2/4
    My strongest relevant skill. Comfortable with ARIMA and decomposition
    conceptually. May need reminders on implementation details and
    interpreting outputs in a scientific context rather than a business one.

  Statistics and uncertainty                                  — Level 1/4
    Know the basics. Error propagation and uncertainty reporting in a
    scientific context is different from business data science — explain
    the research conventions as we go. Always explain what a statistical
    result means physically, not just mathematically.

  Bayesian methods and MCMC                                   — Level 1/4
    Conceptually aware, not hands-on comfortable. Project 3 relies heavily
    on this. Build it up progressively — do not introduce emcee or corner
    plots without first explaining the underlying logic.

  Monte Carlo methods                                         — Level 1/4
    Know the concept (sample many times, look at the distribution of
    results) but have not implemented it for scientific uncertainty work.
    Walk me through the logic before any implementation.

  Python                                                      — Level 2/4
    Comfortable but not expert. Astronomy libraries (astropy, specutils,
    spectral-cube) are largely new to me. Explain library-specific
    patterns and methods — do not assume I know the API.

MATHEMATICS:
  Calculus, linear algebra, probability                       — Level 2/4
    Conceptually solid, may be rusty on specifics. Can follow math
    explanations but may need notation clarified.

================================================================================
## TEACHING APPROACH FOR THIS DIVISION
================================================================================

I am learning two things simultaneously:
  1. The planetary science domain and research methods
  2. The astronomy Python software stack

This means the learning load is high. Pace accordingly.

MY PREFERRED LEARNING STYLE:
  Short focused explanation → I try it → correct and guide → move on.
  Do not front-load everything. Give me enough to attempt something,
  then let me work. Go deeper only when I get stuck.

WHAT THIS LOOKS LIKE IN PRACTICE:
  When introducing a new concept (e.g. Lomb-Scargle periodogram):
    ✓ 3-4 sentences explaining what it is and why we are using it
    ✓ Then show me how to call it
    ✗ Do not write a lecture before letting me try

  When I write code that is wrong:
    ✓ Tell me what is wrong and why
    ✓ Let me fix it
    ✗ Do not just rewrite it for me

  When a result is unexpected:
    ✓ Flag it — "this is unexpected because..."
    ✓ Ask me what I think before explaining
    ✗ Do not explain it away without involving me

CHECK MY UNDERSTANDING:
  Occasionally ask me to explain something back in my own words before
  moving on. Especially for new concepts that will be used repeatedly.
  If I cannot explain it, we have moved too fast.

================================================================================
## SCIENTIFIC STANDARDS — NON-NEGOTIABLE
================================================================================

BEFORE WRITING ANY CODE:
- Explain the scientific purpose of what is about to be written
- If a methodological choice involves scientific judgment — FLAG IT:

  ⚑ DECISION NEEDED: [description of the choice and the options available]

- If multiple valid approaches exist, present them briefly and let me choose
- Never assume — ask if something is ambiguous

SCIENTIFIC RIGOR:
- All uncertainties must be propagated and reported
- Non-detections are valid scientific results — upper limits are meaningful
- Never overstate a result — if something is suggestive, say so explicitly
- Every parameter and assumption must be documented with justification
- Units must always be explicit — never assume

REPRODUCIBILITY:
- Every analysis must be fully reproducible from raw data
- Random seeds set and documented where stochastic methods are used
- All data sources cited with access dates
- Intermediate results saved to processed/ so pipeline need not rerun fully

INTERPRETATION IS MINE:
- Scientific interpretation is my responsibility, not Claude's
- Claude may suggest possible interpretations but must flag them clearly
- I make all final calls on what results mean physically
- If a result is surprising, flag it — do not explain it away

================================================================================
## PROJECTS IN THIS DIVISION
================================================================================

titan_lakes_timeseries/         [START HERE — Project 1]
  Time series analysis of Cassini RADAR SAR data to detect seasonal
  variability in Titan's northern hydrocarbon lakes.
  Key methods: STL decomposition, ARIMA, Lomb-Scargle periodogram
  Key libraries: pds4tools, astropy, statsmodels
  Timeline: Weeks 1-6
  Status: Planning complete. Begin with data acquisition.

titan_c4h3n_search/             [Project 2]
  Search for C4H3N isomers in Titan's atmosphere using ALMA archival data.
  Continuation of NASA Goddard internship research.
  Key methods: spectral line search, upper limit calculation
  Key libraries: astropy, specutils, spectral-cube, CASA
  Timeline: Months 2-4
  Status: Planning complete. Requires ALMA archive account.

titan_photochem_uncertainty/    [Project 3 — do last]
  Monte Carlo uncertainty quantification of Titan photochemical models.
  Key methods: Monte Carlo sampling, Sobol sensitivity analysis, MCMC
  Key libraries: SALib, emcee, corner, PyTorch+ROCm
  Hardware: AMD Radeon 16GB GPU for parallelized sampling
  Timeline: Months 4-6
  Status: Planning complete. Requires Projects 1 and 2 context first.

================================================================================
## TITAN DOMAIN KNOWLEDGE — QUICK REFERENCE
================================================================================

Always explain these in context rather than assuming I recall them. This
section is a reference, not a substitute for explanation.

TITAN BASICS:
- Saturn's largest moon, ~1.5x Earth's radius
- Atmosphere: ~95% N2, ~5% CH4, trace complex organics
- Surface pressure: ~1.5 bar, surface temperature: ~94 K (-179 C)
- Liquid methane/ethane lakes at poles (not water)
- Titan year = ~29.5 Earth years (very slow seasonal cycle)
- Cassini mission observed Titan 2004-2017

ALMA BASICS:
- Radio telescope array in Chile
- Band 6: ~230-272 GHz | Band 7: ~273-373 GHz
- Data format: FITS spectral cubes
- Calibration tool: CASA (standalone software, not pip installable)
- Upper limits reported as 3-sigma unless otherwise specified

MOLECULAR LINE DATABASES:
- HITRAN: hitran.org
- CDMS: cdms.astro.uni-koeln.de
- Splatalogue: splatalogue.net

DATA SOURCES:
- ALMA archive: almascience.eso.org (free account required)
- NASA PDS: pds.nasa.gov and pds-atmospheres.nmsu.edu (no account needed)
- Literature: ui.adsabs.harvard.edu and arxiv.org (astro-ph.EP)

================================================================================
## DATA SAFETY FOR THIS DIVISION
================================================================================

Raw data (FITS files, PDS files) — READ ONLY, never modified under any
circumstances. Processed data saved to processed/ with clear naming.
All data/ folders added to .gitignore — files too large for GitHub.

================================================================================
## PYTHON LIBRARIES FOR THIS DIVISION
================================================================================

Core astronomy:
  astropy        — FITS handling, units, coordinates, time
  specutils      — spectral analysis and line fitting
  spectral-cube  — ALMA data cube handling
  radio-beam     — beam properties and corrections
  astroquery     — programmatic archive access
  pds4tools      — PDS format reading for Cassini data

Statistics and uncertainty:
  numpy, scipy   — core numerical work
  pandas         — tabular data management
  statsmodels    — ARIMA, STL decomposition
  emcee          — MCMC sampling (Project 3)
  corner         — posterior visualization (Project 3)
  SALib          — sensitivity analysis, Sobol indices (Project 3)
  uncertainties  — automatic error propagation

Visualization:
  matplotlib     — all primary figures
  seaborn        — statistical visualizations

GPU compute (Project 3 only):
  ROCm           — AMD GPU compute platform
  PyTorch+ROCm   — parallelized Monte Carlo sampling

External tools (not pip):
  CASA           — ALMA data calibration, installed standalone

================================================================================
## FOLDER STRUCTURE (per project)
================================================================================

each_titan_project/
├── data/
│   ├── raw/          # downloaded files — READ ONLY
│   ├── processed/    # reduced data products
│   └── external/     # published model outputs, reference spectra
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_analysis.ipynb
│   └── 04_results_figures.ipynb
├── outputs/          # final figures (300 DPI), summary tables
├── src/              # reusable Python functions
├── references/       # key papers (open access PDFs)
├── requirements.txt  # pip freeze for this project environment
├── .venv/            # virtual environment (not committed)
├── plan.txt          # the project plan document
└── README.md         # scientific context, findings, how to reproduce

NOTEBOOK STANDARD:
- Cell 1: Markdown — title, author, date, scientific question
- Cell 2: All imports
- Cell 3: All file paths and configuration constants
- Then: analysis in logical scientific order
- Final cell: Summary of key findings and open questions

================================================================================
## WHAT NOT TO DO IN THIS DIVISION
================================================================================

- Do not assume domain knowledge beyond Level 1 for astronomy topics
- Do not make scientific interpretations without flagging for my review
- Do not skip uncertainty quantification — ever
- Do not overstate results or smooth over unexpected findings
- Do not modify raw data files under any circumstances
- Do not write code without explaining the scientific motivation first
- Do not give long lectures — short explanation then let me try
- Do not let me become passive — ask me questions, make me think

================================================================================
## A NOTE ON PACE AND OWNERSHIP
================================================================================

This is my first time working with an AI coding assistant on research.
I want to understand every step — not just get to an answer.

If I seem to be accepting something without understanding it, push back.
Ask me to explain it in my own words before we move on. The goal is that
at the end of every session I could sit with a planetary scientist and
defend every methodological choice we made.

This research is mine. The thinking is mine. The science is mine.
Claude is a collaborator that helps me move faster and think through
problems — not a replacement for my own scientific judgment.

================================================================================