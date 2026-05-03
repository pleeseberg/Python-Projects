# CLAUDE.md — Portfolio Projects Division
# portfolio_projects/ | Paige Leeseberg
# Last updated: April 2026

================================================================================
## WHAT THIS DIVISION IS
================================================================================

Data science and machine learning portfolio projects built to demonstrate
professional competency to both employers and research collaborators.
Every project should be polished, well-documented, and tell a clear story
from problem statement to insight.

================================================================================
## KNOWLEDGE CONTEXT FOR THIS DIVISION
================================================================================

See root CLAUDE.md for the full baseline. Key reminders for portfolio work:

- Predictive modeling is Level 1 — do not assume I remember which model
  to use when, or how to tune parameters. Walk me through the reasoning.
- Statistics is Level 1 — always explain which metric we are using and why
  it is the right one for this problem before reporting it.
- Python is Level 2 — I can follow and write code but may need help with
  library-specific syntax and more advanced patterns.

When introducing a model or method I haven't used recently:
  1. Give a short plain-language explanation of what it does
  2. Explain why it fits this problem
  3. Then help me implement it

Do not jump straight to implementation and assume I will follow along.

================================================================================
## PROJECTS IN THIS DIVISION
================================================================================

customer_segmentation/
  RFM analysis and K-Means clustering on retail transaction data.
  Goal: identify distinct customer segments and characterize them.
  Status: existing project — available for refinement.

customer_support_flow/
  Predicts customer recommendation likelihood from Capterra reviews.
  Models: Logistic Regression, Random Forest, XGBoost with SHAP.
  Status: existing project — available for refinement.

london_bike_sharing/
  Cleans bike-sharing data and exports to Excel for Tableau dashboard.
  Status: existing project — available for refinement.

twitter_sentiment/
  NLP pipeline — TF-IDF feature extraction, multi-classifier sentiment
  classification of tweets.
  Status: existing project — available for refinement.

================================================================================
## AUDIENCE AND TONE
================================================================================

PRIMARY AUDIENCES:
  1. Employers and recruiters evaluating data science competency
  2. Research collaborators evaluating statistical and analytical rigor

Every project must satisfy both:
- Clear business or research question stated upfront
- Rigorous methodology with honest discussion of limitations
- Clean readable code a non-author could follow
- Visualizations that communicate findings, not just display data
- A conclusions section that directly answers the original question

Do not produce work that looks like a tutorial or a Kaggle notebook.
These should read like professional analytical reports.

================================================================================
## TECHNICAL STANDARDS
================================================================================

MODELS AND METHODS:
- Always justify model choice before implementing — why this model?
- Always include a baseline model to compare against
- Report appropriate metrics for the problem type:
    Classification : precision, recall, F1, ROC-AUC
    Clustering     : silhouette score, inertia, segment interpretability
    Forecasting    : RMSE, MAE, MAPE, prediction intervals
- Always discuss where the model fails or has limitations
- SHAP or feature importance should be included where relevant

VISUALIZATIONS:
- Every figure needs a title, labeled axes, and units where applicable
- Use consistent color schemes within a project
- Figures should be self-explanatory without reading surrounding text
- Save to outputs/ folder

DATA:
- Raw data lives in data/raw/ — never modified
- Processed data lives in data/processed/
- Document every transformation applied to raw data
- Note data source, access date, and any licensing in the README

================================================================================
## FOLDER STRUCTURE (per project)
================================================================================

each_project/
├── data/
│   ├── raw/          # original files, never touched
│   └── processed/    # cleaned, transformed outputs
├── notebooks/        # Jupyter notebooks, numbered in order
├── outputs/          # figures, exports, final deliverables
├── src/              # reusable Python scripts if needed
├── requirements.txt  # pip freeze output
├── .venv/            # virtual environment (not committed)
└── README.md         # project overview, findings, how to run

================================================================================
## README STANDARD FOR THIS DIVISION
================================================================================

Every project README must include:
  1. Project title and one-sentence description
  2. The question being answered
  3. Data source with link and access date
  4. Methods used
  5. Key findings in 2-3 sentences
  6. How to run the notebook
  7. Libraries required

================================================================================
## WHAT NOT TO DO IN THIS DIVISION
================================================================================

- Do not build models without explaining why that model was chosen
- Do not report accuracy alone for classification problems
- Do not leave data cleaning steps undocumented
- Do not produce visualizations without titles and axis labels
- Do not skip the limitations section
- Do not assume I remember how a method works — explain before implementing

================================================================================