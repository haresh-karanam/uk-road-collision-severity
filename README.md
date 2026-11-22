# Road collision severity in Great Britain (2021-2023)

A small analysis project built around one question: **among police-reported injury collisions, what circumstances go with a fatal or serious outcome, and how much of that can a model recover on a year it has not seen?**

It uses SQL (SQLite), Python (pandas, scikit-learn), a leakage-aware evaluation and a Streamlit app. It is a standalone piece, different in kind from a typical web or scripting project: this one is evaluation and honest limits.

## Why this topic
The project was scoped in October 2026 against four UK data internship adverts reviewed at the time (United Utilities Data Analyst Intern, Arm Data Analytics Intern, Moloco Data Science Intern London, InPost UK Graduate Data, Analytics & AI Partner). They all asked for SQL or Python (most wanted both), most named a visualisation tool or dashboards, and the ones that mentioned modelling talked about *evaluating* it and explaining findings to non-specialists. So the project has a SQL layer with window functions, a train/test split by year, and a plain-English reading of what the numbers do and do not say.

## Data
DfT Road Safety Data (STATS19) collision files for 2021, 2022 and 2023, downloaded on first run from data.dft.gov.uk. Published by the Department for Transport under the Open Government Licence v3.0. Page: https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data

- Covers **Great Britain**, not Northern Ireland.
- Only collisions reported to the police where someone was injured. Many minor ones are never reported.
- The CSVs are not in this zip. They are fetched on first run (about 60 MB) and cached in `data/`.
- Contains public sector information licensed under the Open Government Licence v3.0.

## Run it on a Mac
Every run behind this README was done by the AI assistant, on Linux with Python 3.10: first in October 2026, then reproduced on 22 November 2026 (12 tests passing, full pipeline re-run on a fresh download of the DfT files, metrics unchanged). It has not been run on a Mac, so if a step fails there, the error message is the lead to follow.

Open Terminal in this folder.

    python3 --version                  # 3.10 or newer recommended
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q                # 12 tests, offline, a few seconds
    python run_analysis.py             # downloads data, prints metrics, writes charts to outputs/
    streamlit run app.py               # opens the app in your browser

The first `run_analysis.py` or app start needs internet.

## What is in the folder
| File | What it does |
|---|---|
| `roadrisk.py` | Download, cleaning, features, Wilson intervals, models, evaluation. Everything else imports this. |
| `sql/queries.sql`, `sqlstore.py` | Four analytical SQL queries (GROUP BY/HAVING, CTE, LAG, RANK, conditional AVG) run on an in-memory SQLite copy. |
| `run_analysis.py` | Command-line run, writes `outputs/metrics.json` and charts. |
| `app.py` | Streamlit app: explore, SQL results, model check, scenario tool. |
| `tests/test_roadrisk.py` | Tests on synthetic data shaped like the DfT file. |
| `outputs/` | Charts and metrics from the most recent run, so you can see results before running anything. |

## Design decisions (good interview material)
- **Target.** Fatal or serious = 1, slight = 0 (about 23% of collisions).
- **Leakage.** The feature list leaves out `number_of_casualties`, police attendance and the DfT adjusted-severity columns. They are recorded after, or computed from, the outcome. A test checks they are not in the feature list, and another checks the model scores near 0.5 AUC on pure noise.
- **Split by time.** Train on 2021-2022, test on 2023. A random split would let the model see the same period it is tested on.
- **Baselines.** "Always predict the average", then logistic regression, then gradient boosting. The simpler model is the thing to beat.
- **Metrics.** ROC AUC, PR AUC (the classes are imbalanced), Brier score for probability quality, top-decile lift, and a calibration chart.
- **Intervals.** Wilson 95% intervals on every rate, and a minimum group size, so small groups do not look dramatic.

## Results
From the assistant's run of 22 November 2026, on a fresh download of the DfT files. Test year 2023, 104,258 collisions, 23.9% fatal or serious.

| Model | ROC AUC | PR AUC | Brier | Top-decile lift |
|---|---|---|---|---|
| Predict the average | 0.500 | 0.239 | 0.182 | n/a |
| Logistic regression | 0.615 | 0.321 | 0.177 | 1.58 |
| Gradient boosting | 0.641 | 0.344 | 0.174 | 1.71 |

That is a modest gain, and the useful finding is how modest. The recorded circumstances carry some signal about severity, but most of what decides severity (impact speed, vehicle types, who was involved, their age) is not in these columns.

Some SQL results: 60 mph rural roads have the highest fatal/serious share (34.2%), 20 mph urban the lowest (17.7%). Darkness raises the share at every speed limit checked. Hours 0-5 have the highest share (about 29-31%) with far fewer collisions.

## Limits you should say out loud
- This is **severity given that a collision was reported**, not the risk of having one. There is no traffic or distance data, so "60 mph rural roads are worst" does not mean those roads are the most dangerous per mile. 70 mph roads show a lower share than 60 mph roads here, which is a reminder that road type and speed limit are tangled together.
- Associations only. Nothing here shows that changing a speed limit would change outcomes.
- The scenario tool is a retrospective illustration, not a forecast and not safety advice.
- Reporting practices and severity classification differ between police forces and change over time. That is not modelled here.
- Three years only; 2020 is excluded because its file format was not checked.

## Ideas to make it yours
Add a map by local authority; add the vehicle and casualty files to study who is involved (then re-check leakage); compare police forces with a multilevel model; add exposure data from DfT traffic counts to move from severity to rate.

## Assistance and attribution
This project was created with the help of an AI assistant (Instinct), which wrote the code, tests and this README from the brief above, and which ran and verified every result quoted here (most recently on 22 November 2026). Say that plainly if asked. Before presenting it, run it, read `roadrisk.py` until you can explain each step, change something and see what breaks. Do not claim analysis you cannot explain.

Data: DfT Road Safety Data, Open Government Licence v3.0. Code: your own to licence as you choose.
