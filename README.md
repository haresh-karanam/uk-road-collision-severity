# Road collision severity in Great Britain (2021-2023)

A small analysis project built around one question: **among police-reported injury collisions, what circumstances go with a fatal or serious outcome, and how much of that can a model recover on a year it has not seen?**

It uses SQL (SQLite), Python (pandas, scikit-learn), a leakage-aware evaluation and a Streamlit app. It is a standalone piece, different in kind from a typical web or scripting project: this one is evaluation and honest limits.

## Why this topic
The project was scoped in October 2026 against four UK data internship adverts reviewed at the time (United Utilities Data Analyst Intern, Arm Data Analytics Intern, Moloco Data Science Intern London, InPost UK Graduate Data, Analytics & AI Partner). They all asked for SQL or Python (most wanted both), most named a visualisation tool or dashboards, and the ones that mentioned modelling talked about *evaluating* it and explaining findings to non-specialists. So the project has a SQL layer with window functions, a train/test split by year, and a plain-English reading of what the numbers do and do not say.

## Data
DfT Road Safety Data (STATS19) collision, casualty and vehicle files for 2021, 2022 and 2023, downloaded on first run from data.dft.gov.uk. Published by the Department for Transport under the Open Government Licence v3.0. Page: https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data

- Covers **Great Britain**, not Northern Ireland.
- Only collisions reported to the police where someone was injured. Many minor ones are never reported.
- The CSVs are not in this repository. They are fetched on first run (about 160 MB across the collision, casualty and vehicle files) and cached in `data/`.
- Contains public sector information licensed under the Open Government Licence v3.0.
- The map step also downloads local-authority boundaries (Local Authority Districts, December 2023, UK BGC) and the 2022-to-2023 code lookup from the Office for National Statistics Open Geography Portal, also Open Government Licence v3.0.

## Run it on a Mac
Every run behind this README was done by the AI assistant, on Linux with Python 3.10: first in October 2026, then reproduced on 22 November 2026, and again on 24 November 2026 (26 tests passing, full pipeline re-run on a fresh download of the DfT files, metrics unchanged). It has not been run on a Mac, so if a step fails there, the error message is the lead to follow.

Open Terminal in this folder.

    python3 --version                  # 3.10 or newer recommended
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q                # 26 tests, offline, a few seconds
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
| `lamap.py` | Local-authority severity map: ONS boundaries, the 2022-to-2023 code lookup, and the choropleth. |
| `whorisk.py` | Who-is-involved views: fatal share by casualty age band, fatal/serious share by vehicle type. Descriptive only. |
| `tests/test_roadrisk.py` | Tests on synthetic data shaped like the DfT file. |
| `tests/test_lamap.py` | Tests for the map: code folding, dropped codes, thin-authority rule, intervals. |
| `tests/test_whorisk.py` | Tests for the who-is-involved views on synthetic casualty and vehicle frames. |
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

## Map: severity share by local authority
`run_analysis.py` also writes `outputs/ksi_by_authority.png`: every GB local authority coloured by its fatal/serious share, with Wilson intervals behind the colour and a colour scale spanning the full range of the coloured authorities (nothing clipped off the ends).

The pattern is rural against urban: the highest shares are in rural Scotland (Aberdeenshire 62.0%, Highland 55.5%, South Ayrshire 54.9%) and the lowest in city areas (Enfield 10.6%, Bristol 11.2%, Brent 11.6%). Read it with two caveats already in the limits below: there is no exposure data, so this is share among reported collisions, not danger per mile; and Scotland records severity through the CRASH injury-based system, which classifies more casualties as serious than officer judgement did, so part of the Scottish-urban gap is reporting, not roads (DfT guide: https://www.gov.uk/government/publications/guide-to-severity-adjustments-for-reported-road-casualty-statistics/guide-to-severity-adjustments-for-reported-road-casualties-great-britain).

Authorities with fewer than 200 collisions are grey, not dramatic. Collision rows from 2021-2022 that name districts abolished in April 2023 (Cumbria, North Yorkshire, Somerset) are folded into their successor authorities through the ONS 2022-to-2023 lookup; 56 rows carrying DfT's special Heathrow code have no authority to fold into and are left out.

## Who is involved
`run_analysis.py` also joins the DfT casualty and vehicle files and writes two more charts, `outputs/who_fatal_share_by_age.png` and `outputs/who_ksi_by_vehicle.png`. These are descriptive views only: they are never fed to the model, so the leakage controls above are unchanged, and the same Wilson intervals and minimum-count rules apply.

Age. Of the 80,106 people killed or seriously injured in 2021-2023, the share who died rises steeply with age. For casualties over 75 it is 13.6% (95% CI 12.7-14.6); for 11-15 year olds it is 1.9%. A serious injury is about seven times as likely to prove fatal at the oldest ages as at the youngest.

Vehicles. Of the 311,349 reported collisions, 32.3% of those involving a motorcycle were fatal or serious, against 21.6% of those involving a car. Heavy goods vehicles (25.9%) and pedal cycles (25.0%) sit between. A collision counts once per vehicle group involved in it, so a car-motorcycle crash appears in both rows. As everywhere in this project, this is share among reported collisions, not danger per mile travelled.

## Limits you should say out loud
- This is **severity given that a collision was reported**, not the risk of having one. There is no traffic or distance data, so "60 mph rural roads are worst" does not mean those roads are the most dangerous per mile. 70 mph roads show a lower share than 60 mph roads here, which is a reminder that road type and speed limit are tangled together.
- Associations only. Nothing here shows that changing a speed limit would change outcomes.
- The scenario tool is a retrospective illustration, not a forecast and not safety advice.
- Reporting practices and severity classification differ between police forces and change over time. That is not modelled here. Most forces now record severity through injury-based reporting systems (39 of 44 by the end of 2025, per DfT), which classifies more casualties as serious than officer judgement did.
- Three years only; 2020 is excluded because its file format was not checked.

## Ideas to make it yours
Add exposure data from DfT traffic counts to move from severity to rate, including a per-mile view for the vehicle types; compare police forces with a multilevel model.

## Assistance and attribution
This project was created with the help of an AI assistant (Instinct), which wrote the code, tests and this README from the brief above, and which ran and verified every result quoted here (most recently on 24 November 2026). Say that plainly if asked. Before presenting it, run it, read `roadrisk.py` until you can explain each step, change something and see what breaks. Do not claim analysis you cannot explain.

Data: DfT Road Safety Data, Open Government Licence v3.0. Code: your own to licence as you choose.
