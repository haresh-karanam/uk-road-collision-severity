"""Streamlit app. Run:  streamlit run app.py"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import roadrisk as rr
import sqlstore

st.set_page_config(page_title="GB road collision severity", layout="wide")


@st.cache_data(show_spinner="Downloading DfT data on first run (about 60 MB)...")
def get_data():
    return rr.load()


@st.cache_resource(show_spinner="Fitting models on 2021-2022...")
def get_models(d):
    train, test = rr.split_by_year(d)
    return rr.evaluate(train, test), test


d = get_data()
res, test = get_models(d)

st.title("Road collision severity in Great Britain, 2021-2023")
st.caption("Source: DfT Road Safety Data (STATS19), police-reported injury collisions in Great Britain, "
           "Open Government Licence v3.0. Great Britain only, not Northern Ireland.")
t1, t2, t3, t4 = st.tabs(["Explore", "SQL queries", "Model check", "Scenario tool"])

with t1:
    years = st.multiselect("Years", sorted(d["collision_year"].unique()), default=sorted(d["collision_year"].unique()))
    f = d[d["collision_year"].isin(years)]
    if f.empty:
        st.warning("Pick at least one year.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    c1.metric("Collisions", f"{len(f):,}")
    c2.metric("Fatal or serious", f"{f['ksi'].mean():.1%}")
    c3.metric("Median hour", int(f["hour"].median()))
    col = st.selectbox("Break down by", ["speed_limit", "hour", "light_conditions", "weather_conditions",
                                           "road_surface_conditions", "urban_or_rural_area", "day_of_week"])
    g = rr.ksi_rate_by(f, col, min_n=100)
    if len(g) == 0:
        st.warning("No group has enough collisions for a stable rate.")
    else:
        names = [rr.label(col, v) if col in rr.LABELS else str(v) for v in g[col]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, g["rate"] * 100, yerr=[(g["rate"] - g["low"]) * 100, (g["high"] - g["rate"]) * 100], capsize=3)
        ax.set_ylabel("% of collisions fatal/serious"); plt.xticks(rotation=40, ha="right"); fig.tight_layout()
        st.pyplot(fig)
    st.info("This is the share of severe outcomes among collisions that were reported. It says nothing about "
            "how risky a journey is: we have no data on how much traffic each road type or hour carries.")

with t2:
    con = sqlstore.connect(d)
    for name in sqlstore.queries():
        st.subheader(name.replace("_", " ").capitalize())
        with st.expander("SQL"):
            st.code(sqlstore.queries()[name], language="sql")
        st.dataframe(sqlstore.run(con, name), use_container_width=True)

with t3:
    st.write(f"Trained on {res['n_train']:,} collisions (2021-2022), tested on {res['n_test']:,} from 2023. "
             f"Fatal/serious share: {res['base_rate_train']:.1%} train, {res['base_rate_test']:.1%} test.")
    rows = [("Always predict the average", res["baseline"]["roc_auc"], res["baseline"]["pr_auc"], res["baseline"]["brier"], None)]
    for k, n in (("logreg", "Logistic regression"), ("hgb", "Gradient boosting")):
        rows.append((n, res[k]["roc_auc"], res[k]["pr_auc"], res[k]["brier"], res[k]["top_decile_lift"]))
    st.dataframe(pd.DataFrame(rows, columns=["Model", "ROC AUC", "PR AUC", "Brier", "Top-decile lift"]).round(3))
    st.write("Reading it: the models rank severity only modestly better than chance (AUC about 0.6 to 0.65). "
             "The top 10% of collisions by predicted score contain roughly 1.6 to 1.7 times the average share of "
             "fatal/serious ones. Circumstances recorded in the report explain a little of severity; most of it "
             "(speed of impact, vehicle types, who was involved, age) is not in these columns.")
    cal = rr.calibration_table(res["models"]["hgb"], test)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(cal["predicted"], cal["observed"], marker="o"); mx = max(cal["predicted"].max(), cal["observed"].max())
    ax.plot([0, mx], [0, mx], "--", color="grey"); ax.set_xlabel("Mean predicted"); ax.set_ylabel("Observed in 2023")
    ax.set_title("Calibration by decile (gradient boosting)"); fig.tight_layout(); st.pyplot(fig)

with t4:
    st.warning("Retrospective illustration only. This estimates how severe a collision with these recorded "
               "circumstances has tended to be, given that one was reported. It is not a forecast of whether a "
               "crash will happen and not safety advice.")
    a, b, c = st.columns(3)
    speed = a.selectbox("Speed limit", [20, 30, 40, 50, 60, 70], index=1)
    area = b.selectbox("Area", [1, 2], format_func=lambda v: rr.label("urban_or_rural_area", v))
    light = c.selectbox("Light", [1, 4, 5, 6], format_func=lambda v: rr.label("light_conditions", v))
    hour = a.slider("Hour", 0, 23, 17)
    wx = b.selectbox("Weather", [1, 2, 7], format_func=lambda v: rr.label("weather_conditions", v))
    surf = c.selectbox("Surface", [1, 2, 4], format_func=lambda v: rr.label("road_surface_conditions", v))
    p = res["models"]["hgb"].predict_proba(rr.scenario_frame(speed_limit=speed, urban_or_rural_area=area,
        light_conditions=light, hour=hour, weather_conditions=wx, road_surface_conditions=surf))[0, 1]
    st.metric("Estimated chance a reported collision like this is fatal/serious", f"{p:.0%}",
              delta=f"{(p - res['base_rate_test']) * 100:+.1f} percentage points vs the 2023 average ({res['base_rate_test']:.1%})",
              delta_color="off")
    st.caption("Everything not shown above is held fixed: 2 vehicles, June, a Wednesday, single carriageway, "
               "A road, not at a junction. To vary them, edit "
               "scenario_frame in roadrisk.py. Nothing about the vehicles, drivers or ages is available to it.")
