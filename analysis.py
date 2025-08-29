# app.py — L&D Training Insights Dashboard (No Upload Version)
# Run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Data source:
#   Place your Excel in the SAME folder and set DATA_PATH below (default: "sample.xlsx").
#   Required columns: ACTIVITY_NAME, COURSE_START_DATE, Duration
#
# What this app includes:
# - Smart grouping of similar activity names (TF‑IDF + Agglomerative; difflib fallback)
# - KPI cards, full Insights section (11 items), and interactive charts (Plotly)
# - Filters (date range, activities, duration) — no file upload needed
# - Sample mapping table and a download of the cleaned/grouped data

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from io import BytesIO
from pathlib import Path

st.set_page_config(
    page_title="L&D Training Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Config ----------
DATA_PATH = "sample.xlsx"  # ← put your file name here

# ---------- Aesthetic tweaks ----------
CUSTOM_CSS = """
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
.big-metric { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
.subtle { color: var(--text-color); opacity: 0.7; }
.card { background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.16); border-radius: 16px; padding: 1rem 1.2rem; }
.insights-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.insight-card { background: rgba(127,127,127,0.08); border: 1px solid rgba(127,127,127,0.18); border-radius: 14px; padding: 14px 16px; }
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

st.title("📊 L&D Training Insights")
st.caption("Smart grouping + interactive analytics — no upload required")

# ---------- Utilities ----------
def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    import re
    s = re.sub(r"[^a-z0-9\s/+-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    repl = {
        "&": "and", " w/ ": " with ", " w ": " with ",
        "certificate": "cert", "certification": "cert",
        "introduction": "intro", "advanced": "adv", "intermediate": "inter",
        "management": "mgmt", "manager": "mgr",
        "workshop": "ws", "training": "train", "course": "crs"
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def group_activity_names(names: pd.Series, threshold: float = 0.35) -> dict:
    """Return map raw_name -> canonical_name by clustering normalized names.
    Uses sklearn if available, else a difflib fallback.
    threshold is a distance threshold on (1 - cosine_similarity); lower = stricter grouping.
    """
    norm = names.fillna("").astype(str).map(normalize_text)
    unique = norm.drop_duplicates().tolist()

    # Frequencies on normalized forms
    norm_freq = norm.value_counts().to_dict()

    representative = {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_distances
        from sklearn.cluster import AgglomerativeClustering

        X = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(unique)
        D = cosine_distances(X)
        clustering = AgglomerativeClustering(
            n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=threshold
        )
        labels = clustering.fit_predict(D)

        clusters = defaultdict(list)
        for n, lab in zip(unique, labels):
            clusters[lab].append(n)

        for lab, names_ in clusters.items():
            rep = max(names_, key=lambda n: (norm_freq.get(n, 0), -len(n)))
            for n in names_:
                representative[n] = rep

    except Exception:
        import difflib
        groups = []
        used = set()
        for n in unique:
            if n in used:
                continue
            group = [n]
            for m in unique:
                if m in used or m == n:
                    continue
                if difflib.SequenceMatcher(None, n, m).ratio() >= 0.75:
                    group.append(m)
            used.update(group)
            groups.append(group)
        for names_ in groups:
            rep = max(names_, key=lambda n: (norm_freq.get(n, 0), -len(n)))
            for n in names_:
                representative[n] = rep

    # Map canonical normalized -> best original spelling
    df_tmp = pd.DataFrame({"raw": names, "norm": norm})
    rep_to_best_original = {}
    for rep in set(representative.values()):
        mask = df_tmp["norm"].map(lambda s: representative.get(s, s) == rep)
        orig_counts = df_tmp.loc[mask, "raw"].value_counts()
        rep_to_best_original[rep] = orig_counts.index[0]

    raw_to_canon = {}
    for raw, nrm in zip(names, norm):
        rep = representative.get(nrm, nrm)
        raw_to_canon[raw] = rep_to_best_original.get(rep, raw)

    return raw_to_canon


@st.cache_data(show_spinner=False)
def load_data_local(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"Data file not found: {path}. Place your Excel in the app folder and set DATA_PATH.")
        return pd.DataFrame(columns=["ACTIVITY_NAME", "COURSE_START_DATE", "Duration"])
    df = pd.read_excel(p)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if "COURSE_START_DATE" in df.columns:
        df["COURSE_START_DATE"] = pd.to_datetime(df["COURSE_START_DATE"], errors="coerce")
    if "Duration" in df.columns and not np.issubdtype(df["Duration"].dtype, np.number):
        df["Duration"] = pd.to_numeric(df["Duration"].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
    return df


# ---------- Data load (no upload) ----------
df = load_data_local(DATA_PATH)
required = ["ACTIVITY_NAME", "COURSE_START_DATE", "Duration"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}. Please provide an Excel with these columns.")
    st.stop()

# ---------- Group activity names ----------
with st.spinner("Grouping similar activity names…"):
    mapping = group_activity_names(df["ACTIVITY_NAME"])  # adjust threshold inside function if needed

df["ACTIVITY_CANON"] = df["ACTIVITY_NAME"].map(mapping)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
min_date, max_date = df["COURSE_START_DATE"].min(), df["COURSE_START_DATE"].max()
if pd.notna(min_date) and pd.notna(max_date):
    date_sel = st.sidebar.date_input("Date range", (min_date.date(), max_date.date()))
    if isinstance(date_sel, (list, tuple)) and len(date_sel) == 2:
        start, end = date_sel
    else:
        start, end = min_date.date(), max_date.date()
    df = df[(df["COURSE_START_DATE"] >= pd.to_datetime(start)) & (df["COURSE_START_DATE"] <= pd.to_datetime(end) + pd.Timedelta(days=1))]

acts = sorted(df["ACTIVITY_CANON"].dropna().unique().tolist())
default_n = min(10, len(acts))
sel_acts = st.sidebar.multiselect("Activities", acts, default=acts[:default_n] if default_n > 0 else [])
if sel_acts:
    df = df[df["ACTIVITY_CANON"].isin(sel_acts)]

# Duration slider handling
st.sidebar.write("Example If you slide it to 0.0 → 2.0, you’ll only see short training sessions under 2 hours in the KPIs, Insights, and plots.If you move it to 5.0 → 20.0, you’ll only see longer workshops/courses")   
if df["Duration"].notna().any():
    dmin_val = np.nanmin(df["Duration"].values)
    dmax_val = np.nanmax(df["Duration"].values)
    dmin = float(dmin_val) if np.isfinite(dmin_val) else 0.0
    dmax = float(dmax_val) if np.isfinite(dmax_val) else 1.0
    dmax = max(dmax, 1.0)
else:
    dmin, dmax = 0.0, 1.0

dur_range = st.sidebar.slider("Duration (hours)", min_value=0.0, max_value=float(dmax), value=(0.0, float(dmax)), step=0.5)
df = df[(df["Duration"].fillna(0) >= dur_range[0]) & (df["Duration"].fillna(0) <= dur_range[1])]

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="big-metric">{len(df):,}</div><div class="subtle">Sessions</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="big-metric">{df["ACTIVITY_CANON"].nunique():,}</div><div class="subtle">Activities (grouped)</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    total_hours = float(df["Duration"].sum(skipna=True))
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="big-metric">{total_hours:.1f}</div><div class="subtle">Total Hours</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with c4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if pd.notna(df["COURSE_START_DATE"].min()):
        date_span = f'{df["COURSE_START_DATE"].min().date()} → {df["COURSE_START_DATE"].max().date()}'
    else:
        date_span = "—"
    st.markdown(f'<div class="big-metric">🗓️</div><div class="subtle">{date_span}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Compute Insights ----------
# 1) Consolidation
n_raw = df["ACTIVITY_NAME"].nunique()
n_canon = df["ACTIVITY_CANON"].nunique()
cons_text = f"Consolidated <b>{n_raw}</b> raw activity names into <b>{n_canon}</b> canonical groups (−{n_raw - n_canon} duplicates)."

# 2) Date coverage & busiest months
date_min = pd.to_datetime(df["COURSE_START_DATE"]).min()
date_max = pd.to_datetime(df["COURSE_START_DATE"]).max()
df_m = df.copy()
df_m["Month"] = df_m["COURSE_START_DATE"].dt.to_period("M").astype(str)
monthly_counts = df_m.groupby("Month").size().sort_values(ascending=False)
busy_months = monthly_counts.head(3).index.tolist()
cover_text = f"Data covers <b>{date_min.date() if pd.notna(date_min) else '—'}</b> to <b>{date_max.date() if pd.notna(date_max) else '—'}</b>; busiest month(s): <b>{busy_months}</b>."

# 3) Duration stats
valid_dur = df["Duration"].dropna()
mean_h = float(valid_dur.mean()) if len(valid_dur) else 0.0
median_h = float(valid_dur.median()) if len(valid_dur) else 0.0
p90_h = float(valid_dur.quantile(0.90)) if len(valid_dur) else 0.0
dur_text = f"Average session duration is <b>{mean_h:.2f} h</b> (median <b>{median_h:.2f} h</b>, 90th pct <b>{p90_h:.2f} h</b>)."

# 4) Top activities by count
vc = df["ACTIVITY_CANON"].value_counts()
most_text = "Most-run activities: " + ", ".join([f"<b>{a}</b> ({c})" for a, c in vc.head(3).items()]) + " (by count)."

# 5) Longest avg duration activity
avg_dur_by_act = df.dropna(subset=["Duration"]).groupby("ACTIVITY_CANON")["Duration"].mean().sort_values(ascending=False)
long_text = ""
if not avg_dur_by_act.empty:
    long_text = f"Longest average-duration activity: <b>{avg_dur_by_act.index[0]}</b> (≈ {avg_dur_by_act.iloc[0]:.2f} h)."

# 6) Weekday skew
wk_counts = df["COURSE_START_DATE"].dt.day_name().value_counts()
weekday_text = ""
if not wk_counts.empty:
    weekday_text = f"Scheduling skews to <b>{wk_counts.idxmax()}</b>; lightest day: <b>{wk_counts.idxmin()}</b>."

# 7) Seasonality (quarter share)
q_counts = df["COURSE_START_DATE"].dt.to_period("Q").astype(str).value_counts()
season_text = ""
if not q_counts.empty:
    q_share = (q_counts / q_counts.sum() * 100).round(1)
    top_q = q_share.sort_values(ascending=False).head(1)
    season_text = f"Seasonality: <b>{top_q.index[0]}</b> accounts for <b>{float(top_q.iloc[0])}%</b> of sessions."

# 8) Duration bucket majority

def bucket_duration(x):
    if pd.isna(x):
        return "Unknown"
    if x < 1:
        return "< 1 hr"
    if x < 2:
        return "1–2 hrs"
    if x < 4:
        return "2–4 hrs"
    if x < 8:
        return "4–8 hrs"
    return "8+ hrs"

if "Duration" in df.columns:
    df["Duration Bucket"] = df["Duration"].map(bucket_duration)
    buck = df["Duration Bucket"].value_counts()
    major_text = ""
    if not buck.empty:
        major_text = f"Majority of sessions fall in <b>{buck.idxmax()}</b> bucket."
else:
    major_text = ""

# 9) Repeat cadence (most frequent among top activities)
rep_text = ""
cadences = {}
for act in df["ACTIVITY_CANON"].value_counts().head(5).index:
    dates = df.loc[df["ACTIVITY_CANON"] == act, "COURSE_START_DATE"].dropna().sort_values()
    if len(dates) >= 2:
        gaps = dates.diff().dropna().dt.days.values
        if len(gaps) > 0:
            cadences[act] = float(np.mean(gaps))
if cadences:
    # Smallest average gap = most frequent
    most_freq_act, gap_days = min(cadences.items(), key=lambda x: x[1])
    rep_text = f"Repeat cadence: <b>{most_freq_act}</b> recurs about every <b>{gap_days:.1f} days</b> on average."

# 10) Total training volume
vol_text = f"Total training volume logged: <b>{float(df['Duration'].sum(skipna=True)):.2f} hours</b> across <b>{len(df)}</b> sessions."

# 11) YoY trend (if >= 2 years present)
trend_text = ""
if df["COURSE_START_DATE"].dt.year.nunique() >= 2:
    yearly = df.groupby(df["COURSE_START_DATE"].dt.year).size().sort_index()
    yoy = yearly.pct_change().dropna()
    if not yoy.empty:
        last_year = int(yoy.index[-1])
        rate = yoy.iloc[-1] * 100
        trend_text = f"Year-over-year change to <b>{last_year}</b>: <b>{rate:.1f}%</b> sessions."

# ---------- Insights section ----------
st.subheader("Insights")
st.markdown('<div class="insights-grid">', unsafe_allow_html=True)
for text in [cons_text, cover_text, dur_text, most_text, long_text, weekday_text, season_text, major_text, rep_text, vol_text, trend_text]:
    if text:
        st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- Interactive Charts ----------
import plotly.express as px

# Monthly trend
if df["COURSE_START_DATE"].notna().any():
    tmp = df.copy()
    tmp["Month"] = tmp["COURSE_START_DATE"].dt.to_period("M").astype(str)
    monthly = tmp.groupby("Month").size().reset_index(name="Sessions")
    fig1 = px.bar(monthly, x="Month", y="Sessions", title="Sessions per Month")
    st.plotly_chart(fig1, use_container_width=True)

# Top activities by count
max_top = max(5, min(20, df["ACTIVITY_CANON"].nunique()))
topn = st.slider("Top N activities", 5, max_top, min(10, max_top))
top_cnt = df["ACTIVITY_CANON"].value_counts().head(topn).reset_index()
top_cnt.columns = ["Activity", "Sessions"]
fig2 = px.bar(top_cnt, x="Activity", y="Sessions", title=f"Top {topn} Activities (by sessions)")
fig2.update_layout(xaxis_tickangle=-30)
st.plotly_chart(fig2, use_container_width=True)

# Duration distribution + avg by activity
if df["Duration"].notna().any():
    fig3 = px.histogram(df, x="Duration", nbins=30, title="Duration distribution (hours)")
    st.plotly_chart(fig3, use_container_width=True)

    avg_by_act = (
        df.dropna(subset=["Duration"]).groupby("ACTIVITY_CANON")["Duration"].mean()
        .sort_values(ascending=False).head(topn).reset_index()
    )
    fig4 = px.bar(avg_by_act, x="ACTIVITY_CANON", y="Duration", title=f"Longest Average Duration by Activity (Top {topn})")
    fig4.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig4, use_container_width=True)

# Weekly pattern
if df["COURSE_START_DATE"].notna().any():
    tmp2 = df.copy()
    tmp2["Weekday"] = tmp2["COURSE_START_DATE"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wk_df = tmp2["Weekday"].value_counts().reindex(order).fillna(0).reset_index()
    wk_df.columns = ["Weekday", "Sessions"]
    fig5 = px.bar(wk_df, x="Weekday", y="Sessions", title="Sessions by Weekday")
    st.plotly_chart(fig5, use_container_width=True)

# ---------- Mapping table & download ----------
st.subheader("Activity Name Grouping (samples)")
sample_map = (
    df[["ACTIVITY_NAME", "ACTIVITY_CANON"]]
    .drop_duplicates()
    .sort_values("ACTIVITY_CANON")
    .head(50)
)
st.dataframe(sample_map, use_container_width=True, hide_index=True)

# Download cleaned / grouped data
out = df.copy()
to_write = BytesIO()
with pd.ExcelWriter(to_write, engine='xlsxwriter') as writer:
    out.to_excel(writer, index=False, sheet_name="cleaned")
to_write.seek(0)
st.download_button(
    "⬇️ Download cleaned & grouped Excel",
    data=to_write,
    file_name="cleaned_grouped.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("Built with ❤️ for L&D teams.")
