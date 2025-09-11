import os
import sqlite3
from typing import List, Tuple

import pandas as pd
import streamlit as st

DB_PATH = os.environ.get("SOLVERDASH_DB", os.path.abspath("solverdash.db"))

@st.cache_data(show_spinner=False)
def load_df(query: str, params: Tuple=()):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, con, params=params)
    con.close()
    return df

st.set_page_config(page_title="SolverDash", layout="wide")
st.title("SolverDash — local runs")

# Sidebar: choose project & run
projects = load_df("SELECT DISTINCT project FROM runs ORDER BY project")["project"].tolist()
project = st.sidebar.selectbox("Project", ["(all)"] + projects)
if project == "(all)":
    runs = load_df("SELECT run_id, name, project, created_at, status FROM runs ORDER BY created_at DESC")
else:
    runs = load_df("SELECT run_id, name, project, created_at, status FROM runs WHERE project=? ORDER BY created_at DESC", (project,))

st.sidebar.write(f"{len(runs)} runs")
run_row = st.sidebar.dataframe(runs, use_container_width=True, hide_index=True)
run_id = st.sidebar.selectbox("Run ID", runs["run_id"].tolist())

# Header + config
run = load_df("SELECT * FROM runs WHERE run_id=?", (run_id,)).iloc[0]
cols = st.columns(4)
cols[0].metric("Status", run["status"])
cols[1].metric("Started", run["created_at"])
cols[2].metric("Finished", run["finished_at"] or "—")
cols[3].metric("Host", run["host"])
st.caption(f"Name: **{run['name']}** · Project: **{run['project']}** · Python: {run['py_version']} · OS: {run['os']}")
if run["notes"]:
    st.info(run["notes"])
with st.expander("Config"):
    st.json(run["config_json"] or "{}")

# Metrics
m = load_df("SELECT step, ts, key, value FROM metrics WHERE run_id=? ORDER BY step", (run_id,))
pivot = m.pivot_table(index="step", columns="key", values="value", aggfunc="last").reset_index().sort_values("step")
if not pivot.empty:
    k_left = st.selectbox("Left y-axis metric", [c for c in pivot.columns if c != "step"], index=([c for c in pivot.columns if c != "step"].index("f") if "f" in pivot.columns else 0))
    k_right = st.selectbox("Right y-axis metric (optional)", ["(none)"] + [c for c in pivot.columns if c not in ("step", k_left)])
    st.line_chart(pivot.set_index("step")[[k_left]])
    if k_right != "(none)":
        st.line_chart(pivot.set_index("step")[[k_right]])

    with st.expander("Raw metrics"):
        st.dataframe(pivot, use_container_width=True)
else:
    st.warning("No metrics logged yet for this run.")

# Events
st.subheader("Events")
ev = load_df("SELECT ts, level, message FROM events WHERE run_id=? ORDER BY ts", (run_id,))
if not ev.empty:
    st.dataframe(ev, use_container_width=True, hide_index=True, height=220)
else:
    st.caption("No events.")

# Timings
st.subheader("Timings (aggregated)")
tm = load_df("SELECT key, seconds, count FROM timings WHERE run_id=? ORDER BY key", (run_id,))
if not tm.empty:
    tm["avg_sec"] = tm["seconds"] / tm["count"].clip(lower=1)
    st.dataframe(tm, use_container_width=True, hide_index=True)
else:
    st.caption("No timings.")

# Artifacts
st.subheader("Artifacts")
art = load_df("SELECT name, type, path, size_bytes, sha256, created_at FROM artifacts WHERE run_id=? ORDER BY created_at DESC", (run_id,))
if not art.empty:
    st.dataframe(art, use_container_width=True, hide_index=True)
else:
    st.caption("No artifacts.")
