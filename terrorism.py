from __future__ import annotations
import os, glob, re
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Optional plotting lib — install if missing: pip install plotly-express
try:
    import plotly.express as px
except Exception:
    px = None

# ---------------------------- App Config ---------------------------- #
st.set_page_config(
    page_title="Global Terrorism Statistic",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join("data", "terrorism")


# Subtle, professional styling
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
        .stPlotlyChart { border-radius: 14px; box-shadow: 0 2px 8px rgba(0,0,0,.06); }
        .stDataFrame, .dataframe { border: 1px solid #e8e8e8; border-radius: 10px; }
        .metric-small .stMetric { padding: .25rem .5rem; }
        h1, h2, h3 { margin-bottom: .5rem; }
        .css-1v0mbdj { padding-top: 0 !important; } /* tighten top gap if present */
        div[data-baseweb="tab"] { font-weight: 600; } /* bolder tab labels */
        section[data-testid="stSidebar"] .stMarkdown h3 { margin-top: .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------- Helpers ---------------------------- #
ALIASES = {
    # canonical -> possible input headers (case-insensitive)
    "Country":   ["entity", "country", "country name", "name"],
    "ISO_Code":  ["code", "iso_code", "iso code", "iso3", "alpha-3"],
    "Year":      ["year", "iyear"],
}

def _trim_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df

def _normalize_headers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Normalize to Country / ISO_Code / Year and keep other columns unchanged."""
    df = _trim_headers(df)
    lower_to_actual = {c.lower(): c for c in df.columns}
    rename_plan: Dict[str, str] = {}

    for canonical, candidates in ALIASES.items():
        for cand in candidates:
            actual = lower_to_actual.get(cand.lower())
            if actual is not None:
                rename_plan[actual] = canonical
                break

    if rename_plan:
        df = df.rename(columns=rename_plan)

    # Standardize a few dtypes
    if "ISO_Code" in df.columns:
        df["ISO_Code"] = df["ISO_Code"].astype(str).str.strip().str.upper()
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    return df, rename_plan

def _scan_files() -> List[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    found = []
    for pat in ("*.csv", "*.parquet", "*.xlsx", "*.xls"):
        found += glob.glob(os.path.join(DATA_DIR, pat))
    return sorted(found)

def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)

def _detect_metric_column(df: pd.DataFrame, kind: str) -> Optional[str]:
    """
    Heuristic detection of the metric column in one file.
    kind ∈ {"attacks","deaths","injuries"}.
    """
    cand_map = {
        "attacks":  [r"^terrorist\s*attacks$", r"attacks?", r"incidents?"],
        "deaths":   [r"^terrorism\s*deaths$", r"deaths?", r"fatalit"],
        "injuries": [r"^injuries\s*from\s*terrorist\s*attacks$", r"injur"],
    }
    cols = list(df.columns)
    for pattern in cand_map[kind]:
        regex = re.compile(pattern, flags=re.I)
        for c in cols:
            if regex.search(str(c)):
                return c
    return None

def _collect_datasets() -> Dict[str, pd.DataFrame]:
    """Load every file, normalize headers, and return a dict by filename stem."""
    data: Dict[str, pd.DataFrame] = {}
    for p in _scan_files():
        try:
            df = _read_any(p)
            df, _ = _normalize_headers(df)
            df["__source_file"] = os.path.basename(p)
            data[os.path.splitext(os.path.basename(p))[0]] = df
        except Exception as e:
            st.warning(f"Skipping {os.path.basename(p)}: {e}")
    return data

def _build_panel(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the Country×Year panel by finding three metrics across the loaded files:
    - Terrorist attacks
    - Terrorism deaths
    - Injuries from terrorist attacks
    Returns panel only (no provenance UI).
    """
    attacks_df = None
    deaths_df = None
    injuries_df = None

    # filename preference
    for name, df in datasets.items():
        lname = name.lower()
        if attacks_df is None and ("terrorist-attacks" in lname or "attacks" in lname):
            col = _detect_metric_column(df, "attacks")
            if col and {"Country", "ISO_Code", "Year"}.issubset(df.columns):
                attacks_df = df[["Country","ISO_Code","Year", col]].rename(columns={col: "Terrorist attacks"})
        if deaths_df is None and ("terrorism-deaths" in lname or "deaths" in lname):
            col = _detect_metric_column(df, "deaths")
            if col and {"Country", "ISO_Code", "Year"}.issubset(df.columns):
                deaths_df = df[["Country","ISO_Code","Year", col]].rename(columns={col: "Terrorism deaths"})
        if injuries_df is None and ("injuries-from-terrorist-attacks" in lname or "injur" in lname):
            col = _detect_metric_column(df, "injuries")
            if col and {"Country", "ISO_Code", "Year"}.issubset(df.columns):
                injuries_df = df[["Country","ISO_Code","Year", col]].rename(columns={col: "Injuries from terrorist attacks"})

    # if still missing, scan all files for the metric column
    def scan_any(kind: str) -> Optional[pd.DataFrame]:
        for df in datasets.values():
            col = _detect_metric_column(df, kind)
            if col and {"Country", "ISO_Code", "Year"}.issubset(df.columns):
                out_name = {
                    "attacks":  "Terrorist attacks",
                    "deaths":   "Terrorism deaths",
                    "injuries": "Injuries from terrorist attacks",
                }[kind]
                return df[["Country","ISO_Code","Year", col]].rename(columns={col: out_name})
        return None

    if attacks_df is None:  attacks_df  = scan_any("attacks")
    if deaths_df is None:   deaths_df   = scan_any("deaths")
    if injuries_df is None: injuries_df = scan_any("injuries")

    # Start merging panel (outer joins to preserve data)
    pieces = [x for x in [attacks_df, deaths_df, injuries_df] if x is not None]
    if not pieces:
        return pd.DataFrame()

    panel = pieces[0].copy()
    for part in pieces[1:]:
        panel = panel.merge(part, on=["Country","ISO_Code","Year"], how="outer")

    # numeric coercion
    for c in ["Terrorist attacks", "Terrorism deaths", "Injuries from terrorist attacks"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    panel = panel.sort_values(["Country","Year"]).reset_index(drop=True)
    return panel

# ---------------------------- Data Loader ---------------------------- #
@st.cache_data(show_spinner=True)
def load_panel() -> pd.DataFrame:
    files = _scan_files()
    if not files:
        return pd.DataFrame()
    datasets = _collect_datasets()
    panel = _build_panel(datasets)
    return panel

# ---------------------------- UI: Sidebar Filters ---------------------------- #
def sidebar_filters(panel: pd.DataFrame) -> Dict:
    st.sidebar.header("Filters")

    if panel.empty:
        return {"years": (1970, 2025), "countries": set()}

    # Year slider
    years_avail = panel["Year"].dropna().astype(int).unique().tolist()
    if years_avail:
        ymin, ymax = int(min(years_avail)), int(max(years_avail))
    else:
        ymin, ymax = 1970, 2025

    years_sel = st.sidebar.slider("Year range", min_value=ymin, max_value=ymax, value=(ymin, ymax), step=1)

    # Country multiselect
    countries = sorted(panel["Country"].dropna().astype(str).unique().tolist())
    countries_sel = set(st.sidebar.multiselect("Country", countries))

    return {"years": years_sel, "countries": countries_sel}

def apply_filters(panel: pd.DataFrame, flt: Dict) -> pd.DataFrame:
    if panel.empty:
        return panel
    lo, hi = flt["years"]
    out = panel[(panel["Year"] >= lo) & (panel["Year"] <= hi)]
    if flt["countries"]:
        out = out[out["Country"].astype(str).isin(flt["countries"])]
    return out

# ---------------------------- App Body ---------------------------- #
st.title("Global Terrorism Statistic")
st.write(
    "Explore terrorism trends by country and year—spot peaks, compare regions, and export the data"
)

# Quick Start callout (compact guidance)
with st.expander("Quick Start", expanded=False):
    st.markdown(
        """
        1) Set a **Year range** and optional **Countries** in the sidebar.  
        2) Use **Overview** to see top countries in the selected period.  
        3) Open **Trends** for global time series across metrics.  
        4) Use **Drilldown** to group and sort by Country or Year, then download tables if needed.  
        """
    )

# Light glossary to explain the metrics
with st.expander("What the metrics mean", expanded=False):
    st.markdown(
        """
        - **Terrorist attacks**: Number of recorded attack incidents.  
        - **Terrorism deaths**: Fatalities attributed to terrorist attacks.  
        - **Injuries from terrorist attacks**: Non-fatal injuries attributed to attacks.  
        *Notes*: Definitions and coverage depend on the data sources you place in `data/terrorism/`.
        """
    )

panel = load_panel()

if panel.empty:
    st.error(
        "Could not assemble the panel dataset. Make sure your files contain 'Country', 'ISO_Code', 'Year' "
        "and at least one of: Attacks / Deaths / Injuries."
    )
    st.stop()

filters = sidebar_filters(panel)
pdf = apply_filters(panel, filters)

# ---------------------------- KPIs ---------------------------- #
st.subheader("Key metrics")
metric_cols = [c for c in ["Terrorist attacks", "Terrorism deaths", "Injuries from terrorist attacks"] if c in pdf.columns]
kpi_cols = st.columns(max(1, len(metric_cols)))
for i, m in enumerate(metric_cols):
    total = int(pd.to_numeric(pdf[m], errors="coerce").fillna(0).sum())
    kpi_cols[i].metric(m, f"{total:,}")

# ---------------------------- Tabs ---------------------------- #
tab_overview, tab_trends, tab_drill, tab_about = st.tabs(["Overview", "Trends", "Drilldown", "About"])

# ---- Overview ---- #
with tab_overview:
    st.subheader("Top countries (totals in selected period)")

    if "Country" not in pdf.columns or pdf.empty:
        st.info("No country information available.")
    else:
        if metric_cols:
            agg = (
                pdf.groupby("Country")[metric_cols].sum(min_count=1)
                   .fillna(0)
                   .sort_values(metric_cols[0], ascending=False)
            )
            st.dataframe(agg.head(25), use_container_width=True)
            # Optional: download full aggregation table
            st.download_button(
                "Download full country totals (CSV)",
                data=agg.reset_index().to_csv(index=False).encode("utf-8"),
                file_name=f"overview_country_totals_{filters['years'][0]}_{filters['years'][1]}.csv",
                mime="text/csv",
            )

            if px is not None and not agg.empty:
                for m in metric_cols:
                    top = agg.sort_values(m, ascending=False).head(15).reset_index()
                    fig = px.bar(top, x=m, y="Country", orientation="h", title=f"Top countries by {m}")
                    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=520)
                    st.plotly_chart(fig, use_container_width=True)

                    # Download data used for this plot
                    st.download_button(
                        f"Download data: Top 15 by {m} (CSV)",
                        data=top.to_csv(index=False).encode("utf-8"),
                        file_name=f"top15_{m.replace(' ', '_').lower()}_{filters['years'][0]}_{filters['years'][1]}.csv",
                        mime="text/csv",
                    )

                    st.caption(
                        f"What you are seeing: Countries ranked by total **{m}** within the selected year range. "
                        "Longer bars indicate higher totals. This helps to quickly identify which countries account "
                        "for the greatest share of observed impact for the chosen metric."
                    )
                    with st.expander("How to read this chart"):
                        st.markdown(
                            f"""
                            - Sort focus: Bars are already sorted by **{m}** (descending).  
                            - Scope: Results reflect only your sidebar filters (year range, countries).  
                            - Drill deeper: Switch to **Drilldown** to compare by Year or export the table.  
                            """
                        )
        else:
            st.info("No metric columns detected for aggregation.")

# ---- Trends ---- #
with tab_trends:
    st.subheader("Yearly trends")
    if pdf.empty or pdf["Year"].dropna().empty or not metric_cols:
        st.info("No year/metric data available.")
    else:
        trend = (
            pdf.groupby("Year")[metric_cols]
               .sum(min_count=1)
               .fillna(0)
               .reset_index()
               .sort_values("Year")
        )
        st.dataframe(trend, use_container_width=True)

        # Download data used for the line plot
        st.download_button(
            "Download data: Global trends by year (CSV)",
            data=trend.to_csv(index=False).encode("utf-8"),
            file_name=f"trends_global_{filters['years'][0]}_{filters['years'][1]}.csv",
            mime="text/csv",
        )

        if px is not None and not trend.empty:
            fig = px.line(
                trend,
                x="Year",
                y=metric_cols,
                markers=True,
                title="Global trends by year",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=520)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "What you are seeing: Global totals per year for each available metric within your selected range. "
                "Rising lines signal increasing totals; dips suggest declines or quieter periods. "
                "This view emphasizes long-term direction and turning points across metrics."
            )
            with st.expander("How to read this chart"):
                st.markdown(
                    """
                    - Compare slopes: Steeper rises imply faster growth in totals for that metric.  
                    - Check alignment: If lines move together, it suggests synchronized dynamics across metrics.  
                    - Hover for exact values and use the legend to toggle series visibility.  
                    """
                )

# ---- Drilldown ---- #
with tab_drill:
    st.subheader("Slice & dice")
    if pdf.empty or not metric_cols:
        st.info("No data to drill down.")
    else:
        group_by = st.selectbox("Group by", ["Country", "Year"])
        metric = st.selectbox("Metric", metric_cols)

        if metric in pdf.columns:
            by = (
                pdf.groupby(group_by)[metric]
                   .sum(min_count=1)
                   .fillna(0)
                   .reset_index(name="value")
                   .sort_values("value", ascending=False)
            )
            st.dataframe(by, use_container_width=True, height=480)

            # Download data used for the drilldown plot
            st.download_button(
                f"Download data: {metric} by {group_by} (CSV)",
                data=by.to_csv(index=False).encode("utf-8"),
                file_name=f"drilldown_{metric.replace(' ', '_').lower()}_by_{group_by.lower()}_{filters['years'][0]}_{filters['years'][1]}.csv",
                mime="text/csv",
            )

            if px is not None and not by.empty:
                if group_by == "Year":
                    fig = px.line(by.sort_values("Year"), x="Year", y="value", markers=True,
                                  title=f"{metric} by Year")
                    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=520)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"What you are seeing: Total **{metric}** per year after applying your filters. "
                        "Peaks mark years with heightened totals, troughs indicate quieter periods."
                    )
                    with st.expander("How to read this view"):
                        st.markdown(
                            """
                            - Use this to isolate time dynamics for a single metric.  
                            - Combine with the **Overview** tab to see which countries are driving peaks.  
                            - Export the table above for offline analysis.  
                            """
                        )
                else:
                    fig = px.bar(by.head(30), x="value", y=group_by, orientation="h",
                                 title=f"Top {group_by} by {metric}")
                    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=520)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        f"What you are seeing: Entities ranked by total **{metric}**. "
                        "The longest bars contribute the most under your current filters."
                    )
                    with st.expander("How to read this view"):
                        st.markdown(
                            f"""
                            - Narrow scope via the sidebar to focus on a subset of countries/years.  
                            - Switch **Group by** to change perspective (Country ↔ Year).  
                            - Click a bar label to quickly find it in the table above.  
                            """
                        )
        else:
            st.info("Chosen metric not present in the current data.")

# ---- About ---- #
with tab_about:
    st.header("About this topic")
    st.markdown(
        """
        This topic compiles terrorism-related indicators from multiple files
        standardizes their keys (Country, ISO_Code, Year), auto-detects the primary
        metrics (attacks, deaths, injuries), and merges them into a single Country × Year panel.

        **What you can do here**
        - **Overview**: Identify top countries by metric totals in your chosen period.  
        - **Trends**: Track global totals by year and compare metrics side by side.  
        - **Drilldown**: Group by Country or Year to explore patterns and export tables.  

        **How to interpret the metrics**
        - Higher **Terrorist attacks** → more incidents recorded.  
        - Higher **Terrorism deaths** → more fatalities attributed to attacks.  
        - Higher **Injuries from terrorist attacks** → more non-fatal victims.  
        """
    )

st.caption("Workflow: multi-file ingest → header normalization → metric detection → panel merge. CSV downloads included for every plot.")
