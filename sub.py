from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ==============================
# Page config (clean + professional)
# ==============================
st.set_page_config(
    page_title="Crime Indicators Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle, professional styling (consistent with your other dashboards)
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
        .stPlotlyChart { border-radius: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
        .stDataFrame, .dataframe { border: 1px solid #e8e8e8; border-radius: 10px; }
        .metric-small .stMetric { padding: 0.25rem 0.5rem; }
        h1, h2, h3 { margin-bottom: 0.5rem; }
        .css-1v0mbdj { padding-top: 0 !important; }
        div[data-baseweb="tab"] { font-weight: 600; }
        section[data-testid="stSidebar"] .stMarkdown h3 { margin-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Robust data dir discovery
# ------------------------------
def find_data_dir() -> Path:
    """
    Try multiple locations to find ./data/sub:
    1) next to this file
    2) current working directory
    3) parent of this file (for some deployment layouts)
    """
    candidates: List[Path] = []
    try:
        here = Path(__file__).resolve().parent
        candidates.append(here / "data" / "sub")
        candidates.append(here.parent / "data" / "sub")
    except NameError:
        pass
    candidates.append(Path.cwd() / "data" / "sub")

    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    # default best-guess
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    return base / "data" / "sub"

DATA_DIR = find_data_dir()

# ------------------------------
# Normalization helpers
# ------------------------------
def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in cols:
        for cand in candidates:
            if cand.lower() in str(c).lower():
                return c
    return None

def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = _drop_unnamed(df.copy())
    col_map = {
        "country": _find_first_col(df, ["Country or Area", "Country", "Area", "LOCATION", "Entity"]),
        "indicator": _find_first_col(df, ["Indicator", "Series", "Series description"]),
        "unit": _find_first_col(df, ["Unit of measurement", "Unit", "Units"]),
        "year": _find_first_col(df, ["Year", "Time", "TIME", "Year Code"]),
        "value": _find_first_col(df, ["VALUE", "Value", "Observation", "OBS_VALUE", "val"]),
        "code": _find_first_col(df, ["ISO3", "ISO Code", "Country Code", "Code", "M49 Code"]),
        "region": _find_first_col(df, ["Region", "Subregion", "M49 Region"]),
    }
    for key in ("year", "value"):
        c = col_map.get(key)
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, col_map

# ------------------------------
# Discovery & loading (CSV + Excel)
# ------------------------------
@st.cache_data(show_spinner=True)
def discover_files(directory: Path) -> Dict[str, Path]:
    """Find CSV/XLS/XLSX files and map them to friendly names."""
    known_map = {
        "access_and_functioning_of_justice": "Access & Justice",
        "intentional_homicide": "Intentional Homicide",
        "prisons_and_prisoners": "Prisons & Prisoners",
        "violent_and_sexual_crime": "Violent & Sexual Crime",
        "glotip": "GLOTIP",
        "firearms_trafficking": "Firearms Trafficking",
        "m49_regions": "UN M49 Regions",
        "sdg_dataset": "SDG Dataset",
    }
    out: Dict[str, Path] = {}
    paths = sorted(list(directory.glob("*.csv")) + list(directory.glob("*.xlsx")) + list(directory.glob("*.xls")))
    for p in paths:
        fname = p.name.lower()
        label = None
        for k, v in known_map.items():
            if k in fname:
                label = v
                break
        if not label:
            label = p.stem.replace("_", " ").title()
        base = label
        i = 2
        while label in out and out[label] != p:
            label = f"{base} ({i})"
            i += 1
        out[label] = p
    return out

@st.cache_data(show_spinner=True)
def get_excel_sheets(path: Path) -> List[str]:
    """List sheet names for an Excel file."""
    try:
        xls = pd.ExcelFile(path)
        return list(xls.sheet_names)
    except Exception:
        return []

@st.cache_data(show_spinner=True)
def load_table(path: Path, sheet: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    Load a table from CSV or Excel (specific sheet when provided),
    try header=2 first (as in typical UN CSVs), then fall back.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            df = pd.read_csv(path, header=2)
        except Exception:
            df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        try:
            if sheet is not None:
                try:
                    df = pd.read_excel(path, sheet_name=sheet, header=2)
                except Exception:
                    df = pd.read_excel(path, sheet_name=sheet)
            else:
                try:
                    df = pd.read_excel(path, sheet_name=0, header=2)
                except Exception:
                    df = pd.read_excel(path, sheet_name=0)
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel: {e}")
    else:
        raise RuntimeError(f"Unsupported file type: {suffix}")

    df, cmap = normalize_columns(df)
    return df, cmap

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = (df.isna().mean() * 100.0).round(2).rename("Missing %")
    return miss.sort_values(ascending=False).to_frame()

# ------------------------------
# Sidebar: discovery + selection
# ------------------------------
st.sidebar.markdown("## Data Source")
st.sidebar.caption(f"Looking in: `{DATA_DIR}`")

if not DATA_DIR.exists():
    st.sidebar.error(
        "Couldn't find the `data/sub/` folder.\n\n"
        "Create it next to this file, e.g.:\n"
        "```\n./app.py\n./data/sub/yourfile.csv\n```"
    )
    st.stop()

file_map = discover_files(DATA_DIR)
st.sidebar.caption(f"Discovered files: **{len(file_map)}**")

if len(file_map) == 0:
    st.sidebar.error(
        "No CSV/XLSX files found in `data/sub/`.\n\n"
        "Please add at least one `.csv` or `.xlsx` file and reload."
    )
    st.stop()

options = list(file_map.keys())
ds_name = st.sidebar.selectbox("Choose a dataset", options=options, index=0, key="ds_select")

if ds_name is None or ds_name not in file_map:
    st.sidebar.error("Invalid dataset selection. Please pick a dataset from the list.")
    st.stop()

ds_path = file_map[ds_name]

# If Excel, allow sheet selection
sheet_choice: Optional[str] = None
if ds_path.suffix.lower() in (".xlsx", ".xls"):
    sheets = get_excel_sheets(ds_path)
    if len(sheets) > 1:
        sheet_choice = st.sidebar.selectbox("Sheet (Excel)", options=sheets, index=0)
    elif len(sheets) == 1:
        sheet_choice = sheets[0]

# Load selected table
try:
    df_raw, cmap = load_table(ds_path, sheet_choice)
except Exception as e:
    st.sidebar.error(f"Failed to load `{ds_path.name}`: {e}")
    st.stop()

df = df_raw.copy()

# ------------------------------
# Filters
# ------------------------------
ycol, vcol, ccol, icol, ucol = (
    cmap.get("year"),
    cmap.get("value"),
    cmap.get("country"),
    cmap.get("indicator"),
    cmap.get("unit"),
)

# Year range
if ycol in df.columns and df[ycol].notna().any():
    y_min, y_max = int(np.nanmin(df[ycol])), int(np.nanmax(df[ycol]))
    year_range = st.sidebar.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max))
    df = df[(df[ycol] >= year_range[0]) & (df[ycol] <= year_range[1])]
else:
    year_range = (None, None)

# Country
country_val = None
if ccol in df.columns:
    countries = sorted([c for c in df[ccol].dropna().astype(str).unique()])
    country_choice = st.sidebar.selectbox("Country (optional)", options=["All"] + countries, index=0)
    if country_choice != "All":
        country_val = country_choice
        df = df[df[ccol] == country_val]

# Indicator
indicator_val = None
if icol in df.columns:
    inds = sorted([c for c in df[icol].dropna().astype(str).unique()])
    indicator_choice = st.sidebar.selectbox("Indicator (optional)", options=["All"] + inds, index=0)
    if indicator_choice != "All":
        indicator_val = indicator_choice
        df = df[df[icol] == indicator_val]

# Unit
if ucol in df.columns:
    units = sorted([c for c in df[ucol].dropna().astype(str).unique()])
    if len(units) > 1:
        unit_choice = st.sidebar.selectbox("Unit (optional)", options=["All"] + units, index=0)
        if unit_choice != "All":
            df = df[df[ucol] == unit_choice]

# ==============================
# Header
# ==============================
st.title("Crime Indicators Explorer")
subtitle = f"Dataset: **{ds_name}**"
if ds_path.suffix.lower() in (".xlsx", ".xls") and sheet_choice:
    subtitle += f" — Sheet: **{sheet_choice}**"

# Brief purpose line (replaceable)
st.write("Crime Indicators Explorer quickly — apply flexible filters (year, country, indicator, unit), see trends and top contrasts at a glance, " \
"check missing values and column types, and download ready-to-use CSVs from each view.")

# Quick Start
with st.expander("Quick Start", expanded=False):
    st.markdown(
        """
        1) Pick a **dataset** (and sheet for Excel).  
        2) Use sidebar filters (Year / Country / Indicator / Unit).  
        3) Check **Visuals** for time series, comparisons, and heatmaps.  
        4) Use **Geography** for a world map and **Data Quality** to inspect missing values.  
        """
    )

# What the columns mean (generic glossary)
with st.expander("What the columns mean", expanded=False):
    st.markdown(
        """
        - **Year**: Period the observation refers to.  
        - **Value**: Numeric observation (sum/mean depends on your analysis).  
        - **Country / Code**: Geographic identifiers (name or ISO/M49).  
        - **Indicator / Unit**: What is being measured and in which unit.  
        *Note*: The app auto-detects column names and adapts charts accordingly.
        """
    )

# ==============================
# KPI Row
# ==============================
def compute_kpis(df: pd.DataFrame, cmap: Dict[str, Optional[str]], sel_years: Tuple[Optional[int], Optional[int]]) -> Dict[str, float]:
    ycol, vcol = cmap.get("year"), cmap.get("value")
    if not (ycol and vcol) or ycol not in df or vcol not in df:
        return {}
    dff = df.dropna(subset=[ycol, vcol]).copy()
    if sel_years[0] is not None:
        dff = dff[dff[ycol] >= sel_years[0]]
    if sel_years[1] is not None:
        dff = dff[dff[ycol] <= sel_years[1]]
    if dff.empty:
        return {}
    total = float(dff[vcol].sum())
    mean_val = float(dff[vcol].mean())
    recent_year = int(np.nanmax(dff[ycol]))
    rec_val = float(dff.loc[dff[ycol] == recent_year, vcol].sum())
    return {"Total (filtered)": total, "Mean per row": mean_val, f"Total in {recent_year}": rec_val}

kpis = compute_kpis(df_raw, cmap, year_range)
if kpis:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total (filtered)", f"{kpis.get('Total (filtered)', 0):,.0f}")
    c2.metric("Mean per row", f"{kpis.get('Mean per row', 0):,.2f}")
    last_key = next((k for k in kpis.keys() if k.startswith("Total in ")), None)
    if last_key:
        c3.metric(last_key, f"{kpis[last_key]:,.0f}")
else:
    st.info("KPIs will appear when standard Year/Value columns are detected.")

# ==============================
# Tabs
# ==============================
tab_explore, tab_visuals, tab_geo, tab_quality, tab_about = st.tabs(
    ["Data Explorer", "Visuals", "Geography", "Data Quality", "About"]
)

with tab_explore:
    st.subheader("Data Preview")
    st.write(f"**Rows:** {len(df):,}  |  **Columns:** {df.shape[1]}")
    st.dataframe(df.head(1000), use_container_width=True)
    st.download_button(
        "Download current slice (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ds_name.replace(' ', '_').lower()}_slice.csv",
    )
    st.caption("What you are seeing: The first 1,000 rows of the filtered table to help you validate columns, units, and coverage.")

# ---------- Plot helpers (now include CSV downloads & reading guides) ----------
def plot_trend(df: pd.DataFrame, cmap: Dict[str, Optional[str]], country: Optional[str], indicator: Optional[str]):
    ycol, vcol, ccol, icol = cmap.get("year"), cmap.get("value"), cmap.get("country"), cmap.get("indicator")
    if not (ycol and vcol) or ycol not in df or vcol not in df:
        st.info("No standard Year/Value columns detected for a trend plot.")
        return
    dff = df.dropna(subset=[ycol, vcol]).copy()
    if ccol and country:
        dff = dff[dff[ccol] == country]
    if icol and indicator:
        dff = dff[dff[icol] == indicator]
    g = dff.groupby(ycol, as_index=False)[vcol].sum()
    if g.empty:
        st.info("No data for selected filters.")
        return
    fig = px.line(g, x=ycol, y=vcol, markers=True, title="Trend Over Time (Summed)")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "Download data: Trend over time (CSV)",
        data=g.to_csv(index=False).encode("utf-8"),
        file_name="trend_over_time.csv",
        mime="text/csv",
    )
    st.caption(
        "What you are seeing: Year-by-year totals (sum of the Value column) after applying your filters. "
        "Rising segments indicate growing totals, dips suggest quieter periods."
    )
    with st.expander("How to read this chart"):
        st.markdown(
            "- Look for slope changes to spot turning points.  \n"
            "- Hover for exact values; use sidebar filters (Country/Indicator) to isolate a series.  \n"
            "- If the line is jagged, consider longer ranges or smoothing offline."
        )

def plot_grouped_bars(df: pd.DataFrame, cmap: Dict[str, Optional[str]], top_n: int = 5):
    ycol, vcol, icol = cmap.get("year"), cmap.get("value"), cmap.get("indicator")
    if not (ycol and vcol and icol) or icol not in df:
        st.info("Need Year/Value/Indicator columns for grouped bars.")
        return
    dff = df.dropna(subset=[ycol, vcol]).copy()
    g = dff.groupby([ycol, icol], as_index=False)[vcol].sum()
    top = g.groupby(icol, as_index=False)[vcol].sum().sort_values(vcol, ascending=False)[icol].head(top_n)
    g2 = g[g[icol].isin(top)]
    if g2.empty:
        st.info("No data available for grouped bars.")
        return
    fig = px.bar(g2, x=ycol, y=vcol, color=icol, barmode="group", title="Top Indicators — Grouped by Year")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "Download data: Grouped bars (CSV)",
        data=g2.to_csv(index=False).encode("utf-8"),
        file_name="grouped_bars_top_indicators.csv",
        mime="text/csv",
    )
    st.caption(
        "What you are seeing: For the highest-impact indicators, bars show how totals change across years. "
        "Compare colors within each year to see which indicators dominate."
    )
    with st.expander("How to read this chart"):
        st.markdown(
            "- Focus on relative bar heights per year to compare indicators.  \n"
            "- Adjust **Top N** (code) if you want to widen/narrow the set.  \n"
            "- If one indicator dwarfs others, consider per-capita normalization offline."
        )

def plot_heatmap(df: pd.DataFrame, cmap: Dict[str, Optional[str]], indicator: Optional[str]):
    icol, ycol, vcol = cmap.get("indicator"), cmap.get("year"), cmap.get("value")
    if not (ycol and vcol):
        st.info("Need Year and Value columns for a heatmap.")
        return
    dff = df.dropna(subset=[ycol, vcol]).copy()
    if icol and indicator:
        dff = dff[dff[icol] == indicator]
    g = dff.groupby([ycol], as_index=False)[vcol].sum()
    if g.empty:
        st.info("No data for heatmap.")
        return
    g["metric"] = "Total"
    pivot = g.pivot(index="metric", columns=ycol, values=vcol)
    fig = px.imshow(pivot, aspect="auto", title="Heatmap — Total by Year")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "Download data: Heatmap (CSV)",
        data=pivot.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="heatmap_total_by_year.csv",
        mime="text/csv",
    )
    st.caption(
        "What you are seeing: A single-row heatmap where each column is a year and cell shading reflects the total. "
        "lighter cells mark higher totals, enabling quick spotting of hot years."
    )
    with st.expander("How to read this heatmap"):
        st.markdown(
            "- Scan for lightest cells to find peak years.  \n"
            "- Filter by **Indicator** to focus the heatmap on one series.  \n"
            "- Use alongside the Trend chart for detail."
        )

def plot_choropleth(df: pd.DataFrame, cmap: Dict[str, Optional[str]], year: Optional[int]):
    ccol, vcol, ycol, code = cmap.get("country"), cmap.get("value"), cmap.get("year"), cmap.get("code")
    if not vcol or (not ccol and not code):
        st.info("Need Country/Code and Value columns for a choropleth.")
        return
    dff = df.dropna(subset=[vcol]).copy()
    if year and ycol in dff.columns:
        dff = dff[dff[ycol] == year]
    locations = code if (code and code in dff.columns) else (ccol if ccol in dff.columns else None)
    if dff.empty or locations is None:
        st.info("No geo data for choropleth with current filters.")
        return
    fig = px.choropleth(
        dff,
        locations=locations,
        color=vcol,
        hover_name=ccol if ccol in dff.columns else None,
        color_continuous_scale="Viridis",
        title="Global Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        f"Download data: Map {year if year else 'all years'} (CSV)",
        data=dff[[c for c in [locations, ccol, ycol, vcol] if c in dff.columns]].to_csv(index=False).encode("utf-8"),
        file_name=f"map_{year if year else 'all'}.csv",
        mime="text/csv",
    )
    st.caption(
        "What you are seeing: A world map shaded by the Value column for the selected year. "
        "Darker shades correspond to higher values, helping you spot geographic hotspots."
    )
    with st.expander("How to read this map"):
        st.markdown(
            "- Compare neighboring countries’ shades to see regional contrasts.  \n"
            "- If country matching fails, ensure your file has ISO3 codes or clean country names.  \n"
            "- Use the year slider to scan how the map changes over time."
        )

with tab_visuals:
    st.subheader("Time Series & Comparisons")
    left, right = st.columns(2)
    with left:
        st.markdown("**Trend Over Time**")
        plot_trend(df_raw, cmap, country_val, indicator_val)
    with right:
        st.markdown("**Grouped Bars (Top Indicators)**")
        plot_grouped_bars(df_raw, cmap, top_n=5)

    st.markdown("---")
    st.markdown("**Heatmap**")
    plot_heatmap(df_raw, cmap, indicator_val)

with tab_geo:
    st.subheader("Global Distribution")
    year_for_map = None
    if ycol in df_raw.columns and df_raw[ycol].notna().any():
        y_min0, y_max0 = int(np.nanmin(df_raw[ycol])), int(np.nanmax(df_raw[ycol]))
        year_for_map = st.slider("Select a year for the map", min_value=y_min0, max_value=y_max0, value=y_max0)
    plot_choropleth(df_raw, cmap, year_for_map)

with tab_quality:
    st.subheader("Data Quality Overview")
    q_left, q_right = st.columns([2, 1], vertical_alignment="top")

    with q_left:
        st.markdown("**Missing Values (%)**")
        miss_tbl = summarize_missing(df_raw)
        st.dataframe(miss_tbl, use_container_width=True, height=400)
        st.download_button(
            "Download data: Missing values (%) (CSV)",
            data=miss_tbl.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="missing_values_percent.csv",
            mime="text/csv",
        )
        st.caption("What you are seeing: Share of missing values per column to help prioritize cleaning.")

        st.markdown("**Type Summary**")
        dtype_summary = pd.DataFrame({
            "Column": df_raw.columns,
            "Dtype": df_raw.dtypes.astype(str).values,
            "Non-Null Count": df_raw.notna().sum().values,
            "Null Count": df_raw.isna().sum().values,
            "Unique Values": [df_raw[c].nunique(dropna=True) for c in df_raw.columns],
        })
        st.dataframe(dtype_summary, use_container_width=True, height=400)
        st.download_button(
            "Download data: Type summary (CSV)",
            data=dtype_summary.to_csv(index=False).encode("utf-8"),
            file_name="type_summary.csv",
            mime="text/csv",
        )
        st.caption("What you are seeing: Basic schema profile — types, nulls, and cardinality per column.")

    with q_right:
        st.markdown("**Quick Stats**")
        stats = df_raw.select_dtypes(include=[np.number]).describe().T
        st.dataframe(stats, use_container_width=True, height=400)
        st.download_button(
            "Download data: Quick stats (CSV)",
            data=stats.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="quick_stats_numeric.csv",
            mime="text/csv",
        )
        st.caption("What you are seeing: Descriptive stats for all numeric columns (count/mean/std/min/percentiles/max).")

with tab_about:
    st.header("About this topic")
    st.markdown(
        """
       This topic is a simple, fast way to look at justice and crime data without opening spreadsheets.
        It reads CSV and Excel files, figures out the common columns
        (Year, Value, Country, Indicator, Unit, Code) even if they’re named a bit differently, and gives
        you interactive views you can filter and export.

        What you can do here
        - Filter by year, country, indicator, and unit to focus on the slice you care about.
        - See quick totals and simple KPIs at the top.
        - Explore trends over time, compare indicators, and view a heatmap of peak years.
        - See a world map for a selected year (works best if your file includes ISO3 codes or clean country names).
        - Inspect data quality (missing values, types, basic stats) and export any table for further analysis.

        Tabs at a glance
        - Data Explorer: preview the filtered table and download the current slice as CSV.
        - Visuals: line chart for trends, grouped bars for top indicators, and a quick heatmap.
        - Geography: choropleth map for one year to spot hotspots at a glance.
        - Data Quality: missing-value percentages, column types, and numeric summary stats.
        - About: this page.

        Downloads and reporting
        - Every chart has a “Download data” button so you can grab exactly what you’re seeing.
        - Use these exports in notebooks, BI tools, or slides without re-building the queries.

        Tips and limitations
        - If results look odd, check “Unit” and “Indicator” in the sidebar—mixed units can be misleading.
        - Country matching for the map is easiest with ISO3 codes; otherwise make sure names are consistent.
        - Very large files can use more memory, especially if you choose the “preload all files” option.
        - The app shows totals (sums) in most charts. If you need per-capita or other normalizations,
          export the data and apply your own transformations.

        Privacy note
        - Files stay on your machine (or wherever you run Streamlit). Nothing is uploaded unless your
          Streamlit deployment does that by design. Remote downloads only happen if you add URLs yourself.

        That’s it—pick a file, set a few filters, and you’re ready to explore and export. 
        """
    )

st.markdown("---")
st.caption("flexible CSV/Excel explorer with automatic Year/Value/Country/Indicator/Unit/Code detection.")
