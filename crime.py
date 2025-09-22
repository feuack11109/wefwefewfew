"""
Global Organized Crime Index 

Expected data location (relative to this file):
    ./data/crime/
        global_oc_index_2021_dataset.xlsx
        global_oc_index_2023_dataset.xlsx

This app is robust to slight file name differences and will search for
any *.xlsx in data/crime/ containing the year string ("2021", "2023").
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------
# Page & Global Config
# ---------------------------
st.set_page_config(
    page_title="Global Organized Crime Index",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light styling polish (cleaner spacing, subtle shadows, consistent radii)
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
        .stPlotlyChart { border-radius: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
        .stDataFrame, .dataframe { border: 1px solid #e8e8e8; border-radius: 10px; }
        .metric-small .stMetric { padding: 0.25rem 0.5rem; }
        /* Tighten up headers slightly */
        h1, h2, h3 { margin-bottom: 0.5rem; }
        /* Reduce extra padding above the top container if present */
        .css-1v0mbdj { padding-top: 0 !important; }
        /* Make tabs look a bit more card-like */
        div[data-baseweb="tab"] { font-weight: 600; }
        /* Sidebar section titles */
        section[data-testid="stSidebar"] .stMarkdown h3 { margin-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path("data/crime")
YEAR_FILES_HINTS = {
    2021: ["2021"],
    2023: ["2023"],
}

# Short explainer text per metric (shown contextually)
METRIC_DESCRIPTIONS: Dict[str, str] = {
    "Criminality": "Overall scale and scope of criminal activity within a country.",
    "Resilience": "The state's ability to withstand and disrupt organized crime.",
    "Criminal markets": "Breadth and intensity of illicit markets operating in the country.",
    "Criminal actors": "Presence, reach, and influence of different criminal actor types.",
    "Human trafficking": "Prevalence of trafficking for exploitation, coercion, or slavery.",
    "Human smuggling": "Scale of illicit facilitation of migration across borders.",
    "Arms trafficking": "Illicit manufacture, transfer, and distribution of firearms.",
    "Flora crimes": "Illegal exploitation and trade of plant species and products.",
    "Fauna crimes": "Illegal exploitation and trade of wildlife and animal products.",
    "Non-renewable resource crimes": "Illicit extraction and trade of minerals, oil, or gas.",
    "Drug trade": "Aggregate across heroin, cocaine, cannabis, and synthetic drug markets.",
}

# ---------------------------------
# Utilities: Loading & Normalization
# ---------------------------------

RENAME_2023 = {
    "Criminality avg.": "Criminality",
    "Criminal markets avg.": "Criminal markets",
    "Human trafficking avg.": "Human trafficking",
    "Human smuggling avg.": "Human smuggling",
    "Arms trafficking avg.": "Arms trafficking",
    "Flora crimes avg.": "Flora crimes",
    "Fauna crimes avg.": "Fauna crimes",
    "Non-renewable resource crimes avg.": "Non-renewable resource crimes",
    "Heroin trade avg.": "Heroin trade",
    "Cocaine trade avg.": "Cocaine trade",
    "Cannabis trade avg.": "Cannabis trade",
    "Synthetic drug trade avg.": "Synthetic drug trade",
    "Criminal actors avg.": "Criminal actors",
    "Mafia-style groups avg.": "Mafia-style groups",
    "Criminal networks avg.": "Criminal networks",
    "State-embedded actors avg.": "State-embedded actors",
    "Foreign actors avg.": "Foreign actors",
    "Resilience avg.": "Resilience",
    "Political leadership and governance avg.": "Political leadership and governance",
    "Government transparency and accountability avg.": "Government transparency and accountability",
    "International cooperation avg.": "International cooperation",
    "National policies and laws avg.": "National policies and laws",
    "Judicial system and detention avg.": "Judicial system and detention",
    "Law enforcement avg.": "Law enforcement",
    "Territorial integrity avg.": "Territorial integrity",
    "Anti-money laundering avg.": "Anti-money laundering",
    "Economic regulatory capacity avg.": "Economic regulatory capacity",
    "Victim and witness support avg.": "Victim and witness support",
    "Prevention avg.": "Prevention",
    "Non-state actors avg.": "Non-state actors",
}

def _normalize_col(c: str) -> str:
    c = c.strip()
    c = re.sub(r"\s+", " ", c)
    c = re.sub(r"\s+avg[\s\.,:;]*$", "", c, flags=re.IGNORECASE)
    return c

@st.cache_data(show_spinner=False)
def find_file_for_year(year: int) -> Optional[Path]:
    if not DATA_DIR.exists():
        return None
    candidates: List[Path] = list(DATA_DIR.glob("*.xlsx"))
    hints = YEAR_FILES_HINTS.get(year, [str(year)])
    for p in candidates:
        name = p.name.lower()
        if any(h.lower() in name for h in hints):
            return p
    # fallback: first excel file (prevents total failure in demo)
    return candidates[0] if candidates else None

@st.cache_data(show_spinner=False)
def load_year(year: int) -> Tuple[pd.DataFrame, Optional[str]]:
    """Load a single year's dataset, attempting light normalization.
    Returns (dataframe, message). Message is informative text about what we did.
    """
    msg = None
    path = find_file_for_year(year)
    if path is None:
        return pd.DataFrame(), f"No file found in {DATA_DIR}/ for {year}."

    df = pd.read_excel(path)

    # Robust basic column normalization for 2023; 2021 usually already clean.
    if year == 2023:
        before_cols = set(df.columns)
        df = df.rename(columns=RENAME_2023)
        df = df.rename(columns=lambda c: _normalize_col(str(c)))
        after_cols = set(df.columns)
        changed = before_cols.symmetric_difference(after_cols)
        if changed:
            msg = f"Normalized {len(changed)} column names for 2023."

    # Ensure standard identity columns exist if present under slightly different names
    col_aliases = {
        "Country": ["country", "COUNTRY", "State"],
        "Region": ["region", "REGION"],
        "Continent": ["continent", "CONTINENT"],
        "ISO3": ["ISO 3", "ISO-3", "Alpha-3", "ISO alpha-3", "iso3"],
    }
    for target, aliases in col_aliases.items():
        if target not in df.columns:
            for alt in df.columns:
                if alt in aliases:
                    df = df.rename(columns={alt: target})
                    break

    # Create Drug trade aggregate if its parts exist
    drug_parts = ["Heroin trade", "Cocaine trade", "Cannabis trade", "Synthetic drug trade"]
    if all(col in df.columns for col in drug_parts) and "Drug trade" not in df.columns:
        df["Drug trade"] = df[drug_parts].mean(axis=1)

    # Cast numeric-looking columns to numeric safely
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass

    return df, msg

@st.cache_data(show_spinner=False)
def load_all() -> Dict[int, pd.DataFrame]:
    data = {}
    for yr in sorted(YEAR_FILES_HINTS.keys()):
        df, _ = load_year(yr)
        if not df.empty:
            data[yr] = df
    return data

def numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def safe_corr(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr() if cols else pd.DataFrame()

# ---------------------------
# Sidebar Controls
# ---------------------------

data_by_year = load_all()

st.sidebar.header("Filters")

available_years = sorted(data_by_year.keys())
if not available_years:
    st.error(
        f"No datasets found. Please place your Excel files under `{DATA_DIR}`.\n"
        "Expected filenames to contain 2021 / 2023."
    )
    st.stop()

selected_year = st.sidebar.selectbox("Primary year", available_years, index=len(available_years) - 1)

# For comparisons
compare_to = st.sidebar.selectbox(
    "Compare with (optional)", ["None"] + [str(y) for y in available_years if y != selected_year], index=0
)
compare_year = int(compare_to) if compare_to != "None" else None

# Build region filter based on primary dataset
_df = data_by_year[selected_year]
regions = sorted(_df["Region"].dropna().unique()) if "Region" in _df.columns else []
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions[:]) if regions else []

# Columns of interest (common metrics)
metric_candidates = [
    "Criminality",
    "Resilience",
    "Criminal markets",
    "Criminal actors",
    "Human trafficking",
    "Human smuggling",
    "Arms trafficking",
    "Flora crimes",
    "Fauna crimes",
    "Non-renewable resource crimes",
    "Drug trade",
]
metric_candidates = [m for m in metric_candidates if m in _df.columns]

primary_metric = st.sidebar.selectbox("Primary metric", metric_candidates, index=0)
secondary_metric = st.sidebar.selectbox(
    "Secondary metric", [m for m in metric_candidates if m != primary_metric], index=0
) if len(metric_candidates) > 1 else None

with st.sidebar.expander("How to use", expanded=False):
    st.markdown(
        """
        - Pick a **Primary year** and (optionally) a comparison year.  
        - Filter by **Regions**.  
        - Choose **Primary/Secondary metrics** to update all views.  
        - Use the tabs to explore maps, regions, correlations, and year-over-year changes.
        """
    )

# ---------------------------
# Filtered DataFrame
# ---------------------------

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if selected_regions and "Region" in df.columns:
        df = df[df["Region"].isin(selected_regions)]
    return df

df_primary = apply_filters(data_by_year[selected_year])
df_compare = apply_filters(data_by_year.get(compare_year, pd.DataFrame())) if compare_year else pd.DataFrame()

# ---------------------------
# Header & KPIs
# ---------------------------

st.title("Global Organized Crime Index — Prime Dashboard")
st.caption(
    "Explore country-level criminality, resilience, markets, and actors. "
    "Use the sidebar to filter regions, change metrics, and compare years."
)

# Quick Start callout (compact guidance without clutter)
with st.expander("Quick Start", expanded=False):
    st.markdown(
        """
        1) Pick a **Primary metric** in the sidebar — all views update automatically.  
        2) Use **Regions** to narrow to specific areas of interest.  
        3) Open **Compare Years** and select another year to see changes.  
        4) Hover on charts for tooltips; download tables/CSVs where available.  
        """
    )

if primary_metric in METRIC_DESCRIPTIONS:
    st.info(f"**{primary_metric}** — {METRIC_DESCRIPTIONS[primary_metric]}")

col_kpi = st.columns(4)

if primary_metric in df_primary.columns:
    with col_kpi[0]:
        st.metric(f"{primary_metric} · mean", f"{df_primary[primary_metric].mean():.2f}")
    with col_kpi[1]:
        st.metric(f"{primary_metric} · median", f"{df_primary[primary_metric].median():.2f}")

if "Criminality" in df_primary.columns and "Resilience" in df_primary.columns:
    with col_kpi[2]:
        corr_val = df_primary[["Criminality", "Resilience"]].corr().iloc[0, 1]
        st.metric("Corr(Criminality, Resilience)", f"{corr_val:.2f}")

with col_kpi[3]:
    st.metric("Countries", f"{len(df_primary)}")

# Top/Bottom table
if "Country" in df_primary.columns and primary_metric in df_primary.columns:
    top = df_primary.sort_values(primary_metric, ascending=False).head(10)[["Country", "Region", primary_metric]]
    bottom = df_primary.sort_values(primary_metric, ascending=True).head(10)[["Country", "Region", primary_metric]]
else:
    top = bottom = pd.DataFrame()

st.divider()

# ---------------------------
# Tabs (added "About")
# ---------------------------

t_overview, t_map, t_regions, t_cor, t_compare, t_country, t_data, t_about = st.tabs(
    [
        "Overview",
        "World Map",
        "Regions",
        "Correlations",
        "Compare Years",
        "Country Detail",
        "Data",
        "About",
    ]
)

# ---- Overview Tab ----
with t_overview:
    st.subheader("Relationship between selected metrics")
    c1, c2 = st.columns([1.3, 1])

    with c1:
        if primary_metric and secondary_metric and all(
            m in df_primary.columns for m in [primary_metric, secondary_metric]
        ):
            fig = px.scatter(
                df_primary,
                x=secondary_metric,
                y=primary_metric,
                color="Region" if "Region" in df_primary.columns else None,
                hover_name="Country" if "Country" in df_primary.columns else None,
                trendline="ols",
                template="plotly_white",
            )
            fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title=secondary_metric,
                yaxis_title=primary_metric,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"What you are seeing: Each circle represents a **country**. The x-axis shows "
                f"**{secondary_metric}** and the y-axis shows **{primary_metric}**. Colors indicate "
                "regional grouping. The dotted trendline illustrates whether countries with higher "
                f"values of {secondary_metric} tend to also have higher (or lower) {primary_metric} scores. "
                "Outliers far from the line highlight unusual cases compared to the global pattern."
            )
            with st.expander("How to read this chart"):
                st.markdown(
                    f"""
                    - Look for overall slope of the trendline to judge the direction of association between **{secondary_metric}** and **{primary_metric}**.  
                    - Dense clusters suggest typical ranges; isolated points are potential outliers to investigate.  
                    - Use the legend to toggle regions on/off and reveal hidden patterns.  
                    - Hover a point to see a country’s exact values.  
                    """
                )
        else:
            st.info("Please ensure both selected metrics exist in the dataset.")

    with c2:
        st.subheader("Leaders & Laggards")
        if not top.empty and not bottom.empty:
            st.write("Top 10 by primary metric")
            st.dataframe(top, use_container_width=True, height=240, hide_index=True)
            st.write("Bottom 10 by primary metric")
            st.dataframe(bottom, use_container_width=True, height=240, hide_index=True)
        else:
            st.info("Not enough information to render ranking tables.")

# ---- World Map Tab ----
with t_map:
    st.subheader("Global view")
    if primary_metric in df_primary.columns:
        # Choose locations: prefer ISO3, else use country names
        locations_col = None
        locationmode = None
        if "ISO3" in df_primary.columns and df_primary["ISO3"].notna().any():
            locations_col = "ISO3"
            locationmode = "ISO-3"
        elif "Country" in df_primary.columns:
            locations_col = "Country"
            locationmode = "country names"

        if locations_col is None:
            st.warning("No suitable location column (ISO3 or Country) found for choropleth.")
        else:
            fig = px.choropleth(
                df_primary,
                locations=locations_col,
                color=primary_metric,
                hover_name="Country" if "Country" in df_primary.columns else None,
                locationmode=locationmode,
                color_continuous_scale="RdYlBu_r",
                template="plotly_white",
            )
            fig.update_layout(height=600, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"What you are seeing: A choropleth world map where countries are shaded according to "
                f"their **{primary_metric}** value. Darker colors indicate higher scores, lighter colors "
                "indicate lower scores. Hovering over a country reveals its name and metric value. "
                "This view helps to quickly identify geographic patterns — for example, whether certain "
                "continents or regions consistently score higher or lower."
            )
            with st.expander("How to read this map"):
                st.markdown(
                    f"""
                    - Compare shading across neighboring countries to spot regional contrasts in **{primary_metric}**.  
                    - Use the sidebar to filter Regions if the map feels crowded.  
                    - Hover for value tooltips; click legend items to isolate patterns.  
                    - If boundaries look odd, switch location mode by ensuring the dataset has **ISO3** codes.  
                    """
                )
    else:
        st.info("Primary metric not available.")

# ---- Regions Tab ----
with t_regions:
    st.subheader("Regional distribution & profiles")
    if "Region" in df_primary.columns and primary_metric in df_primary.columns:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.box(
                df_primary,
                x="Region",
                y=primary_metric,
                points="outliers",
                template="plotly_white",
            )
            fig.update_layout(xaxis_title="Region", yaxis_title=primary_metric, height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"What you are seeing: A **box plot** of {primary_metric} scores across each region. "
                "The line inside the box shows the regional median, while the box covers the middle 50% of values. "
                "Dots represent countries that are statistical outliers. This visualization highlights "
                "regional variation and helps spot which regions have consistently high or low scores, "
                "and where disparities within a region are especially large."
            )
            with st.expander("How to read this box plot"):
                st.markdown(
                    f"""
                    - Compare medians across regions to rank central tendencies of **{primary_metric}**.  
                    - Taller boxes/whiskers indicate more dispersion (greater variability within a region).  
                    - Outlier dots may indicate special cases worth deeper investigation in *Country Detail*.  
                    """
                )

        with c2:
            # Regional means heatmap for key indicators
            num_cols = [c for c in [
                "Criminality",
                "Resilience",
                "Criminal markets",
                "Criminal actors",
                "Drug trade",
            ] if c in df_primary.columns]
            if num_cols:
                region_means = (
                    df_primary.groupby("Region")[num_cols]
                    .mean(numeric_only=True)
                    .sort_values(num_cols[0], ascending=False)
                )
                fig = px.imshow(
                    region_means,
                    text_auto=".1f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                )
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "What you are seeing: A **heatmap** of regional averages for several key indicators. "
                    "Each row is a region, and each column is an indicator (such as Criminality or Resilience). "
                    "Darker cells mean higher average scores. This view allows easy comparison of how "
                    "different regions perform across multiple dimensions at once."
                )
                with st.expander("How to read this heatmap"):
                    st.markdown(
                        """
                        - Scan rows to compare regions; scan columns to compare indicators.  
                        - Dark bands across a row indicate a region scoring higher across many indicators.  
                        - Use this as a guide to where deeper drill-down (map or country detail) may be useful.  
                        """
                    )
            else:
                st.info("Not enough numeric columns to render regional heatmap.")
    else:
        st.info("Region column not found.")

# ---- Correlations Tab ----
with t_cor:
    st.subheader("Correlation matrix")
    nums = numeric_columns(df_primary)
    focus_cols = st.multiselect(
        "Select columns to correlate",
        options=nums,
        default=[c for c in ["Criminality", "Resilience", "Criminal markets", "Criminal actors", "Drug trade"] if c in nums],
    )
    corr = safe_corr(df_primary, focus_cols)
    if not corr.empty:
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download correlation CSV",
            data=corr.to_csv().encode("utf-8"),
            file_name=f"correlations_{selected_year}.csv",
            mime="text/csv",
        )
        st.caption(
            "What you are seeing: A **correlation matrix** showing the statistical relationships between indicators. "
            "Values close to +1 mean that two variables increase together, while values close to -1 mean one increases "
            "when the other decreases. For example, a strong negative correlation between **Criminality** and "
            "**Resilience** would suggest that more resilient states generally experience less criminal activity."
        )
        with st.expander("How to read this matrix"):
            st.markdown(
                """
                - Focus on the largest magnitude values (darkest cells) to find strong relationships.  
                - Correlation ≠ causation: use this to generate hypotheses, not conclusions.  
                - If a pair looks interesting, revisit the **Overview** scatter and plot those two indicators.  
                """
            )
    else:
        st.info("Select at least two numeric columns.")

# ---- Compare Years Tab ----
with t_compare:
    st.subheader("Year-over-year comparison")
    if compare_year and not df_compare.empty and "Country" in df_primary.columns:
        join_cols = [c for c in ["Country", "ISO3", "Region"] if c in df_primary.columns and c in df_compare.columns]
        if "Country" not in join_cols:
            join_cols = ["Country"] if "Country" in df_primary.columns else join_cols
        merged = pd.merge(
            df_primary, df_compare, on=join_cols, suffixes=(f"_{selected_year}", f"_{compare_year}"), how="inner"
        )

        if primary_metric + f"_{selected_year}" in merged.columns and primary_metric + f"_{compare_year}" in merged.columns:
            merged["Δ"] = merged[primary_metric + f"_{selected_year}"] - merged[primary_metric + f"_{compare_year}"]
            st.markdown(f"**Change in {primary_metric}: {selected_year} vs {compare_year}**")
            fig = px.scatter(
                merged,
                x=primary_metric + f"_{compare_year}",
                y=primary_metric + f"_{selected_year}",
                hover_name="Country" if "Country" in merged.columns else None,
                color=("Region" if "Region" in merged.columns else None),
                template="plotly_white",
            )
            fig.add_trace(
                go.Scatter(
                    x=[merged[primary_metric + f"_{compare_year}"].min(), merged[primary_metric + f"_{compare_year}"].max()],
                    y=[merged[primary_metric + f"_{compare_year}"].min(), merged[primary_metric + f"_{compare_year}"].max()],
                    mode='lines', line=dict(dash='dash'), showlegend=False
                )
            )
            fig.update_layout(xaxis_title=str(compare_year), yaxis_title=str(selected_year), height=520)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"What you are seeing: Each point is a country, comparing its **{primary_metric}** score "
                f"in {compare_year} (x-axis) versus {selected_year} (y-axis). The dashed diagonal line represents "
                "no change. Countries above the line improved, while those below declined. The distance from the "
                "line reflects the magnitude of change. This helps identify which countries made the biggest gains "
                "or losses between the two years."
            )
            with st.expander("How to read this comparison"):
                st.markdown(
                    f"""
                    - Points above the dashed line improved their **{primary_metric}**; below the line declined.  
                    - Farther from the line = larger change (scan for biggest movers).  
                    - Use the table below to sort by Δ and download the results for reporting.  
                    """
                )

            st.dataframe(
                merged[["Country", "Region"] + [primary_metric + f"_{selected_year}", primary_metric + f"_{compare_year}", "Δ"]]
                .sort_values("Δ", ascending=False),
                use_container_width=True,
                height=420,
                hide_index=True,
            )
            st.download_button(
                "Download comparison table",
                data=merged.to_csv(index=False).encode("utf-8"),
                file_name=f"compare_{primary_metric}_{selected_year}_vs_{compare_year}.csv",
                mime="text/csv",
            )
        else:
            st.info("Primary metric missing in one of the years after merge.")
    else:
        st.info("Pick a comparison year in the sidebar to unlock this section.")

# ---- Country Detail Tab ----
with t_country:
    st.subheader("Country detail")
    if "Country" in df_primary.columns:
        country = st.selectbox("Country", sorted(df_primary["Country"].unique()))
        sub = df_primary[df_primary["Country"] == country]
        if not sub.empty:
            st.write(f"**{country} — key indicators**")
            show_cols = [c for c in [
                "Criminality", "Resilience", "Criminal markets", "Criminal actors", "Drug trade",
                "Human trafficking", "Human smuggling", "Arms trafficking", "Flora crimes", "Fauna crimes",
                "Non-renewable resource crimes",
            ] if c in sub.columns]
            if show_cols:
                row = sub[show_cols].mean()  # handle duplicates if any
                fig = go.Figure(go.Bar(x=row.index.tolist(), y=row.values.tolist()))
                fig.update_layout(template="plotly_white", height=480, margin=dict(l=10, r=10, t=30, b=30))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "What you are seeing: A **bar chart** of this country's average scores across selected indicators. "
                    "It allows quick comparison of strengths and weaknesses within the same country profile — for example, "
                    "a state may score relatively high in Resilience but still have elevated values in Drug trade."
                )
                with st.expander("How to read this profile"):
                    st.markdown(
                        """
                        - Compare bar heights to identify relative strengths/weaknesses for the selected country.  
                        - Revisit the **World Map** or **Regions** to see how this profile compares to neighbors.  
                        - Download the table to analyze changes offline or blend with external context.  
                        """
                    )
            st.dataframe(sub, use_container_width=True, hide_index=True)
            st.download_button(
                "Download country rows (CSV)",
                data=sub.to_csv(index=False).encode("utf-8"),
                file_name=f"{country}_{selected_year}.csv",
                mime="text/csv",
            )
    else:
        st.info("Country column not found.")

# ---- Data Tab ----
with t_data:
    st.subheader("Raw data")
    st.write(f"**Dataset path:** `{find_file_for_year(selected_year)}`")
    if df_primary.empty:
        st.info("No data loaded.")
    else:
        st.dataframe(df_primary, use_container_width=True, hide_index=True)
        csv = df_primary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered CSV",
            data=csv,
            file_name=f"crime_{selected_year}_filtered.csv",
            mime="text/csv",
        )
        with st.expander("Tips for working with this data"):
            st.markdown(
                """
                - Export CSVs and join with external datasets (e.g., demographics, governance) for deeper analysis.  
                - If column names differ across years, this app applies light normalization for 2023 for consistency.  
                """
            )

# ---- About Tab (NEW) ----
with t_about:
    st.header("About this topic")
    st.markdown(
        """
        This topic presents data from the **Global Organized Crime Index** (2021 & 2023).
        It enables you to:
        - Explore country-level scores across criminality, resilience, markets, and actors  
        - Compare regions and visualize distributions  
        - Inspect correlations between indicators  
        - Track changes across years for any metric

        **How scores are interpreted**
        - Higher **Criminality / Markets / Actors** → more pervasive organized crime
        - Higher **Resilience** → stronger capacity to counter organized crime

        **Notes**
        - “Drug trade” here is a convenience aggregate of heroin, cocaine, cannabis, and synthetic drug markets.
        - Some columns are lightly normalized for 2023 to ensure consistency across editions.

        **Data source:** Global Initiative Against Transnational Organized Crime — Global Organized Crime Index.
        """
    )

st.success("Dashboard ready. Use the sidebar to explore metrics, regions, maps, correlations, and year-over-year changes.")
