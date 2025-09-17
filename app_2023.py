
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Global OC Index — 2023 Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Data helpers ----------
@st.cache_data(show_spinner=False)
def load_2023_sheet(path: str):
    # Load all sheets
    all_sheets = pd.read_excel(path, sheet_name=None)
    # Find a sheet whose name mentions 2023 (case-insensitive)
    pick = None
    for name in all_sheets.keys():
        if "2023" in str(name):
            pick = name
            break
    # Fallback: first sheet if none matched
    if pick is None:
        pick = list(all_sheets.keys())[0]
    df = all_sheets[pick].copy()
    return df, pick, list(all_sheets.keys())

def drop_indexish_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in df.columns if str(c).startswith("Unnamed")]
    out = df.drop(columns=cols_to_drop, errors="ignore")
    all_nan = [c for c in out.columns if out[c].isna().all()]
    out = out.drop(columns=all_nan, errors="ignore")
    return out

def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        lc = str(c).lower()
        if lc in ["date","timestamp","period"]:
            try:
                out[c] = pd.to_datetime(out[c], errors="coerce")
            except Exception:
                pass
        if lc == "year" and pd.api.types.is_integer_dtype(out[c]):
            out["year_datetime"] = pd.to_datetime(out[c].astype(str) + "-01-01", errors="coerce")
    return out

def choose_default_metric(df: pd.DataFrame):
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in numeric if not str(c).lower().startswith("unnamed")]
    if not numeric:
        return None
    scored = []
    for c in numeric:
        s = df[c]
        unique_ratio = s.nunique(dropna=True) / max(len(s), 1)
        scored.append((unique_ratio, c))
    scored.sort(reverse=True)
    return scored[0][1]

# ---------- UI ----------
st.title("Global OC Index — 2023 Dashboard")
st.caption("This app is locked to the 2023 sheet.")

data_path = st.text_input("Data file path", value="global_oc_index.xlsx", help="Path to the Excel file that contains the 2023 sheet.")
try:
    raw_df, picked_sheet, all_sheet_names = load_2023_sheet(data_path)
    st.text(f"Loaded sheet: {picked_sheet}")
except Exception as e:
    st.error(f"Could not load Excel file at '{data_path}'. Details: {e}")
    st.stop()

raw_df = drop_indexish_columns(raw_df)
df = coerce_dates(raw_df)

# ---------- Sidebar Filters ----------
with st.sidebar:
    st.subheader("Filters (2023)")

    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if not str(c).lower().startswith("unnamed")]
    cat_cols = [c for c in cols if c not in numeric_cols]

    date_like = []
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_like.append(c)
        elif str(c).lower() in ["date","timestamp","period","year_datetime"]:
            try:
                _ = pd.to_datetime(df[c], errors="coerce")
                date_like.append(c)
            except Exception:
                pass

    default_metric = choose_default_metric(df)
    metric = st.selectbox(
        "Primary metric",
        numeric_cols,
        index=(numeric_cols.index(default_metric) if default_metric in numeric_cols else 0) if numeric_cols else None,
        placeholder="Select a numeric metric"
    )

    filter_values = {}
    for c in cat_cols:
        nunique = df[c].nunique(dropna=True)
        if 1 < nunique <= 40:
            opts = ["(All)"] + sorted([str(x) for x in df[c].dropna().unique().tolist()])
            sel = st.multiselect(f"{c}", opts, default="(All)")
            if sel and "(All)" not in sel:
                filter_values[c] = sel

    date_col = st.selectbox("Date/Period column", date_like, index=0 if date_like else None, placeholder="Select date column")
    start_date, end_date = None, None
    if date_col:
        valid_dates = pd.to_datetime(df[date_col], errors="coerce")
        if valid_dates.notna().any():
            min_d = valid_dates.min()
            max_d = valid_dates.max()
            start_date, end_date = st.date_input("Date range", [min_d.date(), max_d.date()])

# Apply filters
fdf = df.copy()
for c, sel in filter_values.items():
    fdf = fdf[fdf[c].astype(str).isin(sel)]
if date_col and start_date and end_date and date_col in fdf.columns:
    vd = pd.to_datetime(fdf[date_col], errors="coerce")
    mask = (vd >= pd.to_datetime(start_date)) & (vd <= pd.to_datetime(end_date))
    fdf = fdf[mask]

# ---------- KPIs ----------
st.markdown("### Key Metrics")
kpi_cols = st.columns(4)
if metric and metric in fdf.columns and pd.api.types.is_numeric_dtype(fdf[metric]) and fdf[metric].notna().any():
    total = float(np.nansum(fdf[metric]))
    avg = float(np.nanmean(fdf[metric]))
    med = float(np.nanmedian(fdf[metric]))
    count_rows = int(len(fdf))
    with kpi_cols[0]:
        st.metric(f"Total {metric}", f"{total:,.2f}")
    with kpi_cols[1]:
        st.metric(f"Average {metric}", f"{avg:,.2f}")
    with kpi_cols[2]:
        st.metric(f"Median {metric}", f"{med:,.2f}")
    with kpi_cols[3]:
        st.metric("Row Count", f"{count_rows:,}")
else:
    st.info("Select a numeric primary metric in the sidebar to populate KPIs.")

st.divider()

# ---------- Visuals ----------
tab1, tab2, tab3, tab4 = st.tabs(["Trend", "Ranking", "Distribution", "Geo"])

with tab1:
    st.subheader("Time Trend")
    if metric and date_col and date_col in fdf.columns:
        temp = fdf.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col])
        grp = st.selectbox(
            "Group by (optional)",
            [None] + [c for c in temp.columns if c not in [metric, date_col] and not pd.api.types.is_numeric_dtype(temp[c])],
            key="trend_groupby"
        )
        if grp:
            fig = px.line(temp, x=date_col, y=metric, color=grp, markers=True)
        else:
            fig = px.line(temp, x=date_col, y=metric, markers=True)
        st.plotly_chart(fig, use_container_width=True, key="trend_chart")
    else:
        st.info("Pick a date/period column and a metric to see trend.")

with tab2:
    st.subheader("Top/Bottom Ranking")
    segment_options = [c for c in fdf.columns if c not in [metric] and not pd.api.types.is_numeric_dtype(fdf[c])]
    segment = st.selectbox("Segment by", segment_options, index=0 if segment_options else None, key="rank_segment", placeholder="Choose a categorical column")
    n = st.slider("How many to show", 3, 30, 10, key="rank_n")
    aggfunc = st.selectbox("Aggregate", ["sum","mean","median"], index=0, key="rank_agg")
    if metric and segment:
        grouped = getattr(fdf.groupby(segment)[metric], aggfunc)().reset_index().sort_values(metric, ascending=False)
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Top**")
            st.plotly_chart(px.bar(grouped.head(n), x=segment, y=metric),
                            use_container_width=True, key="top_bar_chart")
        with colB:
            st.markdown("**Bottom**")
            st.plotly_chart(px.bar(grouped.tail(n).sort_values(metric, ascending=True), x=segment, y=metric),
                            use_container_width=True, key="bottom_bar_chart")
    else:
        st.info("Select a categorical segment and metric.")

with tab3:
    st.subheader("Distribution & Relationships")
    col1, col2 = st.columns(2)
    if metric:
        with col1:
            st.markdown("**Histogram**")
            st.plotly_chart(px.histogram(fdf, x=metric, nbins=30),
                            use_container_width=True, key="hist_chart")
        with col2:
            st.markdown("**Box Plot by Category**")
            cat_opts = [c for c in fdf.columns if c != metric and not pd.api.types.is_numeric_dtype(fdf[c])]
            cat = st.selectbox("Category for box plot", cat_opts, index=0 if cat_opts else None, key="boxcat")
            if cat:
                st.plotly_chart(px.box(fdf, x=cat, y=metric, points="suspectedoutliers"),
                                use_container_width=True, key="box_chart")
    num_cols = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])]
    if len(num_cols) >= 2:
        st.markdown("**Scatter**")
        xcol = st.selectbox("X", num_cols, index=0, key="scatter_x")
        ycol = st.selectbox("Y", num_cols, index=1 if len(num_cols) > 1 else 0, key="scatter_y")
        color = st.selectbox("Color (optional)", [None] + [c for c in fdf.columns if not pd.api.types.is_numeric_dtype(fdf[c])], key="scatter_color")
        fig = px.scatter(fdf, x=xcol, y=ycol, color=color) if color else px.scatter(fdf, x=xcol, y=ycol)
        st.plotly_chart(fig, use_container_width=True, key="scatter_chart")

with tab4:
    st.subheader("Geospatial (country-level)")
    country_col = None
    for candidate in ["country","nation","region"]:
        for c in fdf.columns:
            if str(c).lower() == candidate:
                country_col = c
                break
        if country_col:
            break
    if metric and country_col:
        fig = px.choropleth(fdf, locations=country_col, locationmode="country names", color=metric, color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True, key="map_chart")
    else:
        st.info("To enable the map, include a column named 'country' (country names) and select a numeric metric.")

st.divider()
st.subheader("Data Explorer")
st.dataframe(fdf, use_container_width=True)

st.download_button("Download filtered data as CSV", data=fdf.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_data_2023.csv", mime="text/csv")

st.caption("Locked to the 2023 sheet. All charts keyed uniquely to avoid duplicate ID errors.")
