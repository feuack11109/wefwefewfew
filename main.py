# main.py — Switcher for your dashboards (no more first-load error)

from __future__ import annotations
import os
from pathlib import Path
import contextlib
import runpy
import streamlit as st

# --- 1) MUST be first Streamlit command in the entire run
st.set_page_config(
    page_title="Unified Analytics Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2) Make any subsequent calls from sub-apps harmless (no-op)
def _noop_set_page_config(*args, **kwargs):
    # We've already set the page config in main.py
    return None

st.set_page_config = _noop_set_page_config  # type: ignore[attr-defined]

# --- 3) UI switcher (now safe to use Streamlit UI)
choice = st.sidebar.selectbox(
    "Choose a Topic",
    ["Global Organized Crime Index", "Global Terrorism Statistic", "Crime Indicators Explorer"],
    index=0,
)

BASE = Path(__file__).resolve().parent
APP_MAP = {
    "Crime": BASE / "crime.py",
    "Terrorism": BASE / "terrorism.py",
    "Sub_Crime_Analytics": BASE / "sub.py",
}

target = APP_MAP[choice]
if not target.exists():
    st.error(f"Could not find `{target.name}` at: {target}")
    st.stop()


@contextlib.contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))

with _pushd(target.parent):
    # Run the selected app "as main" so its if __name__ == "__main__" blocks work
    runpy.run_path(str(target), run_name="__main__")
