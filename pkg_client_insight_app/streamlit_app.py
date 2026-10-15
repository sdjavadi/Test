"""PKG client insight — one app, several modules (deposit attrition, suspicious activity, payment network) and one shared
customer page.  Run:  streamlit run streamlit_app.py   (PKG_APP_MODE=demo for synthetic data)"""
import os

import streamlit as st

st.set_page_config(page_title="PKG client insight", page_icon="🧭", layout="wide")

from pkg_app import customer, home  # noqa: E402
from pkg_app.attrition import pages as attrition  # noqa: E402
from pkg_app.core import nav, state, ui  # noqa: E402
from pkg_app.fraud import pages as fraud  # noqa: E402
from pkg_app.network import pages as network  # noqa: E402

ui.inject_css()
mods = state.allowed_modules()
start = os.getenv("PKG_APP_START", "home")                     # landing page (home, attrition, fraud, network, customer)
spec = [("home", home.page, "Overview", "🧭", "overview"),
        ("attrition", attrition.cards_page, "Deposit attrition", "📉", "attrition"),
        ("fraud", fraud.cards_page, "Suspicious activity", "🛡️", "suspicious-activity"),
        ("network", network.explorer_page, "Payment network", "🕸️", "network"),
        ("customer", customer.page, "Customer", "👤", "customer")]
spec = [s for s in spec if s[0] in ("home", "customer") or s[0] in mods]
if start not in [s[0] for s in spec]: start = "home"
pages = {k: st.Page(fn, title=t, icon=i, url_path=u, default=(k == start)) for k, fn, t, i, u in spec}
for name, page in pages.items():
    nav.register(name, page)
st.navigation(list(pages.values()), position="top").run()
