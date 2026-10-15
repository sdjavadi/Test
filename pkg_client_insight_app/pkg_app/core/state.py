"""Settings, the data source and who is looking — shared by every page."""
import json

import streamlit as st

from ..config import settings

CFG = settings()
MODE = CFG["mode"]
MODULES = {"attrition": "Deposit attrition", "fraud": "Suspicious activity", "network": "Payment network"}
# "View as" presets: leadership can preview what each audience would see. A preset can only narrow what the viewer is
# already allowed to see (never widen it), so it is safe to leave on.
ROLES = {"Leadership (all modules)": ["attrition", "fraud", "network"],
         "Relationship manager": ["attrition", "network"],
         "Compliance / AML": ["fraud", "network"]}


@st.cache_resource(show_spinner="Connecting to the data…")
def _source(mode):
    if mode == "demo":
        from .demo import DemoSource
        return DemoSource()
    from .data import ImpalaSource
    return ImpalaSource(CFG)


def source():
    return _source(MODE)


def current_user():
    """Posit Connect passes the signed-in user and groups in the RStudio-Connect-Credentials header."""
    try:
        raw = st.context.headers.get("Rstudio-Connect-Credentials") or st.context.headers.get("RStudio-Connect-Credentials")
        if raw:
            d = json.loads(raw)
            return dict(user=d.get("user") or "", groups=list(d.get("groups") or []))
    except Exception:
        pass
    return dict(user="", groups=[])


def allowed_modules():
    """Modules this viewer may open: config `access` maps module → users / groups (empty list = everyone), then the
    optional "View as" preset narrows it."""
    u = current_user(); who = {u["user"], *u["groups"]} - {""}
    rules = CFG.get("access", {}) or {}
    ok = [m for m in MODULES if not rules.get(m) or who & set(rules.get(m, []))]
    role = st.session_state.get("view_as")
    if role in ROLES: ok = [m for m in ok if m in ROLES[role]]
    return ok


def can(module):
    return module in allowed_modules()
