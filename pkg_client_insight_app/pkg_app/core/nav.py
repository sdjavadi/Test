"""Page registry and the hop from any card, chart or table to the shared customer page."""
import streamlit as st

from . import state
from .fmt import cust_key, norm_pwr

PAGES = {}
LENS_LABEL = {"attrition": "📉 Deposit attrition", "fraud": "🛡️ Suspicious activity", "network": "🕸️ Payment network"}


def register(name, page):
    PAGES[name] = page


def open_customer(key, lens=None, entity=None):
    """Go to the customer page for `key`, on the lens of the module the user came from."""
    st.session_state["nav_cust"] = key
    st.session_state["nav_lens"] = lens
    st.session_state["nav_entity"] = entity
    st.switch_page(PAGES["customer"], query_params={"c": key})


def go(name):
    st.switch_page(PAGES[name])


def resolve(key):
    """Everything the customer page needs to know about one customer key, from whichever modules know it."""
    cache = st.session_state.setdefault("_resolved", {})
    if key in cache: return cache[key]
    info = dict(key=key, name=None, pwr=None, mdm_ids=[], naics_desc=None, segment_desc=None, rltn_pwr_id=None, attr_cust=None)
    mods = state.allowed_modules()
    if "fraud" in mods:
        from ..fraud import service as FS
        r = FS.rows_for_key(key)
        if len(r):
            info["name"] = info["name"] or next((x for x in r.customer_name if isinstance(x, str) and x), None)
            info["mdm_ids"] += list(r.mdm_id.unique())
            info["pwr"] = info["pwr"] or next((norm_pwr(x) for x in r.customer_pwr_id if norm_pwr(x)), None)
            info["naics_desc"] = info["naics_desc"] or next((x for x in r.naics_desc if isinstance(x, str) and x), None)
    if "attrition" in mods:
        from ..attrition import service as AS
        a = AS.row_for_key(key)
        if a is not None:
            info["attr_cust"] = a["cust_pwr_id"]
            for k_ in ("customer_name", "naics_desc", "segment_desc", "rltn_pwr_id"):
                tgt = "name" if k_ == "customer_name" else k_
                if not info.get(tgt) and isinstance(a.get(k_), str) and a.get(k_): info[tgt] = a[k_]
            info["mdm_ids"] += [x for x in str(a.get("mdm_ids") or "").split(",") if x]
            info["pwr"] = info["pwr"] or norm_pwr(a["cust_pwr_id"])
    if not info["mdm_ids"] or not info["name"]:
        try:
            lk = state.source().lookup(key)
        except Exception:
            lk = None
        if lk:
            info["name"] = info["name"] or lk.get("name")
            info["mdm_ids"] += lk.get("mdm_ids") or []
            info["pwr"] = info["pwr"] or norm_pwr(lk.get("pwr"))
            info["naics_desc"] = info["naics_desc"] or lk.get("naics_desc")
            info["rltn_pwr_id"] = info["rltn_pwr_id"] or lk.get("master_pwr_id")
    if key.startswith("M:") and key[2:] not in info["mdm_ids"]: info["mdm_ids"].append(key[2:])
    info["mdm_ids"] = sorted(set(info["mdm_ids"]))
    cache[key] = info
    return info


def search(text):
    """Search any client (name contains / exact id) → list of (key, label)."""
    try:
        d = state.source().search_customers(text)
    except Exception as e:
        st.warning(f"Search is unavailable ({type(e).__name__}).")
        return []
    out = {}
    for r in d.itertuples():
        k = cust_key(getattr(r, "customer_pwr_id", None), r.mdm_id)
        if k and k not in out:
            out[k] = f"{r.customer_name or 'Unnamed'} · {getattr(r, 'naics_desc', '') or ''} · ref {k}"
    return list(out.items())
