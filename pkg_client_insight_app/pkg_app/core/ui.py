"""Look and feel shared by every page: palette, CSS, chart defaults, banners and the page header."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import state

NAVY, BLUE, TEAL, ORANGE, RED, GREY, VIOLET, GREEN, LIGHT = ("#0b2545", "#1d4e89", "#12a39a", "#f28c28", "#d1495b", "#8d99ae",
                                                            "#6a4c93", "#2a9d8f", "#f3f6fa")
TIER_COLOR = {"High": "red", "Elevated": "orange", "Watch": "gray"}
FONT = "'IBM Plex Sans', 'Source Sans Pro', -apple-system, 'Segoe UI', sans-serif"

CSS = f"""<style>
.block-container {{ padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1500px; }}
h1 {{ font-weight: 700; color: {NAVY}; letter-spacing: -.01em; }}
.pg-sub {{ color: #5b6876; font-size: .95rem; margin: -.4rem 0 .6rem 0; }}
.card-name {{ font-size: 1.05rem; font-weight: 700; color: {NAVY}; line-height: 1.2; }}
.card-sub {{ color: #6b7785; font-size: .8rem; margin-bottom: .25rem; }}
.why li {{ margin: .1rem 0; font-size: .88rem; }}
.next {{ background: {LIGHT}; border-radius: 8px; padding: .5rem .7rem; font-size: .86rem; margin-top: .3rem; }}
.alt {{ color: #5b6876; font-size: .8rem; border-top: 1px dashed #d5dbe3; padding-top: .35rem; margin-top: .35rem; }}
.play {{ background: #e8f6f4; border-left: 4px solid {TEAL}; border-radius: 6px; padding: .55rem .8rem; margin: .4rem 0; }}
.comp {{ background: #fff6e8; border-left: 4px solid {ORANGE}; border-radius: 6px; padding: .5rem .8rem; margin: .2rem 0 .7rem 0;
         font-size: .86rem; color: #5a4220; }}
.tile-h {{ font-size: 1.15rem; font-weight: 700; color: {NAVY}; }}
.tile-p {{ color: #5b6876; font-size: .88rem; min-height: 2.6rem; }}
.kv {{ font-size: .86rem; color: #3b4652; }} .kv b {{ color: {NAVY}; }}
.ev {{ font-size: .88rem; margin: .1rem 0 .1rem 0; }}
.muted {{ color: #6b7785; font-size: .82rem; }}
div[data-testid="stMetricValue"] {{ font-size: 1.35rem; }}
</style>"""


def inject_css():
    st.html(CSS)


def base_fig(fig, h=320, title=None, legend=None):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=40, r=20, t=40 if title else 10, b=30), title=title,
                      legend=dict(orientation="h", y=-0.2), font=dict(size=12, family=FONT))
    if legend is not None: fig.update_layout(showlegend=legend)
    return fig


def spark_in_out(h, key, in_col="amt_all_in", out_col="amt_all_out"):
    """Tiny chart of money in (teal) and out (orange) per month."""
    fig = go.Figure()
    fig.add_bar(x=h.month, y=h[in_col] / 1e6, marker_color="rgba(18,163,154,.75)", name="in", hovertemplate="%{x} in: $%{y:,.2f}m<extra></extra>")
    fig.add_bar(x=h.month, y=-h[out_col] / 1e6, marker_color="rgba(242,140,40,.75)", name="out",
                hovertemplate="%{x} out: $%{customdata:,.2f}m<extra></extra>", customdata=h[out_col] / 1e6)
    fig.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False),
                      showlegend=False, template="plotly_white", barmode="relative", bargap=.15)
    st.plotly_chart(fig, config={"displayModeBar": False}, key=key)


def compliance_note(short=False):
    txt = ("Patterns worth a review — <b>not evidence of wrongdoing</b>. Do not discuss with the client; route concerns to "
           "Compliance / AML.") if short else (
        "<b>For internal review only.</b> These are payment patterns that resemble known money-laundering typologies; most have an "
        "ordinary business explanation. A pattern is <b>not evidence of wrongdoing</b> and is not a suspicious-activity "
        "determination. <b>Do not discuss it with the client.</b> Concerns go to Compliance / AML through the usual referral route.")
    st.markdown(f"<div class='comp'>⚠️ {txt}</div>", unsafe_allow_html=True)


def page_header(title, sub=None, icon=None):
    """Title on the left; data mode and the "View as" preview on the right."""
    c1, c2 = st.columns([5, 1.25], vertical_alignment="bottom")
    with c1:
        st.title(f"{icon} {title}" if icon else title)
        if sub: st.markdown(f"<div class='pg-sub'>{sub}</div>", unsafe_allow_html=True)
    with c2:
        with st.popover(f"👤 {st.session_state.get('view_as') or 'View as…'}", width="stretch"):
            u = state.current_user()
            st.caption(f"Signed in as **{u['user']}**" if u["user"] else "Not signed in through Posit Connect (local run)")
            st.caption(f"Data: {'synthetic demo' if state.MODE == 'demo' else 'Impala · ' + state.CFG['database']}")
            opts = ["Everything I can access"] + list(state.ROLES)
            cur = st.session_state.get("view_as")
            pick = st.radio("Preview the app as", opts, index=opts.index(cur) if cur in opts else 0, key="view_as_pick",
                            help="Shows what each audience would see. It can only hide modules, never reveal more than your access.")
            new = None if pick == opts[0] else pick
            if new != cur:
                st.session_state.view_as = new; st.rerun()


def empty_state(msg, icon="ℹ️"):
    st.info(msg, icon=icon)


def kpis(items):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        label, value = it[0], it[1]
        c.metric(label, value, help=it[2] if len(it) > 2 else None)


def money_col(df, cols, fmt):
    return df.assign(**{c: df[c].map(fmt) for c in cols if c in df.columns})


def is_txt(v):
    return isinstance(v, str) and v.strip() != "" and v.lower() not in ("nan", "none")


def first_txt(*vals, default=""):
    for v in vals:
        if is_txt(v): return v
    return default


def ok_num(v):
    return v is not None and not pd.isna(v)
