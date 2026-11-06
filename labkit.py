"""
labkit.py: shared helpers for the graph ML workshop labs.

Loading the synthetic data and drawing networks. The graph ideas themselves
live in the notebooks; this file only keeps plotting code out of the way.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:          # the labs need plotly for interactive figures
    go = None


def use_renderer(name=None):
    """Choose how plotly figures appear in the notebook.

    "iframe" (default) writes each figure to a small HTML file with plotly.js inside
    (folder iframe_figures/ next to the notebook) and shows it in an iframe. It works
    offline and without the JupyterLab plotly extension, which is the usual reason
    figures come out blank. Other choices: "iframe_connected" (plotly.js from the
    internet), "notebook", "jupyterlab", "png" (static, needs kaleido).
    Override for all labs with the environment variable PAYGRAPH_PLOTLY.
    """
    import time
    import plotly.io as pio
    name = name or os.environ.get("PAYGRAPH_PLOTLY", "iframe")
    pio.renderers.default = name
    # one folder per kernel session, so two labs never overwrite each other's figures
    session = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    for r in ("iframe", "iframe_connected"):
        pio.renderers[r].html_directory = os.path.join("iframe_figures", session)
    return name


if go is not None:
    try:
        RENDERER = use_renderer()
    except Exception:            # an unknown renderer name: keep plotly's default
        RENDERER = None

# colours (validated categorical palette, fixed order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#a3a29b"
EDGE = "#c9c8c0"
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
IN_COLOR, OUT_COLOR = "#2a78d6", "#eb6834"        # money in = blue, money out = orange

DATA = os.environ.get("PAYGRAPH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
MONTHS = [str(p) for p in pd.period_range("2025-09", "2026-08", freq="M")]


def load(truth: bool = False):
    """Return nodes, edges_monthly (and truth if asked). Truth holds the planted answers."""
    nodes = pd.read_csv(os.path.join(DATA, "nodes.csv"), dtype={"sector_code": str}).fillna({"sector": "", "sector_code": ""})
    edges = pd.read_csv(os.path.join(DATA, "edges_monthly.csv"))
    if truth:
        t = pd.read_csv(os.path.join(DATA, "truth.csv")).fillna({"vendor_scenario": "", "silent_from_month": ""})
        return nodes, edges, t
    return nodes, edges


def load_transactions():
    return pd.read_csv(os.path.join(DATA, "transactions_2026-08.csv"), parse_dates=["date"])


def load_vendor_payments():
    return pd.read_csv(os.path.join(DATA, "vendor_payments.csv"))


def payment_graph(edges: pd.DataFrame, months=None, directed=True, drop=(), weight="amount"):
    """Sum payments over `months` into one graph. Edge attributes: amount, n_txn, weight = log(1 + amount)."""
    e = edges if months is None else edges[edges.month.isin(months)]
    e = e[~e.src.isin(drop) & ~e.dst.isin(drop)]
    g = e.groupby(["src", "dst"], as_index=False)[["amount", "n_txn"]].sum()
    G = nx.DiGraph()
    G.add_weighted_edges_from(zip(g.src, g.dst, np.log1p(g.amount)), weight="weight")
    nx.set_edge_attributes(G, dict(zip(zip(g.src, g.dst), g.amount)), "amount")
    nx.set_edge_attributes(G, dict(zip(zip(g.src, g.dst), g.n_txn)), "n_txn")
    if directed:
        return G
    U = nx.Graph()
    for u, v, d in G.edges(data=True):
        if U.has_edge(u, v):
            U[u][v]["amount"] += d["amount"]
            U[u][v]["n_txn"] += d["n_txn"]
        else:
            U.add_edge(u, v, amount=d["amount"], n_txn=d["n_txn"])
    for u, v, d in U.edges(data=True):
        d["weight"] = float(np.log1p(d["amount"]))
    return U


def usd(x):
    x = float(x)
    for unit, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= d:
            return f"USD {x / d:,.1f}{unit}"
    return f"USD {x:,.0f}"


def colour_map(values, top=len(PALETTE)):
    """Fixed-order colours for the most common categories, grey for the rest."""
    vc = pd.Series(list(values)).value_counts()
    keys = [k for k in vc.index if k not in ("", None)][:top]
    return {k: PALETTE[i] for i, k in enumerate(keys)}


def _layout(fig, title, height=650, legend=True):
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=17, color=INK)),
                      plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, height=height,
                      font=dict(family="Inter, Helvetica, Arial, sans-serif", color=INK),
                      margin=dict(l=10, r=10, t=55, b=10), showlegend=legend,
                      legend=dict(orientation="h", y=-0.02, x=0.0))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return fig


def draw(G, pos=None, color=None, size=None, hover=None, title="", group=None, height=650,
         edge_width=0.6, seed=7, legend=True, arrows=False, edge_colors=None, labels=None):
    """Plotly network. color/size/hover: dicts node -> value. group: dict node -> legend name."""
    pos = pos or nx.spring_layout(G, seed=seed)
    traces = []
    buckets = {}
    for u, v in G.edges():
        c = (edge_colors or {}).get((u, v), EDGE)
        buckets.setdefault(c, ([], []))
        xs, ys = buckets[c]
        xs += [pos[u][0], pos[v][0], None]
        ys += [pos[u][1], pos[v][1], None]
    for c, (xs, ys) in buckets.items():
        traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=edge_width, color=c),
                                 hoverinfo="skip", showlegend=False))
    nodes = list(G.nodes())
    groups = {}
    for n_ in nodes:
        groups.setdefault((group or {}).get(n_, "nodes"), []).append(n_)
    for gname, members in groups.items():
        traces.append(go.Scatter(
            x=[pos[n_][0] for n_ in members], y=[pos[n_][1] for n_ in members],
            mode="markers+text" if labels else "markers",
            text=[(labels or {}).get(n_, "") for n_ in members], textposition="top center",
            textfont=dict(size=12, color=INK),
            marker=dict(size=[(size or {}).get(n_, 7) for n_ in members],
                        color=[(color or {}).get(n_, PALETTE[0]) for n_ in members],
                        line=dict(width=1, color=SURFACE)),
            name=str(gname), hovertext=[(hover or {}).get(n_, str(n_)) for n_ in members], hoverinfo="text",
            showlegend=legend and group is not None))
    fig = go.Figure(traces)
    if arrows:
        for u, v in G.edges():
            fig.add_annotation(x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2,
                               arrowwidth=1.2, arrowcolor=(edge_colors or {}).get((u, v), "#8d8c85"),
                               standoff=8, startstandoff=6)
    return _layout(fig, title, height, legend)


def bar(x, y, title, xlabel="", ylabel="", color=PALETTE[0], height=420, text=None, horizontal=False):
    fig = go.Figure(go.Bar(x=y if horizontal else x, y=x if horizontal else y,
                           orientation="h" if horizontal else "v", marker=dict(color=color),
                           text=text, textposition="outside"))
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=17)), height=height,
                      plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, margin=dict(l=10, r=10, t=55, b=10),
                      xaxis_title=xlabel, yaxis_title=ylabel, font=dict(color=INK))
    fig.update_xaxes(gridcolor="#e7e6e1")
    fig.update_yaxes(gridcolor="#e7e6e1")
    return fig


def style(fig, title, height=450):
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=17)), height=height,
                      plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, margin=dict(l=10, r=10, t=55, b=10),
                      font=dict(color=INK), legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(gridcolor="#e7e6e1", zerolinecolor="#d6d5cf")
    fig.update_yaxes(gridcolor="#e7e6e1", zerolinecolor="#d6d5cf")
    return fig


def ego_layout(G, center, r=1.0):
    """Payers on the left, payees on the right, partners on both sides at the bottom."""
    preds = set(G.predecessors(center)) if G.is_directed() else set()
    succs = set(G.successors(center)) if G.is_directed() else set(G.neighbors(center))
    both, ins, outs = sorted(preds & succs), sorted(preds - succs), sorted(succs - preds)
    pos = {center: np.array([0.0, 0.0])}

    def arc(items, a0, a1):
        for i, n_ in enumerate(items):
            ang = a0 + (a1 - a0) * (i + 1) / (len(items) + 1)
            pos[n_] = np.array([r * np.cos(ang), r * np.sin(ang)])

    arc(ins, np.pi * 0.55, np.pi * 1.25)
    arc(both, np.pi * 1.30, np.pi * 1.70)
    arc(outs, np.pi * 1.75, np.pi * 2.45)
    return pos


def find_hubs(edges, factor=15, floor=400):
    """Hub rule used in every lab: degree above `factor` x median business degree and above `floor` partners."""
    G = payment_graph(edges, directed=False)
    deg = pd.Series(dict(G.degree()))
    b = deg[[n for n in deg.index if str(n).startswith("B")]]
    return sorted(b[(b > factor * b.median()) & (b > floor)].index, key=lambda n: -deg[n])
