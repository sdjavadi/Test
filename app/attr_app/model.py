"""Real-time scoring with the model bundle exported by the notebook."""
import numpy as np
import pandas as pd


def load_bundle(path):
    import joblib
    b = joblib.load(path)
    b["_booster"] = _booster(b)
    return b


def _booster(b):
    if b["engine"] == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_str=b["model"])
    if b["engine"] == "xgboost":
        import xgboost as xgb
        bst = xgb.Booster(); bst.load_model(bytearray(b["model"])); return bst
    raise ValueError(f"unsupported engine {b['engine']}")


def matrix(b, df):
    """Feature matrix in the bundle's column order; Impala lower-cases column names, so match case-insensitively."""
    low = {c.lower(): c for c in df.columns}
    cols = [low.get(f.lower()) for f in b["feats"]]
    X = np.full((len(df), len(cols)), np.nan, dtype=np.float32)
    for j, c in enumerate(cols):
        if c is not None: X[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float32)
    return X


def predict(b, X, contrib=False):
    bst = b["_booster"]
    if b["engine"] == "lightgbm":
        raw = bst.predict(X, num_iteration=b.get("n_iter"))
        C = bst.predict(X, num_iteration=b.get("n_iter"), pred_contrib=True)[:, :-1] if contrib else None
    else:
        import xgboost as xgb
        d = xgb.DMatrix(X, missing=np.nan); rng = (0, int(b.get("n_iter") or 0))
        raw = bst.predict(d, iteration_range=rng)
        C = bst.predict(d, iteration_range=rng, pred_contribs=True)[:, :-1] if contrib else None
    p = b["iso"].predict(raw) if b.get("iso") is not None else raw
    return np.clip(p, 0.0, b.get("p_cap", 0.95)), C


def rank_month(df, alpha):
    s = df.p * df.bal_norm.clip(lower=1.0) ** alpha
    return s.groupby(df.month).rank(ascending=False, method="first").astype(int)


def fired_matrix(b, df):
    """Event-time signals beyond the stayers' threshold (dict feature → (threshold, 'falls'|'rises'))."""
    out = {}
    low = {c.lower(): c for c in df.columns}
    for f, (th, d) in b.get("thr", {}).items():
        c = low.get(f.lower())
        if c is None: continue
        v = pd.to_numeric(df[c], errors="coerce")
        out[f] = (v < th) if d == "falls" else (v > th)
    return pd.DataFrame(out, index=df.index).fillna(False)
