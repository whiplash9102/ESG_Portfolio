# strategies.py
import numpy as np
import pandas as pd

# Optional for Strategy B (min-variance)
try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -----------------------
# Small utilities
# -----------------------
def _normalize(w: pd.Series) -> pd.Series:
    w = w.clip(lower=0.0)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Weights sum to zero.")
    return w / s


def _pct_rank_centered(s: pd.Series) -> pd.Series:
    # percentile rank -> roughly [-0.5, +0.5]
    return s.rank(pct=True) - 0.5


def _compute_score_A(meta: pd.DataFrame, rets: pd.DataFrame, asof: pd.Timestamp, cfg: dict) -> pd.Series:
    """
    Score = w_esg * rank(esg) + w_mom * rank(momentum) + w_lowvol * rank(-vol)
    """
    esg = meta["esg"]

    hist = rets.loc[:asof]
    mom_lb = cfg["mom_lookback"]
    vol_lb = cfg["vol_lookback"]

    mom = (1.0 + hist.tail(mom_lb)).prod() - 1.0
    vol = hist.tail(vol_lb).std()

    df = pd.DataFrame({"esg": esg, "mom": mom, "vol": vol}).dropna()

    score = (
        cfg["w_esg"] * _pct_rank_centered(df["esg"])
        + cfg["w_mom"] * _pct_rank_centered(df["mom"])
        + cfg["w_lowvol"] * _pct_rank_centered(-df["vol"])
    )

    # return score aligned to available names
    return score.sort_values(ascending=False)


# -----------------------
# Strategy A: rotate K
# -----------------------
def target_weights_A(meta: pd.DataFrame, rets: pd.DataFrame, asof: pd.Timestamp, cfg: dict, w_current: pd.Series | None) -> pd.Series:
    """
    Defensive ESG + LowVol + Momentum (score-based), but ONLY rotate K names per rebalance.
    - Keeps N holdings
    - Sells K worst among held, buys K best among not-held
    - Equal weight, with max_weight cap (then renormalize)
    """
    N = int(cfg["n_holdings"])
    K = int(cfg.get("k_rotate", 6))  # <=8 recommended

    score = _compute_score_A(meta, rets, asof, cfg)  # high->low
    names = score.index

    # If first time (no current weights), just take top N
    if w_current is None or float(w_current.sum()) == 0.0:
        topN = names[:N]
        w = pd.Series(0.0, index=names)
        w.loc[topN] = 1.0 / N
        w = w.clip(upper=cfg["max_weight"])
        return _normalize(w)

    # Align current weights to available names
    w_cur = w_current.reindex(names).fillna(0.0)
    held = w_cur[w_cur > 0].index

    # If not enough held (data missing etc.), rebuild top N
    if len(held) < N:
        topN = names[:N]
        w = pd.Series(0.0, index=names)
        w.loc[topN] = 1.0 / N
        w = w.clip(upper=cfg["max_weight"])
        return _normalize(w)

    # SELL: K worst among held (lowest score)
    held_scores = score.reindex(held)
    sell = held_scores.sort_values(ascending=True).head(K).index

    # BUY: K best among not-held
    not_held = names.difference(held)
    buy = score.reindex(not_held).dropna().head(K).index

    new_holdings = held.difference(sell).union(buy)

    # Ensure exactly N names (in case union size drifts)
    new_holdings = score.reindex(new_holdings).dropna().head(N).index

    w = pd.Series(0.0, index=names)
    w.loc[new_holdings] = 1.0 / N
    w = w.clip(upper=cfg["max_weight"])
    return _normalize(w)


# -----------------------
# Strategy B: min-variance (sparse)
# -----------------------
def _shrink_cov(cov: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    diag = np.diag(np.diag(cov))
    return (1 - alpha) * cov + alpha * diag


def _min_var_weights(cov: np.ndarray, max_w: float) -> np.ndarray:
    n = cov.shape[0]
    x0 = np.ones(n) / n

    def obj(w):
        return float(w @ cov @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_w) for _ in range(n)]

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 300})
    if not res.success:
        return x0

    w = np.clip(res.x, 0.0, max_w)
    w = w / w.sum()
    return w


def target_weights_B(meta: pd.DataFrame, rets: pd.DataFrame, asof: pd.Timestamp, cfg: dict, w_current: pd.Series | None) -> pd.Series:
    """
    ESG best-in-class + minimum variance, then SPARSIFY to top N weights.
    If scipy not available, fallback to ESG+LowVol equal weight.
    """
    N = int(cfg["n_holdings"])
    max_w = float(cfg["max_weight"])
    cov_lb = int(cfg["cov_lookback"])

    hist = rets.loc[:asof]
    esg = meta["esg"].reindex(hist.columns).dropna()

    # Best-in-class filter: top 40%
    cut = esg.quantile(0.60)
    eligible = esg[esg >= cut].index

    # Candidates: lowest vol among eligible
    vol = hist[eligible].tail(cfg["vol_lookback"]).std().dropna()
    cand = vol.sort_values().head(max(N * 2, N)).index

    window = hist[cand].tail(cov_lb).dropna(axis=1, how="any")
    if window.shape[1] < 12:
        # fallback to equal-weight ESG top N
        top = esg.sort_values(ascending=False).head(N).index
        w = pd.Series(0.0, index=hist.columns)
        w.loc[top] = 1.0 / len(top)
        return _normalize(w)

    # If no scipy -> fallback to equal-weight low vol among eligible
    if not _HAS_SCIPY:
        top = vol.sort_values().head(N).index
        w = pd.Series(0.0, index=hist.columns)
        w.loc[top] = 1.0 / len(top)
        return _normalize(w)

    cov = window.cov().values
    cov = _shrink_cov(cov, alpha=0.10)
    w_mv = _min_var_weights(cov, max_w=max_w)

    w = pd.Series(0.0, index=hist.columns)
    w.loc[window.columns] = w_mv

    # SPARSIFY: keep only top N weights -> avoids micro-weights / micro-trades
    topN = w.sort_values(ascending=False).head(N).index
    w = w.where(w.index.isin(topN), 0.0)

    # normalize
    return _normalize(w)


# -----------------------
# Dispatcher
# -----------------------
def get_target_weights(meta: pd.DataFrame, rets: pd.DataFrame, asof: pd.Timestamp, cfg: dict, strategy_name: str, w_current: pd.Series | None = None) -> pd.Series:
    s = strategy_name.upper().strip()
    if s == "A":
        return target_weights_A(meta, rets, asof, cfg, w_current=w_current)
    if s == "B":
        return target_weights_B(meta, rets, asof, cfg, w_current=w_current)
    raise ValueError("strategy_name must be 'A' or 'B'")
