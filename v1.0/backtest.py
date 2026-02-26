# backtest.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

from strategies import get_target_weights

# ---------- helpers ----------
def _normalize(w: pd.Series) -> pd.Series:
    w = w.clip(lower=0.0)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Weights sum to zero.")
    return w / s

def get_rebalance_dates(dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
    # Tue=1, Fri=4
    return [d for d in dates if d.weekday() in (1, 4)]

def turnover(w_old: pd.Series, w_new: pd.Series) -> float:
    # 0.5*sum(|Δw|) for fully-invested portfolio
    u = w_old.index.union(w_new.index)
    a = w_old.reindex(u).fillna(0.0)
    b = w_new.reindex(u).fillna(0.0)
    return 0.5 * float((b - a).abs().sum())

def apply_turnover_cap(w_old: pd.Series, w_target: pd.Series, cap: float) -> pd.Series:
    u = w_old.index.union(w_target.index)
    a = w_old.reindex(u).fillna(0.0)
    b = w_target.reindex(u).fillna(0.0)

    to = turnover(a, b)
    if to <= cap + 1e-12:
        return _normalize(b)

    delta = b - a
    scale = cap / to
    w_new = a + scale * delta
    w_new = w_new.clip(lower=0.0)
    return _normalize(w_new)

def transaction_cost(w_old: pd.Series, w_new: pd.Series, tcost_rate: float) -> float:
    u = w_old.index.union(w_new.index)
    a = w_old.reindex(u).fillna(0.0)
    b = w_new.reindex(u).fillna(0.0)
    traded = float((b - a).abs().sum())
    return tcost_rate * traded

def count_trades(w_old: pd.Series, w_new: pd.Series, eps: float = 1e-6) -> int:
    u = w_old.index.union(w_new.index)
    a = w_old.reindex(u).fillna(0.0)
    b = w_new.reindex(u).fillna(0.0)
    return int(((b - a).abs() > eps).sum())

def force_min_two_changes(w_old: pd.Series, w_new: pd.Series) -> pd.Series:
    """
    Ensure at least 2 names change per Tue/Fri (as requested by brief).
    If not, do a tiny swap: move 0.5% from smallest holding to a non-held.
    """
    u = w_old.index.union(w_new.index)
    a = w_old.reindex(u).fillna(0.0)
    b = w_new.reindex(u).fillna(0.0)

    n_changed = int(((b - a).abs() > 1e-6).sum())
    if n_changed >= 2:
        return _normalize(b)

    w = b.copy()
    held = w[w > 0].sort_values()
    nonheld = w[w == 0].index

    if len(held) < 1 or len(nonheld) < 1:
        return _normalize(w)

    smallest = held.index[0]
    add = nonheld[0]
    shift = min(0.005, float(w.loc[smallest]) * 0.5)

    w.loc[smallest] -= shift
    w.loc[add] += shift
    return _normalize(w)

# ---------- result container ----------
@dataclass
class BacktestResult:
    nav: pd.Series
    daily_ret: pd.Series
    weights: Dict[pd.Timestamp, pd.Series]
    turnover: pd.Series
    costs: pd.Series
    trade_count: int

# ---------- main engine ----------
def run_backtest(rets: pd.DataFrame, meta: pd.DataFrame, cfg: dict, strategy_name: str = "A") -> BacktestResult:
    """
    rets: daily returns (date x instrument)
    meta: must include 'esg' column, index=instrument
    """
    dates = rets.index
    rebal_dates = get_rebalance_dates(dates)

    # start late enough so signals can be computed
    min_start_idx = max(cfg.get("cov_lookback", 60), cfg.get("mom_lookback", 20), cfg.get("vol_lookback", 20))
    min_start = dates[min_start_idx] if len(dates) > min_start_idx else dates[0]
    rebal_dates = [d for d in rebal_dates if d >= min_start]

    # initial weights: build at first rebalance date
    w = pd.Series(0.0, index=rets.columns)
    nav = 1.0

    nav_list = []
    ret_list = []
    weights_hist: Dict[pd.Timestamp, pd.Series] = {}
    turnover_hist: Dict[pd.Timestamp, float] = {}
    costs_hist: Dict[pd.Timestamp, float] = {}
    trade_count = 0

    current_w = None

    for d in dates:
        # rebalance
        if d in rebal_dates:
            if current_w is None:
                current_w = w.copy()

            w_target = get_target_weights(meta, rets, d, cfg, strategy_name, w_current=current_w)
            # align to universe columns
            w_target = w_target.reindex(rets.columns).fillna(0.0)
            w_target = _normalize(w_target)

            # enforce min turnover rule (>=2 changes), then turnover cap
            w_target = force_min_two_changes(current_w, w_target)
            w_new = apply_turnover_cap(current_w, w_target, cap=cfg["turnover_cap"])

            # costs + stats
            to = turnover(current_w, w_new)
            c = transaction_cost(current_w, w_new, cfg["tcost_rate"])
            trades = count_trades(current_w, w_new)

            turnover_hist[d] = to
            costs_hist[d] = c
            trade_count += trades

            current_w = w_new
            weights_hist[d] = current_w.copy()

        # daily pnl
        if current_w is None:
            nav_list.append(nav)
            ret_list.append(0.0)
            continue

        r = rets.loc[d].fillna(0.0)
        port_ret = float((current_w * r).sum())

        # apply trading cost on rebalance day
        if d in costs_hist:
            port_ret -= costs_hist[d]

        nav *= (1.0 + port_ret)
        nav_list.append(nav)
        ret_list.append(port_ret)

    nav_s = pd.Series(nav_list, index=dates, name=f"NAV_{strategy_name}")
    ret_s = pd.Series(ret_list, index=dates, name=f"RET_{strategy_name}")

    return BacktestResult(
        nav=nav_s,
        daily_ret=ret_s,
        weights=weights_hist,
        turnover=pd.Series(turnover_hist).sort_index(),
        costs=pd.Series(costs_hist).sort_index(),
        trade_count=trade_count
    )
