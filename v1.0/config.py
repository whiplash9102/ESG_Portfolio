# config.py

CFG = {
    # universe
    "countries": ["France","Germany","Netherlands","Belgium","Luxembourg","Austria","Ireland"],
    "min_esg": 1e-9,

    # lookbacks (trading days)
    "mom_lookback": 20,
    "vol_lookback": 20,
    "cov_lookback": 60,
    "lookback_days_prices": 180,

    # portfolio constraints
    "n_holdings": 35,       # 12–80
    "max_weight": 0.05,
    "turnover_cap": 0.25,   # per Tue/Fri rebalance
    "tcost_rate": 20/10000, # 20 bps
    "k_rotate": 6, 

    # strategy A scoring weights
    "w_esg": 0.35,
    "w_mom": 0.35,
    "w_lowvol": 0.30,

    # stress test
    "horizon_days": 15,
    "bootstrap_paths": 8000,
    "bootstrap_block": 5,
}
