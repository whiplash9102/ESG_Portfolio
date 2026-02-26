import pandas as pd
import numpy as np
import yfinance as yf

# ========= 1) Tickers + weights (từ screenshot của bạn) =========
tickers = ["MSI","MO","O","MCD","PG","SO","GE","PVH","TRV","DG","APAM","USFD"]

market_values_eur = {
    "MSI": 5127202.18,
    "MO": 5763912.65,
    "O": 5087248.92,
    "MCD": 6177011.08,
    "PG": 6316127.33,
    "SO": 6103066.08,
    "GE": 6282228.64,
    "PVH": 8963539.20,
    "TRV": 5207148.28,
    "DG": 9047760.75,
    "APAM": 5290395.70,
    "USFD": 4767310.01,
}

w = pd.Series(market_values_eur).reindex(tickers)
w = w / w.sum()
print("Weights (%):")
print((w * 100).round(2).sort_values(ascending=False))

# ========= 2) Download prices =========
# 6mo đủ để bạn tính mom_3m + risk + drawdown; đổi thành "1y" nếu muốn
px = yf.download(tickers, period="6mo", interval="1d", auto_adjust=True)["Close"]
px = px.dropna(how="all")

# nếu có ticker thiếu dữ liệu -> loại ra để tránh NaN lan
available = [t for t in tickers if t in px.columns and px[t].dropna().shape[0] > 20]
px = px[available]
w = w.reindex(available)
w = w / w.sum()

ret = px.pct_change().dropna()

# ========= 3) Portfolio return series =========
port_ret = (ret * w).sum(axis=1)

# ========= 4) Metrics helpers =========
def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return float(dd.min())

def ann_return(r: pd.Series, periods=252) -> float:
    # geometric annualization
    wealth = (1 + r).prod()
    n = r.shape[0]
    return float(wealth ** (periods / n) - 1)

def ann_vol(r: pd.Series, periods=252) -> float:
    return float(r.std() * np.sqrt(periods))

def sharpe(r: pd.Series, rf_annual=0.0, periods=252) -> float:
    rf_daily = (1 + rf_annual) ** (1 / periods) - 1
    excess = r - rf_daily
    denom = excess.std()
    return float(np.nan) if denom == 0 else float(excess.mean() / denom * np.sqrt(periods))

def sortino(r: pd.Series, rf_annual=0.0, periods=252) -> float:
    rf_daily = (1 + rf_annual) ** (1 / periods) - 1
    excess = r - rf_daily
    downside = excess[excess < 0]
    denom = downside.std()
    return float(np.nan) if denom == 0 else float(excess.mean() / denom * np.sqrt(periods))

def calmar(r: pd.Series, periods=252) -> float:
    ar = ann_return(r, periods)
    mdd = abs(max_drawdown(r))
    return float(np.nan) if mdd == 0 else float(ar / mdd)

# ========= 5) Print portfolio stats =========
stats = {
    "obs_days": int(port_ret.shape[0]),
    "cum_return": float((1 + port_ret).prod() - 1),
    "ann_return": ann_return(port_ret),
    "ann_vol": ann_vol(port_ret),
    "sharpe(0% rf)": sharpe(port_ret, rf_annual=0.0),
    "sortino(0% rf)": sortino(port_ret, rf_annual=0.0),
    "max_drawdown": max_drawdown(port_ret),
    "calmar": calmar(port_ret),
}
print("\nPortfolio stats:")
for k, v in stats.items():
    if "days" in k:
        print(f"{k:>15}: {v}")
    else:
        print(f"{k:>15}: {v:.4f}")

# ========= 6) (Bonus) Risk contribution theo covariance =========
cov = ret.cov() * 252
port_var = float(w.T @ cov @ w)
marginal = cov @ w
risk_contrib = (w * marginal) / port_var

rc_table = pd.DataFrame({
    "weight": w,
    "risk_contrib": risk_contrib
}).sort_values("risk_contrib", ascending=False)

print("\nRisk contribution vs Weight:")
print((rc_table * 100).round(2))  # % form

# ========= 7) (Optional) show top correlations =========
corr = ret.corr()
pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
             .stack()
             .sort_values(ascending=False))
print("\nTop correlations (>|0.75|):")
print(pairs[pairs.abs() > 0.75].head(20))