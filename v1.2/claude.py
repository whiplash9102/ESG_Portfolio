"""
=============================================================================
ESG Portfolio Management Script
MSc FDA – ESSCA 2026 | Momentum + Mean-Reversion + ESG Strategy
=============================================================================

REQUIREMENTS:
    pip install pandas numpy yfinance refinitiv-data matplotlib seaborn openpyxl

REFINITIV NOTE:
    Replace the mock data section with real Refinitiv API calls using:
    import refinitiv.data as rd
    rd.open_session()
    
=============================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "esg_min_score":        50,       # Minimum ESG score filter
    "min_market_cap_eur":   500e6,    # €500M minimum market cap
    "min_avg_volume_eur":   1e6,      # €1M average daily volume
    "momentum_window_long": 126,      # ~6 months in trading days
    "momentum_window_short": 21,      # ~1 month in trading days
    "momentum_top_pct":     0.25,     # Top 25% for momentum core
    "rsi_period":           14,
    "rsi_oversold":         40,       # RSI below this = mean-reversion candidate
    "mr_drawdown_min":      0.10,     # Min 10% drawdown to be mean-reversion candidate
    "mr_drawdown_max":      0.30,     # Max 30% drawdown (avoid fundamentally broken)
    "core_n_stocks":        15,       # Number of core momentum stocks
    "satellite_n_stocks":   7,        # Number of satellite mean-reversion stocks
    "core_weight":          0.65,     # 65% in momentum core
    "satellite_weight":     0.35,     # 35% in mean-reversion satellite
    "exit_momentum_pct":    0.40,     # Exit if falls below top 40% momentum rank
    "exit_mr_profit":       0.08,     # Take profit at +8%
    "exit_mr_stoploss":    -0.05,     # Stop loss at -5%
    "transaction_cost_bps": 20,       # 20 bps transaction cost
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. EUROPEAN STOCK UNIVERSE
#    Replace or extend this list with your Refinitiv-filtered universe
#    These are example tickers from Paris (PA), Amsterdam (AS), Madrid (MC), Brussels (BR)
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSE_TICKERS = [
    # Paris (CAC 40 + mid-cap examples)
    "AI.PA",   "AIR.PA",  "BN.PA",   "BNP.PA",  "CA.PA",
    "CAP.PA",  "CS.PA",   "DG.PA",   "EL.PA",   "EN.PA",
    "ENGI.PA", "FP.PA",   "GLE.PA",  "HO.PA",   "KER.PA",
    "LR.PA",   "MC.PA",   "ML.PA",   "MT.PA",   "ORA.PA",
    "PUB.PA",  "RI.PA",   "RMS.PA",  "SAF.PA",  "SAN.PA",
    "SGO.PA",  "STLAM.MI","STM.PA",  "SU.PA",   "TTE.PA",
    "VIE.PA",  "VIV.PA",  "WLN.PA",

    # Amsterdam (AEX examples)
    "ADYEN.AS","ASML.AS", "DSM.AS",  "HEIA.AS", "IMCD.AS",
    "INGA.AS", "MT.AS",   "NN.AS",   "PHIA.AS", "RAND.AS",
    "REN.AS",  "SHELL.AS","UMG.AS",  "UNA.AS",  "WKL.AS",

    # Madrid (IBEX examples)
    "ACS.MC",  "AMS.MC",  "BBVA.MC", "BKT.MC",  "CABK.MC",
    "ELE.MC",  "FER.MC",  "GRF.MC",  "IBE.MC",  "ITX.MC",
    "MAP.MC",  "REE.MC",  "REP.MC",  "SAB.MC",  "SAN.MC",

    # Brussels (BEL 20 examples)
    "ABI.BR",  "ACKB.BR", "AGS.BR",  "COFB.BR", "GBLB.BR",
    "KBC.BR",  "SOLB.BR", "UCB.BR",  "UMI.BR",
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. MOCK ESG DATA
#    !! REPLACE THIS with real Refinitiv API calls !!
#    Example Refinitiv call:
#
#    import refinitiv.data as rd
#    rd.open_session()
#    df_esg = rd.get_data(
#        universe=UNIVERSE_TICKERS,
#        fields=["TR.TRESGScore", "TR.CO2EmissionTotal", "TR.AnalyticCO2"]
#    )
# ─────────────────────────────────────────────────────────────────────────────

def get_mock_esg_data(tickers):
    """
    Mock ESG data — replace with Refinitiv API call.
    Returns DataFrame with ESG scores and carbon intensity.
    """
    np.random.seed(42)
    return pd.DataFrame({
        "ticker":           tickers,
        "esg_score":        np.random.uniform(30, 90, len(tickers)),
        "carbon_intensity": np.random.uniform(10, 500, len(tickers)),  # tCO2/€M revenue
        "social_score":     np.random.uniform(30, 90, len(tickers)),
        "governance_score": np.random.uniform(30, 90, len(tickers)),
        "controversy_score":np.random.uniform(0, 5, len(tickers)),     # 0=low, 5=high
    }).set_index("ticker")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_price_data(tickers, lookback_days=252):
    """Download historical prices from Yahoo Finance."""
    end   = datetime.today()
    start = end - timedelta(days=lookback_days + 30)

    print(f"\n📥 Downloading price data for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"]
        volume = raw["Volume"]
    else:
        close  = raw[["Close"]]
        volume = raw[["Volume"]]

    # Drop tickers with too much missing data (>20%)
    valid = close.columns[close.isna().mean() < 0.20]
    close  = close[valid].ffill()
    volume = volume[valid].ffill()

    print(f"   ✅ {len(valid)} tickers with sufficient data")
    return close, volume


# ─────────────────────────────────────────────────────────────────────────────
# 5. TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI for a single price series, return latest value."""
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def compute_momentum_score(prices: pd.Series, long_w: int, short_w: int) -> float:
    """
    Momentum = 6-month return minus 1-month return.
    This avoids buying stocks that peaked last week.
    """
    if len(prices) < long_w + 5:
        return np.nan
    ret_long  = (prices.iloc[-1] / prices.iloc[-long_w])  - 1
    ret_short = (prices.iloc[-1] / prices.iloc[-short_w]) - 1
    return ret_long - ret_short


def compute_drawdown_from_peak(prices: pd.Series, window: int = 63) -> float:
    """Max drawdown from peak over recent window (~3 months)."""
    recent = prices.iloc[-window:]
    peak   = recent.cummax()
    dd     = (recent - peak) / peak
    return dd.iloc[-1]   # current drawdown (negative number)


def compute_indicators(close_df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators for every ticker."""
    cfg = CONFIG
    rows = []

    for ticker in close_df.columns:
        prices = close_df[ticker].dropna()
        if len(prices) < cfg["momentum_window_long"] + 10:
            continue

        momentum = compute_momentum_score(
            prices,
            cfg["momentum_window_long"],
            cfg["momentum_window_short"]
        )
        rsi      = compute_rsi(prices, cfg["rsi_period"])
        drawdown = compute_drawdown_from_peak(prices)
        ret_1m   = (prices.iloc[-1] / prices.iloc[-cfg["momentum_window_short"]]) - 1
        ret_3m   = (prices.iloc[-1] / prices.iloc[-63]) - 1 if len(prices) >= 63 else np.nan
        vol_21   = prices.pct_change().iloc[-21:].std() * np.sqrt(252)

        rows.append({
            "ticker":   ticker,
            "momentum": momentum,
            "rsi":      rsi,
            "drawdown": drawdown,
            "ret_1m":   ret_1m,
            "ret_3m":   ret_3m,
            "vol_ann":  vol_21,
            "price":    prices.iloc[-1],
        })

    df = pd.DataFrame(rows).set_index("ticker")

    # Rank momentum (0 = lowest, 1 = highest)
    df["momentum_rank"] = df["momentum"].rank(pct=True)

    return df.dropna(subset=["momentum"])


# ─────────────────────────────────────────────────────────────────────────────
# 6. STOCK SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def filter_esg_universe(indicators: pd.DataFrame, esg_df: pd.DataFrame) -> pd.DataFrame:
    """Merge indicators with ESG data and apply ESG filter."""
    merged = indicators.join(esg_df, how="inner")
    filtered = merged[merged["esg_score"] >= CONFIG["esg_min_score"]].copy()
    print(f"\n🌱 ESG filter (score ≥ {CONFIG['esg_min_score']}): "
          f"{len(indicators)} → {len(filtered)} stocks")
    return filtered


def select_core_momentum(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select top momentum stocks for core portfolio."""
    top_pct = CONFIG["momentum_top_pct"]
    core = df[df["momentum_rank"] >= (1 - top_pct)].copy()
    # Sort by momentum, then by ESG as tiebreaker
    core = core.sort_values(["momentum", "esg_score"], ascending=[False, False])
    selected = core.head(n)
    print(f"\n🚀 Core Momentum: selected {len(selected)} stocks (top {top_pct*100:.0f}% momentum)")
    return selected


def select_satellite_mean_reversion(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select oversold stocks for mean-reversion satellite."""
    cfg = CONFIG
    candidates = df[
        (df["rsi"] < cfg["rsi_oversold"]) &
        (df["drawdown"] <= -cfg["mr_drawdown_min"]) &
        (df["drawdown"] >= -cfg["mr_drawdown_max"])
    ].copy()
    # Sort by lowest RSI (most oversold) and best ESG
    candidates = candidates.sort_values(["rsi", "esg_score"], ascending=[True, False])
    selected = candidates.head(n)
    print(f"📉 Satellite Mean-Reversion: {len(candidates)} candidates → selected {len(selected)}")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 7. PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio(core: pd.DataFrame, satellite: pd.DataFrame,
                    portfolio_value: float = 100_000) -> pd.DataFrame:
    """
    Assign weights and compute position sizes.
    Core = equal weight within 65% allocation
    Satellite = equal weight within 35% allocation
    """
    core      = core.copy()
    satellite = satellite.copy()

    core["segment"]      = "Core_Momentum"
    satellite["segment"] = "Satellite_MeanReversion"

    core["weight"]      = CONFIG["core_weight"]      / len(core)
    satellite["weight"] = CONFIG["satellite_weight"] / len(satellite)

    portfolio = pd.concat([core, satellite])
    portfolio["allocation_eur"]  = portfolio["weight"] * portfolio_value
    portfolio["shares"]          = (portfolio["allocation_eur"] / portfolio["price"]).apply(np.floor)
    portfolio["actual_value_eur"]= portfolio["shares"] * portfolio["price"]
    portfolio["tx_cost_eur"]     = portfolio["actual_value_eur"] * (CONFIG["transaction_cost_bps"] / 10_000)

    total_invested = portfolio["actual_value_eur"].sum()
    cash_remaining = portfolio_value - total_invested - portfolio["tx_cost_eur"].sum()

    print(f"\n💼 Portfolio Summary:")
    print(f"   Total stocks     : {len(portfolio)}")
    print(f"   Total invested   : €{total_invested:,.0f}")
    print(f"   Transaction costs: €{portfolio['tx_cost_eur'].sum():,.0f}")
    print(f"   Cash remaining   : €{cash_remaining:,.0f}")

    return portfolio


# ─────────────────────────────────────────────────────────────────────────────
# 8. ESG PORTFOLIO METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_esg(portfolio: pd.DataFrame) -> dict:
    """Compute weighted ESG metrics for the portfolio."""
    w = portfolio["weight"] / portfolio["weight"].sum()

    metrics = {
        "weighted_esg_score":        (w * portfolio["esg_score"]).sum(),
        "weighted_carbon_intensity": (w * portfolio["carbon_intensity"]).sum(),
        "weighted_social_score":     (w * portfolio["social_score"]).sum(),
        "weighted_governance_score": (w * portfolio["governance_score"]).sum(),
        "avg_controversy_score":     (w * portfolio["controversy_score"]).sum(),
        "pct_high_esg":              (portfolio["esg_score"] >= 70).mean() * 100,
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 9. REBALANCING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def check_rebalancing_signals(portfolio: pd.DataFrame,
                               current_prices: pd.Series,
                               all_indicators: pd.DataFrame) -> dict:
    """
    Check exit signals for each position.
    Returns dict with stocks to SELL and reason.
    """
    sells = {}

    for ticker, row in portfolio.iterrows():
        if ticker not in current_prices.index:
            continue

        current_price = current_prices[ticker]
        entry_price   = row["price"]
        pnl_pct       = (current_price - entry_price) / entry_price

        if row["segment"] == "Core_Momentum":
            # Exit if momentum rank falls below threshold
            if ticker in all_indicators.index:
                cur_rank = all_indicators.loc[ticker, "momentum_rank"]
                if cur_rank < CONFIG["exit_momentum_pct"]:
                    sells[ticker] = f"Momentum rank fell to {cur_rank:.2f} (threshold: {CONFIG['exit_momentum_pct']})"

        elif row["segment"] == "Satellite_MeanReversion":
            # Take profit
            if pnl_pct >= CONFIG["exit_mr_profit"]:
                sells[ticker] = f"Take profit triggered: +{pnl_pct*100:.1f}%"
            # Stop loss
            elif pnl_pct <= CONFIG["exit_mr_stoploss"]:
                sells[ticker] = f"Stop loss triggered: {pnl_pct*100:.1f}%"

    return sells


def get_replacement_candidates(all_indicators: pd.DataFrame,
                                current_portfolio_tickers: list,
                                segment: str,
                                n: int = 5) -> pd.DataFrame:
    """
    Find replacement stocks not already in portfolio.
    """
    candidates = all_indicators[
        ~all_indicators.index.isin(current_portfolio_tickers)
    ].copy()

    if segment == "Core_Momentum":
        return candidates.sort_values("momentum", ascending=False).head(n)
    else:
        cfg = CONFIG
        mr_candidates = candidates[
            (candidates["rsi"] < cfg["rsi_oversold"]) &
            (candidates["drawdown"] <= -cfg["mr_drawdown_min"])
        ]
        return mr_candidates.sort_values("rsi").head(n)


# ─────────────────────────────────────────────────────────────────────────────
# 10. PERFORMANCE TRACKING
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceTracker:
    def __init__(self, portfolio: pd.DataFrame, initial_value: float = 100_000):
        self.initial_value = initial_value
        self.history       = []  # List of dicts per rebalancing date

    def record(self, date, portfolio: pd.DataFrame,
               current_prices: pd.Series, esg_metrics: dict):
        """Record portfolio state at a given date."""
        port = portfolio.copy()
        port["current_price"] = port.index.map(
            lambda t: current_prices.get(t, port.loc[t, "price"])
        )
        port["current_value"] = port["shares"] * port["current_price"]
        total_value = port["current_value"].sum()
        total_return = (total_value - self.initial_value) / self.initial_value

        self.history.append({
            "date":         date,
            "total_value":  total_value,
            "total_return": total_return,
            "n_stocks":     len(port),
            **esg_metrics,
        })

    def get_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.history).set_index("date")


# ─────────────────────────────────────────────────────────────────────────────
# 11. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_portfolio_analysis(portfolio: pd.DataFrame,
                             close_df: pd.DataFrame,
                             esg_metrics: dict):
    """Generate 4-panel portfolio analysis chart."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("ESG Portfolio Analysis — Momentum + Mean-Reversion Strategy",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = {"Core_Momentum": "#2E86AB", "Satellite_MeanReversion": "#E84855"}

    # ── Panel 1: Segment weight breakdown ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    seg_weights = portfolio.groupby("segment")["weight"].sum()
    wedge_colors = [colors.get(s, "gray") for s in seg_weights.index]
    wedges, texts, autotexts = ax1.pie(
        seg_weights.values,
        labels=[s.replace("_", "\n") for s in seg_weights.index],
        autopct="%1.1f%%",
        colors=wedge_colors,
        startangle=90,
        textprops={"fontsize": 9}
    )
    ax1.set_title("Portfolio Allocation", fontweight="bold")

    # ── Panel 2: ESG score distribution ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for seg, grp in portfolio.groupby("segment"):
        ax2.hist(grp["esg_score"], bins=8, alpha=0.7,
                 label=seg.replace("_", " "), color=colors.get(seg, "gray"))
    ax2.axvline(esg_metrics["weighted_esg_score"], color="black",
                linestyle="--", linewidth=1.5, label=f"Wtd Avg: {esg_metrics['weighted_esg_score']:.1f}")
    ax2.set_xlabel("ESG Score")
    ax2.set_ylabel("# Stocks")
    ax2.set_title("ESG Score Distribution", fontweight="bold")
    ax2.legend(fontsize=8)

    # ── Panel 3: Momentum vs RSI scatter ───────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for seg, grp in portfolio.groupby("segment"):
        ax3.scatter(grp["momentum"] * 100, grp["rsi"],
                    c=colors.get(seg, "gray"), label=seg.replace("_", " "),
                    s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
    ax3.axhline(CONFIG["rsi_oversold"], color="red", linestyle="--",
                alpha=0.5, linewidth=1, label=f"RSI oversold ({CONFIG['rsi_oversold']})")
    ax3.set_xlabel("Momentum Score (%)")
    ax3.set_ylabel("RSI (14)")
    ax3.set_title("Momentum vs RSI", fontweight="bold")
    ax3.legend(fontsize=8)

    # ── Panel 4: Top holdings by weight ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    top = portfolio.sort_values("weight", ascending=True).tail(20)
    bar_colors = [colors.get(s, "gray") for s in top["segment"]]
    bars = ax4.barh(top.index, top["weight"] * 100, color=bar_colors, edgecolor="white")
    ax4.set_xlabel("Portfolio Weight (%)")
    ax4.set_title("Portfolio Holdings by Weight", fontweight="bold")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors["Core_Momentum"], label="Core Momentum"),
                       Patch(facecolor=colors["Satellite_MeanReversion"], label="Satellite Mean-Rev")]
    ax4.legend(handles=legend_elements, fontsize=8)

    # ── Panel 5: ESG metrics radar ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    metrics_labels = ["ESG Score\n(/100)", "Social\n(/100)", "Governance\n(/100)",
                      "Low Carbon\n(inv)", "Low Controversy\n(inv)"]
    carbon_norm = max(0, 100 - esg_metrics["weighted_carbon_intensity"] / 5)
    controversy_norm = max(0, 100 - esg_metrics["avg_controversy_score"] * 20)
    values = [
        esg_metrics["weighted_esg_score"],
        esg_metrics["weighted_social_score"],
        esg_metrics["weighted_governance_score"],
        carbon_norm,
        controversy_norm,
    ]
    x = np.arange(len(metrics_labels))
    bars2 = ax5.bar(x, values, color="#2E86AB", alpha=0.8, edgecolor="white")
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics_labels, fontsize=7.5)
    ax5.set_ylim(0, 100)
    ax5.set_ylabel("Score")
    ax5.set_title("Portfolio ESG Profile", fontweight="bold")
    ax5.axhline(CONFIG["esg_min_score"], color="red", linestyle="--",
                alpha=0.5, linewidth=1, label=f"Min threshold ({CONFIG['esg_min_score']})")
    ax5.legend(fontsize=8)

    plt.savefig("/mnt/user-data/outputs/portfolio_analysis.png",
                dpi=150, bbox_inches="tight", facecolor="white")
    print("   📊 Chart saved: portfolio_analysis.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 12. EXCEL TRACKING SHEET
# ─────────────────────────────────────────────────────────────────────────────

def export_tracking_sheet(portfolio: pd.DataFrame,
                           esg_metrics: dict,
                           rebalancing_dates: list):
    """Export a full tracking worksheet to Excel."""
    output_path = "/mnt/user-data/outputs/portfolio_tracking.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # ── Sheet 1: Initial Portfolio ──────────────────────────────────────
        cols = ["segment", "weight", "price", "shares", "allocation_eur",
                "tx_cost_eur", "momentum", "momentum_rank", "rsi",
                "drawdown", "esg_score", "carbon_intensity",
                "social_score", "governance_score", "controversy_score"]
        export_cols = [c for c in cols if c in portfolio.columns]
        port_display = portfolio[export_cols].copy()
        port_display["weight"] = (port_display["weight"] * 100).round(2)
        port_display = port_display.round(4)
        port_display.to_excel(writer, sheet_name="Initial Portfolio", index=True)

        # ── Sheet 2: ESG Summary ────────────────────────────────────────────
        esg_summary = pd.DataFrame([esg_metrics]).T
        esg_summary.columns = ["Portfolio Value"]
        esg_summary.to_excel(writer, sheet_name="ESG Summary")

        # ── Sheet 3: Rebalancing Log (blank template) ───────────────────────
        rebal_template = pd.DataFrame({
            "Date":            rebalancing_dates,
            "Stocks Sold":     [""] * len(rebalancing_dates),
            "Sell Reason":     [""] * len(rebalancing_dates),
            "Stocks Bought":   [""] * len(rebalancing_dates),
            "Buy Reason":      [""] * len(rebalancing_dates),
            "Portfolio Value": [""] * len(rebalancing_dates),
            "Return (%)":      [""] * len(rebalancing_dates),
            "Wtd ESG Score":   [""] * len(rebalancing_dates),
            "Carbon Intensity":[""] * len(rebalancing_dates),
            "Notes":           [""] * len(rebalancing_dates),
        })
        rebal_template.to_excel(writer, sheet_name="Rebalancing Log", index=False)

        # ── Sheet 4: Exit Signal Checklist ──────────────────────────────────
        checklist = pd.DataFrame({
            "Rule":        [
                "Core: Exit if momentum_rank < 0.40",
                "Satellite: Take profit at +8%",
                "Satellite: Stop loss at -5%",
                "Both: Exit if ESG score drops below 50",
                "Mandatory: ≥2 stocks replaced every Tue & Fri",
                "Cap: Max 25% portfolio turnover per session",
            ],
            "Check":       ["☐"] * 6,
            "Notes":       [""] * 6,
        })
        checklist.to_excel(writer, sheet_name="Exit Rules", index=False)

    print(f"   📄 Excel tracking sheet saved: portfolio_tracking.xlsx")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  ESG Portfolio Builder — Momentum + Mean-Reversion Strategy")
    print("=" * 65)

    # Step 1: Load price data
    close_df, volume_df = load_price_data(UNIVERSE_TICKERS)

    # Step 2: Compute technical indicators
    print("\n⚙️  Computing technical indicators...")
    indicators = compute_indicators(close_df)

    # Step 3: Load & merge ESG data
    print("\n🌱 Loading ESG data...")
    esg_df = get_mock_esg_data(list(indicators.index))
    # ── REPLACE ABOVE LINE WITH REFINITIV CALL ──
    # import refinitiv.data as rd
    # rd.open_session()
    # esg_raw = rd.get_data(
    #     universe=list(indicators.index),
    #     fields=["TR.TRESGScore","TR.CO2EmissionTotal","TR.AnalyticSocialPillarScore",
    #             "TR.AnalyticCorporateGovernancePillarScore","TR.TRESGControversiesScore"]
    # )
    # esg_df = esg_raw.set_index("Instrument").rename(columns={
    #     "TR.TRESGScore": "esg_score",
    #     "TR.CO2EmissionTotal": "carbon_intensity",
    #     "TR.AnalyticSocialPillarScore": "social_score",
    #     "TR.AnalyticCorporateGovernancePillarScore": "governance_score",
    #     "TR.TRESGControversiesScore": "controversy_score",
    # })

    # Step 4: Filter universe by ESG
    filtered = filter_esg_universe(indicators, esg_df)

    # Step 5: Select core & satellite
    core      = select_core_momentum(filtered, n=CONFIG["core_n_stocks"])
    satellite = select_satellite_mean_reversion(
        filtered[~filtered.index.isin(core.index)],
        n=CONFIG["satellite_n_stocks"]
    )

    # Step 6: Build portfolio
    portfolio = build_portfolio(core, satellite)

    # Step 7: ESG metrics
    esg_metrics = compute_portfolio_esg(portfolio)
    print(f"\n🌍 Portfolio ESG Metrics:")
    for k, v in esg_metrics.items():
        print(f"   {k:35s}: {v:.2f}")

    # Step 8: Check rebalancing signals (example — use on each rebalancing date)
    current_prices = close_df.iloc[-1]
    sells = check_rebalancing_signals(portfolio, current_prices, filtered)
    if sells:
        print(f"\n⚠️  Rebalancing signals today:")
        for ticker, reason in sells.items():
            print(f"   SELL {ticker}: {reason}")
        replacements = get_replacement_candidates(
            filtered, list(portfolio.index), "Core_Momentum", n=3
        )
        print(f"\n   Replacement candidates:")
        print(replacements[["momentum", "rsi", "esg_score"]].round(3))
    else:
        print("\n✅ No exit signals triggered today")

    # Step 9: Visualize
    print("\n📊 Generating charts...")
    plot_portfolio_analysis(portfolio, close_df, esg_metrics)

    # Step 10: Export tracking Excel
    rebalancing_dates = [
        "2025-02-24", "2025-02-27", "2025-03-03",
        "2025-03-05", "2025-03-10", "2025-03-12",
    ]
    print("\n📄 Exporting tracking sheet...")
    export_tracking_sheet(portfolio, esg_metrics, rebalancing_dates)

    # Step 11: Print final holdings
    print("\n" + "=" * 65)
    print("  FINAL PORTFOLIO HOLDINGS")
    print("=" * 65)
    display_cols = ["segment", "weight", "price", "momentum_rank", "rsi", "esg_score"]
    display_cols = [c for c in display_cols if c in portfolio.columns]
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_rows", 50)
    print(portfolio[display_cols].sort_values("segment").to_string())
    print("\n✅ Done! Files saved to /mnt/user-data/outputs/")


if __name__ == "__main__":
    main()