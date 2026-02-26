# kit.py
import json
import pandas as pd
import refinitiv.data as rd

EXCHANGES = {"XPAR", "XAMS", "XBRU", "XMAD"}  # MIC codes

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_top_esg_stocks(config_path="config.json"):
    cfg = load_config(config_path)

    rd.open_session()

    mic_codes = ", ".join([f"'{m}'" for m in cfg.get("markets", list(EXCHANGES))])

    screener_query = (
        "SCREEN("
        "U(IN(Equity(active,public,primary))), "
        f"IN(TR.ExchangeMarketIdCode, {mic_codes}), "
        "TR.TRESGScore > 0"
        ")"
    )

    fields = [
        "TR.CommonName",
        "TR.ISIN",
        "TR.ExchangeName",
        "TR.ExchangeMarketIdCode",
        "TR.Currency",
        "TR.TRESGScore",
        "TR.TRESGGrade",
        "TR.CompanyMarketCap",
        "TR.PriceClose",
    ]

    print("🔄 Screening")

    df = rd.get_data(universe=screener_query, fields=fields) 

    # RIC in instrument
    if "Instrument" in df.columns:
        df = df.rename(columns={"Instrument": "RIC"})

    # EUR currency
    if "TR.Currency" in df.columns:
        df = df[df["TR.Currency"].eq("EUR")].copy()

    # Sort based on the market cap      
    top_n = int(cfg.get("top_n", 100))
    if "TR.CompanyMarketCap" in df.columns:
        df = df.sort_values("TR.CompanyMarketCap", ascending=False)

    df = df.head(top_n).reset_index(drop=True)

    print(f"✅ Done: {len(df)} stock.")
    return df



