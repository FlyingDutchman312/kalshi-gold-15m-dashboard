"""Score saved Gold forecasts and simulate one contract per qualified market."""

import pandas as pd

CHECKPOINT = 7
FEE = 0.02
SLIPPAGE = 0.01
MIN_EDGE = 0.03
FORECASTS = {
    "Market midpoint": "baseline_mid",
    "Quote-only heuristic": "quote_p_yes",
    "Gold context heuristic": "p_yes",
}


def prepare(rows):
    data = pd.DataFrame(rows)
    for column in ("minute", "yes_won", "yes_bid", "yes_ask", *FORECASTS.values()):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data["observed_at"] = pd.to_datetime(data["observed_at"], utc=True, errors="coerce")
    close = pd.to_datetime(pd.to_numeric(data["close_ts"], errors="coerce"),
                           unit="s", utc=True, errors="coerce")
    elapsed = (data["observed_at"] - close).dt.total_seconds() + 900
    valid_time = elapsed.ge(data["minute"] * 60) & elapsed.lt((data["minute"] + 1) * 60)
    data = (data.loc[valid_time].sort_values("observed_at", kind="stable")
            .drop_duplicates(["market_ticker", "minute"]))
    return data


def complete_forecasts(data):
    comparable = pd.Series(True, index=data.index, dtype=bool)
    for column in FORECASTS.values():
        comparable &= data[column].between(0, 1)
    return comparable


def coverage(rows):
    raw = pd.DataFrame(rows)
    data = prepare(rows)
    checkpoint = data.loc[data["minute"].eq(CHECKPOINT)]
    complete = complete_forecasts(checkpoint)
    settled = checkpoint["yes_won"].isin([0, 1])
    return {
        "Observed markets": raw["market_ticker"].nunique(),
        "Valid minute-7 records": len(checkpoint),
        "Settled minute-7 records": int(settled.sum()),
        "Included in Brier scores": int((complete & settled).sum()),
        "Settled but missing a forecast": int((~complete & settled).sum()),
    }


def scored_forecasts(data):
    comparable = complete_forecasts(data) & data["yes_won"].isin([0, 1])
    scored = data.loc[comparable].copy()
    for label, column in FORECASTS.items():
        scored[label] = (scored[column] - scored["yes_won"]) ** 2
    return scored


def calibration(rows):
    data = prepare(rows)
    matched = data.loc[data["minute"].eq(CHECKPOINT)]
    matched = matched.loc[complete_forecasts(matched) & matched["yes_won"].isin([0, 1])]
    bands = [f"{low}–{low + 10}%" for low in range(0, 100, 10)]
    reports = {}
    for label, column in FORECASTS.items():
        # Exact boundaries go in the higher band, with 100% in the last band.
        band = (matched[column] * 10).astype(int).clip(upper=9)
        summary = matched.groupby(band).agg(
            markets=("yes_won", "size"), forecast=(column, "mean"), actual=("yes_won", "mean"),
        ).reindex(range(10))
        reports[label] = pd.DataFrame({
            "Probability band": bands,
            "Settled markets": summary.markets.fillna(0).astype(int),
            "Average forecast (%)": (summary.forecast * 100).round(2),
            "Actual YES rate (%)": (summary.actual * 100).round(2),
            "Forecast minus actual (pp)": ((summary.forecast - summary.actual) * 100).round(2),
        })
    return reports, len(matched)


def rolling_brier(rows, window):
    data = prepare(rows)
    scored = scored_forecasts(data.loc[data["minute"].eq(CHECKPOINT)])
    differences = pd.DataFrame({
        "vs market midpoint": scored["Gold context heuristic"] - scored["Market midpoint"],
        "vs quote-only heuristic": scored["Gold context heuristic"] - scored["Quote-only heuristic"],
    })
    differences.index = pd.to_datetime(scored["close_ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    differences.index.name = "Market close (Eastern)"
    return differences.sort_index().rolling(window, min_periods=window).mean().dropna(), len(scored)


def evaluate(rows):
    data = prepare(rows)
    scored = scored_forecasts(data)
    by_minute = scored.groupby("minute").agg(
        **{"Markets": ("market_ticker", "size")},
        **{label: (label, "mean") for label in FORECASTS},
    )

    ledger = data.loc[data["minute"].eq(CHECKPOINT)].copy()
    quotes = (ledger.yes_bid.gt(0) & ledger.yes_ask.lt(1)
              & ledger.yes_bid.le(ledger.yes_ask))
    yes_cost = ledger.yes_ask + SLIPPAGE + FEE
    no_cost = 1 - ledger.yes_bid + SLIPPAGE + FEE
    yes_edge = ledger.p_yes - yes_cost
    no_edge = 1 - ledger.p_yes - no_cost
    buy_yes = yes_edge.ge(no_edge)
    cost = yes_cost.where(buy_yes, no_cost)
    edge = yes_edge.where(buy_yes, no_edge)
    valid_forecast = ledger.p_yes.between(0, 1)
    eligible = quotes & valid_forecast & cost.lt(1) & edge.round(8).ge(MIN_EDGE)
    ledger["Estimated edge after costs (¢)"] = (100 * edge).where(quotes & valid_forecast).round(4)
    ledger["Skip reason"] = f"Edge below {MIN_EDGE * 100:g}¢"
    ledger.loc[cost.ge(1), "Skip reason"] = "Entry cost at least $1"
    ledger.loc[~valid_forecast, "Skip reason"] = "Missing or invalid Gold forecast"
    ledger.loc[~quotes, "Skip reason"] = "Missing or invalid quote"
    ledger.loc[eligible, "Skip reason"] = ""
    settled = ledger.yes_won.isin([0, 1])
    payout = ledger.yes_won.where(buy_yes, 1 - ledger.yes_won)
    ledger["Paper decision"] = "Skip"
    ledger.loc[eligible & buy_yes, "Paper decision"] = "YES"
    ledger.loc[eligible & ~buy_yes, "Paper decision"] = "NO"
    ledger["Quoted entry (USD)"] = (cost - SLIPPAGE - FEE).where(eligible).round(4)
    ledger["Assumed fee (USD)"] = FEE
    ledger["Assumed slippage (USD)"] = SLIPPAGE
    ledger["Estimated net P&L (USD)"] = (payout - cost).where(eligible & settled).round(4)
    return by_minute, ledger


def cost_sensitivity(ledger):
    settled = ledger["Estimated net P&L (USD)"].dropna()
    if settled.empty:
        return pd.DataFrame()
    scenarios = ((FEE, SLIPPAGE), (FEE, SLIPPAGE + .01), (FEE + .01, SLIPPAGE + .02))
    rows = []
    for fee, slippage in scenarios:
        net = settled - (fee + slippage - FEE - SLIPPAGE)
        rows.append({
            "Fee (¢)": round(fee * 100),
            "Slippage (¢)": round(slippage * 100),
            "Settled entries": len(net),
            "Net P&L (USD)": round(net.sum(), 2),
            "Average P&L per entry (USD)": round(net.mean(), 4),
        })
    return pd.DataFrame(rows)


def history_health(rows, now):
    data = pd.DataFrame(rows)
    observed = pd.to_datetime(data["observed_at"], utc=True, errors="coerce")
    close = pd.to_datetime(pd.to_numeric(data["close_ts"], errors="coerce"),
                           unit="s", utc=True, errors="coerce")
    outcome = pd.to_numeric(data["yes_won"], errors="coerce").isin([0, 1])
    verified = data.loc[outcome, "market_ticker"]
    pending = close.le(now) & ~data["market_ticker"].isin(verified)
    latest_settlement = pd.NaT
    if "settled_at" in data:
        latest_settlement = pd.to_datetime(
            data.loc[outcome, "settled_at"], utc=True, errors="coerce"
        ).max()
    pending_markets = data.loc[pending, "market_ticker"]
    oldest_close = close.loc[pending].min()
    oldest_ticker = pending_markets.loc[close.loc[pending].eq(oldest_close)].min() if pd.notna(oldest_close) else None
    return observed.max(), latest_settlement, pending_markets.nunique(), oldest_ticker, oldest_close
