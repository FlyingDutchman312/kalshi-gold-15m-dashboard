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


def evaluate(rows):
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
    comparable = data["yes_won"].isin([0, 1])
    for column in FORECASTS.values():
        comparable &= data[column].between(0, 1)
    scored = data.loc[comparable].copy()
    for label, column in FORECASTS.items():
        scored[label] = (scored[column] - scored["yes_won"]) ** 2
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
    eligible = quotes & ledger.p_yes.between(0, 1) & cost.lt(1) & edge.round(8).ge(MIN_EDGE)
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
