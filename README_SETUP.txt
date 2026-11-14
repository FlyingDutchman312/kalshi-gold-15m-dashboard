KALSHI GOLD 15-MINUTE PREDICTION DESK (SEPARATE FROM BTC)
=========================================================

WHAT THIS IS
- A second Streamlit dashboard for Gold KXGOLD15M only, with automatic live market lookup,
  a one-second countdown in Eastern time, YES bid/ask and midpoint, an explicitly
  UNCALIBRATED quote-momentum UP/DOWN observation, market-quote history while open,
  and Supabase forward accuracy/Brier reporting against the original quote midpoint.
- Includes optional always-running Gold observation + settlement worker for a second Railway
  service; it polls every 5 seconds. NO trade/execution code exists in either app or worker.
- Bitcoin app.py, BTC data, GitHub Actions, production/demo credentials, and existing
  BTC Railway service are not changed.
- Gold is a separate market with different settlement rules (Pyth Gold); never copy BTC
  model performance scores into this app.

FILES
  app_gold.py           Streamlit Gold dashboard, make this its main module
  gold_core.py          Gold API, heuristic and storage functions
  gold_worker.py        Optional second Railway worker; observation + settlement only
  gold_store.sql        Creates separate table in existing Supabase
  requirements_gold.txt Dependencies for new Streamlit app
  Dockerfile.gold       Optional Railway Gold worker build

SETUP — FIRST CREATE TABLE
1. Supabase -> SQL Editor -> New query. Run gold_store.sql.
   It adds only public.gold_forward_signals with separate market ticker/clock checks.
   No table is deleted and your BTC database is unaffected.

SETUP — SECOND STREAMLIT APP (NEW GITHUB REPOSITORY RECOMMENDED)
2. Create a NEW GitHub repository named kalshi-gold-15m-dashboard. Upload all
   package files from the ZIP (including requirements.txt) and commit to main.
   This keeps the existing BTC repository and dependencies COMPLETELY untouched.
3. Streamlit Community Cloud -> Create app -> select the NEW Gold GitHub repo.
   Choose a DIFFERENT app name such as 'kalshi-gold-15m-ai-desk'.
   Main file path: app_gold.py. The included requirements.txt will be used automatically.
   (If you instead use the existing BTC repo, do not overwrite its requirements.txt.
   Add streamlit-autorefresh>=1.0.1 to the existing BTC requirements file only.)
4. In the new Gold Streamlit app -> Settings -> Secrets, set the SAME credentials as
   your existing BTC Streamlit [history_store] section:

   [history_store]
   url = "https://YOUR-PROJECT.supabase.co"
   secret_key = "YOUR-SUPABASE-SERVER-SIDE-SECRET"

   Do not publish the secret to GitHub. Never add /rest/v1 to the base URL.
5. Deploy Gold app. Live prices come from public Kalshi; scores/history only after the
   optional worker is running and settlement is published. A browser need not remain
   open to read already-stored Gold observations.

SETUP — OPTIONAL 24/7 AUTOMATIC OBSERVATION + SCORING
6. Commit gold_worker.py and Dockerfile.gold to the GitHub repo.
7. Railway -> same project -> NEW SERVICE -> Deploy from GitHub -> NEW Gold repo.
   Do NOT replace the existing BTC Railway service.
   Settings -> Build -> Dockerfile Path: Dockerfile.gold
   Settings -> Deploy -> Custom Start Command: python gold_worker.py
   Configure a single replica, persistent service, restart on failure.
8. Add Railway Variables in the NEW Gold service:
      SUPABASE_URL = https://YOUR-PROJECT.supabase.co
      SUPABASE_SECRET_KEY = YOUR-SERVER-SECRET
   NO Kalshi API keys are required; gold worker uses PUBLIC market endpoints.
   It cannot submit demo or production orders.
9. Check logs for 'GOLD MARKET FOUND', 'GOLD FORECAST SAVED',
   and 'GOLD SETTLEMENT CHECK'. On weekends/market breaks 'GOLD MARKET CLOSED/UNAVAILABLE'
   is expected. The worker attempts settlement rechecks every 120 seconds.
10. Verify:
    SELECT COUNT(*) total,
           COUNT(*) FILTER(WHERE yes_won IN (0,1)) scored,
           COUNT(*) FILTER(WHERE yes_won IS NULL) pending
    FROM public.gold_forward_signals;

NOTES AND LIMITATIONS
- This is a new, separate Gold market-quote heuristic, not a clone of your trained BTC AI.
- Its displayed score is NOT a calibrated chance of winning. Positive Brier improvement
  over a market midpoint baseline must be evaluated only on forward-settled markets,
  with distinct-market counts and strict chronological holdouts.
- It observes the YES odds, not direct Pyth XAUUSD. No Pyth spot-price chart is shown.
- Kalshi REST market quotes can lag by several seconds. Do not use for execution.
- The worker only records minute 3/5/7/9/11 checkpoints if market discovery and quotes
  are valid. GitHub/Cloud outages can cause missed checkpoints.
- Gold series may be inactive on weekends and holidays.
- No automatic demo trading, 1.75x cash-out, live positions, or real-money trading.
- Keep your existing BTC bot and all existing secrets/risk limits unchanged.
