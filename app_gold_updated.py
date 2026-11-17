"""Independent, read-only Kalshi Gold 15-minute prediction and forward-test dashboard."""
import os
import io
import json
import zipfile
from datetime import timedelta
import requests
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
from gold_core import (SERIES, CHECKPOINTS, discover, bid_ask, estimate, minute_of_market,
                       utc, supa)
from gold_fundamentals import fred_snapshot, news_snapshot, enrich

st.set_page_config(page_title='Gold 15m', page_icon='🥇',layout='wide')
st.markdown('''<style>
.block-container {padding-top:1.3rem;max-width:1200px}
div[data-testid="stMetric"] {background:rgba(126,121,104,.08);border:1px solid rgba(145,143,130,.18);padding:14px;border-radius:12px}
</style>''', unsafe_allow_html=True)
ET=ZoneInfo('America/New_York')


def config():
    """Public dashboard: only a low-privilege Supabase publishable key is accepted.

    Never read full-access secrets from environment variables or Streamlit Secrets.
    This public app is read-only; the separate Railway worker retains its own keys.
    """
    try:
        group = st.secrets.get('history_store', {})
        u = str(group.get('url', '')).strip().rstrip('/')
        if u.endswith('/rest/v1'):
            u = u[:-8]
        key = str(group.get('publishable_key', '')).strip()
        if u.startswith('https://') and u.endswith('.supabase.co') and key.startswith('sb_publishable_'):
            return u, key
    except Exception:
        pass
    return '', ''


def fred_key():
    try:
        return str(st.secrets.get('gold_fundamentals',{}).get('fred_api_key','')).strip() or os.getenv('FRED_API_KEY','').strip()
    except Exception:
        return os.getenv('FRED_API_KEY','').strip()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_macro(key):
    return fred_snapshot(api_key=key)


@st.cache_data(ttl=300, show_spinner=False)
def cached_news():
    return news_snapshot()


def et_str(value):
    value=utc(value) if not isinstance(value,datetime) else value
    return value.astimezone(ET).strftime('%b %d, %I:%M:%S %p %Z') if value else '—'


def timer(close):
    close_unix=int(close.timestamp()*1000)
    components.html('''<div style="font-family:system-ui;color:#e2c982;background:#171a23;padding:12px 20px;border-radius:12px;text-align:center">
      <span style="font-size:12px;letter-spacing:1px">GOLD MARKET CLOSES IN</span>
      <div id="count" style="font-size:38px;font-weight:750;font-variant-numeric:tabular-nums">--:--</div>
      </div><script>const target = '''+str(close_unix)+''';const el=document.getElementById('count');function tick(){let s=Math.max(0,Math.ceil((target-Date.now())/1000));el.textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}tick();setInterval(tick,1000);</script>''',height=112)


def records(cfg):
    if not all(cfg): return []
    # Restrict the dashboard to a specific table and GET only.
    return supa('GET', 'gold_forward_signals', cfg,
                {'select':'market_ticker,minute,observed_at,p_yes,yes_bid,yes_ask,baseline_mid,direction,decision,yes_won,model_version,quote_p_yes',
                 'order':'observed_at.desc', 'limit':'3000'})


def report(data):
    st.subheader('Gold Forward Performance & Model Quality')
    if not data:
        st.info('No Gold observations yet. Install gold_store.sql and start the optional Gold observation worker to record them 24/7.')
        return
    df=pd.DataFrame(data)
    for c in ['minute','p_yes','yes_won','baseline_mid']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    scored=df[df.yes_won.isin([0,1]) & df.p_yes.between(0,1)].copy()
    n=len(scored)
    markets=scored.market_ticker.nunique()
    a,b,c,d=st.columns(4)
    a.metric('Recorded Gold forecasts',len(df))
    b.metric('Scored',n)
    c.metric('Pending',int(df.yes_won.isna().sum()))
    d.metric('Distinct settled markets',markets)
    st.progress(min(n/100,1.0),text=f'Scored observation target: {n}/100')
    st.progress(min(markets/30,1.0),text=f'Distinct settled Gold markets target: {markets}/30')
    if not n:
        st.info('Waiting for official Kalshi Gold settlement outcomes. These are separate from Bitcoin outcomes.')
        st.dataframe(df.head(20),hide_index=True,use_container_width=True)
        return
    scored['correct']=((scored.p_yes>=.5)==(scored.yes_won==1)).astype(int)
    scored['brier']=(scored.p_yes-scored.yes_won)**2
    scored['market_brier']=(scored.baseline_mid-scored.yes_won)**2
    if 'quote_p_yes' in scored:
        scored['quote_p_yes']=pd.to_numeric(scored['quote_p_yes'],errors='coerce')
        scored['quote_brier']=(scored.quote_p_yes-scored.yes_won)**2
    x,y,z=st.columns(3)
    x.metric('Direction accuracy',f'{100*scored.correct.mean():.2f}%')
    y.metric('Gold heuristic Brier',f'{scored.brier.mean():.5f}')
    z.metric('Market midpoint Brier',f'{scored.market_brier.mean():.5f}')
    delta=scored.brier.mean()-scored.market_brier.mean()
    st.caption(f'Same-observation Brier difference (heuristic − market): {delta:+.5f}; lower is better. This is NOT the BTC historical validation gate.')
    if markets<30: st.warning(f'Only {markets} distinct settled gold markets. No calibrated win probabilities or trade permissions can be inferred.')
    by=(scored.groupby('minute').agg(predictions=('correct','size'),markets=('market_ticker','nunique'),
          correct=('correct','sum'),accuracy=('correct','mean'),brier=('brier','mean'),market_brier=('market_brier','mean'))
          .reset_index().sort_values('minute'))
    by['accuracy']=(100*by['accuracy']).round(2)
    st.markdown('**Performance by checkpoint (minute)**')
    st.dataframe(by.rename(columns={'minute':'Minute','predictions':'Forecasts','markets':'Distinct markets',
       'correct':'Correct','accuracy':'Accuracy %','brier':'Heuristic Brier','market_brier':'Market Brier'}),
       use_container_width=True,hide_index=True)
    st.bar_chart(by.set_index('minute')[['brier','market_brier']])
    if 'quote_brier' in scored and scored.quote_brier.notna().any():
        st.caption(f'Original quote-only heuristic Brier: {scored.quote_brier.mean():.5f}. Compare with Gold context Brier above on exactly the same scored records; a model change is not proof of a real advantage.')
    st.markdown('**Recent Gold observations**')
    st.dataframe(df.head(50),use_container_width=True,hide_index=True)
    csv=df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Gold forward results CSV',csv,'kalshi_gold_forward.csv','text/csv')



# ---------- Gold-only market-data layer. No authenticated Kalshi trading calls. ----------
GOLD_SPOT_API = 'https://api.gold-api.com/price/XAU'
TD_API = 'https://api.twelvedata.com/time_series'
SPAN_OPTIONS = {
    '1m': (timedelta(minutes=1), '1min', 30),
    '5m': (timedelta(minutes=5), '1min', 30),
    '10min': (timedelta(minutes=10), '1min', 30),
    '15min': (timedelta(minutes=15), '1min', 40),
    '30min': (timedelta(minutes=30), '1min', 60),
    '1 hour': (timedelta(hours=1), '1min', 110),
    '12 hour': (timedelta(hours=12), '5min', 180),
    '24 hour': (timedelta(hours=24), '15min', 140),
    'weekly': (timedelta(days=7), '1h', 200),
    'monthly': (timedelta(days=30), '4h', 230),
    'yearly': (timedelta(days=365), '1day', 400),
}


def secret_value(section, name):
    try:
        return str(st.secrets.get(section, {}).get(name, '')).strip()
    except Exception:
        return ''


@st.cache_data(ttl=30, show_spinner=False)
def gold_spot():
    r = requests.get(GOLD_SPOT_API, timeout=12)
    r.raise_for_status()
    d = r.json()
    price = float(d.get('price'))
    if not np.isfinite(price) or price <= 0:
        raise ValueError('No valid gold spot price returned')
    return {'price': price, 'updated_at': d.get('updatedAt') or d.get('updated_at') or d.get('timestamp')}


@st.cache_data(ttl=120, show_spinner=False)
def twelve_data_candles(api_key, interval, outputsize):
    if not api_key:
        return pd.DataFrame()
    r = requests.get(TD_API, params={
        'symbol': 'XAU/USD', 'interval': interval, 'outputsize': outputsize,
        'apikey': api_key, 'timezone': 'UTC'}, timeout=18)
    r.raise_for_status()
    payload = r.json()
    if payload.get('status') == 'error' or 'values' not in payload:
        raise ValueError(str(payload.get('message', 'Gold historical feed returned no bars'))[:250])
    frame = pd.DataFrame(payload['values'])
    frame['time'] = pd.to_datetime(frame['datetime'], utc=True, errors='coerce')
    for col in ('open', 'high', 'low', 'close'):
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame['volume'] = pd.to_numeric(frame.get('volume', pd.Series(index=frame.index, dtype=float)), errors='coerce')
    return frame.dropna(subset=['time', 'open', 'high', 'low', 'close']).sort_values('time').reset_index(drop=True)


def sampled_spot_frame():
    samples = st.session_state.setdefault('gold_spot_samples', [])
    return pd.DataFrame(samples, columns=['time', 'close']) if samples else pd.DataFrame(columns=['time', 'close'])


def chart_data(label, key):
    span, interval, size = SPAN_OPTIONS[label]
    if key:
        data = twelve_data_candles(key, interval, size)
    else:
        data = sampled_spot_frame()
    if data.empty:
        return data
    data = data[data.time >= pd.Timestamp.now(tz='UTC') - span].copy()
    return data.sort_values('time').reset_index(drop=True)


def indicator_values(frame):
    if len(frame) < 26:
        return None
    close = pd.to_numeric(frame.close, errors='coerce')
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    ratio = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else float('inf')
    rsi = 100 - 100/(1+ratio)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    return {'EMA 9': float(ema9.iloc[-1]), 'EMA 21': float(ema21.iloc[-1]),
            'EMA direction': 'UP' if ema9.iloc[-1] > ema21.iloc[-1] else 'DOWN',
            'RSI 14': round(float(rsi), 2), 'MACD': round(float(macd.iloc[-1]), 4),
            'MACD signal': round(float(signal.iloc[-1]), 4),
            'Momentum (3 bars %)': round(float(close.pct_change(3).iloc[-1]*100), 4)}


def clean_forward(raw):
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    for c in ('p_yes','yes_won','baseline_mid','yes_bid','yes_ask','minute'):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'observed_at' in df:
        df['observed_at'] = pd.to_datetime(df['observed_at'], utc=True, errors='coerce')
    return df


def calibrated_gold_model(df):
    """Gold-only, chronological holdout; one record per market avoids pseudo replication.

    This makes no calibrated claim until an out-of-sample Brier gate is passed.
    """
    if df.empty or not {'market_ticker','p_yes','yes_won','baseline_mid','observed_at'}.issubset(df.columns):
        return None, 'Not enough settled gold forecasts.'
    eligible = df[df.yes_won.isin([0,1]) & df.p_yes.between(0,1) & df.baseline_mid.between(0,1)].copy()
    eligible = eligible.dropna(subset=['observed_at']).sort_values('observed_at')
    eligible = eligible.drop_duplicates('market_ticker', keep='last')
    n = len(eligible)
    if n < 150:
        return None, f'Need at least 150 distinct settled Gold markets for chronological calibration/holdout; have {n}.'
    split = int(n*.7)
    train, test = eligible.iloc[:split], eligible.iloc[split:]
    if train.yes_won.nunique() < 2 or test.yes_won.nunique() < 2:
        return None, 'Both UP and DOWN outcomes are required in calibration and holdout periods.'
    try:
        from sklearn.isotonic import IsotonicRegression
        calibration = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        calibration.fit(train.p_yes.to_numpy(), train.yes_won.to_numpy())
        predicted = calibration.predict(test.p_yes.to_numpy())
    except ImportError:
        return None, 'scikit-learn is required for Gold probability calibration.'
    model_brier = float(np.mean((predicted - test.yes_won.to_numpy())**2))
    market_brier = float(np.mean((test.baseline_mid.to_numpy() - test.yes_won.to_numpy())**2))
    quality = {'calibration_markets': len(train), 'holdout_markets': len(test),
               'holdout_brier': model_brier, 'market_brier': market_brier,
               'passed': model_brier < market_brier}
    if not quality['passed']:
        return None, 'Gold probability calibration did not beat market midpoint on the untouched holdout.', quality
    return calibration, 'Passed chronological Gold holdout Brier gate.', quality


def pct_text(v):
    try:
        return f'{float(v):.1%}'
    except (ValueError, TypeError):
        return '—'


def paper_backtest(df, calibrated):
    """Offline, single-observation-per-market fee-aware illustrative settlement scenario."""
    if df.empty or calibrated is None:
        return pd.DataFrame()
    valid = df[df.yes_won.isin([0,1]) & df.p_yes.between(0,1) & df.yes_ask.between(0.01,.99)].copy()
    valid = valid.dropna(subset=['observed_at']).sort_values('observed_at').drop_duplicates('market_ticker', keep='last')
    if len(valid) < 150:
        return pd.DataFrame()
    test = valid.iloc[int(len(valid)*.7):].copy()
    test['calibrated_p_up'] = calibrated.predict(test.p_yes.to_numpy())
    test['chosen_side'] = np.where(test.calibrated_p_up >= .5, 'UP', 'DOWN')
    # We cannot accurately price DOWN without the true NO ask at entry, so exclude it.
    test = test[test.chosen_side == 'UP'].copy()
    if test.empty:
        return test
    # $5 max illustrative paper allocation, pessimistic 1c slippage and per-order fee.
    test['entry_price'] = test.yes_ask + .01
    test['contracts'] = np.floor(5 / (test.entry_price + .02)).astype(int)
    test['fee_estimate'] = np.ceil(.07 * test.contracts * test.yes_ask * (1-test.yes_ask) * 100) / 100
    test['cost'] = test.contracts*test.entry_price + test.fee_estimate
    test = test[(test.contracts > 0) & (test.cost <= 5)].copy()
    test['expected_edge'] = test.calibrated_p_up - (test.entry_price + test.fee_estimate / test.contracts)
    test = test[test.expected_edge >= .03].copy()
    test['net_pnl'] = test.contracts*test.yes_won - test.cost
    return test


def export_zip(forward, journal):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('gold_forward_signals.csv', pd.DataFrame(forward).to_csv(index=False))
        z.writestr('gold_session_trade_journal.csv', pd.DataFrame(journal).to_csv(index=False))
        z.writestr('readme.txt', 'Gold dashboard export. Forward signals are read-only Supabase observations. Journal entries are local to this Streamlit session unless stored by a separate backend. Not evidence of real orders.\n')
    return buffer.getvalue()


# ---------- Live dashboard ----------
st.title('🥇 Kalshi Gold · 15-Minute Prediction Desk')
st.caption('Independent from BTC · KXGOLD15M · Eastern Time · read-only account/market access')
st.warning('REAL-MONEY ORDER EXECUTION IS NOT ENABLED. This dashboard does not place or cancel Kalshi orders.')
st_autorefresh(interval=10000, key='gold_dashboard_refresh')
now=datetime.now(timezone.utc)
config_values=config()
market=None; market_error=None; ba=None; minute=None; calc=None
try:
    market=discover(now)
except Exception as exc:
    market_error=f'{type(exc).__name__}: {exc}'

if market:
    try:
        ba=bid_ask(market)
        minute=minute_of_market(market,now)
    except Exception as exc:
        st.warning(f'Kalshi live quote unavailable: {exc}')

try:
    spot_result=gold_spot()
    samples=st.session_state.setdefault('gold_spot_samples', [])
    moment=pd.Timestamp.now(tz='UTC')
    if not samples or (moment-pd.Timestamp(samples[-1][0])).total_seconds()>=25:
        samples.append((moment,spot_result['price']))
    cutoff=moment-timedelta(days=2)
    samples[:]=[x for x in samples if pd.Timestamp(x[0])>=cutoff][-5000:]
    spot_error=None
except Exception as exc:
    spot_result=None; spot_error=str(exc)[:180]

# The worker writes the Gold forward ledger. Never post to Gold/BTC tables from this public UI.
forward_rows=[]; forward_error=None
if all(config_values):
    try:
        forward_rows=records(config_values)
    except Exception as exc:
        forward_error=f'{type(exc).__name__}: {exc}'
forward=clean_forward(forward_rows)

macro_data=cached_macro(fred_key())
news_data=cached_news()
if market and ba:
    bid, ask=ba
    quotes=st.session_state.setdefault('gold_quote_history',{})
    series=quotes.setdefault(market['ticker'], [])
    midpoint=(bid+ask)/2
    if not series or now.timestamp()-series[-1][0]>=8:
        series.append((now.timestamp(),midpoint))
    series[:]=series[-90:]
    older=[v for timestamp,v in series if timestamp<=now.timestamp()-20]
    try:
        calc=enrich(estimate(market,older[-1] if older else None),macro_data,news_data,now)
    except Exception as exc:
        st.warning(f'Gold heuristic unavailable: {exc}')

# 1. A glance: visible by default.
with st.container(border=True):
    st.subheader('Live Market at a Glance')
    if market:
        a,b=st.columns([2,1])
        with a:
            st.markdown(f'**{market.get("title", "Gold 15-minute UP/DOWN")}**')
            st.code(str(market.get('ticker','Unknown')))
            st.caption(f'Closes {et_str(market.get("close_time"))} · Kalshi market data')
        with b:
            close=utc(market.get('close_time'))
            if close: timer(close)
        a,b,c,d=st.columns(4)
        a.metric('YES bid',pct_text(ba[0]) if ba else '—')
        b.metric('YES ask',pct_text(ba[1]) if ba else '—')
        c.metric('YES midpoint',pct_text(sum(ba)/2) if ba else '—')
        d.metric('Elapsed', f'{minute} / 15 min' if minute is not None else '—')
        s1,s2,s3=st.columns(3)
        s1.metric('Gold forecast',str(calc.get('direction','—')) if calc else '—')
        s2.metric('Heuristic YES score',pct_text(calc.get('p_yes')) if calc else '—')
        s3.metric('Spot gold / troy oz (USD)',f'${spot_result["price"]:,.2f}' if spot_result else 'Unavailable')
        st.caption('Kalshi settles against its specified Gold reference, not necessarily this independent XAU/USD spot feed. Heuristic score is NOT calibrated probability.')
        if minute in CHECKPOINTS:
            st.info(f'Observation checkpoint minute {minute}. Your Railway worker records forecasts independently when enabled.')
    else:
        st.info('No active Kalshi 15-minute Gold market detected. The market may be closed.')
        if market_error: st.warning(market_error)
        if spot_result: st.metric('Gold spot / troy oz (USD)',f'${spot_result["price"]:,.2f}')
    if spot_error: st.caption('Spot gold feed issue: '+spot_error)

# 2. Chart (spot XAU/USD, not Kalshi YES contract midpoint)
with st.container(border=True):
    st.subheader('Current Gold Price Chart')
    a,b=st.columns([2,1])
    with a:
        tf=st.selectbox('Chart timeframe (visible history)',list(SPAN_OPTIONS),index=3,key='gold_chart_timeframe')
    with b:
        st.metric('Current gold spot / oz',f'${spot_result["price"]:,.2f}' if spot_result else 'Unavailable')
    td_key=secret_value('gold_market_data','twelvedata_api_key')
    if not td_key:
        st.caption('Showing observed spot-price samples from this browser session only. Add [gold_market_data] twelvedata_api_key for historical 1m–yearly XAU/USD charts.')
    try:
        prices=chart_data(tf,td_key)
        if len(prices)>1:
            import altair as alt
            prices=prices.copy()
            prices['eastern_time']=prices['time'].dt.tz_convert(ET).dt.tz_localize(None)
            line=(alt.Chart(prices).mark_line(strokeWidth=2).encode(
                x=alt.X('eastern_time:T',title='Eastern time',axis=alt.Axis(format='%b %d %I:%M %p')),
                y=alt.Y('close:Q',scale=alt.Scale(zero=False),axis=alt.Axis(orient='right',title='USD per troy oz')),
                tooltip=[alt.Tooltip('time:T',title='UTC'),alt.Tooltip('close:Q',format='$,.2f')]
            ).properties(height=340).interactive())
            st.altair_chart(line,use_container_width=True)
            st.caption(f'{len(prices)} observed bars · independent XAU/USD feed · chart timeframe {tf} · price axis on right')
        else:
            st.info('Not enough observed Gold price points in this timeframe. Historical candles require a configured XAU/USD history provider; no missing points are fabricated.')
    except Exception as exc:
        st.warning('Historical Gold chart unavailable: '+str(exc)[:250])

# 3. Collapsible technical dashboard
with st.expander('Technical Indicator Dashboard — EMA / RSI / MACD / momentum',expanded=False):
    if td_key:
        try:
            candles=twelve_data_candles(td_key,'1min',160)
            indicator=indicator_values(candles)
            if indicator:
                a,b,c=st.columns(3)
                a.metric('EMA direction',indicator['EMA direction'])
                b.metric('RSI 14',f'{indicator["RSI 14"]:.1f}')
                c.metric('MACD',f'{indicator["MACD"]:+.4f}')
                st.dataframe(pd.DataFrame([indicator]).T.rename(columns={0:'Gold 1m reading'}),use_container_width=True)
                st.caption('Technical indicators are computed from independent XAU/USD 1-minute candles, not Kalshi contract quotes. Not validated trade probabilities.')
            else:
                st.info('At least 26 Gold candles are required for the indicator dashboard.')
        except Exception as exc:
            st.warning('Indicator feed unavailable: '+str(exc)[:220])
    else:
        st.info('Set [gold_market_data] twelvedata_api_key in Streamlit Secrets to enable historically sourced EMA, RSI, MACD and momentum. No synthetic readings are displayed.')

# 4. Calibrated probability and entry timing
with st.container(border=True):
    st.subheader('Calibrated Probability & Entry Timing')
    calibration=calibrated_gold_model(forward)
    if len(calibration)==3:
        cal_model, cal_status, quality=calibration
    else:
        cal_model, cal_status=calibration; quality=None
    if quality:
        x,y,z=st.columns(3)
        x.metric('Holdout markets',quality['holdout_markets'])
        y.metric('Calibrated Brier',f'{quality["holdout_brier"]:.5f}')
        z.metric('Market Brier',f'{quality["market_brier"]:.5f}')
    if cal_model and calc:
        p_up=float(cal_model.predict([float(calc['p_yes'])])[0])
        x,y=st.columns(2)
        x.metric('Calibrated Gold UP probability',f'{p_up:.1%}')
        y.metric('Calibrated Gold DOWN probability',f'{1-p_up:.1%}')
        st.success(cal_status)
    else:
        st.info('Calibrated probabilities unavailable: '+cal_status)
        p_up=None
    st.caption('Calibration uses earliest 70% of distinct settled Gold contracts and a later 30% untouched holdout. Approval requires better Brier than the same holdout market midpoint. No BTC models are reused.')
    if market and minute is not None:
        next_checkpoint=next((m for m in sorted(CHECKPOINTS) if m>minute),None)
        st.write('**Next research checkpoint:**',f'Minute {next_checkpoint}' if next_checkpoint else 'No remaining checkpoints this contract')

# 5. Live entry monitor
with st.container(border=True):
    st.subheader('Live Kalshi Entry Timing Monitor')
    if not market:
        st.info('Waiting for the next open Gold market.')
    elif not ba:
        st.warning('No reliable Gold quote; entry observation suppressed.')
    else:
        spread=ba[1]-ba[0]
        x,y,z=st.columns(3)
        x.metric('Market minute',minute if minute is not None else '—')
        y.metric('YES spread',f'{spread*100:.1f}¢')
        z.metric('Checkpoint active','YES' if minute in CHECKPOINTS else 'NO')
        if cal_model is None:
            st.warning('OBSERVATION ONLY — no calibrated Gold model has passed the holdout gate. No live entry approval.')
        elif minute not in CHECKPOINTS:
            st.info('Waiting for a Gold observation checkpoint. No order submission.')
        else:
            yes_edge=(p_up-ba[1]) if p_up is not None else None
            if yes_edge is not None:
                st.metric('Raw YES probability minus ask',f'{yes_edge*100:+.2f}¢')
                st.caption('This raw difference excludes actual fees, slippage, fill odds, and risk controls; it is not a live-order permission.')
    st.caption('Refreshes with the dashboard every 10s; checkpoints: '+', '.join(map(str,CHECKPOINTS))+'. Production/DEMO auto-execution is not implemented here.')

# 6. Forward quality retained from original app
with st.expander('Forward Performance & Model Quality',expanded=True):
    if forward_error: st.error(f'Gold history read failed: {forward_error}. Check RLS and read-only publishable key.')
    if not all(config_values): st.warning('Gold history read-only connection not configured. Set [history_store] url and publishable_key.')
    report(forward_rows)

# 7. Journal and note-taking; local drafts only until secure backend available
with st.expander('Trading History, Notes & Backtesting',expanded=False):
    st.caption('The original Gold app is read-only; it cannot automatically detect real orders/fills or write journal records. A separately authenticated order/fill syncing backend is required for automatic live trade history.')
    st.session_state.setdefault('gold_session_journal',[])
    with st.form('gold_journal_form',clear_on_submit=True):
        a,b,c=st.columns(3)
        side=a.selectbox('Side',['UP / YES','DOWN / NO'])
        amount=b.number_input('Stake ($)',min_value=0.01,max_value=100000.0,value=5.0,step=1.0)
        outcome=c.selectbox('Result',['Pending','Won','Lost'])
        notes=st.text_area('Trade notes (manual journal, not executed by this app)',max_chars=2000)
        submitted=st.form_submit_button('Add manual journal entry')
        if submitted:
            st.session_state.gold_session_journal.append({
                'recorded_at_et':datetime.now(ET).isoformat(),
                'market_ticker':market.get('ticker','') if market else '',
                'side':side,'stake_usd':amount,'outcome':outcome,'notes':notes,
                'source':'manual_session_only'})
            st.success('Manual journal entry added to this browser session. Export it to preserve it.')
    journal=st.session_state.gold_session_journal
    if journal:
        st.dataframe(pd.DataFrame(journal).iloc[::-1],use_container_width=True,hide_index=True)
        st.download_button('Download manual Gold journal CSV',pd.DataFrame(journal).to_csv(index=False),
                           'gold_manual_session_journal.csv','text/csv')
    st.markdown('**Historical research/backtesting — Gold-only**')
    st.caption('Illustrative, chronological holdout-only settlement model; UP/YES entries only because the current Gold ledger lacks actual NO-ask history. No actual fills assumed.')
    if cal_model is None:
        st.info('Backtest unavailable until the Gold-only model passes the distinct-market/Brier quality gate.')
    else:
        tested=paper_backtest(forward,cal_model)
        if not tested.empty:
            a,b,c=st.columns(3)
            a.metric('Illustrative eligible entries',len(tested))
            b.metric('Modeled net P/L',f'${tested.net_pnl.sum():+.2f}')
            c.metric('Modeled win rate',f'{100*(tested.yes_won==1).mean():.1f}%')
            st.dataframe(tested[['market_ticker','observed_at','calibrated_p_up','entry_price','contracts','cost','yes_won','net_pnl']],
                         use_container_width=True,hide_index=True)
        else:
            st.info('No eligible holdout UP/YES examples under the illustrative $5/3¢ edge rule.')

# 8. Long-term analytics, backups
with st.expander('Lifetime History Analytics & Backups',expanded=False):
    st.caption('Lifetime scope is the latest 3,000 records returned by the existing Gold forward ledger, not a guaranteed complete database lifetime.')
    if not forward.empty:
        settled=forward[forward.yes_won.isin([0,1]) & forward.p_yes.between(0,1)].copy()
        x,y,z=st.columns(3)
        x.metric('Loaded forward observations',len(forward))
        y.metric('Distinct Gold contracts',forward.market_ticker.nunique())
        z.metric('Settled observations',len(settled))
        if len(settled):
            settled['correct']=((settled.p_yes>=.5).astype(int)==settled.yes_won).astype(int)
            st.metric('Loaded historical directional accuracy',f'{100*settled.correct.mean():.1f}%')
    else:
        st.info('No Gold forward history available.')
    st.download_button('Download Gold research + session journal backup ZIP',
                       export_zip(forward_rows,st.session_state.setdefault('gold_session_journal',[])),
                       'gold_history_backup.zip','application/zip')
    st.caption('Restore requires a separate protected server process; this public read-only dashboard intentionally cannot upload or overwrite Supabase records.')

# 9. Actual scoreboard instead of made-up public Kalshi leaderboards
with st.expander('Performance Leaderboards',expanded=False):
    st.caption('Ranks the actual Gold observation checkpoints available in your stored forward ledger. Public Kalshi trader rankings are not available from this app; no account names or leaderboards are fabricated.')
    if not forward.empty:
        scored=forward[forward.yes_won.isin([0,1]) & forward.p_yes.between(0,1)].copy()
        if len(scored):
            scored['brier']=(scored.p_yes-scored.yes_won)**2
            scored['hit']=((scored.p_yes>=.5).astype(int)==scored.yes_won).astype(int)
            leaderboard=(scored.groupby('minute').agg(Forecasts=('hit','size'),
                Distinct_markets=('market_ticker','nunique'),Accuracy=('hit','mean'),
                Brier=('brier','mean')).reset_index().sort_values('Brier'))
            leaderboard['Accuracy']=leaderboard['Accuracy'].mul(100).round(1)
            leaderboard.index=range(1,len(leaderboard)+1)
            leaderboard.index.name='Rank'
            st.dataframe(leaderboard.rename(columns={'minute':'Checkpoint minute','Distinct_markets':'Distinct markets','Accuracy':'Accuracy %'}),
                         use_container_width=True)
        else:
            st.info('No scored Gold forecasts yet.')
    else:
        st.info('No Gold observations to rank.')

# Preserve original FRED and news context
with st.expander('Economic Fundamentals & Gold News Context',expanded=False):
    st.caption('Gold-specific FRED macro data and GDELT headlines from the existing application. Context only; neither source proves a short-term trading edge.')
    if macro_data.get('status')=='not_configured':
        st.info('Configure [gold_fundamentals] fred_api_key to enable macro readings.')
    else:
        st.caption('FRED status: '+str(macro_data.get('status'))+' · retrieved: '+et_str(macro_data.get('fetched_at')))
    if macro_data.get('series'):
        macro_rows=[{'Series':sid,'Indicator':obs.get('name'),'Value':obs.get('value'),
                     'Change vs previous':obs.get('change'),'Observation date':obs.get('date')}
                    for sid,obs in macro_data['series'].items()]
        st.dataframe(pd.DataFrame(macro_rows),use_container_width=True,hide_index=True)
    st.caption('News status: '+str(news_data.get('status'))+' · retrieved: '+et_str(news_data.get('fetched_at')))
    if news_data.get('articles'):
        headlines=pd.DataFrame(news_data['articles'])
        columns=[c for c in ('seen_at','title','domain','tone_label','event_watch','url') if c in headlines.columns]
        st.dataframe(headlines[columns].head(15),hide_index=True,use_container_width=True)
    else:
        st.info('No eligible news headlines available.')

st.divider()
st.caption('Gold dashboard research only. Source app_gold.py market discovery, macro/news enrichment, and read-only gold_forward_signals integration retained. BTC tables and services unchanged. No real/demo Kalshi trading, order sync, or automatic write-back is enabled.')
