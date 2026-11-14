"""Independent, read-only Kalshi Gold 15-minute prediction and forward-test dashboard."""
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components
from gold_core import (SERIES, CHECKPOINTS, discover, bid_ask, estimate, minute_of_market,
                       utc, store_config, supa)
from gold_fundamentals import fred_snapshot, news_snapshot, enrich

st.set_page_config(page_title='Kalshi Gold 15-Minute Prediction Desk', page_icon='🥇',layout='wide')
st.markdown('''<style>
.block-container {padding-top:1.3rem;max-width:1200px}
div[data-testid="stMetric"] {background:rgba(126,121,104,.08);border:1px solid rgba(145,143,130,.18);padding:14px;border-radius:12px}
</style>''', unsafe_allow_html=True)
ET=ZoneInfo('America/New_York')


def config():
    u,k=store_config()
    if u and k: return u,k
    try:
        group=st.secrets.get('history_store',{})
        u=str(group.get('url','')).strip().rstrip('/')
        if u.endswith('/rest/v1'): u=u[:-8]
        k=str(group.get('secret_key','')).strip()
    except Exception: pass
    return u,k


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
    return supa('GET','gold_forward_signals',cfg,
                {'select':'market_ticker,minute,observed_at,p_yes,yes_bid,yes_ask,baseline_mid,direction,decision,yes_won,model_version',
                 'order':'observed_at.desc','limit':'3000'})


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


st.title('🥇 Kalshi Gold · 15-Minute Prediction Desk')
st.caption('Separate from your BTC dashboard · KXGOLD15M · Eastern time · quote, macro and headline context · no order execution')
st_autorefresh(interval=10000,key='gold_dashboard_refresh')
now=datetime.now(timezone.utc)
config_values=config()
col1,col2=st.columns([2,1])
with col1:
    st.markdown('**Current Gold 15-minute contract**')
    try:
        market=discover(now)
        error=None
    except Exception as ex:
        market=None;error=f'{type(ex).__name__}: {ex}'
    if market:
        ticker=market['ticker']
        st.code(ticker)
        st.caption(f'Closes: {et_str(market.get("close_time"))} · Data from Kalshi public market API')
        st.write(market.get('title','Gold 15-minute up or down'))
    else:
        st.info('No currently tradable 15-minute Gold contract. Gold markets may pause on weekends or during exchange closures.')
        if error: st.warning(f'Market data unavailable: {error}')
with col2:
    if market: timer(utc(market.get('close_time')))
    else:
        components.html('''<div style="font: 19px system-ui;text-align:center;padding:22px;color:#c4b38c;background:#171a23;border-radius:12px">Market closed / unavailable</div>''',height=86)

if market:
    ba=bid_ask(market)
    minute=minute_of_market(market,now)
    if ba:
        bid,ask=ba
        st.session_state.setdefault('gold_quote_history',{})
        quote_history=st.session_state['gold_quote_history']
        series=quote_history.setdefault(market['ticker'],[])
        mid=(bid+ask)/2
        if not series or now.timestamp()-series[-1][0]>=8:
            series.append((now.timestamp(),mid))
        series[:]=series[-90:]
        history=[v for t,v in series if t<=now.timestamp()-20]
        macro_data=cached_macro(fred_key())
        news_data=cached_news()
        calc=enrich(estimate(market,history[-1] if history else None),macro_data,news_data,now)
        c1,c2,c3,c4=st.columns(4)
        c1.metric('YES bid',f'{bid:.1%}')
        c2.metric('YES ask',f'{ask:.1%}')
        c3.metric('YES quote midpoint',f'{mid:.1%}')
        c4.metric('Market elapsed',f'{minute} / 15 min' if minute is not None else '—')
        st.subheader('Gold UP / DOWN observation')
        st.metric('Uncalibrated Gold context forecast',calc['direction'],
                  f'YES heuristic score {calc["p_yes"]:.1%}')
        st.caption(f'Quote-only YES score {calc["quote_p_yes"]:.1%} · ' +
                   f'Macro adjustment {calc["fundamentals"]["adjustments"]["macro_adjustment"]:+.2%} · ' +
                   f'News adjustment {calc["fundamentals"]["adjustments"]["news_adjustment"]:+.2%}')
        st.caption('The displayed score is a heuristic derived from Kalshi quotes, NOT a calibrated probability, guaranteed confidence, or trade recommendation. ' 
                   'Gold settles using Kalshi’s stated Pyth Gold rules; an unrelated XAU/USD spot feed could differ.')
        if len(series)>1:
            frame=pd.DataFrame(series,columns=['unix','YES midpoint'])
            frame['Eastern time']=pd.to_datetime(frame.unix,unit='s',utc=True).dt.tz_convert(ET)
            st.line_chart(frame.set_index('Eastern time')[['YES midpoint']],height=220)
        if minute in CHECKPOINTS:
            st.info(f'Checkpoint minute {minute}: the separate Gold worker records a prediction automatically if enabled. The web dashboard itself does not write or duplicate it.')
        else:
            st.caption(f'Observation checkpoints: {CHECKPOINTS}')
    else:
        st.warning('Market listed, but no valid YES bid/ask quote. Predictions are suppressed instead of invented.')

st.divider()
st.subheader('Economic fundamentals & news-event context')
macro_data=cached_macro(fred_key())
news_data=cached_news()
st.caption('Daily/monthly macro releases and news headlines offer market context, not a proven 15-minute trading advantage. Live data availability may lag economic releases.')
if macro_data.get('status')=='not_configured':
    st.info('Add a FRED API key in [gold_fundamentals] fred_api_key to enable macro readings. Without it, macro impact is neutral.')
else:
    st.caption('FRED macro status: '+str(macro_data.get('status'))+' · retrieved: '+et_str(macro_data.get('fetched_at')))
if macro_data.get('series'):
    macro_rows=[]
    for sid, obs in macro_data['series'].items():
        macro_rows.append({'Series':sid,'Indicator':obs['name'], 'Value':obs['value'],
                           'Change vs previous':obs['change'],'Observation date':obs['date']})
    st.dataframe(pd.DataFrame(macro_rows),hide_index=True,use_container_width=True)
st.caption('GDELT headline status: '+str(news_data.get('status'))+' · retrieved: '+et_str(news_data.get('fetched_at')) +
           ' · macro-event watch headlines: '+str(news_data.get('event_count',0)))
if news_data.get('articles'):
    df_news=pd.DataFrame(news_data['articles'])
    show=['seen_at','title','domain','tone_label','event_watch','url']
    st.dataframe(df_news[show].head(15),hide_index=True,use_container_width=True,
                 column_config={'url':st.column_config.LinkColumn('Article')})
else:
    st.caption('No eligible current headlines (or news feed unavailable); news adjustment remains neutral.')
st.caption('Headline scores use literal phrase rules only. News can be ambiguous, incorrect, delayed, or already reflected in Kalshi odds. Avoid treating these as calibrated signals.')

st.divider()
if all(config_values):
    try:
        rows=records(config_values)
        st.caption('Gold Supabase history connected — stored in gold_forward_signals (separate from BTC).')
        report(rows)
    except Exception as ex:
        st.error(f'Cannot read Gold forward results: {type(ex).__name__}: {ex}. Run gold_store.sql first, and check [history_store] secrets.')
else:
    st.warning('Persistent Gold history not connected. Add the existing BTC Supabase [history_store] url and secret_key to this new Streamlit app’s Secrets.')
    report([])
st.divider()
st.caption('Observation-only prototype. No automatic demo trading, cash-out, order submission, or claims of a trained Gold model. Existing Bitcoin services and tables are not changed.')
