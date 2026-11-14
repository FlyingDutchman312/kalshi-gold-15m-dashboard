"""Market-quote based Gold KXGOLD15M observation engine. NO order endpoints."""
import os
import time
from datetime import datetime, timezone
from urllib.parse import quote
import requests

SERIES = 'KXGOLD15M'
API = 'https://external-api.kalshi.com/trade-api/v2'
CHECKPOINTS = (3, 5, 7, 9, 11)


def utc(s):
    if not s: return None
    return datetime.fromisoformat(str(s).replace('Z', '+00:00')).astimezone(timezone.utc)


def number(x):
    try:
        z = float(x)
        return z if 0 <= z <= 1 else None
    except (TypeError, ValueError):
        return None


def get_json(path, params=None, session=None):
    client = session or requests
    response = client.get(API + path, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict): raise ValueError('Kalshi response was not a JSON object')
    return data


def discover(now=None, session=None):
    now = now or datetime.now(timezone.utc)
    result = get_json('/markets', {'series_ticker': SERIES, 'status': 'open', 'limit': 100}, session)
    candidates = []
    for m in result.get('markets', []):
        if not str(m.get('ticker', '')).startswith(SERIES + '-'): continue
        close = utc(m.get('close_time'))
        opened = utc(m.get('open_time'))
        if not close or not (0 < (close-now).total_seconds() <= 900): continue
        # Guard against old markets whose trading windows don't match a 15m contract.
        if opened and now < opened: continue
        candidates.append(m)
    candidates.sort(key=lambda m: utc(m['close_time']))
    return candidates[0] if candidates else None


def bid_ask(m):
    bid = number(m.get('yes_bid_dollars'))
    ask = number(m.get('yes_ask_dollars'))
    # Older snapshots may use cents; do not interpret a missing quote as zero.
    if bid is None and m.get('yes_bid') is not None: bid = number(float(m['yes_bid'])/100)
    if ask is None and m.get('yes_ask') is not None: ask = number(float(m['yes_ask'])/100)
    if bid is None or ask is None or bid <= 0 or ask >= 1 or bid > ask: return None
    return bid, ask


def minute_of_market(m, now=None):
    now = now or datetime.now(timezone.utc)
    close = utc(m.get('close_time'))
    if not close: return None
    elapsed = now.timestamp() - (close.timestamp()-900)
    if elapsed < 0 or elapsed >= 900: return None
    return int(elapsed//60)


def estimate(m, previous_mid=None):
    """Uncalibrated quote-momentum heuristic; not a statistical confidence level."""
    ba = bid_ask(m)
    if not ba: return None
    bid, ask = ba
    mid = (bid+ask)/2
    spread = ask-bid
    # 20% of the observed quote midpoint change, capped at 3pp to avoid false precision.
    move = max(-.03, min(.03, .2*(mid-previous_mid))) if previous_mid is not None else 0.0
    p_yes = max(.01, min(.99, mid+move))
    return {'p_yes': p_yes, 'mid': mid, 'bid': bid, 'ask': ask, 'spread': spread,
            'momentum_adjustment':move, 'direction':'UP' if p_yes >= .5 else 'DOWN'}


def settled_result(ticker, session=None):
    if not ticker.startswith(SERIES+'-'): raise ValueError('Non-Gold ticker')
    escaped = quote(ticker, safe='')
    for path in ('/markets/'+escaped, '/historical/markets/'+escaped):
        try: m = get_json(path, session=session).get('market', {})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404: continue
            raise
        result = str(m.get('result') or '').lower()
        if result in ('yes','no') and str(m.get('status','')).lower() == 'settled':
            return 1 if result == 'yes' else 0
    return None


def store_config():
    u=os.getenv('SUPABASE_URL','').strip().rstrip('/')
    k=os.getenv('SUPABASE_SECRET_KEY','').strip()
    if u.endswith('/rest/v1'): u=u[:-8]
    return u,k


def headers(key, prefer=None):
    h={'apikey':key, 'Content-Type':'application/json'}
    if not key.startswith('sb_secret_'): h['Authorization']='Bearer '+key
    if prefer: h['Prefer']=prefer
    return h


def supa(method, table, config, params=None, data=None):
    url,key=config
    if not url or not key: raise RuntimeError('SUPABASE_URL and SUPABASE_SECRET_KEY required')
    pref='resolution=ignore-duplicates,return=minimal' if method=='POST' else None
    resp=requests.request(method, f'{url}/rest/v1/{table}', params=params, headers=headers(key,pref), json=data, timeout=20)
    resp.raise_for_status()
    return resp.json() if resp.content else []


def save_prediction(m, minute, estimate_data, config, now=None):
    now=now or datetime.now(timezone.utc)
    ticker=str(m['ticker'])
    if not ticker.startswith(SERIES+'-') or minute not in CHECKPOINTS: raise ValueError('Invalid Gold market or minute')
    close=utc(m['close_time'])
    if not close or now >= close: raise ValueError('Closed market')
    row={'market_ticker':ticker, 'minute':minute, 'close_ts':int(close.timestamp()),
         'observed_at':now.isoformat(), 'p_yes':estimate_data['p_yes'],
         'yes_bid':estimate_data['bid'], 'yes_ask':estimate_data['ask'],
         'baseline_mid':estimate_data['mid'], 'direction':estimate_data['direction'],
         'decision':'OBSERVATION', 'model_version':'gold_quote_momentum_v1'}
    supa('POST','gold_forward_signals',config,{'on_conflict':'market_ticker,minute'},row)
    return row


def pending(config, limit=120):
    return supa('GET','gold_forward_signals',config,{'select':'market_ticker,close_ts','yes_won':'is.null','order':'close_ts.asc','limit':str(limit)})


def settle_pending(config, now=None):
    now=now or datetime.now(timezone.utc)
    found={}
    for row in pending(config):
        ticker=row['market_ticker']
        if ticker in found: continue
        if int(row['close_ts']) > now.timestamp()-120: continue
        result=settled_result(ticker)
        found[ticker]=result
        if result is not None:
            # Update only currently-null predictions, maintaining original probabilities.
            supa('PATCH','gold_forward_signals',config,
                 {'market_ticker':'eq.'+ticker,'yes_won':'is.null'},
                 {'yes_won':result,'settled_at':now.isoformat()})
    return found
