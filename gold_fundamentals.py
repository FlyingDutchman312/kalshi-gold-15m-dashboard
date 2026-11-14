"""As-observed macro/news context for KXGOLD15M; never an execution signal.

FRED daily/monthly observations are NOT high-frequency ticks. A reading is only
usable once it has actually been retrieved during the running system, with the
retrieval timestamp persisted alongside the forward forecast. Never use this
module to reconstruct historical information availability from FRED dates.
"""
import os
from datetime import datetime, timezone, timedelta
import requests

FRED_URL = 'https://api.stlouisfed.org/fred/series/observations'
GDELT_URL = 'https://api.gdeltproject.org/api/v2/doc/doc'
SERIES = {
    'DFII10': ('10Y real Treasury yield', 'pct'),
    'DGS10': ('10Y nominal Treasury yield', 'pct'),
    'DTWEXBGS': ('Broad trade-weighted USD', 'index'),
    'VIXCLS': ('VIX volatility index', 'index'),
    'CPIAUCSL': ('US CPI (monthly)', 'index'),
    'UNRATE': ('US unemployment rate (monthly)', 'pct'),
}
NEWS_QUERY = '(gold OR bullion) (Federal Reserve OR inflation OR interest rates OR Treasury OR dollar OR central bank)'
POSITIVE_PHRASES = ('gold rises', 'gold rallies', 'gold gains', 'gold jumps', 'gold surges',
                    'gold climbs', 'gold advances', 'gold rebounds', 'dollar weakens',
                    'dollar falls', 'yields fall', 'yields decline', 'rates cut', 'rate cut')
NEGATIVE_PHRASES = ('gold falls', 'gold drops', 'gold slides', 'gold tumbles', 'gold sinks',
                    'gold declines', 'gold retreats', 'dollar strengthens', 'dollar rises',
                    'yields rise', 'yields jump', 'rates hike', 'rate hike')
EVENT_PHRASES = ('cpi', 'inflation report', 'jobs report', 'nonfarm payroll', 'fomc',
                 'interest rate decision', 'federal reserve', 'fed chair', 'powell',
                 'pce inflation', 'unemployment')


def stamp(dt):
    return dt.astimezone(timezone.utc).isoformat()


def parse_dt(raw):
    if not raw: return None
    s = str(raw).strip()
    if s.endswith('Z'): s = s[:-1] + '+00:00'
    if len(s) == 14 and s.isdigit():
        return datetime.strptime(s, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
    try:
        value = datetime.fromisoformat(s)
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    except ValueError:
        return None


def fred_snapshot(api_key=None, now=None, session=None):
    now = now or datetime.now(timezone.utc)
    api_key = api_key or os.environ.get('FRED_API_KEY', '').strip()
    if not api_key:
        return {'status':'not_configured', 'fetched_at':stamp(now), 'series':{},
                'message':'Add FRED_API_KEY to use macro data.'}
    client = session or requests
    observations, failures = {}, {}
    for sid, (title, unit) in SERIES.items():
        try:
            response = client.get(FRED_URL, params={'series_id':sid, 'api_key':api_key,
                'file_type':'json', 'sort_order':'desc', 'limit':12}, timeout=12)
            response.raise_for_status()
            if 'json' not in response.headers.get('Content-Type','').lower():
                raise ValueError('FRED response was not JSON')
            payload = response.json()
            if not isinstance(payload,dict): raise ValueError('Invalid FRED document')
            valid = []
            for obs in payload.get('observations', []):
                try:
                    value = float(obs.get('value'))
                    if not (-1e9 < value < 1e9): continue
                    valid.append((obs['date'], value))
                except (TypeError, ValueError, KeyError): continue
            if valid:
                newest, latest = valid[0]
                prev = valid[1][1] if len(valid)>1 else None
                observations[sid] = {'name':title, 'unit':unit, 'date':newest,
                    'value':latest, 'previous':prev,
                    'change':latest-prev if prev is not None else None}
            else: failures[sid] = 'No numeric observations'
        except (requests.RequestException, ValueError, KeyError) as ex:
            failures[sid] = type(ex).__name__
    status = 'ok' if len(observations)==len(SERIES) else 'partial' if observations else 'unavailable'
    return {'status':status, 'fetched_at':stamp(now), 'series':observations,
            'errors':failures, 'source':'FRED / Federal Reserve Bank of St. Louis'}


def news_snapshot(now=None, session=None):
    now=now or datetime.now(timezone.utc)
    client=session or requests
    try:
        response=client.get(GDELT_URL, params={'query':NEWS_QUERY, 'mode':'ArtList',
            'format':'json', 'timespan':'12h', 'sort':'DateDesc', 'maxrecords':35}, timeout=15)
        response.raise_for_status()
        if 'json' not in response.headers.get('Content-Type','').lower():
            raise ValueError('GDELT response was not JSON')
        data=response.json()
        if not isinstance(data,dict): raise ValueError('Invalid GDELT response')
        articles=[]
        seen_urls=set()
        for item in data.get('articles', []):
            title=str(item.get('title') or '')[:250]
            seen=parse_dt(item.get('seendate'))
            # Only use articles observed by GDELT on or before forecast creation.
            if not seen or seen>now or (now-seen)>timedelta(hours=8): continue
            url=str(item.get('url') or '')[:600]
            if not title or not url or url in seen_urls: continue
            seen_urls.add(url)
            lower=title.lower()
            pos=any(phrase in lower for phrase in POSITIVE_PHRASES)
            neg=any(phrase in lower for phrase in NEGATIVE_PHRASES)
            direction = 1 if pos and not neg else -1 if neg and not pos else 0
            event=any(term in lower for term in EVENT_PHRASES)
            articles.append({'title':title, 'url':url, 'seen_at':stamp(seen),
                'domain':str(item.get('domain') or '')[:120],
                'tone_label': 'gold_supportive' if direction>0 else 'gold_adverse' if direction<0 else 'unclear',
                'direction':direction, 'event_watch':event})
        score=sum(x['direction'] for x in articles)
        return {'status':'ok', 'fetched_at':stamp(now), 'articles':articles[:20],
                'directional_balance':score, 'event_count':sum(x['event_watch'] for x in articles),
                'source':'GDELT DOC 2.0', 'method':'headline phrase rules; not sentiment ML'}
    except (requests.RequestException, ValueError) as ex:
        return {'status':'unavailable', 'fetched_at':stamp(now), 'articles':[],
                'directional_balance':0, 'event_count':0,
                'source':'GDELT DOC 2.0', 'error':type(ex).__name__}


def context_adjustments(macro, news, now=None):
    """Very small, explicitly uncalibrated, transparent contextual nudges.

    No guessing from historical releases. Stale/missing sources contribute zero.
    Rate changes are percentage points (e.g. 0.03 is 3bp), USD change index pts.
    """
    now=now or datetime.now(timezone.utc)
    macro_adj=0.0; news_adj=0.0; reasons=[]
    mtime=parse_dt((macro or {}).get('fetched_at'))
    if mtime and timedelta(0)<=now-mtime<=timedelta(hours=6) and macro.get('status') in ('ok','partial'):
        entries=macro.get('series', {})
        real=entries.get('DFII10',{}).get('change')
        usd=entries.get('DTWEXBGS',{}).get('change')
        # Heuristic daily-context signs, NOT fitted predictive coefficients.
        if real is not None and abs(real)>=0.02:
            macro_adj+=(-0.004 if real>0 else .004)
            reasons.append('real yields ' + ('higher' if real>0 else 'lower'))
        if usd is not None and abs(usd)>=0.1:
            macro_adj+=(-0.004 if usd>0 else .004)
            reasons.append('broad USD ' + ('higher' if usd>0 else 'lower'))
    ntime=parse_dt((news or {}).get('fetched_at'))
    if ntime and timedelta(0)<=now-ntime<=timedelta(minutes=20) and news.get('status')=='ok':
        balance=int(news.get('directional_balance') or 0)
        if abs(balance)>=2:
            news_adj=.005 if balance>0 else -.005
            reasons.append('headline balance ' + ('supportive' if balance>0 else 'adverse'))
    macro_adj=max(-.008,min(.008,macro_adj))
    news_adj=max(-.005,min(.005,news_adj))
    return {'macro_adjustment':macro_adj,'news_adjustment':news_adj,
            'total_adjustment':macro_adj+news_adj,'reasons':reasons,
            'status':'context_available' if reasons else 'neutral_or_unavailable',
            'caution':'Heuristic context; not validated for 15-minute gold markets.'}


def enrich(base, macro=None, news=None, now=None):
    if not base: return None
    now=now or datetime.now(timezone.utc)
    signals=context_adjustments(macro,news,now)
    p=max(.01,min(.99,float(base['p_yes'])+signals['total_adjustment']))
    result=dict(base)
    result.update({'p_yes':p,'direction':'UP' if p>=.5 else 'DOWN',
                   'quote_p_yes':float(base['p_yes']),'fundamentals':{
                       'asof':stamp(now), 'macro':macro or {}, 'news':news or {},
                       'adjustments':signals},
                   'model_version':'gold_macro_news_heuristic_v2'})
    return result
