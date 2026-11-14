"""Separate 24/7 Gold OBSERVATION worker; zero trading capability."""
import logging
import time
from collections import deque
from datetime import datetime, timezone
from gold_core import (CHECKPOINTS, discover, estimate, minute_of_market, bid_ask,
                       store_config, save_prediction, settle_pending)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
LOG=logging.getLogger('gold-worker')


def main():
    config=store_config()
    if not all(config): raise RuntimeError('Supabase variables missing. Gold worker refused to start.')
    history={}
    attempted=set()
    previous_ticker=None
    last_settle=0
    last_idle=0
    LOG.info('KXGOLD15M gold observation worker started: 5-second polling; NO ORDER EXECUTION')
    while True:
        now=datetime.now(timezone.utc)
        try:
            market=discover(now)
            if market:
                ticker=market['ticker']
                if ticker!=previous_ticker:
                    LOG.info('GOLD MARKET FOUND ticker=%s close=%s',ticker,market.get('close_time'))
                    previous_ticker=ticker
                minute=minute_of_market(market,now)
                ba=bid_ask(market)
                if ba:
                    series=history.setdefault(ticker,deque(maxlen=100))
                    mid=sum(ba)/2
                    series.append((now.timestamp(),mid))
                    older=[v for t,v in series if t <= now.timestamp()-20]
                    prev=older[-1] if older else None
                    if minute in CHECKPOINTS and (ticker,minute) not in attempted:
                        est=estimate(market,prev)
                        if est:
                            row=save_prediction(market,minute,est,config,now)
                            attempted.add((ticker,minute))
                            LOG.info('GOLD FORECAST SAVED ticker=%s minute=%s direction=%s p_yes=%.4f',
                                     ticker,minute,row['direction'],row['p_yes'])
                elif minute in CHECKPOINTS and (ticker,minute) not in attempted:
                    LOG.warning('GOLD CHECKPOINT WAITING: invalid/missing YES bid and ask ticker=%s minute=%s',ticker,minute)
            elif time.monotonic()-last_idle > 60:
                LOG.info('GOLD MARKET CLOSED/UNAVAILABLE: waiting for an open KXGOLD15M market')
                last_idle=time.monotonic()
            if time.monotonic()-last_settle > 120:
                states=settle_pending(config,now)
                if states: LOG.info('GOLD SETTLEMENT CHECK: checked=%s resolved=%s',len(states),sum(v is not None for v in states.values()))
                last_settle=time.monotonic()
            if len(attempted)>2000:
                attempted={x for x in attempted if x[0]==previous_ticker}
            if len(history)>8:
                history={k:v for k,v in history.items() if k==previous_ticker}
        except Exception as exc:
            LOG.warning('GOLD FAIL-CLOSED (observation only): %s: %s',type(exc).__name__,exc)
        time.sleep(5)

if __name__=='__main__': main()
