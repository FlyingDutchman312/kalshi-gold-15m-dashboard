"""Offline regression tests (no network, no order endpoints)."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import gold_core as g

NOW=datetime(2026,11,16,13,5,tzinfo=timezone.utc)

def market(close=NOW+timedelta(minutes=10)):
    return {'ticker':'KXGOLD15M-26NOV161300-00', 'status':'active', 'open_time':(close-timedelta(minutes=15)).isoformat(),
            'close_time':close.isoformat(),'yes_bid_dollars':'0.6000','yes_ask_dollars':'0.6200'}

class GoldTest(unittest.TestCase):
    def test_discovery_and_minute(self):
        with patch.object(g,'get_json',return_value={'markets':[market()]}):
            m=g.discover(NOW)
        self.assertEqual(g.minute_of_market(m,NOW),5)
    def test_quotes_and_heuristic(self):
        est=g.estimate(market(), .58)
        self.assertAlmostEqual(est['mid'],.61)
        self.assertAlmostEqual(est['p_yes'],.616)
        self.assertEqual(est['direction'],'UP')
    def test_no_missing_quote_as_zero(self):
        m=market();m.pop('yes_bid_dollars')
        self.assertIsNone(g.estimate(m))
    def test_no_wrong_series(self):
        m=market();m['ticker']='KXBTC15M-26NOV161300-00'
        with patch.object(g,'get_json',return_value={'markets':[m]}):
            self.assertIsNone(g.discover(NOW))
    def test_save_observation(self):
        with patch.object(g,'supa',return_value=[]) as f:
            row=g.save_prediction(market(),5,g.estimate(market()),('u','k'),NOW)
            self.assertEqual(row['decision'],'OBSERVATION')
            self.assertEqual(f.call_args.args[1],'gold_forward_signals')
    def test_never_store_after_close(self):
        with self.assertRaises(ValueError):
            g.save_prediction(market(),5,g.estimate(market()),('u','k'),NOW+timedelta(minutes=11))
    def test_gold_settlement_yes(self):
        with patch.object(g,'get_json',return_value={'market':{'status':'settled','result':'yes'}}):
            self.assertEqual(g.settled_result('KXGOLD15M-26NOV161300-00'),1)
    def test_open_result_not_scored(self):
        with patch.object(g,'get_json',return_value={'market':{'status':'open','result':'yes'}}):
            self.assertIsNone(g.settled_result('KXGOLD15M-26NOV161300-00'))

if __name__=='__main__':unittest.main()
