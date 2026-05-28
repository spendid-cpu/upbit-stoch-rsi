"""
scanner.py v3.1.0 (Fixed for JSON Serialization & Safe Loop) - 1부
변경사항:
- v3.1.0: 다이버전스 탐지 추가, JSON 직렬화 오류 해결을 위한 안정성 강화 버전
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

import mtf_setup as _mtf
from mtf_setup import VERSION as MTF_VERSION

VERSION = 'v3.1.0'
PORT    = int(os.environ.get('PORT', '8080'))
KST     = timezone(timedelta(hours=9))

def _now() -> datetime:
    return datetime.now(KST)

def _now_iso() -> str:
    return _now().strftime('%Y-%m-%dT%H:%M:%S')

def _now_hm() -> str:
    return _now().strftime('%H:%M')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

SCAN_INTERVAL_MIN           = int(os.environ.get('SCAN_INTERVAL_MIN',         '60'))
WATCH_RESCAN_INTERVAL_MIN = int(os.environ.get('WATCH_RESCAN_INTERVAL_MIN', '15'))
PRICE_CHECK_INTERVAL_MIN  = int(os.environ.get('PRICE_CHECK_INTERVAL_MIN',  '1'))
ACTIVE_CHECK_INTERVAL_MIN = int(os.environ.get('ACTIVE_CHECK_INTERVAL_MIN', '1'))
DEEP_SCAN_INTERVAL_MIN    = int(os.environ.get('DEEP_SCAN_INTERVAL_MIN',    '5'))
DAILY_SUMMARY_HOUR_KST    = int(os.environ.get('DAILY_SUMMARY_HOUR_KST',   '9'))
DEEP_REBOUND_VALID_HOURS  = int(os.environ.get('DEEP_REBOUND_VALID_HOURS', '2'))

REQUEST_DELAY = float(os.environ.get('REQUEST_DELAY', '0.35'))
MAX_WORKERS   = int(os.environ.get('MAX_WORKERS',    '3'))
CANDLE_COUNT  = int(os.environ.get('CANDLE_COUNT',   '100'))

TRADE_TP_PCT    = float(os.environ.get('TRADE_TP_PCT',    '5.0'))
TRADE_SL_PCT    = float(os.environ.get('TRADE_SL_PCT',    '3.0'))
TRADE_TIMEOUT_H = int(os.environ.get('TRADE_TIMEOUT_H',  '48'))

BTC_DROP_1H_PCT = float(os.environ.get('BTC_DROP_1H_PCT', '-1.0'))
BTC_DROP_4H_PCT = float(os.environ.get('BTC_DROP_4H_PCT', '-2.0'))

WATCH_DROP_PCT = float(os.environ.get('WATCH_DROP_PCT', '-5.0'))
WATCH_RISE_PCT = float(os.environ.get('WATCH_RISE_PCT',  '8.0'))

WATCH_EXPIRE_DAYS = {'S': 7, 'A': 7, 'B': 5, 'C': 3, 'X': 1, '-': 1}
ALLOWED_GRADES    = {'S', 'A', 'B'}

STABLE_COINS = {
    'USDT','USDC','DAI','BUSD','TUSD','USDP','USDD',
    'USD1','FDUSD','PYUSD','SUSD','GUSD',
    'STETH','WBTC','CBBTC',
}

BASE_DIR     = os.environ.get('DATA_DIR', '/app/data')
WATCH_FILE   = os.path.join(BASE_DIR, 'watch_list.json')
ACTIVE_FILE  = os.path.join(BASE_DIR, 'active_list.json')
HISTORY_FILE = os.path.join(BASE_DIR, 'trade_history.json')
DEEP_FILE    = os.path.join(BASE_DIR, 'deep_list.json')
STATE_FILE   = os.path.join(BASE_DIR, 'scanner_state.json')
EVENT_FILE   = os.path.join(BASE_DIR, 'events.json')

os.makedirs(BASE_DIR, exist_ok=True)

_state_lock    = threading.Lock()
_scanner_state = {
    'version':              VERSION,
    'mtf_version':          MTF_VERSION,
    'running':              False,
    'watch_rescanning':      False,
    'price_checking':       False,
    'deep_scanning':        False,
    'last_scan':            None,
    'last_watch_rescan':    None,
    'last_price_check':     None,
    'last_deep_scan':       None,
    'next_scan':            None,
    'next_deep_scan':       None,
    'scan_count':           0,
    'watch_count':          0,
    'active_count':         0,
    'deep_count':           0,
    'btc_price':            None,
    'btc_price_usd':        None,
    'usdt_rate':            None,
    'btc_daily_ma20':       None,
    'btc_daily_ma20_usd':   None,
    'btc_daily_above':      None,
    'btc_daily_pct':        None,
    'btc_weekly_ma20':      None,
    'btc_weekly_ma20_usd':  None,
    'btc_weekly_above':     None,
    'btc_weekly_pct':       None,
    'btc_1h_pct':           None,
    'btc_4h_pct':           None,
    'btc_d_short_cycle':    None,
    'btc_d_mid_cycle':      None,
    'btc_h4_cycle':         None,
    'btc_h1_cycle':         None,
    'btc_m15_cycle':        None,
    'btc_m15_gc':           False,
    'btc_m5_cycle':         None,
    'btc_m5_gc':            False,
    'btc_entry_signal':     None,
    'btc_rebound_detected': False,
    'btc_rebound_at':       None,
    'total_trades':         0,
    'win_trades':           0,
    'total_pnl':            0.0,
    'error':                None,
}

def _load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        log.warning(f'JSON 로드 실패 {path}: {e}')
    return default

def _save_json(path: str, data):
    try:
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log.error(f'JSON 저장 실패 {path}: {e}')

def load_watch_list():   return _load_json(WATCH_FILE,   [])
def save_watch_list(d):  _save_json(WATCH_FILE,   d)
def load_active_list():  return _load_json(ACTIVE_FILE,  [])
def save_active_list(d): _save_json(ACTIVE_FILE,  d)
def load_history():      return _load_json(HISTORY_FILE, [])
def save_history(d):     _save_json(HISTORY_FILE, d)
def load_deep_list():    return _load_json(DEEP_FILE,    [])
def save_deep_list(d):   _save_json(DEEP_FILE,    d)
def load_events():       return _load_json(EVENT_FILE,   [])
def save_state():        _save_json(STATE_FILE, _scanner_state)

def _is_expired(item: dict) -> bool:
    try:
        exp = item.get('expire_at', '')
        if not exp: return False
        exp_dt = datetime.fromisoformat(exp)
        if exp_dt.tzinfo is None: exp_dt = exp_dt.replace(tzinfo=KST)
        return _now() > exp_dt
    except Exception: return False

def _calc_btc_entry_signal(d_short, d_mid, h4, h1, m15_gc=False, m5_gc=False) -> str:
    bad = ('PEAK', 'FALLING')
    if d_short in bad and d_mid in bad: return 'BLOCK'
    if d_short in bad: return 'CAUTION'
    if h4 in bad or h1 in bad: return 'CAUTION'
    if d_short in ('BOTTOM', 'RISING') and d_mid in ('BOTTOM', 'RISING'):
        if m15_gc and m5_gc: return 'GOOD+'
        return 'GOOD'
    return 'CAUTION'

def _get_deep_rs_grade(ticker: str) -> str:
    try:
        deep_list = load_deep_list()
        for item in deep_list:
            if item.get('ticker') == ticker: return item.get('deep_grade', '-')
    except Exception: pass
    return '-'

_event_lock = threading.Lock()
def add_event(emoji: str, msg: str):
    event = {'time': _now_hm(), 'emoji': emoji, 'msg': msg}
    with _event_lock:
        events = load_events()
        events.append(event)
        if len(events) > 50: events = events[-50:]
        _save_json(EVENT_FILE, events)
    log.info(f'{emoji} {msg}')

def _upbit_get(url: str, params: dict = None, retries: int = 3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1: time.sleep(REQUEST_DELAY * (attempt + 2))
            else: log.warning(f'Upbit API 실패 {url}: {e}')
    return None

def get_krw_tickers() -> list:
    data = _upbit_get('https://api.upbit.com/v1/market/all', {'isDetails': 'false'})
    if not data: return []
    return [
        item['market'] for item in data
        if item.get('market','').startswith('KRW-')
        and item['market'].replace('KRW-','') not in STABLE_COINS
    ]

def get_candles(market: str, unit: str, count: int = CANDLE_COUNT) -> list:
    if unit == 'days': url = 'https://api.upbit.com/v1/candles/days'
    elif unit == 'weeks': url = 'https://api.upbit.com/v1/candles/weeks'
    else: url = f'https://api.upbit.com/v1/candles/{unit}'
    data = _upbit_get(url, {'market': market, 'count': count})
    if not data: return []
    return [float(c['trade_price']) for c in reversed(data)]

def get_current_price(market: str):
    data = _upbit_get('https://api.upbit.com/v1/ticker', {'markets': market})
    if data and len(data) > 0: return float(data[0]['trade_price'])
    return None

def get_usdt_rate() -> float:
    data = _upbit_get('https://api.upbit.com/v1/ticker', {'markets': 'KRW-USDT'})
    if data and len(data) > 0: return float(data[0]['trade_price'])
    return 1300.0

def get_volume_ratio(market: str) -> float:
    data = _upbit_get('https://api.upbit.com/v1/candles/days', {'market': market, 'count': 21})
    if not data or len(data) < 2: return 1.0
    vols    = [float(c['candle_acc_trade_volume']) for c in reversed(data)]
    avg_vol = sum(vols[:-1]) / len(vols[:-1])
    return round(vols[-1] / avg_vol, 2) if avg_vol > 0 else 1.0
