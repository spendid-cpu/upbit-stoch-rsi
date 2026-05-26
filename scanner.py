"""
scanner.py v3.0.2
변경사항:
- v3.0.1: USDT 환율, 주봉 MA20, 이벤트 로그, 스캔 상태 플래그 세분화, 중복 스캔 방지
- v3.0.2: analyze_ticker() 반환값에 사이클 정보 추가
           (d_short_cycle, d_mid_cycle, h4_cycle, h1_cycle)
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

import mtf_setup as _mtf                  # ← 하나로 통합
from mtf_setup import VERSION as MTF_VERSION

VERSION = 'v3.0.2'

# ── 로깅 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

# ── 환경변수 ──────────────────────────────────────────────────────
TELEGRAM_TOKEN       = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID     = os.environ.get('TELEGRAM_CHAT_ID', '')

SCAN_INTERVAL_MIN         = int(os.environ.get('SCAN_INTERVAL_MIN',         '60'))
WATCH_RESCAN_INTERVAL_MIN = int(os.environ.get('WATCH_RESCAN_INTERVAL_MIN', '15'))
PRICE_CHECK_INTERVAL_MIN  = int(os.environ.get('PRICE_CHECK_INTERVAL_MIN',  '1'))
ACTIVE_CHECK_INTERVAL_MIN = int(os.environ.get('ACTIVE_CHECK_INTERVAL_MIN', '1'))
DEEP_SCAN_INTERVAL_MIN    = int(os.environ.get('DEEP_SCAN_INTERVAL_MIN',    '5'))
DAILY_SUMMARY_HOUR_KST    = int(os.environ.get('DAILY_SUMMARY_HOUR_KST',   '9'))

REQUEST_DELAY  = float(os.environ.get('REQUEST_DELAY', '0.35'))
MAX_WORKERS    = int(os.environ.get('MAX_WORKERS',    '3'))
CANDLE_COUNT   = int(os.environ.get('CANDLE_COUNT',   '100'))

TRADE_TP_PCT    = float(os.environ.get('TRADE_TP_PCT',    '5.0'))
TRADE_SL_PCT    = float(os.environ.get('TRADE_SL_PCT',    '3.0'))
TRADE_TIMEOUT_H = int(os.environ.get('TRADE_TIMEOUT_H',  '48'))

BTC_DROP_1H_PCT = float(os.environ.get('BTC_DROP_1H_PCT', '-1.0'))
BTC_DROP_4H_PCT = float(os.environ.get('BTC_DROP_4H_PCT', '-2.0'))

WATCH_EXPIRE_DAYS = {'S': 7, 'A': 7, 'B': 5, 'C': 3, 'X': 1, '-': 1}

# 스테이블 코인 제외
STABLE_COINS = {
    'USDT','USDC','DAI','BUSD','TUSD','USDP','USDD',
    'USD1','FDUSD','PYUSD','SUSD','GUSD',
    'STETH','WBTC','CBBTC',
}

# ── 파일 경로 ─────────────────────────────────────────────────────
BASE_DIR     = os.environ.get('DATA_DIR', '/app/data')
WATCH_FILE   = os.path.join(BASE_DIR, 'watch_list.json')
ACTIVE_FILE  = os.path.join(BASE_DIR, 'active_list.json')
HISTORY_FILE = os.path.join(BASE_DIR, 'trade_history.json')
DEEP_FILE    = os.path.join(BASE_DIR, 'deep_list.json')
STATE_FILE   = os.path.join(BASE_DIR, 'scanner_state.json')
EVENT_FILE   = os.path.join(BASE_DIR, 'events.json')

os.makedirs(BASE_DIR, exist_ok=True)

# ── 전역 상태 ─────────────────────────────────────────────────────
_state_lock    = threading.Lock()
_scanner_state = {
    'version':          VERSION,
    'mtf_version':      MTF_VERSION,

    'running':            False,
    'watch_rescanning':   False,
    'price_checking':     False,
    'deep_scanning':      False,

    'last_scan':          None,
    'last_watch_rescan':  None,
    'last_price_check':   None,
    'last_deep_scan':     None,
    'next_scan':          None,
    'next_deep_scan':     None,
    'scan_count':         0,

    'watch_count':        0,
    'active_count':       0,
    'deep_count':         0,

    'btc_price':          None,
    'btc_price_usd':      None,
    'usdt_rate':          None,
    'btc_daily_ma20':     None,
    'btc_daily_ma20_usd': None,
    'btc_daily_above':    None,
    'btc_daily_pct':      None,
    'btc_weekly_ma20':    None,
    'btc_weekly_ma20_usd':None,
    'btc_weekly_above':   None,
    'btc_weekly_pct':     None,
    'btc_1h_pct':         None,
    'btc_4h_pct':         None,

    'total_trades':  0,
    'win_trades':    0,
    'total_pnl':     0.0,

    'error': None,
}

# ── JSON 유틸 ─────────────────────────────────────────────────────
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

def save_state():
    _save_json(STATE_FILE, _scanner_state)

# ══════════════════════════════════════════════════════════════════
# 이벤트 로그
# ══════════════════════════════════════════════════════════════════

_event_lock = threading.Lock()

def add_event(emoji: str, msg: str):
    event = {
        'time':  datetime.now().strftime('%H:%M'),
        'emoji': emoji,
        'msg':   msg,
    }
    with _event_lock:
        events = load_events()
        events.append(event)
        if len(events) > 50:
            events = events[-50:]
        _save_json(EVENT_FILE, events)
    log.info(f'{emoji} {msg}')

# ══════════════════════════════════════════════════════════════════
# Upbit API
# ══════════════════════════════════════════════════════════════════

def _upbit_get(url: str, params: dict = None, retries: int = 3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(REQUEST_DELAY * (attempt + 2))
            else:
                log.warning(f'Upbit API 실패 {url}: {e}')
    return None

def get_krw_tickers() -> list:
    data = _upbit_get('https://api.upbit.com/v1/market/all', {'isDetails': 'false'})
    if not data:
        return []
    tickers = []
    for item in data:
        mkt  = item.get('market', '')
        if not mkt.startswith('KRW-'):
            continue
        coin = mkt.replace('KRW-', '')
        if coin not in STABLE_COINS:
            tickers.append(mkt)
    return tickers

def get_candles(market: str, unit: str, count: int = CANDLE_COUNT) -> list:
    if unit == 'days':
        url = 'https://api.upbit.com/v1/candles/days'
    elif unit == 'weeks':
        url = 'https://api.upbit.com/v1/candles/weeks'
    else:
        url = f'https://api.upbit.com/v1/candles/{unit}'

    data = _upbit_get(url, {'market': market, 'count': count})
    if not data:
        return []
    return [float(c['trade_price']) for c in reversed(data)]

def get_current_price(market: str):
    data = _upbit_get('https://api.upbit.com/v1/ticker', {'markets': market})
    if data and len(data) > 0:
        return float(data[0]['trade_price'])
    return None

def get_usdt_rate() -> float:
    data = _upbit_get('https://api.upbit.com/v1/ticker', {'markets': 'KRW-USDT'})
    if data and len(data) > 0:
        return float(data[0]['trade_price'])
    return 1300.0

def get_volume_ratio(market: str) -> float:
    data = _upbit_get(
        'https://api.upbit.com/v1/candles/days',
        {'market': market, 'count': 21}
    )
    if not data or len(data) < 2:
        return 1.0
    vols    = [float(c['candle_acc_trade_volume']) for c in reversed(data)]
    avg_vol = sum(vols[:-1]) / len(vols[:-1])
    return round(vols[-1] / avg_vol, 2) if avg_vol > 0 else 1.0

def get_btc_info() -> dict:
    usdt_rate      = get_usdt_rate()
    time.sleep(REQUEST_DELAY)

    closes_daily   = get_candles('KRW-BTC', 'days',        count=30)
    time.sleep(REQUEST_DELAY)
    closes_weekly  = get_candles('KRW-BTC', 'weeks',       count=25)
    time.sleep(REQUEST_DELAY)
    closes_h4      = get_candles('KRW-BTC', 'minutes/240', count=10)
    time.sleep(REQUEST_DELAY)
    closes_h1      = get_candles('KRW-BTC', 'minutes/60',  count=5)

    ma20_info = _mtf.btc_ma20_signal(closes_daily, closes_weekly)

    price  = ma20_info.get('price')
    pct_4h = None
    pct_1h = None

    if len(closes_h4) >= 2:
        pct_4h = round((closes_h4[-1] - closes_h4[-2]) / closes_h4[-2] * 100, 2)
    if len(closes_h1) >= 2:
        pct_1h = round((closes_h1[-1] - closes_h1[-2]) / closes_h1[-2] * 100, 2)

    def to_usd(krw):
        if krw and usdt_rate and usdt_rate > 0:
            return round(krw / usdt_rate, 1)
        return None

    return {
        'price':              price,
        'price_usd':          to_usd(price),
        'usdt_rate':          usdt_rate,
        'daily_ma20':         ma20_info.get('daily_ma20'),
        'daily_ma20_usd':     to_usd(ma20_info.get('daily_ma20')),
        'daily_above':        ma20_info.get('daily_above'),
        'daily_pct':          ma20_info.get('daily_pct'),
        'weekly_ma20':        ma20_info.get('weekly_ma20'),
        'weekly_ma20_usd':    to_usd(ma20_info.get('weekly_ma20')),
        'weekly_above':       ma20_info.get('weekly_above'),
        'weekly_pct':         ma20_info.get('weekly_pct'),
        'pct_1h':             pct_1h,
        'pct_4h':             pct_4h,
    }

def get_price_change_pct(market: str, unit: str, periods: int = 1):
    candles = get_candles(market, unit, count=periods + 1)
    if len(candles) < 2:
        return None
    old = candles[-(periods + 1)]
    new = candles[-1]
    return round((new - old) / old * 100, 2) if old != 0 else None

# ══════════════════════════════════════════════════════════════════
# Telegram
# ══════════════════════════════════════════════════════════════════

def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
            timeout=10
        )
    except Exception as e:
        log.warning(f'텔레그램 전송 실패: {e}')

def _fmt_watch_msg(item: dict) -> str:
    return (
        f'📋 <b>Watch 등록</b>\n'
        f'종목: <b>{item["ticker"]}</b> | 등급: <b>{item.get("grade","-")}</b>\n'
        f'등록가: {item.get("reg_price",0):,.0f} KRW\n'
        f'점수: {item.get("score",0)}점 | 일봉장기K: {item.get("daily_long_k","-")}\n'
        f'⏰ {datetime.now().strftime("%m/%d %H:%M")}'
    )

def _fmt_active_msg(item: dict, trade_type: str = 'auto') -> str:
    label = '🤖 자동' if trade_type == 'auto' else '👤 수동'
    return (
        f'✅ <b>Active 진입</b> ({label})\n'
        f'종목: <b>{item["ticker"]}</b> | 등급: <b>{item.get("grade","-")}</b>\n'
        f'진입가: {item.get("entry_price",0):,.0f} KRW\n'
        f'TP: {item.get("tp_price",0):,.0f} (+{TRADE_TP_PCT}%) | '
        f'SL: {item.get("sl_price",0):,.0f} (-{TRADE_SL_PCT}%)\n'
        f'⏰ {datetime.now().strftime("%m/%d %H:%M")}'
    )

def _fmt_close_msg(item: dict, reason: str, pnl_pct: float) -> str:
    emoji = '🟢' if pnl_pct >= 0 else '🔴'
    return (
        f'{emoji} <b>포지션 종료</b>\n'
        f'종목: <b>{item.get("ticker","")}</b> | 사유: {reason}\n'
        f'수익률: {pnl_pct:+.2f}%\n'
        f'⏰ {datetime.now().strftime("%m/%d %H:%M")}'
    )

def _fmt_deep_msg(items: list, btc_pct: str) -> str:
    lines = [f'🔥 <b>DEEP 상대강도 감지</b> (BTC {btc_pct}%)\n']
    for it in items[:5]:
        lines.append(
            f'  <b>{it["ticker"]}</b> [{it.get("deep_grade","-")}] '
            f'RS: +{it.get("rs",0)}% | 변화: {it.get("coin_pct","?")}%'
        )
    return '\n'.join(lines) + f'\n⏰ {datetime.now().strftime("%m/%d %H:%M")}'

# ══════════════════════════════════════════════════════════════════
# 코인 분석
# ══════════════════════════════════════════════════════════════════

def analyze_ticker(market: str):
    try:
        time.sleep(REQUEST_DELAY)
        daily = get_candles(market, 'days',        count=CANDLE_COUNT)
        time.sleep(REQUEST_DELAY)
        h4    = get_candles(market, 'minutes/240', count=CANDLE_COUNT)
        time.sleep(REQUEST_DELAY)
        h1    = get_candles(market, 'minutes/60',  count=CANDLE_COUNT)

        if len(daily) < 60 or len(h4) < 60 or len(h1) < 60:
            return None

        mtf     = _mtf.analyze_mtf({'daily': daily, 'h4': h4, 'h1': h1})
        summary = mtf['summary']

        vol_ratio   = get_volume_ratio(market)
        bottom_days = _count_bottom_days(
            get_candles(market, 'days', count=30), 'long'
        )

        return {
            'market':   market,
            'ticker':   market.replace('KRW-', ''),
            'price':    daily[-1] if daily else None,

            'daily_long_k':       mtf['daily']['long'].get('k'),
            'daily_long_d':       mtf['daily']['long'].get('d'),
            'daily_mid_k':        mtf['daily']['mid'].get('k'),
            'daily_mid_d':        mtf['daily']['mid'].get('d'),
            'daily_short_k':      mtf['daily']['short'].get('k'),
            'daily_short_d':      mtf['daily']['short'].get('d'),
            'daily_long_signal':  mtf['daily']['long'].get('signal'),
            'daily_short_signal': mtf['daily']['short'].get('signal'),

            'h4_short_k':      mtf['h4']['short'].get('k'),
            'h4_short_d':      mtf['h4']['short'].get('d'),
            'h4_short_signal': mtf['h4']['short'].get('signal'),
            'h4_gc':           summary.get('h4_gc', False),

            'h1_short_k':      mtf['h1']['short'].get('k'),
            'h1_short_d':      mtf['h1']['short'].get('d'),
            'h1_short_signal': mtf['h1']['short'].get('signal'),
            'h1_gc':           summary.get('h1_gc', False),

            'grade':          summary.get('grade', '-'),
            'score':          summary.get('score', 0),
            'watch_eligible': summary.get('watch_eligible', False),
            'auto_entry':     summary.get('auto_entry', False),
            'any_buy_no':     summary.get('any_buy_no', False),

            'vol_ratio':   vol_ratio,
            'bottom_days': bottom_days,

            # v3.0.2: 사이클 정보
            'd_short_cycle': mtf['daily']['short'].get('cycle', 'RISING'),
            'd_mid_cycle':   mtf['daily']['mid'].get('cycle',   'RISING'),
            'h4_cycle':      mtf['h4']['short'].get('cycle',    'RISING'),
            'h1_cycle':      mtf['h1']['short'].get('cycle',    'RISING'),

            'analyzed_at': datetime.now().isoformat(),
        }

    except Exception as e:
        log.warning(f'분석 실패 {market}: {e}')
        return None


def _count_bottom_days(closes: list, term: str) -> int:
    if len(closes) < 30:
        return 0
    count = 0
    for i in range(len(closes) - 1, max(len(closes) - 15, -1), -1):
        r = _mtf.calc_stoch_rsi(closes[:i+1], term)
        if r.get('k') is not None and r['k'] <= 20:
            count += 1
        else:
            break
    return count

# ══════════════════════════════════════════════════════════════════
# Watch / Active 빌더
# ══════════════════════════════════════════════════════════════════

def _make_watch_item(res: dict) -> dict:
    now       = datetime.now().isoformat()
    expire_at = (
        datetime.now() + timedelta(days=WATCH_EXPIRE_DAYS.get(res.get('grade','-'), 3))
    ).isoformat()
    return {
        'ticker':        res['ticker'],
        'market':        res['market'],
        'grade':         res.get('grade', '-'),
        'score':         res.get('score', 0),
        'reg_price':     res.get('price'),

        'daily_long_k':  res.get('daily_long_k'),
        'daily_long_d':  res.get('daily_long_d'),
        'daily_mid_k':   res.get('daily_mid_k'),
        'daily_mid_d':   res.get('daily_mid_d'),
        'daily_short_k': res.get('daily_short_k'),
        'daily_short_d': res.get('daily_short_d'),

        'h4_short_k':    res.get('h4_short_k'),
        'h4_short_d':    res.get('h4_short_d'),
        'h4_gc':         res.get('h4_gc', False),

        'h1_short_k':    res.get('h1_short_k'),
        'h1_short_d':    res.get('h1_short_d'),
        'h1_gc':         res.get('h1_gc', False),

        'vol_ratio':     res.get('vol_ratio', 1.0),
        'bottom_days':   res.get('bottom_days', 0),

        # v3.0.2: 사이클 정보
        'd_short_cycle': res.get('d_short_cycle', 'RISING'),
        'd_mid_cycle':   res.get('d_mid_cycle',   'RISING'),
        'h4_cycle':      res.get('h4_cycle',       'RISING'),
        'h1_cycle':      res.get('h1_cycle',       'RISING'),

        'added_at':      now,
        'expire_at':     expire_at,
        'score_history': [res.get('score', 0)],
        'rescan_count':  0,
    }


def _make_active_item(watch_item: dict, price: float, trade_type: str = 'auto') -> dict:
    now    = datetime.now().isoformat()
    tp     = round(price * (1 + TRADE_TP_PCT / 100), 2)
    sl     = round(price * (1 - TRADE_SL_PCT / 100), 2)
    expire = (datetime.now() + timedelta(hours=TRADE_TIMEOUT_H)).isoformat()

    return {
        'ticker':        watch_item['ticker'],
        'market':        watch_item['market'],
        'grade':         watch_item.get('grade', '-'),
        'score':         watch_item.get('score', 0),
        'entry_price':   price,
        'tp_price':      tp,
        'sl_price':      sl,
        'trade_type':    trade_type,
        'entry_at':      now,
        'expire_at':     expire,
        'current_price': price,
        'pnl_pct':       0.0,
        'max_price':     price,
        'min_price':     price,

        'daily_long_k':  watch_item.get('daily_long_k'),
        'daily_short_k': watch_item.get('daily_short_k'),
        'h4_short_k':    watch_item.get('h4_short_k'),
        'h4_short_d':    watch_item.get('h4_short_d'),
        'h4_gc':         watch_item.get('h4_gc', False),
        'h1_short_k':    watch_item.get('h1_short_k'),
        'h1_short_d':    watch_item.get('h1_short_d'),
        'h1_gc':         watch_item.get('h1_gc', False),
        'vol_ratio':     watch_item.get('vol_ratio', 1.0),
        'bottom_days':   watch_item.get('bottom_days', 0),

        # v3.0.2: 사이클 정보
        'd_short_cycle': watch_item.get('d_short_cycle', 'RISING'),
        'd_mid_cycle':   watch_item.get('d_mid_cycle',   'RISING'),
        'h4_cycle':      watch_item.get('h4_cycle',       'RISING'),
        'h1_cycle':      watch_item.get('h1_cycle',       'RISING'),
    }


def _is_watch_expired(item: dict) -> bool:
    try:
        return datetime.now() > datetime.fromisoformat(item.get('expire_at', ''))
    except Exception:
        return False


def close_active_item(item: dict, reason: str, close_price: float) -> dict:
    entry   = item.get('entry_price', close_price)
    pnl_pct = round((close_price - entry) / entry * 100, 2) if entry > 0 else 0.0

    closed = {**item}
    closed.update({
        'close_price': close_price,
        'close_at':    datetime.now().isoformat(),
        'close_reason':reason,
        'pnl_pct':     pnl_pct,
    })

    history = load_history()
    history.append(closed)
    save_history(history)

    emoji = '🟢' if pnl_pct >= 0 else '🔴'
    add_event(emoji, f'{item.get("ticker","")} [{reason}] {pnl_pct:+.2f}%')
    send_telegram(_fmt_close_msg(item, reason, pnl_pct))

    with _state_lock:
        _scanner_state['total_trades'] += 1
        if pnl_pct > 0:
            _scanner_state['win_trades'] += 1
        _scanner_state['total_pnl'] = round(
            _scanner_state['total_pnl'] + pnl_pct, 2
        )
    return closed

# ══════════════════════════════════════════════════════════════════
# DEEP 스캔
# ══════════════════════════════════════════════════════════════════

def run_deep_scan(btc_info: dict):
    with _state_lock:
        _scanner_state['deep_scanning'] = True
    add_event('🔥', 'DEEP 스캔 시작')

    try:
        tickers  = get_krw_tickers()
        btc_pct  = min(btc_info.get('pct_1h') or 0, btc_info.get('pct_4h') or 0)
        results  = []

        def _check(market):
            time.sleep(REQUEST_DELAY)
            coin_pct = get_price_change_pct(market, 'minutes/60', 1)
            if coin_pct is None:
                return None
            rs_info = _mtf.calc_relative_strength(coin_pct, btc_pct)
            if rs_info['grade'] == '-':
                return None
            daily = get_candles(market, 'days', count=30)
            if daily:
                r = _mtf.calc_stoch_rsi(daily, 'long')
                if r.get('k') is not None and r['k'] >= 70:
                    return None
            time.sleep(REQUEST_DELAY)
            vol_ratio = get_volume_ratio(market)
            return {
                'ticker':     market.replace('KRW-', ''),
                'market':     market,
                'coin_pct':   coin_pct,
                'btc_pct':    btc_pct,
                'rs':         rs_info['rs'],
                'deep_grade': rs_info['grade'],
                'signal':     rs_info['signal'],
                'vol_ratio':  vol_ratio,
                'scanned_at': datetime.now().isoformat(),
            }

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for res in as_completed({ex.submit(_check, m): m for m in tickers}):
                r = res.result()
                if r:
                    results.append(r)

        results.sort(key=lambda x: x['rs'], reverse=True)
        top = [r for r in results if r['deep_grade'] in ('S', 'A', 'B')]
        save_deep_list(top)

        with _state_lock:
            _scanner_state['last_deep_scan'] = datetime.now().isoformat()
            _scanner_state['deep_count']     = len(top)
            _scanner_state['deep_scanning']  = False

        add_event('🔥', f'DEEP 스캔 완료 {len(top)}개 감지')

        alert = [r for r in top if r['deep_grade'] in ('S', 'A')]
        if alert:
            send_telegram(_fmt_deep_msg(alert, f'{btc_pct:+.1f}'))

    except Exception as e:
        log.error(f'DEEP 스캔 오류: {e}')
        with _state_lock:
            _scanner_state['deep_scanning'] = False

# ══════════════════════════════════════════════════════════════════
# 수동 관리 API
# ══════════════════════════════════════════════════════════════════

def manual_add_watch(ticker: str) -> dict:
    ticker  = ticker.upper().replace('KRW-', '')
    market  = f'KRW-{ticker}'
    watches = load_watch_list()
    if any(w['ticker'] == ticker for w in watches):
        return {'success': False, 'message': f'{ticker} 이미 Watch에 있습니다.'}
    res = analyze_ticker(market)
    if not res:
        return {'success': False, 'message': f'{ticker} 분석 실패'}
    item = _make_watch_item(res)
    watches.append(item)
    save_watch_list(watches)
    add_event('📋', f'{ticker} 수동 Watch 등록 [{item["grade"]}]')
    send_telegram(_fmt_watch_msg(item))
    return {'success': True, 'message': f'{ticker} Watch 등록 완료', 'item': item}


def manual_remove_watch(ticker: str) -> dict:
    ticker  = ticker.upper().replace('KRW-', '')
    watches = load_watch_list()
    new     = [w for w in watches if w['ticker'] != ticker]
    if len(new) == len(watches):
        return {'success': False, 'message': f'{ticker} Watch에 없습니다.'}
    save_watch_list(new)
    add_event('🗑️', f'{ticker} Watch 제거')
    return {'success': True, 'message': f'{ticker} Watch 제거 완료'}


def manual_activate_watch(ticker: str) -> dict:
    ticker  = ticker.upper().replace('KRW-', '')
    watches = load_watch_list()
    actives = load_active_list()
    if any(a['ticker'] == ticker for a in actives):
        return {'success': False, 'message': f'{ticker} 이미 Active에 있습니다.'}
    watch_item = next((w for w in watches if w['ticker'] == ticker), None)
    if not watch_item:
        return {'success': False, 'message': f'{ticker} Watch에 없습니다.'}
    price = get_current_price(f'KRW-{ticker}')
    if not price:
        return {'success': False, 'message': f'{ticker} 현재가 조회 실패'}
    active = _make_active_item(watch_item, price, 'manual')
    actives.append(active)
    save_active_list(actives)
    watches = [w for w in watches if w['ticker'] != ticker]
    save_watch_list(watches)
    add_event('✅', f'{ticker} 수동 Active 전환 @ {price:,.0f}')
    send_telegram(_fmt_active_msg(active, 'manual'))
    with _state_lock:
        _scanner_state['active_count'] = len(actives)
        _scanner_state['watch_count']  = len(watches)
    return {'success': True, 'message': f'{ticker} Active 전환 완료', 'item': active}


def manual_close_active(ticker: str, reason: str = '수동종료') -> dict:
    ticker  = ticker.upper().replace('KRW-', '')
    actives = load_active_list()
    item    = next((a for a in actives if a['ticker'] == ticker), None)
    if not item:
        return {'success': False, 'message': f'{ticker} Active에 없습니다.'}
    price = get_current_price(f'KRW-{ticker}') or item.get('current_price', item.get('entry_price', 0))
    close_active_item(item, reason, price)
    actives = [a for a in actives if a['ticker'] != ticker]
    save_active_list(actives)
    with _state_lock:
        _scanner_state['active_count'] = len(actives)
    return {'success': True, 'message': f'{ticker} 수동 종료 완료'}


def run_single_scan() -> dict:
    with _state_lock:
        if _scanner_state.get('running'):
            return {'success': False, 'message': '스캔이 이미 실행 중입니다.'}
    try:
        _run_full_scan()
        return {'success': True, 'message': '스캔 완료'}
    except Exception as e:
        return {'success': False, 'message': str(e)}


def reset_watch_list() -> dict:
    save_watch_list([])
    with _state_lock:
        _scanner_state['watch_count'] = 0
    add_event('🔄', 'Watch 목록 초기화')
    return {'success': True, 'message': 'Watch 목록 초기화 완료'}

# ══════════════════════════════════════════════════════════════════
# 스캔 루틴
# ══════════════════════════════════════════════════════════════════

def _run_full_scan():
    with _state_lock:
        if _scanner_state.get('running'):
            log.info('⏭️ 스캔 이미 실행 중 - 스킵')
            return
        _scanner_state['running'] = True

    add_event('📡', '전체 스캔 시작')
    log.info('🚀 전체 스캔 시작')

    try:
        tickers = get_krw_tickers()
        log.info(f'  대상: {len(tickers)}개 종목')

        btc = get_btc_info()
        with _state_lock:
            _scanner_state['btc_price']            = btc.get('price')
            _scanner_state['btc_price_usd']        = btc.get('price_usd')
            _scanner_state['usdt_rate']             = btc.get('usdt_rate')
            _scanner_state['btc_daily_ma20']        = btc.get('daily_ma20')
            _scanner_state['btc_daily_ma20_usd']    = btc.get('daily_ma20_usd')
            _scanner_state['btc_daily_above']       = btc.get('daily_above')
            _scanner_state['btc_daily_pct']         = btc.get('daily_pct')
            _scanner_state['btc_weekly_ma20']       = btc.get('weekly_ma20')
            _scanner_state['btc_weekly_ma20_usd']   = btc.get('weekly_ma20_usd')
            _scanner_state['btc_weekly_above']      = btc.get('weekly_above')
            _scanner_state['btc_weekly_pct']        = btc.get('weekly_pct')
            _scanner_state['btc_1h_pct']            = btc.get('pct_1h')
            _scanner_state['btc_4h_pct']            = btc.get('pct_4h')

        watches        = load_watch_list()
        actives        = load_active_list()
        watch_tickers  = {w['ticker'] for w in watches}
        active_tickers = {a['ticker'] for a in actives}

        new_watches = []

        def _process(market):
            res = analyze_ticker(market)
            if not res:
                return
            ticker = res['ticker']
            if (
                res.get('watch_eligible') and
                not res.get('any_buy_no') and
                ticker not in watch_tickers and
                ticker not in active_tickers
            ):
                item = _make_watch_item(res)
                new_watches.append(item)
                log.info(
                    f'  📋 Watch 등록: {ticker} [{res["grade"]}] {res["score"]}점 '
                    f'd_short:{res["d_short_cycle"]} d_mid:{res["d_mid_cycle"]}'
                )
                add_event('📋', f'{ticker} Watch 등록 [{res["grade"]}] {res["score"]}점')
                send_telegram(_fmt_watch_msg(item))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            list(ex.map(_process, tickers))

        if new_watches:
            watches.extend(new_watches)
            watches = [w for w in watches if not _is_watch_expired(w)]
            save_watch_list(watches)

        now       = datetime.now().isoformat()
        next_scan = (datetime.now() + timedelta(minutes=SCAN_INTERVAL_MIN)).isoformat()

        with _state_lock:
            _scanner_state['running']     = False
            _scanner_state['last_scan']   = now
            _scanner_state['next_scan']   = next_scan
            _scanner_state['scan_count'] += 1
            _scanner_state['watch_count'] = len(watches)
            _scanner_state['error']       = None

        save_state()
        add_event('✅', f'스캔 완료 | Watch {len(watches)}개')
        log.info(f'✅ 전체 스캔 완료 | Watch: {len(watches)}개')

    except Exception as e:
        log.error(f'스캔 오류: {e}')
        with _state_lock:
            _scanner_state['running'] = False
            _scanner_state['error']   = str(e)
        add_event('❌', f'스캔 오류: {e}')


def _run_watch_rescan():
    with _state_lock:
        if _scanner_state.get('watch_rescanning'):
            return
        _scanner_state['watch_rescanning'] = True

    watches = load_watch_list()
    actives = load_active_list()

    if not watches:
        with _state_lock:
            _scanner_state['watch_rescanning'] = False
        return

    add_event('🔍', f'Watch 재스캔 시작 ({len(watches)}개)')
    log.info(f'🔄 Watch 재스캔: {len(watches)}개')
    active_tickers  = {a['ticker'] for a in actives}
    updated_watches = []
    new_actives     = []

    try:
        for item in watches:
            if _is_watch_expired(item):
                add_event('⏰', f'{item["ticker"]} Watch 만료')
                continue

            ticker = item['ticker']
            market = item['market']

            if ticker in active_tickers:
                updated_watches.append(item)
                continue

            res = analyze_ticker(market)
            if not res:
                updated_watches.append(item)
                continue

            if res.get('any_buy_no'):
                add_event('❌', f'{ticker} BUY_NO 감지 → Watch 제거')
                continue

            if res.get('auto_entry'):
                price = get_current_price(market)
                if price:
                    updated_item = {**item, **{
                        'grade':         res['grade'],
                        'score':         res['score'],
                        'h4_short_k':    res.get('h4_short_k'),
                        'h4_short_d':    res.get('h4_short_d'),
                        'h4_gc':         res.get('h4_gc', False),
                        'h1_short_k':    res.get('h1_short_k'),
                        'h1_short_d':    res.get('h1_short_d'),
                        'h1_gc':         res.get('h1_gc', False),
                        'daily_short_k': res.get('daily_short_k'),
                        'daily_long_k':  res.get('daily_long_k'),
                        'd_short_cycle': res.get('d_short_cycle', 'RISING'),
                        'd_mid_cycle':   res.get('d_mid_cycle',   'RISING'),
                        'h4_cycle':      res.get('h4_cycle',       'RISING'),
                        'h1_cycle':      res.get('h1_cycle',       'RISING'),
                    }}
                    active = _make_active_item(updated_item, price, 'auto')
                    new_actives.append(active)
                    active_tickers.add(ticker)
                    add_event('✅', f'{ticker} 자동 Active 전환 [{res["grade"]}] @ {price:,.0f}')
                    send_telegram(_fmt_active_msg(active, 'auto'))
                    continue

            item_u = {**item}
            item_u.update({
                'grade':         res['grade'],
                'score':         res['score'],
                'h4_short_k':    res.get('h4_short_k'),
                'h4_short_d':    res.get('h4_short_d'),
                'h4_gc':         res.get('h4_gc', False),
                'h1_short_k':    res.get('h1_short_k'),
                'h1_short_d':    res.get('h1_short_d'),
                'h1_gc':         res.get('h1_gc', False),
                'daily_long_k':  res.get('daily_long_k'),
                'daily_short_k': res.get('daily_short_k'),
                'd_short_cycle': res.get('d_short_cycle', 'RISING'),
                'd_mid_cycle':   res.get('d_mid_cycle',   'RISING'),
                'h4_cycle':      res.get('h4_cycle',       'RISING'),
                'h1_cycle':      res.get('h1_cycle',       'RISING'),
            })
            sh = item_u.get('score_history', [])
            sh.append(res['score'])
            item_u['score_history'] = sh[-10:]
            item_u['rescan_count']  = item_u.get('rescan_count', 0) + 1
            updated_watches.append(item_u)

        if new_actives:
            actives.extend(new_actives)
            save_active_list(actives)

        save_watch_list(updated_watches)

        now = datetime.now().isoformat()
        with _state_lock:
            _scanner_state['watch_count']       = len(updated_watches)
            _scanner_state['active_count']      = len(actives)
            _scanner_state['last_watch_rescan'] = now
            _scanner_state['watch_rescanning']  = False

        add_event('🔍', f'Watch 재스캔 완료 | Watch {len(updated_watches)} / Active {len(actives)}')

    except Exception as e:
        log.error(f'Watch 재스캔 오류: {e}')
        with _state_lock:
            _scanner_state['watch_rescanning'] = False


def _run_price_check():
    with _state_lock:
        _scanner_state['price_checking'] = True

    actives = load_active_list()
    if not actives:
        with _state_lock:
            _scanner_state['price_checking'] = False
        return

    remaining = []
    for item in actives:
        price = get_current_price(item['market'])
        if not price:
            remaining.append(item)
            continue

        item['current_price'] = price
        entry = item.get('entry_price', price)
        if entry > 0:
            item['pnl_pct'] = round((price - entry) / entry * 100, 2)

        item['max_price'] = max(item.get('max_price', price), price)
        item['min_price'] = min(item.get('min_price', price), price)

        if price >= item.get('tp_price', float('inf')):
            close_active_item(item, 'TP', price)
            continue

        if price <= item.get('sl_price', 0):
            close_active_item(item, 'SL', price)
            continue

        try:
            if datetime.now() > datetime.fromisoformat(item.get('expire_at', '')):
                close_active_item(item, '시간만료', price)
                continue
        except Exception:
            pass

        remaining.append(item)

    save_active_list(remaining)

    with _state_lock:
        _scanner_state['active_count']     = len(remaining)
        _scanner_state['last_price_check'] = datetime.now().isoformat()
        _scanner_state['price_checking']   = False

# ══════════════════════════════════════════════════════════════════
# 루프 함수
# ══════════════════════════════════════════════════════════════════

def scanner_loop():
    log.info(f'🚀 scanner_loop 시작 (주기: {SCAN_INTERVAL_MIN}분)')
    while True:
        try:
            _run_full_scan()
        except Exception as e:
            log.error(f'scanner_loop 오류: {e}')
        time.sleep(SCAN_INTERVAL_MIN * 60)


def watch_rescan_loop():
    log.info(f'🔄 watch_rescan_loop 시작 (주기: {WATCH_RESCAN_INTERVAL_MIN}분)')
    time.sleep(90)
    while True:
        try:
            _run_watch_rescan()
        except Exception as e:
            log.error(f'watch_rescan_loop 오류: {e}')
        time.sleep(WATCH_RESCAN_INTERVAL_MIN * 60)


def price_check_loop():
    log.info('💰 price_check_loop 시작 (주기: 1분)')
    while True:
        try:
            _run_price_check()
        except Exception as e:
            log.error(f'price_check_loop 오류: {e}')
        time.sleep(PRICE_CHECK_INTERVAL_MIN * 60)


def active_monitor_loop():
    log.info('📊 active_monitor_loop 시작')
    while True:
        try:
            actives = load_active_list()
            if actives:
                _run_price_check()
        except Exception as e:
            log.error(f'active_monitor_loop 오류: {e}')
        time.sleep(ACTIVE_CHECK_INTERVAL_MIN * 60)


def deep_scan_loop():
    log.info(f'🔥 deep_scan_loop 시작 (주기: {DEEP_SCAN_INTERVAL_MIN}분)')
    while True:
        try:
            btc_info = get_btc_info()
            p1h = btc_info.get('pct_1h') or 0
            p4h = btc_info.get('pct_4h') or 0
            if p1h <= BTC_DROP_1H_PCT or p4h <= BTC_DROP_4H_PCT:
                add_event('🔥', f'BTC 급락 감지 (1h:{p1h}% 4h:{p4h}%) → DEEP 스캔')
                run_deep_scan(btc_info)

            with _state_lock:
                _scanner_state['next_deep_scan'] = (
                    datetime.now() + timedelta(minutes=DEEP_SCAN_INTERVAL_MIN)
                ).isoformat()

        except Exception as e:
            log.error(f'deep_scan_loop 오류: {e}')
        time.sleep(DEEP_SCAN_INTERVAL_MIN * 60)


def daily_summary_loop():
    log.info('📅 daily_summary_loop 시작')
    while True:
        try:
            now    = datetime.now()
            target = now.replace(
                hour=DAILY_SUMMARY_HOUR_KST, minute=0, second=0, microsecond=0
            )
            if now >= target:
                target += timedelta(days=1)
            time.sleep((target - now).total_seconds())

            watches = load_watch_list()
            actives = load_active_list()
            history = load_history()
            today   = datetime.now().strftime('%Y-%m-%d')
            today_h = [h for h in history if h.get('close_at','').startswith(today)]
            wins    = sum(1 for h in today_h if h.get('pnl_pct', 0) > 0)
            pnl_sum = sum(h.get('pnl_pct', 0) for h in today_h)

            msg = (
                f'📅 <b>일일 요약</b> {today}\n'
                f'Watch: {len(watches)}개 | Active: {len(actives)}개\n'
                f'오늘 종료: {len(today_h)}건 | 승: {wins}건\n'
                f'오늘 수익 합계: {pnl_sum:+.2f}%'
            )
            send_telegram(msg)
            add_event('📅', f'일일 요약 발송 | 오늘 {len(today_h)}건 {pnl_sum:+.1f}%')

        except Exception as e:
            log.error(f'daily_summary_loop 오류: {e}')


def get_scanner_state() -> dict:
    with _state_lock:
        return dict(_scanner_state)


print(f'✅ Scanner v{VERSION} 로드 완료')
print(f'   MTF: {MTF_VERSION}')
