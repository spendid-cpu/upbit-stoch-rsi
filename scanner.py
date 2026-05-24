"""
scanner.py v3.1.0
변경사항:
- v3.0.1: USDT 환율, 주봉 MA20, 이벤트 시스템, 스캔 상태 플래그
- v3.0.2: C등급 Watch 차단, 현재가/변동률 저장
- v3.0.3: timing_warning/overbought_warning 저장
- v3.0.4: entry_price 보존, price_change = 등록가 대비 변동률
- v3.0.5: 일봉 단기/중기 K 방향 점수 반영
- v3.0.6: 일봉 단기 K>15 페널티
- v3.1.0: 사이클 감지 (BOTTOM/RISING/PEAK/FALLING) 저장
           cycle_block Watch 차단, 사이클 배지 저장
"""

import os
import json
import time
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import os
import json
import time
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

import mtf_setup as _mtf
from mtf_setup import (
    VERSION as MTF_VERSION,
    analyze_mtf, btc_ma20_signal, calc_relative_strength,
    PARAMS, _stoch_rsi_k, _sma, _summarize
)


import mtf_setup as _mtf
from mtf_setup import (
    VERSION as MTF_VERSION,
    analyze_mtf, btc_ma20_signal, calc_relative_strength,
    PARAMS, _stoch_rsi_k, _sma, _summarize
)

# ── 로깅 ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

VERSION = 'v3.1.0'

# ── 환경변수 ──────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
UPBIT_ACCESS_KEY = os.environ.get('UPBIT_ACCESS_KEY', '')
UPBIT_SECRET_KEY = os.environ.get('UPBIT_SECRET_KEY', '')

SCAN_INTERVAL_MIN   = int(os.environ.get('SCAN_INTERVAL_MIN',   60))
RESCAN_INTERVAL_MIN = int(os.environ.get('RESCAN_INTERVAL_MIN', 15))
PRICE_CHECK_SEC     = int(os.environ.get('PRICE_CHECK_SEC',     60))
DEEP_CHECK_SEC      = int(os.environ.get('DEEP_CHECK_SEC',      300))
REQUEST_DELAY       = float(os.environ.get('REQUEST_DELAY',     0.35))
MAX_WORKERS         = int(os.environ.get('MAX_WORKERS',         3))
CANDLE_COUNT        = int(os.environ.get('CANDLE_COUNT',        100))

TP_PCT             = float(os.environ.get('TP_PCT',             5.0))
SL_PCT             = float(os.environ.get('SL_PCT',             3.0))
WATCH_EXPIRE_DAYS  = int(os.environ.get('WATCH_EXPIRE_DAYS',    7))
BTC_DROP_THRESHOLD = float(os.environ.get('BTC_DROP_THRESHOLD', -1.0))
PORT               = int(os.environ.get('PORT',                 8080))

# ── 파일 경로 ─────────────────────────────────────────
BASE_DIR     = os.environ.get('DATA_DIR', '/app/data')
os.makedirs(BASE_DIR, exist_ok=True)

WATCH_FILE   = os.path.join(BASE_DIR, 'watch_list.json')
ACTIVE_FILE  = os.path.join(BASE_DIR, 'active_list.json')
HISTORY_FILE = os.path.join(BASE_DIR, 'trade_history.json')
DEEP_FILE    = os.path.join(BASE_DIR, 'deep_list.json')
STATE_FILE   = os.path.join(BASE_DIR, 'scanner_state.json')
EVENT_FILE   = os.path.join(BASE_DIR, 'events.json')

# ── 글로벌 상태 ───────────────────────────────────────
_scanner_state = {
    'version':          VERSION,
    'mtf_version':      MTF_VERSION,
    'running':          False,
    'watch_rescanning': False,
    'price_checking':   False,
    'deep_scanning':    False,
    'last_scan':        None,
    'next_scan':        None,
    'scan_count':       0,
    'total_symbols':    0,
    'btc_price':        0,
    'btc_daily_ma20':   0,
    'btc_weekly_ma20':  0,
    'btc_daily_signal': 'UNKNOWN',
    'btc_weekly_signal':'UNKNOWN',
    'btc_change_1h':    0.0,
    'usdt_rate':        1450.0,
    'total_trades':     0,
    'win_trades':       0,
    'total_pnl':        0.0,
}
_state_lock = threading.Lock()

# ── JSON 유틸 ─────────────────────────────────────────
def _load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        log.warning(f'_load_json {path}: {e}')
    return default

def _save_json(path, data):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f'_save_json {path}: {e}')

# ── 이벤트 로그 ───────────────────────────────────────
def _log_event(event_type, message, data=None):
    events = _load_json(EVENT_FILE, [])
    events.append({
        'time':    datetime.now().strftime('%m/%d %H:%M'),
        'type':    event_type,
        'message': message,
        'data':    data or {}
    })
    events = events[-100:]
    _save_json(EVENT_FILE, events)

# ── 텔레그램 ──────────────────────────────────────────
def _send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
            timeout=10
        )
    except Exception as e:
        log.warning(f'Telegram: {e}')

# ── Upbit API ─────────────────────────────────────────
def _upbit_get(endpoint, params=None):
    url = f'https://api.upbit.com/v1{endpoint}'
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 429:
            log.warning(f'Upbit 429: {endpoint}')
            time.sleep(1.0)
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f'Upbit API {endpoint}: {e}')
        return None

def get_krw_tickers():
    data = _upbit_get('/market/all', {'isDetails': 'false'})
    if not data:
        return []
    return [d['market'] for d in data if d['market'].startswith('KRW-')]

def get_candles(market, unit='days', count=100):
    if unit == 'days':
        endpoint = f'/candles/days'
    elif unit == 'weeks':
        endpoint = f'/candles/weeks'
    else:
        endpoint = f'/candles/minutes/{unit}'
    data = _upbit_get(endpoint, {'market': market, 'count': count})
    if not data:
        return []
    return [float(c['trade_price']) for c in reversed(data)]

def get_current_price(market):
    data = _upbit_get('/ticker', {'markets': market})
    if not data:
        return None
    return float(data[0]['trade_price'])

def get_usdt_rate():
    data = _upbit_get('/ticker', {'markets': 'KRW-USDT'})
    if not data:
        return 1450.0
    return float(data[0]['trade_price'])

def get_volume_ratio(market):
    """최근 거래량 대비 20일 평균 거래량 비율"""
    data = _upbit_get('/candles/days', {'market': market, 'count': 21})
    if not data or len(data) < 2:
        return 1.0
    vols = [float(c['candle_acc_trade_volume']) for c in data]
    avg  = np.mean(vols[1:]) if len(vols) > 1 else vols[0]
    return round(vols[0] / avg, 2) if avg > 0 else 1.0

def get_price_change_pct(market):
    data = _upbit_get('/ticker', {'markets': market})
    if not data:
        return 0.0
    return round(float(data[0].get('signed_change_rate', 0)) * 100, 2)

def get_btc_info():
    """BTC 가격, 1h 변동률, 일봉/주봉 MA20"""
    daily  = get_candles('KRW-BTC', 'days',  21)
    weekly = get_candles('KRW-BTC', 'weeks', 21)
    h1     = get_candles('KRW-BTC', '60',    3)

    price     = daily[-1]  if daily  else 0
    change_1h = round((h1[-1] - h1[-2]) / h1[-2] * 100, 2) if len(h1) >= 2 else 0.0

    daily_sig  = btc_ma20_signal(daily,  20)
    weekly_sig = btc_ma20_signal(weekly, 20)

    return {
        'price':          price,
        'change_1h':      change_1h,
        'daily_ma20':     daily_sig.get('ma20', 0),
        'weekly_ma20':    weekly_sig.get('ma20', 0),
        'daily_signal':   daily_sig.get('signal', 'UNKNOWN'),
        'weekly_signal':  weekly_sig.get('signal', 'UNKNOWN'),
    }

# ── 종목 분석 ─────────────────────────────────────────
def analyze_ticker(market):
    """단일 종목 MTF 분석 → summary 반환"""
    try:
        daily = get_candles(market, 'days',  CANDLE_COUNT)
        h4    = get_candles(market, '240',   CANDLE_COUNT)
        h1    = get_candles(market, '60',    CANDLE_COUNT)

        if len(daily) < 50 or len(h4) < 50 or len(h1) < 50:
            return None

        time.sleep(REQUEST_DELAY)

        mtf     = analyze_mtf(daily, h4, h1)
        summary = _summarize(mtf)

        # 거래량 / 현재가 / 변동률
        volume_ratio = get_volume_ratio(market)
        current_price = daily[-1] if daily else 0
        price_change  = get_price_change_pct(market)

        # 바닥일수 (일봉 장기 oversold 연속일)
        bottom_days = 0
        long_p = PARAMS['long']
        k_arr  = _stoch_rsi_k(daily, long_p['rsi'], long_p['stoch'], long_p['k_smooth'])
        d_arr  = _sma(k_arr, long_p['d_smooth'])
        valid  = ~np.isnan(d_arr)
        if valid.sum() > 0:
            vi = np.where(valid)[0]
            for i in reversed(vi):
                if k_arr[i] <= 20:
                    bottom_days += 1
                else:
                    break

        # 상대강도
        btc_daily = get_candles('KRW-BTC', 'days', CANDLE_COUNT)
        rs_data   = calc_relative_strength(daily, btc_daily)

        return {
            **summary,
            'market':        market,
            'current_price': current_price,
            'price_change':  price_change,
            'volume_ratio':  volume_ratio,
            'bottom_days':   bottom_days,
            'rs':            rs_data.get('rs', 0),
            'rs_grade':      rs_data.get('grade', 'B'),
            # 사이클 정보
            'd_short_cycle': summary.get('d_short_cycle', 'RISING'),
            'd_mid_cycle':   summary.get('d_mid_cycle',   'RISING'),
            'h4_cycle':      summary.get('h4_cycle',      'RISING'),
            'h1_cycle':      summary.get('h1_cycle',      'RISING'),
            'cycle_block':   summary.get('cycle_block',   False),
        }
    except Exception as e:
        log.warning(f'analyze_ticker {market}: {e}')
        return None

# ── entry_price 대비 변동률 ───────────────────────────
def _calc_price_change_from_entry(current_price, entry_price):
    if not entry_price or entry_price == 0:
        return 0.0
    return round((current_price - entry_price) / entry_price * 100, 2)

# ── Watch 아이템 생성 ─────────────────────────────────
def _make_watch_item(result):
    now = datetime.now()
    return {
        'market':             result['market'],
        'grade':              result['grade'],
        'score':              result['score'],
        'entry_price':        result['current_price'],
        'current_price':      result['current_price'],
        'price_change':       0.0,
        'volume_ratio':       result['volume_ratio'],
        'bottom_days':        result['bottom_days'],
        'aligned':            result['aligned'],
        'watch_eligible':     result['watch_eligible'],
        'auto_entry':         result['auto_entry'],
        'timing_warning':     result['timing_warning'],
        'overbought_warning': result['overbought_warning'],
        'cycle_block':        result['cycle_block'],
        'd_short_cycle':      result['d_short_cycle'],
        'd_mid_cycle':        result['d_mid_cycle'],
        'h4_cycle':           result['h4_cycle'],
        'h1_cycle':           result['h1_cycle'],
        'h4_gc':              result['h4_gc'],
        'h1_gc':              result['h1_gc'],
        'daily_gc':           result['daily_gc'],
        'rs':                 result['rs'],
        'rs_grade':           result['rs_grade'],
        'registered_at':      now.strftime('%Y-%m-%d %H:%M'),
        'expire_at':          (now + timedelta(days=WATCH_EXPIRE_DAYS)).strftime('%Y-%m-%d %H:%M'),
        'status':             'watch',
    }

# ── Active 아이템 생성 ────────────────────────────────
def _make_active_item(watch_item, current_price):
    now = datetime.now()
    ep  = watch_item.get('entry_price', current_price)
    return {
        **watch_item,
        'entry_price':   ep,
        'current_price': current_price,
        'price_change':  _calc_price_change_from_entry(current_price, ep),
        'tp_price':      round(ep * (1 + TP_PCT / 100), 4),
        'sl_price':      round(ep * (1 - SL_PCT / 100), 4),
        'entered_at':    now.strftime('%Y-%m-%d %H:%M'),
        'status':        'active',
    }

# ── Watch 관리 API ────────────────────────────────────
def add_watch(market):
    watch = _load_json(WATCH_FILE, [])
    if any(w['market'] == market for w in watch):
        return {'ok': False, 'msg': f'{market} 이미 등록됨'}
    result = analyze_ticker(market)
    if not result:
        return {'ok': False, 'msg': '분석 실패'}
    item = _make_watch_item(result)
    watch.append(item)
    _save_json(WATCH_FILE, watch)
    _log_event('WATCH_ADD', f'{market} Watch 수동 등록', {'grade': item['grade'], 'score': item['score']})
    return {'ok': True, 'item': item}

def remove_watch(market):
    watch = _load_json(WATCH_FILE, [])
    before = len(watch)
    watch  = [w for w in watch if w['market'] != market]
    _save_json(WATCH_FILE, watch)
    _log_event('WATCH_REMOVE', f'{market} Watch 제거')
    return {'ok': True, 'removed': before - len(watch)}

def activate_watch(market):
    watch  = _load_json(WATCH_FILE,  [])
    active = _load_json(ACTIVE_FILE, [])
    item   = next((w for w in watch if w['market'] == market), None)
    if not item:
        return {'ok': False, 'msg': '없는 종목'}
    price = get_current_price(market) or item['current_price']
    active_item = _make_active_item(item, price)
    active.append(active_item)
    watch = [w for w in watch if w['market'] != market]
    _save_json(WATCH_FILE,  watch)
    _save_json(ACTIVE_FILE, active)
    _log_event('ACTIVE_ENTER', f'{market} 진입', {'price': price, 'grade': item['grade']})
    _send_telegram(f'🟢 진입: {market}\n등급: {item["grade"]} {item["score"]}점\n가격: {price:,.0f}')
    return {'ok': True, 'item': active_item}

def close_active(market, reason='manual'):
    active  = _load_json(ACTIVE_FILE,  [])
    history = _load_json(HISTORY_FILE, [])
    item    = next((a for a in active if a['market'] == market), None)
    if not item:
        return {'ok': False, 'msg': '없는 종목'}
    price  = get_current_price(market) or item['current_price']
    ep     = item.get('entry_price', price)
    pnl    = round((price - ep) / ep * 100, 2)
    closed = {
        **item,
        'exit_price': price,
        'pnl':        pnl,
        'closed_at':  datetime.now().strftime('%Y-%m-%d %H:%M'),
        'reason':     reason,
        'status':     'closed',
    }
    history.append(closed)
    active = [a for a in active if a['market'] != market]
    _save_json(ACTIVE_FILE,  active)
    _save_json(HISTORY_FILE, history)
    # 통계 업데이트
    with _state_lock:
        _scanner_state['total_trades'] += 1
        if pnl > 0:
            _scanner_state['win_trades'] += 1
        _scanner_state['total_pnl'] = round(_scanner_state['total_pnl'] + pnl, 2)
    emoji = '🟢' if pnl > 0 else '🔴'
    _log_event('ACTIVE_CLOSE', f'{market} 청산 {pnl:+.2f}%', {'price': price, 'pnl': pnl, 'reason': reason})
    _send_telegram(f'{emoji} 청산: {market}\n수익: {pnl:+.2f}%\n사유: {reason}')
    return {'ok': True, 'pnl': pnl}

def reset_watch_list():
    _save_json(WATCH_FILE, [])
    _log_event('WATCH_RESET', 'Watch 목록 초기화')
    return {'ok': True}

# ── 단일 스캔 (수동) ──────────────────────────────────
def run_single_scan():
    t = threading.Thread(target=_run_full_scan, daemon=True)
    t.start()
    return {'ok': True, 'msg': '스캔 시작'}

# ── 전체 스캔 ─────────────────────────────────────────
def _run_full_scan():
    with _state_lock:
        if _scanner_state['running']:
            return
        _scanner_state['running'] = True
        _scanner_state['last_scan'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    log.info('=== 전체 스캔 시작 ===')
    try:
        # BTC 정보
        btc = get_btc_info()
        usdt_rate = get_usdt_rate()
        with _state_lock:
            _scanner_state.update({
                'btc_price':        btc['price'],
                'btc_daily_ma20':   btc['daily_ma20'],
                'btc_weekly_ma20':  btc['weekly_ma20'],
                'btc_daily_signal': btc['daily_signal'],
                'btc_weekly_signal':btc['weekly_signal'],
                'btc_change_1h':    btc['change_1h'],
                'usdt_rate':        usdt_rate,
            })

        tickers = get_krw_tickers()
        with _state_lock:
            _scanner_state['total_symbols'] = len(tickers)
        log.info(f'대상 종목: {len(tickers)}개')

        watch  = _load_json(WATCH_FILE,  [])
        active = _load_json(ACTIVE_FILE, [])
        watch_markets  = {w['market'] for w in watch}
        active_markets = {a['market'] for a in active}

        new_watch  = []
        new_active = []

        def _process(market):
            if market in active_markets:
                return None
            time.sleep(REQUEST_DELAY)
            return analyze_ticker(market)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(_process, t): t for t in tickers}
            for future in as_completed(futures):
                result = future.result()
                if not result:
                    continue
                market = result['market']

                # 이미 Watch에 있으면 entry_price 보존 후 업데이트
                if market in watch_markets:
                    for w in watch:
                        if w['market'] == market:
                            ep = w.get('entry_price', result['current_price'])
                            w.update({
                                'grade':              result['grade'],
                                'score':              result['score'],
                                'current_price':      result['current_price'],
                                'price_change':       _calc_price_change_from_entry(result['current_price'], ep),
                                'volume_ratio':       result['volume_ratio'],
                                'aligned':            result['aligned'],
                                'watch_eligible':     result['watch_eligible'],
                                'auto_entry':         result['auto_entry'],
                                'timing_warning':     result['timing_warning'],
                                'overbought_warning': result['overbought_warning'],
                                'cycle_block':        result['cycle_block'],
                                'd_short_cycle':      result['d_short_cycle'],
                                'd_mid_cycle':        result['d_mid_cycle'],
                                'h4_cycle':           result['h4_cycle'],
                                'h1_cycle':           result['h1_cycle'],
                                'h4_gc':              result['h4_gc'],
                                'h1_gc':              result['h1_gc'],
                                'daily_gc':           result['daily_gc'],
                            })
                            # cycle_block이 되면 Watch 제거
                            if result['cycle_block'] or result['grade'] == 'X':
                                watch = [w2 for w2 in watch if w2['market'] != market]
                                _log_event('WATCH_REMOVE', f'{market} 사이클 차단으로 제거',
                                           {'cycle': result['d_short_cycle']})
                    continue

                # 신규 Watch 등록 조건
                if result['watch_eligible'] and result['grade'] in ('S', 'A', 'B'):
                    item = _make_watch_item(result)
                    new_watch.append(item)
                    log.info(f'📋 Watch 등록: {market} [{result["grade"]}] {result["score"]}점 '
                             f'd_short:{result["d_short_cycle"]} d_mid:{result["d_mid_cycle"]}')
                    _log_event('WATCH_ADD', f'{market} Watch 등록',
                               {'grade': result['grade'], 'score': result['score'],
                                'd_short_cycle': result['d_short_cycle'],
                                'd_mid_cycle': result['d_mid_cycle']})

                    # 자동진입
                    if result['auto_entry']:
                        active_item = _make_active_item(item, result['current_price'])
                        new_active.append(active_item)
                        _log_event('ACTIVE_ENTER', f'{market} 자동진입',
                                   {'price': result['current_price'], 'grade': result['grade']})
                        _send_telegram(
                            f'🟢 자동진입: {market}\n'
                            f'등급: {result["grade"]} {result["score"]}점\n'
                            f'd_short: {result["d_short_cycle"]} / d_mid: {result["d_mid_cycle"]}\n'
                            f'가격: {result["current_price"]:,.0f}'
                        )

        # 만료 Watch 제거
        now_dt = datetime.now()
        watch = [w for w in watch if
                 datetime.strptime(w.get('expire_at', '2099-01-01 00:00'), '%Y-%m-%d %H:%M') > now_dt]

        watch  += new_watch
        active += new_active

        _save_json(WATCH_FILE,  watch)
        _save_json(ACTIVE_FILE, active)

        with _state_lock:
            _scanner_state['scan_count'] += 1
            _scanner_state['next_scan']   = (
                datetime.now() + timedelta(minutes=SCAN_INTERVAL_MIN)
            ).strftime('%Y-%m-%d %H:%M:%S')

        log.info(f'=== 스캔 완료: Watch {len(watch)}개 / Active {len(active)}개 ===')

    except Exception as e:
        log.error(f'_run_full_scan error: {e}', exc_info=True)
    finally:
        with _state_lock:
            _scanner_state['running'] = False

# ── Watch 재스캔 ──────────────────────────────────────
def _run_watch_rescan():
    with _state_lock:
        if _scanner_state['watch_rescanning']:
            return
        _scanner_state['watch_rescanning'] = True

    try:
        watch  = _load_json(WATCH_FILE,  [])
        active = _load_json(ACTIVE_FILE, [])
        active_markets = {a['market'] for a in active}

        new_active = []
        remove_markets = []

        for item in watch:
            market = item['market']
            if market in active_markets:
                continue
            try:
                time.sleep(REQUEST_DELAY)
                result = analyze_ticker(market)
                if not result:
                    continue

                ep = item.get('entry_price', result['current_price'])

                # 사이클 차단 → Watch 제거
                if result['cycle_block'] or result['grade'] == 'X':
                    remove_markets.append(market)
                    _log_event('WATCH_REMOVE', f'{market} 재스캔: 사이클 차단',
                               {'d_short_cycle': result['d_short_cycle'],
                                'd_mid_cycle':   result['d_mid_cycle']})
                    continue

                item.update({
                    'grade':              result['grade'],
                    'score':              result['score'],
                    'current_price':      result['current_price'],
                    'price_change':       _calc_price_change_from_entry(result['current_price'], ep),
                    'volume_ratio':       result['volume_ratio'],
                    'aligned':            result['aligned'],
                    'watch_eligible':     result['watch_eligible'],
                    'auto_entry':         result['auto_entry'],
                    'timing_warning':     result['timing_warning'],
                    'overbought_warning': result['overbought_warning'],
                    'cycle_block':        result['cycle_block'],
                    'd_short_cycle':      result['d_short_cycle'],
                    'd_mid_cycle':        result['d_mid_cycle'],
                    'h4_cycle':           result['h4_cycle'],
                    'h1_cycle':           result['h1_cycle'],
                    'h4_gc':              result['h4_gc'],
                    'h1_gc':              result['h1_gc'],
                    'daily_gc':           result['daily_gc'],
                })

                # 자동진입 조건 충족
                if result['auto_entry'] and not item.get('timing_warning') and not item.get('overbought_warning'):
                    active_item = _make_active_item(item, result['current_price'])
                    new_active.append(active_item)
                    remove_markets.append(market)
                    _log_event('ACTIVE_ENTER', f'{market} 재스캔 자동진입',
                               {'price': result['current_price'], 'grade': result['grade']})
                    _send_telegram(
                        f'🟢 자동진입(재스캔): {market}\n'
                        f'등급: {result["grade"]} {result["score"]}점\n'
                        f'd_short: {result["d_short_cycle"]} / d_mid: {result["d_mid_cycle"]}\n'
                        f'가격: {result["current_price"]:,.0f}'
                    )

            except Exception as e:
                log.warning(f'watch_rescan {market}: {e}')

        # 만료 제거
        now_dt = datetime.now()
        watch = [w for w in watch if
                 w['market'] not in remove_markets and
                 datetime.strptime(w.get('expire_at', '2099-01-01 00:00'), '%Y-%m-%d %H:%M') > now_dt]

        active += new_active
        _save_json(WATCH_FILE,  watch)
        _save_json(ACTIVE_FILE, active)
        log.info(f'Watch 재스캔 완료: {len(watch)}개')

    except Exception as e:
        log.error(f'_run_watch_rescan error: {e}', exc_info=True)
    finally:
        with _state_lock:
            _scanner_state['watch_rescanning'] = False

# ── 가격 체크 (TP/SL) ────────────────────────────────
def _run_price_check():
    with _state_lock:
        if _scanner_state['price_checking']:
            return
        _scanner_state['price_checking'] = True
    try:
        active = _load_json(ACTIVE_FILE, [])
        for item in active:
            market = item['market']
            try:
                price = get_current_price(market)
                if not price:
                    continue
                ep = item.get('entry_price', price)
                item['current_price'] = price
                item['price_change']  = _calc_price_change_from_entry(price, ep)

                if price >= item.get('tp_price', float('inf')):
                    close_active(market, reason='TP')
                elif price <= item.get('sl_price', 0):
                    close_active(market, reason='SL')

            except Exception as e:
                log.warning(f'price_check {market}: {e}')
            time.sleep(REQUEST_DELAY)

        _save_json(ACTIVE_FILE, active)
    except Exception as e:
        log.error(f'_run_price_check error: {e}', exc_info=True)
    finally:
        with _state_lock:
            _scanner_state['price_checking'] = False

# ── DEEP 스캔 ─────────────────────────────────────────
def run_deep_scan():
    with _state_lock:
        if _scanner_state['deep_scanning']:
            return
        _scanner_state['deep_scanning'] = True
    try:
        btc_change = _scanner_state.get('btc_change_1h', 0)
        if btc_change > BTC_DROP_THRESHOLD:
            return

        log.info(f'DEEP 스캔 시작 (BTC 1h: {btc_change}%)')
        tickers   = get_krw_tickers()
        btc_daily = get_candles('KRW-BTC', 'days', CANDLE_COUNT)
        results   = []

        for market in tickers[:50]:
            try:
                daily = get_candles(market, 'days', CANDLE_COUNT)
                rs    = calc_relative_strength(daily, btc_daily)
                if rs['grade'] in ('S', 'A'):
                    results.append({'market': market, 'rs': rs['rs'], 'rs_grade': rs['grade']})
                time.sleep(REQUEST_DELAY)
            except:
                pass

        results.sort(key=lambda x: x['rs'], reverse=True)
        _save_json(DEEP_FILE, results[:30])
        _log_event('DEEP_SCAN', f'DEEP 스캔 완료: {len(results)}개 발견')
    except Exception as e:
        log.error(f'run_deep_scan error: {e}', exc_info=True)
    finally:
        with _state_lock:
            _scanner_state['deep_scanning'] = False

# ── 루프들 ────────────────────────────────────────────
def scanner_loop():
    time.sleep(5)
    while True:
        try:
            _run_full_scan()
        except Exception as e:
            log.error(f'scanner_loop: {e}')
        time.sleep(SCAN_INTERVAL_MIN * 60)

def watch_rescan_loop():
    time.sleep(90)
    while True:
        try:
            _run_watch_rescan()
        except Exception as e:
            log.error(f'watch_rescan_loop: {e}')
        time.sleep(RESCAN_INTERVAL_MIN * 60)

def price_check_loop():
    time.sleep(30)
    while True:
        try:
            _run_price_check()
        except Exception as e:
            log.error(f'price_check_loop: {e}')
        time.sleep(PRICE_CHECK_SEC)

def active_monitor_loop():
    time.sleep(60)
    while True:
        try:
            active = _load_json(ACTIVE_FILE, [])
            if active:
                _run_price_check()
        except Exception as e:
            log.error(f'active_monitor_loop: {e}')
        time.sleep(30)

def deep_scan_loop():
    time.sleep(120)
    while True:
        try:
            run_deep_scan()
        except Exception as e:
            log.error(f'deep_scan_loop: {e}')
        time.sleep(DEEP_CHECK_SEC)

def daily_summary_loop():
    while True:
        try:
            now = datetime.now()
            # 매일 오전 9시 요약
            target = now.replace(hour=9, minute=0, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            time.sleep((target - now).total_seconds())

            state   = _scanner_state
            watch   = _load_json(WATCH_FILE,  [])
            active  = _load_json(ACTIVE_FILE, [])
            history = _load_json(HISTORY_FILE, [])

            msg = (
                f'📊 일간 요약 {now.strftime("%m/%d")}\n'
                f'BTC: {state["btc_price"]:,.0f}원\n'
                f'Watch: {len(watch)}개 / Active: {len(active)}개\n'
                f'총 거래: {state["total_trades"]}회 / 승률: '
                f'{round(state["win_trades"]/state["total_trades"]*100) if state["total_trades"] else 0}%\n'
                f'누적 PnL: {state["total_pnl"]:+.2f}%'
            )
            _send_telegram(msg)
            _log_event('DAILY_SUMMARY', msg)
        except Exception as e:
            log.error(f'daily_summary_loop: {e}')
            time.sleep(3600)

# ── 상태 조회 ─────────────────────────────────────────
def get_scanner_state():
    with _state_lock:
        return dict(_scanner_state)

# ── 메인 ──────────────────────────────────────────────
if __name__ == '__main__':
    print(f'✅ Scanner {VERSION} + MTF {MTF_VERSION} 시작')
    print(f'   사이클 감지: BOTTOM/RISING/PEAK/FALLING')
    print(f'   PEAK/FALLING → Watch 자동 차단 ❌')
    print(f'   entry_price 보존: price_change = 등록가 대비')
    print(f'   스캔 주기: {SCAN_INTERVAL_MIN}분 / 재스캔: {RESCAN_INTERVAL_MIN}분')
