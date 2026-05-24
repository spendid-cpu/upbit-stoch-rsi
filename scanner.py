"""
scanner.py  v3.0.4
─────────────────────────────────────────────
변경사항:
  v3.0.1  USDT 환율, 주봉 MA20, 이벤트 시스템, 스캔상태 플래그
  v3.0.2  C등급 Watch 등록 차단, 현재가+변동률 저장
  v3.0.3  타이밍경고(timing_warning/overbought_warning) Watch 아이템 저장
  v3.0.4  entry_price 유지 (재스캔 시 덮어쓰기 방지)
          price_change → 등록가 대비 변동률로 수정
          전체스캔 시 기존 Watch 현재가 갱신 로직 추가
─────────────────────────────────────────────
"""

VERSION = 'v3.0.4'

import os, json, time, logging, threading
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import mtf_setup as _mtf
from mtf_setup import (
    VERSION          as MTF_VERSION,
    analyze_mtf,
    btc_ma20_signal,
    calc_relative_strength,
    PARAMS,
    _stoch_rsi_k,
    _sma,
)

# ── 로깅 ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── 환경변수 ──────────────────────────────────────────
TELEGRAM_TOKEN    = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID  = os.environ.get('TELEGRAM_CHAT_ID', '')
UPBIT_ACCESS      = os.environ.get('UPBIT_ACCESS_KEY', '')
UPBIT_SECRET      = os.environ.get('UPBIT_SECRET_KEY', '')

SCAN_INTERVAL_MIN   = int(os.environ.get('SCAN_INTERVAL_MIN',   '60'))
RESCAN_INTERVAL_MIN = int(os.environ.get('RESCAN_INTERVAL_MIN', '15'))
PRICE_CHECK_SEC     = int(os.environ.get('PRICE_CHECK_SEC',     '60'))
DEEP_CHECK_SEC      = int(os.environ.get('DEEP_CHECK_SEC',      '300'))
REQUEST_DELAY       = float(os.environ.get('REQUEST_DELAY',     '0.35'))
MAX_WORKERS         = int(os.environ.get('MAX_WORKERS',         '3'))
CANDLE_COUNT        = int(os.environ.get('CANDLE_COUNT',        '100'))

TP_PCT            = float(os.environ.get('TP_PCT',   '5.0'))
SL_PCT            = float(os.environ.get('SL_PCT',   '3.0'))
WATCH_EXPIRE_DAYS = int(os.environ.get('WATCH_EXPIRE_DAYS', '7'))

BTC_DROP_THRESHOLD = float(os.environ.get('BTC_DROP_THRESHOLD', '-1.0'))

PORT = int(os.environ.get('PORT', '8080'))

# ── 파일 경로 ─────────────────────────────────────────
BASE_DIR     = os.environ.get('DATA_DIR', '/app/data')
os.makedirs(BASE_DIR, exist_ok=True)

WATCH_FILE   = os.path.join(BASE_DIR, 'watch_list.json')
ACTIVE_FILE  = os.path.join(BASE_DIR, 'active_list.json')
HISTORY_FILE = os.path.join(BASE_DIR, 'trade_history.json')
DEEP_FILE    = os.path.join(BASE_DIR, 'deep_list.json')
STATE_FILE   = os.path.join(BASE_DIR, 'scanner_state.json')
EVENT_FILE   = os.path.join(BASE_DIR, 'events.json')

# ── 전역 상태 ─────────────────────────────────────────
_state_lock = threading.Lock()
_scanner_state = {
    'version':           VERSION,
    'mtf_version':       MTF_VERSION,
    'running':           False,
    'watch_rescanning':  False,
    'price_checking':    False,
    'deep_scanning':     False,
    'last_scan':         None,
    'next_scan':         None,
    'scan_count':        0,
    'total_symbols':     0,
    'btc_price':         None,
    'btc_daily_ma20':    None,
    'btc_weekly_ma20':   None,
    'btc_daily_signal':  None,
    'btc_weekly_signal': None,
    'btc_change_1h':     None,
    'usdt_rate':         None,
    'total_trades':      0,
    'win_trades':        0,
    'total_pnl':         0.0,
}


# ── JSON 헬퍼 ─────────────────────────────────────────
def _load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        log.warning(f'JSON load error {path}: {e}')
    return default

def _save_json(path: str, data):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f'JSON save error {path}: {e}')


# ── 이벤트 로그 ───────────────────────────────────────
def _log_event(msg: str, etype: str = 'info'):
    events = _load_json(EVENT_FILE, [])
    events.append({
        'time':    datetime.now().strftime('%H:%M:%S'),
        'message': msg,
        'type':    etype,
    })
    events = events[-20:]
    _save_json(EVENT_FILE, events)


# ── Upbit API ─────────────────────────────────────────
UPBIT_BASE = 'https://api.upbit.com/v1'

def _upbit_get(endpoint: str, params: dict = None, retries: int = 3):
    url = f'{UPBIT_BASE}{endpoint}'
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 429:
                log.warning(f'429 Too Many Requests – {endpoint} (attempt {attempt+1})')
                time.sleep(REQUEST_DELAY * (attempt + 2))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f'Upbit API 실패 {url}: {e}')
            time.sleep(REQUEST_DELAY)
    return None

def get_krw_tickers() -> list:
    data = _upbit_get('/market/all', {'isDetails': 'false'})
    if not data:
        return []
    return [d['market'] for d in data if d['market'].startswith('KRW-')]

def get_candles(market: str, unit: str, count: int = 100) -> list:
    if unit == 'days':
        endpoint = '/candles/days'
        params   = {'market': market, 'count': count}
    elif unit == 'weeks':
        endpoint = '/candles/weeks'
        params   = {'market': market, 'count': count}
    else:
        endpoint = f'/candles/{unit}'
        params   = {'market': market, 'count': count}
    data = _upbit_get(endpoint, params)
    if not data:
        return []
    return [c['trade_price'] for c in reversed(data)]

def get_current_price(market: str):
    data = _upbit_get('/ticker', {'markets': market})
    if data and isinstance(data, list):
        return data[0].get('trade_price')
    return None

def get_usdt_rate() -> float:
    data = _upbit_get('/ticker', {'markets': 'KRW-USDT'})
    if data and isinstance(data, list):
        return float(data[0].get('trade_price', 1350))
    return 1350.0

def get_volume_ratio(market: str) -> float:
    try:
        data = _upbit_get('/candles/days', {'market': market, 'count': 21})
        if not data or len(data) < 2:
            return 1.0
        volumes    = [c['candle_acc_trade_volume'] for c in data]
        recent_vol = volumes[0]
        avg_vol    = sum(volumes[1:]) / len(volumes[1:])
        return round(recent_vol / avg_vol, 2) if avg_vol > 0 else 1.0
    except:
        return 1.0

def get_price_change_pct(market: str) -> float:
    data = _upbit_get('/ticker', {'markets': market})
    if data and isinstance(data, list):
        return round(float(data[0].get('signed_change_rate', 0)) * 100, 2)
    return 0.0

def get_btc_info() -> dict:
    try:
        usdt_rate     = get_usdt_rate()
        daily_closes  = get_candles('KRW-BTC', 'days',  count=30)
        weekly_closes = get_candles('KRW-BTC', 'weeks', count=25)
        ma_info       = btc_ma20_signal(daily_closes, weekly_closes)
        btc_price     = get_current_price('KRW-BTC') or 0
        btc_1h_chg    = get_price_change_pct('KRW-BTC')
        return {
            'price':         btc_price,
            'daily_ma20':    ma_info.get('daily_ma20'),
            'weekly_ma20':   ma_info.get('weekly_ma20'),
            'daily_signal':  ma_info.get('daily_signal'),
            'weekly_signal': ma_info.get('weekly_signal'),
            'usdt_rate':     usdt_rate,
            'change_1h':     btc_1h_chg,
        }
    except Exception as e:
        log.warning(f'get_btc_info error: {e}')
        return {}


# ── 텔레그램 ──────────────────────────────────────────
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
        log.warning(f'Telegram error: {e}')


# ── 티커 분석 ─────────────────────────────────────────
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

        mtf     = analyze_mtf(daily, h4, h1)
        summary = _mtf._summarize(mtf)

        time.sleep(REQUEST_DELAY)
        volume_ratio  = get_volume_ratio(market)
        time.sleep(REQUEST_DELAY)
        current_price = get_current_price(market) or 0

        # 바닥일수: 일봉 장기 K ≤ 20 연속일
        bottom_days = 0
        p     = PARAMS['long']
        arr   = np.array(daily, dtype=float)
        k_arr = _stoch_rsi_k(arr, p['rsi'], p['stoch'], p['k_smooth'])
        for v in reversed(k_arr):
            if not np.isnan(v) and v <= 20:
                bottom_days += 1
            else:
                break

        return {
            'market':        market,
            'summary':       summary,
            'mtf':           mtf,
            'volume_ratio':  volume_ratio,
            'current_price': current_price,
            'bottom_days':   bottom_days,
        }
    except Exception as e:
        log.warning(f'analyze_ticker error {market}: {e}')
        return None


# ── 등록가 대비 변동률 계산 ───────────────────────────
def _calc_price_change_from_entry(entry_price: float, current_price: float) -> float:
    if not entry_price or entry_price == 0:
        return 0.0
    return round((current_price - entry_price) / entry_price * 100, 2)


# ── Watch / Active 아이템 생성 ────────────────────────
def _make_watch_item(market: str, result: dict) -> dict:
    summary = result['summary']
    mtf     = result['mtf']
    now     = datetime.now()
    price   = result['current_price']
    return {
        'market':             market,
        'grade':              summary['grade'],
        'score':              summary['score'],
        'aligned':            summary.get('aligned', 0),
        'entry_price':        price,   # 등록 시점 가격 – 이후 변경 안 함
        'current_price':      price,
        'price_change':       0.0,     # 등록 시점은 0%
        'volume_ratio':       result['volume_ratio'],
        'bottom_days':        result['bottom_days'],
        'timing_warning':     summary.get('timing_warning',    False),
        'overbought_warning': summary.get('overbought_warning', False),
        'h4_k':               summary.get('h4_k', 50.0),
        'h1_k':               summary.get('h1_k', 50.0),
        'd_long_k':           mtf['daily']['long'].get('k',   50.0),
        'd_long_d':           mtf['daily']['long'].get('d',   50.0),
        'd_mid_k':            mtf['daily']['mid'].get('k',    50.0),
        'd_mid_d':            mtf['daily']['mid'].get('d',    50.0),
        'd_short_k':          mtf['daily']['short'].get('k',  50.0),
        'd_short_d':          mtf['daily']['short'].get('d',  50.0),
        'h4_k_val':           mtf['h4']['short'].get('k',     50.0),
        'h4_d_val':           mtf['h4']['short'].get('d',     50.0),
        'h1_k_val':           mtf['h1']['short'].get('k',     50.0),
        'h1_d_val':           mtf['h1']['short'].get('d',     50.0),
        'h4_gc':              summary.get('h4_gc',    False),
        'h1_gc':              summary.get('h1_gc',    False),
        'daily_gc':           summary.get('daily_gc', False),
        'registered_at':      now.strftime('%Y-%m-%d %H:%M'),
        'expire_at':          (now + timedelta(days=WATCH_EXPIRE_DAYS)).strftime('%Y-%m-%d'),
    }

def _make_active_item(watch_item: dict, current_price: float) -> dict:
    now = datetime.now()
    return {
        'market':        watch_item['market'],
        'grade':         watch_item['grade'],
        'score':         watch_item['score'],
        'entry_price':   current_price,
        'current_price': current_price,
        'tp_price':      round(current_price * (1 + TP_PCT / 100), 4),
        'sl_price':      round(current_price * (1 - SL_PCT / 100), 4),
        'tp_pct':        TP_PCT,
        'sl_pct':        SL_PCT,
        'volume_ratio':  watch_item.get('volume_ratio', 1.0),
        'bottom_days':   watch_item.get('bottom_days',  0),
        'entered_at':    now.strftime('%Y-%m-%d %H:%M'),
    }


# ── Watch 관리 API ────────────────────────────────────
def add_watch(market: str) -> dict:
    market = market.upper()
    if not market.startswith('KRW-'):
        market = f'KRW-{market}'
    watch = _load_json(WATCH_FILE, [])
    if any(w['market'] == market for w in watch):
        return {'success': False, 'message': f'{market} 이미 Watch 중'}
    result = analyze_ticker(market)
    if not result:
        return {'success': False, 'message': '분석 실패'}
    item = _make_watch_item(market, result)
    watch.append(item)
    _save_json(WATCH_FILE, watch)
    _log_event(f'📋 {market} 수동 Watch 등록 [{item["grade"]}] {item["score"]}점', 'watch')
    return {'success': True, 'message': f'{market} Watch 등록 완료', 'item': item}

def remove_watch(market: str) -> dict:
    market = market.upper()
    watch  = _load_json(WATCH_FILE, [])
    before = len(watch)
    watch  = [w for w in watch if w['market'] != market]
    _save_json(WATCH_FILE, watch)
    removed = before - len(watch)
    return {'success': removed > 0, 'message': f'{market} 제거 완료' if removed else '없음'}

def activate_watch(market: str) -> dict:
    market = market.upper()
    watch  = _load_json(WATCH_FILE, [])
    active = _load_json(ACTIVE_FILE, [])
    item   = next((w for w in watch if w['market'] == market), None)
    if not item:
        return {'success': False, 'message': f'{market} Watch 목록에 없음'}
    if any(a['market'] == market for a in active):
        return {'success': False, 'message': f'{market} 이미 Active'}
    price = get_current_price(market)
    if not price:
        return {'success': False, 'message': '현재가 조회 실패'}
    active_item = _make_active_item(item, price)
    active.append(active_item)
    watch = [w for w in watch if w['market'] != market]
    _save_json(ACTIVE_FILE, active)
    _save_json(WATCH_FILE,  watch)
    msg = f'✅ {market} 수동 진입 @ {price:,.0f}원'
    send_telegram(msg)
    _log_event(msg, 'active')
    return {'success': True, 'message': msg, 'item': active_item}

def close_active(market: str, reason: str = 'manual') -> dict:
    market  = market.upper()
    active  = _load_json(ACTIVE_FILE,  [])
    history = _load_json(HISTORY_FILE, [])
    item    = next((a for a in active if a['market'] == market), None)
    if not item:
        return {'success': False, 'message': f'{market} Active 없음'}
    price = get_current_price(market) or item['entry_price']
    pnl   = round((price - item['entry_price']) / item['entry_price'] * 100, 2)
    closed = {**item, 'close_price': price, 'pnl': pnl,
              'reason': reason,
              'closed_at': datetime.now().strftime('%Y-%m-%d %H:%M')}
    history.append(closed)
    active = [a for a in active if a['market'] != market]
    _save_json(ACTIVE_FILE,  active)
    _save_json(HISTORY_FILE, history)
    emoji = '🟢' if pnl >= 0 else '🔴'
    msg   = f'{emoji} {market} {reason} 종료 @ {price:,.0f}원 ({pnl:+.2f}%)'
    send_telegram(msg)
    _log_event(msg, 'close')
    return {'success': True, 'message': msg}

def reset_watch_list() -> dict:
    _save_json(WATCH_FILE, [])
    _log_event('🔄 Watch 목록 초기화', 'system')
    return {'success': True, 'message': 'Watch 목록 초기화 완료'}

def run_single_scan() -> dict:
    with _state_lock:
        if _scanner_state.get('running'):
            return {'success': False, 'message': '스캔 이미 실행 중'}
    log.info('🔄 수동 스캔 트리거')
    try:
        _run_full_scan()
        return {'success': True, 'message': '스캔 완료'}
    except Exception as e:
        return {'success': False, 'message': str(e)}


# ── 전체 스캔 ─────────────────────────────────────────
def _run_full_scan():
    with _state_lock:
        if _scanner_state.get('running'):
            log.info('⏭️ 스캔 이미 실행 중 – skip')
            return
        _scanner_state['running'] = True

    log.info('🚀 전체 스캔 시작')
    _log_event('📡 전체 스캔 시작', 'system')

    try:
        # BTC 정보 업데이트
        btc = get_btc_info()
        if btc:
            with _state_lock:
                _scanner_state.update({
                    'btc_price':         btc.get('price'),
                    'btc_daily_ma20':    btc.get('daily_ma20'),
                    'btc_weekly_ma20':   btc.get('weekly_ma20'),
                    'btc_daily_signal':  btc.get('daily_signal'),
                    'btc_weekly_signal': btc.get('weekly_signal'),
                    'usdt_rate':         btc.get('usdt_rate'),
                    'btc_change_1h':     btc.get('change_1h'),
                })

        tickers = get_krw_tickers()
        with _state_lock:
            _scanner_state['total_symbols'] = len(tickers)
        log.info(f'  대상: {len(tickers)}개 종목')

        watch          = _load_json(WATCH_FILE,  [])
        active         = _load_json(ACTIVE_FILE, [])
        watch_tickers  = {w['market'] for w in watch}
        active_tickers = {a['market'] for a in active}

        new_watch  = []
        new_active = []
        # 기존 Watch 현재가 업데이트용 dict
        watch_price_updates = {}

        def _process(ticker):
            result = analyze_ticker(ticker)
            if not result:
                return

            summary = result['summary']
            grade   = summary.get('grade', '-')

            # ── 이미 Watch에 있는 종목 → 현재가만 갱신 (entry_price 유지) ──
            if ticker in watch_tickers:
                existing = next((w for w in watch if w['market'] == ticker), None)
                if existing:
                    entry_price = existing.get('entry_price', result['current_price'])
                    watch_price_updates[ticker] = {
                        'current_price': result['current_price'],
                        'price_change':  _calc_price_change_from_entry(
                                             entry_price, result['current_price']),
                    }
                return

            # C등급 이하 제외
            if grade in ('C', '-', 'X'):
                return

            # 신규 Watch 등록
            if (summary.get('watch_eligible') and
                ticker not in active_tickers):
                item   = _make_watch_item(ticker, result)
                new_watch.append(item)
                symbol = ticker.replace('KRW-', '')
                warn   = ' ⚠️' if item.get('timing_warning')    else ''
                warn  += ' 🔴' if item.get('overbought_warning') else ''
                log.info(f'  📋 Watch 등록: {symbol} [{grade}] {summary["score"]}점{warn}')
                _log_event(
                    f'📋 {symbol} Watch 등록 [{grade}] {summary["score"]}점{warn}',
                    'watch'
                )

            # 자동 진입 (기존 Watch에서 GC 발생)
            if (summary.get('auto_entry') and
                ticker in watch_tickers and
                ticker not in active_tickers):
                watch_item = next((w for w in watch if w['market'] == ticker), None)
                if watch_item:
                    price       = result['current_price']
                    active_item = _make_active_item(watch_item, price)
                    new_active.append({'ticker': ticker, 'item': active_item})
                    symbol = ticker.replace('KRW-', '')
                    msg    = f'✅ {symbol} 자동 진입 @ {price:,.0f}원 [{grade}]'
                    send_telegram(msg)
                    _log_event(msg, 'active')
                    log.info(msg)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(_process, t): t for t in tickers}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    log.warning(f'process error {futures[f]}: {e}')

        # 기존 Watch 현재가 갱신 (entry_price 건드리지 않음)
        for w in watch:
            if w['market'] in watch_price_updates:
                upd = watch_price_updates[w['market']]
                w['current_price'] = upd['current_price']
                w['price_change']  = upd['price_change']

        # 신규 Watch 추가
        if new_watch:
            existing_markets = {w['market'] for w in watch}
            for item in new_watch:
                if item['market'] not in existing_markets:
                    watch.append(item)
        _save_json(WATCH_FILE, watch)

        # Active 전환
        if new_active:
            active = _load_json(ACTIVE_FILE, [])
            act_m  = {a['market'] for a in active}
            for entry in new_active:
                ticker = entry['ticker']
                if ticker not in act_m:
                    active.append(entry['item'])
                    watch = [w for w in watch if w['market'] != ticker]
            _save_json(ACTIVE_FILE, active)
            _save_json(WATCH_FILE,  watch)

        now = datetime.now()
        with _state_lock:
            _scanner_state['scan_count'] += 1
            _scanner_state['last_scan']   = now.strftime('%Y-%m-%d %H:%M')
            _scanner_state['next_scan']   = (
                now + timedelta(minutes=SCAN_INTERVAL_MIN)
            ).strftime('%Y-%m-%d %H:%M')

        watch_count = len(_load_json(WATCH_FILE, []))
        msg = f'📡 스캔완료 Watch {watch_count}개 (신규 {len(new_watch)}개)'
        log.info(msg)
        _log_event(msg, 'system')

    except Exception as e:
        log.error(f'전체 스캔 오류: {e}')
        _log_event(f'❌ 스캔 오류: {e}', 'error')
    finally:
        with _state_lock:
            _scanner_state['running'] = False


# ── Watch 재스캔 ──────────────────────────────────────
def _run_watch_rescan():
    with _state_lock:
        _scanner_state['watch_rescanning'] = True
    log.info('🔍 Watch 재스캔 시작')
    try:
        watch  = _load_json(WATCH_FILE,  [])
        active = _load_json(ACTIVE_FILE, [])
        if not watch:
            return

        active_tickers = {a['market'] for a in active}
        to_activate    = []
        to_remove      = []
        updated_watch  = []

        for item in watch:
            market = item['market']
            result = analyze_ticker(market)
            if not result:
                updated_watch.append(item)
                continue

            summary = result['summary']
            grade   = summary.get('grade', '-')

            # 만료 체크
            try:
                expire = datetime.strptime(item['expire_at'], '%Y-%m-%d')
                if datetime.now() > expire:
                    to_remove.append(market)
                    _log_event(
                        f'⏰ {market.replace("KRW-","")} Watch 만료 제거', 'system')
                    continue
            except:
                pass

            # 등급 하락 → 제거
            if grade in ('C', '-', 'X'):
                to_remove.append(market)
                _log_event(
                    f'📉 {market.replace("KRW-","")} 등급하락 [{grade}] Watch 제거',
                    'system')
                continue

            # ── 현재가 갱신 (entry_price 유지) ──────────────
            entry_price   = item.get('entry_price', result['current_price'])
            current_price = result['current_price']
            price_change  = _calc_price_change_from_entry(entry_price, current_price)

            item.update({
                'current_price':      current_price,
                'price_change':       price_change,   # 등록가 대비
                'grade':              grade,
                'score':              summary['score'],
                'timing_warning':     summary.get('timing_warning',    False),
                'overbought_warning': summary.get('overbought_warning', False),
                'h4_k':               summary.get('h4_k', 50.0),
                'h1_k':               summary.get('h1_k', 50.0),
                'h4_gc':              summary.get('h4_gc',    False),
                'h1_gc':              summary.get('h1_gc',    False),
                # entry_price 는 업데이트하지 않음 ✅
            })

            # 자동 진입 체크
            if (summary.get('auto_entry') and
                market not in active_tickers):
                to_activate.append((item, current_price))
            else:
                updated_watch.append(item)

        # 제거 목록 반영
        updated_watch = [w for w in updated_watch
                         if w['market'] not in to_remove]

        # 자동 진입
        for watch_item, price in to_activate:
            market      = watch_item['market']
            active_item = _make_active_item(watch_item, price)
            active.append(active_item)
            symbol = market.replace('KRW-', '')
            msg    = (f'✅ {symbol} 자동 진입 @ {price:,.0f}원 '
                      f'[{watch_item["grade"]}]')
            send_telegram(msg)
            _log_event(msg, 'active')
            log.info(msg)

        _save_json(WATCH_FILE,  updated_watch)
        _save_json(ACTIVE_FILE, active)
        log.info(
            f'Watch 재스캔 완료 – {len(updated_watch)}개 유지, '
            f'{len(to_activate)}개 진입, {len(to_remove)}개 제거'
        )

    except Exception as e:
        log.error(f'Watch 재스캔 오류: {e}')
    finally:
        with _state_lock:
            _scanner_state['watch_rescanning'] = False


# ── 가격 체크 ─────────────────────────────────────────
def _run_price_check():
    with _state_lock:
        _scanner_state['price_checking'] = True
    try:
        active  = _load_json(ACTIVE_FILE,  [])
        history = _load_json(HISTORY_FILE, [])
        if not active:
            return

        to_close = []
        for item in active:
            market = item['market']
            price  = get_current_price(market)
            if not price:
                continue
            item['current_price'] = price
            pnl = (price - item['entry_price']) / item['entry_price'] * 100

            if price >= item['tp_price']:
                to_close.append((item, price, 'TP', pnl))
            elif price <= item['sl_price']:
                to_close.append((item, price, 'SL', pnl))

        closed_markets = set()
        for item, price, reason, pnl in to_close:
            market = item['market']
            symbol = market.replace('KRW-', '')
            closed = {**item,
                      'close_price': price,
                      'pnl':         round(pnl, 2),
                      'reason':      reason,
                      'closed_at':   datetime.now().strftime('%Y-%m-%d %H:%M')}
            history.append(closed)
            closed_markets.add(market)
            emoji = '🟢' if reason == 'TP' else '🔴'
            msg   = f'{emoji} {symbol} {reason} @ {price:,.0f}원 ({pnl:+.2f}%)'
            send_telegram(msg)
            _log_event(msg, 'close')
            log.info(msg)

        active = [a for a in active if a['market'] not in closed_markets]
        _save_json(ACTIVE_FILE,  active)
        _save_json(HISTORY_FILE, history)

        # 통계 업데이트
        wins      = [h for h in history if h.get('pnl', 0) > 0]
        total_pnl = sum(h.get('pnl', 0) for h in history)
        with _state_lock:
            _scanner_state['total_trades'] = len(history)
            _scanner_state['win_trades']   = len(wins)
            _scanner_state['total_pnl']    = round(total_pnl, 2)

    except Exception as e:
        log.error(f'가격 체크 오류: {e}')
    finally:
        with _state_lock:
            _scanner_state['price_checking'] = False


# ── DEEP 스캔 ─────────────────────────────────────────
def run_deep_scan():
    with _state_lock:
        _scanner_state['deep_scanning'] = True
    log.info('🔥 DEEP 스캔 시작')
    _log_event('🔥 BTC 급락 감지 → DEEP 스캔 시작', 'deep')
    try:
        tickers    = get_krw_tickers()
        btc_closes = get_candles('KRW-BTC', 'days', count=20)
        deep_list  = []

        def _deep_process(ticker):
            try:
                time.sleep(REQUEST_DELAY)
                closes = get_candles(ticker, 'days', count=20)
                if len(closes) < 15:
                    return
                rs = calc_relative_strength(closes, btc_closes)
                if rs['grade'] in ('S', 'A'):
                    price = get_current_price(ticker) or 0
                    deep_list.append({
                        'market':     ticker,
                        'rs':         rs['rs'],
                        'rs_grade':   rs['grade'],
                        'price':      price,
                        'scanned_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    })
            except:
                pass

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            list(ex.map(_deep_process, tickers))

        deep_list.sort(key=lambda x: x['rs'], reverse=True)
        _save_json(DEEP_FILE, deep_list[:30])
        msg = f'🔥 DEEP 스캔 완료 – 강도 상위 {len(deep_list[:30])}개'
        log.info(msg)
        _log_event(msg, 'deep')

    except Exception as e:
        log.error(f'DEEP 스캔 오류: {e}')
    finally:
        with _state_lock:
            _scanner_state['deep_scanning'] = False


# ── 루프 함수 ─────────────────────────────────────────
def scanner_loop():
    log.info(f'🚀 scanner_loop 시작 (주기: {SCAN_INTERVAL_MIN}분)')
    while True:
        try:
            _run_full_scan()
        except Exception as e:
            log.error(f'scanner_loop 오류: {e}')
        time.sleep(SCAN_INTERVAL_MIN * 60)

def watch_rescan_loop():
    log.info(f'🔄 watch_rescan_loop 시작 (주기: {RESCAN_INTERVAL_MIN}분)')
    time.sleep(90)
    while True:
        try:
            _run_watch_rescan()
        except Exception as e:
            log.error(f'watch_rescan_loop 오류: {e}')
        time.sleep(RESCAN_INTERVAL_MIN * 60)

def price_check_loop():
    log.info(f'💰 price_check_loop 시작 (주기: {PRICE_CHECK_SEC}초)')
    time.sleep(30)
    while True:
        try:
            _run_price_check()
        except Exception as e:
            log.error(f'price_check_loop 오류: {e}')
        time.sleep(PRICE_CHECK_SEC)

def active_monitor_loop():
    log.info('📊 active_monitor_loop 시작')
    time.sleep(60)
    while True:
        try:
            active = _load_json(ACTIVE_FILE, [])
            if active:
                _run_price_check()
        except Exception as e:
            log.error(f'active_monitor_loop 오류: {e}')
        time.sleep(PRICE_CHECK_SEC * 2)

def deep_scan_loop():
    log.info(f'🔥 deep_scan_loop 시작 (주기: {DEEP_CHECK_SEC}초)')
    time.sleep(120)
    while True:
        try:
            with _state_lock:
                btc_chg = _scanner_state.get('btc_change_1h', 0) or 0
            if btc_chg <= BTC_DROP_THRESHOLD:
                run_deep_scan()
        except Exception as e:
            log.error(f'deep_scan_loop 오류: {e}')
        time.sleep(DEEP_CHECK_SEC)

def daily_summary_loop():
    log.info('📅 daily_summary_loop 시작')
    while True:
        try:
            now = datetime.now()
            if now.hour == 9 and now.minute < 5:
                history = _load_json(HISTORY_FILE, [])
                today   = now.strftime('%Y-%m-%d')
                today_h = [h for h in history
                           if h.get('closed_at', '').startswith(today)]
                if today_h:
                    wins = [h for h in today_h if h.get('pnl', 0) > 0]
                    pnl  = sum(h.get('pnl', 0) for h in today_h)
                    msg  = (f'📊 일일 요약 {today}\n'
                            f'종료: {len(today_h)}건 | 승: {len(wins)}건\n'
                            f'총 PnL: {pnl:+.2f}%')
                    send_telegram(msg)
                    _log_event(msg, 'system')
        except Exception as e:
            log.error(f'daily_summary_loop 오류: {e}')
        time.sleep(300)


# ── 상태 조회 ─────────────────────────────────────────
def get_scanner_state() -> dict:
    with _state_lock:
        return dict(_scanner_state)


if __name__ == '__main__':
    print(f'scanner.py {VERSION} 로드 완료 ✅')
    print(f'  MTF Setup: {MTF_VERSION}')
    print(f'  Watch 허용 등급: B 이상 (score≥55 + aligned≥2)')
    print(f'  entry_price 보존: 재스캔 시 등록가 유지')
    print(f'  price_change: 등록가 대비 변동률')
