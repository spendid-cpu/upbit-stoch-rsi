"""
scanner.py v3.0.0
Upbit MTF 자동 스캐너
- StochRSI 단기/중기/장기 K+D 기반 Watch/Active 관리
- DEEP 상대강도 스캐너 (BTC 하락 시 5분 주기)
- 자동 Active 전환 (골든크로스 기반)
- Telegram 알림 통합
"""

import os
import json
import time
import logging
import threading
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import mtf_setup

VERSION = 'v3.0.0'

# ── 로깅 설정 ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ── 환경변수 ──────────────────────────────────────────────────
TELEGRAM_TOKEN      = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID    = os.environ.get('TELEGRAM_CHAT_ID', '')
UPBIT_ACCESS_KEY    = os.environ.get('UPBIT_ACCESS_KEY', '')
UPBIT_SECRET_KEY    = os.environ.get('UPBIT_SECRET_KEY', '')

SCAN_INTERVAL_MIN        = int(os.environ.get('SCAN_INTERVAL_MIN',        '60'))
WATCH_RESCAN_INTERVAL_MIN= int(os.environ.get('WATCH_RESCAN_INTERVAL_MIN','15'))
PRICE_CHECK_INTERVAL_MIN = int(os.environ.get('PRICE_CHECK_INTERVAL_MIN', '1'))
ACTIVE_CHECK_INTERVAL_MIN= int(os.environ.get('ACTIVE_CHECK_INTERVAL_MIN','1'))
DEEP_SCAN_INTERVAL_MIN   = int(os.environ.get('DEEP_SCAN_INTERVAL_MIN',   '5'))
DAILY_SUMMARY_HOUR_KST   = int(os.environ.get('DAILY_SUMMARY_HOUR_KST',  '9'))

REQUEST_DELAY   = float(os.environ.get('REQUEST_DELAY', '0.12'))
MAX_WORKERS     = int(os.environ.get('MAX_WORKERS', '6'))
CANDLE_COUNT    = int(os.environ.get('CANDLE_COUNT', '120'))

TRADE_TP_PCT    = float(os.environ.get('TRADE_TP_PCT', '5.0'))
TRADE_SL_PCT    = float(os.environ.get('TRADE_SL_PCT', '3.0'))
TRADE_TIMEOUT_H = int(os.environ.get('TRADE_TIMEOUT_H', '48'))

WATCH_EXPIRE_DAYS = {
    'S': 7, 'A': 7, 'B': 5, 'C': 3, 'X': 1, '-': 1
}

# BTC 하락 감지 임계값
BTC_DROP_1H_PCT  = float(os.environ.get('BTC_DROP_1H_PCT',  '-1.0'))
BTC_DROP_4H_PCT  = float(os.environ.get('BTC_DROP_4H_PCT',  '-2.0'))

# 스테이블 코인 제외
STABLE_COINS = {
    'USDT','USDC','DAI','BUSD','TUSD','USDP','USDD',
    'USD1','FDUSD','PYUSD','SUSD','GUSD',
    'STETH','WBTC','CBBTC',
}

# ── 파일 경로 ─────────────────────────────────────────────────
BASE_DIR        = os.environ.get('DATA_DIR', '/app/data')
WATCH_FILE      = os.path.join(BASE_DIR, 'watch_list.json')
ACTIVE_FILE     = os.path.join(BASE_DIR, 'active_list.json')
HISTORY_FILE    = os.path.join(BASE_DIR, 'trade_history.json')
DEEP_FILE       = os.path.join(BASE_DIR, 'deep_list.json')
STATE_FILE      = os.path.join(BASE_DIR, 'scanner_state.json')

os.makedirs(BASE_DIR, exist_ok=True)

# ── 전역 상태 ─────────────────────────────────────────────────
_state_lock  = threading.Lock()
_scanner_state = {
    'version':          VERSION,
    'mtf_version':      mtf_setup.VERSION,
    'last_scan':        None,
    'last_deep_scan':   None,
    'next_scan':        None,
    'next_deep_scan':   None,
    'scan_count':       0,
    'watch_count':      0,
    'active_count':     0,
    'deep_count':       0,
    'btc_price':        None,
    'btc_ma20':         None,
    'btc_above_ma20':   None,
    'btc_1h_pct':       None,
    'btc_4h_pct':       None,
    'total_trades':     0,
    'win_trades':       0,
    'total_pnl':        0.0,
    'running':          False,
    'error':            None,
}


# ══════════════════════════════════════════════════════════════
# JSON I/O
# ══════════════════════════════════════════════════════════════

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


def load_watch_list():
    return _load_json(WATCH_FILE, [])

def save_watch_list(data):
    _save_json(WATCH_FILE, data)

def load_active_list():
    return _load_json(ACTIVE_FILE, [])

def save_active_list(data):
    _save_json(ACTIVE_FILE, data)

def load_history():
    return _load_json(HISTORY_FILE, [])

def save_history(data):
    _save_json(HISTORY_FILE, data)

def load_deep_list():
    return _load_json(DEEP_FILE, [])

def save_deep_list(data):
    _save_json(DEEP_FILE, data)

def load_state():
    return _load_json(STATE_FILE, {})

def save_state():
    _save_json(STATE_FILE, _scanner_state)


# ══════════════════════════════════════════════════════════════
# Upbit API
# ══════════════════════════════════════════════════════════════

def _upbit_get(url: str, params: dict = None, retries: int = 3) -> dict | list | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(REQUEST_DELAY * (attempt + 1))
            else:
                log.warning(f'Upbit API 실패 {url}: {e}')
    return None


def get_krw_tickers() -> list[str]:
    """KRW 마켓 전체 티커 조회"""
    data = _upbit_get('https://api.upbit.com/v1/market/all', {'isDetails': 'false'})
    if not data:
        return []
    tickers = []
    for item in data:
        mkt = item.get('market', '')
        if not mkt.startswith('KRW-'):
            continue
        coin = mkt.replace('KRW-', '')
        if coin not in STABLE_COINS:
            tickers.append(mkt)
    return tickers


def get_candles(market: str, unit: str, count: int = CANDLE_COUNT) -> list[float]:
    """
    캔들 종가 리스트 반환
    unit: 'days' | 'minutes/240' | 'minutes/60'
    """
    if unit == 'days':
        url = 'https://api.upbit.com/v1/candles/days'
    else:
        url = f'https://api.upbit.com/v1/candles/{unit}'

    data = _upbit_get(url, {'market': market, 'count': count})
    if not data:
        return []
    return [float(c['trade_price']) for c in reversed(data)]


def get_current_price(market: str) -> float | None:
    """현재가 조회"""
    data = _upbit_get('https://api.upbit.com/v1/ticker', {'markets': market})
    if data and len(data) > 0:
        return float(data[0]['trade_price'])
    return None


def get_price_change_pct(market: str, unit: str, periods: int = 1) -> float | None:
    """
    특정 봉 기준 변화율(%) 계산
    unit: 'minutes/60'(1시간), 'minutes/240'(4시간)
    """
    candles = get_candles(market, unit, count=periods + 1)
    if len(candles) < 2:
        return None
    old = candles[-(periods + 1)]
    new = candles[-1]
    if old == 0:
        return None
    return round((new - old) / old * 100, 2)


def get_volume_ratio(market: str) -> float:
    """최근 거래량 / 20일 평균 거래량"""
    url  = 'https://api.upbit.com/v1/candles/days'
    data = _upbit_get(url, {'market': market, 'count': 21})
    if not data or len(data) < 2:
        return 1.0
    vols    = [float(c['candle_acc_trade_volume']) for c in reversed(data)]
    avg_vol = sum(vols[:-1]) / len(vols[:-1])
    if avg_vol == 0:
        return 1.0
    return round(vols[-1] / avg_vol, 2)


def get_btc_info() -> dict:
    """BTC 현재가, MA20, 변화율 조회"""
    closes_daily = get_candles('KRW-BTC', 'days', count=30)
    closes_h4    = get_candles('KRW-BTC', 'minutes/240', count=10)
    closes_h1    = get_candles('KRW-BTC', 'minutes/60',  count=5)

    ma20_info = mtf_setup.btc_ma20_signal(closes_daily)

    pct_4h = None
    pct_1h = None

    if len(closes_h4) >= 2:
        pct_4h = round((closes_h4[-1] - closes_h4[-2]) / closes_h4[-2] * 100, 2)
    if len(closes_h1) >= 2:
        pct_1h = round((closes_h1[-1] - closes_h1[-2]) / closes_h1[-2] * 100, 2)

    return {
        'price':      ma20_info.get('price'),
        'ma20':       ma20_info.get('ma20'),
        'above_ma20': ma20_info.get('above'),
        'ma20_pct':   ma20_info.get('pct'),
        'pct_1h':     pct_1h,
        'pct_4h':     pct_4h,
    }


# ══════════════════════════════════════════════════════════════
# Telegram
# ══════════════════════════════════════════════════════════════

def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
        requests.post(url, json={
            'chat_id':    TELEGRAM_CHAT_ID,
            'text':       msg,
            'parse_mode': 'HTML',
        }, timeout=10)
    except Exception as e:
        log.warning(f'텔레그램 전송 실패: {e}')


def _fmt_watch_msg(item: dict) -> str:
    g = item.get('grade', '-')
    ticker = item.get('ticker', '')
    price  = item.get('reg_price', 0)
    score  = item.get('score', 0)
    d_long_k = item.get('daily_long_k', '-')
    return (
        f'📋 <b>Watch 등록</b>\n'
        f'종목: <b>{ticker}</b> | 등급: <b>{g}</b>\n'
        f'등록가: {price:,.0f} KRW\n'
        f'점수: {score}점 | 일봉장기K: {d_long_k}\n'
        f'⏰ {datetime.now().strftime("%m/%d %H:%M")}'
    )


def _fmt_active_msg(item: dict, trade_type: str = 'auto') -> str:
    label = '🤖 자동' if trade_type == 'auto' else '👤 수동'
    ticker = item.get('ticker', '')
    entry  = item.get('entry_price', 0)
    tp     = item.get('tp_price', 0)
    sl     = item.get('sl_price', 0)
    grade  = item.get('grade', '-')
    return (
        f'✅ <b>Active 진입</b> ({label})\n'
        f'종목: <b>{ticker}</b> | 등급: <b>{grade}</b>\n'
        f'진입가: {entry:,.0f} KRW\n'
        f'TP: {tp:,.0f} (+{TRADE_TP_PCT}%) | SL: {sl:,.0f} (-{TRADE_SL_PCT}%)\n'
        f'⏰ {datetime.now().strftime("%m/%d %H:%M")}'
    )


def _fmt_close_msg(item: dict, reason: str, pnl_pct: float) -> str:
    emoji = '🟢' if pnl_pct >= 0 else '🔴'
    return (
        f'{emoji} <b>종료</b> [{reason}]\n'
        f'종목: <b>{item.get("ticker","")}</b>\n'
        f'수익: <b>{pnl_pct:+.2f}%</b>\n'
        f'⏰ {datetime.now().strftime("%m/%d %H:%M")}'
    )


def _fmt_deep_msg(items: list, btc_pct: str) -> str:
    lines = [f'🔥 <b>DEEP 상대강도 감지</b> (BTC {btc_pct}%)\n']
    for it in items[:5]:
        rs    = it.get('rs', 0)
        grade = it.get('deep_grade', '-')
        lines.append(
            f'  <b>{it["ticker"]}</b> [{grade}] RS: +{rs}% | 변화: {it.get("coin_pct","?")}%'
        )
    return '\n'.join(lines) + f'\n⏰ {datetime.now().strftime("%m/%d %H:%M")}'


# ══════════════════════════════════════════════════════════════
# 코인 분석
# ══════════════════════════════════════════════════════════════

def analyze_ticker(market: str) -> dict | None:
    """
    단일 티커 MTF 분석
    Returns None if insufficient data
    """
    try:
        time.sleep(REQUEST_DELAY)
        daily = get_candles(market, 'days',         count=CANDLE_COUNT)
        time.sleep(REQUEST_DELAY)
        h4    = get_candles(market, 'minutes/240',  count=CANDLE_COUNT)
        time.sleep(REQUEST_DELAY)
        h1    = get_candles(market, 'minutes/60',   count=CANDLE_COUNT)

        if len(daily) < 60 or len(h4) < 60 or len(h1) < 60:
            return None

        mtf = mtf_setup.analyze_mtf({'daily': daily, 'h4': h4, 'h1': h1})
        summary = mtf['summary']

        price = daily[-1] if daily else None
        vol_ratio = get_volume_ratio(market)
        time.sleep(REQUEST_DELAY)

        # 바닥 지속 일수 (일봉 장기 K ≤ 20 연속 일수)
        bottom_days = _count_bottom_days(
            get_candles(market, 'days', count=30),
            'long'
        )

        return {
            'market':   market,
            'ticker':   market.replace('KRW-', ''),
            'price':    price,

            # 일봉
            'daily_long_k':  mtf['daily']['long'].get('k'),
            'daily_long_d':  mtf['daily']['long'].get('d'),
            'daily_mid_k':   mtf['daily']['mid'].get('k'),
            'daily_mid_d':   mtf['daily']['mid'].get('d'),
            'daily_short_k': mtf['daily']['short'].get('k'),
            'daily_short_d': mtf['daily']['short'].get('d'),
            'daily_long_signal':  mtf['daily']['long'].get('signal'),
            'daily_short_signal': mtf['daily']['short'].get('signal'),

            # 4시간
            'h4_short_k': mtf['h4']['short'].get('k'),
            'h4_short_d': mtf['h4']['short'].get('d'),
            'h4_short_signal': mtf['h4']['short'].get('signal'),
            'h4_gc': summary.get('h4_gc', False),

            # 1시간
            'h1_short_k': mtf['h1']['short'].get('k'),
            'h1_short_d': mtf['h1']['short'].get('d'),
            'h1_short_signal': mtf['h1']['short'].get('signal'),
            'h1_gc': summary.get('h1_gc', False),

            # 종합
            'grade':          summary.get('grade', '-'),
            'score':          summary.get('score', 0),
            'watch_eligible': summary.get('watch_eligible', False),
            'auto_entry':     summary.get('auto_entry', False),
            'any_buy_no':     summary.get('any_buy_no', False),

            # 보조지표
            'vol_ratio':   vol_ratio,
            'bottom_days': bottom_days,

            'analyzed_at': datetime.now().isoformat(),
        }

    except Exception as e:
        log.warning(f'분석 실패 {market}: {e}')
        return None


def _count_bottom_days(closes: list, term: str) -> int:
    """일봉 K ≤ 20 연속 일수"""
    if len(closes) < 30:
        return 0
    count = 0
    for i in range(len(closes) - 1, max(len(closes) - 15, -1), -1):
        sub = closes[:i+1]
        r = mtf_setup.calc_stoch_rsi(sub, term)
        if r.get('k') is not None and r['k'] <= 20:
            count += 1
        else:
            break
    return count


# ══════════════════════════════════════════════════════════════
# Watch 관리
# ══════════════════════════════════════════════════════════════

def _make_watch_item(res: dict) -> dict:
    now = datetime.now().isoformat()
    expire_days = WATCH_EXPIRE_DAYS.get(res.get('grade', '-'), 3)
    expire_at = (datetime.now() + timedelta(days=expire_days)).isoformat()

    return {
        'ticker':       res['ticker'],
        'market':       res['market'],
        'grade':        res.get('grade', '-'),
        'score':        res.get('score', 0),
        'reg_price':    res.get('price'),

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

        'added_at':  now,
        'expire_at': expire_at,
        'score_history': [res.get('score', 0)],
        'rescan_count':  0,
    }


def _is_watch_expired(item: dict) -> bool:
    try:
        exp = datetime.fromisoformat(item.get('expire_at', ''))
        return datetime.now() > exp
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════
# Active 관리
# ══════════════════════════════════════════════════════════════

def _make_active_item(watch_item: dict, price: float, trade_type: str = 'auto') -> dict:
    now    = datetime.now().isoformat()
    tp     = round(price * (1 + TRADE_TP_PCT / 100), 2)
    sl     = round(price * (1 - TRADE_SL_PCT / 100), 2)
    expire = (datetime.now() + timedelta(hours=TRADE_TIMEOUT_H)).isoformat()

    return {
        'ticker':      watch_item['ticker'],
        'market':      watch_item['market'],
        'grade':       watch_item.get('grade', '-'),
        'score':       watch_item.get('score', 0),
        'entry_price': price,
        'tp_price':    tp,
        'sl_price':    sl,
        'trade_type':  trade_type,
        'entry_at':    now,
        'expire_at':   expire,
        'current_price': price,
        'pnl_pct':     0.0,
        'max_price':   price,
        'min_price':   price,

        # 진입 시점 지표 스냅샷
        'daily_long_k':  watch_item.get('daily_long_k'),
        'daily_short_k': watch_item.get('daily_short_k'),
        'h4_short_k':    watch_item.get('h4_short_k'),
        'h4_gc':         watch_item.get('h4_gc', False),
        'h1_gc':         watch_item.get('h1_gc', False),
        'vol_ratio':     watch_item.get('vol_ratio', 1.0),
        'bottom_days':   watch_item.get('bottom_days', 0),
    }


def close_active_item(item: dict, reason: str, close_price: float) -> dict:
    """Active 종료 처리 → History 저장"""
    entry = item.get('entry_price', close_price)
    if entry > 0:
        pnl_pct = round((close_price - entry) / entry * 100, 2)
    else:
        pnl_pct = 0.0

    closed = {**item}
    closed.update({
        'close_price':  close_price,
        'close_reason': reason,
        'close_at':     datetime.now().isoformat(),
        'pnl_pct':      pnl_pct,
    })

    history = load_history()
    history.append(closed)
    save_history(history)

    send_telegram(_fmt_close_msg(item, reason, pnl_pct))

    with _state_lock:
        _scanner_state['total_trades'] += 1
        if pnl_pct > 0:
            _scanner_state['win_trades'] += 1
        _scanner_state['total_pnl'] = round(
            _scanner_state['total_pnl'] + pnl_pct, 2
        )

    return closed


# ══════════════════════════════════════════════════════════════
# DEEP 스캐너
# ══════════════════════════════════════════════════════════════

def run_deep_scan(btc_info: dict):
    """BTC 하락 감지 시 상대강도 스캔"""
    log.info('🔥 DEEP 스캔 시작')
    tickers = get_krw_tickers()
    deep_results = []

    btc_pct_1h = btc_info.get('pct_1h', 0) or 0
    btc_pct_4h = btc_info.get('pct_4h', 0) or 0
    btc_pct    = min(btc_pct_1h, btc_pct_4h)   # 더 큰 하락 기준

    def _check(market):
        time.sleep(REQUEST_DELAY)
        # 1시간 변화율
        coin_pct = get_price_change_pct(market, 'minutes/60', 1)
        if coin_pct is None:
            return None

        rs_info = mtf_setup.calc_relative_strength(coin_pct, btc_pct)
        if rs_info['grade'] == '-':
            return None

        # 과매수 필터 (일봉 장기 K ≥ 70 제외)
        daily = get_candles(market, 'days', count=30)
        if daily:
            r = mtf_setup.calc_stoch_rsi(daily, 'long')
            k = r.get('k')
            if k is not None and k >= 70:
                return None

        vol_ratio = get_volume_ratio(market)
        time.sleep(REQUEST_DELAY)

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
        futures = {ex.submit(_check, m): m for m in tickers}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                deep_results.append(res)

    # 상대강도 순 정렬
    deep_results.sort(key=lambda x: x['rs'], reverse=True)

    # 상위 등급만 저장
    top = [r for r in deep_results if r['deep_grade'] in ('S', 'A', 'B')]
    save_deep_list(top)

    with _state_lock:
        _scanner_state['last_deep_scan'] = datetime.now().isoformat()
        _scanner_state['deep_count'] = len(top)

    # S/A 등급 텔레그램 알림
    alert_items = [r for r in top if r['deep_grade'] in ('S', 'A')]
    if alert_items:
        send_telegram(_fmt_deep_msg(
            alert_items,
            f'{btc_pct:+.1f}'
        ))

    log.info(f'🔥 DEEP 스캔 완료: {len(top)}개 감지')
    return top


# ══════════════════════════════════════════════════════════════
# 수동 관리 API
# ══════════════════════════════════════════════════════════════

def manual_add_watch(ticker: str) -> dict:
    """수동 Watch 추가"""
    ticker  = ticker.upper().replace('KRW-', '')
    market  = f'KRW-{ticker}'
    watches = load_watch_list()

    if any(w['ticker'] == ticker for w in watches):
        return {'success': False, 'message': f'{ticker} 이미 Watch 목록에 있습니다.'}

    res = analyze_ticker(market)
    if not res:
        return {'success': False, 'message': f'{ticker} 분석 실패'}

    item = _make_watch_item(res)
    watches.append(item)
    save_watch_list(watches)
    send_telegram(_fmt_watch_msg(item))
    return {'success': True, 'message': f'{ticker} Watch 등록 완료', 'item': item}


def manual_remove_watch(ticker: str) -> dict:
    """수동 Watch 제거"""
    ticker  = ticker.upper().replace('KRW-', '')
    watches = load_watch_list()
    new     = [w for w in watches if w['ticker'] != ticker]
    if len(new) == len(watches):
        return {'success': False, 'message': f'{ticker} Watch 목록에 없습니다.'}
    save_watch_list(new)
    return {'success': True, 'message': f'{ticker} Watch 제거 완료'}


def manual_activate_watch(ticker: str) -> dict:
    """수동 Watch → Active 전환"""
    ticker  = ticker.upper().replace('KRW-', '')
    watches = load_watch_list()
    actives = load_active_list()

    if any(a['ticker'] == ticker for a in actives):
        return {'success': False, 'message': f'{ticker} 이미 Active 목록에 있습니다.'}

    watch_item = next((w for w in watches if w['ticker'] == ticker), None)
    if not watch_item:
        return {'success': False, 'message': f'{ticker} Watch 목록에 없습니다.'}

    price = get_current_price(f'KRW-{ticker}')
    if not price:
        return {'success': False, 'message': f'{ticker} 현재가 조회 실패'}

    active = _make_active_item(watch_item, price, trade_type='manual')
    actives.append(active)
    save_active_list(actives)

    watches = [w for w in watches if w['ticker'] != ticker]
    save_watch_list(watches)

    send_telegram(_fmt_active_msg(active, 'manual'))

    with _state_lock:
        _scanner_state['active_count'] = len(actives)
        _scanner_state['watch_count']  = len(watches)

    return {'success': True, 'message': f'{ticker} Active 전환 완료', 'item': active}


def manual_close_active(ticker: str, reason: str = '수동종료') -> dict:
    """수동 Active 종료"""
    ticker  = ticker.upper().replace('KRW-', '')
    actives = load_active_list()
    item    = next((a for a in actives if a['ticker'] == ticker), None)

    if not item:
        return {'success': False, 'message': f'{ticker} Active 목록에 없습니다.'}

    price = get_current_price(f'KRW-{ticker}')
    if not price:
        price = item.get('current_price', item.get('entry_price', 0))

    close_active_item(item, reason, price)
    actives = [a for a in actives if a['ticker'] != ticker]
    save_active_list(actives)

    with _state_lock:
        _scanner_state['active_count'] = len(actives)

    return {'success': True, 'message': f'{ticker} 수동 종료 완료'}


def run_single_scan() -> dict:
    """수동 즉시 스캔 트리거"""
    log.info('🔄 수동 스캔 트리거')
    try:
        _run_full_scan()
        return {'success': True, 'message': '스캔 완료'}
    except Exception as e:
        return {'success': False, 'message': str(e)}


def reset_watch_list() -> dict:
    """Watch 목록 초기화"""
    save_watch_list([])
    with _state_lock:
        _scanner_state['watch_count'] = 0
    return {'success': True, 'message': 'Watch 목록 초기화 완료'}


# ══════════════════════════════════════════════════════════════
# 메인 스캔 로직
# ══════════════════════════════════════════════════════════════

def _run_full_scan():
    """전체 시장 MTF 스캔"""
    log.info('🚀 전체 스캔 시작')
    with _state_lock:
        _scanner_state['running'] = True

    try:
        tickers = get_krw_tickers()
        log.info(f'  대상: {len(tickers)}개 종목')

        btc_info = get_btc_info()
        with _state_lock:
            _scanner_state['btc_price']      = btc_info.get('price')
            _scanner_state['btc_ma20']       = btc_info.get('ma20')
            _scanner_state['btc_above_ma20'] = btc_info.get('above_ma20')
            _scanner_state['btc_1h_pct']     = btc_info.get('pct_1h')
            _scanner_state['btc_4h_pct']     = btc_info.get('pct_4h')

        watches = load_watch_list()
        actives = load_active_list()
        watch_tickers  = {w['ticker'] for w in watches}
        active_tickers = {a['ticker'] for a in actives}

        new_watches = []

        def _process(market):
            res = analyze_ticker(market)
            if not res:
                return
            ticker = res['ticker']

            # Watch 등록 조건: 일봉 장기 과매도 + BUY_NO 아닌 경우
            if (
                res.get('watch_eligible') and
                not res.get('any_buy_no') and
                ticker not in watch_tickers and
                ticker not in active_tickers
            ):
                item = _make_watch_item(res)
                new_watches.append(item)
                log.info(f'  📋 Watch 등록: {ticker} [{res["grade"]}] {res["score"]}점')
                send_telegram(_fmt_watch_msg(item))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            list(ex.map(_process, tickers))

        if new_watches:
            watches.extend(new_watches)
            # 만료 항목 정리
            watches = [w for w in watches if not _is_watch_expired(w)]
            save_watch_list(watches)

        now = datetime.now().isoformat()
        next_scan = (datetime.now() + timedelta(minutes=SCAN_INTERVAL_MIN)).isoformat()

        with _state_lock:
            _scanner_state['last_scan']   = now
            _scanner_state['next_scan']   = next_scan
            _scanner_state['scan_count'] += 1
            _scanner_state['watch_count'] = len(watches)
            _scanner_state['running']     = False
            _scanner_state['error']       = None

        save_state()
        log.info(f'✅ 전체 스캔 완료 | Watch: {len(watches)}개')

    except Exception as e:
        log.error(f'스캔 오류: {e}')
        with _state_lock:
            _scanner_state['running'] = False
            _scanner_state['error']   = str(e)


def _run_watch_rescan():
    """Watch 목록 재스캔 및 자동 Active 전환"""
    watches = load_watch_list()
    actives = load_active_list()

    if not watches:
        return

    log.info(f'🔄 Watch 재스캔: {len(watches)}개')
    active_tickers = {a['ticker'] for a in actives}
    updated_watches = []
    new_actives     = []

    for item in watches:
        if _is_watch_expired(item):
            log.info(f'  ⏰ 만료: {item["ticker"]}')
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

        # 진입 차단 체크
        if res.get('any_buy_no'):
            log.info(f'  ❌ BUY_NO: {ticker} 제거')
            continue

        # 자동 진입 조건: 골든크로스 발생
        if res.get('auto_entry'):
            price = get_current_price(market)
            if price:
                # Watch 정보 업데이트
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
                }}
                active = _make_active_item(updated_item, price, 'auto')
                new_actives.append(active)
                active_tickers.add(ticker)
                log.info(f'  ✅ 자동 Active: {ticker} [{res["grade"]}] @ {price:,.0f}')
                send_telegram(_fmt_active_msg(active, 'auto'))
                continue

        # Watch 정보 업데이트
        item_updated = {**item}
        item_updated.update({
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
        })
        score_hist = item_updated.get('score_history', [])
        score_hist.append(res['score'])
        item_updated['score_history']  = score_hist[-10:]
        item_updated['rescan_count']   = item_updated.get('rescan_count', 0) + 1
        updated_watches.append(item_updated)

    if new_actives:
        actives.extend(new_actives)
        save_active_list(actives)

    save_watch_list(updated_watches)

    with _state_lock:
        _scanner_state['watch_count']  = len(updated_watches)
        _scanner_state['active_count'] = len(actives)

    log.info(f'✅ Watch 재스캔 완료 | Watch: {len(updated_watches)} | Active: {len(actives)}')


def _run_price_check():
    """Active 종목 가격 모니터링 (TP/SL/만료 체크)"""
    actives = load_active_list()
    if not actives:
        return

    remaining = []
    for item in actives:
        ticker = item['ticker']
        market = item['market']

        price = get_current_price(market)
        if not price:
            remaining.append(item)
            continue

        item['current_price'] = price
        entry = item.get('entry_price', price)
        if entry > 0:
            item['pnl_pct'] = round((price - entry) / entry * 100, 2)
        item['max_price'] = max(item.get('max_price', price), price)
        item['min_price'] = min(item.get('min_price', price), price)

        tp = item.get('tp_price', float('inf'))
        sl = item.get('sl_price', 0)

        if price >= tp:
            close_active_item(item, 'TP', price)
            log.info(f'  🟢 TP 도달: {ticker} @ {price:,.0f}')
            continue

        if price <= sl:
            close_active_item(item, 'SL', price)
            log.info(f'  🔴 SL 도달: {ticker} @ {price:,.0f}')
            continue

        # 만료 체크
        try:
            exp = datetime.fromisoformat(item.get('expire_at', ''))
            if datetime.now() > exp:
                close_active_item(item, '시간만료', price)
                log.info(f'  ⏰ 시간만료: {ticker}')
                continue
        except Exception:
            pass

        remaining.append(item)

    save_active_list(remaining)

    with _state_lock:
        _scanner_state['active_count'] = len(remaining)


# ══════════════════════════════════════════════════════════════
# 루프 함수들
# ══════════════════════════════════════════════════════════════

def scanner_loop():
    """메인 스캔 루프 (60분 주기)"""
    log.info(f'🚀 scanner_loop 시작 (주기: {SCAN_INTERVAL_MIN}분)')
    while True:
        try:
            _run_full_scan()
        except Exception as e:
            log.error(f'scanner_loop 오류: {e}')
        time.sleep(SCAN_INTERVAL_MIN * 60)


def watch_rescan_loop():
    """Watch 재스캔 루프 (15분 주기)"""
    log.info(f'🔄 watch_rescan_loop 시작 (주기: {WATCH_RESCAN_INTERVAL_MIN}분)')
    time.sleep(60)   # 초기 딜레이
    while True:
        try:
            _run_watch_rescan()
        except Exception as e:
            log.error(f'watch_rescan_loop 오류: {e}')
        time.sleep(WATCH_RESCAN_INTERVAL_MIN * 60)


def price_check_loop():
    """가격 모니터링 루프 (1분 주기)"""
    log.info('💰 price_check_loop 시작 (주기: 1분)')
    while True:
        try:
            _run_price_check()
        except Exception as e:
            log.error(f'price_check_loop 오류: {e}')
        time.sleep(PRICE_CHECK_INTERVAL_MIN * 60)


def active_monitor_loop():
    """Active 상세 모니터링 루프 (1분 주기)"""
    log.info('📊 active_monitor_loop 시작')
    while True:
        try:
            actives = load_active_list()
            with _state_lock:
                _scanner_state['active_count'] = len(actives)
            save_state()
        except Exception as e:
            log.error(f'active_monitor_loop 오류: {e}')
        time.sleep(ACTIVE_CHECK_INTERVAL_MIN * 60)


def deep_scan_loop():
    """DEEP 스캔 루프 (5분 주기, BTC 하락 시만 실행)"""
    log.info(f'🔥 deep_scan_loop 시작 (주기: {DEEP_SCAN_INTERVAL_MIN}분)')
    while True:
        try:
            btc_info = get_btc_info()
            pct_1h = btc_info.get('pct_1h') or 0
            pct_4h = btc_info.get('pct_4h') or 0

            if pct_1h <= BTC_DROP_1H_PCT or pct_4h <= BTC_DROP_4H_PCT:
                log.info(f'🔥 BTC 하락 감지 (1h: {pct_1h}%, 4h: {pct_4h}%) → DEEP 스캔')
                run_deep_scan(btc_info)

            with _state_lock:
                _scanner_state['next_deep_scan'] = (
                    datetime.now() + timedelta(minutes=DEEP_SCAN_INTERVAL_MIN)
                ).isoformat()

        except Exception as e:
            log.error(f'deep_scan_loop 오류: {e}')
        time.sleep(DEEP_SCAN_INTERVAL_MIN * 60)


def daily_summary_loop():
    """일일 요약 루프 (매일 오전 9시)"""
    log.info('📅 daily_summary_loop 시작')
    while True:
        try:
            now = datetime.now()
            target = now.replace(
                hour=DAILY_SUMMARY_HOUR_KST, minute=0, second=0, microsecond=0
            )
            if now >= target:
                target += timedelta(days=1)
            wait = (target - now).total_seconds()
            time.sleep(wait)

            watches = load_watch_list()
            actives = load_active_list()
            history = load_history()
            today_h = [
                h for h in history
                if h.get('close_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))
            ]
            wins    = sum(1 for h in today_h if h.get('pnl_pct', 0) > 0)
            pnl_sum = sum(h.get('pnl_pct', 0) for h in today_h)

            msg = (
                f'📅 <b>일일 요약</b> {datetime.now().strftime("%Y-%m-%d")}\n'
                f'Watch: {len(watches)}개 | Active: {len(actives)}개\n'
                f'오늘 종료: {len(today_h)}건 | 승: {wins}건\n'
                f'오늘 수익 합계: {pnl_sum:+.2f}%\n'
                f'BTC: {_scanner_state.get("btc_price","?"):,} KRW'
            )
            send_telegram(msg)

        except Exception as e:
            log.error(f'daily_summary_loop 오류: {e}')
            time.sleep(3600)
