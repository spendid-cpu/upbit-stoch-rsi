"""
scanner.py v3.2.0 (Advanced Risk Management & Stable Version)
변경사항:
- v3.1.0: 다이버전스 탐지 추가
- v3.2.0: 🌟 트레일링 스탑 고도화 (+2% 수익 도달 시 활성화, 고점 대비 1% 하락 시 청산)
           🌟 ATR(Average True Range) 기반 코인별 가변 손절매(SL) 로직 주입
           🌟 문법 오류(SyntaxError) 및 JSON 직렬화 안정성 완벽 검증
           🌟 누락되었던 send_telegram, get_price_change_pct 함수 복구 완료
           🌟 레일웨이 환경변수(TELEGRAM_BOT_TOKEN) 매칭 완료
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

VERSION = 'v3.2.0'
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

# 🌟 레일웨이 설정에 맞춰 TELEGRAM_BOT_TOKEN 으로 수정 완료
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

SCAN_INTERVAL_MIN         = int(os.environ.get('SCAN_INTERVAL_MIN',         '60'))
WATCH_RESCAN_INTERVAL_MIN = int(os.environ.get('WATCH_RESCAN_INTERVAL_MIN', '15'))
PRICE_CHECK_INTERVAL_MIN  = int(os.environ.get('PRICE_CHECK_INTERVAL_MIN',  '1'))
ACTIVE_CHECK_INTERVAL_MIN = int(os.environ.get('ACTIVE_CHECK_INTERVAL_MIN', '1'))
DEEP_SCAN_INTERVAL_MIN    = int(os.environ.get('DEEP_SCAN_INTERVAL_MIN',    '5'))
DAILY_SUMMARY_HOUR_KST    = int(os.environ.get('DAILY_SUMMARY_HOUR_KST',   '9'))

DEEP_REBOUND_VALID_HOURS  = int(os.environ.get('DEEP_REBOUND_VALID_HOURS', '2'))

REQUEST_DELAY = float(os.environ.get('REQUEST_DELAY', '0.35'))
MAX_WORKERS   = int(os.environ.get('MAX_WORKERS',    '3'))
CANDLE_COUNT  = int(os.environ.get('CANDLE_COUNT',   '100'))

# ── [전략 고도화 파라미터 세팅] ───────────────────────────────────
TRADE_TP_PCT       = float(os.environ.get('TRADE_TP_PCT',    '5.0'))   # 고정 목표가 (최대 익절 제한)
TRADE_TIMEOUT_H    = int(os.environ.get('TRADE_TIMEOUT_H',  '48'))     # 포지션 최대 유지 시간
TRAILING_START_PCT = 2.0   # 🌟 트레일링 스탑 활성화 기준 수익률 (+2%)
TRAILING_DROP_PCT  = 1.0   # 🌟 트레일링 활성화 후 고점 대비 허용 하락폭 (1%)
ATR_MULTIPLIER     = 1.5   # 🌟 ATR 기반 손절가 설정 변동성 배수
# ───────────────────────────────────────────────────────────────────

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
    'watch_rescanning':     False,
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

def save_state():
    _save_json(STATE_FILE, _scanner_state)


def _is_expired(item: dict) -> bool:
    try:
        exp = item.get('expire_at', '')
        if not exp:
            return False
        exp_dt = datetime.fromisoformat(exp)
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=KST)
        return _now() > exp_dt
    except Exception:
        return False


def _calc_btc_entry_signal(
    d_short, d_mid, h4, h1,
    m15_gc=False, m5_gc=False
) -> str:
    bad = ('PEAK', 'FALLING')
    if d_short in bad and d_mid in bad:
        return 'BLOCK'
    if d_short in bad:
        return 'CAUTION'
    if h4 in bad or h1 in bad:
        return 'CAUTION'
    if d_short in ('BOTTOM', 'RISING') and d_mid in ('BOTTOM', 'RISING'):
        if m15_gc and m5_gc:
            return 'GOOD+'
        return 'GOOD'
    return 'CAUTION'


def _get_deep_rs_grade(ticker: str) -> str:
    try:
        deep_list = load_deep_list()
        for item in deep_list:
            if item.get('ticker') == ticker:
                return item.get('deep_grade', '-')
    except Exception:
        pass
    return '-'


_event_lock = threading.Lock()

def add_event(emoji: str, msg: str):
    event = {'time': _now_hm(), 'emoji': emoji, 'msg': msg}
    with _event_lock:
        events = load_events()
        events.append(event)
        if len(events) > 50:
            events = events[-50:]
        _save_json(EVENT_FILE, events)
    log.info(f'{emoji} {msg}')


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
    return [
        item['market'] for item in data
        if item.get('market','').startswith('KRW-')
        and item['market'].replace('KRW-','') not in STABLE_COINS
    ]


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
    data = _upbit_get('https://api.upbit.com/v1/candles/days',
                      {'market': market, 'count': 21})
    if not data or len(data) < 2:
        return 1.0
    vols    = [float(c['candle_acc_trade_volume']) for c in reversed(data)]
    avg_vol = sum(vols[:-1]) / len(vols[:-1])
    return round(vols[-1] / avg_vol, 2) if avg_vol > 0 else 1.0


def get_btc_info() -> dict:
    usdt_rate     = get_usdt_rate();   time.sleep(REQUEST_DELAY)
    closes_daily  = get_candles('KRW-BTC', 'days',        count=CANDLE_COUNT); time.sleep(REQUEST_DELAY)
    closes_weekly = get_candles('KRW-BTC', 'weeks',       count=25);           time.sleep(REQUEST_DELAY)
    closes_h4     = get_candles('KRW-BTC', 'minutes/240', count=CANDLE_COUNT); time.sleep(REQUEST_DELAY)
    closes_h1     = get_candles('KRW-BTC', 'minutes/60',  count=CANDLE_COUNT)

    ma20_info = _mtf.btc_ma20_signal(closes_daily, closes_weekly)
    price     = ma20_info.get('price')

    pct_4h  = round((closes_h4[-1]-closes_h4[-2])/closes_h4[-2]*100,2) if len(closes_h4)>=2 else None
    pct_1h  = round((closes_h1[-1]-closes_h1[-2])/closes_h1[-2]*100,2) if len(closes_h1)>=2 else None
    pct_24h = round((closes_daily[-1]-closes_daily[-2])/closes_daily[-2]*100,2) if len(closes_daily)>=2 else None

    btc_mtf = _mtf.analyze_mtf({
        'daily': closes_daily,
        'h4':    closes_h4,
        'h1':    closes_h1,
    })
    d_short_cycle = btc_mtf['daily']['short'].get('cycle', 'RISING')
    d_mid_cycle   = btc_mtf['daily']['mid'].get('cycle',   'RISING')
    h4_cycle      = btc_mtf['h4']['short'].get('cycle',    'RISING')
    h1_cycle      = btc_mtf['h1']['short'].get('cycle',    'RISING')

    with _state_lock:
        m15_gc = _scanner_state.get('btc_m15_gc', False)
        m5_gc  = _scanner_state.get('btc_m5_gc',  False)

    entry_signal = _calc_btc_entry_signal(
        d_short_cycle, d_mid_cycle, h4_cycle, h1_cycle,
        m15_gc=m15_gc, m5_gc=m5_gc
    )

    def to_usd(krw):
        return round(krw/usdt_rate,1) if krw and usdt_rate else None

    return {
        'price':            price,
        'price_usd':        to_usd(price),
        'usdt_rate':        usdt_rate,
        'daily_ma20':       ma20_info.get('daily_ma20'),
        'daily_ma20_usd':   to_usd(ma20_info.get('daily_ma20')),
        'daily_above':      ma20_info.get('daily_above'),
        'daily_pct':        ma20_info.get('daily_pct'),
        'weekly_ma20':      ma20_info.get('weekly_ma20'),
        'weekly_ma20_usd':  to_usd(ma20_info.get('weekly_ma20')),
        'weekly_above':     ma20_info.get('weekly_above'),
        'weekly_pct':       ma20_info.get('weekly_pct'),
        'pct_1h':           pct_1h,
        'pct_4h':           pct_4h,
        'pct_24h':          pct_24h,
        'd_short_cycle':    d_short_cycle,
        'd_mid_cycle':      d_mid_cycle,
        'h4_cycle':         h4_cycle,
        'h1_cycle':         h1_cycle,
        'entry_signal':     entry_signal,
    }

# 🌟 누락되었던 핵심 함수 2개 추가 완료!
def get_price_change_pct(market: str, unit: str, periods: int = 1):
    candles = get_candles(market, unit, count=periods+1)
    if len(candles) < 2:
        return None
    old = candles[-(periods+1)]
    new = candles[-1]
    return round((new-old)/old*100, 2) if old != 0 else None


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
    deep_tag = f' 🔥RS-{item["deep_rs_grade"]}' if item.get('deep_rs_grade') not in (None,'-') else ''
    reg_tag  = ' 📡반등감지' if item.get('reg_from') == 'deep_rebound' else ''
    div_tag  = ''
    if item.get('bull_div'):
        div_tag = ' 🔼BULL DIV'
    elif item.get('hidden_bull'):
        div_tag = ' ↗HID BULL'
    return (
        f'📋 <b>Watch 등록</b>{reg_tag}{div_tag}\n'
        f'종목: <b>{item["ticker"]}</b>{deep_tag} | 등급: <b>{item.get("grade","-")}</b>\n'
        f'등록가: {item.get("reg_price",0):,.0f} KRW\n'
        f'점수: {item.get("score",0)}점\n'
        f'⏰ {_now_hm()}'
    )


def _fmt_active_msg(item: dict, trade_type: str = 'auto') -> str:
    label = '🤖 자동' if trade_type == 'auto' else '👤 수동'
    return (
        f'✅ <b>Active 진입</b> ({label})\n'
        f'종목: <b>{item["ticker"]}</b> | 등급: <b>{item.get("grade","-")}</b>\n'
        f'진입가: {item.get("entry_price",0):,.0f} KRW\n'
        f'초기 가변SL: {item.get("sl_price",0):,.0f} | Fixed TP: {item.get("tp_price",0):,.0f}\n'
        f'⏰ {_now_hm()}'
    )


def _fmt_close_msg(item: dict, reason: str, pnl_pct: float) -> str:
    emoji = '🟢' if pnl_pct >= 0 else '🔴'
    return (
        f'{emoji} <b>포지션 종료</b>\n'
        f'종목: <b>{item.get("ticker","")}</b> | 사유: <b>{reason}</b>\n'
        f'수익률: {pnl_pct:+.2f}%\n'
        f'⏰ {_now_hm()}'
    )


def _fmt_deep_msg(items: list, btc_pct: str) -> str:
    lines = [f'🔥 <b>DEEP 상대강도 감지</b> (BTC {btc_pct}%)\n']
    for it in items[:5]:
        rs_detail = ''
        if it.get('rs_1h') is not None:
            rs_detail += f'1h:{it["rs_1h"]:+.1f}%'
        if it.get('rs_4h') is not None:
            rs_detail += f' 4h:{it["rs_4h"]:+.1f}%'
        if it.get('rs_24h') is not None:
            rs_detail += f' 24h:{it["rs_24h"]:+.1f}%'
        lines.append(
            f'  <b>{it["ticker"]}</b> [{it.get("deep_grade","-")}] '
            f'RS종합: {it.get("rs",0):+.1f}% ({rs_detail})'
        )
    return '\n'.join(lines) + f'\n⏰ {_now_hm()}'


def _fmt_rebound_msg(items: list, registered: list) -> str:
    lines = [f'⚡ <b>BTC 반등 감지!</b> (1h FALLING→BOTTOM)\n']
    lines.append(f'📋 RS 강세 {len(registered)}개 Watch 자동 등록\n')
    for it in items[:5]:
        tag = '✅' if it['ticker'] in [r['ticker'] for r in registered] else '⏭️'
        lines.append(
            f'  {tag} <b>{it["ticker"]}</b> [{it.get("deep_grade","-")}] '
            f'RS: {it.get("rs",0):+.1f}%'
        )
    return '\n'.join(lines) + f'\n⏰ {_now_hm()}'


# ══════════════════════════════════════════════════════════════════
# v3.1.0: 다이버전스 분석 헬퍼
# ══════════════════════════════════════════════════════════════════
def _analyze_divergence(daily: list, h4: list, h1: list) -> dict:
    div_daily = _mtf.detect_divergence(daily, 'long',  lookback=60)
    div_h4    = _mtf.detect_divergence(h4,    'short', lookback=60)
    div_h1    = _mtf.detect_divergence(h1,    'short', lookback=40)

    div_bonus = 0

    if div_daily.get('bull_div'):
        div_bonus += 15 if div_daily['div_strength'] == 'STRONG' else 8
    if div_h4.get('bull_div'):
        div_bonus += 10 if div_h4['div_strength'] == 'STRONG' else 5
    if div_h1.get('bull_div'):
        div_bonus += 5
        
    if div_daily.get('hidden_bull') or div_h4.get('hidden_bull'):
        div_bonus += 5
    if div_h1.get('hidden_bull'):
        div_bonus += 3

    if div_daily.get('bear_div'):
        div_bonus -= 15 if div_daily['div_strength'] == 'STRONG' else 8
    if div_h4.get('bear_div'):
        div_bonus -= 10 if div_h4['div_strength'] == 'STRONG' else 5
    if div_h1.get('bear_div'):
        div_bonus -= 5

    if div_daily.get('bull_div') and div_h4.get('bull_div'):
        div_bonus += 10
    if div_daily.get('bear_div') and div_h4.get('bear_div'):
        div_bonus -= 10

    if div_daily.get('bull_div'):
        div_type = 'BULL'
    elif div_h4.get('bull_div'):
        div_type = 'BULL'
    elif div_h1.get('bull_div'):
        div_type = 'BULL'
    elif div_daily.get('bear_div'):
        div_type = 'BEAR'
    elif div_h4.get('bear_div'):
        div_type = 'BEAR'
    elif div_h1.get('bear_div'):
        div_type = 'BEAR'
    elif div_daily.get('hidden_bull') or div_h4.get('hidden_bull') or div_h1.get('hidden_bull'):
        div_type = 'HIDDEN_BULL'
    elif div_daily.get('hidden_bear') or div_h4.get('hidden_bear'):
        div_type = 'HIDDEN_BEAR'
    else:
        div_type = 'NONE'

    div_strength = (
        div_daily.get('div_strength') or
        div_h4.get('div_strength')    or
        div_h1.get('div_strength')    or
        'NONE'
    )
    if div_strength == 'NONE' and div_type != 'NONE':
        div_strength = 'WEAK'

    return {
        'bull_div':       div_type in ('BULL',),
        'bear_div':       div_type in ('BEAR',),
        'hidden_bull':    div_type == 'HIDDEN_BULL',
        'hidden_bear':    div_type == 'HIDDEN_BEAR',
        'div_type':       div_type,
        'div_strength':   div_strength,
        'div_daily':      div_daily.get('div_type', 'NONE'),
        'div_h4':         div_h4.get('div_type',    'NONE'),
        'div_h1':         div_h1.get('div_type',    'NONE'),
        'div_bonus':      div_bonus,
        '_div_h4_raw':    div_h4,
    }


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

        # 🌟 고도화: 코인별 변동성 지표(Close-to-Close ATR) 1h 기준 실시간 계산
        atr_h1 = 0.0
        if len(h1) >= 15:
            deltas = np.abs(np.diff(h1))
            atr_h1 = float(np.mean(deltas[-14:]))
        else:
            atr_h1 = daily[-1] * 0.03  # 데이터 부족 시 3% 대용 활용

        mtf     = _mtf.analyze_mtf({'daily': daily, 'h4': h4, 'h1': h1})
        summary = mtf['summary']

        ticker        = market.replace('KRW-', '')
        deep_rs_grade = _get_deep_rs_grade(ticker)
        base_score    = summary.get('score', 0)

        deep_bonus  = {'S': 15, 'A': 10, 'B': 5}
        bonus       = deep_bonus.get(deep_rs_grade, 0)

        div_info  = _analyze_divergence(daily, h4, h1)
        div_bonus = div_info['div_bonus']

        final_score = min(base_score + bonus + div_bonus, 100)
        final_score = max(final_score, 0)

        grade = summary.get('grade', '-')
        if (bonus > 0 or div_bonus > 0) and not summary.get('any_buy_no'):
            if final_score >= 80:   grade = 'S'
            elif final_score >= 65: grade = 'A'
            elif final_score >= 45: grade = 'B'

        vol_ratio   = get_volume_ratio(market)
        bottom_days = _count_bottom_days(get_candles(market,'days',count=30), 'long')

        return {
            'market':             market,
            'ticker':             ticker,
            'price':              daily[-1] if daily else None,
            'daily_long_k':       mtf['daily']['long'].get('k'),
            'daily_long_d':       mtf['daily']['long'].get('d'),
            'daily_mid_k':        mtf['daily']['mid'].get('k'),
            'daily_mid_d':        mtf['daily']['mid'].get('d'),
            'daily_short_k':      mtf['daily']['short'].get('k'),
            'daily_short_d':      mtf['daily']['short'].get('d'),
            'daily_long_signal':  mtf['daily']['long'].get('signal'),
            'daily_short_signal': mtf['daily']['short'].get('signal'),
            'h4_short_k':         mtf['h4']['short'].get('k'),
            'h4_short_d':         mtf['h4']['short'].get('d'),
            'h4_short_signal':    mtf['h4']['short'].get('signal'),
            'h4_gc':              summary.get('h4_gc', False),
            'h1_short_k':         mtf['h1']['short'].get('k'),
            'h1_short_d':         mtf['h1']['short'].get('d'),
            'h1_short_signal':    mtf['h1']['short'].get('signal'),
            'h1_gc':              summary.get('h1_gc', False),
            'grade':              grade,
            'score':              final_score,
            'watch_eligible':     summary.get('watch_eligible', False),
            'auto_entry':         summary.get('auto_entry', False),
            'any_buy_no':         summary.get('any_buy_no', False),
            'vol_ratio':          vol_ratio,
            'bottom_days':        bottom_days,
            'd_short_cycle':      mtf['daily']['short'].get('cycle', 'RISING'),
            'd_mid_cycle':        mtf['daily']['mid'].get('cycle',   'RISING'),
            'h4_cycle':           mtf['h4']['short'].get('cycle',    'RISING'),
            'h1_cycle':           mtf['h1']['short'].get('cycle',    'RISING'),
            'deep_rs_grade':      deep_rs_grade,
            'bull_div':           div_info['bull_div'],
            'bear_div':           div_info['bear_div'],
            'hidden_bull':        div_info['hidden_bull'],
            'hidden_bear':        div_info['hidden_bear'],
            'div_type':           div_info['div_type'],
            'div_strength':       div_info['div_strength'],
            'div_daily':          div_info['div_daily'],
            'div_h4':             div_info['div_h4'],
            'div_h1':             div_info['div_h1'],
            'div_bonus':          div_bonus,
            'atr_h1':             atr_h1,
            'analyzed_at':        _now_iso(),
        }
    except Exception as e:
        log.warning(f'분석 실패 {market}: {e}')
        return None


def _count_bottom_days(closes: list, term: str) -> int:
    if len(closes) < 30:
        return 0
    count = 0
    for i in range(len(closes)-1, max(len(closes)-15,-1), -1):
        r = _mtf.calc_stoch_rsi(closes[:i+1], term)
        if r.get('k') is not None and r['k'] <= 20:
            count += 1
        else:
            break
    return count


def _make_watch_item(res: dict, reg_from: str = 'scan') -> dict:
    now       = _now_iso()
    expire_at = (_now() + timedelta(
        days=WATCH_EXPIRE_DAYS.get(res.get('grade','-'), 3)
    )).strftime('%Y-%m-%dT%H:%M:%S')
    return {
        'ticker':        res['ticker'],
        'market':        res['market'],
        'grade':         res.get('grade', '-'),
        'score':         res.get('score', 0),
        'reg_price':     res.get('price'),
        'current_price': res.get('price'),
        'price_change':  0.0,
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
        'd_short_cycle': res.get('d_short_cycle', 'RISING'),
        'd_mid_cycle':   res.get('d_mid_cycle',   'RISING'),
        'h4_cycle':      res.get('h4_cycle',       'RISING'),
        'h1_cycle':      res.get('h1_cycle',       'RISING'),
        'deep_rs_grade': res.get('deep_rs_grade', '-'),
        'bull_div':      res.get('bull_div',    False),
        'bear_div':      res.get('bear_div',    False),
        'hidden_bull':   res.get('hidden_bull', False),
        'div_type':      res.get('div_type',    'NONE'),
        'div_strength':  res.get('div_strength','NONE'),
        'div_daily':     res.get('div_daily',   'NONE'),
        'div_h4':        res.get('div_h4',      'NONE'),
        'div_h1':        res.get('div_h1',      'NONE'),
        'atr_h1':        res.get('atr_h1', 0.0),
        'added_at':      now,
        'expire_at':     expire_at,
        'score_history': [res.get('score', 0)],
        'rescan_count':  0,
        'reg_from':      reg_from,
    }


def _make_active_item(watch_item: dict, price: float, trade_type: str = 'auto') -> dict:
    now    = _now_iso()
    tp     = round(price * (1 + TRADE_TP_PCT / 100), 2)
    
    atr = watch_item.get('atr_h1', 0.0)
    if atr > 0:
        sl_buffer = atr * ATR_MULTIPLIER
        sl = round(max(price - sl_buffer, price * 0.95), 2)
    else:
        sl = round(price * 0.97, 2)
        
    expire = (_now() + timedelta(hours=TRADE_TIMEOUT_H)).strftime('%Y-%m-%dT%H:%M:%S')
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
        'trailing_activated': False,
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
        'd_short_cycle': watch_item.get('d_short_cycle', 'RISING'),
        'd_mid_cycle':   watch_item.get('d_mid_cycle',   'RISING'),
        'h4_cycle':      watch_item.get('h4_cycle',       'RISING'),
        'h1_cycle':      watch_item.get('h1_cycle',       'RISING'),
        'deep_rs_grade': watch_item.get('deep_rs_grade', '-'),
        'bull_div':      watch_item.get('bull_div',    False),
        'bear_div':      watch_item.get('bear_div',    False),
        'hidden_bull':   watch_item.get('hidden_bull', False),
        'div_type':      watch_item.get('div_type',    'NONE'),
        'div_strength':  watch_item.get('div_strength','NONE'),
        'reg_from':      watch_item.get('reg_from', 'scan'),
    }


def close_active_item(item: dict, reason: str, close_price: float) -> dict:
    entry   = item.get('entry_price', close_price)
    pnl_pct = round((close_price-entry)/entry*100, 2) if entry > 0 else 0.0
    closed  = {**item}
    closed.update({
        'close_price':  close_price,
        'close_at':     _now_iso(),
        'close_reason': reason,
        'pnl_pct':      pnl_pct,
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
        _scanner_state['total_pnl'] = round(_scanner_state['total_pnl'] + pnl_pct, 2)
    return closed


def _is_deep_data_valid(item: dict) -> bool:
    try:
        scanned_at = item.get('scanned_at', '')
        if not scanned_at:
            return False
        dt = datetime.fromisoformat(scanned_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return (_now() - dt).total_seconds() < DEEP_REBOUND_VALID_HOURS * 3600
    except Exception:
        return False


def _auto_watch_from_deep(strong_items: list) -> list:
    watches        = load_watch_list()
    actives        = load_active_list()
    watch_tickers  = {w['ticker'] for w in watches}
    active_tickers = {a['ticker'] for a in actives}
    registered     = []

    for deep_item in strong_items:
        ticker = deep_item.get('ticker', '')
        market = f'KRW-{ticker}'

        if ticker in watch_tickers or ticker in active_tickers:
            log.info(f'  ⏭️ {ticker} 이미 Watch/Active에 존재 - 스킵')
            continue

        try:
            res = analyze_ticker(market)
            if not res:
                log.warning(f'  ⚠️ {ticker} 분석 실패 - 스킵')
                continue
            if res.get('grade') not in ALLOWED_GRADES:
                log.info(f'  ⏭️ {ticker} 등급 {res.get("grade")} - 스킵')
                continue
            if res.get('any_buy_no'):
                log.info(f'  ⏭️ {ticker} BUY_NO - 스킵')
                continue

            item = _make_watch_item(res, reg_from='deep_rebound')
            watches.append(item)
            watch_tickers.add(ticker)
            registered.append(item)

            add_event('⚡', f'{ticker} 반등감지 Watch 등록 [{res["grade"]}] RS-{deep_item.get("deep_grade","-")}')
            send_telegram(_fmt_watch_msg(item))
            log.info(f'  ⚡ {ticker} 반등감지 Watch 등록 [{res["grade"]}] {res["score"]}점')

        except Exception as e:
            log.warning(f'  _auto_watch_from_deep {ticker}: {e}')
            continue

    if registered:
        save_watch_list(watches)
        with _state_lock:
            _scanner_state['watch_count'] = len(watches)

    return registered


def btc_fast_loop():
    log.info('⚡ btc_fast_loop 시작 (10초 주기, 5m/15m + 반등감지)')
    _prev_h1_cycle = None

    while True:
        try:
            closes_m15 = get_candles('KRW-BTC', 'minutes/15', count=50)
            closes_m5  = get_candles('KRW-BTC', 'minutes/5',  count=50)

            m15_gc, m15_cycle = False, 'RISING'
            m5_gc,  m5_cycle  = False, 'RISING'

            if len(closes_m15) >= 30:
                r15       = _mtf.calc_stoch_rsi(closes_m15, 'short')
                m15_cycle = r15.get('cycle', 'RISING')
                k15, d15  = r15.get('k'), r15.get('d')
                if k15 is not None and d15 is not None:
                    m15_gc = bool((k15 > d15) and (k15 <= 40))

            if len(closes_m5) >= 30:
                r5       = _mtf.calc_stoch_rsi(closes_m5, 'short')
                m5_cycle = r5.get('cycle', 'RISING')
                k5, d5    = r5.get('k'), r5.get('d')
                if k5 is not None and d5 is not None:
                    m5_gc = bool((k5 > d5) and (k5 <= 40))

            with _state_lock:
                _scanner_state['btc_m15_cycle'] = m15_cycle
                _scanner_state['btc_m15_gc']    = m15_gc
                _scanner_state['btc_m5_cycle']  = m5_cycle
                _scanner_state['btc_m5_gc']     = m5_gc

                curr_h1_cycle = _scanner_state.get('btc_h1_cycle', 'RISING')

                entry_signal = _calc_btc_entry_signal(
                    _scanner_state.get('btc_d_short_cycle', 'RISING'),
                    _scanner_state.get('btc_d_mid_cycle',   'RISING'),
                    _scanner_state.get('btc_h4_cycle',      'RISING'),
                    curr_h1_cycle,
                    m15_gc=m15_gc,
                    m5_gc=m5_gc,
                )
                _scanner_state['btc_entry_signal'] = entry_signal

            if (
                _prev_h1_cycle == 'FALLING' and
                curr_h1_cycle  == 'BOTTOM'
            ):
                log.info('🔄 BTC 1h FALLING→BOTTOM 전환 감지!')
                deep_list    = load_deep_list()
                valid_strong = [
                    d for d in deep_list
                    if d.get('deep_grade') in ('S', 'A')
                    and _is_deep_data_valid(d)
                ]
                if valid_strong:
                    log.info(f'  RS 강세 유효 종목: {len(valid_strong)}개 → Watch 자동 등록 시작')
                    registered = _auto_watch_from_deep(valid_strong)
                    if registered:
                        add_event('⚡', f'BTC 반등! {len(registered)}개 Watch 자동 등록')
                        send_telegram(_fmt_rebound_msg(valid_strong, registered))
                        with _state_lock:
                            _scanner_state['btc_rebound_detected'] = True
                            _scanner_state['btc_rebound_at']       = _now_iso()
                    else:
                        add_event('⚡', 'BTC 반등 감지 (등록 가능 종목 없음)')
                else:
                    log.info('  유효한 DEEP RS 데이터 없음 (만료 또는 미스캔)')
                    add_event('⚡', 'BTC 반등 감지 (DEEP RS 데이터 없음)')

            _prev_h1_cycle = curr_h1_cycle

        except Exception as e:
            log.error(f'btc_fast_loop: {e}')

        time.sleep(10)


def run_deep_scan(btc_info: dict):
    with _state_lock:
        _scanner_state['deep_scanning'] = True
    add_event('🔥', 'DEEP 스캔 시작')
    try:
        tickers     = get_krw_tickers()
        btc_pct_1h  = btc_info.get('pct_1h')  or 0
        btc_pct_4h  = btc_info.get('pct_4h')  or 0
        btc_pct_24h = btc_info.get('pct_24h') or 0
        results     = []

        def _check(market):
            try:
                time.sleep(REQUEST_DELAY)
                coin_pct_1h = get_price_change_pct(market, 'minutes/60',  1)
                if coin_pct_1h is None:
                    return None
                time.sleep(REQUEST_DELAY)
                coin_pct_4h  = get_price_change_pct(market, 'minutes/240', 1)
                time.sleep(REQUEST_DELAY)
                coin_pct_24h = get_price_change_pct(market, 'days', 1)

                rs_info = _mtf.calc_relative_strength(
                    coin_pct_1h=coin_pct_1h, btc_pct_1h=btc_pct_1h,
                    coin_pct_4h=coin_pct_4h, btc_pct_4h=btc_pct_4h,
                    coin_pct_24h=coin_pct_24h, btc_pct_24h=btc_pct_24h,
                )
                if rs_info['grade'] in ('-', 'C'):
                    return None

                time.sleep(REQUEST_DELAY)
                daily = get_candles(market, 'days', count=30)
                if daily and _mtf.calc_stoch_rsi(daily, 'long').get('k', 50) >= 70:
                    return None

                time.sleep(REQUEST_DELAY)
                vol_ratio = get_volume_ratio(market)

                return {
                    'ticker':       market.replace('KRW-',''),
                    'market':       market,
                    'coin_pct_1h':  coin_pct_1h,
                    'coin_pct_4h':  coin_pct_4h,
                    'coin_pct_24h': coin_pct_24h,
                    'btc_pct_1h':   btc_info.get('pct_1h',0),
                    'btc_pct_4h':   btc_info.get('pct_4h',0),
                    'btc_pct_24h':  btc_info.get('pct_24h',0),
                    'rs':           rs_info['rs'],
                    'rs_1h':        rs_info['rs_1h'],
                    'rs_4h':        rs_info['rs_4h'],
                    'rs_24h':       rs_info['rs_24h'],
                    'deep_grade':   rs_info['grade'],
                    'signal':       rs_info['signal'],
                    'vol_ratio':    vol_ratio,
                    'scanned_at':   _now_iso(),
                }
            except Exception as e:
                log.warning(f'DEEP _check {market}: {e}')
                return None

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for res in as_completed({ex.submit(_check, m): m for m in tickers}):
                r = res.result()
                if r:
                    results.append(r)

        results.sort(key=lambda x: x['rs'], reverse=True)
        top = [r for r in results if r['deep_grade'] in ('S','A','B')]
        save_deep_list(top)

        with _state_lock:
            _scanner_state['last_deep_scan']        = _now_iso()
            _scanner_state['deep_count']            = len(top)
            _scanner_state['deep_scanning']         = False
            _scanner_state['btc_rebound_detected']  = False

        add_event('🔥', f'DEEP 스캔 완료 {len(top)}개 감지')
        alert = [r for r in top if r['deep_grade'] in ('S','A')]
        if alert:
            send_telegram(_fmt_deep_msg(alert, f'{btc_info.get("pct_1h",0):+.1f}'))

    except Exception as e:
        log.error(f'DEEP 스캔 오류: {e}')
        with _state_lock:
            _scanner_state['deep_scanning'] = False


def manual_add_watch(ticker: str) -> dict:
    ticker  = ticker.upper().replace('KRW-','')
    market  = f'KRW-{ticker}'
    watches = load_watch_list()
    if any(w['ticker'] == ticker for w in watches):
        return {'success': False, 'message': f'{ticker} 이미 Watch에 있습니다.'}
    res = analyze_ticker(market)
    if not res:
        return {'success': False, 'message': f'{ticker} 분석 실패'}
    item = _make_watch_item(res, reg_from='manual')
    watches.append(item)
    save_watch_list(watches)
    add_event('📋', f'{ticker} 수동 Watch 등록 [{item["grade"]}]')
    send_telegram(_fmt_watch_msg(item))
    return {'success': True, 'message': f'{ticker} Watch 등록 완료', 'item': item}


def manual_remove_watch(ticker: str) -> dict:
    ticker  = ticker.upper().replace('KRW-','')
    watches = load_watch_list()
    new     = [w for w in watches if w['ticker'] != ticker]
    if len(new) == len(watches):
        return {'success': False, 'message': f'{ticker} Watch에 없습니다.'}
    save_watch_list(new)
    add_event('🗑️', f'{ticker} Watch 제거')
    return {'success': True, 'message': f'{ticker} Watch 제거 완료'}


def manual_activate_watch(ticker: str) -> dict:
    ticker  = ticker.upper().replace('KRW-','')
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
    ticker  = ticker.upper().replace('KRW-','')
    actives = load_active_list()
    item    = next((a for a in actives if a['ticker'] == ticker), None)
    if not item:
        return {'success': False, 'message': f'{ticker} Active에 없습니다.'}
    price = get_current_price(f'KRW-{ticker}') or item.get('current_price', item.get('entry_price',0))
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
            _scanner_state['btc_price']           = btc.get('price')
            _scanner_state['btc_price_usd']       = btc.get('price_usd')
            _scanner_state['usdt_rate']            = btc.get('usdt_rate')
            _scanner_state['btc_daily_ma20']       = btc.get('daily_ma20')
            _scanner_state['btc_daily_ma20_usd']   = btc.get('daily_ma20_usd')
            _scanner_state['btc_daily_above']      = btc.get('daily_above')
            _scanner_state['btc_daily_pct']        = btc.get('daily_pct')
            _scanner_state['btc_weekly_ma20']      = btc.get('weekly_ma20')
            _scanner_state['btc_weekly_ma20_usd']  = btc.get('weekly_ma20_usd')
            _scanner_state['btc_weekly_above']     = btc.get('weekly_above')
            _scanner_state['btc_weekly_pct']       = btc.get('weekly_pct')
            _scanner_state['btc_1h_pct']           = btc.get('pct_1h')
            _scanner_state['btc_4h_pct']           = btc.get('pct_4h')
            _scanner_state['btc_d_short_cycle']    = btc.get('d_short_cycle')
            _scanner_state['btc_d_mid_cycle']      = btc.get('d_mid_cycle')
            _scanner_state['btc_h4_cycle']         = btc.get('h4_cycle')
            _scanner_state['btc_h1_cycle']         = btc.get('h1_cycle')
            _scanner_state['btc_entry_signal']     = btc.get('entry_signal')

        watches        = load_watch_list()
        actives        = load_active_list()
        watch_tickers  = {w['ticker'] for w in watches}
        active_tickers = {a['ticker'] for a in actives}
        new_watches    = []

        def _process(market):
            res = analyze_ticker(market)
            if not res:
                return
            ticker = res['ticker']
            if (
                res.get('watch_eligible') and
                not res.get('any_buy_no') and
                res.get('grade') in ALLOWED_GRADES and
                ticker not in watch_tickers and
                ticker not in active_tickers
            ):
                item     = _make_watch_item(res, reg_from='scan')
                new_watches.append(item)
                deep_tag = f' 🔥RS-{res["deep_rs_grade"]}' if res.get('deep_rs_grade','-') != '-' else ''
                div_tag  = f' 🔼BULL' if res.get('bull_div') else (f' ↗HID' if res.get('hidden_bull') else '')
                log.info(f'  📋 Watch: {ticker} [{res["grade"]}] {res["score"]}점{deep_tag}{div_tag}')
                add_event('📋', f'{ticker} Watch 등록 [{res["grade"]}] {res["score"]}점{deep_tag}{div_tag}')
                send_telegram(_fmt_watch_msg(item))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            list(ex.map(_process, tickers))

        if new_watches:
            watches.extend(new_watches)
            watches = [w for w in watches if not _is_expired(w)]
            save_watch_list(watches)

        now       = _now_iso()
        next_scan = (_now()+timedelta(minutes=SCAN_INTERVAL_MIN)).strftime('%Y-%m-%dT%H:%M:%S')

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

    add_event('🔍', f'Watch 재스캔 ({len(watches)}개)')
    active_tickers  = {a['ticker'] for a in actives}
    updated_watches = []
    new_actives     = []
    btc_signal      = _scanner_state.get('btc_entry_signal', 'CAUTION')

    try:
        for item in watches:
            ticker = item['ticker']
            market = item['market']

            if _is_expired(item):
                add_event('⏰', f'{ticker} Watch 만료 해제')
                continue

            if ticker in active_tickers:
                updated_watches.append(item)
                continue

            price = get_current_price(market)
            if price:
                reg = item.get('reg_price') or price
                pct = round((price-reg)/reg*100, 2)
                item['current_price'] = price
                item['price_change']  = pct

                if pct <= WATCH_DROP_PCT or pct >= WATCH_RISE_PCT:
                    add_event('📉' if pct<=0 else '📈', f'{ticker} {pct:+.1f}% 돌파 → Watch 해제')
                    continue

            res = analyze_ticker(market)
            if not res:
                updated_watches.append(item)
                continue

            if res.get('any_buy_no') or (res.get('d_short_cycle') in ('PEAK','FALLING') and res.get('d_mid_cycle') in ('PEAK','FALLING')):
                add_event('❌', f'{ticker} 조건 악화 → Watch 해제')
                continue

            if res.get('bear_div') and res.get('div_strength') == 'STRONG' and res.get('d_short_cycle') == 'PEAK':
                add_event('🔽', f'{ticker} 약세다이버전스 → Watch 해제')
                continue

            if (
                res.get('auto_entry') and
                res.get('grade') in ALLOWED_GRADES and
                btc_signal not in ('BLOCK',)
            ):
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
                        'deep_rs_grade': res.get('deep_rs_grade', '-'),
                        'bull_div':      res.get('bull_div',    False),
                        'bear_div':      res.get('bear_div',    False),
                        'hidden_bull':   res.get('hidden_bull', False),
                        'div_type':      res.get('div_type',    'NONE'),
                        'div_strength':  res.get('div_strength','NONE'),
                    }}
                    active = _make_active_item(updated_item, price, 'auto')
                    new_actives.append(active)
                    active_tickers.add(ticker)
                    add_event('✅', f'{ticker} 자동 Active [{res["grade"]}] @ {price:,.0f}')
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
                'current_price': price or item.get('current_price'),
                'price_change':  item.get('price_change', 0.0),
                'deep_rs_grade': res.get('deep_rs_grade', '-'),
                'bull_div':      res.get('bull_div',    False),
                'bear_div':      res.get('bear_div',    False),
                'hidden_bull':   res.get('hidden_bull', False),
                'div_type':      res.get('div_type',    'NONE'),
                'div_strength':  res.get('div_strength','NONE'),
                'div_daily':     res.get('div_daily',   'NONE'),
                'div_h4':        res.get('div_h4',      'NONE'),
                'div_h1':        res.get('div_h1',      'NONE'),
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

        with _state_lock:
            _scanner_state['watch_count']       = len(updated_watches)
            _scanner_state['active_count']      = len(actives)
            _scanner_state['last_watch_rescan'] = _now_iso()
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
            item['pnl_pct'] = round((price-entry)/entry*100, 2)
            
        # 🌟 고도화: 트레일링 스탑 추적을 위한 실시간 최고가 업데이트
        item['max_price'] = max(item.get('max_price', price), price)
        item['min_price'] = min(item.get('min_price', price), price)
        
        # 🌟 고도화: 수익률 2% 첫 터치 시 트레일링 가동 플래그 ON
        if item['pnl_pct'] >= TRAILING_START_PCT and not item.get('trailing_activated', False):
            item['trailing_activated'] = True
            add_event('⚡', f'{item["ticker"]} 익절권 진입 (+{item["pnl_pct"]:.1f}%) → 트레일링 감시 가동')

        # 🌟 청산 로직 1: 트레일링 스탑 발동 시 (고점 대비 1% 하락 체크)
        if item.get('trailing_activated', False):
            trail_trigger_price = item['max_price'] * (1 - TRAILING_DROP_PCT / 100)
            if price <= trail_trigger_price:
                close_active_item(item, 'Trailing Stop(익절)', price)
                continue

        # 🌟 청산 로직 2: 가변 ATR 기반 초기 손절가 터치 시
        if price <= item.get('sl_price', 0):
            close_active_item(item, 'SL(손절)', price)
            continue
            
        # 🌟 청산 로직 3: 하드캡 목표 익절가 터치 시
        if price >= item.get('tp_price', float('inf')):
            close_active_item(item, 'TP(지정가익절)', price)
            continue
            
        try:
            exp = datetime.fromisoformat(item.get('expire_at','')).replace(tzinfo=KST)
            if _now() > exp:
                close_active_item(item, '시간만료', price)
                continue
        except Exception:
            pass
        remaining.append(item)

    save_active_list(remaining)
    with _state_lock:
        _scanner_state.update({'active_count': len(remaining), 'last_price_check': _now_iso(), 'price_checking': False})


def active_price_loop():
    log.info('⚡ active_price_loop 시작 (30초 주기)')
    while True:
        try:
            actives = load_active_list()
            if actives:
                updated = []
                for item in actives:
                    price = get_current_price(item['market'])
                    if price:
                        entry = item.get('entry_price', price)
                        item['current_price'] = price
                        
                        if entry > 0:
                            item['pnl_pct'] = round((price-entry)/entry*100, 2)
                            
                        # 🌟 고도화: 빠른 루프(30초) 내에서도 최고가 추적 및 트레일링 연동
                        item['max_price'] = max(item.get('max_price', price), price)
                        item['min_price'] = min(item.get('min_price', price), price)
                        
                        if item['pnl_pct'] >= TRAILING_START_PCT and not item.get('trailing_activated', False):
                            item['trailing_activated'] = True
                            add_event('⚡', f'{item["ticker"]} 익절권 진입 (+{item["pnl_pct"]:.1f}%) → 트레일링 감시 가동(30s)')
                    updated.append(item)
                save_active_list(updated)
        except Exception as e:
            log.error(f'active_price_loop: {e}')
        time.sleep(30)


def scanner_loop():
    log.info(f'🚀 scanner_loop (주기: {SCAN_INTERVAL_MIN}분)')
    while True:
        try:
            _run_full_scan()
        except Exception as e:
            log.error(f'scanner_loop: {e}')
        time.sleep(SCAN_INTERVAL_MIN * 60)


def watch_rescan_loop():
    log.info(f'🔄 watch_rescan_loop (주기: {WATCH_RESCAN_INTERVAL_MIN}분)')
    time.sleep(90)
    while True:
        try:
            _run_watch_rescan()
        except Exception as e:
            log.error(f'watch_rescan_loop: {e}')
        time.sleep(WATCH_RESCAN_INTERVAL_MIN * 60)


def price_check_loop():
    log.info('💰 price_check_loop (주기: 1분)')
    while True:
        try:
            _run_price_check()
        except Exception as e:
            log.error(f'price_check_loop: {e}')
        time.sleep(PRICE_CHECK_INTERVAL_MIN * 60)


def active_monitor_loop():
    log.info('📊 active_monitor_loop')
    while True:
        try:
            if load_active_list():
                _run_price_check()
        except Exception as e:
            log.error(f'active_monitor_loop: {e}')
        time.sleep(ACTIVE_CHECK_INTERVAL_MIN * 60)


def deep_scan_loop():
    log.info(f'🔥 deep_scan_loop (주기: {DEEP_SCAN_INTERVAL_MIN}분)')
    while True:
        try:
            btc_info = get_btc_info()
            p1h = btc_info.get('pct_1h') or 0
            p4h = btc_info.get('pct_4h') or 0
            if p1h <= BTC_DROP_1H_PCT or p4h <= BTC_DROP_4H_PCT:
                add_event('🔥', f'BTC 급락 (1h:{p1h}% 4h:{p4h}%) → DEEP 스캔')
                run_deep_scan(btc_info)
            with _state_lock:
                _scanner_state['next_deep_scan'] = (
                    _now()+timedelta(minutes=DEEP_SCAN_INTERVAL_MIN)
                ).strftime('%Y-%m-%dT%H:%M:%S')
        except Exception as e:
            log.error(f'deep_scan_loop: {e}')
        time.sleep(DEEP_SCAN_INTERVAL_MIN * 60)


def daily_summary_loop():
    log.info('📅 daily_summary_loop')
    while True:
        try:
            now = _now()
            target = now.replace(hour=DAILY_SUMMARY_HOUR_KST, minute=0, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            time.sleep((target-now).total_seconds())
            
            watches = load_watch_list()
            actives = load_active_list()
            history = load_history()
            today   = _now().strftime('%Y-%m-%d')
            today_h = [h for h in history if h.get('close_at','').startswith(today)]
            wins    = sum(1 for h in today_h if h.get('pnl_pct',0) > 0)
            pnl_sum = sum(h.get('pnl_pct',0) for h in today_h)
            
            send_telegram(
                f'📅 <b>일일 요약</b> {today}\n'
                f'Watch: {len(watches)}개 | Active: {len(actives)}개\n'
                f'오늘 종료: {len(today_h)}건 | 승: {wins}건\n'
                f'오늘 수익: {pnl_sum:+.2f}%'
            )
            add_event('📅', f'일일 요약 | {len(today_h)}건 {pnl_sum:+.1f}%')
        except Exception as e:
            log.error(f'daily_summary_loop: {e}')


def get_scanner_state() -> dict:
    with _state_lock:
        return dict(_scanner_state)

print(f'✅ Scanner {VERSION} 로드 완료')
print(f'   MTF: {MTF_VERSION}')
print(f'   다이버전스 탐지: BULL / BEAR / HIDDEN_BULL ✅')
print(f'   하락다이버전스 + PEAK → Watch 자동 해제 ✅')
print(f'   BTC 전용 빠른 루프: 5분봉/15분봉 10초 갱신 ✅')
print(f'   ⚡ GOOD+ 신호: 1h바닥 + 15m GC + 5m GC ✅')
print(f'   🔄 BTC 반등 감지: FALLING→BOTTOM + DEEP RS 자동 Watch 등록 ✅')
print(f'   DEEP RS: 1h(50%) + 4h(30%) + 24h(20%) 가중 평균 ✅')
print(f'   Active 30초 가격 및 트레일링 스탑 업데이트 ✅')
