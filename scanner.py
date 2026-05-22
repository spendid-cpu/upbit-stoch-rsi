# -*- coding: utf-8 -*-
"""
scanner.py — Upbit MTF 자동 스캐너
Version : v2.2.1
Changelog:
  v2.2.1 - daily_k 반환 버그 수정 (safe_k 방식 적용)
           신규Watch 중복 표시 버그 수정 (newly_added 방식)
           _clear_entries 타이머 중복 방지
  v2.2.0 - watch_rescan_loop / active_monitor_loop 추가 (4단계 루프)
  v2.1.0 - Watch 점수/등급 시스템 도입, 히스토리 구조 개선
  v2.0.0 - Active 상태 분리, 자동삭제 정책, 승률 통계
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import mtf_setup

# ── 로깅 ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ── 버전 ────────────────────────────────────────────────────────
VERSION = 'v2.2.1'

# ── 환경변수 ────────────────────────────────────────────────────
SCAN_INTERVAL_MIN         = int(os.getenv('SCAN_INTERVAL_MIN',         '60'))
WATCH_RESCAN_INTERVAL_MIN = int(os.getenv('WATCH_RESCAN_INTERVAL_MIN', '15'))
PRICE_CHECK_INTERVAL_MIN  = int(os.getenv('PRICE_CHECK_INTERVAL_MIN',  '5'))
ACTIVE_CHECK_INTERVAL_MIN = int(os.getenv('ACTIVE_CHECK_INTERVAL_MIN', '1'))

REQUEST_DELAY   = float(os.getenv('REQUEST_DELAY',   '0.12'))
MAX_WORKERS     = int(os.getenv('MAX_WORKERS',       '6'))
CANDLE_COUNT    = int(os.getenv('CANDLE_COUNT',      '200'))
MIN_TRADE_VALUE = float(os.getenv('MIN_TRADE_VALUE_KRW', '0'))

WATCH_LIST_FILE    = os.getenv('WATCH_LIST_FILE',    'watch_list.json')
ACTIVE_TRADES_FILE = os.getenv('ACTIVE_TRADES_FILE', 'active_trades.json')
TRADE_HISTORY_FILE = os.getenv('TRADE_HISTORY_FILE', 'trade_history.json')

TELEGRAM_TOKEN   = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID',   '')

TRADE_TP_PCT      = float(os.getenv('TRADE_TP_PCT',      '5.0'))
TRADE_SL_PCT      = float(os.getenv('TRADE_SL_PCT',      '3.0'))
TRADE_TIMEOUT_H   = float(os.getenv('TRADE_TIMEOUT_H',   '48.0'))

# ── 제외 종목 ────────────────────────────────────────────────────
STABLE_COINS = {
    'KRW-USDT','KRW-USDC','KRW-USDS','KRW-USDE',
    'KRW-TUSD','KRW-USD1','KRW-XAUT','KRW-BTC',
}

# ── 전역 상태 ────────────────────────────────────────────────────
_state_lock = threading.Lock()
_manual_scan_event = threading.Event()
_clear_timer = None   # 중복 타이머 방지용

scanner_state = {
    'version':             VERSION,
    'status':              'idle',
    'last_scan_at':        None,
    'last_watch_rescan_at': None,
    'last_price_check_at': None,
    'last_active_check_at': None,
    'next_scan_at':        None,
    'scan_count':          0,
    'watch_count':         0,
    'active_count':        0,
    'watch_list':          [],
    'active_trades':       [],
    'new_entries':         [],
    'removed_items':       [],
    'macro': {
        'btc_weekly_ma20': None,
        'btc_daily_ma20':  None,
        'btc_price':       None,
        'macro_ok':        None,
    },
    'stats':  {},
    'errors': [],
}

# ═══════════════════════════════════════════════════════════════
# JSON I/O
# ═══════════════════════════════════════════════════════════════
def _load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"JSON 로드 실패 {path}: {e}")
    return default

def _save_json(path, data):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"JSON 저장 실패 {path}: {e}")

def load_watch_list():
    raw = _load_json(WATCH_LIST_FILE, [])
    result = []
    for item in raw:
        # 구버전 마이그레이션
        if 'snapshot' not in item:
            item = {
                'ticker':     item.get('ticker', ''),
                'status':     item.get('status', 'watch'),
                'grade':      item.get('grade', 'C'),
                'snapshot': {
                    'registered_at': item.get('registered_at', datetime.now(timezone.utc).isoformat()),
                    'entry_price':   item.get('entry_price', item.get('current_price', 0)),
                    'score':         item.get('score', 0),
                    'grade':         item.get('grade', 'C'),
                    'daily_k':       item.get('daily_short_k', None),
                    'h4_k':          item.get('h4_short_k', None),
                    'h1_k':          item.get('h1_short_k', None),
                },
                'current': {
                    'price':   item.get('current_price', 0),
                    'score':   item.get('score', 0),
                    'grade':   item.get('grade', 'C'),
                    'daily_k': item.get('daily_short_k', None),
                    'h4_k':    item.get('h4_short_k', None),
                    'h1_k':    item.get('h1_short_k', None),
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                },
            }
        result.append(item)
    return result

def save_watch_list(data):
    _save_json(WATCH_LIST_FILE, data)

def load_active_trades():
    return _load_json(ACTIVE_TRADES_FILE, [])

def save_active_trades(data):
    _save_json(ACTIVE_TRADES_FILE, data)

def load_trade_history():
    return _load_json(TRADE_HISTORY_FILE, [])

def append_history(record):
    history = load_trade_history()
    history.append(record)
    if len(history) > 2000:
        history = history[-2000:]
    _save_json(TRADE_HISTORY_FILE, history)

# ═══════════════════════════════════════════════════════════════
# Upbit API
# ═══════════════════════════════════════════════════════════════
_UPBIT_BASE = 'https://api.upbit.com/v1'

def _get(url, params=None, timeout=10):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug(f"API 오류 {url}: {e}")
        return None

def get_all_krw_markets():
    data = _get(f'{_UPBIT_BASE}/market/all', {'isDetails': 'true'})
    if not data:
        return []
    result = []
    for m in data:
        code = m.get('market', '')
        if not code.startswith('KRW-'):
            continue
        if code in STABLE_COINS:
            continue
        if m.get('market_warning') == 'CAUTION':
            continue
        result.append(code)
    return result

def get_closes(ticker, unit='days', count=200):
    """캔들 종가 리스트 반환 (최신이 마지막)"""
    if unit == 'days':
        url = f'{_UPBIT_BASE}/candles/days'
        params = {'market': ticker, 'count': count}
    elif unit == 'weeks':
        url = f'{_UPBIT_BASE}/candles/weeks'
        params = {'market': ticker, 'count': count}
    elif unit == 'minutes':
        url = f'{_UPBIT_BASE}/candles/minutes/240'
        params = {'market': ticker, 'count': count}
    elif unit == 'minutes60':
        url = f'{_UPBIT_BASE}/candles/minutes/60'
        params = {'market': ticker, 'count': count}
    else:
        return []

    data = _get(url, params)
    if not data:
        return []
    closes = [c['trade_price'] for c in reversed(data)]
    return closes

def get_volumes(ticker, unit='days', count=30):
    """거래량 리스트 반환 (최신이 마지막)"""
    if unit == 'days':
        url = f'{_UPBIT_BASE}/candles/days'
        params = {'market': ticker, 'count': count}
    elif unit == 'minutes':
        url = f'{_UPBIT_BASE}/candles/minutes/240'
        params = {'market': ticker, 'count': count}
    else:
        return []

    data = _get(url, params)
    if not data:
        return []
    vols = [c.get('candle_acc_trade_volume', 0) for c in reversed(data)]
    return vols

def get_btc_closes():
    weekly = get_closes('KRW-BTC', unit='weeks', count=25)
    daily  = get_closes('KRW-BTC', unit='days',  count=25)
    return weekly, daily

def get_current_prices(tickers):
    """종목 리스트의 현재가 일괄 조회 (배치 100개)"""
    result = {}
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        data = _get(f'{_UPBIT_BASE}/ticker', {'markets': ','.join(batch)})
        if data:
            for d in data:
                result[d['market']] = d.get('trade_price', 0)
        time.sleep(REQUEST_DELAY)
    return result

# ═══════════════════════════════════════════════════════════════
# 텔레그램
# ═══════════════════════════════════════════════════════════════
def send_telegram(msg, token=None, chat_id=None):
    token   = token   or TELEGRAM_TOKEN
    chat_id = chat_id or TELEGRAM_CHAT_ID
    if not token or not chat_id:
        return
    try:
        requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            json={'chat_id': chat_id, 'text': msg, 'parse_mode': 'HTML'},
            timeout=10
        )
    except Exception as e:
        log.warning(f"텔레그램 전송 실패: {e}")

def _grade_emoji(grade):
    return {'S': '🔴', 'A': '🟠', 'B': '🟡', 'C': '⚪'}.get(grade, '⚪')

def build_active_msg(item, reason='grade_upgrade'):
    g  = item.get('grade', '?')
    tk = item.get('ticker', '')
    sc = item.get('current', {}).get('score', 0)
    ep = item.get('snapshot', {}).get('entry_price', 0)
    cp = item.get('current',  {}).get('price', 0)
    dk = item.get('current',  {}).get('daily_k', '-')
    hk = item.get('current',  {}).get('h4_k',   '-')
    lk = item.get('current',  {}).get('h1_k',   '-')

    emoji = _grade_emoji(g)
    return (
        f"{emoji} <b>[ACTIVE 전환] {tk}</b>\n"
        f"등급: {g}등급 ({sc}점)\n"
        f"등록가: {ep:,.4f} | 현재가: {cp:,.4f}\n"
        f"일봉K: {dk} | 4hK: {hk} | 1hK: {lk}\n"
        f"TP: +{TRADE_TP_PCT}% / SL: -{TRADE_SL_PCT}%"
    )

def build_close_msg(trade, result):
    tk   = trade.get('ticker', '')
    ep   = trade.get('entry_price', 0)
    cp   = trade.get('close_price', 0)
    pnl  = trade.get('pnl_pct', 0)
    held = trade.get('hours_held', 0)
    icons = {'tp': '✅', 'sl': '❌', 'timeout': '⏰', 'manual': '🖐️'}
    icon  = icons.get(result, '?')
    return (
        f"{icon} <b>[청산] {tk}</b>\n"
        f"결과: {result.upper()}\n"
        f"진입가: {ep:,.4f} → 청산가: {cp:,.4f}\n"
        f"수익률: {pnl:+.2f}%\n"
        f"보유시간: {held:.1f}h"
    )

# ═══════════════════════════════════════════════════════════════
# 분석
# ═══════════════════════════════════════════════════════════════
def analyze_ticker(ticker):
    """
    일봉 게이트 통과 시 4h/1h 스코어 계산
    반환: dict 또는 None
    """
    try:
        # 캔들 수집
        daily_closes = get_closes(ticker, unit='days',     count=CANDLE_COUNT)
        h4_closes    = get_closes(ticker, unit='minutes',  count=CANDLE_COUNT)
        h1_closes    = get_closes(ticker, unit='minutes60',count=CANDLE_COUNT)
        daily_vols   = get_volumes(ticker, unit='days',    count=30)

        if len(daily_closes) < 30:
            return None

        time.sleep(REQUEST_DELAY)

        # Stoch RSI 프리셋 계산
        daily_presets = mtf_setup.calc_all_presets(daily_closes)
        h4_presets    = mtf_setup.calc_all_presets(h4_closes) if len(h4_closes) >= 30 else {}
        h1_presets    = mtf_setup.calc_all_presets(h1_closes) if len(h1_closes) >= 30 else {}

        # 일봉 게이트
        gate = mtf_setup.evaluate_daily_gate(daily_presets)
        if not gate.get('pass'):
            return None

        # 거래량 비율
        vol_ratio = 1.0
        if len(daily_vols) >= 21:
            avg_vol = sum(daily_vols[-21:-1]) / 20
            cur_vol = daily_vols[-1]
            vol_ratio = (cur_vol / avg_vol) if avg_vol > 0 else 1.0

        # Watch 점수 계산
        score_result = mtf_setup.calc_watch_score(
            daily_presets=daily_presets,
            h4_presets=h4_presets,
            h1_presets=h1_presets,
            vol_ratio=vol_ratio,
        )

        return {
            'ticker':      ticker,
            'score':       score_result.get('score', 0),
            'grade':       score_result.get('grade', 'C'),
            'breakdown':   score_result.get('breakdown', {}),
            'daily_dir':   score_result.get('daily_dir', ''),
            'h4_dir':      score_result.get('h4_dir', ''),
            'h1_dir':      score_result.get('h1_dir', ''),
            'daily_k':     score_result.get('daily_k'),
            'h4_k':        score_result.get('h4_k'),
            'h1_k':        score_result.get('h1_k'),
            'vol_ratio':   round(vol_ratio, 2),
            'current_price': daily_closes[-1] if daily_closes else 0,
        }

    except Exception as e:
        log.debug(f"analyze_ticker 오류 {ticker}: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# 통계
# ═══════════════════════════════════════════════════════════════
def calc_stats():
    history = load_trade_history()
    if not history:
        return {
            'total': 0, 'activated': 0, 'expired': 0,
            'tp': 0, 'sl': 0, 'timeout': 0, 'manual': 0,
            'tp_rate': 0, 'sl_rate': 0, 'win_rate': 0,
            'avg_pnl': 0, 'best_pnl': 0, 'worst_pnl': 0,
            'grade_stats': {},
        }

    total     = len(history)
    activated = sum(1 for h in history if h.get('result') == 'activated')
    expired   = sum(1 for h in history if h.get('result') == 'expired')
    tp_cnt    = sum(1 for h in history if h.get('result') == 'tp')
    sl_cnt    = sum(1 for h in history if h.get('result') == 'sl')
    to_cnt    = sum(1 for h in history if h.get('result') == 'timeout')
    mn_cnt    = sum(1 for h in history if h.get('result') == 'manual')

    closed = [h for h in history if h.get('result') in ('tp','sl','timeout','manual')]
    closed_cnt = len(closed)

    pnls = [h.get('pnl_pct', 0) for h in closed if h.get('pnl_pct') is not None]

    grade_stats = {}
    for grade in ['S','A','B','C']:
        g_closed = [h for h in closed if h.get('grade') == grade]
        g_tp = sum(1 for h in g_closed if h.get('result') == 'tp')
        grade_stats[grade] = {
            'total':    len(g_closed),
            'tp':       g_tp,
            'win_rate': round(g_tp / len(g_closed) * 100, 1) if g_closed else 0,
        }

    return {
        'total':     total,
        'activated': activated,
        'expired':   expired,
        'tp':        tp_cnt,
        'sl':        sl_cnt,
        'timeout':   to_cnt,
        'manual':    mn_cnt,
        'tp_rate':   round(tp_cnt / closed_cnt * 100, 1) if closed_cnt else 0,
        'sl_rate':   round(sl_cnt / closed_cnt * 100, 1) if closed_cnt else 0,
        'win_rate':  round(tp_cnt / closed_cnt * 100, 1) if closed_cnt else 0,
        'avg_pnl':   round(sum(pnls) / len(pnls), 2) if pnls else 0,
        'best_pnl':  round(max(pnls), 2) if pnls else 0,
        'worst_pnl': round(min(pnls), 2) if pnls else 0,
        'grade_stats': grade_stats,
    }

# ═══════════════════════════════════════════════════════════════
# Active 전환
# ═══════════════════════════════════════════════════════════════
def activate_item(watch_item):
    """Watch → Active 전환"""
    now_utc = datetime.now(timezone.utc)
    ticker  = watch_item.get('ticker', '')
    snap    = watch_item.get('snapshot', {})
    cur     = watch_item.get('current',  {})

    entry_price = cur.get('price') or snap.get('entry_price', 0)

    active = {
        'ticker':       ticker,
        'grade':        cur.get('grade', watch_item.get('grade', 'C')),
        'entry_price':  entry_price,
        'entry_score':  cur.get('score', snap.get('score', 0)),
        'activated_at': now_utc.isoformat(),
        'registered_at': snap.get('registered_at', now_utc.isoformat()),
        'tp_price':     round(entry_price * (1 + TRADE_TP_PCT / 100), 8),
        'sl_price':     round(entry_price * (1 - TRADE_SL_PCT / 100), 8),
        'timeout_at':   (now_utc + timedelta(hours=TRADE_TIMEOUT_H)).isoformat(),
        'current_price': entry_price,
        'pnl_pct':      0.0,
        'snapshot':     snap,
        'current':      cur,
    }

    # 히스토리: watch → activated
    reg_at = snap.get('registered_at', now_utc.isoformat())
    try:
        watch_hours = (now_utc - datetime.fromisoformat(reg_at.replace('Z','+00:00'))).total_seconds() / 3600
    except Exception:
        watch_hours = 0

    append_history({
        'ticker':       ticker,
        'result':       'activated',
        'grade':        active['grade'],
        'registered_at': reg_at,
        'activated_at': now_utc.isoformat(),
        'watch_hours':  round(watch_hours, 2),
        'entry_price':  entry_price,
        'snapshot_score': snap.get('score', 0),
        'entry_score':  active['entry_score'],
    })

    return active

# ═══════════════════════════════════════════════════════════════
# Active TP/SL/Timeout 체크
# ═══════════════════════════════════════════════════════════════
def check_active_trades(price_map=None):
    """Active 종목 TP/SL/Timeout 체크"""
    active_trades = load_active_trades()
    if not active_trades:
        return []

    if price_map is None:
        tickers   = [t['ticker'] for t in active_trades]
        price_map = get_current_prices(tickers)

    now_utc    = datetime.now(timezone.utc)
    remaining  = []
    closed_out = []

    for trade in active_trades:
        ticker     = trade['ticker']
        cur_price  = price_map.get(ticker, trade.get('current_price', 0))
        entry      = trade['entry_price']
        tp         = trade['tp_price']
        sl         = trade['sl_price']
        timeout_at = trade.get('timeout_at')

        trade['current_price'] = cur_price
        pnl = ((cur_price - entry) / entry * 100) if entry else 0
        trade['pnl_pct'] = round(pnl, 2)

        # 청산 판정
        close_result = None
        if cur_price >= tp:
            close_result = 'tp'
        elif cur_price <= sl:
            close_result = 'sl'
        elif timeout_at:
            try:
                t_at = datetime.fromisoformat(timeout_at.replace('Z','+00:00'))
                if now_utc >= t_at:
                    close_result = 'timeout'
            except Exception:
                pass

        if close_result:
            act_at = trade.get('activated_at', trade.get('registered_at', now_utc.isoformat()))
            try:
                held = (now_utc - datetime.fromisoformat(act_at.replace('Z','+00:00'))).total_seconds() / 3600
            except Exception:
                held = 0

            trade['close_price'] = cur_price
            trade['close_result'] = close_result
            trade['closed_at']   = now_utc.isoformat()
            trade['hours_held']  = round(held, 2)
            trade['pnl_pct']     = round(pnl, 2)

            append_history({
                'ticker':      ticker,
                'result':      close_result,
                'grade':       trade.get('grade', '?'),
                'registered_at': trade.get('registered_at'),
                'activated_at':  trade.get('activated_at'),
                'closed_at':     now_utc.isoformat(),
                'entry_price':   entry,
                'close_price':   cur_price,
                'pnl_pct':       round(pnl, 2),
                'hours_held':    round(held, 2),
                'entry_score':   trade.get('entry_score', 0),
            })

            send_telegram(build_close_msg(trade, close_result))
            closed_out.append(trade)
            log.info(f"[{close_result.upper()}] {ticker} | PnL: {pnl:+.2f}%")
        else:
            remaining.append(trade)

    save_active_trades(remaining)
    return closed_out

# ═══════════════════════════════════════════════════════════════
# Watch 수동 관리
# ═══════════════════════════════════════════════════════════════
def add_manual_watch(ticker):
    ticker = ticker.upper()
    if not ticker.startswith('KRW-'):
        ticker = f'KRW-{ticker}'

    watch_list = load_watch_list()
    existing   = [w['ticker'] for w in watch_list]
    if ticker in existing:
        return {'ok': False, 'msg': f'{ticker} 이미 등록됨'}

    result = analyze_ticker(ticker)
    now_utc = datetime.now(timezone.utc)

    if result:
        item = _build_watch_item(result, now_utc, manual=True)
    else:
        # 분석 실패 시 기본값으로 등록
        price_map = get_current_prices([ticker])
        price = price_map.get(ticker, 0)
        item = {
            'ticker':  ticker,
            'status':  'watch',
            'grade':   'C',
            'manual':  True,
            'snapshot': {
                'registered_at': now_utc.isoformat(),
                'entry_price':   price,
                'score':         0,
                'grade':         'C',
                'daily_k':       None,
                'h4_k':          None,
                'h1_k':          None,
            },
            'current': {
                'price':      price,
                'score':      0,
                'grade':      'C',
                'daily_k':    None,
                'h4_k':       None,
                'h1_k':       None,
                'updated_at': now_utc.isoformat(),
            },
        }

    watch_list.append(item)
    save_watch_list(watch_list)

    with _state_lock:
        scanner_state['watch_list'] = watch_list

    return {'ok': True, 'msg': f'{ticker} 등록 완료', 'item': item}

def remove_watch(ticker):
    ticker     = ticker.upper()
    watch_list = load_watch_list()
    new_list   = [w for w in watch_list if w['ticker'] != ticker]
    removed    = [w for w in watch_list if w['ticker'] == ticker]

    if removed:
        # 수동 삭제 히스토리
        now_utc = datetime.now(timezone.utc)
        for w in removed:
            snap = w.get('snapshot', {})
            reg  = snap.get('registered_at', now_utc.isoformat())
            try:
                wh = (now_utc - datetime.fromisoformat(reg.replace('Z','+00:00'))).total_seconds() / 3600
            except Exception:
                wh = 0
            append_history({
                'ticker':       ticker,
                'result':       'manual_remove',
                'grade':        w.get('grade', 'C'),
                'registered_at': reg,
                'closed_at':    now_utc.isoformat(),
                'watch_hours':  round(wh, 2),
                'entry_price':  snap.get('entry_price', 0),
                'close_price':  w.get('current', {}).get('price', 0),
            })

    save_watch_list(new_list)
    with _state_lock:
        scanner_state['watch_list'] = new_list

    return {'ok': bool(removed), 'msg': f'{ticker} 삭제 완료' if removed else f'{ticker} 없음'}

def manual_close_trade(ticker):
    active_trades = load_active_trades()
    target = [t for t in active_trades if t['ticker'] == ticker.upper()]
    remain = [t for t in active_trades if t['ticker'] != ticker.upper()]

    if not target:
        return {'ok': False, 'msg': f'{ticker} 없음'}

    now_utc   = datetime.now(timezone.utc)
    price_map = get_current_prices([ticker.upper()])

    for trade in target:
        cur_price = price_map.get(trade['ticker'], trade.get('current_price', 0))
        entry     = trade['entry_price']
        pnl       = ((cur_price - entry) / entry * 100) if entry else 0
        act_at    = trade.get('activated_at', now_utc.isoformat())
        try:
            held = (now_utc - datetime.fromisoformat(act_at.replace('Z','+00:00'))).total_seconds() / 3600
        except Exception:
            held = 0

        trade.update({
            'close_price':  cur_price,
            'close_result': 'manual',
            'closed_at':    now_utc.isoformat(),
            'hours_held':   round(held, 2),
            'pnl_pct':      round(pnl, 2),
        })

        append_history({
            'ticker':      trade['ticker'],
            'result':      'manual',
            'grade':       trade.get('grade', '?'),
            'registered_at': trade.get('registered_at'),
            'activated_at':  trade.get('activated_at'),
            'closed_at':     now_utc.isoformat(),
            'entry_price':   entry,
            'close_price':   cur_price,
            'pnl_pct':       round(pnl, 2),
            'hours_held':    round(held, 2),
        })
        send_telegram(build_close_msg(trade, 'manual'))

    save_active_trades(remain)
    with _state_lock:
        scanner_state['active_trades'] = remain

    return {'ok': True, 'msg': f'{ticker} 수동 청산 완료'}

# ═══════════════════════════════════════════════════════════════
# 헬퍼
# ═══════════════════════════════════════════════════════════════
def _build_watch_item(result, now_utc, manual=False):
    return {
        'ticker': result['ticker'],
        'status': 'watch',
        'grade':  result['grade'],
        'manual': manual,
        'snapshot': {
            'registered_at': now_utc.isoformat(),
            'entry_price':   result['current_price'],
            'score':         result['score'],
            'grade':         result['grade'],
            'daily_k':       result.get('daily_k'),
            'h4_k':          result.get('h4_k'),
            'h1_k':          result.get('h1_k'),
            'vol_ratio':     result.get('vol_ratio', 1.0),
            'daily_dir':     result.get('daily_dir', ''),
            'h4_dir':        result.get('h4_dir', ''),
            'h1_dir':        result.get('h1_dir', ''),
        },
        'current': {
            'price':      result['current_price'],
            'score':      result['score'],
            'grade':      result['grade'],
            'daily_k':    result.get('daily_k'),
            'h4_k':       result.get('h4_k'),
            'h1_k':       result.get('h1_k'),
            'vol_ratio':  result.get('vol_ratio', 1.0),
            'daily_dir':  result.get('daily_dir', ''),
            'h4_dir':     result.get('h4_dir', ''),
            'h1_dir':     result.get('h1_dir', ''),
            'updated_at': now_utc.isoformat(),
        },
    }

def _clear_entries():
    """new_entries / removed_items 5분 후 클리어 (중복 방지)"""
    with _state_lock:
        scanner_state['new_entries']   = []
        scanner_state['removed_items'] = []
    log.debug("new_entries / removed_items 클리어")

# ═══════════════════════════════════════════════════════════════
# 메인 스캔 (60분)
# ═══════════════════════════════════════════════════════════════
def run_scan():
    global _clear_timer

    now_utc = datetime.now(timezone.utc)
    log.info(f"[SCAN START] {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    with _state_lock:
        scanner_state['status'] = 'scanning'

    try:
        # ── 매크로 필터 ──────────────────────────────────────────
        macro_ok   = True
        btc_weekly = None
        btc_daily  = None
        btc_price  = None

        try:
            weekly_closes, daily_closes = get_btc_closes()
            if len(weekly_closes) >= 20:
                btc_weekly = round(sum(weekly_closes[-20:]) / 20, 0)
                btc_price  = weekly_closes[-1]
                macro_ok   = btc_price >= btc_weekly
            if len(daily_closes) >= 20:
                btc_daily = round(sum(daily_closes[-20:]) / 20, 0)
        except Exception as e:
            log.warning(f"BTC 매크로 오류: {e}")

        # ── 기존 Watch/Active 로드 ───────────────────────────────
        watch_list    = load_watch_list()
        active_trades = load_active_trades()
        existing_tickers = {w['ticker'] for w in watch_list}
        active_tickers   = {a['ticker'] for a in active_trades}

        # ── 전체 종목 수집 ───────────────────────────────────────
        all_markets = get_all_krw_markets()
        targets     = [m for m in all_markets if m not in active_tickers]
        log.info(f"스캔 대상: {len(targets)}종목")

        # ── 병렬 분석 ────────────────────────────────────────────
        scan_results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(analyze_ticker, t): t for t in targets}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        scan_results[ticker] = result
                except Exception as e:
                    log.debug(f"분석 오류 {ticker}: {e}")

        log.info(f"일봉 게이트 통과: {len(scan_results)}종목")

        # ── 현재가 일괄 조회 ─────────────────────────────────────
        all_scan_tickers = list(scan_results.keys())
        price_map = get_current_prices(all_scan_tickers) if all_scan_tickers else {}

        # ── Watch 업데이트 & 신규 추가 ───────────────────────────
        new_watch_list = []
        newly_added    = []   # ← 이번 스캔에서 진짜 신규 등록된 것만
        removed_items  = []

        cfg            = mtf_setup.get_module_config()
        expiry_days    = cfg.get('WATCH_EXPIRY_DAYS', 7)

        # 기존 Watch 업데이트
        for item in watch_list:
            ticker = item['ticker']

            if ticker in scan_results:
                r   = scan_results[ticker]
                cur = item.get('current', {})
                cur.update({
                    'price':      price_map.get(ticker, cur.get('price', 0)),
                    'score':      r['score'],
                    'grade':      r['grade'],
                    'daily_k':    r.get('daily_k'),
                    'h4_k':       r.get('h4_k'),
                    'h1_k':       r.get('h1_k'),
                    'vol_ratio':  r.get('vol_ratio', 1.0),
                    'daily_dir':  r.get('daily_dir', ''),
                    'h4_dir':     r.get('h4_dir', ''),
                    'h1_dir':     r.get('h1_dir', ''),
                    'updated_at': now_utc.isoformat(),
                })
                item['current'] = cur
                item['grade']   = r['grade']
            else:
                # 일봉 게이트 탈락 → 만료 처리 (보호 등급 제외)
                grade = item.get('grade', 'C')
                if grade in ('A', 'S') or item.get('manual'):
                    new_watch_list.append(item)
                    continue

                reg = item.get('snapshot', {}).get('registered_at', now_utc.isoformat())
                try:
                    age_days = (now_utc - datetime.fromisoformat(reg.replace('Z','+00:00'))).days
                except Exception:
                    age_days = 0

                if age_days >= expiry_days:
                    snap = item.get('snapshot', {})
                    cur  = item.get('current', {})
                    try:
                        wh = (now_utc - datetime.fromisoformat(reg.replace('Z','+00:00'))).total_seconds() / 3600
                    except Exception:
                        wh = 0
                    append_history({
                        'ticker':       ticker,
                        'result':       'expired',
                        'grade':        grade,
                        'registered_at': reg,
                        'closed_at':    now_utc.isoformat(),
                        'watch_hours':  round(wh, 2),
                        'entry_price':  snap.get('entry_price', 0),
                        'close_price':  cur.get('price', snap.get('entry_price', 0)),
                        'pnl_pct':      round(
                            ((cur.get('price', 0) - snap.get('entry_price', 0)) / snap.get('entry_price', 1) * 100)
                            if snap.get('entry_price', 0) else 0, 2
                        ),
                        'snapshot_score': snap.get('score', 0),
                    })
                    removed_items.append(item)
                    log.info(f"[EXPIRED] {ticker} (등급:{grade}, {age_days}일)")
                    continue

            new_watch_list.append(item)

        # 신규 Watch 등록 (이번 스캔에서 처음 발견된 것만)
        current_watch_tickers = {w['ticker'] for w in new_watch_list}
        for ticker, result in scan_results.items():
            if ticker in current_watch_tickers:
                continue
            if ticker in active_tickers:
                continue
            result['current_price'] = price_map.get(ticker, result['current_price'])
            new_item = _build_watch_item(result, now_utc)
            new_watch_list.append(new_item)
            newly_added.append(new_item)   # ← 진짜 신규만 추적
            log.info(f"[NEW WATCH] {ticker} | {result['grade']}등급 | {result['score']}점")

        # ── 저장 ─────────────────────────────────────────────────
        save_watch_list(new_watch_list)
        stats = calc_stats()

        # ── 5분 후 new_entries 클리어 (중복 타이머 방지) ──────────
        if _clear_timer is not None:
            _clear_timer.cancel()
        _clear_timer = threading.Timer(300, _clear_entries)
        _clear_timer.daemon = True
        _clear_timer.start()

        # ── State 업데이트 ────────────────────────────────────────
        next_scan = (now_utc + timedelta(minutes=SCAN_INTERVAL_MIN)).isoformat()
        with _state_lock:
            scanner_state['status']        = 'idle'
            scanner_state['last_scan_at']  = now_utc.isoformat()
            scanner_state['next_scan_at']  = next_scan
            scanner_state['scan_count']    += 1
            scanner_state['watch_count']   = len(new_watch_list)
            scanner_state['active_count']  = len(active_trades)
            scanner_state['watch_list']    = new_watch_list
            scanner_state['active_trades'] = active_trades
            scanner_state['new_entries']   = newly_added
            scanner_state['removed_items'] = removed_items
            scanner_state['stats']         = stats
            scanner_state['macro'] = {
                'btc_weekly_ma20': btc_weekly,
                'btc_daily_ma20':  btc_daily,
                'btc_price':       btc_price,
                'macro_ok':        macro_ok,
            }

        log.info(
            f"[SCAN DONE] Watch:{len(new_watch_list)} | "
            f"신규:{len(newly_added)} | 만료:{len(removed_items)} | "
            f"Active:{len(active_trades)}"
        )

    except Exception as e:
        log.error(f"run_scan 오류: {e}", exc_info=True)
        with _state_lock:
            scanner_state['status'] = 'error'
            scanner_state['errors'].append({
                'time': now_utc.isoformat(),
                'msg':  str(e),
            })
            if len(scanner_state['errors']) > 20:
                scanner_state['errors'] = scanner_state['errors'][-20:]

# ═══════════════════════════════════════════════════════════════
# Watch 재스캔 루프 (15분)
# ═══════════════════════════════════════════════════════════════
def watch_rescan_loop():
    """Watch 종목만 재분석 → 점수/등급 업데이트, A/S → Active 전환"""
    log.info("watch_rescan_loop 시작")
    while True:
        try:
            time.sleep(WATCH_RESCAN_INTERVAL_MIN * 60)
            now_utc = datetime.now(timezone.utc)
            log.info(f"[WATCH RESCAN] {now_utc.strftime('%H:%M:%S UTC')}")

            watch_list    = load_watch_list()
            active_trades = load_active_trades()
            active_tickers = {a['ticker'] for a in active_trades}

            new_watch_list = []
            new_actives    = []

            for item in watch_list:
                ticker = item['ticker']
                if ticker in active_tickers:
                    continue

                result = analyze_ticker(ticker)
                if not result:
                    new_watch_list.append(item)
                    continue

                # 현재 정보 업데이트
                cur = item.get('current', {})
                price_map = get_current_prices([ticker])
                cur.update({
                    'price':      price_map.get(ticker, cur.get('price', 0)),
                    'score':      result['score'],
                    'grade':      result['grade'],
                    'daily_k':    result.get('daily_k'),
                    'h4_k':       result.get('h4_k'),
                    'h1_k':       result.get('h1_k'),
                    'vol_ratio':  result.get('vol_ratio', 1.0),
                    'daily_dir':  result.get('daily_dir', ''),
                    'h4_dir':     result.get('h4_dir', ''),
                    'h1_dir':     result.get('h1_dir', ''),
                    'updated_at': now_utc.isoformat(),
                })
                item['current'] = cur
                item['grade']   = result['grade']

                old_grade = item.get('snapshot', {}).get('grade', 'C')
                new_grade = result['grade']

                # A/S 등급이면 Active 전환
                if new_grade in ('A', 'S'):
                    active_item = activate_item(item)
                    active_trades.append(active_item)
                    new_actives.append(active_item)
                    send_telegram(build_active_msg(item))
                    log.info(f"[ACTIVE] {ticker} | {new_grade}등급 | {result['score']}점")
                else:
                    # 등급 상승 알림 (B→A는 이미 Active 처리됨, C→B 알림)
                    if old_grade == 'C' and new_grade == 'B':
                        send_telegram(
                            f"🟡 [등급상승] {ticker}\n"
                            f"C → B등급 ({result['score']}점)\n"
                            f"일봉K: {result.get('daily_k','-')} | 4hK: {result.get('h4_k','-')}"
                        )
                    new_watch_list.append(item)

            if new_actives:
                save_active_trades(active_trades)

            save_watch_list(new_watch_list)
            stats = calc_stats()

            with _state_lock:
                scanner_state['watch_list']            = new_watch_list
                scanner_state['active_trades']         = active_trades
                scanner_state['last_watch_rescan_at']  = now_utc.isoformat()
                scanner_state['watch_count']           = len(new_watch_list)
                scanner_state['active_count']          = len(active_trades)
                scanner_state['stats']                 = stats
                scanner_state['new_entries']           = []   # 재스캔 시 신규Watch 초기화

            log.info(f"[WATCH RESCAN DONE] Watch:{len(new_watch_list)} | 신규Active:{len(new_actives)}")

        except Exception as e:
            log.error(f"watch_rescan_loop 오류: {e}", exc_info=True)

# ═══════════════════════════════════════════════════════════════
# 가격 체크 루프 (5분)
# ═══════════════════════════════════════════════════════════════
def price_check_loop():
    """Watch 종목 현재가 업데이트"""
    log.info("price_check_loop 시작")
    while True:
        try:
            time.sleep(PRICE_CHECK_INTERVAL_MIN * 60)
            now_utc    = datetime.now(timezone.utc)
            watch_list = load_watch_list()

            if not watch_list:
                continue

            tickers   = [w['ticker'] for w in watch_list]
            price_map = get_current_prices(tickers)

            for item in watch_list:
                tk = item['ticker']
                if tk in price_map:
                    item.setdefault('current', {})['price']      = price_map[tk]
                    item.setdefault('current', {})['updated_at'] = now_utc.isoformat()

            save_watch_list(watch_list)

            with _state_lock:
                scanner_state['watch_list']           = watch_list
                scanner_state['last_price_check_at']  = now_utc.isoformat()

        except Exception as e:
            log.error(f"price_check_loop 오류: {e}")

# ═══════════════════════════════════════════════════════════════
# Active 모니터링 루프 (1분)
# ═══════════════════════════════════════════════════════════════
def active_monitor_loop():
    """Active 종목 TP/SL/Timeout 1분 간격 체크"""
    log.info("active_monitor_loop 시작")
    while True:
        try:
            time.sleep(ACTIVE_CHECK_INTERVAL_MIN * 60)
            now_utc = datetime.now(timezone.utc)

            active_trades = load_active_trades()
            if not active_trades:
                with _state_lock:
                    scanner_state['last_active_check_at'] = now_utc.isoformat()
                continue

            tickers   = [t['ticker'] for t in active_trades]
            price_map = get_current_prices(tickers)

            check_active_trades(price_map=price_map)
            active_trades = load_active_trades()  # 청산 후 재로드

            with _state_lock:
                scanner_state['active_trades']         = active_trades
                scanner_state['active_count']          = len(active_trades)
                scanner_state['last_active_check_at']  = now_utc.isoformat()

        except Exception as e:
            log.error(f"active_monitor_loop 오류: {e}")

# ═══════════════════════════════════════════════════════════════
# 메인 스캔 루프 (60분)
# ═══════════════════════════════════════════════════════════════
def manual_scan():
    """대시보드에서 수동 스캔 트리거"""
    _manual_scan_event.set()

def scanner_loop():
    log.info("scanner_loop 시작")
    # 최초 실행
    run_scan()
    while True:
        try:
            interval_sec = SCAN_INTERVAL_MIN * 60
            triggered    = _manual_scan_event.wait(timeout=interval_sec)
            if triggered:
                _manual_scan_event.clear()
                log.info("[MANUAL SCAN 트리거]")
            run_scan()
        except Exception as e:
            log.error(f"scanner_loop 오류: {e}")
            time.sleep(60)

# ═══════════════════════════════════════════════════════════════
# 직접 실행
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    run_scan()
