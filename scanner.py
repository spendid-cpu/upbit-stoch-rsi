# -*- coding: utf-8 -*-
"""
scanner.py — Upbit MTF 자동 스캐너
Version : v2.4.0
Changelog:
  v2.4.0 - DEEP Watch 시스템 추가
           get_ticker_change_24h: 24h 변화율 조회
           get_weekly_k: 주봉 Stoch RSI K값
           analyze_deep: DEEP 조건 분석
           run_scan / watch_rescan_loop 에 DEEP 감지 통합
           DEEP Active는 type='deep' 으로 구분
           DEEP 전용 텔레그램 메시지
  v2.3.1 - entry_strength grade 상한 폐지
  v2.3.0 - 등급별 만료, snap_k, 점수급등 알림, 일일요약
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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

VERSION = 'v2.4.0'

# ── 환경변수 ────────────────────────────────────────────────────
SCAN_INTERVAL_MIN         = int(os.getenv('SCAN_INTERVAL_MIN',         '60'))
WATCH_RESCAN_INTERVAL_MIN = int(os.getenv('WATCH_RESCAN_INTERVAL_MIN', '15'))
PRICE_CHECK_INTERVAL_MIN  = int(os.getenv('PRICE_CHECK_INTERVAL_MIN',  '5'))
ACTIVE_CHECK_INTERVAL_MIN = int(os.getenv('ACTIVE_CHECK_INTERVAL_MIN', '1'))
DAILY_SUMMARY_HOUR_KST    = int(os.getenv('DAILY_SUMMARY_HOUR_KST',    '9'))

REQUEST_DELAY         = float(os.getenv('REQUEST_DELAY',         '0.12'))
MAX_WORKERS           = int(os.getenv('MAX_WORKERS',             '6'))
CANDLE_COUNT          = int(os.getenv('CANDLE_COUNT',            '200'))
SCORE_SURGE_THRESHOLD = int(os.getenv('SCORE_SURGE_THRESHOLD',   '10'))

WATCH_LIST_FILE    = os.getenv('WATCH_LIST_FILE',    'watch_list.json')
ACTIVE_TRADES_FILE = os.getenv('ACTIVE_TRADES_FILE', 'active_trades.json')
TRADE_HISTORY_FILE = os.getenv('TRADE_HISTORY_FILE', 'trade_history.json')

TELEGRAM_TOKEN   = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID',   '')

TRADE_TP_PCT    = float(os.getenv('TRADE_TP_PCT',    '5.0'))
TRADE_SL_PCT    = float(os.getenv('TRADE_SL_PCT',    '3.0'))
TRADE_TIMEOUT_H = float(os.getenv('TRADE_TIMEOUT_H', '48.0'))

STABLE_COINS = {
    'KRW-USDT','KRW-USDC','KRW-USDS','KRW-USDE',
    'KRW-TUSD','KRW-USD1','KRW-XAUT','KRW-BTC',
}

# ── 전역 상태 ────────────────────────────────────────────────────
_state_lock        = threading.Lock()
_manual_scan_event = threading.Event()
_clear_timer       = None
_last_summary_date = None

scanner_state = {
    'version':               VERSION,
    'status':                'idle',
    'last_scan_at':          None,
    'last_watch_rescan_at':  None,
    'last_price_check_at':   None,
    'last_active_check_at':  None,
    'next_scan_at':          None,
    'scan_count':            0,
    'watch_count':           0,
    'active_count':          0,
    'deep_count':            0,
    'watch_list':            [],
    'active_trades':         [],
    'new_entries':           [],
    'removed_items':         [],
    'failed_tickers':        [],
    'macro': {
        'btc_weekly_ma20': None,
        'btc_daily_ma20':  None,
        'btc_price':       None,
        'btc_change_24h':  None,
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
        if 'snapshot' not in item:
            item = {
                'ticker':  item.get('ticker', ''),
                'status':  item.get('status', 'watch'),
                'grade':   item.get('grade', 'C'),
                'manual':  item.get('manual', False),
                'type':    item.get('type', 'normal'),
                'snapshot': {
                    'registered_at': item.get('registered_at',
                        datetime.now(timezone.utc).isoformat()),
                    'entry_price':   item.get('entry_price', 0),
                    'score':         item.get('score', 0),
                    'grade':         item.get('grade', 'C'),
                    'daily_k':       None,
                    'h4_k':          None,
                    'h1_k':          None,
                },
                'current': {
                    'price':      item.get('current_price', 0),
                    'score':      item.get('score', 0),
                    'grade':      item.get('grade', 'C'),
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                },
                'score_history': [],
            }
        if 'score_history' not in item:
            item['score_history'] = []
        if 'type' not in item:
            item['type'] = 'normal'
        result.append(item)
    return result

def save_watch_list(data):    _save_json(WATCH_LIST_FILE,    data)
def load_active_trades():     return _load_json(ACTIVE_TRADES_FILE, [])
def save_active_trades(data): _save_json(ACTIVE_TRADES_FILE, data)
def load_trade_history():     return _load_json(TRADE_HISTORY_FILE, [])

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
    return [
        m['market'] for m in data
        if m['market'].startswith('KRW-')
        and m['market'] not in STABLE_COINS
        and m.get('market_warning') != 'CAUTION'
    ]

def get_closes(ticker, unit='days', count=200):
    url_map = {
        'days':      f'{_UPBIT_BASE}/candles/days',
        'weeks':     f'{_UPBIT_BASE}/candles/weeks',
        'minutes':   f'{_UPBIT_BASE}/candles/minutes/240',
        'minutes60': f'{_UPBIT_BASE}/candles/minutes/60',
    }
    url = url_map.get(unit)
    if not url:
        return []
    data = _get(url, {'market': ticker, 'count': count})
    if not data:
        return []
    return [c['trade_price'] for c in reversed(data)]

def get_volumes(ticker, unit='days', count=30):
    url_map = {
        'days':    f'{_UPBIT_BASE}/candles/days',
        'minutes': f'{_UPBIT_BASE}/candles/minutes/240',
    }
    url = url_map.get(unit)
    if not url:
        return []
    data = _get(url, {'market': ticker, 'count': count})
    if not data:
        return []
    return [c.get('candle_acc_trade_volume', 0) for c in reversed(data)]

def get_btc_closes():
    weekly = get_closes('KRW-BTC', unit='weeks', count=25)
    daily  = get_closes('KRW-BTC', unit='days',  count=25)
    return weekly, daily

def get_current_prices(tickers):
    result = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i+100]
        data  = _get(f'{_UPBIT_BASE}/ticker', {'markets': ','.join(batch)})
        if data:
            for d in data:
                result[d['market']] = d.get('trade_price', 0)
        time.sleep(REQUEST_DELAY)
    return result

def get_ticker_change_24h(tickers):
    """24h 변화율 일괄 조회"""
    result = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i+100]
        data  = _get(f'{_UPBIT_BASE}/ticker', {'markets': ','.join(batch)})
        if data:
            for d in data:
                result[d['market']] = round(d.get('signed_change_rate', 0) * 100, 2)
        time.sleep(REQUEST_DELAY)
    return result

def get_weekly_k(ticker):
    """주봉 Stoch RSI K값 조회"""
    try:
        weekly_closes = get_closes(ticker, unit='weeks', count=100)
        if len(weekly_closes) < 30:
            return None
        presets = mtf_setup.calc_all_presets(weekly_closes)
        return mtf_setup._safe_k(presets, 'short')
    except Exception:
        return None

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
    return {
        'S': '🔴', 'A': '🟠', 'B': '🟡', 'C': '⚪',
        'DEEP-S': '💎', 'DEEP-A': '🔵', 'DEEP-B': '🟣',
    }.get(grade, '⚪')

def build_deep_msg(item):
    """DEEP Watch Active 전환 텔레그램 메시지"""
    tk    = item.get('ticker', '')
    cur   = item.get('current', {})
    snap  = item.get('snapshot', {})
    deep  = item.get('deep_data', {})
    grade = item.get('grade', 'DEEP-?')

    breakdown = deep.get('breakdown', {})
    btc_ch    = breakdown.get('btc_change', '-')
    coin_ch   = breakdown.get('coin_change', '-')
    relative  = breakdown.get('relative', '-')
    bottom_d  = breakdown.get('bottom_days', 0)
    vol_r     = breakdown.get('vol_ratio', '-')
    wk        = breakdown.get('weekly_k', '-')

    return (
        f"💎 <b>[DEEP 저점 포착 → ACTIVE] {tk}</b>\n\n"
        f"등급: {_grade_emoji(grade)}{grade} ({deep.get('score', 0)}점)\n"
        f"일봉K: {cur.get('daily_k', '-')} "
        f"({bottom_d}일째 ≤{mtf_setup.DEEP_K_THRESHOLD} 유지)\n\n"
        f"💪 상대강도\n"
        f"  BTC 24h: {btc_ch}%\n"
        f"  종목 24h: {coin_ch}%\n"
        f"  상대 버팀: +{relative}%\n\n"
        f"📦 거래량: 평균의 {vol_r}배\n"
        f"📅 주봉K: {wk}\n\n"
        f"진입가: {snap.get('entry_price', 0):,.4f}\n"
        f"TP: +{TRADE_TP_PCT}% / SL: -{TRADE_SL_PCT}%"
    )

def build_active_msg(item):
    g    = item.get('grade', '?')
    tk   = item.get('ticker', '')
    cur  = item.get('current', {})
    snap = item.get('snapshot', {})
    es   = cur.get('entry_strength', {})
    return (
        f"{_grade_emoji(g)} <b>[ACTIVE 전환] {tk}</b>\n"
        f"등급: {g}등급 ({cur.get('score', 0)}점)\n"
        f"신호강도: {es.get('icon','')}{es.get('label','')}\n"
        f"등록가: {snap.get('entry_price',0):,.4f} | "
        f"현재가: {cur.get('price',0):,.4f}\n"
        f"일봉K: {cur.get('daily_k','-')} | "
        f"4hK: {cur.get('h4_k','-')} | "
        f"1hK: {cur.get('h1_k','-')}\n"
        f"TP: +{TRADE_TP_PCT}% / SL: -{TRADE_SL_PCT}%"
    )

def build_close_msg(trade, result):
    tk    = trade.get('ticker', '')
    ep    = trade.get('entry_price', 0)
    cp    = trade.get('close_price', 0)
    pnl   = trade.get('pnl_pct', 0)
    held  = trade.get('hours_held', 0)
    ttype = trade.get('type', 'normal')
    icons = {'tp': '✅', 'sl': '❌', 'timeout': '⏰', 'manual': '🖐️'}
    deep_tag = ' [DEEP]' if ttype == 'deep' else ''
    return (
        f"{icons.get(result,'?')} <b>[청산{deep_tag}] {tk}</b>\n"
        f"결과: {result.upper()}\n"
        f"진입가: {ep:,.4f} → 청산가: {cp:,.4f}\n"
        f"수익률: {pnl:+.2f}%\n"
        f"보유시간: {held:.1f}h"
    )

def build_surge_msg(ticker, old_score, new_score, grade, cur):
    diff = new_score - old_score
    es   = cur.get('entry_strength', {})
    return (
        f"⚡ <b>[점수급등] {ticker}</b>\n"
        f"등급: {_grade_emoji(grade)}{grade} | {old_score}→{new_score}점 (+{diff})\n"
        f"신호강도: {es.get('icon','')}{es.get('label','')}\n"
        f"일봉K: {cur.get('daily_k','-')} | "
        f"4hK: {cur.get('h4_k','-')} | "
        f"1hK: {cur.get('h1_k','-')}"
    )

def build_daily_summary():
    watch_list    = load_watch_list()
    active_trades = load_active_trades()
    stats         = calc_stats()
    deep_list     = [w for w in watch_list if w.get('type') == 'deep']
    top5 = sorted(watch_list,
                  key=lambda x: x.get('current', {}).get('score', 0),
                  reverse=True)[:5]
    top5_lines = '\n'.join([
        f"  {i+1}. {w['ticker']} {_grade_emoji(w.get('grade','C'))}"
        f"{w.get('grade','C')} {w.get('current',{}).get('score',0)}점"
        for i, w in enumerate(top5)
    ])
    now_kst = datetime.now(timezone(timedelta(hours=9)))
    return (
        f"📊 <b>일일 요약 {now_kst.strftime('%Y-%m-%d')}</b>\n\n"
        f"📋 Watch: {len(watch_list)}개 (DEEP: {len(deep_list)}개)\n"
        f"🔵 Active: {len(active_trades)}개\n"
        f"✅ 전체 승률: {stats.get('win_rate', 0)}%\n"
        f"💎 DEEP 승률: {stats.get('deep_win_rate', 0)}%\n"
        f"📈 평균 PnL: {stats.get('avg_pnl', 0):+.2f}%\n\n"
        f"🏆 Watch 상위 5종목:\n{top5_lines}"
    )

# ═══════════════════════════════════════════════════════════════
# 분석
# ═══════════════════════════════════════════════════════════════
def analyze_ticker(ticker, snap_k=None):
    try:
        daily_closes = get_closes(ticker, unit='days',      count=CANDLE_COUNT)
        h4_closes    = get_closes(ticker, unit='minutes',   count=CANDLE_COUNT)
        h1_closes    = get_closes(ticker, unit='minutes60', count=CANDLE_COUNT)
        daily_vols   = get_volumes(ticker, unit='days',     count=30)

        if len(daily_closes) < 30:
            return None

        time.sleep(REQUEST_DELAY)

        daily_presets = mtf_setup.calc_all_presets(daily_closes)
        h4_presets    = mtf_setup.calc_all_presets(h4_closes) if len(h4_closes) >= 30 else {}
        h1_presets    = mtf_setup.calc_all_presets(h1_closes) if len(h1_closes) >= 30 else {}

        gate = mtf_setup.evaluate_daily_gate(daily_presets)
        if not gate.get('pass'):
            return None

        vol_ratio = 1.0
        if len(daily_vols) >= 21:
            avg_vol   = sum(daily_vols[-21:-1]) / 20
            vol_ratio = (daily_vols[-1] / avg_vol) if avg_vol > 0 else 1.0

        score_result = mtf_setup.calc_watch_score(
            daily_presets=daily_presets,
            h4_presets=h4_presets,
            h1_presets=h1_presets,
            vol_ratio=vol_ratio,
            snap_k=snap_k,
        )

        return {
            'ticker':         ticker,
            'score':          score_result.get('score', 0),
            'grade':          score_result.get('grade', 'C'),
            'breakdown':      score_result.get('breakdown', {}),
            'daily_dir':      score_result.get('daily_dir', ''),
            'h4_dir':         score_result.get('h4_dir', ''),
            'h1_dir':         score_result.get('h1_dir', ''),
            'h4_golden':      score_result.get('h4_golden', False),
            'h1_golden':      score_result.get('h1_golden', False),
            'entry_strength': score_result.get('entry_strength', {}),
            'daily_k':        score_result.get('daily_k'),
            'h4_k':           score_result.get('h4_k'),
            'h1_k':           score_result.get('h1_k'),
            'vol_ratio':      round(vol_ratio, 2),
            'daily_vols':     daily_vols,
            'current_price':  daily_closes[-1] if daily_closes else 0,
        }

    except Exception as e:
        log.debug(f"analyze_ticker 오류 {ticker}: {e}")
        return None


def analyze_deep(ticker, daily_k, btc_change_24h, coin_change_24h,
                 vol_ratio=1.0, days_at_bottom=0):
    """DEEP 조건 분석 및 점수 계산"""
    # 필수 조건 체크
    cond = mtf_setup.evaluate_deep_condition(daily_k, btc_change_24h, coin_change_24h)
    if not cond.get('pass'):
        return None

    # 주봉K 조회 (간략히)
    weekly_k = get_weekly_k(ticker)

    # DEEP 점수 계산
    deep_result = mtf_setup.calc_deep_score(
        daily_k=daily_k,
        btc_change_24h=btc_change_24h,
        coin_change_24h=coin_change_24h,
        days_at_bottom=days_at_bottom,
        vol_ratio=vol_ratio,
        weekly_k=weekly_k,
    )

    grade = mtf_setup.get_deep_grade(deep_result['score'])

    return {
        'score':     deep_result['score'],
        'grade':     grade,
        'breakdown': deep_result['breakdown'],
        'condition': cond,
        'weekly_k':  weekly_k,
    }

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
            'avg_watch_hours': 0,
            'deep_total': 0, 'deep_tp': 0, 'deep_win_rate': 0,
            'deep_avg_pnl': 0,
            'grade_stats': {},
        }

    total     = len(history)
    activated = sum(1 for h in history if h.get('result') == 'activated')
    expired   = sum(1 for h in history if h.get('result') == 'expired')
    tp_cnt    = sum(1 for h in history if h.get('result') == 'tp')
    sl_cnt    = sum(1 for h in history if h.get('result') == 'sl')
    to_cnt    = sum(1 for h in history if h.get('result') == 'timeout')
    mn_cnt    = sum(1 for h in history if h.get('result') == 'manual')

    closed     = [h for h in history if h.get('result') in ('tp','sl','timeout','manual')]
    closed_cnt = len(closed)
    pnls       = [h.get('pnl_pct', 0) for h in closed if h.get('pnl_pct') is not None]

    wh_list         = [h.get('watch_hours', 0) for h in history if h.get('watch_hours')]
    avg_watch_hours = round(sum(wh_list) / len(wh_list), 1) if wh_list else 0

    # DEEP 전용 통계
    deep_closed   = [h for h in closed if h.get('type') == 'deep']
    deep_tp       = sum(1 for h in deep_closed if h.get('result') == 'tp')
    deep_pnls     = [h.get('pnl_pct', 0) for h in deep_closed if h.get('pnl_pct') is not None]

    grade_stats = {}
    for grade in ['S', 'A', 'B', 'C', 'DEEP-S', 'DEEP-A', 'DEEP-B']:
        g_closed = [h for h in closed if h.get('grade') == grade]
        g_tp     = sum(1 for h in g_closed if h.get('result') == 'tp')
        grade_stats[grade] = {
            'total':    len(g_closed),
            'tp':       g_tp,
            'win_rate': round(g_tp / len(g_closed) * 100, 1) if g_closed else 0,
        }

    return {
        'total':           total,
        'activated':       activated,
        'expired':         expired,
        'tp':              tp_cnt,
        'sl':              sl_cnt,
        'timeout':         to_cnt,
        'manual':          mn_cnt,
        'tp_rate':         round(tp_cnt / closed_cnt * 100, 1) if closed_cnt else 0,
        'sl_rate':         round(sl_cnt / closed_cnt * 100, 1) if closed_cnt else 0,
        'win_rate':        round(tp_cnt / closed_cnt * 100, 1) if closed_cnt else 0,
        'avg_pnl':         round(sum(pnls) / len(pnls), 2) if pnls else 0,
        'best_pnl':        round(max(pnls), 2) if pnls else 0,
        'worst_pnl':       round(min(pnls), 2) if pnls else 0,
        'avg_watch_hours': avg_watch_hours,
        'deep_total':      len(deep_closed),
        'deep_tp':         deep_tp,
        'deep_win_rate':   round(deep_tp / len(deep_closed) * 100, 1) if deep_closed else 0,
        'deep_avg_pnl':    round(sum(deep_pnls) / len(deep_pnls), 2) if deep_pnls else 0,
        'grade_stats':     grade_stats,
    }

# ═══════════════════════════════════════════════════════════════
# Active 전환
# ═══════════════════════════════════════════════════════════════
def activate_item(watch_item, trade_type='normal'):
    now_utc     = datetime.now(timezone.utc)
    ticker      = watch_item.get('ticker', '')
    snap        = watch_item.get('snapshot', {})
    cur         = watch_item.get('current',  {})
    entry_price = cur.get('price') or snap.get('entry_price', 0)

    active = {
        'ticker':         ticker,
        'type':           trade_type,
        'grade':          watch_item.get('grade', 'C'),
        'entry_price':    entry_price,
        'entry_score':    cur.get('score', snap.get('score', 0)),
        'entry_strength': cur.get('entry_strength', {}),
        'deep_data':      watch_item.get('deep_data', {}),
        'activated_at':   now_utc.isoformat(),
        'registered_at':  snap.get('registered_at', now_utc.isoformat()),
        'tp_price':       round(entry_price * (1 + TRADE_TP_PCT / 100), 8),
        'sl_price':       round(entry_price * (1 - TRADE_SL_PCT / 100), 8),
        'timeout_at':     (now_utc + timedelta(hours=TRADE_TIMEOUT_H)).isoformat(),
        'current_price':  entry_price,
        'pnl_pct':        0.0,
        'snapshot':       snap,
        'current':        cur,
    }

    reg_at = snap.get('registered_at', now_utc.isoformat())
    try:
        watch_hours = (now_utc - datetime.fromisoformat(
            reg_at.replace('Z', '+00:00'))).total_seconds() / 3600
    except Exception:
        watch_hours = 0

    append_history({
        'ticker':         ticker,
        'result':         'activated',
        'type':           trade_type,
        'grade':          active['grade'],
        'registered_at':  reg_at,
        'activated_at':   now_utc.isoformat(),
        'watch_hours':    round(watch_hours, 2),
        'entry_price':    entry_price,
        'snapshot_score': snap.get('score', 0),
        'entry_score':    active['entry_score'],
        'deep_data':      watch_item.get('deep_data', {}),
    })
    return active

# ═══════════════════════════════════════════════════════════════
# Active TP/SL/Timeout 체크
# ═══════════════════════════════════════════════════════════════
def check_active_trades(price_map=None):
    active_trades = load_active_trades()
    if not active_trades:
        return []

    if price_map is None:
        price_map = get_current_prices([t['ticker'] for t in active_trades])

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

        close_result = None
        if cur_price >= tp:
            close_result = 'tp'
        elif cur_price <= sl:
            close_result = 'sl'
        elif timeout_at:
            try:
                if now_utc >= datetime.fromisoformat(timeout_at.replace('Z', '+00:00')):
                    close_result = 'timeout'
            except Exception:
                pass

        if close_result:
            act_at = trade.get('activated_at', now_utc.isoformat())
            try:
                held = (now_utc - datetime.fromisoformat(
                    act_at.replace('Z', '+00:00'))).total_seconds() / 3600
            except Exception:
                held = 0

            trade.update({
                'close_price':  cur_price,
                'close_result': close_result,
                'closed_at':    now_utc.isoformat(),
                'hours_held':   round(held, 2),
                'pnl_pct':      round(pnl, 2),
            })
            append_history({
                'ticker':        ticker,
                'result':        close_result,
                'type':          trade.get('type', 'normal'),
                'grade':         trade.get('grade', '?'),
                'registered_at': trade.get('registered_at'),
                'activated_at':  trade.get('activated_at'),
                'closed_at':     now_utc.isoformat(),
                'entry_price':   entry,
                'close_price':   cur_price,
                'pnl_pct':       round(pnl, 2),
                'hours_held':    round(held, 2),
                'entry_score':   trade.get('entry_score', 0),
                'deep_data':     trade.get('deep_data', {}),
            })
            send_telegram(build_close_msg(trade, close_result))
            closed_out.append(trade)
            log.info(f"[{close_result.upper()}] {ticker} | PnL: {pnl:+.2f}% | type:{trade.get('type','normal')}")
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
    if ticker in [w['ticker'] for w in watch_list]:
        return {'ok': False, 'msg': f'{ticker} 이미 등록됨'}

    result  = analyze_ticker(ticker)
    now_utc = datetime.now(timezone.utc)

    if result:
        item = _build_watch_item(result, now_utc, manual=True)
    else:
        price = get_current_prices([ticker]).get(ticker, 0)
        item  = {
            'ticker':  ticker, 'status': 'watch',
            'grade':   'C',    'manual': True, 'type': 'normal',
            'snapshot': {
                'registered_at': now_utc.isoformat(),
                'entry_price': price, 'score': 0, 'grade': 'C',
                'daily_k': None, 'h4_k': None, 'h1_k': None,
            },
            'current': {
                'price': price, 'score': 0, 'grade': 'C',
                'daily_k': None, 'h4_k': None, 'h1_k': None,
                'entry_strength': {}, 'updated_at': now_utc.isoformat(),
            },
            'score_history': [],
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
        now_utc = datetime.now(timezone.utc)
        for w in removed:
            snap = w.get('snapshot', {})
            reg  = snap.get('registered_at', now_utc.isoformat())
            try:
                wh = (now_utc - datetime.fromisoformat(
                    reg.replace('Z', '+00:00'))).total_seconds() / 3600
            except Exception:
                wh = 0
            append_history({
                'ticker':        ticker,
                'result':        'manual_remove',
                'type':          w.get('type', 'normal'),
                'grade':         w.get('grade', 'C'),
                'registered_at': reg,
                'closed_at':     now_utc.isoformat(),
                'watch_hours':   round(wh, 2),
                'entry_price':   snap.get('entry_price', 0),
                'close_price':   w.get('current', {}).get('price', 0),
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
            held = (now_utc - datetime.fromisoformat(
                act_at.replace('Z', '+00:00'))).total_seconds() / 3600
        except Exception:
            held = 0

        trade.update({
            'close_price': cur_price, 'close_result': 'manual',
            'closed_at': now_utc.isoformat(),
            'hours_held': round(held, 2), 'pnl_pct': round(pnl, 2),
        })
        append_history({
            'ticker':        trade['ticker'],
            'result':        'manual',
            'type':          trade.get('type', 'normal'),
            'grade':         trade.get('grade', '?'),
            'registered_at': trade.get('registered_at'),
            'activated_at':  trade.get('activated_at'),
            'closed_at':     now_utc.isoformat(),
            'entry_price':   entry,
            'close_price':   cur_price,
            'pnl_pct':       round(pnl, 2),
            'hours_held':    round(held, 2),
            'deep_data':     trade.get('deep_data', {}),
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
        'ticker':  result['ticker'],
        'status':  'watch',
        'grade':   result['grade'],
        'type':    'normal',
        'manual':  manual,
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
            'price':          result['current_price'],
            'score':          result['score'],
            'grade':          result['grade'],
            'daily_k':        result.get('daily_k'),
            'h4_k':           result.get('h4_k'),
            'h1_k':           result.get('h1_k'),
            'vol_ratio':      result.get('vol_ratio', 1.0),
            'daily_dir':      result.get('daily_dir', ''),
            'h4_dir':         result.get('h4_dir', ''),
            'h1_dir':         result.get('h1_dir', ''),
            'h4_golden':      result.get('h4_golden', False),
            'h1_golden':      result.get('h1_golden', False),
            'entry_strength': result.get('entry_strength', {}),
            'updated_at':     now_utc.isoformat(),
        },
        'score_history': [{
            'time':  now_utc.isoformat(),
            'score': result['score'],
            'grade': result['grade'],
        }],
    }

def _build_deep_watch_item(ticker, price, deep_result, now_utc):
    """DEEP Watch 아이템 생성"""
    return {
        'ticker':    ticker,
        'status':    'watch',
        'grade':     deep_result['grade'],
        'type':      'deep',
        'manual':    False,
        'deep_data': deep_result,
        'snapshot': {
            'registered_at': now_utc.isoformat(),
            'entry_price':   price,
            'score':         deep_result['score'],
            'grade':         deep_result['grade'],
            'daily_k':       deep_result['breakdown'].get('k_position'),
            'btc_change':    deep_result['breakdown'].get('btc_change'),
            'coin_change':   deep_result['breakdown'].get('coin_change'),
            'relative':      deep_result['breakdown'].get('relative'),
        },
        'current': {
            'price':      price,
            'score':      deep_result['score'],
            'grade':      deep_result['grade'],
            'updated_at': now_utc.isoformat(),
        },
        'score_history': [{
            'time':  now_utc.isoformat(),
            'score': deep_result['score'],
            'grade': deep_result['grade'],
        }],
    }

def _clear_entries():
    with _state_lock:
        scanner_state['new_entries']   = []
        scanner_state['removed_items'] = []

def _update_score_history(item, new_score, new_grade, now_utc):
    history = item.get('score_history', [])
    history.append({'time': now_utc.isoformat(), 'score': new_score, 'grade': new_grade})
    if len(history) > 50:
        history = history[-50:]
    item['score_history'] = history

def _count_days_at_bottom(item):
    """score_history에서 K≤5 유지 일수 추정"""
    history = item.get('score_history', [])
    if not history:
        return 0
    try:
        reg = item.get('snapshot', {}).get('registered_at', '')
        if reg:
            reg_dt  = datetime.fromisoformat(reg.replace('Z', '+00:00'))
            now_utc = datetime.now(timezone.utc)
            days    = (now_utc - reg_dt).days
            return max(1, days)
    except Exception:
        pass
    return 1

# ═══════════════════════════════════════════════════════════════
# 메인 스캔 (60분)
# ═══════════════════════════════════════════════════════════════
def run_scan():
    global _clear_timer

    now_utc = datetime.now(timezone.utc)
    log.info(f"[SCAN START] {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    with _state_lock:
        scanner_state['status'] = 'scanning'
        failed_tickers = list(scanner_state.get('failed_tickers', []))

    try:
        # ── 매크로 + BTC 24h 변화율 ──────────────────────────────
        macro_ok = True
        btc_weekly = btc_daily = btc_price = btc_change_24h = None
        try:
            weekly_closes, daily_closes = get_btc_closes()
            if len(weekly_closes) >= 20:
                btc_weekly = round(sum(weekly_closes[-20:]) / 20, 0)
                btc_price  = weekly_closes[-1]
                macro_ok   = btc_price >= btc_weekly
            if len(daily_closes) >= 20:
                btc_daily = round(sum(daily_closes[-20:]) / 20, 0)
            # BTC 24h 변화율
            btc_ticker_data = get_ticker_change_24h(['KRW-BTC'])
            btc_change_24h  = btc_ticker_data.get('KRW-BTC')
        except Exception as e:
            log.warning(f"BTC 매크로 오류: {e}")

        watch_list     = load_watch_list()
        active_trades  = load_active_trades()
        active_tickers = {a['ticker'] for a in active_trades}

        all_markets = get_all_krw_markets()
        targets     = list(set(
            [m for m in all_markets if m not in active_tickers] +
            [t for t in failed_tickers if t not in active_tickers]
        ))
        log.info(f"스캔 대상: {len(targets)}종목")

        snap_k_map = {
            w['ticker']: w.get('snapshot', {}).get('daily_k')
            for w in watch_list
        }

        scan_results = {}
        new_failed   = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(analyze_ticker, t, snap_k_map.get(t)): t
                for t in targets
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        scan_results[ticker] = result
                    else:
                        new_failed.append(ticker)
                except Exception as e:
                    log.debug(f"분석 오류 {ticker}: {e}")
                    new_failed.append(ticker)

        log.info(f"일봉 게이트 통과: {len(scan_results)}종목")

        # ── 24h 변화율 일괄 조회 (DEEP 판단용) ──────────────────
        change_24h_map = {}
        if btc_change_24h is not None:
            scan_tickers = list(scan_results.keys())
            change_24h_map = get_ticker_change_24h(scan_tickers)

        price_map = get_current_prices(list(scan_results.keys())) if scan_results else {}

        new_watch_list = []
        newly_added    = []
        removed_items  = []
        new_actives    = []

        existing_tickers = {w['ticker'] for w in watch_list}

        # 기존 Watch 업데이트
        for item in watch_list:
            ticker = item['ticker']

            if ticker in scan_results:
                r         = scan_results[ticker]
                cur       = item.get('current', {})
                old_score = cur.get('score', 0)
                new_score = r['score']

                cur.update({
                    'price':          price_map.get(ticker, cur.get('price', 0)),
                    'score':          new_score,
                    'grade':          r['grade'],
                    'daily_k':        r.get('daily_k'),
                    'h4_k':           r.get('h4_k'),
                    'h1_k':           r.get('h1_k'),
                    'vol_ratio':      r.get('vol_ratio', 1.0),
                    'daily_dir':      r.get('daily_dir', ''),
                    'h4_dir':         r.get('h4_dir', ''),
                    'h1_dir':         r.get('h1_dir', ''),
                    'h4_golden':      r.get('h4_golden', False),
                    'h1_golden':      r.get('h1_golden', False),
                    'entry_strength': r.get('entry_strength', {}),
                    'updated_at':     now_utc.isoformat(),
                })
                item['current'] = cur
                item['grade']   = r['grade']
                _update_score_history(item, new_score, r['grade'], now_utc)

                if (new_score - old_score) >= SCORE_SURGE_THRESHOLD:
                    send_telegram(build_surge_msg(ticker, old_score, new_score, r['grade'], cur))

                # DEEP 체크 (일반 Watch 종목도 DEEP 조건 충족 시 전환)
                if (item.get('type') != 'deep'
                        and btc_change_24h is not None
                        and r.get('daily_k') is not None
                        and r.get('daily_k', 99) <= mtf_setup.DEEP_K_THRESHOLD):
                    coin_ch = change_24h_map.get(ticker)
                    days_bt = _count_days_at_bottom(item)
                    deep_r  = analyze_deep(
                        ticker, r['daily_k'], btc_change_24h, coin_ch,
                        vol_ratio=r.get('vol_ratio', 1.0),
                        days_at_bottom=days_bt,
                    )
                    if deep_r and deep_r['grade'] in ('DEEP-S', 'DEEP-A'):
                        item['type']      = 'deep'
                        item['grade']     = deep_r['grade']
                        item['deep_data'] = deep_r
                        active_item = activate_item(item, trade_type='deep')
                        active_trades.append(active_item)
                        new_actives.append(active_item)
                        send_telegram(build_deep_msg(item))
                        log.info(f"[DEEP ACTIVE] {ticker} | {deep_r['grade']} | {deep_r['score']}점")
                        continue  # Watch에서 제거 (Active로 이동)

            else:
                grade = item.get('grade', 'C')
                # 보호 등급 (A/S/DEEP 계열 / manual)
                if grade in ('A', 'S', 'DEEP-S', 'DEEP-A', 'DEEP-B') or item.get('manual'):
                    new_watch_list.append(item)
                    continue

                reg = item.get('snapshot', {}).get('registered_at', now_utc.isoformat())
                try:
                    age_days = (now_utc - datetime.fromisoformat(
                        reg.replace('Z', '+00:00'))).days
                except Exception:
                    age_days = 0

                expiry = mtf_setup.get_expiry_days(grade)
                if expiry is not None and age_days >= expiry:
                    snap = item.get('snapshot', {})
                    cur  = item.get('current',  {})
                    try:
                        wh = (now_utc - datetime.fromisoformat(
                            reg.replace('Z', '+00:00'))).total_seconds() / 3600
                    except Exception:
                        wh = 0
                    ep = snap.get('entry_price', 0)
                    cp = cur.get('price', ep)
                    append_history({
                        'ticker':         ticker,
                        'result':         'expired',
                        'type':           item.get('type', 'normal'),
                        'grade':          grade,
                        'registered_at':  reg,
                        'closed_at':      now_utc.isoformat(),
                        'watch_hours':    round(wh, 2),
                        'entry_price':    ep,
                        'close_price':    cp,
                        'pnl_pct':        round((cp - ep) / ep * 100, 2) if ep else 0,
                        'snapshot_score': snap.get('score', 0),
                    })
                    removed_items.append(item)
                    log.info(f"[EXPIRED] {ticker} ({grade}, {age_days}일)")
                    continue

            new_watch_list.append(item)

        # 신규 Watch 등록
        current_tickers = {w['ticker'] for w in new_watch_list}
        for ticker, result in scan_results.items():
            if ticker in current_tickers or ticker in active_tickers:
                continue
            result['current_price'] = price_map.get(ticker, result['current_price'])

            # 신규 종목도 DEEP 체크
            if (btc_change_24h is not None
                    and result.get('daily_k') is not None
                    and result.get('daily_k', 99) <= mtf_setup.DEEP_K_THRESHOLD):
                coin_ch = change_24h_map.get(ticker)
                deep_r  = analyze_deep(
                    ticker, result['daily_k'], btc_change_24h, coin_ch,
                    vol_ratio=result.get('vol_ratio', 1.0),
                    days_at_bottom=1,
                )
                if deep_r and deep_r['grade'] in ('DEEP-S', 'DEEP-A'):
                    deep_item = _build_deep_watch_item(
                        ticker, result['current_price'], deep_r, now_utc)
                    active_item = activate_item(deep_item, trade_type='deep')
                    active_trades.append(active_item)
                    new_actives.append(active_item)
                    send_telegram(build_deep_msg(deep_item))
                    log.info(f"[DEEP NEW ACTIVE] {ticker} | {deep_r['grade']} | {deep_r['score']}점")
                    continue

            new_item = _build_watch_item(result, now_utc)
            new_watch_list.append(new_item)
            newly_added.append(new_item)
            log.info(f"[NEW WATCH] {ticker} | {result['grade']}등급 | {result['score']}점")

        if new_actives:
            save_active_trades(active_trades)

        save_watch_list(new_watch_list)
        stats = calc_stats()

        if _clear_timer is not None:
            _clear_timer.cancel()
        _clear_timer = threading.Timer(300, _clear_entries)
        _clear_timer.daemon = True
        _clear_timer.start()

        deep_count = sum(1 for w in new_watch_list if w.get('type') == 'deep')
        next_scan  = (now_utc + timedelta(minutes=SCAN_INTERVAL_MIN)).isoformat()

        with _state_lock:
            scanner_state.update({
                'status':         'idle',
                'last_scan_at':   now_utc.isoformat(),
                'next_scan_at':   next_scan,
                'scan_count':     scanner_state['scan_count'] + 1,
                'watch_count':    len(new_watch_list),
                'active_count':   len(active_trades),
                'deep_count':     deep_count,
                'watch_list':     new_watch_list,
                'active_trades':  active_trades,
                'new_entries':    newly_added,
                'removed_items':  removed_items,
                'failed_tickers': new_failed[-50:],
                'stats':          stats,
                'macro': {
                    'btc_weekly_ma20': btc_weekly,
                    'btc_daily_ma20':  btc_daily,
                    'btc_price':       btc_price,
                    'btc_change_24h':  btc_change_24h,
                    'macro_ok':        macro_ok,
                },
            })

        log.info(
            f"[SCAN DONE] Watch:{len(new_watch_list)} (DEEP:{deep_count}) | "
            f"신규:{len(newly_added)} | 만료:{len(removed_items)} | "
            f"DEEP Active:{len(new_actives)}"
        )

    except Exception as e:
        log.error(f"run_scan 오류: {e}", exc_info=True)
        with _state_lock:
            scanner_state['status'] = 'error'
            scanner_state['errors'].append({'time': now_utc.isoformat(), 'msg': str(e)})
            if len(scanner_state['errors']) > 20:
                scanner_state['errors'] = scanner_state['errors'][-20:]

# ═══════════════════════════════════════════════════════════════
# Watch 재스캔 루프 (15분) - DEEP 체크 포함
# ═══════════════════════════════════════════════════════════════
def watch_rescan_loop():
    log.info("watch_rescan_loop 시작")
    while True:
        try:
            time.sleep(WATCH_RESCAN_INTERVAL_MIN * 60)
            now_utc = datetime.now(timezone.utc)
            log.info(f"[WATCH RESCAN] {now_utc.strftime('%H:%M:%S UTC')}")

            watch_list     = load_watch_list()
            active_trades  = load_active_trades()
            active_tickers = {a['ticker'] for a in active_trades}

            # BTC 24h 변화율 갱신
            btc_change_24h = None
            try:
                btc_data = get_ticker_change_24h(['KRW-BTC'])
                btc_change_24h = btc_data.get('KRW-BTC')
            except Exception:
                pass

            new_watch_list = []
            new_actives    = []

            for item in watch_list:
                ticker = item['ticker']
                if ticker in active_tickers:
                    continue

                snap_k = item.get('snapshot', {}).get('daily_k')
                result = analyze_ticker(ticker, snap_k=snap_k)

                if not result:
                    new_watch_list.append(item)
                    continue

                cur       = item.get('current', {})
                old_score = cur.get('score', 0)
                old_grade = item.get('snapshot', {}).get('grade', 'C')
                new_score = result['score']
                new_grade = result['grade']

                price_map = get_current_prices([ticker])
                cur.update({
                    'price':          price_map.get(ticker, cur.get('price', 0)),
                    'score':          new_score,
                    'grade':          new_grade,
                    'daily_k':        result.get('daily_k'),
                    'h4_k':           result.get('h4_k'),
                    'h1_k':           result.get('h1_k'),
                    'vol_ratio':      result.get('vol_ratio', 1.0),
                    'daily_dir':      result.get('daily_dir', ''),
                    'h4_dir':         result.get('h4_dir', ''),
                    'h1_dir':         result.get('h1_dir', ''),
                    'h4_golden':      result.get('h4_golden', False),
                    'h1_golden':      result.get('h1_golden', False),
                    'entry_strength': result.get('entry_strength', {}),
                    'updated_at':     now_utc.isoformat(),
                })
                item['current'] = cur
                item['grade']   = new_grade
                _update_score_history(item, new_score, new_grade, now_utc)

                if (new_score - old_score) >= SCORE_SURGE_THRESHOLD:
                    send_telegram(build_surge_msg(ticker, old_score, new_score, new_grade, cur))

                # DEEP 조건 체크 (15분마다)
                deep_activated = False
                if (item.get('type') != 'deep'
                        and btc_change_24h is not None
                        and result.get('daily_k') is not None
                        and result.get('daily_k', 99) <= mtf_setup.DEEP_K_THRESHOLD):
                    try:
                        coin_ch = get_ticker_change_24h([ticker]).get(ticker)
                        days_bt = _count_days_at_bottom(item)
                        deep_r  = analyze_deep(
                            ticker, result['daily_k'], btc_change_24h, coin_ch,
                            vol_ratio=result.get('vol_ratio', 1.0),
                            days_at_bottom=days_bt,
                        )
                        if deep_r and deep_r['grade'] in ('DEEP-S', 'DEEP-A'):
                            item['type']      = 'deep'
                            item['grade']     = deep_r['grade']
                            item['deep_data'] = deep_r
                            active_item = activate_item(item, trade_type='deep')
                            active_trades.append(active_item)
                            new_actives.append(active_item)
                            send_telegram(build_deep_msg(item))
                            log.info(f"[DEEP ACTIVE] {ticker} | {deep_r['grade']} | {deep_r['score']}점")
                            deep_activated = True
                    except Exception as e:
                        log.debug(f"DEEP 체크 오류 {ticker}: {e}")

                if deep_activated:
                    continue  # Watch에서 제거

                # 일반 A/S → Active 전환
                if new_grade in ('A', 'S'):
                    active_item = activate_item(item, trade_type='normal')
                    active_trades.append(active_item)
                    new_actives.append(active_item)
                    send_telegram(build_active_msg(item))
                    log.info(f"[ACTIVE] {ticker} | {new_grade}등급 | {new_score}점")
                else:
                    if old_grade == 'C' and new_grade == 'B':
                        send_telegram(
                            f"🟡 [등급상승] {ticker}\n"
                            f"C → B등급 ({new_score}점)\n"
                            f"일봉K: {result.get('daily_k','-')} | "
                            f"4hK: {result.get('h4_k','-')}"
                        )
                    new_watch_list.append(item)

            if new_actives:
                save_active_trades(active_trades)

            save_watch_list(new_watch_list)
            stats      = calc_stats()
            deep_count = sum(1 for w in new_watch_list if w.get('type') == 'deep')

            with _state_lock:
                scanner_state['watch_list']           = new_watch_list
                scanner_state['active_trades']        = active_trades
                scanner_state['last_watch_rescan_at'] = now_utc.isoformat()
                scanner_state['watch_count']          = len(new_watch_list)
                scanner_state['active_count']         = len(active_trades)
                scanner_state['deep_count']           = deep_count
                scanner_state['stats']                = stats
                scanner_state['new_entries']          = []
                scanner_state['macro']['btc_change_24h'] = btc_change_24h

            log.info(
                f"[WATCH RESCAN DONE] Watch:{len(new_watch_list)} | "
                f"신규Active:{len(new_actives)}"
            )

        except Exception as e:
            log.error(f"watch_rescan_loop 오류: {e}", exc_info=True)

# ═══════════════════════════════════════════════════════════════
# 가격 체크 루프 (5분)
# ═══════════════════════════════════════════════════════════════
def price_check_loop():
    log.info("price_check_loop 시작")
    while True:
        try:
            time.sleep(PRICE_CHECK_INTERVAL_MIN * 60)
            now_utc    = datetime.now(timezone.utc)
            watch_list = load_watch_list()
            if not watch_list:
                with _state_lock:
                    scanner_state['last_price_check_at'] = now_utc.isoformat()
                continue

            price_map = get_current_prices([w['ticker'] for w in watch_list])
            for item in watch_list:
                tk = item['ticker']
                if tk in price_map:
                    item.setdefault('current', {})['price']      = price_map[tk]
                    item.setdefault('current', {})['updated_at'] = now_utc.isoformat()

            save_watch_list(watch_list)
            with _state_lock:
                scanner_state['watch_list']          = watch_list
                scanner_state['last_price_check_at'] = now_utc.isoformat()

        except Exception as e:
            log.error(f"price_check_loop 오류: {e}")

# ═══════════════════════════════════════════════════════════════
# Active 모니터링 루프 (1분)
# ═══════════════════════════════════════════════════════════════
def active_monitor_loop():
    log.info("active_monitor_loop 시작")
    while True:
        try:
            time.sleep(ACTIVE_CHECK_INTERVAL_MIN * 60)
            now_utc       = datetime.now(timezone.utc)
            active_trades = load_active_trades()

            if active_trades:
                price_map = get_current_prices([t['ticker'] for t in active_trades])
                check_active_trades(price_map=price_map)
                active_trades = load_active_trades()

            with _state_lock:
                scanner_state['active_trades']        = active_trades
                scanner_state['active_count']         = len(active_trades)
                scanner_state['last_active_check_at'] = now_utc.isoformat()

        except Exception as e:
            log.error(f"active_monitor_loop 오류: {e}")

# ═══════════════════════════════════════════════════════════════
# 일일 요약 루프
# ═══════════════════════════════════════════════════════════════
def daily_summary_loop():
    global _last_summary_date
    log.info("daily_summary_loop 시작")
    while True:
        try:
            time.sleep(60)
            now_kst  = datetime.now(timezone(timedelta(hours=9)))
            today    = now_kst.date()
            if now_kst.hour == DAILY_SUMMARY_HOUR_KST and _last_summary_date != today:
                send_telegram(build_daily_summary())
                _last_summary_date = today
                log.info("[DAILY SUMMARY] 전송 완료")
        except Exception as e:
            log.error(f"daily_summary_loop 오류: {e}")

# ═══════════════════════════════════════════════════════════════
# 메인 루프
# ═══════════════════════════════════════════════════════════════
def manual_scan():
    _manual_scan_event.set()

def scanner_loop():
    log.info("scanner_loop 시작")
    run_scan()
    while True:
        try:
            triggered = _manual_scan_event.wait(timeout=SCAN_INTERVAL_MIN * 60)
            if triggered:
                _manual_scan_event.clear()
                log.info("[MANUAL SCAN 트리거]")
            run_scan()
        except Exception as e:
            log.error(f"scanner_loop 오류: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_scan()
