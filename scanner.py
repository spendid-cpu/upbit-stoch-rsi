"""
scanner.py v2.4.2
Upbit MTF 자동 스캐너

변경사항 (v2.4.2):
- manual_activate_watch: Watch → 수동 Active 즉시 전환
- activate_item: trade_type 'manual' 지원
- 텔레그램 수동 진입 알림 메시지 추가
"""

import os
import json
import time
import math
import logging
import threading
import traceback
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import mtf_setup

VERSION = 'v2.4.2'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('scanner')

# ── 환경변수 ─────────────────────────────────────────────────
SCAN_INTERVAL_MIN         = int(os.environ.get('SCAN_INTERVAL_MIN', 60))
WATCH_RESCAN_INTERVAL_MIN = int(os.environ.get('WATCH_RESCAN_INTERVAL_MIN', 15))
PRICE_CHECK_INTERVAL_MIN  = int(os.environ.get('PRICE_CHECK_INTERVAL_MIN', 5))
ACTIVE_CHECK_INTERVAL_MIN = int(os.environ.get('ACTIVE_CHECK_INTERVAL_MIN', 1))
DAILY_SUMMARY_HOUR_KST    = int(os.environ.get('DAILY_SUMMARY_HOUR_KST', 9))

REQUEST_DELAY         = float(os.environ.get('REQUEST_DELAY', 0.12))
MAX_WORKERS           = int(os.environ.get('MAX_WORKERS', 6))
CANDLE_COUNT          = int(os.environ.get('CANDLE_COUNT', 200))
MIN_TRADE_VALUE       = float(os.environ.get('MIN_TRADE_VALUE', 0))
SCORE_SURGE_THRESHOLD = int(os.environ.get('SCORE_SURGE_THRESHOLD', 10))

WATCH_LIST_FILE    = os.environ.get('WATCH_LIST_FILE',    'watch_list.json')
ACTIVE_TRADES_FILE = os.environ.get('ACTIVE_TRADES_FILE', 'active_trades.json')
TRADE_HISTORY_FILE = os.environ.get('TRADE_HISTORY_FILE', 'trade_history.json')

TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

TRADE_TP_PCT    = float(os.environ.get('TRADE_TP_PCT', 5.0))
TRADE_SL_PCT    = float(os.environ.get('TRADE_SL_PCT', 3.0))
TRADE_TIMEOUT_H = int(os.environ.get('TRADE_TIMEOUT_H', 48))

STABLE_COINS = {
    'KRW-USDT','KRW-USDC','KRW-DAI','KRW-TUSD','KRW-BUSD',
    'KRW-LUSD','KRW-FRAX','KRW-GUSD',
}

# ── 글로벌 상태 ──────────────────────────────────────────────
scanner_state = {
    'version':        VERSION,
    'status':         'idle',
    'last_scan_at':   None,
    'next_scan_at':   None,
    'scan_count':     0,
    'watch_count':    0,
    'active_count':   0,
    'deep_count':     0,
    'watch_list':     [],
    'active_trades':  [],
    'deep_watch':     [],
    'new_entries':    [],
    'removed_items':  [],
    'macro':          {},
    'stats':          {},
    'errors':         [],
    'failed_tickers': [],
}

_state_lock   = threading.Lock()
_clear_timer  = None
_scan_trigger = threading.Event()
KST = timezone(timedelta(hours=9))


# ── JSON I/O ─────────────────────────────────────────────────
def _load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f'_load_json {path}: {e}')
    return default


def _save_json(path: str, data):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f'_save_json {path}: {e}')


def load_watch_list():
    data = _load_json(WATCH_LIST_FILE, [])
    return [_migrate_watch_item(item) for item in data]


def _migrate_watch_item(item: dict) -> dict:
    """레거시 string dir_info → dict 변환"""
    for key in ('daily_dir_info', 'h4_dir_info', 'h1_dir_info'):
        val = item.get(key)
        if not isinstance(val, dict):
            dir_str = item.get(key.replace('_info', ''), '횡보')
            item[key] = {
                'direction':    dir_str if dir_str in ('상승','반등','횡보','하락') else '횡보',
                'strength':     0,
                'golden_cross': False,
            }
    return item


def save_watch_list(data):    _save_json(WATCH_LIST_FILE, data)
def load_active_trades():     return _load_json(ACTIVE_TRADES_FILE, [])
def save_active_trades(data): _save_json(ACTIVE_TRADES_FILE, data)
def load_trade_history():     return _load_json(TRADE_HISTORY_FILE, [])


def append_history(record: dict):
    history = load_trade_history()
    history.append(record)
    _save_json(TRADE_HISTORY_FILE, history)


# ── Upbit API ─────────────────────────────────────────────────
def _upbit_get(url: str, params: dict = None, retries: int = 3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
    return None


def get_all_krw_markets() -> list:
    data = _upbit_get('https://api.upbit.com/v1/market/all', {'isDetails': 'false'})
    return [
        m['market'] for m in (data or [])
        if m['market'].startswith('KRW-') and m['market'] not in STABLE_COINS
    ]


def get_closes(market: str, unit: str = 'days',
               count: int = 200, minutes: int = 60) -> list:
    if unit == 'days':
        url    = 'https://api.upbit.com/v1/candles/days'
        params = {'market': market, 'count': count}
    elif unit == 'weeks':
        url    = 'https://api.upbit.com/v1/candles/weeks'
        params = {'market': market, 'count': count}
    elif unit == 'minutes':
        url    = f'https://api.upbit.com/v1/candles/minutes/{minutes}'
        params = {'market': market, 'count': count}
    else:
        return []
    try:
        data = _upbit_get(url, params)
        if not data:
            return []
        return [c['trade_price'] for c in reversed(data)]
    except Exception as e:
        logger.debug(f'get_closes {market} {unit}: {e}')
        return []


def get_btc_closes(unit: str = 'weeks', count: int = 50) -> list:
    return get_closes('KRW-BTC', unit=unit, count=count)


def get_current_prices(markets: list) -> dict:
    result = {}
    for i in range(0, len(markets), 100):
        chunk = markets[i: i + 100]
        try:
            data = _upbit_get(
                'https://api.upbit.com/v1/ticker',
                {'markets': ','.join(chunk)}
            )
            for item in (data or []):
                result[item['market']] = {
                    'trade_price':     item.get('trade_price', 0),
                    'acc_trade_price': item.get('acc_trade_price_24h', 0),
                    'change_rate':     item.get('signed_change_rate', 0),
                }
        except Exception as e:
            logger.warning(f'get_current_prices chunk {i}: {e}')
        time.sleep(REQUEST_DELAY)
    return result


# ── 텔레그램 ─────────────────────────────────────────────────
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f'send_telegram: {e}')


def _grade_emoji(grade: str) -> str:
    return {'S': '🏆', 'A': '⭐', 'B': '📌', 'C': '📋'}.get(grade, '📋')


def build_active_msg(item: dict) -> str:
    g = item.get('grade', '?')
    return (
        f"{_grade_emoji(g)} <b>[Active 전환]</b> {item['ticker']}\n"
        f"등급: {g}  점수: {item.get('score', 0)}\n"
        f"진입가: {item.get('entry_price', 0):,.0f}원\n"
        f"TP: {item.get('tp_price', 0):,.0f}  SL: {item.get('sl_price', 0):,.0f}\n"
        f"일봉K: {item.get('daily_k', '-')}  "
        f"4hK: {item.get('h4_k', '-')}  "
        f"1hK: {item.get('h1_k', '-')}\n"
        f"진입강도: {item.get('entry_label', '?')}"
    )


def build_manual_active_msg(item: dict) -> str:
    g = item.get('grade', '?')
    return (
        f"👆 <b>[수동 진입]</b> {item['ticker']}\n"
        f"등급: {g}  점수: {item.get('score', 0)}\n"
        f"진입가: {item.get('entry_price', 0):,.0f}원\n"
        f"TP: {item.get('tp_price', 0):,.0f}  SL: {item.get('sl_price', 0):,.0f}\n"
        f"일봉K: {item.get('daily_k', '-')}  "
        f"4hK: {item.get('h4_k', '-')}  "
        f"1hK: {item.get('h1_k', '-')}\n"
        f"진입강도: {item.get('entry_label', '?')}\n"
        f"⏱ 타임아웃: {TRADE_TIMEOUT_H}시간"
    )


def build_close_msg(item: dict, reason: str, pnl: float) -> str:
    emoji = '✅' if pnl >= 0 else '❌'
    trade_type = item.get('trade_type', 'normal')
    type_label = {'manual': '👆수동', 'deep': '🔴DEEP', 'normal': '자동'}.get(trade_type, trade_type)
    return (
        f"{emoji} <b>[청산]</b> {item['ticker']}  ({reason} / {type_label})\n"
        f"PnL: {pnl:+.2f}%\n"
        f"진입가: {item.get('entry_price', 0):,.0f}  "
        f"청산가: {item.get('current_price', 0):,.0f}"
    )


def build_deep_msg(item: dict) -> str:
    return (
        f"🔴 <b>[DEEP Watch → Auto Active]</b> {item['ticker']}\n"
        f"DEEP등급: {item.get('deep_grade','?')}  점수: {item.get('deep_score',0)}\n"
        f"일봉K: {item.get('daily_k','-')}\n"
        f"BTC 24h: {item.get('btc_change',0):+.2f}%  "
        f"코인 24h: {item.get('coin_change',0):+.2f}%\n"
        f"상대강도: +{item.get('relative_strength',0):.2f}%p\n"
        f"진입가: {item.get('entry_price',0):,.0f}원\n"
        f"TP: {item.get('tp_price',0):,.0f}  SL: {item.get('sl_price',0):,.0f}"
    )


def build_surge_msg(ticker: str, old_score: int,
                    new_score: int, grade: str, entry_label: str) -> str:
    return (
        f"📈 <b>[점수 급등]</b> {ticker}\n"
        f"점수: {old_score} → {new_score} (+{new_score - old_score})\n"
        f"등급: {grade}  진입강도: {entry_label}"
    )


def build_daily_summary_msg(state: dict) -> str:
    stats   = state.get('stats', {})
    now_kst = datetime.now(KST).strftime('%Y-%m-%d %H:%M')
    wl      = state.get('watch_list', [])
    top5    = sorted(wl, key=lambda x: x.get('score', 0), reverse=True)[:5]
    top5_str = '\n'.join(
        f"  {i+1}. {t['ticker']} {t.get('grade','?')} {t.get('score',0)}점"
        for i, t in enumerate(top5)
    )
    return (
        f"📊 <b>[일일 요약]</b> {now_kst}\n"
        f"Watch: {state.get('watch_count',0)}개  "
        f"Active: {state.get('active_count',0)}개\n"
        f"승률: {stats.get('win_rate',0):.1f}%  "
        f"평균PnL: {stats.get('avg_pnl',0):+.2f}%\n"
        f"누적 거래: {stats.get('total_trades',0)}건\n"
        f"\nTop 5 Watch:\n{top5_str}"
    )


# ── 분석 ─────────────────────────────────────────────────────
def analyze_ticker(ticker: str,
                   snap_k: 'float | None' = None,
                   btc_change: 'float | None' = None) -> 'dict | None':
    try:
        time.sleep(REQUEST_DELAY)
        daily_closes = get_closes(ticker, 'days',    CANDLE_COUNT)
        h4_closes    = get_closes(ticker, 'minutes', CANDLE_COUNT, 240)
        h1_closes    = get_closes(ticker, 'minutes', CANDLE_COUNT, 60)

        if len(daily_closes) < 30 or len(h4_closes) < 30:
            return None

        daily_presets = mtf_setup.calc_all_presets(daily_closes)
        if not mtf_setup.evaluate_daily_gate(daily_presets):
            return None

        h4_presets = mtf_setup.calc_all_presets(h4_closes) if len(h4_closes) >= 30 else {}
        h1_presets = mtf_setup.calc_all_presets(h1_closes) if len(h1_closes) >= 30 else {}

        recent_vol = daily_closes[-1] if daily_closes else 1
        avg_vol    = sum(daily_closes[-20:]) / min(20, len(daily_closes)) if daily_closes else 1
        vol_ratio  = recent_vol / avg_vol if avg_vol > 0 else 1.0

        score_result = mtf_setup.calc_watch_score(
            daily_presets, h4_presets, h1_presets,
            vol_ratio=vol_ratio, snap_k=snap_k,
        )

        # 주봉 K (DEEP용)
        weekly_closes  = get_closes(ticker, 'weeks', 60)
        weekly_presets = mtf_setup.calc_all_presets(weekly_closes) if len(weekly_closes) >= 20 else {}
        weekly_k       = mtf_setup._safe_k(weekly_presets) if weekly_presets else None

        # DEEP 조건
        deep_info = None
        dk = score_result.get('daily_k')
        if dk is not None and dk <= mtf_setup.DEEP_K_THRESHOLD and btc_change is not None:
            coin_data   = get_current_prices([ticker])
            coin_change = coin_data.get(ticker, {}).get('change_rate', 0) * 100
            if mtf_setup.evaluate_deep_condition(dk, btc_change, coin_change):
                k_series       = daily_presets.get('short', {}).get('k_series', [])
                days_at_bottom = sum(
                    1 for kv in reversed(k_series)
                    if kv is not None and kv <= mtf_setup.DEEP_K_THRESHOLD
                )
                deep_result = mtf_setup.calc_deep_score(
                    daily_k=dk, btc_change=btc_change,
                    coin_change=coin_change,
                    days_at_bottom=days_at_bottom,
                    vol_ratio=vol_ratio, weekly_k=weekly_k,
                )
                deep_grade = mtf_setup.get_deep_grade(deep_result['deep_score'])
                deep_info  = {
                    'deep_score':        deep_result['deep_score'],
                    'deep_grade':        deep_grade,
                    'deep_breakdown':    deep_result['breakdown'],
                    'btc_change':        round(btc_change, 2),
                    'coin_change':       round(coin_change, 2),
                    'relative_strength': round(coin_change - btc_change, 2),
                    'days_at_bottom':    days_at_bottom,
                    'weekly_k':          weekly_k,
                }

        return {
            'ticker':         ticker,
            'score':          score_result['score'],
            'grade':          score_result['grade'],
            'breakdown':      score_result['breakdown'],
            'daily_dir':      score_result['daily_dir'],
            'h4_dir':         score_result['h4_dir'],
            'h1_dir':         score_result['h1_dir'],
            'daily_dir_info': score_result.get('daily_dir_info', {}),
            'h4_dir_info':    score_result.get('h4_dir_info', {}),
            'h1_dir_info':    score_result.get('h1_dir_info', {}),
            'entry_level':    score_result['entry_level'],
            'entry_label':    score_result['entry_label'],
            'daily_k':        score_result['daily_k'],
            'h4_k':           score_result['h4_k'],
            'h1_k':           score_result['h1_k'],
            'weekly_k':       weekly_k,
            'vol_ratio':      round(vol_ratio, 2),
            'current_price':  daily_closes[-1] if daily_closes else 0,
            'deep':           deep_info,
        }

    except Exception as e:
        logger.debug(f'analyze_ticker {ticker}: {e}')
        return None


# ── 통계 ─────────────────────────────────────────────────────
def calc_stats() -> dict:
    history = load_trade_history()
    if not history:
        return {
            'total_trades': 0, 'win_count': 0, 'loss_count': 0,
            'win_rate': 0.0, 'avg_pnl': 0.0, 'grade_stats': {},
        }
    wins    = [h for h in history if h.get('pnl_pct', 0) >= 0]
    all_pnl = [h.get('pnl_pct', 0) for h in history]

    grade_stats = {}
    for h in history:
        g = h.get('grade', '?')
        if g not in grade_stats:
            grade_stats[g] = {'total': 0, 'wins': 0, 'pnl_list': []}
        grade_stats[g]['total'] += 1
        if h.get('pnl_pct', 0) >= 0:
            grade_stats[g]['wins'] += 1
        grade_stats[g]['pnl_list'].append(h.get('pnl_pct', 0))

    for g in grade_stats:
        pl = grade_stats[g]['pnl_list']
        grade_stats[g]['avg_pnl']  = round(sum(pl) / len(pl), 2) if pl else 0.0
        grade_stats[g]['win_rate'] = round(
            grade_stats[g]['wins'] / grade_stats[g]['total'] * 100, 1
        )
        del grade_stats[g]['pnl_list']

    return {
        'total_trades': len(history),
        'win_count':    len(wins),
        'loss_count':   len(history) - len(wins),
        'win_rate':     round(len(wins) / len(history) * 100, 1),
        'avg_pnl':      round(sum(all_pnl) / len(all_pnl), 2),
        'grade_stats':  grade_stats,
    }


# ── Watch 아이템 빌더 ────────────────────────────────────────
def _build_watch_item(result: dict, now_utc: datetime,
                      manual: bool = False) -> dict:
    expiry_days = mtf_setup.get_expiry_days(result['grade'])
    expire_at   = (now_utc + timedelta(days=expiry_days)).isoformat()
    return {
        'ticker':         result['ticker'],
        'score':          result['score'],
        'grade':          result['grade'],
        'breakdown':      result['breakdown'],
        'daily_dir':      result['daily_dir'],
        'h4_dir':         result['h4_dir'],
        'h1_dir':         result['h1_dir'],
        'daily_dir_info': result.get('daily_dir_info', {}),
        'h4_dir_info':    result.get('h4_dir_info', {}),
        'h1_dir_info':    result.get('h1_dir_info', {}),
        'entry_level':    result['entry_level'],
        'entry_label':    result['entry_label'],
        'daily_k':        result['daily_k'],
        'h4_k':           result['h4_k'],
        'h1_k':           result['h1_k'],
        'weekly_k':       result.get('weekly_k'),
        'vol_ratio':      result['vol_ratio'],
        'current_price':  result['current_price'],
        'snapshot':       {'daily_k': result['daily_k'], 'score': result['score']},
        'score_history':  [result['score']],
        'registered_at':  now_utc.isoformat(),
        'updated_at':     now_utc.isoformat(),
        'expire_at':      expire_at,
        'manual':         manual,
        'type':           'normal',
    }


def _build_deep_watch_item(result: dict, deep_info: dict,
                           now_utc: datetime) -> dict:
    item = _build_watch_item(result, now_utc)
    item.update({
        'type':              'deep',
        'deep_score':        deep_info['deep_score'],
        'deep_grade':        deep_info['deep_grade'],
        'deep_breakdown':    deep_info['deep_breakdown'],
        'btc_change':        deep_info['btc_change'],
        'coin_change':       deep_info['coin_change'],
        'relative_strength': deep_info['relative_strength'],
        'days_at_bottom':    deep_info['days_at_bottom'],
        'weekly_k':          deep_info.get('weekly_k'),
        'expire_at':         None,
    })
    return item


# ── Active 전환 ──────────────────────────────────────────────
def activate_item(watch_item: dict, price_map: dict,
                  trade_type: str = 'normal') -> 'dict | None':
    ticker     = watch_item['ticker']
    price_info = price_map.get(ticker)
    if not price_info:
        return None

    entry_price = price_info['trade_price']
    tp_price    = entry_price * (1 + TRADE_TP_PCT / 100)
    sl_price    = entry_price * (1 - TRADE_SL_PCT / 100)
    now_utc     = datetime.now(timezone.utc)
    timeout_at  = (now_utc + timedelta(hours=TRADE_TIMEOUT_H)).isoformat()

    active_item = {
        **watch_item,
        'entry_price':   entry_price,
        'tp_price':      tp_price,
        'sl_price':      sl_price,
        'current_price': entry_price,
        'pnl_pct':       0.0,
        'activated_at':  now_utc.isoformat(),
        'timeout_at':    timeout_at,
        'status':        'active',
        'trade_type':    trade_type,
    }

    append_history({
        'type':        'activate',
        'ticker':      ticker,
        'trade_type':  trade_type,
        'score':       watch_item.get('score', 0),
        'grade':       watch_item.get('grade', '?'),
        'entry_price': entry_price,
        'activated_at': now_utc.isoformat(),
    })
    return active_item


# ── Active 모니터링 ──────────────────────────────────────────
def check_active_trades(price_map: dict) -> list:
    active_trades = load_active_trades()
    now_utc       = datetime.now(timezone.utc)
    remaining     = []

    for item in active_trades:
        ticker = item['ticker']
        p      = price_map.get(ticker)
        if p:
            item['current_price'] = p['trade_price']
            ep = item.get('entry_price', 0)
            item['pnl_pct'] = round(
                (item['current_price'] - ep) / ep * 100, 2
            ) if ep > 0 else 0.0

        current = item.get('current_price', 0)
        tp      = item.get('tp_price', 0)
        sl      = item.get('sl_price', 0)
        timeout = item.get('timeout_at')

        close_reason = None
        if current >= tp:
            close_reason = 'TP'
        elif current <= sl:
            close_reason = 'SL'
        elif timeout:
            try:
                if now_utc >= datetime.fromisoformat(timeout):
                    close_reason = 'TIMEOUT'
            except Exception:
                pass

        if close_reason:
            pnl = item.get('pnl_pct', 0.0)
            append_history({
                'type':         'close',
                'ticker':       ticker,
                'trade_type':   item.get('trade_type', 'normal'),
                'close_reason': close_reason,
                'grade':        item.get('grade', '?'),
                'entry_price':  item.get('entry_price', 0),
                'close_price':  current,
                'pnl_pct':      pnl,
                'activated_at': item.get('activated_at'),
                'closed_at':    now_utc.isoformat(),
            })
            send_telegram(build_close_msg(item, close_reason, pnl))
        else:
            remaining.append(item)

    if len(remaining) < len(active_trades):
        save_active_trades(remaining)

    return remaining


# ── 수동 Watch 관리 ──────────────────────────────────────────
def add_manual_watch(ticker: str) -> dict:
    ticker = ticker.upper()
    if not ticker.startswith('KRW-'):
        ticker = f'KRW-{ticker}'

    watch_list = load_watch_list()
    for item in watch_list:
        if item['ticker'] == ticker:
            return {'success': False, 'message': f'{ticker} 이미 Watch 목록에 있습니다.'}

    result = analyze_ticker(ticker)
    if not result:
        return {'success': False, 'message': f'{ticker} 분석 실패 (일봉K>20 또는 데이터 부족)'}

    now_utc  = datetime.now(timezone.utc)
    new_item = _build_watch_item(result, now_utc, manual=True)
    watch_list.append(new_item)
    save_watch_list(watch_list)

    with _state_lock:
        scanner_state['watch_list']  = watch_list
        scanner_state['watch_count'] = len(watch_list)
        scanner_state['new_entries'] = [new_item]

    _reset_clear_timer()
    return {'success': True, 'message': f'{ticker} Watch 추가 완료', 'item': new_item}


def remove_watch(ticker: str) -> dict:
    ticker     = ticker.upper()
    watch_list = load_watch_list()
    before     = len(watch_list)
    watch_list = [w for w in watch_list if w['ticker'] != ticker]
    if len(watch_list) == before:
        return {'success': False, 'message': f'{ticker} 없음'}
    save_watch_list(watch_list)
    with _state_lock:
        scanner_state['watch_list']  = watch_list
        scanner_state['watch_count'] = len(watch_list)
    return {'success': True, 'message': f'{ticker} 제거 완료'}


def manual_close_trade(ticker: str) -> dict:
    ticker        = ticker.upper()
    active_trades = load_active_trades()
    target        = None
    remaining     = []
    for item in active_trades:
        if item['ticker'] == ticker:
            target = item
        else:
            remaining.append(item)
    if not target:
        return {'success': False, 'message': f'{ticker} Active 없음'}

    now_utc = datetime.now(timezone.utc)
    pnl     = target.get('pnl_pct', 0.0)
    append_history({
        'type':         'close',
        'ticker':       ticker,
        'trade_type':   target.get('trade_type', 'normal'),
        'close_reason': 'MANUAL',
        'grade':        target.get('grade', '?'),
        'entry_price':  target.get('entry_price', 0),
        'close_price':  target.get('current_price', 0),
        'pnl_pct':      pnl,
        'activated_at': target.get('activated_at'),
        'closed_at':    now_utc.isoformat(),
    })
    send_telegram(build_close_msg(target, 'MANUAL', pnl))
    save_active_trades(remaining)
    with _state_lock:
        scanner_state['active_trades'] = remaining
        scanner_state['active_count']  = len(remaining)
    return {'success': True, 'message': f'{ticker} 수동 청산 완료', 'pnl_pct': pnl}


def manual_activate_watch(ticker: str) -> dict:
    """Watch → 수동 즉시 Active 전환"""
    ticker = ticker.upper()
    if not ticker.startswith('KRW-'):
        ticker = f'KRW-{ticker}'

    watch_list    = load_watch_list()
    active_trades = load_active_trades()

    # 이미 Active 확인
    for a in active_trades:
        if a['ticker'] == ticker:
            return {'success': False, 'message': f'{ticker} 이미 Active 상태입니다.'}

    # Watch에서 찾기
    target          = None
    remaining_watch = []
    for item in watch_list:
        if item['ticker'] == ticker:
            target = item
        else:
            remaining_watch.append(item)

    if not target:
        return {'success': False, 'message': f'{ticker} Watch 목록에 없습니다.'}

    # 현재가 조회
    price_map = get_current_prices([ticker])
    if not price_map.get(ticker):
        return {'success': False, 'message': f'{ticker} 현재가 조회 실패'}

    # Active 전환
    active_item = activate_item(target, price_map, trade_type='manual')
    if not active_item:
        return {'success': False, 'message': f'{ticker} Active 전환 실패'}

    active_trades.append(active_item)
    save_watch_list(remaining_watch)
    save_active_trades(active_trades)

    # 텔레그램
    send_telegram(build_manual_active_msg(active_item))

    # 상태 업데이트
    with _state_lock:
        scanner_state['watch_list']    = remaining_watch
        scanner_state['watch_count']   = len(remaining_watch)
        scanner_state['active_trades'] = active_trades
        scanner_state['active_count']  = len(active_trades)
        scanner_state['stats']         = calc_stats()

    logger.info(f'수동 Active 전환: {ticker} '
                f'진입가={active_item["entry_price"]:,.0f} '
                f'TP={active_item["tp_price"]:,.0f} '
                f'SL={active_item["sl_price"]:,.0f}')

    return {
        'success': True,
        'message': f'{ticker} 수동 Active 전환 완료',
        'item':    active_item,
    }


# ── 타이머 헬퍼 ──────────────────────────────────────────────
def _clear_entries():
    with _state_lock:
        scanner_state['new_entries']   = []
        scanner_state['removed_items'] = []


def _reset_clear_timer():
    global _clear_timer
    if _clear_timer is not None:
        _clear_timer.cancel()
    _clear_timer = threading.Timer(300, _clear_entries)
    _clear_timer.daemon = True
    _clear_timer.start()


# ── 메인 스캔 ────────────────────────────────────────────────
def run_scan():
    with _state_lock:
        scanner_state['status'] = 'scanning'

    now_utc = datetime.now(timezone.utc)
    logger.info(f'[{VERSION}] run_scan 시작 {now_utc.isoformat()}')

    try:
        btc_weekly = get_btc_closes('weeks', 50)
        macro      = mtf_setup.evaluate_macro_filter(btc_weekly)
        btc_daily  = get_btc_closes('days', 2)
        btc_change = None
        if len(btc_daily) >= 2:
            btc_change = (btc_daily[-1] - btc_daily[-2]) / btc_daily[-2] * 100

        with _state_lock:
            scanner_state['macro'] = macro

        existing_watch  = load_watch_list()
        existing_active = load_active_trades()
        existing_tickers = {w['ticker'] for w in existing_watch}
        active_tickers   = {a['ticker'] for a in existing_active}

        with _state_lock:
            failed = list(scanner_state.get('failed_tickers', []))

        all_markets = get_all_krw_markets()
        logger.info(f'마켓 수: {len(all_markets)}')

        results    = {}
        new_failed = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(
                    analyze_ticker,
                    ticker,
                    next((w['snapshot'].get('daily_k')
                          for w in existing_watch if w['ticker'] == ticker), None),
                    btc_change,
                ): ticker
                for ticker in all_markets
                if ticker not in active_tickers
            }
            for future in as_completed(future_map):
                ticker = future_map[future]
                try:
                    res = future.result()
                    if res:
                        results[ticker] = res
                except Exception as e:
                    new_failed.append(ticker)
                    logger.debug(f'future {ticker}: {e}')

        all_tickers = list(results.keys()) + [a['ticker'] for a in existing_active]
        price_map   = get_current_prices(list(set(all_tickers)))

        new_watch_list = []
        newly_added    = []
        removed_items  = []

        # 기존 Watch 업데이트
        for item in existing_watch:
            ticker = item['ticker']
            if ticker in results:
                res       = results[ticker]
                old_score = item.get('score', 0)
                new_score = res['score']

                hist = item.get('score_history', [])
                hist.append(new_score)
                if len(hist) > 50:
                    hist = hist[-50:]

                item.update({
                    'score':          new_score,
                    'grade':          res['grade'],
                    'breakdown':      res['breakdown'],
                    'daily_dir':      res['daily_dir'],
                    'h4_dir':         res['h4_dir'],
                    'h1_dir':         res['h1_dir'],
                    'daily_dir_info': res.get('daily_dir_info', {}),
                    'h4_dir_info':    res.get('h4_dir_info', {}),
                    'h1_dir_info':    res.get('h1_dir_info', {}),
                    'entry_level':    res['entry_level'],
                    'entry_label':    res['entry_label'],
                    'daily_k':        res['daily_k'],
                    'h4_k':           res['h4_k'],
                    'h1_k':           res['h1_k'],
                    'weekly_k':       res.get('weekly_k'),
                    'vol_ratio':      res['vol_ratio'],
                    'current_price':  price_map.get(ticker, {}).get(
                        'trade_price', item.get('current_price', 0)),
                    'score_history':  hist,
                    'updated_at':     now_utc.isoformat(),
                })

                if new_score - old_score >= SCORE_SURGE_THRESHOLD:
                    send_telegram(build_surge_msg(
                        ticker, old_score, new_score,
                        res['grade'], res['entry_label']
                    ))

                expire_at = item.get('expire_at')
                if expire_at:
                    try:
                        if now_utc >= datetime.fromisoformat(expire_at):
                            removed_items.append(item)
                            continue
                    except Exception:
                        pass

                new_watch_list.append(item)
            else:
                new_watch_list.append(item)

        # 신규 Watch 추가
        for ticker, res in results.items():
            if ticker in existing_tickers or ticker in active_tickers:
                continue

            deep_info = res.get('deep')
            if deep_info and deep_info['deep_grade'] in ('DEEP-S', 'DEEP-A'):
                watch_item  = _build_deep_watch_item(res, deep_info, now_utc)
                active_item = activate_item(watch_item, price_map, trade_type='deep')
                if active_item:
                    existing_active.append(active_item)
                    send_telegram(build_deep_msg({**watch_item, **active_item}))
            else:
                new_item = _build_watch_item(res, now_utc)
                new_item['current_price'] = price_map.get(ticker, {}).get(
                    'trade_price', res['current_price'])
                new_watch_list.append(new_item)
                newly_added.append(new_item)

        new_watch_list.sort(key=lambda x: x.get('score', 0), reverse=True)
        save_active_trades(existing_active)
        save_watch_list(new_watch_list)
        stats = calc_stats()

        with _state_lock:
            scanner_state.update({
                'status':         'idle',
                'last_scan_at':   now_utc.isoformat(),
                'scan_count':     scanner_state.get('scan_count', 0) + 1,
                'watch_count':    len(new_watch_list),
                'active_count':   len(existing_active),
                'deep_count':     sum(1 for a in existing_active
                                      if a.get('trade_type') == 'deep'),
                'watch_list':     new_watch_list,
                'active_trades':  existing_active,
                'new_entries':    newly_added,
                'removed_items':  removed_items,
                'stats':          stats,
                'failed_tickers': new_failed,
                'errors':         [],
            })

        _reset_clear_timer()
        logger.info(
            f'스캔 완료: Watch={len(new_watch_list)} '
            f'Active={len(existing_active)} '
            f'신규={len(newly_added)} 제거={len(removed_items)}'
        )

    except Exception as e:
        logger.error(f'run_scan 오류: {e}\n{traceback.format_exc()}')
        with _state_lock:
            scanner_state['status'] = 'error'
            scanner_state['errors'] = [str(e)]


# ── Watch 재스캔 루프 ─────────────────────────────────────────
def watch_rescan_loop():
    while True:
        time.sleep(WATCH_RESCAN_INTERVAL_MIN * 60)
        try:
            with _state_lock:
                watch_list    = list(scanner_state.get('watch_list', []))
                active_trades = list(scanner_state.get('active_trades', []))

            if not watch_list:
                continue

            active_tickers = {a['ticker'] for a in active_trades}
            now_utc        = datetime.now(timezone.utc)
            price_map      = get_current_prices([w['ticker'] for w in watch_list])
            updated        = []
            new_actives    = []

            for item in watch_list:
                ticker = item['ticker']
                if ticker in active_tickers:
                    continue

                snap_k = item.get('snapshot', {}).get('daily_k')
                res    = analyze_ticker(ticker, snap_k=snap_k)
                if not res:
                    updated.append(item)
                    continue

                old_score = item.get('score', 0)
                new_score = res['score']
                hist      = item.get('score_history', [])
                hist.append(new_score)
                if len(hist) > 50:
                    hist = hist[-50:]

                item.update({
                    'score':          new_score,
                    'grade':          res['grade'],
                    'breakdown':      res['breakdown'],
                    'daily_dir':      res['daily_dir'],
                    'h4_dir':         res['h4_dir'],
                    'h1_dir':         res['h1_dir'],
                    'daily_dir_info': res.get('daily_dir_info', {}),
                    'h4_dir_info':    res.get('h4_dir_info', {}),
                    'h1_dir_info':    res.get('h1_dir_info', {}),
                    'entry_level':    res['entry_level'],
                    'entry_label':    res['entry_label'],
                    'daily_k':        res['daily_k'],
                    'h4_k':           res['h4_k'],
                    'h1_k':           res['h1_k'],
                    'current_price':  price_map.get(ticker, {}).get(
                        'trade_price', item.get('current_price', 0)),
                    'score_history':  hist,
                    'updated_at':     now_utc.isoformat(),
                })

                if res['grade'] in ('S', 'A'):
                    active_item = activate_item(item, price_map)
                    if active_item:
                        new_actives.append(active_item)
                        send_telegram(build_active_msg(active_item))
                        continue

                if new_score - old_score >= SCORE_SURGE_THRESHOLD:
                    send_telegram(build_surge_msg(
                        ticker, old_score, new_score,
                        res['grade'], res['entry_label']
                    ))

                updated.append(item)

            updated.sort(key=lambda x: x.get('score', 0), reverse=True)
            all_active = active_trades + new_actives
            save_watch_list(updated)
            save_active_trades(all_active)

            with _state_lock:
                scanner_state['watch_list']    = updated
                scanner_state['active_trades'] = all_active
                scanner_state['watch_count']   = len(updated)
                scanner_state['active_count']  = len(all_active)
                scanner_state['stats']         = calc_stats()

        except Exception as e:
            logger.error(f'watch_rescan_loop: {e}')


# ── 가격 체크 루프 ────────────────────────────────────────────
def price_check_loop():
    while True:
        time.sleep(PRICE_CHECK_INTERVAL_MIN * 60)
        try:
            with _state_lock:
                watch_list    = list(scanner_state.get('watch_list', []))
                active_trades = list(scanner_state.get('active_trades', []))

            tickers   = list({w['ticker'] for w in watch_list} |
                             {a['ticker'] for a in active_trades})
            price_map = get_current_prices(tickers)

            for item in watch_list:
                p = price_map.get(item['ticker'])
                if p:
                    item['current_price'] = p['trade_price']

            for item in active_trades:
                p = price_map.get(item['ticker'])
                if p:
                    item['current_price'] = p['trade_price']
                    ep = item.get('entry_price', 0)
                    if ep > 0:
                        item['pnl_pct'] = round(
                            (item['current_price'] - ep) / ep * 100, 2)

            with _state_lock:
                scanner_state['watch_list']    = watch_list
                scanner_state['active_trades'] = active_trades

        except Exception as e:
            logger.error(f'price_check_loop: {e}')


# ── Active 모니터링 루프 ──────────────────────────────────────
def active_monitor_loop():
    while True:
        time.sleep(ACTIVE_CHECK_INTERVAL_MIN * 60)
        try:
            with _state_lock:
                active_trades = list(scanner_state.get('active_trades', []))
            if not active_trades:
                continue

            price_map = get_current_prices([a['ticker'] for a in active_trades])
            remaining = check_active_trades(price_map)

            with _state_lock:
                scanner_state['active_trades'] = remaining
                scanner_state['active_count']  = len(remaining)
                scanner_state['deep_count']    = sum(
                    1 for a in remaining if a.get('trade_type') == 'deep')
                scanner_state['stats']         = calc_stats()

        except Exception as e:
            logger.error(f'active_monitor_loop: {e}')


# ── 일일 요약 루프 ────────────────────────────────────────────
def daily_summary_loop():
    sent_today = None
    while True:
        time.sleep(60)
        try:
            now_kst = datetime.now(KST)
            today   = now_kst.date()
            if now_kst.hour == DAILY_SUMMARY_HOUR_KST and sent_today != today:
                with _state_lock:
                    state = dict(scanner_state)
                send_telegram(build_daily_summary_msg(state))
                sent_today = today
        except Exception as e:
            logger.error(f'daily_summary_loop: {e}')


# ── 스캐너 메인 루프 ──────────────────────────────────────────
def scanner_loop():
    logger.info(f'scanner_loop 시작 (간격 {SCAN_INTERVAL_MIN}분)')
    run_scan()
    while True:
        triggered = _scan_trigger.wait(timeout=SCAN_INTERVAL_MIN * 60)
        if triggered:
            _scan_trigger.clear()
        run_scan()


def manual_scan():
    _scan_trigger.set()


# ── 실행 진입점 ───────────────────────────────────────────────
if __name__ == '__main__':
    threads = [
        threading.Thread(target=scanner_loop,        daemon=True, name='scanner_loop'),
        threading.Thread(target=watch_rescan_loop,   daemon=True, name='watch_rescan'),
        threading.Thread(target=price_check_loop,    daemon=True, name='price_check'),
        threading.Thread(target=active_monitor_loop, daemon=True, name='active_monitor'),
        threading.Thread(target=daily_summary_loop,  daemon=True, name='daily_summary'),
    ]
    for t in threads:
        t.start()
    logger.info(f'[{VERSION}] 모든 루프 시작됨')
    for t in threads:
        t.join()
