# -*- coding: utf-8 -*-
"""
scanner.py — Upbit MTF 자동 스캐너 (v2.1)
변경사항:
  - 3단계 루프 구조
    * 전체 스캔       : 60분 (SCAN_INTERVAL_MIN)
    * Watch 재스캔    : 15분 (WATCH_RESCAN_INTERVAL_MIN)
    * 가격 체크       :  5분 (PRICE_CHECK_INTERVAL_MIN)
    * Active 모니터링 :  1분 (ACTIVE_CHECK_INTERVAL_MIN)
"""

import os, time, json, logging, threading
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import mtf_setup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger(__name__)

# ── 환경변수 ──────────────────────────────────────
SCAN_INTERVAL_MIN          = int(float(os.getenv("SCAN_INTERVAL_MIN",          "60")))
WATCH_RESCAN_INTERVAL_MIN  = int(float(os.getenv("WATCH_RESCAN_INTERVAL_MIN",  "15")))
PRICE_CHECK_INTERVAL_MIN   = int(float(os.getenv("PRICE_CHECK_INTERVAL_MIN",   "5")))
ACTIVE_CHECK_INTERVAL_MIN  = int(float(os.getenv("ACTIVE_CHECK_INTERVAL_MIN",  "1")))

MIN_TRADE_VALUE_KRW = float(os.getenv("MIN_TRADE_VALUE_KRW", "0"))
REQUEST_DELAY       = float(os.getenv("REQUEST_DELAY",       "0.12"))
MAX_WORKERS         = int(os.getenv("MAX_WORKERS",           "6"))
CANDLE_COUNT        = int(os.getenv("CANDLE_COUNT",          "200"))

WATCH_LIST_FILE    = os.getenv("WATCH_LIST_FILE",    "watch_list.json")
ACTIVE_TRADES_FILE = os.getenv("ACTIVE_TRADES_FILE", "active_trades.json")
TRADE_HISTORY_FILE = os.getenv("TRADE_HISTORY_FILE", "trade_history.json")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")

TRADE_TP_PCT    = float(os.getenv("TRADE_TP_PCT",    "5.0"))
TRADE_SL_PCT    = float(os.getenv("TRADE_SL_PCT",    "3.0"))
TRADE_TIMEOUT_H = float(os.getenv("TRADE_TIMEOUT_H", "48.0"))

# ── 스테이블코인 제외 ─────────────────────────────
STABLE_COINS = {
    "KRW-USDT", "KRW-USDC", "KRW-BUSD", "KRW-DAI",
    "KRW-TUSD", "KRW-USDP", "KRW-USDD", "KRW-USDS",
    "KRW-USDE", "KRW-USD1", "KRW-XAUT", "KRW-FF",
}

# ── 공유 상태 ─────────────────────────────────────
_state_lock        = threading.Lock()
_manual_scan_event = threading.Event()

scanner_state = {
    'status':                  'idle',
    'last_scan_at':            None,
    'next_scan_at':            None,
    'last_watch_rescan_at':    None,
    'last_price_check_at':     None,
    'last_active_check_at':    None,
    'total_scans':             0,
    'total_scanned':           0,
    'watch_list':              [],
    'active_trades':           [],
    'new_entries':             [],
    'removed_items':           [],
    'new_actives':             [],
    'macro':                   {},
    'stats':                   {},
    'error':                   None,
}


# ════════════════════════════════════════════════
# 파일 I/O
# ════════════════════════════════════════════════

def _load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"load {path} failed: {e}")
    return default

def _save_json(path, data):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"save {path} failed: {e}")

def load_watch_list():
    data = _load_json(WATCH_LIST_FILE, [])
    migrated = []
    for item in data:
        if 'snapshot' not in item:
            now_str = item.get('registered_at', datetime.now(timezone.utc).isoformat())
            ep  = item.get('entry_price') or item.get('current', {}).get('price', 0) or 0
            dk  = item.get('daily_short_k') or item.get('current', {}).get('daily_k')
            h4k = item.get('h4_short_k')    or item.get('current', {}).get('h4_k')
            h1k = item.get('h1_short_k')    or item.get('current', {}).get('h1_k')
            old_score = item.get('score', 0)
            old_grade = item.get('grade', 'C')
            item = {
                'ticker': item['ticker'],
                'manual': item.get('manual', False),
                'status': 'watch',
                'snapshot': {
                    'registered_at': now_str,
                    'entry_price':   ep,
                    'score':         old_score,
                    'grade':         old_grade,
                    'daily_k':       dk,
                    'h4_k':          h4k,
                    'h1_k':          h1k,
                    'btc_price':     0,
                },
                'current': {
                    'score':      old_score,
                    'grade':      old_grade,
                    'daily_k':    dk,
                    'h4_k':       h4k,
                    'h1_k':       h1k,
                    'vol_ratio':  1.0,
                    'price':      ep,
                    'change_pct': 0,
                },
            }
        migrated.append(item)
    return migrated

def save_watch_list(d):    _save_json(WATCH_LIST_FILE,    d)
def load_active_trades():  return _load_json(ACTIVE_TRADES_FILE, [])
def save_active_trades(d): _save_json(ACTIVE_TRADES_FILE, d)
def load_trade_history():  return _load_json(TRADE_HISTORY_FILE, [])

def append_history(record):
    history = load_trade_history()
    history.append(record)
    if len(history) > 2000:
        history = history[-2000:]
    _save_json(TRADE_HISTORY_FILE, history)


# ════════════════════════════════════════════════
# Upbit API
# ════════════════════════════════════════════════

def _get(url, params=None, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(1 + i)

def get_all_krw_markets():
    data = _get("https://api.upbit.com/v1/market/all", {"isDetails": "true"})
    markets = []
    for m in data:
        code = m['market']
        if not code.startswith('KRW-'):          continue
        if code in STABLE_COINS:                 continue
        if code == 'KRW-BTC':                    continue
        if m.get('market_warning') == 'CAUTION': continue
        markets.append(code)
    log.info(f"스캔 대상: {len(markets)}종목")
    return markets

def get_closes(ticker, interval, count=None):
    count = count or CANDLE_COUNT
    url_map = {
        'days':       'https://api.upbit.com/v1/candles/days',
        'weeks':      'https://api.upbit.com/v1/candles/weeks',
        'minutes240': 'https://api.upbit.com/v1/candles/minutes/240',
        'minutes60':  'https://api.upbit.com/v1/candles/minutes/60',
    }
    data    = _get(url_map[interval], {'market': ticker, 'count': count})
    closes  = [c['trade_price']             for c in reversed(data)]
    volumes = [c['candle_acc_trade_volume'] for c in reversed(data)]
    return closes, volumes

def get_btc_closes():
    daily,  _ = get_closes('KRW-BTC', 'days',  100)
    weekly, _ = get_closes('KRW-BTC', 'weeks', 30)
    return daily, weekly

def get_current_prices(tickers):
    result = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i+100]
        try:
            data = _get(
                "https://api.upbit.com/v1/ticker",
                {"markets": ",".join(chunk)}
            )
            for d in data:
                result[d['market']] = {
                    'price':      d['trade_price'],
                    'change_pct': d['signed_change_rate'] * 100,
                }
        except Exception as e:
            log.warning(f"price fetch error: {e}")
        time.sleep(REQUEST_DELAY)
    return result


# ════════════════════════════════════════════════
# Telegram
# ════════════════════════════════════════════════

def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
            timeout=10
        )
    except Exception as e:
        log.warning(f"telegram error: {e}")

GRADE_EMOJI = {'S': '🔴', 'A': '🟠', 'B': '🟡', 'C': '⚪'}

def build_active_msg(item, prev_grade):
    g      = item.get('grade', '?')
    ticker = item['ticker'].replace('KRW-', '')
    score  = item.get('entry_score', 0)
    ep     = item.get('entry_price', 0)
    return (
        f"{GRADE_EMOJI.get(g,'')} [Active 전환] {ticker}\n"
        f"{prev_grade}등급 → {g}등급 ({score}점)\n"
        f"진입가: {ep:,.0f}\n"
        f"TP: +{TRADE_TP_PCT}% | SL: -{TRADE_SL_PCT}% | "
        f"Timeout: {TRADE_TIMEOUT_H:.0f}h"
    )

def build_close_msg(record):
    result_map = {
        'tp':      '✅ TP 청산',
        'sl':      '❌ SL 청산',
        'timeout': '⏱ Timeout 청산',
        'manual':  '🖐 수동 청산',
        'expired': '🗑 Watch 만료',
    }
    label   = result_map.get(record['result'], record['result'])
    ticker  = record['ticker'].replace('KRW-', '')
    pnl     = record.get('pnl_pct')
    pnl_str = f"{pnl:+.2f}%" if pnl is not None else '-'
    return (
        f"{label} {ticker}\n"
        f"등급: {record.get('grade','?')} | 점수: {record.get('score','?')}\n"
        f"수익률: {pnl_str} | "
        f"보유: {record.get('watch_hours', record.get('hours_held', 0)):.1f}h"
    )


# ════════════════════════════════════════════════
# 종목 분석
# ════════════════════════════════════════════════

def analyze_ticker(ticker):
    try:
        time.sleep(REQUEST_DELAY)
        daily_closes, daily_vols = get_closes(ticker, 'days')
        if len(daily_closes) < 60:
            return None

        daily_presets = mtf_setup.calc_all_presets(daily_closes)
        gate          = mtf_setup.evaluate_daily_gate(daily_presets)
        if not gate['pass']:
            return None

        time.sleep(REQUEST_DELAY)
        h4_closes, _ = get_closes(ticker, 'minutes240')
        time.sleep(REQUEST_DELAY)
        h1_closes, _ = get_closes(ticker, 'minutes60')

        h4_presets = mtf_setup.calc_all_presets(h4_closes) if len(h4_closes) >= 60 else {}
        h1_presets = mtf_setup.calc_all_presets(h1_closes) if len(h1_closes) >= 60 else {}

        vol_ratio = 1.0
        if len(daily_vols) >= 21:
            avg_vol   = sum(daily_vols[-21:-1]) / 20
            vol_ratio = daily_vols[-1] / avg_vol if avg_vol > 0 else 1.0

        score_result = mtf_setup.calc_watch_score(
            daily_presets=daily_presets,
            h4_presets=h4_presets,
            h1_presets=h1_presets,
            volume_ratio=vol_ratio,
        )

        chart_data = {
            'daily': {
                'k': daily_presets.get('short',{}).get('k_series',[]),
                'd': daily_presets.get('short',{}).get('d_series',[]),
            },
            'h4': {
                'k': h4_presets.get('short',{}).get('k_series',[]) if h4_presets else [],
                'd': h4_presets.get('short',{}).get('d_series',[]) if h4_presets else [],
            },
            'h1': {
                'k': h1_presets.get('short',{}).get('k_series',[]) if h1_presets else [],
                'd': h1_presets.get('short',{}).get('d_series',[]) if h1_presets else [],
            },
        }

        return {
            'ticker':        ticker,
            'score_result':  score_result,
            'vol_ratio':     round(vol_ratio, 2),
            'chart_data':    chart_data,
            'daily_presets': daily_presets,
            'h4_presets':    h4_presets,
            'h1_presets':    h1_presets,
        }
    except Exception as e:
        log.debug(f"analyze {ticker} error: {e}")
        return None


# ════════════════════════════════════════════════
# 통계
# ════════════════════════════════════════════════

def calc_stats():
    history = load_trade_history()
    if not history:
        return {
            'total': 0, 'activated': 0, 'expired': 0,
            'tp': 0, 'sl': 0, 'timeout': 0, 'manual': 0,
            'watch_to_active_rate': 0,
            'tp_rate': 0, 'avg_pnl': 0, 'best_pnl': 0, 'worst_pnl': 0,
            'grade_stats': {},
        }

    total     = len(history)
    expired   = sum(1 for h in history if h['result'] == 'expired')
    activated = sum(1 for h in history if h['result'] == 'activated')
    tp        = sum(1 for h in history if h['result'] == 'tp')
    sl        = sum(1 for h in history if h['result'] == 'sl')
    timeout   = sum(1 for h in history if h['result'] == 'timeout')
    manual    = sum(1 for h in history if h['result'] == 'manual')

    watch_total          = expired + activated
    watch_to_active_rate = round(activated / watch_total * 100, 1) if watch_total > 0 else 0

    closed    = [h for h in history
                 if h['result'] in ('tp','sl','timeout','manual')
                 and h.get('pnl_pct') is not None]
    tp_rate   = round(tp / len(closed) * 100, 1) if closed else 0
    avg_pnl   = round(sum(h['pnl_pct'] for h in closed) / len(closed), 2) if closed else 0
    best_pnl  = round(max((h['pnl_pct'] for h in closed), default=0), 2)
    worst_pnl = round(min((h['pnl_pct'] for h in closed), default=0), 2)

    grade_stats = {}
    for grade in ('S','A','B','C'):
        g_closed = [h for h in closed if h.get('grade') == grade]
        g_tp     = sum(1 for h in g_closed if h['result'] == 'tp')
        grade_stats[grade] = {
            'total':   len(g_closed),
            'tp':      g_tp,
            'tp_rate': round(g_tp / len(g_closed) * 100, 1) if g_closed else 0,
            'avg_pnl': round(sum(h['pnl_pct'] for h in g_closed) / len(g_closed), 2) if g_closed else 0,
        }

    return {
        'total': total, 'activated': activated, 'expired': expired,
        'tp': tp, 'sl': sl, 'timeout': timeout, 'manual': manual,
        'watch_to_active_rate': watch_to_active_rate,
        'tp_rate': tp_rate, 'avg_pnl': avg_pnl,
        'best_pnl': best_pnl, 'worst_pnl': worst_pnl,
        'grade_stats': grade_stats,
    }


# ════════════════════════════════════════════════
# Active 전환
# ════════════════════════════════════════════════

def activate_item(item, current_price, reason_grade):
    now  = datetime.now(timezone.utc).isoformat()
    snap = item.get('snapshot', {})

    active = {
        'ticker':        item['ticker'],
        'activated_at':  now,
        'entry_price':   current_price,
        'entry_score':   item.get('current', {}).get('score', 0),
        'grade':         reason_grade,
        'snapshot':      snap,
        'current_price': current_price,
        'pnl_pct':       0.0,
        'manual':        item.get('manual', False),
    }

    watch_hours = 0
    reg_at = snap.get('registered_at', '')
    if reg_at:
        try:
            reg_dt      = datetime.fromisoformat(reg_at.replace('Z', '+00:00'))
            watch_hours = (datetime.now(timezone.utc) - reg_dt).total_seconds() / 3600
        except Exception:
            pass

    append_history({
        'ticker':        item['ticker'],
        'result':        'activated',
        'grade':         reason_grade,
        'score':         item.get('current', {}).get('score', 0),
        'registered_at': reg_at,
        'closed_at':     now,
        'watch_hours':   round(watch_hours, 1),
        'snapshot':      snap,
        'pnl_pct':       None,
        'entry_price':   current_price,
    })
    return active


# ════════════════════════════════════════════════
# TP / SL / Timeout 체크
# ════════════════════════════════════════════════

def check_active_trades(current_prices):
    active_trades = load_active_trades()
    remaining     = []
    closed_items  = []
    now           = datetime.now(timezone.utc)

    for trade in active_trades:
        ticker  = trade['ticker']
        ep      = trade.get('entry_price', 0)
        if not ep:
            remaining.append(trade)
            continue

        cp_data = current_prices.get(ticker)
        if not cp_data:
            remaining.append(trade)
            continue

        cp  = cp_data['price']
        pnl = (cp - ep) / ep * 100
        trade['current_price'] = cp
        trade['pnl_pct']       = round(pnl, 2)

        hours_held = 0
        try:
            act_at     = datetime.fromisoformat(trade['activated_at'].replace('Z', '+00:00'))
            hours_held = (now - act_at).total_seconds() / 3600
        except Exception:
            pass

        result = None
        if pnl >= TRADE_TP_PCT:
            result = 'tp'
        elif pnl <= -TRADE_SL_PCT:
            result = 'sl'
        elif hours_held >= TRADE_TIMEOUT_H:
            result = 'timeout'

        if result:
            record = {
                'ticker':        ticker,
                'result':        result,
                'grade':         trade.get('grade', '?'),
                'score':         trade.get('entry_score', 0),
                'registered_at': trade.get('snapshot', {}).get('registered_at'),
                'activated_at':  trade.get('activated_at'),
                'closed_at':     now.isoformat(),
                'entry_price':   ep,
                'exit_price':    cp,
                'pnl_pct':       round(pnl, 2),
                'hours_held':    round(hours_held, 1),
                'snapshot':      trade.get('snapshot', {}),
            }
            append_history(record)
            closed_items.append(record)
            send_telegram(build_close_msg(record))
            log.info(f"[{result.upper()}] {ticker} pnl={pnl:+.2f}%")
        else:
            remaining.append(trade)

    if len(remaining) != len(active_trades):
        save_active_trades(remaining)

    return remaining, closed_items


# ════════════════════════════════════════════════
# 수동 Watch 등록 / 삭제 / 청산
# ════════════════════════════════════════════════

def add_manual_watch(ticker, entry_price):
    ticker = ticker.upper()
    if not ticker.startswith('KRW-'):
        ticker = f'KRW-{ticker}'
    watch_list = load_watch_list()
    if any(w['ticker'] == ticker for w in watch_list):
        return False, '이미 등록된 종목입니다'
    now = datetime.now(timezone.utc).isoformat()
    item = {
        'ticker': ticker,
        'manual': True,
        'status': 'watch',
        'snapshot': {
            'registered_at': now,
            'entry_price':   float(entry_price),
            'score': 0, 'grade': 'C',
            'daily_k': None, 'h4_k': None, 'h1_k': None, 'btc_price': 0,
        },
        'current': {
            'score': 0, 'grade': 'C',
            'daily_k': None, 'h4_k': None, 'h1_k': None,
            'vol_ratio': 1.0,
            'price': float(entry_price), 'change_pct': 0,
        },
    }
    watch_list.append(item)
    save_watch_list(watch_list)
    with _state_lock:
        scanner_state['watch_list'] = watch_list
    return True, '등록 완료'

def remove_watch(ticker):
    ticker     = ticker.upper()
    watch_list = load_watch_list()
    new_list   = [w for w in watch_list if w['ticker'] != ticker]
    if len(new_list) == len(watch_list):
        return False, '종목을 찾을 수 없습니다'
    save_watch_list(new_list)
    with _state_lock:
        scanner_state['watch_list'] = new_list
    return True, '삭제 완료'

def manual_close_trade(ticker, current_price=None):
    active_trades = load_active_trades()
    remaining     = []
    closed        = False
    now           = datetime.now(timezone.utc)
    for trade in active_trades:
        if trade['ticker'] == ticker.upper():
            ep  = trade.get('entry_price', 0)
            cp  = current_price or trade.get('current_price', ep)
            pnl = (cp - ep) / ep * 100 if ep else 0
            hours_held = 0
            try:
                act_at     = datetime.fromisoformat(trade['activated_at'].replace('Z', '+00:00'))
                hours_held = (now - act_at).total_seconds() / 3600
            except Exception:
                pass
            record = {
                'ticker':        ticker.upper(),
                'result':        'manual',
                'grade':         trade.get('grade', '?'),
                'score':         trade.get('entry_score', 0),
                'registered_at': trade.get('snapshot', {}).get('registered_at'),
                'activated_at':  trade.get('activated_at'),
                'closed_at':     now.isoformat(),
                'entry_price':   ep,
                'exit_price':    cp,
                'pnl_pct':       round(pnl, 2),
                'hours_held':    round(hours_held, 1),
                'snapshot':      trade.get('snapshot', {}),
            }
            append_history(record)
            send_telegram(build_close_msg(record))
            closed = True
        else:
            remaining.append(trade)
    if closed:
        save_active_trades(remaining)
        with _state_lock:
            scanner_state['active_trades'] = remaining
        return True, '청산 완료'
    return False, '종목을 찾을 수 없습니다'


# ════════════════════════════════════════════════
# 메인 스캔 (60분)
# ════════════════════════════════════════════════

def run_scan():
    now_utc = datetime.now(timezone.utc)
    log.info("=== 전체 스캔 시작 ===")
    with _state_lock:
        scanner_state['status'] = 'scanning'
        scanner_state['error']  = None
    try:
        btc_daily, btc_weekly = get_btc_closes()
        macro = mtf_setup.evaluate_macro_filter(btc_weekly, btc_daily)
        btc_change_pct = (
            (btc_daily[-1] - btc_daily[-2]) / btc_daily[-2] * 100
            if len(btc_daily) >= 2 else 0
        )

        watch_list    = load_watch_list()
        active_trades = load_active_trades()
        targets       = get_all_krw_markets()

        results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(analyze_ticker, t): t for t in targets}
            for f in as_completed(futures):
                r = f.result()
                if r:
                    results[r['ticker']] = r

        log.info(f"분석 완료: {len(results)}/{len(targets)}")

        all_tickers = list(set(
            [w['ticker'] for w in watch_list] +
            [a['ticker'] for a in active_trades] +
            list(results.keys())
        ))
        current_prices = get_current_prices(all_tickers)

        existing_tickers = {w['ticker'] for w in watch_list}
        new_watch_list   = []
        new_entries      = []
        removed_items    = []
        new_actives      = []

        # 기존 Watch 업데이트
        for item in watch_list:
            ticker    = item['ticker']
            snap      = item.get('snapshot', {})
            is_manual = item.get('manual', False)

            if ticker in current_prices:
                if 'current' not in item:
                    item['current'] = {}
                item['current']['price']      = current_prices[ticker]['price']
                item['current']['change_pct'] = current_prices[ticker]['change_pct']

            if ticker in results:
                r        = results[ticker]
                coin_chg = current_prices.get(ticker, {}).get('change_pct', 0)
                sr2 = mtf_setup.calc_watch_score(
                    daily_presets   = r['daily_presets'],
                    h4_presets      = r.get('h4_presets', {}),
                    h1_presets      = r.get('h1_presets', {}),
                    volume_ratio    = r['vol_ratio'],
                    initial_daily_k = snap.get('daily_k'),
                    btc_change_pct  = btc_change_pct,
                    coin_change_pct = coin_chg,
                )
                prev_grade = item.get('current', {}).get('grade', 'C')
                item['current'] = {
                    'score':      sr2['score'],
                    'grade':      sr2['grade'],
                    'daily_k':    sr2['daily_k'],
                    'h4_k':       sr2['h4_k'],
                    'h1_k':       sr2['h1_k'],
                    'vol_ratio':  r['vol_ratio'],
                    'price':      current_prices.get(ticker, {}).get('price', 0),
                    'change_pct': current_prices.get(ticker, {}).get('change_pct', 0),
                }
                item['chart_data'] = r['chart_data']
                new_grade = sr2['grade']

                if new_grade in ('A', 'S'):
                    cp     = current_prices.get(ticker, {}).get('price', 0)
                    active = activate_item(item, cp, new_grade)
                    active_trades.append(active)
                    new_actives.append(active)
                    save_active_trades(active_trades)
                    send_telegram(build_active_msg(active, prev_grade))
                    log.info(f"[ACTIVE] {ticker} {new_grade}등급 ({sr2['score']}점)")
                    continue

                if not is_manual and new_grade in ('C', 'B'):
                    reg_at = snap.get('registered_at', '')
                    try:
                        reg_dt   = datetime.fromisoformat(reg_at.replace('Z', '+00:00'))
                        age_days = (now_utc - reg_dt).total_seconds() / 86400
                    except Exception:
                        age_days = 0
                    if age_days >= mtf_setup.WATCH_EXPIRY_DAYS:
                        watch_hours = age_days * 24
                        append_history({
                            'ticker': ticker, 'result': 'expired',
                            'grade': new_grade, 'score': sr2['score'],
                            'registered_at': reg_at,
                            'closed_at': now_utc.isoformat(),
                            'watch_hours': round(watch_hours, 1),
                            'snapshot': snap, 'pnl_pct': None,
                        })
                        removed_items.append(item)
                        send_telegram(build_close_msg({
                            'result': 'expired', 'ticker': ticker,
                            'grade': new_grade, 'score': sr2['score'],
                            'watch_hours': watch_hours,
                        }))
                        log.info(f"[EXPIRED] {ticker} {age_days:.1f}일 경과")
                        continue

            new_watch_list.append(item)

        # 신규 등록
        for ticker, r in results.items():
            if ticker in existing_tickers:
                continue
            sr    = r['score_result']
            grade = sr['grade']
            cp    = current_prices.get(ticker, {}).get('price', 0)
            snap  = {
                'registered_at': now_utc.isoformat(),
                'entry_price':   cp,
                'score':         sr['score'],
                'grade':         grade,
                'daily_k':       sr['daily_k'],
                'h4_k':          sr['h4_k'],
                'h1_k':          sr['h1_k'],
                'btc_price':     btc_daily[-1] if btc_daily else 0,
            }
            if grade in ('A', 'S'):
                dummy  = {'ticker': ticker, 'snapshot': snap, 'current': {'score': sr['score']}, 'manual': False}
                active = activate_item(dummy, cp, grade)
                active_trades.append(active)
                new_actives.append(active)
                save_active_trades(active_trades)
                send_telegram(build_active_msg(active, '-'))
                log.info(f"[NEW ACTIVE] {ticker} {grade}등급 ({sr['score']}점)")
                continue

            new_item = {
                'ticker': ticker, 'manual': False, 'status': 'watch',
                'snapshot': snap,
                'current': {
                    'score': sr['score'], 'grade': grade,
                    'daily_k': sr['daily_k'], 'h4_k': sr['h4_k'], 'h1_k': sr['h1_k'],
                    'vol_ratio': r['vol_ratio'],
                    'price': cp,
                    'change_pct': current_prices.get(ticker, {}).get('change_pct', 0),
                },
                'chart_data': r['chart_data'],
            }
            new_watch_list.append(new_item)
            new_entries.append(new_item)
            log.info(f"[WATCH] {ticker} {grade}등급 ({sr['score']}점)")

        save_watch_list(new_watch_list)
        stats = calc_stats()

        with _state_lock:
            scanner_state.update({
                'status':        'done',
                'last_scan_at':  now_utc.isoformat(),
                'total_scans':   scanner_state['total_scans'] + 1,
                'total_scanned': len(targets),
                'watch_list':    new_watch_list,
                'active_trades': active_trades,
                'new_entries':   new_entries,
                'removed_items': removed_items,
                'new_actives':   new_actives,
                'macro':         macro,
                'stats':         stats,
                'error':         None,
            })

        log.info(
            f"=== 전체 스캔 완료 === "
            f"Watch:{len(new_watch_list)} Active:{len(active_trades)} "
            f"신규Watch:{len(new_entries)} 신규Active:{len(new_actives)}"
        )
    except Exception as e:
        log.error(f"run_scan error: {e}", exc_info=True)
        with _state_lock:
            scanner_state['status'] = 'error'
            scanner_state['error']  = str(e)


# ════════════════════════════════════════════════
# Watch 재스캔 루프 (15분)
# ════════════════════════════════════════════════

def watch_rescan_loop():
    """Watch 종목만 빠르게 재스캔 → 점수/등급 업데이트 → A/S 시 Active 전환"""
    while True:
        try:
            time.sleep(WATCH_RESCAN_INTERVAL_MIN * 60)
            log.info("=== Watch 재스캔 시작 ===")

            with _state_lock:
                watch_list    = list(scanner_state['watch_list'])
                active_trades = list(scanner_state['active_trades'])
                macro         = dict(scanner_state.get('macro', {}))

            if not watch_list:
                continue

            btc_daily, _ = get_btc_closes()
            btc_change_pct = (
                (btc_daily[-1] - btc_daily[-2]) / btc_daily[-2] * 100
                if len(btc_daily) >= 2 else 0
            )

            tickers        = [w['ticker'] for w in watch_list]
            current_prices = get_current_prices(tickers)

            # Watch 종목만 병렬 분석
            results = {}
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as ex:
                futures = {ex.submit(analyze_ticker, t): t for t in tickers}
                for f in as_completed(futures):
                    r = f.result()
                    if r:
                        results[r['ticker']] = r

            new_watch_list = []
            new_actives    = []
            now_utc        = datetime.now(timezone.utc)

            for item in watch_list:
                ticker    = item['ticker']
                snap      = item.get('snapshot', {})

                # 현재가 업데이트
                if ticker in current_prices:
                    if 'current' not in item:
                        item['current'] = {}
                    item['current']['price']      = current_prices[ticker]['price']
                    item['current']['change_pct'] = current_prices[ticker]['change_pct']

                # 점수 재계산
                if ticker in results:
                    r        = results[ticker]
                    coin_chg = current_prices.get(ticker, {}).get('change_pct', 0)
                    sr2 = mtf_setup.calc_watch_score(
                        daily_presets   = r['daily_presets'],
                        h4_presets      = r.get('h4_presets', {}),
                        h1_presets      = r.get('h1_presets', {}),
                        volume_ratio    = r['vol_ratio'],
                        initial_daily_k = snap.get('daily_k'),
                        btc_change_pct  = btc_change_pct,
                        coin_change_pct = coin_chg,
                    )
                    prev_grade = item.get('current', {}).get('grade', 'C')
                    item['current'] = {
                        'score':      sr2['score'],
                        'grade':      sr2['grade'],
                        'daily_k':    sr2['daily_k'],
                        'h4_k':       sr2['h4_k'],
                        'h1_k':       sr2['h1_k'],
                        'vol_ratio':  r['vol_ratio'],
                        'price':      current_prices.get(ticker, {}).get('price', 0),
                        'change_pct': current_prices.get(ticker, {}).get('change_pct', 0),
                    }
                    item['chart_data'] = r['chart_data']
                    new_grade = sr2['grade']

                    # A/S → Active 전환
                    if new_grade in ('A', 'S'):
                        cp     = current_prices.get(ticker, {}).get('price', 0)
                        active = activate_item(item, cp, new_grade)
                        active_trades.append(active)
                        new_actives.append(active)
                        save_active_trades(active_trades)
                        send_telegram(build_active_msg(active, prev_grade))
                        log.info(f"[WATCH→ACTIVE] {ticker} {new_grade}등급 ({sr2['score']}점)")
                        continue

                new_watch_list.append(item)

            save_watch_list(new_watch_list)
            stats = calc_stats()

            with _state_lock:
                scanner_state['watch_list']           = new_watch_list
                scanner_state['active_trades']        = active_trades
                scanner_state['last_watch_rescan_at'] = now_utc.isoformat()
                scanner_state['stats']                = stats
                if new_actives:
                    scanner_state['new_actives'] = new_actives

            log.info(
                f"=== Watch 재스캔 완료 === "
                f"Watch:{len(new_watch_list)} 신규Active:{len(new_actives)}"
            )

        except Exception as e:
            log.warning(f"watch_rescan_loop error: {e}", exc_info=True)


# ════════════════════════════════════════════════
# 가격 체크 루프 (5분) - Watch 현재가
# ════════════════════════════════════════════════

def price_check_loop():
    """Watch 종목 현재가만 빠르게 업데이트"""
    while True:
        try:
            time.sleep(PRICE_CHECK_INTERVAL_MIN * 60)

            with _state_lock:
                watch_list = list(scanner_state['watch_list'])

            if not watch_list:
                continue

            tickers        = [w['ticker'] for w in watch_list]
            current_prices = get_current_prices(tickers)

            for item in watch_list:
                t = item['ticker']
                if t in current_prices:
                    if 'current' not in item:
                        item['current'] = {}
                    item['current']['price']      = current_prices[t]['price']
                    item['current']['change_pct'] = current_prices[t]['change_pct']

            with _state_lock:
                scanner_state['watch_list']         = watch_list
                scanner_state['last_price_check_at'] = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            log.warning(f"price_check_loop error: {e}")


# ════════════════════════════════════════════════
# Active 모니터링 루프 (1분) - TP/SL/Timeout
# ════════════════════════════════════════════════

def active_monitor_loop():
    """Active 종목만 1분마다 TP/SL/Timeout 체크"""
    while True:
        try:
            time.sleep(ACTIVE_CHECK_INTERVAL_MIN * 60)

            with _state_lock:
                active_trades = list(scanner_state['active_trades'])

            if not active_trades:
                continue

            tickers        = [a['ticker'] for a in active_trades]
            current_prices = get_current_prices(tickers)

            remaining, closed = check_active_trades(current_prices)

            with _state_lock:
                scanner_state['active_trades']         = remaining
                scanner_state['last_active_check_at']  = datetime.now(timezone.utc).isoformat()
                if closed:
                    scanner_state['stats'] = calc_stats()

            if closed:
                log.info(f"[ACTIVE MONITOR] 청산 {len(closed)}건")

        except Exception as e:
            log.warning(f"active_monitor_loop error: {e}")


# ════════════════════════════════════════════════
# 스캔 루프 / 수동 트리거
# ════════════════════════════════════════════════

def manual_scan():
    _manual_scan_event.set()

def scanner_loop():
    run_scan()
    while True:
        next_scan = datetime.now(timezone.utc) + timedelta(minutes=SCAN_INTERVAL_MIN)
        with _state_lock:
            scanner_state['next_scan_at'] = next_scan.isoformat()
        triggered = _manual_scan_event.wait(timeout=SCAN_INTERVAL_MIN * 60)
        if triggered:
            _manual_scan_event.clear()
            log.info("수동 스캔 트리거")
        run_scan()


if __name__ == "__main__":
    run_scan()
