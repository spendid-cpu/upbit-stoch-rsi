# -*- coding: utf-8 -*-
"""
scanner.py — Upbit MTF 자동 스캐너 (v1.4)
변경사항:
  - TP/SL/타임아웃 트레이드 모니터링 추가
  - active_trades.json: 활성 포지션 관리
  - trade_history.json: 청산 기록 + 승률 계산
  - TP=+5%, SL=-3%, 타임아웃=48H (환경변수로 조정 가능)
  - entry_price 최초 1회만 저장 (덮어쓰기 금지)
"""

import os
import time
import json
import logging
import requests
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import mtf_setup

# ── 로깅 ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── 환경변수 ──────────────────────────────────────
SCAN_INTERVAL_MIN   = int(os.environ.get('SCAN_INTERVAL_MIN', 60))
MIN_TRADE_VALUE_KRW = float(os.environ.get('MIN_TRADE_VALUE_KRW', 0))
REQUEST_DELAY       = float(os.environ.get('REQUEST_DELAY', 0.12))
MAX_WORKERS         = int(os.environ.get('MAX_WORKERS', 6))
CANDLE_COUNT        = int(os.environ.get('CANDLE_COUNT', 200))
WATCH_LIST_FILE     = os.environ.get('WATCH_LIST_FILE',     'watch_list.json')
SIGNAL_HISTORY_FILE = os.environ.get('SIGNAL_HISTORY_FILE', 'signal_history.json')
ACTIVE_TRADES_FILE  = os.environ.get('ACTIVE_TRADES_FILE',  'active_trades.json')
TRADE_HISTORY_FILE  = os.environ.get('TRADE_HISTORY_FILE',  'trade_history.json')

# ── TP/SL/타임아웃 설정 ────────────────────────────
TRADE_TP_PCT      = float(os.environ.get('TRADE_TP_PCT',      5.0))   # +5%
TRADE_SL_PCT      = float(os.environ.get('TRADE_SL_PCT',      3.0))   # -3%
TRADE_TIMEOUT_H   = float(os.environ.get('TRADE_TIMEOUT_H',   48.0))  # 48시간

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT  = os.environ.get('TELEGRAM_CHAT_ID',   '')

# ── 스테이블코인 제외 목록 ─────────────────────────
STABLE_COINS = {
    'KRW-USDT', 'KRW-USDC', 'KRW-BUSD', 'KRW-DAI',
    'KRW-TUSD', 'KRW-USDP', 'KRW-GUSD', 'KRW-FRAX',
    'KRW-USDD', 'KRW-FDUSD', 'KRW-PYUSD',
    'KRW-USDE', 'KRW-USDS', 'KRW-USD0',
}

# ── 공유 상태 ─────────────────────────────────────
_state_lock   = threading.Lock()
_manual_event = threading.Event()

scanner_state = {
    'status'        : 'idle',
    'last_scan_at'  : None,
    'next_scan_at'  : None,
    'scan_count'    : 0,
    'watch_list'    : [],
    'new_entries'   : [],
    'entry_signals' : [],
    'removed'       : [],
    'macro'         : {},
    'total_scanned' : 0,
    'current_prices': {},
    'chart_data'    : {},
    # TP/SL 모니터링
    'active_trades' : [],   # 현재 활성 포지션
    'closed_trades' : [],   # 이번 스캔에서 청산된 포지션
    'win_rate'      : None, # 전체 승률 (%)
    'trade_stats'   : {},   # { total, win, loss, timeout, avg_pnl }
    'error'         : None,
}

# ===================== Upbit API =====================
def _get(url, params=None, retry=3):
    for i in range(retry):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == retry - 1:
                raise
            time.sleep(0.5 * (i + 1))

def get_all_krw_markets():
    data = _get('https://api.upbit.com/v1/market/all', {'isDetails': 'true'})
    markets = []
    for d in data:
        market = d['market']
        if not market.startswith('KRW-'):
            continue
        if market in STABLE_COINS:
            continue
        if market == 'KRW-BTC':
            continue
        if d.get('market_warning', 'NONE') != 'NONE':
            continue
        markets.append(market)
    log.info(f'전체 스캔 대상: {len(markets)}개')
    return markets

def get_closes(ticker, interval, count=200):
    try:
        url_map = {
            'day'     : 'https://api.upbit.com/v1/candles/days',
            'week'    : 'https://api.upbit.com/v1/candles/weeks',
            'minutes4': 'https://api.upbit.com/v1/candles/minutes/240',
            'minutes1': 'https://api.upbit.com/v1/candles/minutes/60',
        }
        url = url_map.get(interval)
        if not url:
            return []
        data = _get(url, {'market': ticker, 'count': count})
        data.sort(key=lambda x: x['candle_date_time_utc'])
        return [float(c['trade_price']) for c in data]
    except Exception as e:
        log.debug(f'{ticker} {interval} 캔들 오류: {e}')
        return []

def get_btc_closes():
    daily  = get_closes('KRW-BTC', 'day',  count=60)
    time.sleep(REQUEST_DELAY)
    weekly = get_closes('KRW-BTC', 'week', count=30)
    return daily, weekly

def get_current_prices(tickers):
    if not tickers:
        return {}
    result = {}
    chunks = [tickers[i:i+100] for i in range(0, len(tickers), 100)]
    for chunk in chunks:
        try:
            data = _get('https://api.upbit.com/v1/ticker',
                        {'markets': ','.join(chunk)})
            for d in data:
                result[d['market']] = {
                    'price'     : d.get('trade_price', 0),
                    'change_pct': d.get('signed_change_rate', 0) * 100,
                }
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log.debug(f'현재가 조회 오류: {e}')
    return result

# ===================== 파일 I/O =====================
def load_watch_list():
    if not os.path.exists(WATCH_LIST_FILE):
        return []
    try:
        with open(WATCH_LIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_watch_list(wl):
    with open(WATCH_LIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)

def load_signal_history():
    if not os.path.exists(SIGNAL_HISTORY_FILE):
        return []
    try:
        with open(SIGNAL_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def append_signal_history(record):
    history = load_signal_history()
    history.append(record)
    history = history[-500:]
    with open(SIGNAL_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_active_trades():
    if not os.path.exists(ACTIVE_TRADES_FILE):
        return []
    try:
        with open(ACTIVE_TRADES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_active_trades(trades):
    with open(ACTIVE_TRADES_FILE, 'w', encoding='utf-8') as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)

def load_trade_history():
    if not os.path.exists(TRADE_HISTORY_FILE):
        return []
    try:
        with open(TRADE_HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def append_trade_history(record):
    history = load_trade_history()
    history.append(record)
    history = history[-1000:]
    with open(TRADE_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ===================== 승률 계산 =====================
def calc_trade_stats():
    """
    trade_history.json 기반 승률·통계 계산.
    Returns: { total, win, loss, timeout, win_rate, avg_pnl, best_pnl, worst_pnl }
    """
    history = load_trade_history()
    if not history:
        return {
            'total': 0, 'win': 0, 'loss': 0, 'timeout': 0,
            'win_rate': None, 'avg_pnl': None,
            'best_pnl': None, 'worst_pnl': None,
        }

    total   = len(history)
    win     = sum(1 for h in history if h.get('result') == 'TP')
    loss    = sum(1 for h in history if h.get('result') == 'SL')
    timeout = sum(1 for h in history if h.get('result') == 'TIMEOUT')
    pnls    = [h.get('pnl_pct', 0) for h in history if h.get('pnl_pct') is not None]

    return {
        'total'    : total,
        'win'      : win,
        'loss'     : loss,
        'timeout'  : timeout,
        'win_rate' : round(win / total * 100, 1) if total > 0 else None,
        'avg_pnl'  : round(sum(pnls) / len(pnls), 2) if pnls else None,
        'best_pnl' : round(max(pnls), 2) if pnls else None,
        'worst_pnl': round(min(pnls), 2) if pnls else None,
    }

# ===================== TP/SL 모니터링 =====================
def check_active_trades(current_prices):
    """
    활성 트레이드 TP/SL/타임아웃 체크.
    Returns: (remaining_trades, closed_trades)
    """
    trades   = load_active_trades()
    now_utc  = datetime.now(timezone.utc)
    remaining = []
    closed    = []

    for trade in trades:
        ticker     = trade['ticker']
        entry_px   = trade.get('entry_price')
        entered_at = trade.get('entered_at')

        if not entry_px or entry_px <= 0:
            remaining.append(trade)
            continue

        # 현재가
        cp = current_prices.get(ticker, {}).get('price')
        if not cp:
            remaining.append(trade)
            continue

        pnl_pct = (cp - entry_px) / entry_px * 100

        # 경과 시간
        hours_elapsed = 0
        if entered_at:
            try:
                entered_dt = datetime.fromisoformat(entered_at)
                if entered_dt.tzinfo is None:
                    entered_dt = entered_dt.replace(tzinfo=timezone.utc)
                hours_elapsed = (now_utc - entered_dt).total_seconds() / 3600
            except Exception:
                pass

        # 청산 조건 판정
        result = None
        if pnl_pct >= TRADE_TP_PCT:
            result = 'TP'
        elif pnl_pct <= -TRADE_SL_PCT:
            result = 'SL'
        elif hours_elapsed >= TRADE_TIMEOUT_H:
            result = 'TIMEOUT'

        if result:
            closed_record = {
                **trade,
                'result'       : result,
                'exit_price'   : cp,
                'pnl_pct'      : round(pnl_pct, 3),
                'hours_held'   : round(hours_elapsed, 1),
                'closed_at'    : now_utc.isoformat(),
            }
            closed.append(closed_record)
            append_trade_history(closed_record)
            log.info(
                f'[{result}] {ticker} | '
                f'진입 {fmt_price(entry_px)} → 현재 {fmt_price(cp)} | '
                f'PnL {pnl_pct:+.2f}% | {hours_elapsed:.1f}h'
            )
            # 텔레그램 청산 알림
            send_telegram(build_trade_close_msg(closed_record))
        else:
            # 현재 PnL 업데이트
            trade['current_price'] = cp
            trade['pnl_pct']       = round(pnl_pct, 3)
            trade['hours_held']    = round(hours_elapsed, 1)
            remaining.append(trade)

    save_active_trades(remaining)
    return remaining, closed

def open_trade(signal, entry_price):
    """진입 신호 발생 시 active_trades에 추가."""
    now_utc = datetime.now(timezone.utc)
    trade = {
        'ticker'      : signal['ticker'],
        'entered_at'  : now_utc.isoformat(),
        'entry_price' : entry_price,
        'tp_price'    : round(entry_price * (1 + TRADE_TP_PCT / 100), 6),
        'sl_price'    : round(entry_price * (1 - TRADE_SL_PCT / 100), 6),
        'timeout_at'  : datetime.fromtimestamp(
                            now_utc.timestamp() + TRADE_TIMEOUT_H * 3600,
                            tz=timezone.utc
                        ).isoformat(),
        'trigger'     : signal.get('trigger', {}),
        'current_price': entry_price,
        'pnl_pct'     : 0.0,
        'hours_held'  : 0.0,
    }
    trades = load_active_trades()
    # 동일 종목 중복 방지
    if not any(t['ticker'] == signal['ticker'] for t in trades):
        trades.append(trade)
        save_active_trades(trades)
        log.info(
            f'트레이드 오픈: {signal["ticker"]} @ {fmt_price(entry_price)}원 | '
            f'TP {fmt_price(trade["tp_price"])} / SL {fmt_price(trade["sl_price"])}'
        )
    return trade

# ===================== 텔레그램 =====================
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    url     = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    MAX_LEN = 4000
    for chunk in [text[i:i+MAX_LEN] for i in range(0, len(text), MAX_LEN)]:
        try:
            requests.post(url, json={
                'chat_id': TELEGRAM_CHAT, 'text': chunk, 'parse_mode': 'HTML'
            }, timeout=10)
        except Exception as e:
            log.warning(f'텔레그램 오류: {e}')

def fmt_price(p):
    if p is None: return '—'
    if p >= 100:  return f'{p:,.0f}'
    if p >= 1:    return f'{p:,.2f}'
    return f'{p:,.4f}'

def build_new_entry_msg(items, macro_state, prices):
    lines = [f'📋 <b>Watch List 신규 등록 {len(items)}개</b>']
    icon  = '✅' if macro_state['safe'] else '🚫'
    w     = macro_state.get('weekly_distance_pct', 0) or 0
    d     = macro_state.get('daily_distance_pct',  0) or 0
    lines.append(f'BTC 주봉MA20({icon}): {w:+.2f}% | 일봉MA20: {d:+.2f}%')
    lines.append('')
    for item in items:
        t   = item['ticker']
        p   = prices.get(t, {})
        prc = fmt_price(p.get('price'))
        chg = p.get('change_pct', 0)
        lines.append(f'• <b>{t}</b>  일봉K {item["daily_short_k"]:.1f} | {prc}원 ({chg:+.2f}%)')
    return '\n'.join(lines)

def build_entry_signal_msg(items, prices):
    lines = [f'🚀 <b>진입 트리거 {len(items)}개</b>']
    lines.append(f'TP +{TRADE_TP_PCT}% / SL -{TRADE_SL_PCT}% / 타임아웃 {TRADE_TIMEOUT_H:.0f}h')
    lines.append('')
    for item in items:
        t   = item['ticker']
        tr  = item['trigger']
        p   = prices.get(t, {})
        prc = fmt_price(p.get('price'))
        chg = p.get('change_pct', 0)
        ep  = item.get('entry_price')
        tp  = fmt_price(ep * (1 + TRADE_TP_PCT / 100)) if ep else '—'
        sl  = fmt_price(ep * (1 - TRADE_SL_PCT / 100)) if ep else '—'
        lines.append(
            f'⚡ <b>{t}</b>  {prc}원 ({chg:+.2f}%)\n'
            f'   4h K {tr["h4_short_k"]:.1f} | 1h K {tr["h1_short_k"]:.1f} | {tr["h1_trigger_type"]}\n'
            f'   🎯 TP {tp} / 🛑 SL {sl}'
        )
    return '\n'.join(lines)

def build_trade_close_msg(trade):
    icons = {'TP': '🎯', 'SL': '🛑', 'TIMEOUT': '⏰'}
    icon  = icons.get(trade['result'], '📌')
    pnl   = trade.get('pnl_pct', 0)
    pnl_icon = '▲' if pnl >= 0 else '▼'
    stats = calc_trade_stats()
    wr    = f"{stats['win_rate']:.1f}%" if stats['win_rate'] is not None else '—'
    return (
        f'{icon} <b>{trade["result"]} — {trade["ticker"]}</b>\n'
        f'진입 {fmt_price(trade["entry_price"])}원 → 청산 {fmt_price(trade["exit_price"])}원\n'
        f'{pnl_icon} PnL <b>{pnl:+.2f}%</b> | 보유 {trade["hours_held"]:.1f}h\n'
        f'📊 누적 승률 {wr} ({stats["win"]}승 {stats["loss"]}패 {stats["timeout"]}타임아웃)'
    )

# ===================== 단일 종목 분석 =====================
def analyze_ticker(ticker, watch_list_map):
    try:
        daily = get_closes(ticker, 'day', count=CANDLE_COUNT)
        time.sleep(REQUEST_DELAY)
        if len(daily) < 60:
            return None

        watch_result  = mtf_setup.evaluate_watch_list_entry(daily, ticker)
        daily_presets = mtf_setup.calc_all_presets(daily)
        chart = {
            'daily_k': daily_presets['short'].get('k_line', [])[-50:],
            'daily_d': daily_presets['short'].get('d_line', [])[-50:],
            'h4_k': [], 'h4_d': [],
            'h1_k': [], 'h1_d': [],
        }

        entry_result = None
        if ticker in watch_list_map or watch_result['should_register']:
            h4 = get_closes(ticker, 'minutes4', count=CANDLE_COUNT)
            time.sleep(REQUEST_DELAY)
            h1 = get_closes(ticker, 'minutes1', count=CANDLE_COUNT)
            time.sleep(REQUEST_DELAY)

            h4_presets = mtf_setup.calc_all_presets(h4)
            h1_presets = mtf_setup.calc_all_presets(h1)
            chart['h4_k'] = h4_presets['short'].get('k_line', [])[-50:]
            chart['h4_d'] = h4_presets['short'].get('d_line', [])[-50:]
            chart['h1_k'] = h1_presets['short'].get('k_line', [])[-50:]
            chart['h1_d'] = h1_presets['short'].get('d_line', [])[-50:]
            entry_result  = mtf_setup.evaluate_entry_trigger(h4, h1, ticker)

        return {
            'ticker'       : ticker,
            'watch_result' : watch_result,
            'entry_result' : entry_result,
            'daily_short_k': watch_result.get('daily_short_k'),
            'chart'        : chart,
        }
    except Exception as e:
        log.debug(f'{ticker} 분석 오류: {e}')
        return None

# ===================== 수동 스캔 =====================
def manual_scan():
    _manual_event.set()

# ===================== 메인 스캔 =====================
def run_scan():
    now_utc = datetime.now(timezone.utc)
    log.info(f'=== 스캔 시작 {now_utc.strftime("%Y-%m-%d %H:%M UTC")} ===')

    with _state_lock:
        scanner_state['status']        = 'scanning'
        scanner_state['error']         = None
        scanner_state['new_entries']   = []
        scanner_state['entry_signals'] = []
        scanner_state['removed']       = []
        scanner_state['closed_trades'] = []

    try:
        # 1) BTC 매크로
        btc_daily, btc_weekly = get_btc_closes()
        macro  = mtf_setup.evaluate_macro_filter(btc_daily, btc_weekly, ticker='BTC')
        w_dist = macro.get('weekly_distance_pct', 0) or 0
        d_dist = macro.get('daily_distance_pct',  0) or 0
        log.info(f'매크로: safe={macro["safe"]} | 주봉MA20 {w_dist:+.2f}% | 일봉MA20 {d_dist:+.2f}%')

        # 2) Watch List
        watch_list     = load_watch_list()
        watch_list_map = {w['ticker']: w for w in watch_list}

        # 3) 전 종목 스캔
        targets = get_all_krw_markets()

        # 4) 병렬 분석
        results    = []
        chart_data = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(analyze_ticker, t, watch_list_map): t for t in targets}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)
                    if r.get('chart'):
                        chart_data[r['ticker']] = r['chart']

        # 5) Watch List 무효화
        new_watch_list = []
        removed_list   = []
        for item in watch_list:
            ticker = item['ticker']
            daily  = get_closes(ticker, 'day', count=CANDLE_COUNT)
            time.sleep(REQUEST_DELAY)
            inv = mtf_setup.evaluate_watch_invalidation(daily, item, ticker)
            if inv['should_remove']:
                removed_list.append({
                    'ticker'      : ticker,
                    'reason'      : inv['reason'],
                    'removal_type': inv['removal_type'],
                    'removed_at'  : now_utc.isoformat(),
                })
                log.info(f'Watch 제거: {ticker} — {inv["reason"]}')
            else:
                new_watch_list.append(item)

        # 6) 신규 등록 (entry_price=None, 현재가 조회 후 채움)
        new_entries      = []
        existing_tickers = {w['ticker'] for w in new_watch_list}
        for r in results:
            wr     = r['watch_result']
            ticker = r['ticker']
            if wr['should_register'] and ticker not in existing_tickers:
                if mtf_setup.USE_MACRO_FILTER and not macro['safe']:
                    log.info(f'Watch 보류(매크로 차단): {ticker}')
                    continue
                entry = {
                    'ticker'       : ticker,
                    'registered_at': now_utc.isoformat(),
                    'daily_short_k': wr['daily_short_k'],
                    'reason'       : wr['reason'],
                    'entry_price'  : None,
                    'current_price': None,
                    'change_pct'   : None,
                }
                new_watch_list.append(entry)
                new_entries.append(entry)
                existing_tickers.add(ticker)
                log.info(f'Watch 등록: {ticker} K={wr["daily_short_k"]:.1f}')

        # 7) 현재가 일괄 조회 (Watch List 전체)
        all_tickers    = [w['ticker'] for w in new_watch_list]
        current_prices = get_current_prices(all_tickers)

        # 8) Watch List 현재가 업데이트 + entry_price 최초 1회 설정
        for item in new_watch_list:
            t = item['ticker']
            if t in current_prices:
                item['current_price'] = current_prices[t]['price']
                item['change_pct']    = current_prices[t]['change_pct']
                if item.get('entry_price') is None:
                    item['entry_price'] = current_prices[t]['price']
                    log.info(f'entry_price 설정: {t} = {fmt_price(item["entry_price"])}원')

        # 9) 진입 트리거 → active_trades 등록
        entry_signals = []
        for r in results:
            er = r.get('entry_result')
            if er and er['should_enter']:
                watch_item = next(
                    (w for w in new_watch_list if w['ticker'] == r['ticker']), {}
                )
                ep = watch_item.get('entry_price') or \
                     current_prices.get(r['ticker'], {}).get('price')

                signal = {
                    'ticker'      : r['ticker'],
                    'triggered_at': now_utc.isoformat(),
                    'trigger'     : er,
                    'entry_price' : ep,
                }
                entry_signals.append(signal)
                append_signal_history(signal)
                log.info(f'진입 신호: {r["ticker"]} — {er["reason"]}')

                # ★ active_trades에 포지션 오픈
                if ep:
                    open_trade(signal, ep)

        # 10) TP/SL/타임아웃 체크
        #     active_trades에 있는 종목의 현재가도 조회
        active_trades   = load_active_trades()
        active_tickers  = [t['ticker'] for t in active_trades
                           if t['ticker'] not in current_prices]
        if active_tickers:
            extra_prices = get_current_prices(active_tickers)
            current_prices.update(extra_prices)

        active_trades, closed_trades = check_active_trades(current_prices)

        # 11) 저장
        save_watch_list(new_watch_list)

        # 12) 텔레그램
        if new_entries:
            send_telegram(build_new_entry_msg(new_entries, macro, current_prices))
        if entry_signals:
            send_telegram(build_entry_signal_msg(entry_signals, current_prices))

        # 13) 통계
        stats = calc_trade_stats()

        # 14) 상태 업데이트
        with _state_lock:
            scanner_state.update({
                'status'        : 'done',
                'last_scan_at'  : now_utc.isoformat(),
                'scan_count'    : scanner_state['scan_count'] + 1,
                'watch_list'    : new_watch_list,
                'new_entries'   : new_entries,
                'entry_signals' : entry_signals,
                'removed'       : removed_list,
                'macro'         : macro,
                'total_scanned' : len(targets),
                'current_prices': current_prices,
                'chart_data'    : chart_data,
                'active_trades' : active_trades,
                'closed_trades' : closed_trades,
                'win_rate'      : stats['win_rate'],
                'trade_stats'   : stats,
            })

        log.info(
            f'스캔 완료 | {len(targets)}개 | Watch {len(new_watch_list)}개 | '
            f'진입 {len(entry_signals)}개 | 활성 {len(active_trades)}개 | '
            f'청산 {len(closed_trades)}개 | 승률 {stats["win_rate"]}%'
        )

    except Exception as e:
        log.error(f'스캔 오류: {e}', exc_info=True)
        with _state_lock:
            scanner_state['status'] = 'error'
            scanner_state['error']  = str(e)


def scanner_loop():
    while True:
        run_scan()
        next_at = datetime.now(timezone.utc).timestamp() + SCAN_INTERVAL_MIN * 60
        with _state_lock:
            scanner_state['next_scan_at'] = datetime.fromtimestamp(
                next_at, tz=timezone.utc
            ).isoformat()
        log.info(f'다음 스캔까지 {SCAN_INTERVAL_MIN}분 대기')
        _manual_event.wait(timeout=SCAN_INTERVAL_MIN * 60)
        _manual_event.clear()


if __name__ == '__main__':
    run_scan()
