# -*- coding: utf-8 -*-
"""
scanner.py — Upbit MTF 자동 스캐너 (v1.1)
변경사항:
  - 스테이블코인 제외 전 종목 스캔 (SCAN_TARGET_COUNT 환경변수 무시)
  - STABLE_COINS 집합으로 스테이블코인 필터링
  - macro 정보를 scanner_state에 포함
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
MIN_TRADE_VALUE_KRW = float(os.environ.get('MIN_TRADE_VALUE_KRW', 0))  # 0 = 제한 없음
REQUEST_DELAY       = float(os.environ.get('REQUEST_DELAY', 0.12))
MAX_WORKERS         = int(os.environ.get('MAX_WORKERS', 6))
CANDLE_COUNT        = int(os.environ.get('CANDLE_COUNT', 200))
WATCH_LIST_FILE     = os.environ.get('WATCH_LIST_FILE', 'watch_list.json')
SIGNAL_HISTORY_FILE = os.environ.get('SIGNAL_HISTORY_FILE', 'signal_history.json')

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT  = os.environ.get('TELEGRAM_CHAT_ID', '')

# ── 스테이블코인 제외 목록 ─────────────────────────
STABLE_COINS = {
    'KRW-USDT', 'KRW-USDC', 'KRW-BUSD', 'KRW-DAI',
    'KRW-TUSD', 'KRW-USDP', 'KRW-GUSD', 'KRW-FRAX',
    'KRW-USDD', 'KRW-FDUSD', 'KRW-PYUSD',
}

# ── 공유 상태 ─────────────────────────────────────
_state_lock  = threading.Lock()
scanner_state = {
    'status'       : 'idle',
    'last_scan_at' : None,
    'next_scan_at' : None,
    'scan_count'   : 0,
    'watch_list'   : [],
    'new_entries'  : [],
    'entry_signals': [],
    'removed'      : [],
    'macro'        : {},   # ← 매크로 정보 추가
    'total_scanned': 0,    # ← 이번 스캔 종목 수
    'error'        : None,
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
    """스테이블코인·BTC 제외 전체 KRW 마켓 반환."""
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
    log.info(f'전체 스캔 대상: {len(markets)}개 (스테이블·BTC 제외)')
    return markets

def get_closes(ticker, interval, count=200):
    try:
        url_map = {
            'day'     : 'https://api.upbit.com/v1/candles/days',
            'week'    : 'https://api.upbit.com/v1/candles/weeks',
            'minutes4': 'https://api.upbit.com/v1/candles/minutes/240',
            'minutes1': 'https://api.upbit.com/v1/candles/minutes/60',
        }
        url  = url_map.get(interval)
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

# ===================== Watch List I/O =====================
def load_watch_list():
    if not os.path.exists(WATCH_LIST_FILE):
        return []
    try:
        with open(WATCH_LIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_watch_list(watch_list):
    with open(WATCH_LIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(watch_list, f, ensure_ascii=False, indent=2)

def append_signal_history(record):
    history = []
    if os.path.exists(SIGNAL_HISTORY_FILE):
        try:
            with open(SIGNAL_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(record)
    history = history[-500:]
    with open(SIGNAL_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ===================== 텔레그램 =====================
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    url     = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    MAX_LEN = 4000
    chunks  = [text[i:i+MAX_LEN] for i in range(0, len(text), MAX_LEN)]
    for chunk in chunks:
        try:
            requests.post(url, json={
                'chat_id'   : TELEGRAM_CHAT,
                'text'      : chunk,
                'parse_mode': 'HTML'
            }, timeout=10)
        except Exception as e:
            log.warning(f'텔레그램 전송 오류: {e}')

def build_new_entry_msg(items, macro_state):
    lines = [f'📋 <b>Watch List 신규 등록 {len(items)}개</b>']
    macro_icon = '✅' if macro_state['safe'] else '🚫'
    w_dist = macro_state.get('weekly_distance_pct', 0) or 0
    d_dist = macro_state.get('daily_distance_pct', 0) or 0
    lines.append(f'BTC 주봉MA20({macro_icon}): {w_dist:+.2f}% | 일봉MA20(참고): {d_dist:+.2f}%')
    lines.append('')
    for item in items:
        lines.append(f'• <b>{item["ticker"]}</b>  일봉단기K {item["daily_short_k"]:.1f}')
    return '\n'.join(lines)

def build_entry_signal_msg(items):
    lines = [f'🚀 <b>진입 트리거 {len(items)}개</b>']
    lines.append('')
    for item in items:
        t = item['trigger']
        lines.append(
            f'⚡ <b>{item["ticker"]}</b>\n'
            f'   4h K {t["h4_short_k"]:.1f} | '
            f'1h K {t["h1_short_k"]:.1f} | '
            f'{t["h1_trigger_type"]}\n'
            f'   {t["reason"]}'
        )
    return '\n'.join(lines)

# ===================== 단일 종목 분석 =====================
def analyze_ticker(ticker, watch_list_map):
    try:
        daily = get_closes(ticker, 'day', count=CANDLE_COUNT)
        time.sleep(REQUEST_DELAY)
        if len(daily) < 60:
            return None

        watch_result = mtf_setup.evaluate_watch_list_entry(daily, ticker)

        entry_result = None
        if ticker in watch_list_map or watch_result['should_register']:
            h4 = get_closes(ticker, 'minutes4', count=CANDLE_COUNT)
            time.sleep(REQUEST_DELAY)
            h1 = get_closes(ticker, 'minutes1', count=CANDLE_COUNT)
            time.sleep(REQUEST_DELAY)
            entry_result = mtf_setup.evaluate_entry_trigger(h4, h1, ticker)

        return {
            'ticker'       : ticker,
            'watch_result' : watch_result,
            'entry_result' : entry_result,
            'daily_short_k': watch_result.get('daily_short_k'),
        }
    except Exception as e:
        log.debug(f'{ticker} 분석 오류: {e}')
        return None

# ===================== 메인 스캔 =====================
def run_scan():
    now_utc = datetime.now(timezone.utc)
    log.info(f'=== 스캔 시작 {now_utc.strftime("%Y-%m-%d %H:%M UTC")} ===')

    with _state_lock:
        scanner_state['status']       = 'scanning'
        scanner_state['error']        = None
        scanner_state['new_entries']  = []
        scanner_state['entry_signals']= []
        scanner_state['removed']      = []

    try:
        # 1) BTC 매크로 (주봉 MA20 기준)
        btc_daily, btc_weekly = get_btc_closes()
        macro = mtf_setup.evaluate_macro_filter(btc_daily, btc_weekly, ticker='BTC')
        w_dist = macro.get('weekly_distance_pct', 0) or 0
        d_dist = macro.get('daily_distance_pct', 0) or 0
        log.info(f'매크로: safe={macro["safe"]} | 주봉MA20 {w_dist:+.2f}% | 일봉MA20 {d_dist:+.2f}% (참고)')

        # 2) Watch List
        watch_list     = load_watch_list()
        watch_list_map = {w['ticker']: w for w in watch_list}

        # 3) 전 종목 스캔
        targets = get_all_krw_markets()

        # 4) 병렬 분석
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(analyze_ticker, t, watch_list_map): t
                for t in targets
            }
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)

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

        # 6) 신규 등록
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
                }
                new_watch_list.append(entry)
                new_entries.append(entry)
                existing_tickers.add(ticker)
                log.info(f'Watch 등록: {ticker} K={wr["daily_short_k"]:.1f}')

        # 7) 진입 트리거
        entry_signals = []
        for r in results:
            er = r.get('entry_result')
            if er and er['should_enter']:
                signal = {
                    'ticker'      : r['ticker'],
                    'triggered_at': now_utc.isoformat(),
                    'trigger'     : er,
                }
                entry_signals.append(signal)
                append_signal_history(signal)
                log.info(f'진입 신호: {r["ticker"]} — {er["reason"]}')

        # 8) 저장
        save_watch_list(new_watch_list)

        # 9) 텔레그램
        if new_entries:
            send_telegram(build_new_entry_msg(new_entries, macro))
        if entry_signals:
            send_telegram(build_entry_signal_msg(entry_signals))

        # 10) 상태 업데이트
        with _state_lock:
            scanner_state.update({
                'status'       : 'done',
                'last_scan_at' : now_utc.isoformat(),
                'scan_count'   : scanner_state['scan_count'] + 1,
                'watch_list'   : new_watch_list,
                'new_entries'  : new_entries,
                'entry_signals': entry_signals,
                'removed'      : removed_list,
                'macro'        : macro,
                'total_scanned': len(targets),
            })

        log.info(f'스캔 완료 | 대상 {len(targets)}개 | Watch {len(new_watch_list)}개 | '
                 f'신규 {len(new_entries)}개 | 진입 {len(entry_signals)}개')

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
        time.sleep(SCAN_INTERVAL_MIN * 60)


if __name__ == '__main__':
    run_scan()
