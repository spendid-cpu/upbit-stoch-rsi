# -*- coding: utf-8 -*-
"""
scanner.py — Upbit MTF 자동 스캐너 (v1.0)
dashboard.py의 백그라운드 스레드에서 호출되거나 단독 실행 가능.
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
SCAN_TARGET_COUNT   = int(os.environ.get('SCAN_TARGET_COUNT', 60))
MIN_TRADE_VALUE_KRW = float(os.environ.get('MIN_TRADE_VALUE_KRW', 3_000_000_000))
REQUEST_DELAY       = float(os.environ.get('REQUEST_DELAY', 0.12))
MAX_WORKERS         = int(os.environ.get('MAX_WORKERS', 6))
CANDLE_COUNT        = int(os.environ.get('CANDLE_COUNT', 200))
WATCH_LIST_FILE     = os.environ.get('WATCH_LIST_FILE', 'watch_list.json')
SIGNAL_HISTORY_FILE = os.environ.get('SIGNAL_HISTORY_FILE', 'signal_history.json')

TELEGRAM_TOKEN  = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT   = os.environ.get('TELEGRAM_CHAT_ID', '')

# ── 공유 상태 (dashboard.py에서 읽음) ────────────
_state_lock = threading.Lock()
scanner_state = {
    'status'       : 'idle',       # idle | scanning | done | error
    'last_scan_at' : None,
    'next_scan_at' : None,
    'scan_count'   : 0,
    'watch_list'   : [],           # 현재 Watch List
    'new_entries'  : [],           # 이번 스캔에서 신규 등록된 종목
    'entry_signals': [],           # 진입 트리거 발동 종목
    'removed'      : [],           # 이번 스캔에서 제거된 종목
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

def get_krw_markets():
    data = _get('https://api.upbit.com/v1/market/all', {'isDetails': 'true'})
    return [
        d['market'] for d in data
        if d['market'].startswith('KRW-')
        and d.get('market_warning', 'NONE') == 'NONE'
        and d['market'] not in ('KRW-BTC', 'KRW-USDT')
    ]

def get_top_markets(limit=60):
    markets = get_krw_markets()
    chunks  = [markets[i:i+100] for i in range(0, len(markets), 100)]
    tickers = []
    for chunk in chunks:
        data = _get('https://api.upbit.com/v1/ticker',
                    {'markets': ','.join(chunk)})
        tickers.extend(data)
        time.sleep(REQUEST_DELAY)
    tickers = [t for t in tickers
               if t.get('acc_trade_price_24h', 0) >= MIN_TRADE_VALUE_KRW]
    tickers.sort(key=lambda x: x['acc_trade_price_24h'], reverse=True)
    return [t['market'] for t in tickers[:limit]]

def get_closes(ticker, interval, count=200):
    """종가 리스트만 반환."""
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
        # Upbit API는 최신 → 과거 순으로 반환 → 역순 정렬
        data.sort(key=lambda x: x['candle_date_time_utc'])
        return [float(c['trade_price']) for c in data]
    except Exception as e:
        log.debug(f'{ticker} {interval} 캔들 오류: {e}')
        return []

def get_btc_closes():
    """BTC 일봉·주봉 종가 반환."""
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
    history = history[-500:]  # 최대 500개 보관
    with open(SIGNAL_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ===================== 텔레그램 =====================
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    MAX_LEN = 4000
    chunks = [text[i:i+MAX_LEN] for i in range(0, len(text), MAX_LEN)]
    for chunk in chunks:
        try:
            requests.post(url, json={
                'chat_id': TELEGRAM_CHAT,
                'text': chunk,
                'parse_mode': 'HTML'
            }, timeout=10)
        except Exception as e:
            log.warning(f'텔레그램 전송 오류: {e}')

def build_new_entry_msg(items, macro_state):
    lines = [f'📋 <b>Watch List 신규 등록 {len(items)}개</b>']
    macro_icon = '✅' if macro_state['safe'] else '🚫'
    lines.append(f'매크로({macro_icon}): {macro_state["reason"]}')
    lines.append('')
    for item in items:
        lines.append(
            f'• <b>{item["ticker"]}</b>  '
            f'일봉단기K {item["daily_short_k"]:.1f}'
        )
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
def analyze_ticker(ticker, btc_macro, watch_list_map):
    """
    한 종목 분석.
    Returns: dict { ticker, watch_result, entry_result, daily_k, status }
    """
    try:
        daily = get_closes(ticker, 'day',   count=CANDLE_COUNT)
        time.sleep(REQUEST_DELAY)

        if len(daily) < 60:
            return None

        watch_result = mtf_setup.evaluate_watch_list_entry(daily, ticker)

        # 진입 트리거는 Watch List에 있는 종목만
        entry_result = None
        if ticker in watch_list_map or watch_result['should_register']:
            h4 = get_closes(ticker, 'minutes4', count=CANDLE_COUNT)
            time.sleep(REQUEST_DELAY)
            h1 = get_closes(ticker, 'minutes1', count=CANDLE_COUNT)
            time.sleep(REQUEST_DELAY)
            entry_result = mtf_setup.evaluate_entry_trigger(h4, h1, ticker)

        return {
            'ticker'      : ticker,
            'watch_result': watch_result,
            'entry_result': entry_result,
            'daily_short_k': watch_result.get('daily_short_k'),
        }
    except Exception as e:
        log.debug(f'{ticker} 분석 오류: {e}')
        return None

# ===================== 메인 스캔 루프 =====================
def run_scan():
    """한 번의 전체 스캔 실행."""
    now_utc = datetime.now(timezone.utc)
    log.info(f'=== 스캔 시작 {now_utc.strftime("%Y-%m-%d %H:%M UTC")} ===')

    with _state_lock:
        scanner_state['status']    = 'scanning'
        scanner_state['error']     = None
        scanner_state['new_entries']   = []
        scanner_state['entry_signals'] = []
        scanner_state['removed']       = []

    try:
        # 1) BTC 매크로 필터
        btc_daily, btc_weekly = get_btc_closes()
        macro = mtf_setup.evaluate_macro_filter(
            btc_daily, btc_weekly, ticker='BTC'
        )
        log.info(f'매크로: {macro["reason"]}')

        # 2) Watch List 불러오기
        watch_list = load_watch_list()
        watch_list_map = {w['ticker']: w for w in watch_list}

        # 3) 스캔 대상 종목 가져오기
        targets = get_top_markets(limit=SCAN_TARGET_COUNT)
        log.info(f'스캔 대상: {len(targets)}개')

        # 4) 병렬 분석
        new_entries    = []
        entry_signals  = []
        removed_list   = []
        results        = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(analyze_ticker, t, macro, watch_list_map): t
                for t in targets
            }
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)

        # 5) Watch List 무효화 처리 (스캔 대상에 없는 종목도 포함)
        results_map = {r['ticker']: r for r in results}
        new_watch_list = []
        for item in watch_list:
            ticker = item['ticker']
            # 무효화 판정
            daily = get_closes(ticker, 'day', count=CANDLE_COUNT)
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

        # 6) 신규 Watch List 등록
        existing_tickers = {w['ticker'] for w in new_watch_list}
        for r in results:
            wr = r['watch_result']
            ticker = r['ticker']
            if wr['should_register'] and ticker not in existing_tickers:
                # 매크로 필터 통과 시에만 등록
                if mtf_setup.USE_MACRO_FILTER and not macro['safe']:
                    log.info(f'Watch 등록 보류(매크로 차단): {ticker}')
                    continue
                entry = {
                    'ticker'        : ticker,
                    'registered_at' : now_utc.isoformat(),
                    'daily_short_k' : wr['daily_short_k'],
                    'reason'        : wr['reason'],
                }
                new_watch_list.append(entry)
                new_entries.append(entry)
                existing_tickers.add(ticker)
                log.info(f'Watch 등록: {ticker} K={wr["daily_short_k"]:.1f}')

        # 7) 진입 트리거 체크
        for r in results:
            er = r.get('entry_result')
            if er and er['should_enter']:
                signal = {
                    'ticker'     : r['ticker'],
                    'triggered_at': now_utc.isoformat(),
                    'trigger'    : er,
                }
                entry_signals.append(signal)
                append_signal_history(signal)
                log.info(f'진입 신호: {r["ticker"]} — {er["reason"]}')

        # 8) Watch List 저장
        save_watch_list(new_watch_list)

        # 9) 텔레그램 알림
        if new_entries:
            send_telegram(build_new_entry_msg(new_entries, macro))
        if entry_signals:
            send_telegram(build_entry_signal_msg(entry_signals))

        # 10) 상태 업데이트
        with _state_lock:
            scanner_state['status']        = 'done'
            scanner_state['last_scan_at']  = now_utc.isoformat()
            scanner_state['scan_count']   += 1
            scanner_state['watch_list']    = new_watch_list
            scanner_state['new_entries']   = new_entries
            scanner_state['entry_signals'] = entry_signals
            scanner_state['removed']       = removed_list

        log.info(f'스캔 완료 | Watch {len(new_watch_list)}개 | '
                 f'신규 {len(new_entries)}개 | 진입 {len(entry_signals)}개')

    except Exception as e:
        log.error(f'스캔 오류: {e}', exc_info=True)
        with _state_lock:
            scanner_state['status'] = 'error'
            scanner_state['error']  = str(e)


def scanner_loop():
    """백그라운드 스캔 루프 (dashboard.py에서 스레드로 실행)."""
    while True:
        run_scan()
        next_at = datetime.now(timezone.utc).timestamp() + SCAN_INTERVAL_MIN * 60
        with _state_lock:
            scanner_state['next_scan_at'] = datetime.fromtimestamp(
                next_at, tz=timezone.utc
            ).isoformat()
        log.info(f'다음 스캔까지 {SCAN_INTERVAL_MIN}분 대기')
        time.sleep(SCAN_INTERVAL_MIN * 60)


# 단독 실행 시
if __name__ == '__main__':
    run_scan()
