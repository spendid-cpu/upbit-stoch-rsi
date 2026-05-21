# -*- coding: utf-8 -*-
"""
scanner.py — Upbit MTF 자동 스캐너 (v1.2)
변경사항:
  - USDE(Ethena) 스테이블코인 추가
  - Watch 등록 시 entry_price(등록 당시 가격) 저장
  - analyze_ticker에서 Stoch RSI K/D 시계열 저장 (차트용)
  - get_current_prices() 현재가 일괄 조회
  - manual_scan() 수동 스캔 트리거
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
WATCH_LIST_FILE     = os.environ.get('WATCH_LIST_FILE', 'watch_list.json')
SIGNAL_HISTORY_FILE = os.environ.get('SIGNAL_HISTORY_FILE', 'signal_history.json')

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT  = os.environ.get('TELEGRAM_CHAT_ID', '')

# ── 스테이블코인 제외 목록 ─────────────────────────
STABLE_COINS = {
    'KRW-USDT', 'KRW-USDC', 'KRW-BUSD', 'KRW-DAI',
    'KRW-TUSD', 'KRW-USDP', 'KRW-GUSD', 'KRW-FRAX',
    'KRW-USDD', 'KRW-FDUSD', 'KRW-PYUSD',
    'KRW-USDE',   # Ethena USDe
    'KRW-USDS',   # USDS
    'KRW-USD0',   # USD0
}

# ── 공유 상태 ─────────────────────────────────────
_state_lock   = threading.Lock()
_manual_event = threading.Event()  # 수동 스캔 트리거

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
    'current_prices': {},   # { ticker: {price, change_pct} }
    'chart_data'    : {},   # { ticker: {daily_k, daily_d, h4_k, h4_d, h1_k, h1_d} }
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

def get_current_prices(tickers):
    """
    여러 종목의 현재가·등락률 일괄 조회.
    Returns: { ticker: {'price': float, 'change_pct': float} }
    """
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
                    'price'      : d.get('trade_price', 0),
                    'change_pct' : d.get('signed_change_rate', 0) * 100,
                }
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log.debug(f'현재가 조회 오류: {e}')
    return result

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

def fmt_price(p):
    """가격 포맷 (1원 미만은 소수점 표시)."""
    if p is None:
        return '—'
    if p >= 100:
        return f'{p:,.0f}'
    elif p >= 1:
        return f'{p:,.2f}'
    else:
        return f'{p:,.4f}'

def build_new_entry_msg(items, macro_state, prices):
    lines = [f'📋 <b>Watch List 신규 등록 {len(items)}개</b>']
    macro_icon = '✅' if macro_state['safe'] else '🚫'
    w_dist = macro_state.get('weekly_distance_pct', 0) or 0
    d_dist = macro_state.get('daily_distance_pct',  0) or 0
    lines.append(f'BTC 주봉MA20({macro_icon}): {w_dist:+.2f}% | 일봉MA20(참고): {d_dist:+.2f}%')
    lines.append('')
    for item in items:
        t   = item['ticker']
        p   = prices.get(t, {})
        prc = fmt_price(p.get('price'))
        chg = p.get('change_pct', 0)
        lines.append(
            f'• <b>{t}</b>  일봉K {item["daily_short_k"]:.1f} | '
            f'현재가 {prc}원 ({chg:+.2f}%)'
        )
    return '\n'.join(lines)

def build_entry_signal_msg(items, prices):
    lines = [f'🚀 <b>진입 트리거 {len(items)}개</b>']
    lines.append('')
    for item in items:
        t   = item['ticker']
        tr  = item['trigger']
        p   = prices.get(t, {})
        prc = fmt_price(p.get('price'))
        chg = p.get('change_pct', 0)
        ep  = fmt_price(item.get('entry_price'))
        lines.append(
            f'⚡ <b>{t}</b>  현재가 {prc}원 ({chg:+.2f}%)\n'
            f'   4h K {tr["h4_short_k"]:.1f} | '
            f'1h K {tr["h1_short_k"]:.1f} | '
            f'{tr["h1_trigger_type"]}\n'
            f'   Watch 등록가: {ep}원'
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

        # Stoch RSI 시계열 (차트용) — 최근 50개만
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

            entry_result = mtf_setup.evaluate_entry_trigger(h4, h1, ticker)

        return {
            'ticker'       : ticker,
            'watch_result' : watch_result,
            'entry_result' : entry_result,
            'daily_short_k': watch_result.get('daily_short_k'),
            'chart'        : chart,
            'current_price': daily[-1] if daily else None,
        }
    except Exception as e:
        log.debug(f'{ticker} 분석 오류: {e}')
        return None

# ===================== 수동 스캔 트리거 =====================
def manual_scan():
    """dashboard.py의 /api/scan 엔드포인트에서 호출."""
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

    try:
        # 1) BTC 매크로
        btc_daily, btc_weekly = get_btc_closes()
        macro  = mtf_setup.evaluate_macro_filter(btc_daily, btc_weekly, ticker='BTC')
        w_dist = macro.get('weekly_distance_pct', 0) or 0
        d_dist = macro.get('daily_distance_pct',  0) or 0
        log.info(f'매크로: safe={macro["safe"]} | 주봉MA20 {w_dist:+.2f}% | 일봉MA20 {d_dist:+.2f}%')

        # 2) Watch List 불러오기
        watch_list     = load_watch_list()
        watch_list_map = {w['ticker']: w for w in watch_list}

        # 3) 전 종목 스캔
        targets = get_all_krw_markets()

        # 4) 병렬 분석
        results    = []
        chart_data = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(analyze_ticker, t, watch_list_map): t
                for t in targets
            }
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

        # 6) 신규 등록 (등록 당시 가격 저장)
        new_entries      = []
        existing_tickers = {w['ticker'] for w in new_watch_list}
        results_map      = {r['ticker']: r for r in results}

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
                    'entry_price'  : r.get('current_price'),  # ← 등록 시 가격
                }
                new_watch_list.append(entry)
                new_entries.append(entry)
                existing_tickers.add(ticker)
                log.info(f'Watch 등록: {ticker} K={wr["daily_short_k"]:.1f} '
                         f'@ {fmt_price(entry["entry_price"])}원')

        # 7) 진입 트리거
        entry_signals = []
        for r in results:
            er = r.get('entry_result')
            if er and er['should_enter']:
                # Watch List에 있는 종목의 entry_price 참조
                watch_item = next(
                    (w for w in new_watch_list if w['ticker'] == r['ticker']), {}
                )
                signal = {
                    'ticker'      : r['ticker'],
                    'triggered_at': now_utc.isoformat(),
                    'trigger'     : er,
                    'entry_price' : watch_item.get('entry_price'),
                }
                entry_signals.append(signal)
                append_signal_history(signal)
                log.info(f'진입 신호: {r["ticker"]} — {er["reason"]}')

        # 8) 현재가 조회 (Watch List 종목 + 진입 신호 종목)
        price_tickers = list({
            w['ticker'] for w in new_watch_list
        } | {s['ticker'] for s in entry_signals})
        current_prices = get_current_prices(price_tickers)

        # Watch List에 현재가 업데이트
        for item in new_watch_list:
            t = item['ticker']
            if t in current_prices:
                item['current_price']  = current_prices[t]['price']
                item['change_pct']     = current_prices[t]['change_pct']

        # 9) 저장
        save_watch_list(new_watch_list)

        # 10) 텔레그램
        if new_entries:
            send_telegram(build_new_entry_msg(new_entries, macro, current_prices))
        if entry_signals:
            send_telegram(build_entry_signal_msg(entry_signals, current_prices))

        # 11) 상태 업데이트
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
            })

        log.info(f'스캔 완료 | {len(targets)}개 | Watch {len(new_watch_list)}개 | '
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
        # 수동 스캔 이벤트 대기 (타임아웃 = 주기)
        _manual_event.wait(timeout=SCAN_INTERVAL_MIN * 60)
        _manual_event.clear()


if __name__ == '__main__':
    run_scan()
