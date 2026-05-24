"""
mtf_setup.py v3.0.1
변경사항:
- 점수 기반 등급 일치 수정
- 주봉 MA20 계산 추가
- BTC 일봉/주봉 MA20 동시 반환
"""

import numpy as np
from typing import Optional

VERSION = 'v3.0.1'

# ── 파라미터 정의 ─────────────────────────────────────────────
PARAMS = {
    'short':  {'rsi': 5,  'stoch': 5,  'k_smooth': 3,  'd_smooth': 3},
    'mid':    {'rsi': 10, 'stoch': 10, 'k_smooth': 6,  'd_smooth': 6},
    'long':   {'rsi': 20, 'stoch': 20, 'k_smooth': 12, 'd_smooth': 12},
}

OVERSOLD   = 20.0
OVERBOUGHT = 80.0


# ══════════════════════════════════════════════════════════════
# 핵심 계산 함수
# ══════════════════════════════════════════════════════════════

def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    if len(closes) < period + 1:
        return np.full(len(closes), np.nan)
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full(len(closes), np.nan)
    avg_loss = np.full(len(closes), np.nan)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    with np.errstate(divide='ignore', invalid='ignore'):
        rs  = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi = np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))

    rsi[:period] = np.nan
    return rsi


def _stoch_rsi_k(rsi_arr: np.ndarray, stoch_period: int, k_smooth: int) -> np.ndarray:
    n     = len(rsi_arr)
    raw_k = np.full(n, np.nan)

    for i in range(stoch_period - 1, n):
        window = rsi_arr[i - stoch_period + 1 : i + 1]
        valid  = window[~np.isnan(window)]
        if len(valid) < stoch_period:
            continue
        lo, hi = valid.min(), valid.max()
        raw_k[i] = 50.0 if hi == lo else (rsi_arr[i] - lo) / (hi - lo) * 100.0

    k_smooth_arr = np.full(n, np.nan)
    for i in range(k_smooth - 1, n):
        window = raw_k[i - k_smooth + 1 : i + 1]
        valid  = window[~np.isnan(window)]
        if len(valid) == k_smooth:
            k_smooth_arr[i] = valid.mean()

    return k_smooth_arr


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1 : i + 1]
        valid  = window[~np.isnan(window)]
        if len(valid) == period:
            result[i] = valid.mean()
    return result


def calc_stoch_rsi(closes: list, term: str) -> dict:
    p          = PARAMS[term]
    closes_arr = np.array(closes, dtype=float)
    min_len    = p['rsi'] + p['stoch'] + p['k_smooth'] + p['d_smooth'] + 5

    if len(closes_arr) < min_len:
        return _empty_result()

    rsi_arr = _rsi(closes_arr, p['rsi'])
    k_arr   = _stoch_rsi_k(rsi_arr, p['stoch'], p['k_smooth'])
    d_arr   = _sma(k_arr, p['d_smooth'])

    k_vals = k_arr[~np.isnan(k_arr)]
    d_vals = d_arr[~np.isnan(d_arr)]

    if len(k_vals) < 2 or len(d_vals) < 2:
        return _empty_result()

    k      = round(float(k_vals[-1]), 2)
    k_prev = round(float(k_vals[-2]), 2)
    d      = round(float(d_vals[-1]), 2)
    d_prev = round(float(d_vals[-2]), 2)

    golden_cross = (k_prev <= d_prev) and (k > d)
    dead_cross   = (k_prev >= d_prev) and (k < d)

    if k <= OVERSOLD:
        zone = 'oversold'
    elif k >= OVERBOUGHT:
        zone = 'overbought'
    else:
        zone = 'neutral'

    signal = _get_signal(k, d, k_prev, d_prev, golden_cross, dead_cross, zone)

    return {
        'k':            k,
        'd':            d,
        'k_prev':       k_prev,
        'd_prev':       d_prev,
        'golden_cross': golden_cross,
        'dead_cross':   dead_cross,
        'zone':         zone,
        'signal':       signal,
    }


def _get_signal(k, d, k_prev, d_prev, golden_cross, dead_cross, zone) -> str:
    if zone == 'overbought' and (dead_cross or k < d):
        return 'BUY_NO'
    if zone == 'oversold' and golden_cross:
        return 'BUY_OK'
    if zone == 'oversold' and k <= d:
        return 'WATCH'
    if k > OVERSOLD and k_prev <= OVERSOLD and k > d:
        return 'BUY_OK'
    return 'NEUTRAL'


def _empty_result() -> dict:
    return {
        'k': None, 'd': None,
        'k_prev': None, 'd_prev': None,
        'golden_cross': False, 'dead_cross': False,
        'zone': 'unknown', 'signal': 'NEUTRAL',
    }


# ══════════════════════════════════════════════════════════════
# 멀티 타임프레임 통합 분석
# ══════════════════════════════════════════════════════════════

def analyze_mtf(candles: dict) -> dict:
    result = {}
    for tf in ('daily', 'h4', 'h1'):
        closes     = candles.get(tf, [])
        result[tf] = {}
        for term in ('short', 'mid', 'long'):
            result[tf][term] = calc_stoch_rsi(closes, term)

    result['summary'] = _summarize(result)
    return result


def _summarize(mtf: dict) -> dict:
    daily = mtf.get('daily', {})
    h4    = mtf.get('h4',    {})
    h1    = mtf.get('h1',    {})

    d_long   = daily.get('long',  _empty_result())
    d_mid    = daily.get('mid',   _empty_result())
    d_short  = daily.get('short', _empty_result())
    h4_short = h4.get('short',    _empty_result())
    h1_short = h1.get('short',    _empty_result())

    # 진입 차단
    any_buy_no = any([
        d_long.get('signal')   == 'BUY_NO',
        d_short.get('signal')  == 'BUY_NO',
        h4_short.get('signal') == 'BUY_NO',
        h1_short.get('signal') == 'BUY_NO',
    ])

    # 과매도 여부
    d_long_os  = d_long.get('zone')  == 'oversold'
    d_mid_os   = d_mid.get('zone')   == 'oversold'
    d_short_os = d_short.get('zone') == 'oversold'

    # 골든크로스
    h4_gc = h4_short.get('signal') == 'BUY_OK'
    h1_gc = h1_short.get('signal') == 'BUY_OK'
    d_gc  = d_short.get('signal')  == 'BUY_OK'

    # Watch 등록 조건
    watch_eligible = d_long_os and not any_buy_no

    # 점수 계산
    score = _calc_score(d_long, d_mid, d_short, h4_short, h1_short)

    # ── 점수 기반 등급 (점수와 등급 완전 일치) ──
    if any_buy_no:
        grade = 'X'
    elif score >= 80:
        grade = 'S'
    elif score >= 65:
        grade = 'A'
    elif score >= 45:
        grade = 'B'
    elif d_long_os:
        grade = 'C'
    else:
        grade = '-'

    # 자동진입: 일봉 장기 과매도 + 4h 또는 1h 골든크로스
    auto_entry = (
        not any_buy_no and
        d_long_os and
        (h4_gc or h1_gc)
    )

    return {
        'grade':          grade,
        'watch_eligible': watch_eligible,
        'auto_entry':     auto_entry,
        'any_buy_no':     any_buy_no,
        'score':          score,
        'd_long_os':      d_long_os,
        'd_mid_os':       d_mid_os,
        'd_short_os':     d_short_os,
        'h4_gc':          h4_gc,
        'h1_gc':          h1_gc,
        'd_gc':           d_gc,
    }


def _calc_score(d_long, d_mid, d_short, h4_short, h1_short) -> int:
    score = 0

    # 일봉 장기 과매도 (30점 + 극단 보너스 10점)
    if d_long.get('zone') == 'oversold':
        k = d_long.get('k') or 50
        score += 30
        if k <= 10:
            score += 10

    # 일봉 장기 골든크로스 (10점)
    if d_long.get('signal') == 'BUY_OK':
        score += 10

    # 일봉 중기 과매도 (10점) + 골든크로스 (5점)
    if d_mid.get('zone') == 'oversold':
        score += 10
    if d_mid.get('signal') == 'BUY_OK':
        score += 5

    # 일봉 단기 과매도 (5점) + 골든크로스 (5점)
    if d_short.get('zone') == 'oversold':
        score += 5
    if d_short.get('signal') == 'BUY_OK':
        score += 5

    # 4h 골든크로스 (20점) / 과매도만 (8점)
    if h4_short.get('signal') == 'BUY_OK':
        score += 20
    elif h4_short.get('zone') == 'oversold':
        score += 8

    # 1h 골든크로스 (15점) / 과매도만 (5점)
    if h1_short.get('signal') == 'BUY_OK':
        score += 15
    elif h1_short.get('zone') == 'oversold':
        score += 5

    # 패널티
    if d_long.get('signal')   == 'BUY_NO':
        score = max(0, score - 40)
    if h4_short.get('signal') == 'BUY_NO':
        score = max(0, score - 20)
    if h1_short.get('signal') == 'BUY_NO':
        score = max(0, score - 10)

    return min(score, 100)


# ══════════════════════════════════════════════════════════════
# BTC MA20 (일봉 + 주봉)
# ══════════════════════════════════════════════════════════════

def btc_ma20_signal(btc_daily_closes: list, btc_weekly_closes: list = None) -> dict:
    """
    일봉 MA20 + 주봉 MA20 동시 계산
    Returns:
    {
      'price':          현재가,
      'daily_ma20':     일봉 MA20,
      'daily_above':    현재가 > 일봉MA20,
      'daily_pct':      일봉MA20 대비 %,
      'weekly_ma20':    주봉 MA20,
      'weekly_above':   현재가 > 주봉MA20,
      'weekly_pct':     주봉MA20 대비 %,
    }
    """
    result = {
        'price':        None,
        'daily_ma20':   None,
        'daily_above':  None,
        'daily_pct':    0.0,
        'weekly_ma20':  None,
        'weekly_above': None,
        'weekly_pct':   0.0,
    }

    if not btc_daily_closes or len(btc_daily_closes) < 20:
        return result

    price = float(btc_daily_closes[-1])
    result['price'] = price

    # 일봉 MA20
    daily_arr       = np.array(btc_daily_closes[-20:], dtype=float)
    daily_ma20      = float(daily_arr.mean())
    result['daily_ma20']  = round(daily_ma20, 0)
    result['daily_above'] = price > daily_ma20
    result['daily_pct']   = round((price - daily_ma20) / daily_ma20 * 100, 2)

    # 주봉 MA20
    if btc_weekly_closes and len(btc_weekly_closes) >= 20:
        weekly_arr        = np.array(btc_weekly_closes[-20:], dtype=float)
        weekly_ma20       = float(weekly_arr.mean())
        result['weekly_ma20']  = round(weekly_ma20, 0)
        result['weekly_above'] = price > weekly_ma20
        result['weekly_pct']   = round((price - weekly_ma20) / weekly_ma20 * 100, 2)

    return result


# ══════════════════════════════════════════════════════════════
# DEEP 상대강도
# ══════════════════════════════════════════════════════════════

def calc_relative_strength(coin_pct: float, btc_pct: float) -> dict:
    rs = round(coin_pct - btc_pct, 2)

    if rs >= 5.0:
        grade, signal = 'S', 'DEEP_STRONG'
    elif rs >= 3.0:
        grade, signal = 'A', 'DEEP_GOOD'
    elif rs >= 2.0:
        grade, signal = 'B', 'DEEP_WATCH'
    elif rs >= 1.0:
        grade, signal = 'C', 'DEEP_MONITOR'
    else:
        grade, signal = '-', 'NEUTRAL'

    return {'rs': rs, 'grade': grade, 'signal': signal}
