"""
mtf_setup.py v3.0.0
Stochastic RSI 단기/중기/장기 K+D 계산
- 단기: RSI5, Stoch5, K3, D3
- 중기: RSI10, Stoch10, K6, D6
- 장기: RSI20, Stoch20, K12, D12
"""

import numpy as np
from typing import Optional

VERSION = 'v3.0.0'

# ── 파라미터 정의 ─────────────────────────────────────────────
PARAMS = {
    'short':  {'rsi': 5,  'stoch': 5,  'k_smooth': 3,  'd_smooth': 3},
    'mid':    {'rsi': 10, 'stoch': 10, 'k_smooth': 6,  'd_smooth': 6},
    'long':   {'rsi': 20, 'stoch': 20, 'k_smooth': 12, 'd_smooth': 12},
}

OVERSOLD  = 20.0
OVERBOUGHT = 80.0


# ── 핵심 계산 함수 ────────────────────────────────────────────

def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """RSI 계산"""
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

    rs  = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi


def _stoch_rsi_k(rsi_arr: np.ndarray, stoch_period: int, k_smooth: int) -> np.ndarray:
    """Stochastic RSI Raw K → Smoothed K 계산"""
    n = len(rsi_arr)
    raw_k = np.full(n, np.nan)

    for i in range(stoch_period - 1, n):
        window = rsi_arr[i - stoch_period + 1 : i + 1]
        valid  = window[~np.isnan(window)]
        if len(valid) < stoch_period:
            continue
        lo, hi = valid.min(), valid.max()
        if hi == lo:
            raw_k[i] = 50.0
        else:
            raw_k[i] = (rsi_arr[i] - lo) / (hi - lo) * 100.0

    # K 스무딩 (SMA)
    k_smooth_arr = np.full(n, np.nan)
    for i in range(k_smooth - 1, n):
        window = raw_k[i - k_smooth + 1 : i + 1]
        valid  = window[~np.isnan(window)]
        if len(valid) == k_smooth:
            k_smooth_arr[i] = valid.mean()

    return k_smooth_arr


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """단순 이동평균"""
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1 : i + 1]
        valid  = window[~np.isnan(window)]
        if len(valid) == period:
            result[i] = valid.mean()
    return result


def calc_stoch_rsi(closes: list, term: str) -> dict:
    """
    단일 텀의 StochRSI K, D 계산
    Returns:
        {
          'k': float,        # 현재 K값
          'd': float,        # 현재 D값
          'k_prev': float,   # 이전 K값 (골든크로스 판단용)
          'd_prev': float,   # 이전 D값
          'golden_cross': bool,   # K가 D를 상향 돌파
          'dead_cross':   bool,   # K가 D를 하향 돌파
          'zone': str,       # 'oversold' / 'overbought' / 'neutral'
          'signal': str,     # 'BUY_OK' / 'BUY_NO' / 'WATCH' / 'NEUTRAL'
        }
    """
    p = PARAMS[term]
    closes_arr = np.array(closes, dtype=float)

    # 최소 데이터 길이 확인
    min_len = p['rsi'] + p['stoch'] + p['k_smooth'] + p['d_smooth'] + 5
    if len(closes_arr) < min_len:
        return _empty_result()

    rsi_arr = _rsi(closes_arr, p['rsi'])
    k_arr   = _stoch_rsi_k(rsi_arr, p['stoch'], p['k_smooth'])
    d_arr   = _sma(k_arr, p['d_smooth'])

    # 현재/이전 값 추출
    k_vals = k_arr[~np.isnan(k_arr)]
    d_vals = d_arr[~np.isnan(d_arr)]

    if len(k_vals) < 2 or len(d_vals) < 2:
        return _empty_result()

    k      = round(float(k_vals[-1]), 2)
    k_prev = round(float(k_vals[-2]), 2)
    d      = round(float(d_vals[-1]), 2)
    d_prev = round(float(d_vals[-2]), 2)

    # 골든크로스 / 데드크로스 판단
    golden_cross = (k_prev <= d_prev) and (k > d)
    dead_cross   = (k_prev >= d_prev) and (k < d)

    # 구간 판단
    if k <= OVERSOLD:
        zone = 'oversold'
    elif k >= OVERBOUGHT:
        zone = 'overbought'
    else:
        zone = 'neutral'

    # 시그널 판단
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
    """
    시그널 판단 로직
    - BUY_OK : 과매도 구간에서 K가 D 상향 돌파 (바닥 반등 시작)
    - BUY_NO : 과매수 구간에서 K가 D 하향 돌파 (하락 중)
    - WATCH  : 과매도 구간이지만 아직 골든크로스 미발생
    - NEUTRAL: 그 외
    """
    # 절대 NO: 과매수 + 데드크로스
    if zone == 'overbought' and dead_cross:
        return 'BUY_NO'
    if zone == 'overbought' and k < d:
        return 'BUY_NO'

    # 최적 진입: 과매도 + 골든크로스
    if zone == 'oversold' and golden_cross:
        return 'BUY_OK'

    # 과매도 대기: 골든크로스 미발생
    if zone == 'oversold' and k <= d:
        return 'WATCH'

    # 과매도 상승 중 (골든크로스 이후 K가 20 돌파)
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


# ── 멀티 타임프레임 통합 분석 ─────────────────────────────────

def analyze_mtf(candles: dict) -> dict:
    """
    candles = {
        'daily': [close, close, ...],   # 최소 100개 권장
        'h4':    [close, close, ...],   # 최소 100개 권장
        'h1':    [close, close, ...],   # 최소 100개 권장
    }
    Returns full MTF analysis result.
    """
    result = {}
    for tf in ('daily', 'h4', 'h1'):
        closes = candles.get(tf, [])
        result[tf] = {}
        for term in ('short', 'mid', 'long'):
            result[tf][term] = calc_stoch_rsi(closes, term)

    # 종합 판단
    result['summary'] = _summarize(result)
    return result


def _summarize(mtf: dict) -> dict:
    """
    종합 시그널 및 등급 산출
    등급 기준:
      S: 일봉 장기+중기+단기 과매도 + 4h/1h 단기 골든크로스
      A: 일봉 장기+단기 과매도 + 4h 단기 골든크로스
      B: 일봉 장기 과매도 + (4h 또는 1h) 단기 골든크로스
      C: 일봉 장기 과매도만
      -: 조건 미충족
    """
    daily = mtf.get('daily', {})
    h4    = mtf.get('h4', {})
    h1    = mtf.get('h1', {})

    d_long  = daily.get('long',  _empty_result())
    d_mid   = daily.get('mid',   _empty_result())
    d_short = daily.get('short', _empty_result())
    h4_short = h4.get('short',  _empty_result())
    h1_short = h1.get('short',  _empty_result())

    # 진입 차단 조건
    any_buy_no = any([
        d_long.get('signal')  == 'BUY_NO',
        d_short.get('signal') == 'BUY_NO',
        h4_short.get('signal') == 'BUY_NO',
        h1_short.get('signal') == 'BUY_NO',
    ])

    # 과매도 여부
    d_long_os  = d_long.get('zone')  == 'oversold'
    d_mid_os   = d_mid.get('zone')   == 'oversold'
    d_short_os = d_short.get('zone') == 'oversold'

    # 골든크로스 여부
    h4_gc  = h4_short.get('golden_cross', False) or h4_short.get('signal') == 'BUY_OK'
    h1_gc  = h1_short.get('golden_cross', False) or h1_short.get('signal') == 'BUY_OK'
    d_gc   = d_short.get('golden_cross', False)  or d_short.get('signal')  == 'BUY_OK'

    # Watch 등록 조건: 일봉 장기 과매도
    watch_eligible = d_long_os

    # 등급 산출
    if any_buy_no:
        grade = 'X'   # 진입 금지
    elif d_long_os and d_mid_os and d_short_os and h4_gc and h1_gc:
        grade = 'S'
    elif d_long_os and d_short_os and h4_gc:
        grade = 'A'
    elif d_long_os and (h4_gc or h1_gc):
        grade = 'B'
    elif d_long_os:
        grade = 'C'
    else:
        grade = '-'

    # 자동 진입 가능 여부
    auto_entry = (
        not any_buy_no and
        d_long_os and
        (h4_gc or h1_gc) and
        grade in ('S', 'A', 'B')
    )

    # 진입 강도 점수 (0~100)
    score = _calc_score(d_long, d_mid, d_short, h4_short, h1_short)

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
    """진입 강도 점수 산출 (0~100)"""
    score = 0

    # 일봉 장기 과매도 (핵심, 30점)
    if d_long.get('zone') == 'oversold':
        k = d_long.get('k') or 50
        score += 30
        if k <= 10:
            score += 10   # 극단 과매도 보너스

    # 일봉 중기 과매도 (15점)
    if d_mid.get('zone') == 'oversold':
        score += 15

    # 일봉 단기 과매도 (10점)
    if d_short.get('zone') == 'oversold':
        score += 10

    # 4h 골든크로스 (20점)
    if h4_short.get('signal') in ('BUY_OK',):
        score += 20
    elif h4_short.get('zone') == 'oversold':
        score += 8

    # 1h 골든크로스 (15점)
    if h1_short.get('signal') in ('BUY_OK',):
        score += 15
    elif h1_short.get('zone') == 'oversold':
        score += 5

    # 진입 차단 패널티
    if d_long.get('signal') == 'BUY_NO':
        score = max(0, score - 40)
    if h4_short.get('signal') == 'BUY_NO':
        score = max(0, score - 20)

    return min(score, 100)


# ── BTC MA20 필터 ─────────────────────────────────────────────

def btc_ma20_signal(btc_closes: list) -> dict:
    """
    BTC 가격이 MA20 대비 위/아래 여부 판단
    Returns:
        { 'price': float, 'ma20': float, 'above': bool, 'pct': float }
    """
    if len(btc_closes) < 20:
        return {'price': None, 'ma20': None, 'above': None, 'pct': 0.0}

    arr   = np.array(btc_closes[-20:], dtype=float)
    ma20  = float(arr.mean())
    price = float(btc_closes[-1])
    pct   = round((price - ma20) / ma20 * 100, 2)

    return {
        'price': price,
        'ma20':  round(ma20, 2),
        'above': price > ma20,
        'pct':   pct,
    }


# ── DEEP 상대강도 계산 ────────────────────────────────────────

def calc_relative_strength(coin_pct: float, btc_pct: float) -> dict:
    """
    BTC 대비 상대강도 계산
    coin_pct, btc_pct: 변화율(%) 예) -2.5, -5.0
    Returns:
        { 'rs': float, 'grade': str, 'signal': str }
    """
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
