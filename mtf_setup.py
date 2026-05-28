"""
mtf_setup.py v3.0.5 (Fixed for JSON Serialization)
변경사항:
- v3.0.3: DEEP 상대강도 보너스 추가
- v3.0.4: calc_relative_strength 가중 멀티타임프레임 평균
- v3.0.5: detect_divergence() 추가
          analyze_mtf() 반환 구조 scanner.py 호환으로 수정
- JSON 직렬화 오류 방지를 위해 Numpy 타입을 Python 기본 타입으로 캐스팅 추가 완료
"""

import numpy as np

VERSION     = 'v3.0.5'
MTF_VERSION = VERSION   # 호환용

PARAMS = {
    'short': {'rsi':  5, 'stoch':  5, 'k_smooth':  3, 'd_smooth':  3},
    'mid':   {'rsi': 10, 'stoch': 10, 'k_smooth':  5, 'd_smooth':  3},
    'long':  {'rsi': 14, 'stoch': 14, 'k_smooth': 10, 'd_smooth':  3},
}

OVERSOLD   = 20.0
OVERBOUGHT = 80.0


# ── RSI ───────────────────────────────────────────────────────────
def _rsi(closes, period):
    closes = np.array(closes, dtype=float)
    if len(closes) < period + 1:
        return []
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi_vals = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100.0
        rsi_vals.append(100.0 - 100.0 / (1.0 + rs))
    return rsi_vals


def _sma(values, period):
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(sum(values[i - period + 1:i + 1]) / period)
    return result


def _stoch_rsi_k(closes, rsi_period, stoch_window, smooth_k):
    rsi_vals = _rsi(closes, rsi_period)
    stoch_k  = []
    for i in range(len(rsi_vals)):
        if i < stoch_window - 1:
            stoch_k.append(None)
            continue
        window = rsi_vals[i - stoch_window + 1:i + 1]
        lo, hi = min(window), max(window)
        stoch_k.append(
            50.0 if hi == lo
            else (rsi_vals[i] - lo) / (hi - lo) * 100.0
        )
    valid    = [v for v in stoch_k if v is not None]
    smoothed = _sma(valid, smooth_k)
    return smoothed


def _detect_cycle(k_vals, n=5):
    recent = [v for v in k_vals[-n:] if v is not None]
    if len(recent) < 2:
        return 'RISING'
    first, last = recent[0], recent[-1]
    avg = sum(recent) / len(recent)
    if avg <= 30 and last <= 35:
        return 'BOTTOM'
    if avg >= 70 and last >= 65:
        return 'PEAK'
    if last > first:
        return 'RISING'
    return 'FALLING'


def _get_signal(zone, cross_up, cross_down):
    if zone == 'oversold' and cross_up:
        return 'BUY_OK'
    if zone == 'oversold':
        return 'BUY_WAIT'
    if zone == 'overbought' and cross_down:
        return 'BUY_NO'
    if zone == 'overbought':
        return 'BUY_NO'
    if cross_up:
        return 'BUY_OK'
    if cross_down:
        return 'BUY_NO'
    return 'NEUTRAL'


# ── 단일 StochRSI 계산 ────────────────────────────────────────────
def calc_stoch_rsi(closes, term='short'):
    p      = PARAMS[term]
    k_vals = _stoch_rsi_k(
        closes,
        p['rsi'], p['stoch'], p['k_smooth']
    )
    d_vals = _sma([v for v in k_vals if v is not None], p['d_smooth'])

    valid_k = [v for v in k_vals if v is not None]
    valid_d = [v for v in d_vals if v is not None]

    k = valid_k[-1] if valid_k else 50.0
    d = valid_d[-1] if valid_d else 50.0

    cross_up = cross_down = False
    if len(valid_k) >= 2 and len(valid_d) >= 2:
        cross_up   = valid_k[-2] <= valid_d[-2] and valid_k[-1] > valid_d[-1]
        cross_down = valid_k[-2] >= valid_d[-2] and valid_k[-1] < valid_d[-1]

    zone_str = 'oversold' if k <= OVERSOLD else 'overbought' if k >= OVERBOUGHT else 'neutral'
    signal   = _get_signal(zone_str, cross_up, cross_down)
    cycle    = _detect_cycle(valid_k)

    # 파이썬 기본 Primitive 타입 캐스팅으로 직렬화 보장
    return {
        'k':          float(round(k, 2)),
        'd':          float(round(d, 2)),
        'cross_up':   bool(cross_up),
        'cross_down': bool(cross_down),
        'zone':       str(zone_str),
        'signal':     str(signal),
        'cycle':      str(cycle),
    }


# ── MTF 분석 (scanner.py 호환 구조) ──────────────────────────────
def analyze_mtf(candle_dict):
    """
    반환 구조:
    {
        'daily': {'short': {...}, 'mid': {...}, 'long': {...}},
        'h4':    {'short': {...}, 'mid': {...}, 'long': {...}},
        'h1':    {'short': {...}, 'mid': {...}, 'long': {...}},
        'summary': {...}
    }
    """
    _empty = {
        'k': 50.0, 'd': 50.0,
        'cross_up': False, 'cross_down': False,
        'zone': 'neutral', 'signal': 'NEUTRAL', 'cycle': 'RISING',
    }

    results = {}
    for tf, closes in candle_dict.items():
        tf_result = {}
        for term in ['short', 'mid', 'long']:
            if not closes or len(closes) < 20:
                tf_result[term] = dict(_empty)
            else:
                try:
                    tf_result[term] = calc_stoch_rsi(closes, term)
                except Exception:
                    tf_result[term] = dict(_empty)
        results[tf] = tf_result

    summary = _summarize(results)
    return {**results, 'summary': summary}


def _summarize(results):
    d_long  = results.get('daily', {}).get('long',  {})
    d_mid   = results.get('daily', {}).get('mid',   {})
    d_short = results.get('daily', {}).get('short', {})
    h4      = results.get('h4',    {}).get('short', {})
    h1      = results.get('h1',    {}).get('short', {})

    score = _calc_score(d_long, d_mid, d_short, h4, h1)
    grade = _calc_grade(score)

    d_short_cycle = d_short.get('cycle', 'RISING')
    d_mid_cycle   = d_mid.get('cycle',   'RISING')
    h4_cycle      = h4.get('cycle',      'RISING')
    h1_cycle      = h1.get('cycle',      'RISING')

    h4_gc    = h4.get('cross_up',    False)
    h1_gc    = h1.get('cross_up',    False)
    daily_gc = d_short.get('cross_up', False)

    any_buy_no = (
        d_long.get('signal')  == 'BUY_NO' or
        h4.get('signal')      == 'BUY_NO' or
        h1.get('signal')      == 'BUY_NO'
    )

    bad = ('PEAK', 'FALLING')
    watch_eligible = (
        not any_buy_no and
        d_short_cycle not in bad and
        score >= 45
    )
    auto_entry = (
        watch_eligible and
        h1_gc and
        h1_cycle in ('BOTTOM', 'RISING') and
        score >= 60
    )

    return {
        'score':          int(score),
        'grade':          str(grade),
        'd_short_cycle':  str(d_short_cycle),
        'd_mid_cycle':    str(d_mid_cycle),
        'h4_cycle':       str(h4_cycle),
        'h1_cycle':       str(h1_cycle),
        'h4_gc':          bool(h4_gc),
        'h1_gc':          bool(h1_gc),
        'daily_gc':       bool(daily_gc),
        'any_buy_no':     bool(any_buy_no),
        'watch_eligible': bool(watch_eligible),
        'auto_entry':     bool(auto_entry),
    }


def _calc_score(d_long, d_mid, d_short, h4_short, h1_short,
                deep_rs_grade: str = None) -> int:
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

    # v3.0.3: DEEP RS 보너스
    deep_bonus = {'S': 15, 'A': 10, 'B': 5}
    if deep_rs_grade in deep_bonus:
        score += deep_bonus[deep_rs_grade]

    # 패널티
    if d_long.get('signal')   == 'BUY_NO':
        score = max(0, score - 40)
    if h4_short.get('signal') == 'BUY_NO':
        score = max(0, score - 20)
    if h1_short.get('signal') == 'BUY_NO':
        score = max(0, score - 10)

    # 사이클 패널티
    d_short_cycle = d_short.get('cycle', 'RISING')
    d_mid_cycle   = d_mid.get('cycle',   'RISING')
    bad = ('PEAK', 'FALLING')
    if d_short_cycle in bad:
        score = max(0, score - 15)
    if d_mid_cycle in bad:
        score = max(0, score - 10)

    return int(min(100, max(0, score)))


def _calc_grade(score, deep_rs_grade=None):
    bonus    = {'S': 15, 'A': 10, 'B': 5}.get(deep_rs_grade, 0)
    adjusted = min(100, score + bonus)
    if adjusted >= 75: return 'S'
    if adjusted >= 60: return 'A'
    if adjusted >= 45: return 'B'
    if adjusted >= 30: return 'C'
    if adjusted >= 15: return 'X'
    return '-'


# ── BTC MA20 신호 ─────────────────────────────────────────────────
def btc_ma20_signal(daily_closes, weekly_closes=None):
    if not daily_closes or len(daily_closes) < 20:
        return {
            'price':        None,
            'daily_above':  None,
            'weekly_above': None,
            'daily_ma20':   0,
            'weekly_ma20':  0,
            'daily_pct':    None,
            'weekly_pct':   None,
        }

    price       = daily_closes[-1]
    daily_ma20  = sum(daily_closes[-20:]) / 20
    daily_above = price > daily_ma20
    daily_pct   = round((price - daily_ma20) / daily_ma20 * 100, 2)

    weekly_above = None
    weekly_ma20  = 0
    weekly_pct   = None
    if weekly_closes and len(weekly_closes) >= 20:
        weekly_ma20  = sum(weekly_closes[-20:]) / 20
        weekly_above = price > weekly_ma20
        weekly_pct   = round((price - weekly_ma20) / weekly_ma20 * 100, 2)

    return {
        'price':        float(price) if price is not None else None,
        'daily_above':  bool(daily_above) if daily_above is not None else None,
        'weekly_above': bool(weekly_above) if weekly_above is not None else None,
        'daily_ma20':   int(round(daily_ma20)),
        'weekly_ma20':  int(round(weekly_ma20)),
        'daily_pct':    float(daily_pct) if daily_pct is not None else None,
        'weekly_pct':   float(weekly_pct) if weekly_pct is not None else None,
    }


# ── 상대강도 계산 (v3.0.4: 가중 멀티타임프레임) ───────────────────
def calc_relative_strength(
    coin_pct_1h=None,  btc_pct_1h=None,
    coin_pct_4h=None,  btc_pct_4h=None,
    coin_pct_24h=None, btc_pct_24h=None,
):
    weights = {'1h': 0.5, '4h': 0.3, '24h': 0.2}
    pairs   = [
        ('1h',  coin_pct_1h,  btc_pct_1h),
        ('4h',  coin_pct_4h,  btc_pct_4h),
        ('24h', coin_pct_24h, btc_pct_24h),
    ]

    rs_components = {}
    valid_weight  = 0.0
    weighted_sum  = 0.0

    for key, coin_pct, btc_pct in pairs:
        if coin_pct is None or btc_pct is None:
            continue
        rs = coin_pct - btc_pct
        rs_components[key] = round(rs, 4)
        weighted_sum  += rs * weights[key]
        valid_weight  += weights[key]

    if valid_weight == 0:
        return {
            'rs': 0, 'grade': '-', 'signal': 'NEUTRAL',
            'rs_1h': None, 'rs_4h': None, 'rs_24h': None,
        }

    rs_value = weighted_sum / valid_weight

    if rs_value >= 5:    grade, signal = 'S', 'STRONG_BUY'
    elif rs_value >= 2:  grade, signal = 'A', 'BUY'
    elif rs_value >= 0:  grade, signal = 'B', 'WATCH'
    elif rs_value >= -2: grade, signal = 'C', 'NEUTRAL'
    else:                grade, signal = '-', 'WEAK'

    return {
        'rs':     float(round(rs_value, 4)),
        'grade':  str(grade),
        'signal': str(signal),
        'rs_1h':  float(rs_components['1h']) if rs_components.get('1h') is not None else None,
        'rs_4h':  float(rs_components['4h']) if rs_components.get('4h') is not None else None,
        'rs_24h': float(rs_components['24h']) if rs_components.get('24h') is not None else None,
    }


# ── v3.0.5: 다이버전스 탐지 ──────────────────────────────────────
def _find_local_lows(values, lookback=5, min_gap=3):
    lows = []
    for i in range(1, len(values) - 1):
        if values[i] <= values[i-1] and values[i] <= values[i+1]:
            if not lows or (i - lows[-1]) >= min_gap:
                lows.append(i)
    return lows[-lookback:]


def _find_local_highs(values, lookback=5, min_gap=3):
    highs = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i-1] and values[i] >= values[i+1]:
            if not highs or (i - highs[-1]) >= min_gap:
                highs.append(i)
    return highs[-lookback:]


def detect_divergence(closes, term='short', lookback=60):
    result = {
        'div_type':    'NONE',
        'bull_div':    False,
        'bear_div':    False,
        'hidden_bull': False,
        'hidden_bear': False,
        'div_strength': 'NONE',
        'k_low1':      None, 'k_low2':      None,
        'price_low1': None, 'price_low2': None,
    }

    if not closes or len(closes) < 50:
        return result

    closes_use = closes[-lookback:]
    prices     = list(closes_use)
    p          = PARAMS.get(term, PARAMS['short'])
    k_vals_raw = _stoch_rsi_k(
        closes_use,
        p['rsi'], p['stoch'], p['k_smooth'],
    )
    k_vals = [v for v in k_vals_raw if v is not None]

    min_len = min(len(prices), len(k_vals))
    if min_len < 10:
        return result

    prices = prices[-min_len:]
    k_vals = k_vals[-min_len:]

    price_lows  = _find_local_lows(prices,  lookback=5, min_gap=3)
    price_highs = _find_local_highs(prices, lookback=5, min_gap=3)
    k_lows      = _find_local_lows(k_vals,  lookback=5, min_gap=3)
    k_highs     = _find_local_highs(k_vals, lookback=5, min_gap=3)

    bull_div     = False
    bear_div     = False
    hidden_bull  = False
    hidden_bear  = False
    div_strength = 'NONE'
    k_low1 = k_low2 = price_low1 = price_low2 = None

    # 일반 강세 다이버전스
    if len(price_lows) >= 2 and len(k_lows) >= 2:
        pi1, pi2 = price_lows[-2],  price_lows[-1]
        ki1, ki2 = k_lows[-2],      k_lows[-1]
        if prices[pi2] < prices[pi1] and k_vals[ki2] > k_vals[ki1] and k_vals[ki2] <= 35:
            bull_div               = True
            k_low1, k_low2         = float(round(k_vals[ki1], 2)), float(round(k_vals[ki2], 2))
            price_low1, price_low2 = float(round(prices[pi1], 4)), float(round(prices[pi2], 4))
            k_diff                 = k_vals[ki2] - k_vals[ki1]
            div_strength
