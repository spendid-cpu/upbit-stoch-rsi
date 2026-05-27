"""
mtf_setup.py v3.0.5
변경사항:
- v3.0.3: DEEP 상대강도 보너스 추가
- v3.0.4: calc_relative_strength 가중 멀티타임프레임 평균 (RS_1h 50% + RS_4h 30% + RS_24h 20%)
- v3.0.5: detect_divergence() 추가 (StochRSI 다이버전스 탐지 - BULL/BEAR/HIDDEN_BULL/NONE)
"""

import numpy as np

MTF_VERSION = 'v3.0.5'
VERSION = MTF_VERSION

OVERSOLD  = 20.0
OVERBOUGHT = 80.0

# ── 파라미터 세트 ──────────────────────────────────────────────────
PARAMS = {
    'short': {'rsi_period': 14, 'stoch_window': 14, 'smooth_k': 3, 'smooth_d': 3},
    'mid':   {'rsi_period': 21, 'stoch_window': 21, 'smooth_k': 5, 'smooth_d': 3},
    'long':  {'rsi_period': 28, 'stoch_window': 28, 'smooth_k': 7, 'smooth_d': 3},
}

# ── 내부 유틸 ─────────────────────────────────────────────────────
def _rsi(closes, period=14):
    closes = np.array(closes, dtype=float)
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

def _stoch_rsi_k(closes, rsi_period=14, stoch_window=14, smooth_k=3):
    rsi_vals = _rsi(closes, rsi_period)
    stoch_k  = []
    for i in range(len(rsi_vals)):
        if i < stoch_window - 1:
            stoch_k.append(None)
            continue
        window = rsi_vals[i - stoch_window + 1:i + 1]
        lo, hi = min(window), max(window)
        stoch_k.append(50.0 if hi == lo else (rsi_vals[i] - lo) / (hi - lo) * 100.0)
    valid = [v for v in stoch_k if v is not None]
    smoothed = _sma(valid, smooth_k)
    return smoothed  # %K smoothed

def _detect_cycle(k_vals, n=5):
    """최근 n개 K값으로 사이클 판단"""
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
    if zone == 'OVERSOLD' and cross_up:
        return 'BUY'
    if zone == 'OVERSOLD':
        return 'BUY'
    return 'NEUTRAL'

# ── 메인 StochRSI 계산 ────────────────────────────────────────────
def calc_stoch_rsi(closes, term='short'):
    p = PARAMS[term]
    k_vals = _stoch_rsi_k(closes, p['rsi_period'], p['stoch_window'], p['smooth_k'])
    d_vals = _sma([v for v in k_vals if v is not None], p['smooth_d'])

    valid_k = [v for v in k_vals if v is not None]
    valid_d = [v for v in d_vals if v is not None]

    k = valid_k[-1] if valid_k else 50.0
    d = valid_d[-1] if valid_d else 50.0

    # 크로스 감지
    cross_up   = False
    cross_down = False
    if len(valid_k) >= 2 and len(valid_d) >= 2:
        cross_up   = valid_k[-2] <= valid_d[-2] and valid_k[-1] > valid_d[-1]
        cross_down = valid_k[-2] >= valid_d[-2] and valid_k[-1] < valid_d[-1]

    zone = 'OVERSOLD' if k <= OVERSOLD else 'OVERBOUGHT' if k >= OVERBOUGHT else 'NEUTRAL'
    signal = _get_signal(zone, cross_up, cross_down)
    cycle  = _detect_cycle(valid_k)

    return {
        'k': round(k, 2),
        'd': round(d, 2),
        'cross_up':   cross_up,
        'cross_down': cross_down,
        'zone':       zone,
        'signal':     signal,
        'cycle':      cycle,
    }

# ── MTF 분석 ─────────────────────────────────────────────────────
def analyze_mtf(candle_dict):
    """
    candle_dict = {'daily': [...], 'h4': [...], 'h1': [...]}
    각 리스트는 close 가격 리스트
    """
    results = {}
    for tf, closes in candle_dict.items():
        term = 'long' if tf == 'daily' else 'short'
        results[tf] = calc_stoch_rsi(closes, term)
    summary = _summarize(results)
    return {'results': results, 'summary': summary}

def _summarize(results):
    score = _calc_score(results)
    grade = _calc_grade(score)
    cycles = {tf: r.get('cycle', 'RISING') for tf, r in results.items()}
    return {
        'score': score,
        'grade': grade,
        'cycles': cycles,
        'd_short_cycle': cycles.get('daily', 'RISING'),
        'd_mid_cycle':   cycles.get('daily', 'RISING'),
        'h4_cycle':      cycles.get('h4',    'RISING'),
        'h1_cycle':      cycles.get('h1',    'RISING'),
        'h4_gc': results.get('h4', {}).get('cross_up',   False),
        'h1_gc': results.get('h1', {}).get('cross_up',   False),
        'daily_gc': results.get('daily', {}).get('cross_up', False),
    }

def _calc_score(results):
    score = 0
    weights = {'daily': 40, 'h4': 35, 'h1': 25}
    for tf, weight in weights.items():
        r = results.get(tf, {})
        k = r.get('k', 50)
        cycle = r.get('cycle', 'RISING')
        cross_up = r.get('cross_up', False)
        if cycle == 'BOTTOM':  score += weight * 1.0
        elif cycle == 'RISING': score += weight * 0.7
        elif cycle == 'PEAK':   score += weight * 0.2
        else:                   score += weight * 0.1
        if cross_up:            score += weight * 0.3
        if k <= OVERSOLD:       score += weight * 0.2
    return min(100, round(score))

def _calc_grade(score, deep_rs_grade=None):
    # DEEP RS 보너스
    bonus = 0
    if deep_rs_grade == 'S': bonus = 15
    elif deep_rs_grade == 'A': bonus = 10
    elif deep_rs_grade == 'B': bonus = 5
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
        return {'daily_above': None, 'weekly_above': None,
                'daily_ma20': 0, 'weekly_ma20': 0}
    daily_ma20 = sum(daily_closes[-20:]) / 20
    daily_above = daily_closes[-1] > daily_ma20
    weekly_above = None
    weekly_ma20  = 0
    if weekly_closes and len(weekly_closes) >= 20:
        weekly_ma20  = sum(weekly_closes[-20:]) / 20
        weekly_above = weekly_closes[-1] > weekly_ma20
    return {
        'daily_above':  daily_above,
        'weekly_above': weekly_above,
        'daily_ma20':   round(daily_ma20),
        'weekly_ma20':  round(weekly_ma20),
    }

# ── 상대강도 계산 (v3.0.4: 가중 멀티타임프레임) ───────────────────
def calc_relative_strength(coin_1h, coin_4h, coin_24h,
                            btc_1h,  btc_4h,  btc_24h):
    """
    가중 RS = RS_1h×50% + RS_4h×30% + RS_24h×20%
    타임프레임 누락 시 가중치 재분배
    """
    weights = {'1h': 0.5, '4h': 0.3, '24h': 0.2}
    pairs   = [
        ('1h',  coin_1h,  btc_1h),
        ('4h',  coin_4h,  btc_4h),
        ('24h', coin_24h, btc_24h),
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
        return {'rs': 0, 'grade': '-', 'signal': 'NEUTRAL',
                'rs_1h': None, 'rs_4h': None, 'rs_24h': None}

    # 가중치 재정규화
    rs_value = weighted_sum / valid_weight

    # 등급
    if rs_value >= 5:    grade, signal = 'S', 'STRONG_BUY'
    elif rs_value >= 2:  grade, signal = 'A', 'BUY'
    elif rs_value >= 0:  grade, signal = 'B', 'WATCH'
    elif rs_value >= -2: grade, signal = 'C', 'NEUTRAL'
    else:                grade, signal = '-', 'WEAK'

    return {
        'rs':      round(rs_value, 4),
        'grade':   grade,
        'signal':  signal,
        'rs_1h':   rs_components.get('1h'),
        'rs_4h':   rs_components.get('4h'),
        'rs_24h':  rs_components.get('24h'),
    }

# ── v3.0.5: 다이버전스 탐지 ──────────────────────────────────────
def _find_local_lows(values, lookback=5, min_gap=3):
    """로컬 저점 인덱스 리스트 반환"""
    lows = []
    for i in range(1, len(values) - 1):
        if values[i] <= values[i-1] and values[i] <= values[i+1]:
            if not lows or (i - lows[-1]) >= min_gap:
                lows.append(i)
    return lows[-lookback:]

def _find_local_highs(values, lookback=5, min_gap=3):
    """로컬 고점 인덱스 리스트 반환"""
    highs = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i-1] and values[i] >= values[i+1]:
            if not highs or (i - highs[-1]) >= min_gap:
                highs.append(i)
    return highs[-lookback:]

def detect_divergence(closes, term='short', lookback=60):
    """
    StochRSI 다이버전스 탐지

    Returns:
        dict:
            div_type:     'BULL' | 'BEAR' | 'HIDDEN_BULL' | 'HIDDEN_BEAR' | 'NONE'
            bull_div:     bool
            bear_div:     bool
            hidden_bull:  bool
            hidden_bear:  bool
            div_strength: 'STRONG' | 'NORMAL' | 'NONE'
            k_low1, k_low2:     다이버전스 비교 K값 (저점 다이버전스)
            price_low1, price_low2: 다이버전스 비교 가격 (저점 다이버전스)
    """
    result = {
        'div_type':    'NONE',
        'bull_div':    False,
        'bear_div':    False,
        'hidden_bull': False,
        'hidden_bear': False,
        'div_strength': 'NONE',
        'k_low1': None, 'k_low2': None,
        'price_low1': None, 'price_low2': None,
    }

    if not closes or len(closes) < 50:
        return result

    # 최근 lookback 개 캔들만 사용
    closes_use = closes[-lookback:]
    prices     = list(closes_use)

    p = PARAMS.get(term, PARAMS['short'])
    k_vals_raw = _stoch_rsi_k(
        closes_use,
        p['rsi_period'],
        p['stoch_window'],
        p['smooth_k']
    )
    k_vals = [v for v in k_vals_raw if v is not None]

    # 가격과 K 길이 맞추기
    min_len = min(len(prices), len(k_vals))
    if min_len < 10:
        return result

    prices = prices[-min_len:]
    k_vals = k_vals[-min_len:]

    # 로컬 저점/고점 탐색
    price_lows  = _find_local_lows(prices,  lookback=5, min_gap=3)
    price_highs = _find_local_highs(prices, lookback=5, min_gap=3)
    k_lows      = _find_local_lows(k_vals,  lookback=5, min_gap=3)
    k_highs     = _find_local_highs(k_vals, lookback=5, min_gap=3)

    # ── 일반 강세 다이버전스: 가격 하락저점 + K 상승저점 ──────────
    # 조건: K <= 30 (과매도 구간)
    bull_div    = False
    div_strength = 'NONE'
    k_low1 = k_low2 = price_low1 = price_low2 = None

    if len(price_lows) >= 2 and len(k_lows) >= 2:
        # 가장 최근 두 저점 비교
        pi1, pi2 = price_lows[-2], price_lows[-1]  # 이전, 최근
        ki1, ki2 = k_lows[-2],     k_lows[-1]

        price_falling = prices[pi2] < prices[pi1]
        k_rising      = k_vals[ki2] > k_vals[ki1]
        k_oversold    = k_vals[ki2] <= 35  # 완화된 과매도 기준

        if price_falling and k_rising and k_oversold:
            bull_div = True
            k_low1, k_low2       = round(k_vals[ki1], 2), round(k_vals[ki2], 2)
            price_low1, price_low2 = round(prices[pi1], 4), round(prices[pi2], 4)
            # 강도 판단: K 차이가 클수록 STRONG
            k_diff = k_vals[ki2] - k_vals[ki1]
            div_strength = 'STRONG' if k_diff >= 8 and k_vals[ki2] <= 25 else 'NORMAL'

    # ── 일반 약세 다이버전스: 가격 상승고점 + K 하락고점 ──────────
    bear_div = False
    if len(price_highs) >= 2 and len(k_highs) >= 2:
        pi1, pi2 = price_highs[-2], price_highs[-1]
        ki1, ki2 = k_highs[-2],     k_highs[-1]

        price_rising  = prices[pi2] > prices[pi1]
        k_falling     = k_vals[ki2] < k_vals[ki1]
        k_overbought  = k_vals[ki2] >= 65

        if price_rising and k_falling and k_overbought:
            bear_div = True
            if div_strength == 'NONE':
                k_diff = k_vals[ki1] - k_vals[ki2]
                div_strength = 'STRONG' if k_diff >= 8 and k_vals[ki2] >= 75 else 'NORMAL'

    # ── 히든 강세 다이버전스: 가격 상승저점 + K 하락저점 ──────────
    # (추세 지속 신호 — 상승 추세 중 눌림목)
    hidden_bull = False
    if len(price_lows) >= 2 and len(k_lows) >= 2:
        pi1, pi2 = price_lows[-2], price_lows[-1]
        ki1, ki2 = k_lows[-2],     k_lows[-1]

        price_rising = prices[pi2] > prices[pi1]
        k_falling    = k_vals[ki2] < k_vals[ki1]
        k_mid        = k_vals[ki2] <= 50  # 중간 이하 구간

        if price_rising and k_falling and k_mid and not bull_div:
            hidden_bull = True

    # ── 히든 약세 다이버전스: 가격 하락고점 + K 상승고점 ──────────
    hidden_bear = False
    if len(price_highs) >= 2 and len(k_highs) >= 2:
        pi1, pi2 = price_highs[-2], price_highs[-1]
        ki1, ki2 = k_highs[-2],     k_highs[-1]

        price_falling = prices[pi2] < prices[pi1]
        k_rising      = k_vals[ki2] > k_vals[ki1]
        k_mid         = k_vals[ki2] >= 50

        if price_falling and k_rising and k_mid and not bear_div:
            hidden_bear = True

    # ── 최종 div_type 결정 (우선순위: BULL > BEAR > HIDDEN_BULL > HIDDEN_BEAR) ──
    if bull_div:
        div_type = 'BULL'
    elif bear_div:
        div_type = 'BEAR'
    elif hidden_bull:
        div_type = 'HIDDEN_BULL'
    elif hidden_bear:
        div_type = 'HIDDEN_BEAR'
    else:
        div_type = 'NONE'

    result.update({
        'div_type':    div_type,
        'bull_div':    bull_div,
        'bear_div':    bear_div,
        'hidden_bull': hidden_bull,
        'hidden_bear': hidden_bear,
        'div_strength': div_strength,
        'k_low1':     k_low1,
        'k_low2':     k_low2,
        'price_low1': price_low1,
        'price_low2': price_low2,
    })
    return result


# ── 스크립트 직접 실행 시 확인 출력 ──────────────────────────────
if __name__ == '__main__':
    print(f'✅ mtf_setup {MTF_VERSION} 로드 완료')
    print(f'   사이클 유형: BOTTOM / RISING / PEAK / FALLING')
    print(f'   DEEP RS 가중치: 1h×50% + 4h×30% + 24h×20%')
    print(f'   RS 보너스: S+15 / A+10 / B+5')
    print(f'   다이버전스: BULL / BEAR / HIDDEN_BULL / HIDDEN_BEAR ✅')
