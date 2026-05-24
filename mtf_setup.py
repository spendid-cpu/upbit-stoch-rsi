"""
mtf_setup.py v3.1.1
변경사항:
- v3.0.1: StochRSI K+D 단기/중기/장기 + 통합 점수 기반 등급
- v3.0.2: C등급 Watch 등록 차단
- v3.0.3: 타임프레임 정렬 강화, 등급 강화 (4h K≥70 타이밍경고)
- v3.0.4: entry_price 보존, price_change = 등록가 대비
- v3.0.5: 일봉 단기/중기 K 방향 점수 반영
- v3.0.6: 일봉 단기 K>15 페널티
- v3.1.0: 사이클 감지 (BOTTOM/RISING/PEAK/FALLING), K 5개 시점
- v3.1.1: cycle_block Watch 완전차단 제거 → 점수 페널티로만 반영
           PEAK/FALLING은 점수로 자연 필터링, Watch 차단은 BUY_NO+등급만
"""

import numpy as np

VERSION = 'v3.1.1'

# ── 파라미터 ──────────────────────────────────────────
PARAMS = {
    'short': {'rsi': 14, 'stoch': 14, 'k_smooth': 3,  'd_smooth': 3},
    'mid':   {'rsi': 21, 'stoch': 21, 'k_smooth': 5,  'd_smooth': 5},
    'long':  {'rsi': 42, 'stoch': 42, 'k_smooth': 9,  'd_smooth': 9},
}
OVERSOLD   = 20.0
OVERBOUGHT = 80.0

# ── 기본 계산 함수 ────────────────────────────────────
def _rsi(closes, period):
    closes = np.array(closes, dtype=float)
    if len(closes) < period + 1:
        return np.full(len(closes), 50.0)
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = np.mean(gains[:period])
    avg_l  = np.mean(losses[:period])
    rsi_vals = [50.0] * (period + 1)
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i])  / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs    = avg_g / avg_l if avg_l != 0 else 1e9
        rsi_vals.append(100 - 100 / (1 + rs))
    result = np.full(len(closes), 50.0)
    result[1:] = rsi_vals
    return result

def _sma(arr, period):
    arr = np.array(arr, dtype=float)
    if len(arr) < period:
        return arr
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.mean(arr[i - period + 1:i + 1])
    return result

def _stoch_rsi_k(closes, rsi_period, stoch_period, k_smooth):
    rsi_vals = _rsi(closes, rsi_period)
    k_arr    = np.full(len(rsi_vals), 50.0)
    for i in range(stoch_period - 1, len(rsi_vals)):
        window = rsi_vals[i - stoch_period + 1:i + 1]
        lo, hi = np.min(window), np.max(window)
        k_arr[i] = (rsi_vals[i] - lo) / (hi - lo) * 100 if hi != lo else 50.0
    k_smooth_arr = _sma(k_arr, k_smooth)
    return k_smooth_arr

# ── 사이클 감지 ───────────────────────────────────────
def _detect_cycle(k_series):
    """
    k_series: [k_prev4, k_prev3, k_prev2, k_prev1, k_now] 5개
    반환: 'BOTTOM' / 'RISING' / 'PEAK' / 'FALLING'
    """
    if len(k_series) < 3:
        return 'RISING'

    k_now   = k_series[-1]
    k_prev1 = k_series[-2]
    k_prev2 = k_series[-3]

    slope_now  = k_now   - k_prev1
    slope_prev = k_prev1 - k_prev2

    # BOTTOM: 과매도 구간에서 올라오는 중
    if k_now <= OVERSOLD and slope_now >= 0:
        return 'BOTTOM'

    # FALLING → BOTTOM 진입 직전
    if k_now <= OVERSOLD and slope_now < 0:
        return 'FALLING'

    # PEAK: 고점에서 막 꺾임
    if slope_now < 0 and slope_prev > 0 and k_now > OVERSOLD:
        return 'PEAK'

    # FALLING: 하락 가속
    if slope_now < 0 and slope_prev <= 0 and k_now > OVERSOLD:
        return 'FALLING'

    # RISING: 상승 중
    if slope_now >= 0 and k_now > OVERSOLD:
        return 'RISING'

    return 'RISING'

# ── StochRSI 계산 ─────────────────────────────────────
def calc_stoch_rsi(closes, rsi_period, stoch_period, k_smooth, d_smooth):
    closes = np.array(closes, dtype=float)
    min_len = rsi_period + stoch_period + max(k_smooth, d_smooth) + 10
    if len(closes) < min_len:
        return {
            'k': 50.0, 'd': 50.0,
            'prev_k': 50.0, 'prev_d': 50.0,
            'k_series': [50.0] * 5,
            'gc': False, 'dc': False,
            'zone': 'neutral', 'signal': 'NEUTRAL',
            'cycle': 'RISING'
        }

    k_arr = _stoch_rsi_k(closes, rsi_period, stoch_period, k_smooth)
    d_arr = _sma(k_arr, d_smooth)

    valid = ~np.isnan(d_arr)
    if valid.sum() < 6:
        return {
            'k': 50.0, 'd': 50.0,
            'prev_k': 50.0, 'prev_d': 50.0,
            'k_series': [50.0] * 5,
            'gc': False, 'dc': False,
            'zone': 'neutral', 'signal': 'NEUTRAL',
            'cycle': 'RISING'
        }

    valid_idx = np.where(valid)[0]
    idx       = valid_idx[-1]

    k_now   = float(k_arr[idx])
    d_now   = float(d_arr[idx])
    k_prev1 = float(k_arr[idx - 1]) if idx >= 1 else k_now
    d_prev1 = float(d_arr[idx - 1]) if idx >= 1 else d_now
    k_prev2 = float(k_arr[idx - 2]) if idx >= 2 else k_prev1
    k_prev3 = float(k_arr[idx - 3]) if idx >= 3 else k_prev2
    k_prev4 = float(k_arr[idx - 4]) if idx >= 4 else k_prev3

    k_series = [k_prev4, k_prev3, k_prev2, k_prev1, k_now]
    cycle    = _detect_cycle(k_series)

    # 골든크로스 / 데드크로스
    gc = (k_prev1 <= d_prev1) and (k_now > d_now)
    dc = (k_prev1 >= d_prev1) and (k_now < d_now)

    # 구간
    if k_now <= OVERSOLD:
        zone = 'oversold'
    elif k_now >= OVERBOUGHT:
        zone = 'overbought'
    else:
        zone = 'neutral'

    # 시그널 - PEAK/FALLING이어도 BUY_NO 대신 NEUTRAL로 완화
    if zone == 'overbought':
        signal = 'BUY_NO'
    elif zone == 'oversold' and cycle == 'BOTTOM':
        signal = 'BUY_OK'
    elif zone == 'oversold' and gc:
        signal = 'BUY_OK'
    elif zone == 'oversold':
        signal = 'WATCH'
    elif cycle in ('PEAK', 'FALLING') and k_now > 40:
        # 높은 구간에서 꺾이는 경우만 BUY_NO
        signal = 'BUY_NO'
    else:
        signal = 'NEUTRAL'

    return {
        'k':        k_now,
        'd':        d_now,
        'prev_k':   k_prev1,
        'prev_d':   d_prev1,
        'k_series': k_series,
        'gc':       gc,
        'dc':       dc,
        'zone':     zone,
        'signal':   signal,
        'cycle':    cycle
    }

# ── MTF 분석 ──────────────────────────────────────────
def analyze_mtf(daily_closes, h4_closes, h1_closes):
    results = {}
    for tf, closes in [('daily', daily_closes), ('h4', h4_closes), ('h1', h1_closes)]:
        for label, p in PARAMS.items():
            key = f'{tf}_{label}'
            results[key] = calc_stoch_rsi(
                closes,
                p['rsi'], p['stoch'], p['k_smooth'], p['d_smooth']
            )
    return results

# ── 타임프레임 정렬 카운트 ────────────────────────────
def _count_aligned_timeframes(d_long, d_mid, d_short, h4_short, h1_short):
    count = 0
    if d_long.get('zone')   == 'oversold':                     count += 1
    if d_mid.get('cycle')   in ('BOTTOM', 'RISING'):           count += 1
    if d_short.get('cycle') == 'BOTTOM':                       count += 1
    if h4_short.get('signal') in ('BUY_OK', 'WATCH'):         count += 1
    if h1_short.get('signal') in ('BUY_OK', 'WATCH'):         count += 1
    return count

# ── 점수 계산 ─────────────────────────────────────────
def _calc_score(mtf: dict) -> int:
    d_long  = mtf.get('daily_long',  {})
    d_mid   = mtf.get('daily_mid',   {})
    d_short = mtf.get('daily_short', {})
    h4_s    = mtf.get('h4_short',    {})
    h1_s    = mtf.get('h1_short',    {})

    score = 0

    # ── 일봉 장기 ──────────────────────────────────────
    if d_long.get('zone') == 'oversold':
        score += 15
    if d_long.get('signal') == 'BUY_OK':
        score += 10
    if d_long.get('gc'):
        score += 5

    # ── 일봉 중기 사이클 기반 ──────────────────────────
    mid_cycle = d_mid.get('cycle', 'RISING')
    if mid_cycle == 'BOTTOM':
        score += 20
    elif mid_cycle == 'RISING':
        score += 8
    elif mid_cycle == 'PEAK':
        score -= 15
    elif mid_cycle == 'FALLING':
        score -= 20

    if d_mid.get('signal') == 'BUY_OK':
        score += 10
    if d_mid.get('gc'):
        score += 5

    # ── 일봉 단기 사이클 기반 ──────────────────────────
    short_cycle = d_short.get('cycle', 'RISING')
    if short_cycle == 'BOTTOM':
        score += 20
    elif short_cycle == 'RISING':
        score += 5
    elif short_cycle == 'PEAK':
        score -= 15
    elif short_cycle == 'FALLING':
        score -= 20

    if d_short.get('signal') == 'BUY_OK':
        score += 5
    if d_short.get('gc'):
        score += 3

    # ── 단기+중기 동시 PEAK/FALLING 추가 페널티 ────────
    bad_cycles = {'PEAK', 'FALLING'}
    if short_cycle in bad_cycles and mid_cycle in bad_cycles:
        score -= 15

    # ── 4h 단기 ────────────────────────────────────────
    h4_cycle = h4_s.get('cycle', 'RISING')
    h4_k     = h4_s.get('k', 50.0)
    if h4_cycle == 'BOTTOM':
        score += 10
    elif h4_cycle == 'RISING' and h4_k <= 35:
        score += 5
    elif h4_cycle == 'PEAK':
        score -= 10
    elif h4_cycle == 'FALLING':
        score -= 10

    if h4_s.get('signal') == 'BUY_OK':
        score += 5
    if h4_s.get('gc'):
        score += 3

    # ── 1h 단기 ────────────────────────────────────────
    h1_cycle = h1_s.get('cycle', 'RISING')
    h1_k     = h1_s.get('k', 50.0)
    if h1_cycle == 'BOTTOM':
        score += 8
    elif h1_cycle == 'RISING' and h1_k <= 35:
        score += 3
    elif h1_cycle == 'PEAK':
        score -= 8
    elif h1_cycle == 'FALLING':
        score -= 8

    if h1_s.get('signal') == 'BUY_OK':
        score += 3
    if h1_s.get('gc'):
        score += 5

    # ── BUY_NO 페널티 (overbought만 강하게) ───────────
    for key, data in [('daily_long', d_long), ('daily_mid', d_mid),
                      ('daily_short', d_short), ('h4_short', h4_s), ('h1_short', h1_s)]:
        if data.get('signal') == 'BUY_NO':
            score -= 15

    return max(0, min(100, score))

# ── 요약 / 등급 ───────────────────────────────────────
def _summarize(mtf: dict) -> dict:
    d_long  = mtf.get('daily_long',  {})
    d_mid   = mtf.get('daily_mid',   {})
    d_short = mtf.get('daily_short', {})
    h4_s    = mtf.get('h4_short',    {})
    h1_s    = mtf.get('h1_short',    {})

    score   = _calc_score(mtf)
    aligned = _count_aligned_timeframes(d_long, d_mid, d_short, h4_s, h1_s)

    # 골든크로스 여부
    daily_gc = d_long.get('gc') or d_mid.get('gc') or d_short.get('gc')
    h4_gc    = h4_s.get('gc', False)
    h1_gc    = h1_s.get('gc', False)

    # BUY_NO: overbought만 적용
    any_buy_no = any(
        mtf.get(k, {}).get('signal') == 'BUY_NO'
        for k in ['daily_long', 'daily_mid', 'daily_short', 'h4_short', 'h1_short']
    )

    # 사이클 정보 (표시용 - 차단 아님)
    short_cycle = d_short.get('cycle', 'RISING')
    mid_cycle   = d_mid.get('cycle',   'RISING')

    # cycle_block: 정보 표시용만 (Watch 차단 안 함)
    cycle_block = short_cycle in ('PEAK', 'FALLING') and mid_cycle in ('PEAK', 'FALLING')

    # 타이밍 경고
    h4_k = h4_s.get('k', 0.0)
    h1_k = h1_s.get('k', 0.0)
    timing_warning    = h4_k >= 70
    overbought_warning = h1_k >= 80

    # 등급 - cycle_block은 등급에 영향 없음, 점수로만 반영
    if any_buy_no:
        grade = 'X'
    elif score >= 85 and daily_gc and (h4_gc or h1_gc):
        grade = 'S'
    elif score >= 70 and (h4_gc or h1_gc):
        grade = 'A'
    elif score >= 55 and aligned >= 2:
        grade = 'B'
    elif d_long.get('zone') == 'oversold':
        grade = 'C'
    else:
        grade = '-'

    watch_eligible = grade in ('S', 'A', 'B') and not timing_warning and not overbought_warning
    auto_entry     = watch_eligible and (h4_gc or h1_gc) and not timing_warning and not overbought_warning

    return {
        'grade':              grade,
        'score':              score,
        'aligned':            aligned,
        'watch_eligible':     watch_eligible,
        'auto_entry':         auto_entry,
        'timing_warning':     timing_warning,
        'overbought_warning': overbought_warning,
        'daily_gc':           daily_gc,
        'h4_gc':              h4_gc,
        'h1_gc':              h1_gc,
        'cycle_block':        cycle_block,
        'd_short_cycle':      short_cycle,
        'd_mid_cycle':        mid_cycle,
        'd_long_zone':        d_long.get('zone', 'neutral'),
        'h4_cycle':           h4_s.get('cycle', 'RISING'),
        'h1_cycle':           h1_s.get('cycle', 'RISING'),
    }

# ── BTC MA20 시그널 ───────────────────────────────────
def btc_ma20_signal(closes, period=20):
    closes = np.array(closes, dtype=float)
    if len(closes) < period:
        return {'signal': 'UNKNOWN', 'ma20': None, 'price': float(closes[-1]) if len(closes) else 0}
    ma20  = float(np.mean(closes[-period:]))
    price = float(closes[-1])
    return {
        'signal': 'ABOVE' if price > ma20 else 'BELOW',
        'ma20':   round(ma20, 2),
        'price':  round(price, 2)
    }

# ── 상대강도 ──────────────────────────────────────────
def calc_relative_strength(coin_closes, btc_closes, period=20):
    if len(coin_closes) < period + 1 or len(btc_closes) < period + 1:
        return {'rs': 1.0, 'grade': 'B'}
    coin_ret = (coin_closes[-1] - coin_closes[-period]) / coin_closes[-period] * 100
    btc_ret  = (btc_closes[-1]  - btc_closes[-period])  / btc_closes[-period]  * 100
    rs = coin_ret - btc_ret
    if   rs >= 10:  rs_grade = 'S'
    elif rs >= 3:   rs_grade = 'A'
    elif rs >= -3:  rs_grade = 'B'
    elif rs >= -10: rs_grade = 'C'
    else:           rs_grade = '-'
    return {'rs': round(rs, 2), 'grade': rs_grade}

# ── 전체 래퍼 ─────────────────────────────────────────
def full_analyze(daily_closes, h4_closes, h1_closes):
    mtf     = analyze_mtf(daily_closes, h4_closes, h1_closes)
    summary = _summarize(mtf)
    return {**mtf, **summary}

# ── 모듈 로드 확인 ────────────────────────────────────
if __name__ == '__main__' or True:
    print(f'mtf_setup.py {VERSION} 로드 완료 ✅')
    print(f'  사이클 감지: BOTTOM🟢 / RISING🔵 / PEAK🔴 / FALLING⚫')
    print(f'  cycle_block = 표시용만 (Watch 차단 없음)')
    print(f'  PEAK/FALLING → 점수 페널티로만 필터링')
    print(f'  등급: S(≥85+GC) / A(≥70+GC) / B(≥55+aligned≥2) / X(overbought BUY_NO만)')
