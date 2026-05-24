"""
mtf_setup.py  v3.0.5
─────────────────────────────────────────────
변경사항:
  v3.0.1  StochRSI K+D 단/중/장기, 점수 기반 등급 통일
  v3.0.2  C등급 Watch 등록 차단
  v3.0.3  타임프레임 정렬 카운트 기반 등급 강화
          4h K≥70 → timing_warning
          1h K≥80 → overbought_warning
  v3.0.4  entry_price 보존 로직 (scanner 측 수정)
  v3.0.5  일봉 단기 K방향(K>D) 점수 반영 (+5 / -5)
          일봉 중기 K방향(K>D) 점수 반영 (+5 / -5)
          K방향 정보 summarize 결과에 포함
─────────────────────────────────────────────
"""

VERSION = 'v3.0.5'

import numpy as np

# ── 파라미터 ──────────────────────────────────────────
PARAMS = {
    'short': {'rsi': 14, 'stoch': 14, 'k_smooth': 3,  'd_smooth': 3},
    'mid':   {'rsi': 21, 'stoch': 21, 'k_smooth': 5,  'd_smooth': 5},
    'long':  {'rsi': 42, 'stoch': 42, 'k_smooth': 9,  'd_smooth': 9},
}

OVERSOLD   = 20.0
OVERBOUGHT = 80.0


# ── 내부 계산 함수 ────────────────────────────────────

def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.zeros(len(closes))
    avg_loss = np.zeros(len(closes))
    if len(gains) < period:
        return np.full(len(closes), 50.0)
    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1])  / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period
    rs  = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50.0
    return rsi


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = arr[i-period+1:i+1].mean()
    return out


def _stoch_rsi_k(closes: np.ndarray, rsi_p: int,
                 stoch_p: int, k_smooth: int) -> np.ndarray:
    rsi_vals = _rsi(closes, rsi_p)
    raw_k    = np.full(len(closes), 50.0)
    for i in range(stoch_p - 1, len(closes)):
        window = rsi_vals[i-stoch_p+1:i+1]
        lo, hi = window.min(), window.max()
        raw_k[i] = (50.0 if hi == lo
                    else (rsi_vals[i] - lo) / (hi - lo) * 100)
    return _sma(raw_k, k_smooth)


def _empty_result() -> dict:
    return {
        'k': 50.0, 'd': 50.0,
        'prev_k': 50.0, 'prev_d': 50.0,
        'gc': False, 'dc': False,
        'zone': 'neutral', 'signal': 'NEUTRAL',
    }


def calc_stoch_rsi(closes: list, term: str) -> dict:
    p   = PARAMS[term]
    arr = np.array(closes, dtype=float)
    if len(arr) < p['rsi'] + p['stoch'] + p['k_smooth'] + p['d_smooth']:
        return _empty_result()
    k_arr  = _stoch_rsi_k(arr, p['rsi'], p['stoch'], p['k_smooth'])
    d_arr  = _sma(k_arr, p['d_smooth'])
    valid  = ~np.isnan(k_arr) & ~np.isnan(d_arr)
    if valid.sum() < 2:
        return _empty_result()
    idx        = np.where(valid)[0]
    ci, pi     = idx[-1], idx[-2]
    k, d       = float(k_arr[ci]), float(d_arr[ci])
    prev_k, prev_d = float(k_arr[pi]), float(d_arr[pi])
    gc     = (prev_k <= prev_d) and (k > d)
    dc     = (prev_k >= prev_d) and (k < d)
    zone   = ('oversold'   if k <= OVERSOLD  else
              'overbought' if k >= OVERBOUGHT else 'neutral')
    signal = _get_signal(k, d, prev_k, prev_d, gc, dc, zone)
    return {
        'k': round(k, 1), 'd': round(d, 1),
        'prev_k': round(prev_k, 1), 'prev_d': round(prev_d, 1),
        'gc': gc, 'dc': dc,
        'zone': zone, 'signal': signal,
    }


def _get_signal(k, d, prev_k, prev_d, gc, dc, zone) -> str:
    if dc and zone != 'oversold':
        return 'BUY_NO'
    if gc and zone == 'oversold':
        return 'BUY_OK'
    if zone == 'oversold' and k > prev_k:
        return 'WATCH'
    return 'NEUTRAL'


# ── MTF 분석 ─────────────────────────────────────────

def analyze_mtf(daily: list, h4: list, h1: list) -> dict:
    return {
        'daily': {
            'short': calc_stoch_rsi(daily, 'short'),
            'mid':   calc_stoch_rsi(daily, 'mid'),
            'long':  calc_stoch_rsi(daily, 'long'),
        },
        'h4': {'short': calc_stoch_rsi(h4, 'short')},
        'h1': {'short': calc_stoch_rsi(h1, 'short')},
    }


def _count_aligned_timeframes(d_long, d_mid, d_short,
                               h4_short, h1_short) -> int:
    """과매도 or 상승 전환 중인 타임프레임 수 카운트"""
    count = 0
    if d_long.get('zone') == 'oversold':
        count += 1
    if d_mid.get('zone') == 'oversold' or d_mid.get('signal') == 'BUY_OK':
        count += 1
    if d_short.get('signal') in ('BUY_OK', 'WATCH'):
        count += 1
    if h4_short.get('signal') in ('BUY_OK', 'WATCH'):
        count += 1
    if h1_short.get('signal') in ('BUY_OK', 'WATCH'):
        count += 1
    return count


def _summarize(mtf: dict) -> dict:
    daily    = mtf.get('daily', {})
    h4       = mtf.get('h4', {})
    h1       = mtf.get('h1', {})
    d_long   = daily.get('long',  _empty_result())
    d_mid    = daily.get('mid',   _empty_result())
    d_short  = daily.get('short', _empty_result())
    h4_short = h4.get('short',    _empty_result())
    h1_short = h1.get('short',    _empty_result())

    any_buy_no = any([
        d_long.get('signal')   == 'BUY_NO',
        d_short.get('signal')  == 'BUY_NO',
        h4_short.get('signal') == 'BUY_NO',
        h1_short.get('signal') == 'BUY_NO',
    ])

    d_long_os = d_long.get('zone') == 'oversold'
    daily_gc  = d_short.get('signal') == 'BUY_OK'
    h4_gc     = h4_short.get('signal') == 'BUY_OK'
    h1_gc     = h1_short.get('signal') == 'BUY_OK'

    # ── K 방향 (K > D 여부) ──────────────────────────
    d_short_k_rising = d_short.get('k', 50.0) > d_short.get('d', 50.0)
    d_mid_k_rising   = d_mid.get('k',   50.0) > d_mid.get('d',   50.0)

    score   = _calc_score(d_long, d_mid, d_short, h4_short, h1_short,
                          d_short_k_rising, d_mid_k_rising)
    aligned = _count_aligned_timeframes(
                  d_long, d_mid, d_short, h4_short, h1_short)

    # ── 타이밍 경고 ──────────────────────────────────
    h4_k = h4_short.get('k', 50.0)
    h1_k = h1_short.get('k', 50.0)
    timing_warning     = h4_k >= 70
    overbought_warning = h1_k >= 80

    # ── 등급 판정 ─────────────────────────────────────
    if any_buy_no:
        grade = 'X'
    elif score >= 85 and daily_gc and (h4_gc or h1_gc):
        grade = 'S'
    elif score >= 70 and (h4_gc or h1_gc):
        grade = 'A'
    elif score >= 55 and aligned >= 2:
        grade = 'B'
    elif d_long_os:
        grade = 'C'
    else:
        grade = '-'

    watch_eligible = d_long_os and not any_buy_no and grade not in ('C', '-', 'X')
    auto_entry     = (watch_eligible and
                      (h4_gc or h1_gc) and
                      not timing_warning and
                      not overbought_warning)

    return {
        'grade':               grade,
        'watch_eligible':      watch_eligible,
        'auto_entry':          auto_entry,
        'any_buy_no':          any_buy_no,
        'score':               score,
        'aligned':             aligned,
        'd_long_os':           d_long_os,
        'daily_gc':            daily_gc,
        'h4_gc':               h4_gc,
        'h1_gc':               h1_gc,
        'timing_warning':      timing_warning,
        'overbought_warning':  overbought_warning,
        'h4_k':                round(h4_k, 1),
        'h1_k':                round(h1_k, 1),
        # v3.0.5 추가
        'd_short_k_rising':    d_short_k_rising,
        'd_mid_k_rising':      d_mid_k_rising,
    }


def _calc_score(d_long, d_mid, d_short, h4_short, h1_short,
                d_short_k_rising: bool = True,
                d_mid_k_rising:   bool = True) -> int:
    score = 0

    # 일봉 장기 (핵심 – 최대 40점)
    if d_long.get('zone') == 'oversold':
        score += 30
        if (d_long.get('k') or 50) <= 10:
            score += 10
    if d_long.get('signal') == 'BUY_OK':
        score += 10

    # 일봉 중기 (최대 20점 → 방향 포함)
    if d_mid.get('zone') == 'oversold':
        score += 10
    if d_mid.get('signal') == 'BUY_OK':
        score += 5
    # K 방향 보너스/페널티
    if d_mid_k_rising:
        score += 5   # K > D → 상승 중
    else:
        score = max(0, score - 5)  # K < D → 하락 중

    # 일봉 단기 (최대 15점 → 방향 포함)
    if d_short.get('zone') == 'oversold':
        score += 5
    if d_short.get('signal') == 'BUY_OK':
        score += 5
    # K 방향 보너스/페널티
    if d_short_k_rising:
        score += 5   # K > D → 상승 중
    else:
        score = max(0, score - 5)  # K < D → 하락 중

    # 4h (최대 20점)
    if h4_short.get('signal') == 'BUY_OK':
        score += 20
    elif h4_short.get('zone') == 'oversold':
        score += 8

    # 1h (최대 15점)
    if h1_short.get('signal') == 'BUY_OK':
        score += 15
    elif h1_short.get('zone') == 'oversold':
        score += 5

    # 페널티
    if d_long.get('signal')   == 'BUY_NO': score = max(0, score - 40)
    if h4_short.get('signal') == 'BUY_NO': score = max(0, score - 20)
    if h1_short.get('signal') == 'BUY_NO': score = max(0, score - 10)

    return min(score, 100)


# ── BTC MA20 신호 ────────────────────────────────────

def btc_ma20_signal(daily_closes: list, weekly_closes: list) -> dict:
    result = {
        'daily_ma20':   None, 'daily_signal':   'NEUTRAL',
        'weekly_ma20':  None, 'weekly_signal':  'NEUTRAL',
        'price':        None,
    }
    if daily_closes and len(daily_closes) >= 20:
        arr   = np.array(daily_closes, dtype=float)
        ma20  = float(arr[-20:].mean())
        price = float(arr[-1])
        result['daily_ma20']   = round(ma20)
        result['price']        = round(price)
        result['daily_signal'] = 'ABOVE' if price >= ma20 else 'BELOW'
    if weekly_closes and len(weekly_closes) >= 20:
        arr   = np.array(weekly_closes, dtype=float)
        ma20  = float(arr[-20:].mean())
        price = float(arr[-1])
        result['weekly_ma20']   = round(ma20)
        result['weekly_signal'] = 'ABOVE' if price >= ma20 else 'BELOW'
    return result


# ── 상대강도 ─────────────────────────────────────────

def calc_relative_strength(coin_closes: list,
                            btc_closes:  list,
                            period:      int = 14) -> dict:
    if len(coin_closes) < period + 1 or len(btc_closes) < period + 1:
        return {'rs': 0.0, 'grade': '-'}
    c_arr = np.array(coin_closes[-period-1:], dtype=float)
    b_arr = np.array(btc_closes[-period-1:],  dtype=float)
    c_chg = (c_arr[-1] - c_arr[0]) / c_arr[0] * 100 if c_arr[0] != 0 else 0
    b_chg = (b_arr[-1] - b_arr[0]) / b_arr[0] * 100 if b_arr[0] != 0 else 0
    rs    = round(c_chg - b_chg, 2)
    grade = ('S' if rs >=  5 else
             'A' if rs >=  2 else
             'B' if rs >=  0 else
             'C' if rs >= -3 else '-')
    return {'rs': rs, 'grade': grade}


if __name__ == '__main__':
    print(f'mtf_setup.py {VERSION} 로드 완료 ✅')
    print(f'  등급기준: S(≥85+일봉GC+4h/1hGC) A(≥70+4h/1hGC) B(≥55+aligned≥2)')
    print(f'  v3.0.5: 일봉 단기/중기 K방향 점수 반영 (+5/-5)')
