# -*- coding: utf-8 -*-
"""
mtf_setup.py — Upbit MTF Stochastic RSI Analysis Module (v4.0)
변경사항:
  - Watch 점수 시스템 추가 (calc_watch_score)
  - 일봉 게이트 → 4h/1h 계층적 평가
  - 방향성 판단 (상승/하락/횡보/골든크로스)
  - 등급 산정 (S/A/B/C)
"""

import os

# ── 버전 ──────────────────────────────────────────
VERSION = "4.0.0"

# ── Stoch RSI 프리셋 ──────────────────────────────
PRESET_SHORT  = tuple(int(x) for x in os.getenv("PRESET_SHORT",  "5,5,3,3").split(","))
PRESET_MID    = tuple(int(x) for x in os.getenv("PRESET_MID",   "10,10,6,6").split(","))
PRESET_LONG   = tuple(int(x) for x in os.getenv("PRESET_LONG",  "20,20,12,12").split(","))

# ── 임계값 ────────────────────────────────────────
OVERSOLD_THRESHOLD  = float(os.getenv("OVERSOLD_THRESHOLD",  "20.0"))
RECOVERY_K          = float(os.getenv("RECOVERY_K",          "50.0"))

# ── 매크로 필터 ───────────────────────────────────
MACRO_FILTER_ENABLED = os.getenv("MACRO_FILTER_ENABLED", "true").lower() == "true"
MACRO_MA_PERIOD      = int(os.getenv("MACRO_MA_PERIOD", "20"))

# ── Watch 점수 임계값 ─────────────────────────────
GRADE_S_THRESHOLD = int(os.getenv("GRADE_S_THRESHOLD", "80"))
GRADE_A_THRESHOLD = int(os.getenv("GRADE_A_THRESHOLD", "60"))
GRADE_B_THRESHOLD = int(os.getenv("GRADE_B_THRESHOLD", "40"))

# ── Watch 유효기간 (C/B등급만 적용) ──────────────
WATCH_EXPIRY_DAYS = int(os.getenv("WATCH_EXPIRY_DAYS", "7"))


# ════════════════════════════════════════════════
# 기술 계산 함수
# ════════════════════════════════════════════════

def sma(series, period):
    result = []
    for i in range(len(series)):
        if i < period - 1:
            result.append(float('nan'))
        else:
            result.append(sum(series[i-period+1:i+1]) / period)
    return result

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return [float('nan')] * len(closes)
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi_values = [float('nan')] * (period)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period-1) + gains[i]) / period
        avg_loss = (avg_loss * (period-1) + losses[i]) / period
        rs  = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi_values.append(100 - 100/(1+rs))
    return [float('nan')] + rsi_values

def calc_stoch_rsi(closes, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    rsi_vals = calc_rsi(closes, rsi_period)
    k_vals, d_vals = [], []
    for i in range(len(rsi_vals)):
        window = [v for v in rsi_vals[max(0,i-stoch_period+1):i+1] if not (v!=v)]
        if len(window) < stoch_period:
            k_vals.append(float('nan')); continue
        mn, mx = min(window), max(window)
        raw_k = (rsi_vals[i]-mn)/(mx-mn)*100 if mx != mn else 50.0
        k_vals.append(raw_k)
    smoothed_k = sma(k_vals, k_period)
    smoothed_d = sma(smoothed_k, d_period)
    return smoothed_k, smoothed_d

def calc_all_presets(closes):
    results = {}
    for name, preset in [("short", PRESET_SHORT), ("mid", PRESET_MID), ("long", PRESET_LONG)]:
        k, d = calc_stoch_rsi(closes, *preset)
        valid_k = [v for v in k if v == v]
        valid_d = [v for v in d if v == v]
        results[name] = {
            "k": round(valid_k[-1], 2) if valid_k else float('nan'),
            "d": round(valid_d[-1], 2) if valid_d else float('nan'),
            "k_series": [round(v,2) if v==v else None for v in k[-10:]],
            "d_series": [round(v,2) if v==v else None for v in d[-10:]],
        }
    return results


# ════════════════════════════════════════════════
# 방향성 판단
# ════════════════════════════════════════════════

def calc_direction(k_series: list, d_series: list = None) -> dict:
    """
    K값 시리즈로 방향성 판단
    Returns: {
        direction: 'rising'|'falling'|'sideways'|'reversing_up'|'reversing_down',
        golden_cross: bool,
        dead_cross: bool,
        strength: 'strong'|'weak',
    }
    """
    # nan 제거 후 최근 3개
    clean_k = [v for v in k_series if v is not None and v == v]
    if len(clean_k) < 3:
        return {
            'direction': 'unknown',
            'golden_cross': False,
            'dead_cross': False,
            'strength': 'weak'
        }

    k0 = clean_k[-3]  # 3캔들 전
    k1 = clean_k[-2]  # 2캔들 전
    k2 = clean_k[-1]  # 현재

    # 방향 판단
    if k2 > k1 and k1 > k0:
        direction = 'rising'         # 연속 상승
    elif k2 > k1 and k1 <= k0:
        direction = 'reversing_up'   # 하락 후 반등
    elif k2 < k1 and k1 < k0:
        direction = 'falling'        # 연속 하락
    elif k2 < k1 and k1 >= k0:
        direction = 'reversing_down' # 상승 후 하락
    else:
        direction = 'sideways'       # 횡보

    # 강도 판단
    change = abs(k2 - k0)
    strength = 'strong' if change > 5 else 'weak'

    # 골든/데드 크로스
    golden_cross = False
    dead_cross   = False
    if d_series:
        clean_d = [v for v in d_series if v is not None and v == v]
        if len(clean_d) >= 2 and len(clean_k) >= 2:
            k_prev, k_curr = clean_k[-2], clean_k[-1]
            d_prev, d_curr = clean_d[-2], clean_d[-1]
            if k_prev <= d_prev and k_curr > d_curr:
                golden_cross = True
            elif k_prev >= d_prev and k_curr < d_curr:
                dead_cross = True

    return {
        'direction':    direction,
        'golden_cross': golden_cross,
        'dead_cross':   dead_cross,
        'strength':     strength,
    }


# ════════════════════════════════════════════════
# 일봉 게이트 판단
# ════════════════════════════════════════════════

def evaluate_daily_gate(daily_presets: dict) -> dict:
    """
    일봉 게이트: K≤20 + 하락 or 횡보일 때만 통과
    Returns: { pass: bool, reason: str, k: float, direction: dict }
    """
    short = daily_presets.get('short', {})
    k     = short.get('k', float('nan'))
    k_ser = short.get('k_series', [])
    d_ser = short.get('d_series', [])

    if k != k or k > OVERSOLD_THRESHOLD:
        return {
            'pass': False,
            'reason': f'일봉 K {k:.1f} > {OVERSOLD_THRESHOLD} (과매도 아님)',
            'k': k,
            'direction': None
        }

    direction = calc_direction(k_ser, d_ser)
    d = direction['direction']

    # 이미 강하게 반등 중이면 탈락 (늦은 진입)
    if d == 'rising' and direction['strength'] == 'strong':
        return {
            'pass': False,
            'reason': f'일봉 K {k:.1f} 이미 강한 반등 중 (늦은 진입)',
            'k': k,
            'direction': direction
        }

    return {
        'pass': True,
        'reason': f'일봉 K {k:.1f} 과매도 + {d}',
        'k': k,
        'direction': direction
    }


# ════════════════════════════════════════════════
# Watch 점수 계산
# ════════════════════════════════════════════════

def calc_watch_score(
    daily_presets:  dict,
    h4_presets:     dict,
    h1_presets:     dict,
    volume_ratio:   float = 1.0,   # 현재 거래량 / 20일 평균
    initial_daily_k: float = None, # 최초 등록 시 일봉 K (추가하락 보너스용)
    btc_change_pct:  float = 0.0,  # BTC 최근 변화율
    coin_change_pct: float = 0.0,  # 해당 코인 최근 변화율
) -> dict:
    """
    점수 계산 (총 100점)
    일봉 30점 + 4h 40점 + 1h 20점 + 거래량 5점 + 추가하락 5점
    """
    score = 0
    breakdown = {}

    # ── 일봉 (30점) ──────────────────────────────
    daily_short = daily_presets.get('short', {})
    dk  = daily_short.get('k', float('nan'))
    dk_ser = daily_short.get('k_series', [])
    dd_ser = daily_short.get('d_series', [])
    daily_dir = calc_direction(dk_ser, dd_ser)

    # 위치 점수 (최대 15점)
    if dk <= 5:   dp = 15
    elif dk <= 10: dp = 12
    elif dk <= 15: dp = 8
    else:          dp = 4
    score += dp
    breakdown['daily_position'] = dp

    # 방향 점수 (최대 15점)
    d_dir = daily_dir['direction']
    if daily_dir['golden_cross']:               dd = 15
    elif d_dir in ('reversing_up',):            dd = 12
    elif d_dir == 'sideways':                   dd = 8
    elif d_dir == 'rising' and dk <= 10:        dd = 10
    elif d_dir == 'falling':                    dd = 3
    else:                                       dd = 5
    score += dd
    breakdown['daily_direction'] = dd

    # ── 4h (40점) ────────────────────────────────
    h4_short = h4_presets.get('short', {}) if h4_presets else {}
    hk  = h4_short.get('k', float('nan'))
    hk_ser = h4_short.get('k_series', [])
    hd_ser = h4_short.get('d_series', [])
    h4_dir = calc_direction(hk_ser, hd_ser)

    # 위치 점수 (최대 20점)
    if hk != hk:    hp = 0
    elif hk <= 5:   hp = 20
    elif hk <= 10:  hp = 15
    elif hk <= 20:  hp = 8
    else:           hp = 0
    score += hp
    breakdown['h4_position'] = hp

    # 방향/골든크로스 점수 (최대 20점)
    h_dir = h4_dir['direction']
    if h4_dir['golden_cross']:                  hd = 20
    elif h_dir == 'reversing_up':               hd = 15
    elif h_dir == 'rising' and hk <= 20:        hd = 12
    elif h_dir == 'sideways' and hk <= 20:      hd = 8
    elif h_dir == 'falling':                    hd = 2
    else:                                       hd = 4
    score += hd
    breakdown['h4_direction'] = hd

    # ── 1h (20점) ────────────────────────────────
    h1_short = h1_presets.get('short', {}) if h1_presets else {}
    lk  = h1_short.get('k', float('nan'))
    lk_ser = h1_short.get('k_series', [])
    ld_ser = h1_short.get('d_series', [])
    h1_dir = calc_direction(lk_ser, ld_ser)

    # 위치 점수 (최대 10점)
    if lk != lk:    lp = 0
    elif lk <= 5:   lp = 10
    elif lk <= 10:  lp = 7
    elif lk <= 20:  lp = 4
    else:           lp = 0
    score += lp
    breakdown['h1_position'] = lp

    # 방향/골든크로스 점수 (최대 10점)
    l_dir = h1_dir['direction']
    if h1_dir['golden_cross']:                  ld = 10
    elif l_dir == 'reversing_up':               ld = 8
    elif l_dir == 'rising' and lk <= 20:        ld = 6
    elif l_dir == 'sideways' and lk <= 20:      ld = 4
    elif l_dir == 'falling':                    ld = 1
    else:                                       ld = 2
    score += ld
    breakdown['h1_direction'] = ld

    # ── 거래량 보너스 (최대 5점) ─────────────────
    if volume_ratio >= 3.0:    vb = 5
    elif volume_ratio >= 2.0:  vb = 3
    elif volume_ratio >= 1.5:  vb = 1
    else:                      vb = 0
    score += vb
    breakdown['volume_bonus'] = vb

    # ── 추가하락 보너스 (최대 5점) ───────────────
    ab = 0
    if initial_daily_k is not None and dk == dk:
        drop = initial_daily_k - dk
        if drop >= 5:    ab = 5
        elif drop >= 3:  ab = 3
        elif drop >= 1:  ab = 1
        elif drop < 0:   ab = -3  # 오히려 상승 패널티
    score += ab
    breakdown['additional_drop_bonus'] = ab

    # ── 최종 점수/등급 ────────────────────────────
    score = max(0, min(100, score))

    if score >= GRADE_S_THRESHOLD:   grade = 'S'
    elif score >= GRADE_A_THRESHOLD: grade = 'A'
    elif score >= GRADE_B_THRESHOLD: grade = 'B'
    else:                            grade = 'C'

    return {
        'score':      score,
        'grade':      grade,
        'breakdown':  breakdown,
        'daily_dir':  daily_dir,
        'h4_dir':     h4_dir,
        'h1_dir':     h1_dir,
        'daily_k':    round(dk, 2) if dk == dk else None,
        'h4_k':       round(hk, 2) if hk == hk else None,
        'h1_k':       round(lk, 2) if lk == lk else None,
    }


# ════════════════════════════════════════════════
# 매크로 필터
# ════════════════════════════════════════════════

def evaluate_macro_filter(weekly_closes: list, daily_closes: list) -> dict:
    if not MACRO_FILTER_ENABLED:
        return {'safe': True, 'reason': 'macro filter disabled',
                'weekly_ma20': None, 'weekly_distance_pct': None,
                'daily_ma20': None, 'daily_distance_pct': None}
    if len(weekly_closes) < MACRO_MA_PERIOD:
        return {'safe': False, 'reason': 'insufficient weekly data',
                'weekly_ma20': None, 'weekly_distance_pct': None,
                'daily_ma20': None, 'daily_distance_pct': None}

    w_ma  = sum(weekly_closes[-MACRO_MA_PERIOD:]) / MACRO_MA_PERIOD
    w_cur = weekly_closes[-1]
    w_dist = (w_cur - w_ma) / w_ma * 100

    d_ma, d_dist = None, None
    if len(daily_closes) >= MACRO_MA_PERIOD:
        d_ma   = sum(daily_closes[-MACRO_MA_PERIOD:]) / MACRO_MA_PERIOD
        d_cur  = daily_closes[-1]
        d_dist = (d_cur - d_ma) / d_ma * 100

    safe   = w_cur >= w_ma
    reason = f"BTC 주봉 MA20 {'위' if safe else '아래'} ({w_dist:+.2f}%)"

    return {
        'safe':                 safe,
        'reason':               reason,
        'weekly_ma20':          round(w_ma, 0),
        'weekly_distance_pct':  round(w_dist, 2),
        'daily_ma20':           round(d_ma, 0) if d_ma else None,
        'daily_distance_pct':   round(d_dist, 2) if d_dist else None,
    }


def get_module_config() -> dict:
    return {
        'version':            VERSION,
        'preset_short':       PRESET_SHORT,
        'preset_mid':         PRESET_MID,
        'preset_long':        PRESET_LONG,
        'oversold_threshold': OVERSOLD_THRESHOLD,
        'recovery_k':         RECOVERY_K,
        'macro_filter':       MACRO_FILTER_ENABLED,
        'macro_ma_period':    MACRO_MA_PERIOD,
        'grade_s':            GRADE_S_THRESHOLD,
        'grade_a':            GRADE_A_THRESHOLD,
        'grade_b':            GRADE_B_THRESHOLD,
        'watch_expiry_days':  WATCH_EXPIRY_DAYS,
    }
