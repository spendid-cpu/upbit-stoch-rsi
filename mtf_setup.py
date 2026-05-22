# -*- coding: utf-8 -*-
"""
mtf_setup.py — Upbit MTF Stochastic RSI 분석 모듈
Version : v4.2.0
Changelog:
  v4.2.0 - DEEP Watch 시스템 추가
           calc_deep_score: K극단 + BTC상대강도 + 바닥다지기 + 거래량소진 + 주봉
           evaluate_deep_condition: DEEP 진입 조건 판단
           get_deep_grade: DEEP 점수 → 등급
  v4.1.1 - calc_entry_strength grade 상한 적용 (이후 폐지)
  v4.1.0 - 4hK 과열 페널티, 등급별 만료, calc_entry_strength
  v4.0.1 - safe_k 방식 daily_k 버그 수정
  v4.0.0 - Watch 점수/등급 시스템
"""

import os
import math

VERSION = 'v4.2.0'

# ── Stoch RSI 프리셋 ─────────────────────────────────────────────
PRESET_SHORT = (
    int(os.getenv('PRESET_SHORT_RSI',     '5')),
    int(os.getenv('PRESET_SHORT_STOCH',   '5')),
    int(os.getenv('PRESET_SHORT_SMOOTHK', '3')),
    int(os.getenv('PRESET_SHORT_SMOOTHD', '3')),
)
PRESET_MID = (
    int(os.getenv('PRESET_MID_RSI',     '10')),
    int(os.getenv('PRESET_MID_STOCH',   '10')),
    int(os.getenv('PRESET_MID_SMOOTHK', '6')),
    int(os.getenv('PRESET_MID_SMOOTHD', '6')),
)
PRESET_LONG = (
    int(os.getenv('PRESET_LONG_RSI',     '20')),
    int(os.getenv('PRESET_LONG_STOCH',   '20')),
    int(os.getenv('PRESET_LONG_SMOOTHK', '12')),
    int(os.getenv('PRESET_LONG_SMOOTHD', '12')),
)

# ── 임계값 ───────────────────────────────────────────────────────
OVERSOLD_THRESHOLD = float(os.getenv('OVERSOLD_THRESHOLD', '20.0'))
RECOVERY_K         = float(os.getenv('RECOVERY_K',         '50.0'))
MACRO_ENABLED      = os.getenv('MACRO_ENABLED', 'true').lower() == 'true'
MACRO_MA_PERIOD    = int(os.getenv('MACRO_MA_PERIOD', '20'))

# ── 등급 기준 ────────────────────────────────────────────────────
GRADE_S_THRESHOLD = int(os.getenv('GRADE_S_THRESHOLD', '80'))
GRADE_A_THRESHOLD = int(os.getenv('GRADE_A_THRESHOLD', '60'))
GRADE_B_THRESHOLD = int(os.getenv('GRADE_B_THRESHOLD', '40'))

# ── Watch 만료 (등급별) ──────────────────────────────────────────
WATCH_EXPIRY_DAYS_C = int(os.getenv('WATCH_EXPIRY_DAYS_C', '3'))
WATCH_EXPIRY_DAYS_B = int(os.getenv('WATCH_EXPIRY_DAYS_B', '5'))
WATCH_EXPIRY_DAYS   = int(os.getenv('WATCH_EXPIRY_DAYS',   '7'))

# ── 4h 과열 페널티 ───────────────────────────────────────────────
H4_OVERHEAT_THRESHOLD = float(os.getenv('H4_OVERHEAT_THRESHOLD', '80.0'))
H4_OVERHEAT_PENALTY   = int(os.getenv('H4_OVERHEAT_PENALTY',     '10'))
H4_WARM_THRESHOLD     = float(os.getenv('H4_WARM_THRESHOLD',     '50.0'))
H4_WARM_PENALTY       = int(os.getenv('H4_WARM_PENALTY',         '5'))

# ── DEEP Watch 기준 ──────────────────────────────────────────────
DEEP_K_THRESHOLD         = float(os.getenv('DEEP_K_THRESHOLD',         '5.0'))
DEEP_BTC_DROP_MIN        = float(os.getenv('DEEP_BTC_DROP_MIN',        '-2.0'))
DEEP_RELATIVE_MIN        = float(os.getenv('DEEP_RELATIVE_MIN',        '3.0'))
DEEP_GRADE_S_THRESHOLD   = int(os.getenv('DEEP_GRADE_S_THRESHOLD',     '80'))
DEEP_GRADE_A_THRESHOLD   = int(os.getenv('DEEP_GRADE_A_THRESHOLD',     '60'))
DEEP_GRADE_B_THRESHOLD   = int(os.getenv('DEEP_GRADE_B_THRESHOLD',     '40'))
DEEP_BOTTOM_DAYS_STRONG  = int(os.getenv('DEEP_BOTTOM_DAYS_STRONG',    '3'))
DEEP_BOTTOM_DAYS_MID     = int(os.getenv('DEEP_BOTTOM_DAYS_MID',       '2'))
DEEP_VOL_EXHAUSTED       = float(os.getenv('DEEP_VOL_EXHAUSTED',       '0.3'))
DEEP_VOL_LOW             = float(os.getenv('DEEP_VOL_LOW',             '0.5'))


# ═══════════════════════════════════════════════════════════════
# 기본 계산 함수
# ═══════════════════════════════════════════════════════════════
def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return []
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0)      for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]

    avg_gain = sum(gains[:period])  / period
    avg_loss = sum(losses[:period]) / period

    rsi_values = []
    for i in range(period, len(deltas)):
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - 100 / (1 + rs))
        avg_gain = (avg_gain * (period - 1) + gains[i])  / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    return rsi_values


def calc_stoch_rsi(closes, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    rsi_vals = calc_rsi(closes, rsi_period)
    if len(rsi_vals) < stoch_period:
        return {'k': None, 'd': None, 'k_series': [], 'd_series': []}

    raw_k = []
    for i in range(stoch_period - 1, len(rsi_vals)):
        window = rsi_vals[i - stoch_period + 1: i + 1]
        lo, hi = min(window), max(window)
        denom  = hi - lo
        raw_k.append((rsi_vals[i] - lo) / denom * 100 if denom != 0 else 50.0)

    if len(raw_k) < smooth_k:
        return {'k': None, 'd': None, 'k_series': [], 'd_series': []}

    k_series = []
    for i in range(smooth_k - 1, len(raw_k)):
        k_series.append(sum(raw_k[i - smooth_k + 1: i + 1]) / smooth_k)

    if len(k_series) < smooth_d:
        return {'k': None, 'd': None, 'k_series': k_series, 'd_series': []}

    d_series = []
    for i in range(smooth_d - 1, len(k_series)):
        d_series.append(sum(k_series[i - smooth_d + 1: i + 1]) / smooth_d)

    return {
        'k':        k_series[-1] if k_series else None,
        'd':        d_series[-1] if d_series else None,
        'k_series': k_series,
        'd_series': d_series,
    }


def calc_all_presets(closes):
    result = {}
    for name, preset in [('short', PRESET_SHORT), ('mid', PRESET_MID), ('long', PRESET_LONG)]:
        rp, sp, sk, sd = preset
        result[name] = calc_stoch_rsi(closes, rp, sp, sk, sd)
    return result


# ═══════════════════════════════════════════════════════════════
# 방향성 분석
# ═══════════════════════════════════════════════════════════════
def calc_direction(k_series, d_series=None):
    if not k_series or len(k_series) < 2:
        return {'direction': '알수없음', 'strength': '약', 'golden_cross': False}

    k_cur   = k_series[-1]
    k_prev1 = k_series[-2]
    k_prev2 = k_series[-3] if len(k_series) >= 3 else k_prev1

    if k_cur > k_prev1 and k_prev1 > k_prev2:
        direction = '상승'
        strength  = '강' if (k_cur - k_prev2) > 10 else '보통'
    elif k_cur > k_prev1:
        direction = '반등'
        strength  = '약'
    elif k_cur < k_prev1 and k_prev1 < k_prev2:
        direction = '하락'
        strength  = '강' if (k_prev2 - k_cur) > 10 else '보통'
    elif k_cur < k_prev1:
        direction = '하락'
        strength  = '약'
    else:
        direction = '횡보'
        strength  = '약'

    golden_cross = False
    if d_series and len(d_series) >= 2 and len(k_series) >= 2:
        golden_cross = (k_series[-2] <= d_series[-2]) and (k_series[-1] > d_series[-1])

    return {'direction': direction, 'strength': strength, 'golden_cross': golden_cross}


def get_direction_icon(direction, golden_cross=False):
    if golden_cross:
        return {'icon': '✨', 'css': 'text-yellow'}
    return {
        '상승':    {'icon': '↑',  'css': 'text-green'},
        '반등':    {'icon': '↗',  'css': 'text-lime'},
        '횡보':    {'icon': '→',  'css': 'text-gray'},
        '하락':    {'icon': '↓',  'css': 'text-red'},
        '알수없음': {'icon': '-',  'css': 'text-gray'},
    }.get(direction, {'icon': '-', 'css': 'text-gray'})


# ═══════════════════════════════════════════════════════════════
# 일봉 게이트
# ═══════════════════════════════════════════════════════════════
def evaluate_daily_gate(daily_presets):
    short = daily_presets.get('short', {})
    k_val = short.get('k')

    if k_val is None:
        return {'pass': False, 'reason': '데이터 부족', 'daily_k': None}
    if k_val > OVERSOLD_THRESHOLD:
        return {
            'pass':    False,
            'reason':  f'일봉K({k_val:.1f}) > {OVERSOLD_THRESHOLD}',
            'daily_k': round(k_val, 2),
        }

    k_series = short.get('k_series', [])
    d_series = short.get('d_series', [])
    dir_info = calc_direction(k_series, d_series)

    return {
        'pass':      True,
        'reason':    f'일봉K({k_val:.1f}) 과매도, 방향:{dir_info["direction"]}',
        'daily_k':   round(k_val, 2),
        'direction': dir_info['direction'],
    }


# ═══════════════════════════════════════════════════════════════
# safe_k 헬퍼
# ═══════════════════════════════════════════════════════════════
def _safe_k(presets, preset_key='short'):
    if not presets:
        return None
    sub = presets.get(preset_key, {})
    if not sub:
        return None
    val = sub.get('k', None)
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 2)
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════
# 진입 추천 강도 (등급 상한 없음 - 순수 방향 기반)
# ═══════════════════════════════════════════════════════════════
def calc_entry_strength(daily_dir, h4_dir, h1_dir,
                        h4_golden=False, h1_golden=False):
    strong_dirs = {'상승', '반등'}

    if h4_golden or h1_golden:
        raw_level = 3
    elif daily_dir in strong_dirs and h4_dir in strong_dirs and h1_dir in strong_dirs:
        raw_level = 3
    elif h4_dir in strong_dirs and h1_dir in strong_dirs:
        raw_level = 2
    elif h4_dir in strong_dirs or (daily_dir in strong_dirs and h4_dir == '횡보'):
        raw_level = 1
    else:
        raw_level = 0

    labels = {
        3: ('강한신호', '🚀'),
        2: ('진입고려', '🎯'),
        1: ('관찰',    '👀'),
        0: ('대기',    '⏳'),
    }
    label, icon = labels[raw_level]
    return {'level': raw_level, 'label': label, 'icon': icon}


# ═══════════════════════════════════════════════════════════════
# DEEP Watch 점수 계산 (핵심 신규)
# ═══════════════════════════════════════════════════════════════
def calc_deep_score(daily_k, btc_change_24h, coin_change_24h,
                    days_at_bottom=0, vol_ratio=1.0,
                    weekly_k=None):
    """
    DEEP Watch 점수 계산 (최대 100점)

    구성:
      K값 위치     최대 30점
      상대강도     최대 30점
      바닥 다지기  최대 20점
      거래량 소진  최대 10점
      주봉 과매도  최대 10점
    """
    score     = 0
    breakdown = {}

    # ── K값 위치 점수 (30점) ─────────────────────────────────────
    if daily_k is None:
        k_score = 0
    elif daily_k <= 2:
        k_score = 30
    elif daily_k <= 3:
        k_score = 27
    elif daily_k <= 5:
        k_score = 25
    else:
        k_score = 0
    score += k_score
    breakdown['k_position'] = k_score

    # ── 상대강도 점수 (30점) ─────────────────────────────────────
    # BTC 하락 중 + 종목 버팀
    rel_score = 0
    if btc_change_24h is not None and coin_change_24h is not None:
        try:
            btc_ch  = float(btc_change_24h)
            coin_ch = float(coin_change_24h)
            # 상대강도 = 종목변화 - BTC변화 (클수록 버팀)
            relative = coin_ch - btc_ch

            if btc_ch <= -3.0 and relative >= 4.0:
                rel_score = 30
            elif btc_ch <= -3.0 and relative >= 3.0:
                rel_score = 25
            elif btc_ch <= -2.0 and relative >= 4.0:
                rel_score = 25
            elif btc_ch <= -2.0 and relative >= 3.0:
                rel_score = 20
            elif btc_ch <= -1.0 and relative >= 3.0:
                rel_score = 10
            else:
                rel_score = 0

            breakdown['btc_change']  = round(btc_ch,  2)
            breakdown['coin_change'] = round(coin_ch, 2)
            breakdown['relative']    = round(relative, 2)
        except (TypeError, ValueError):
            pass
    score += rel_score
    breakdown['relative_strength'] = rel_score

    # ── 바닥 다지기 점수 (20점) ──────────────────────────────────
    # K≤5 구간 유지 일수
    if days_at_bottom >= DEEP_BOTTOM_DAYS_STRONG:
        bottom_score = 20
    elif days_at_bottom >= DEEP_BOTTOM_DAYS_MID:
        bottom_score = 10
    elif days_at_bottom >= 1:
        bottom_score = 5
    else:
        bottom_score = 0
    score += bottom_score
    breakdown['bottom_days']  = days_at_bottom
    breakdown['bottom_score'] = bottom_score

    # ── 거래량 소진 점수 (10점) ──────────────────────────────────
    if vol_ratio <= DEEP_VOL_EXHAUSTED:
        vol_score = 10
    elif vol_ratio <= DEEP_VOL_LOW:
        vol_score = 5
    elif vol_ratio <= 0.8:
        vol_score = 2
    else:
        vol_score = 0
    score += vol_score
    breakdown['vol_ratio'] = round(vol_ratio, 2)
    breakdown['vol_score'] = vol_score

    # ── 주봉 과매도 보너스 (10점) ────────────────────────────────
    weekly_score = 0
    if weekly_k is not None:
        try:
            wk = float(weekly_k)
            if wk <= 10:
                weekly_score = 10
            elif wk <= 20:
                weekly_score = 7
            elif wk <= 30:
                weekly_score = 3
        except (TypeError, ValueError):
            pass
    score += weekly_score
    breakdown['weekly_k']    = weekly_k
    breakdown['weekly_score']= weekly_score

    score = max(0, min(100, score))
    return {
        'score':     score,
        'breakdown': breakdown,
    }


def evaluate_deep_condition(daily_k, btc_change_24h, coin_change_24h):
    """
    DEEP Watch 진입 필수 조건 체크
    반환: {'pass': bool, 'reason': str}
    """
    # 필수 조건 1: 일봉K 극단 과매도
    if daily_k is None or daily_k > DEEP_K_THRESHOLD:
        return {
            'pass':   False,
            'reason': f'일봉K({daily_k}) > {DEEP_K_THRESHOLD}',
        }

    # 필수 조건 2: BTC 하락 중
    if btc_change_24h is None or btc_change_24h > DEEP_BTC_DROP_MIN:
        return {
            'pass':   False,
            'reason': f'BTC 24h({btc_change_24h}%) > {DEEP_BTC_DROP_MIN}%',
        }

    # 필수 조건 3: 종목 상대 버팀
    if coin_change_24h is None:
        return {'pass': False, 'reason': '종목 변화율 없음'}

    relative = coin_change_24h - btc_change_24h
    if relative < DEEP_RELATIVE_MIN:
        return {
            'pass':   False,
            'reason': f'상대강도({relative:.1f}%) < {DEEP_RELATIVE_MIN}%',
        }

    return {
        'pass':     True,
        'reason':   f'K:{daily_k:.1f} BTC:{btc_change_24h:.1f}% 종목:{coin_change_24h:.1f}% 상대:{relative:.1f}%',
        'relative': round(relative, 2),
    }


def get_deep_grade(deep_score):
    """DEEP 점수 → 등급"""
    if deep_score >= DEEP_GRADE_S_THRESHOLD:
        return 'DEEP-S'
    elif deep_score >= DEEP_GRADE_A_THRESHOLD:
        return 'DEEP-A'
    elif deep_score >= DEEP_GRADE_B_THRESHOLD:
        return 'DEEP-B'
    else:
        return 'DEEP-C'


# ═══════════════════════════════════════════════════════════════
# Watch 점수 계산
# ═══════════════════════════════════════════════════════════════
def calc_watch_score(daily_presets, h4_presets=None, h1_presets=None,
                     vol_ratio=1.0, snap_k=None):
    score     = 0
    breakdown = {}

    # ── 일봉 ─────────────────────────────────────────────────────
    d_short    = daily_presets.get('short', {}) if daily_presets else {}
    d_k        = d_short.get('k') or 0
    d_k_series = d_short.get('k_series', [])
    d_d_series = d_short.get('d_series', [])
    d_dir      = calc_direction(d_k_series, d_d_series)
    daily_dir  = d_dir['direction']

    if d_k <= 5:    d_pos = 15
    elif d_k <= 10: d_pos = 12
    elif d_k <= 15: d_pos = 8
    else:           d_pos = 4
    score += d_pos
    breakdown['daily_position'] = d_pos

    d_dir_map   = {'상승': 15, '반등': 12, '횡보': 6, '하락': 2, '알수없음': 0}
    d_dir_score = d_dir_map.get(daily_dir, 0)
    if d_dir.get('golden_cross'):
        d_dir_score = min(15, d_dir_score + 5)
    score += d_dir_score
    breakdown['daily_direction'] = d_dir_score

    # ── 4h ──────────────────────────────────────────────────────
    h4_dir_label = '알수없음'
    h4_golden    = False
    h4_k_val     = 0

    if h4_presets:
        h4_short    = h4_presets.get('short', {})
        h4_k_val    = h4_short.get('k') or 0
        h4_k_series = h4_short.get('k_series', [])
        h4_d_series = h4_short.get('d_series', [])
        h4_dir      = calc_direction(h4_k_series, h4_d_series)
        h4_dir_label = h4_dir['direction']
        h4_golden    = h4_dir.get('golden_cross', False)

        if h4_k_val <= 5:    h4_pos = 20
        elif h4_k_val <= 10: h4_pos = 16
        elif h4_k_val <= 20: h4_pos = 10
        else:                h4_pos = 0
        score += h4_pos
        breakdown['h4_position'] = h4_pos

        h4_dir_map   = {'상승': 20, '반등': 16, '횡보': 8, '하락': 2, '알수없음': 0}
        h4_dir_score = h4_dir_map.get(h4_dir_label, 0)
        if h4_golden:
            h4_dir_score = min(20, h4_dir_score + 8)
        score += h4_dir_score
        breakdown['h4_direction'] = h4_dir_score

        h4_penalty = 0
        if h4_k_val > H4_OVERHEAT_THRESHOLD:
            h4_penalty = -H4_OVERHEAT_PENALTY
        elif h4_k_val > H4_WARM_THRESHOLD:
            h4_penalty = -H4_WARM_PENALTY
        score += h4_penalty
        breakdown['h4_penalty'] = h4_penalty
    else:
        breakdown['h4_position']  = 0
        breakdown['h4_direction'] = 0
        breakdown['h4_penalty']   = 0

    # ── 1h ──────────────────────────────────────────────────────
    h1_dir_label = '알수없음'
    h1_golden    = False

    if h1_presets:
        h1_short    = h1_presets.get('short', {})
        h1_k_val    = h1_short.get('k') or 0
        h1_k_series = h1_short.get('k_series', [])
        h1_d_series = h1_short.get('d_series', [])
        h1_dir      = calc_direction(h1_k_series, h1_d_series)
        h1_dir_label = h1_dir['direction']
        h1_golden    = h1_dir.get('golden_cross', False)

        if h1_k_val <= 5:    h1_pos = 10
        elif h1_k_val <= 10: h1_pos = 8
        elif h1_k_val <= 20: h1_pos = 5
        else:                h1_pos = 0
        score += h1_pos
        breakdown['h1_position'] = h1_pos

        h1_dir_map   = {'상승': 10, '반등': 8, '횡보': 4, '하락': 1, '알수없음': 0}
        h1_dir_score = h1_dir_map.get(h1_dir_label, 0)
        if h1_golden:
            h1_dir_score = min(10, h1_dir_score + 4)
        score += h1_dir_score
        breakdown['h1_direction'] = h1_dir_score
    else:
        breakdown['h1_position']  = 0
        breakdown['h1_direction'] = 0

    # ── 거래량 보너스 (5점) ──────────────────────────────────────
    if vol_ratio >= 3.0:   vol_bonus = 5
    elif vol_ratio >= 2.0: vol_bonus = 3
    elif vol_ratio >= 1.5: vol_bonus = 2
    else:                  vol_bonus = 0
    score += vol_bonus
    breakdown['volume_bonus'] = vol_bonus

    # ── 추가 하락 보너스 (5점) ───────────────────────────────────
    drop_bonus = 0
    if snap_k is not None and d_k is not None:
        try:
            drop = float(snap_k) - float(d_k)
            if drop >= 5:   drop_bonus = 5
            elif drop >= 3: drop_bonus = 3
            elif drop >= 1: drop_bonus = 1
        except (TypeError, ValueError):
            pass
    score += drop_bonus
    breakdown['drop_bonus'] = drop_bonus

    # ── 진입강도 (등급 상한 없음) ────────────────────────────────
    entry_strength = calc_entry_strength(
        daily_dir=daily_dir,
        h4_dir=h4_dir_label,
        h1_dir=h1_dir_label,
        h4_golden=h4_golden,
        h1_golden=h1_golden,
    )

    # ── 등급 확정 ────────────────────────────────────────────────
    score = max(0, min(100, score))
    if score >= GRADE_S_THRESHOLD:   grade = 'S'
    elif score >= GRADE_A_THRESHOLD: grade = 'A'
    elif score >= GRADE_B_THRESHOLD: grade = 'B'
    else:                            grade = 'C'

    return {
        'score':          score,
        'grade':          grade,
        'breakdown':      breakdown,
        'daily_dir':      daily_dir,
        'h4_dir':         h4_dir_label,
        'h1_dir':         h1_dir_label,
        'h4_golden':      h4_golden,
        'h1_golden':      h1_golden,
        'entry_strength': entry_strength,
        'daily_k':        _safe_k(daily_presets, 'short'),
        'h4_k':           _safe_k(h4_presets,    'short'),
        'h1_k':           _safe_k(h1_presets,    'short'),
    }


# ═══════════════════════════════════════════════════════════════
# 등급별 만료 기간
# ═══════════════════════════════════════════════════════════════
def get_expiry_days(grade):
    return {
        'C':      WATCH_EXPIRY_DAYS_C,
        'B':      WATCH_EXPIRY_DAYS_B,
        'A':      None,
        'S':      None,
        'DEEP-S': None,
        'DEEP-A': None,
        'DEEP-B': None,
        'DEEP-C': WATCH_EXPIRY_DAYS_C,
    }.get(grade, WATCH_EXPIRY_DAYS)


# ═══════════════════════════════════════════════════════════════
# 매크로 필터
# ═══════════════════════════════════════════════════════════════
def evaluate_macro_filter(weekly_closes, daily_closes=None):
    if not MACRO_ENABLED:
        return {'ok': True, 'weekly_ma20': None, 'daily_ma20': None, 'btc_price': None}
    if len(weekly_closes) < MACRO_MA_PERIOD:
        return {'ok': True, 'weekly_ma20': None, 'daily_ma20': None, 'btc_price': None}

    weekly_ma20 = sma(weekly_closes, MACRO_MA_PERIOD)
    btc_price   = weekly_closes[-1]
    macro_ok    = btc_price >= weekly_ma20

    daily_ma20 = None
    if daily_closes and len(daily_closes) >= MACRO_MA_PERIOD:
        daily_ma20 = sma(daily_closes, MACRO_MA_PERIOD)

    return {
        'ok':          macro_ok,
        'weekly_ma20': round(weekly_ma20, 0) if weekly_ma20 else None,
        'daily_ma20':  round(daily_ma20,  0) if daily_ma20  else None,
        'btc_price':   round(btc_price,   0) if btc_price   else None,
    }


# ═══════════════════════════════════════════════════════════════
# 설정 반환
# ═══════════════════════════════════════════════════════════════
def get_module_config():
    return {
        'VERSION':               VERSION,
        'PRESET_SHORT':          PRESET_SHORT,
        'PRESET_MID':            PRESET_MID,
        'PRESET_LONG':           PRESET_LONG,
        'OVERSOLD_THRESHOLD':    OVERSOLD_THRESHOLD,
        'RECOVERY_K':            RECOVERY_K,
        'MACRO_ENABLED':         MACRO_ENABLED,
        'MACRO_MA_PERIOD':       MACRO_MA_PERIOD,
        'GRADE_S_THRESHOLD':     GRADE_S_THRESHOLD,
        'GRADE_A_THRESHOLD':     GRADE_A_THRESHOLD,
        'GRADE_B_THRESHOLD':     GRADE_B_THRESHOLD,
        'WATCH_EXPIRY_DAYS_C':   WATCH_EXPIRY_DAYS_C,
        'WATCH_EXPIRY_DAYS_B':   WATCH_EXPIRY_DAYS_B,
        'H4_OVERHEAT_THRESHOLD': H4_OVERHEAT_THRESHOLD,
        'H4_OVERHEAT_PENALTY':   H4_OVERHEAT_PENALTY,
        'H4_WARM_THRESHOLD':     H4_WARM_THRESHOLD,
        'H4_WARM_PENALTY':       H4_WARM_PENALTY,
        'DEEP_K_THRESHOLD':      DEEP_K_THRESHOLD,
        'DEEP_BTC_DROP_MIN':     DEEP_BTC_DROP_MIN,
        'DEEP_RELATIVE_MIN':     DEEP_RELATIVE_MIN,
        'DEEP_GRADE_S_THRESHOLD':DEEP_GRADE_S_THRESHOLD,
        'DEEP_GRADE_A_THRESHOLD':DEEP_GRADE_A_THRESHOLD,
    }
