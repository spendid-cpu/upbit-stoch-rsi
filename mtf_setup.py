"""
mtf_setup.py v4.2.1
Upbit MTF Stochastic RSI 분석 모듈

변경사항 (v4.2.1):
- calc_entry_strength: 4hK > 80이면 강도 1단계 강제 하향
- calc_entry_strength: 일봉K=0~2 극단 과매도 + 방향 있으면 최소 레벨 2(👀) 보장
- get_direction_icon: GX 포맷 통일 (✨GX로 변경)
- calc_deep_score: 일봉K=0 케이스 보너스 추가
- evaluate_deep_condition: BTC 하락 기준 완화 옵션 추가 (DEEP_BTC_DROP_MIN 환경변수)
"""

import os
import math
import logging

logger = logging.getLogger(__name__)

VERSION = 'v4.2.1'

# ── 기본 상수 ────────────────────────────────────────────────
OVERSOLD_THRESHOLD   = float(os.environ.get('OVERSOLD_THRESHOLD', 20.0))
RECOVERY_K           = float(os.environ.get('RECOVERY_K', 50.0))
MACRO_ENABLED        = os.environ.get('MACRO_ENABLED', 'true').lower() == 'true'
MACRO_MA_PERIOD      = int(os.environ.get('MACRO_MA_PERIOD', 20))

GRADE_S_THRESHOLD    = int(os.environ.get('GRADE_S_THRESHOLD', 80))
GRADE_A_THRESHOLD    = int(os.environ.get('GRADE_A_THRESHOLD', 60))
GRADE_B_THRESHOLD    = int(os.environ.get('GRADE_B_THRESHOLD', 40))

WATCH_EXPIRY_DAYS    = int(os.environ.get('WATCH_EXPIRY_DAYS', 7))
WATCH_EXPIRY_DAYS_B  = int(os.environ.get('WATCH_EXPIRY_DAYS_B', 5))
WATCH_EXPIRY_DAYS_C  = int(os.environ.get('WATCH_EXPIRY_DAYS_C', 3))

# 4h 과열 임계값
H4_OVERHEAT_THRESHOLD = float(os.environ.get('H4_OVERHEAT_THRESHOLD', 80.0))
H4_WARM_THRESHOLD     = float(os.environ.get('H4_WARM_THRESHOLD', 50.0))

# DEEP Watch 상수
DEEP_K_THRESHOLD     = float(os.environ.get('DEEP_K_THRESHOLD', 5.0))
DEEP_BTC_DROP_MIN    = float(os.environ.get('DEEP_BTC_DROP_MIN', -2.0))   # BTC 24h 변동 임계값
DEEP_RELATIVE_MIN    = float(os.environ.get('DEEP_RELATIVE_MIN', 3.0))    # 코인이 BTC보다 N% 이상 버텨야
DEEP_GRADE_S         = int(os.environ.get('DEEP_GRADE_S', 80))
DEEP_GRADE_A         = int(os.environ.get('DEEP_GRADE_A', 60))
DEEP_GRADE_B         = int(os.environ.get('DEEP_GRADE_B', 40))

# Stochastic RSI 프리셋 (환경변수로 조정 가능)
_SHORT_PARAMS = (
    int(os.environ.get('SHORT_RSI_PERIOD', 5)),
    int(os.environ.get('SHORT_STOCH_PERIOD', 5)),
    int(os.environ.get('SHORT_K_PERIOD', 3)),
    int(os.environ.get('SHORT_D_PERIOD', 3)),
)
_MID_PARAMS = (
    int(os.environ.get('MID_RSI_PERIOD', 10)),
    int(os.environ.get('MID_STOCH_PERIOD', 10)),
    int(os.environ.get('MID_K_PERIOD', 6)),
    int(os.environ.get('MID_D_PERIOD', 6)),
)
_LONG_PARAMS = (
    int(os.environ.get('LONG_RSI_PERIOD', 20)),
    int(os.environ.get('LONG_STOCH_PERIOD', 20)),
    int(os.environ.get('LONG_K_PERIOD', 12)),
    int(os.environ.get('LONG_D_PERIOD', 12)),
)


# ── 수학 헬퍼 ────────────────────────────────────────────────
def _is_valid(v):
    """NaN / Inf 체크"""
    if v is None:
        return False
    try:
        f = float(v)
        return not (math.isnan(f) or math.isinf(f))
    except Exception:
        return False


def _safe_float(v, default=None):
    if not _is_valid(v):
        return default
    return float(v)


# ── SMA ──────────────────────────────────────────────────────
def sma(data: list, period: int) -> list:
    result = []
    for i in range(len(data)):
        if i < period - 1:
            result.append(None)
        else:
            window = data[i - period + 1: i + 1]
            if any(v is None for v in window):
                result.append(None)
            else:
                result.append(sum(window) / period)
    return result


# ── RSI ──────────────────────────────────────────────────────
def calc_rsi(closes: list, period: int = 14) -> list:
    if len(closes) < period + 1:
        return [None] * len(closes)

    rsi_values = [None] * period
    gains, losses = [], []

    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - 100 / (1 + rs))

    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = max(diff, 0)
        loss = max(-diff, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - 100 / (1 + rs))

    return rsi_values


# ── Stochastic RSI ───────────────────────────────────────────
def calc_stoch_rsi(closes: list, rsi_period: int = 14, stoch_period: int = 14,
                   k_period: int = 3, d_period: int = 3):
    """
    Returns: (k_value, d_value, k_series)
    """
    rsi_vals = calc_rsi(closes, rsi_period)
    valid_rsi = [v for v in rsi_vals if v is not None]

    if len(valid_rsi) < stoch_period:
        return None, None, []

    stoch_k_raw = []
    for i in range(len(rsi_vals)):
        if rsi_vals[i] is None:
            stoch_k_raw.append(None)
            continue
        window = [v for v in rsi_vals[max(0, i - stoch_period + 1): i + 1] if v is not None]
        if len(window) < stoch_period:
            stoch_k_raw.append(None)
            continue
        lo, hi = min(window), max(window)
        if hi == lo:
            stoch_k_raw.append(50.0)
        else:
            stoch_k_raw.append((rsi_vals[i] - lo) / (hi - lo) * 100)

    # K 스무딩
    k_series_raw = sma(stoch_k_raw, k_period)
    k_series = [v for v in k_series_raw if v is not None]

    if not k_series:
        return None, None, []

    # D 스무딩
    d_series_raw = sma(k_series_raw, d_period)
    d_series = [v for v in d_series_raw if v is not None]

    k_val = k_series[-1]
    d_val = d_series[-1] if d_series else None

    return round(k_val, 2), (round(d_val, 2) if d_val is not None else None), k_series


# ── 전체 프리셋 계산 ──────────────────────────────────────────
def calc_all_presets(closes: list) -> dict:
    """
    Returns: {
        'short': {'k': float, 'd': float, 'k_series': list},
        'mid':   {...},
        'long':  {...}
    }
    """
    result = {}
    for label, params in [('short', _SHORT_PARAMS), ('mid', _MID_PARAMS), ('long', _LONG_PARAMS)]:
        k, d, ks = calc_stoch_rsi(closes, *params)
        result[label] = {'k': k, 'd': d, 'k_series': ks}
    return result


# ── 방향 분석 ────────────────────────────────────────────────
def calc_direction(presets: dict) -> dict:
    """
    Returns: {
        'direction': str,   # 상승/반등/횡보/하락
        'strength': int,    # 0~3
        'golden_cross': bool
    }
    """
    short = presets.get('short', {})
    mid   = presets.get('mid', {})

    k_s = _safe_float(short.get('k'))
    d_s = _safe_float(short.get('d'))
    k_m = _safe_float(mid.get('k'))
    k_s_series = short.get('k_series', [])

    golden_cross = False
    if k_s is not None and d_s is not None and len(k_s_series) >= 2:
        prev_k = k_s_series[-2] if len(k_s_series) >= 2 else None
        if prev_k is not None and prev_k <= d_s and k_s > d_s:
            golden_cross = True

    if k_s is None:
        return {'direction': '횡보', 'strength': 0, 'golden_cross': False}

    if k_s > 50 and (k_m is None or k_m > 30):
        direction = '상승'
        strength  = 3 if golden_cross else 2
    elif k_s > 20:
        direction = '반등'
        strength  = 2 if golden_cross else 1
    elif k_s <= 20:
        direction = '하락'
        strength  = 0
    else:
        direction = '횡보'
        strength  = 0

    return {'direction': direction, 'strength': strength, 'golden_cross': golden_cross}


# ── 방향 아이콘 ──────────────────────────────────────────────
def get_direction_icon(direction: str, golden_cross: bool = False) -> str:
    """GX는 ✨GX 포맷으로 통일"""
    icon_map = {
        '상승': '↑',
        '반등': '↗',
        '횡보': '→',
        '하락': '↓',
    }
    icon = icon_map.get(direction, '?')
    if golden_cross:
        return f'✨GX'
    return icon


# ── Daily 게이트 ─────────────────────────────────────────────
def evaluate_daily_gate(daily_presets: dict) -> bool:
    """일봉 K ≤ OVERSOLD_THRESHOLD 이면 통과"""
    if not daily_presets:
        return False
    k = _safe_float(daily_presets.get('short', {}).get('k'))
    if k is None:
        return False
    return k <= OVERSOLD_THRESHOLD


# ── 안전한 K 추출 ────────────────────────────────────────────
def _safe_k(presets: dict, label: str = '') -> float | None:
    if not presets:
        return None
    k = presets.get('short', {}).get('k')
    return _safe_float(k)


# ── 진입강도 계산 ────────────────────────────────────────────
def calc_entry_strength(daily_dir: dict, h4_dir: dict, h1_dir: dict,
                        daily_k: float | None = None,
                        h4_k: float | None = None) -> int:
    """
    진입강도 레벨 (0~3):
      0 → ⏳ 대기
      1 → 👀 관찰
      2 → 🎯 진입고려
      3 → 🚀 강한신호

    규칙:
    - 기본 점수: daily/h4/h1 방향 strength 합산
    - 4hK > 80: 1단계 강제 하향 (과열)
    - 일봉K ≤ 2 극단 과매도 + direction 있으면 최소 레벨 2 보장
    - 등급 캡 제거 (v4.1.1에서 복원)
    """
    if daily_dir is None:
        daily_dir = {}
    if h4_dir is None:
        h4_dir = {}
    if h1_dir is None:
        h1_dir = {}

    d_str = int(daily_dir.get('strength', 0) or 0)
    h_str = int(h4_dir.get('strength', 0) or 0)
    h1_str = int(h1_dir.get('strength', 0) or 0)

    total = d_str + h_str + h1_str  # max 9

    if total >= 7:
        level = 3
    elif total >= 4:
        level = 2
    elif total >= 2:
        level = 1
    else:
        level = 0

    # 4hK 과열 → 1단계 하향
    h4k = _safe_float(h4_k)
    if h4k is not None and h4k > H4_OVERHEAT_THRESHOLD:
        level = max(0, level - 1)

    # 일봉K 극단 과매도(≤2) + direction 있으면 최소 레벨 2 보장
    dk = _safe_float(daily_k)
    daily_direction = daily_dir.get('direction', '횡보')
    if dk is not None and dk <= 2.0 and daily_direction in ('상승', '반등'):
        level = max(level, 2)

    return level


def entry_strength_label(level: int) -> str:
    return {3: '🚀강한신호', 2: '🎯진입고려', 1: '👀관찰', 0: '⏳대기'}.get(level, '⏳대기')


# ── Watch 만료일 ─────────────────────────────────────────────
def get_expiry_days(grade: str) -> int:
    if grade == 'C':
        return WATCH_EXPIRY_DAYS_C
    if grade == 'B':
        return WATCH_EXPIRY_DAYS_B
    return WATCH_EXPIRY_DAYS  # A/S → 7일 (자동 만료 없음에 가까움)


# ── Watch 점수 계산 ──────────────────────────────────────────
def calc_watch_score(daily_presets: dict, h4_presets: dict, h1_presets: dict,
                     vol_ratio: float = 1.0, snap_k: float | None = None) -> dict:
    """
    최대 100점 구성:
      - 일봉 위치 (K값 과매도 정도)   : 15점
      - 일봉 방향                     : 15점
      - 4h 위치                       : 20점
      - 4h 방향                       : 20점
      - 1h 위치                       : 10점
      - 1h 방향                       : 10점
      - 거래량 보너스                  : 5점
      - 하락 보너스 (snap_k 대비)      : 5점
      합계: 100점
      패널티:
      - 4hK > 80: -10점
      - 4hK > 50: -5점
    """
    score     = 0
    breakdown = {}

    # ── 일봉 위치 (15점) ─────────────────────────────────────
    dk = _safe_k(daily_presets, 'daily')
    if dk is not None:
        if dk <= 5:
            pos_score = 15
        elif dk <= 10:
            pos_score = 12
        elif dk <= 15:
            pos_score = 8
        elif dk <= 20:
            pos_score = 5
        else:
            pos_score = 0
    else:
        pos_score = 0
    score += pos_score
    breakdown['daily_position'] = pos_score

    # ── 일봉 방향 (15점) ─────────────────────────────────────
    daily_dir_info = calc_direction(daily_presets) if daily_presets else {}
    daily_dir      = daily_dir_info.get('direction', '횡보')
    daily_str      = daily_dir_info.get('strength', 0)
    daily_gx       = daily_dir_info.get('golden_cross', False)

    dir_score = 0
    if daily_dir == '상승':
        dir_score = 15
    elif daily_dir == '반등':
        dir_score = 10
    elif daily_dir == '횡보':
        dir_score = 3
    if daily_gx:
        dir_score = min(15, dir_score + 3)
    score += dir_score
    breakdown['daily_direction'] = dir_score

    # ── 4h 위치 (20점) ───────────────────────────────────────
    h4k = _safe_k(h4_presets, 'h4')
    if h4k is not None:
        if h4k <= 5:
            h4_pos = 20
        elif h4k <= 10:
            h4_pos = 16
        elif h4k <= 20:
            h4_pos = 12
        elif h4k <= 50:
            h4_pos = 6
        else:
            h4_pos = 0
    else:
        h4_pos = 0
    score += h4_pos
    breakdown['h4_position'] = h4_pos

    # ── 4h 방향 (20점) ───────────────────────────────────────
    h4_dir_info = calc_direction(h4_presets) if h4_presets else {}
    h4_dir      = h4_dir_info.get('direction', '횡보')
    h4_gx       = h4_dir_info.get('golden_cross', False)

    h4_dir_score = 0
    if h4_dir == '상승':
        h4_dir_score = 20
    elif h4_dir == '반등':
        h4_dir_score = 13
    elif h4_dir == '횡보':
        h4_dir_score = 4
    if h4_gx:
        h4_dir_score = min(20, h4_dir_score + 4)
    score += h4_dir_score
    breakdown['h4_direction'] = h4_dir_score

    # ── 1h 위치 (10점) ───────────────────────────────────────
    h1k = _safe_k(h1_presets, 'h1')
    if h1k is not None:
        if h1k <= 10:
            h1_pos = 10
        elif h1k <= 20:
            h1_pos = 7
        elif h1k <= 50:
            h1_pos = 3
        else:
            h1_pos = 0
    else:
        h1_pos = 0
    score += h1_pos
    breakdown['h1_position'] = h1_pos

    # ── 1h 방향 (10점) ───────────────────────────────────────
    h1_dir_info = calc_direction(h1_presets) if h1_presets else {}
    h1_dir      = h1_dir_info.get('direction', '횡보')
    h1_gx       = h1_dir_info.get('golden_cross', False)

    h1_dir_score = 0
    if h1_dir == '상승':
        h1_dir_score = 10
    elif h1_dir == '반등':
        h1_dir_score = 6
    elif h1_dir == '횡보':
        h1_dir_score = 2
    if h1_gx:
        h1_dir_score = min(10, h1_dir_score + 2)
    score += h1_dir_score
    breakdown['h1_direction'] = h1_dir_score

    # ── 거래량 보너스 (5점) ──────────────────────────────────
    vol = _safe_float(vol_ratio, 1.0)
    if vol >= 3.0:
        vol_bonus = 5
    elif vol >= 2.0:
        vol_bonus = 3
    elif vol >= 1.5:
        vol_bonus = 1
    else:
        vol_bonus = 0
    score += vol_bonus
    breakdown['volume_bonus'] = vol_bonus

    # ── 하락 보너스 (5점) ────────────────────────────────────
    drop_bonus = 0
    if snap_k is not None and dk is not None:
        drop = snap_k - dk  # 등록 시점 대비 K 하락폭
        if drop >= 20:
            drop_bonus = 5
        elif drop >= 10:
            drop_bonus = 3
        elif drop >= 5:
            drop_bonus = 1
    score += drop_bonus
    breakdown['drop_bonus'] = drop_bonus

    # ── 4hK 과열 패널티 ──────────────────────────────────────
    penalty = 0
    if h4k is not None:
        if h4k > H4_OVERHEAT_THRESHOLD:
            penalty = -10
        elif h4k > H4_WARM_THRESHOLD:
            penalty = -5
    score += penalty
    breakdown['h4_penalty'] = penalty

    # ── 최종 등급 ────────────────────────────────────────────
    score = max(0, min(100, score))
    if score >= GRADE_S_THRESHOLD:
        grade = 'S'
    elif score >= GRADE_A_THRESHOLD:
        grade = 'A'
    elif score >= GRADE_B_THRESHOLD:
        grade = 'B'
    else:
        grade = 'C'

    # ── 진입강도 ─────────────────────────────────────────────
    entry_level = calc_entry_strength(
        daily_dir_info, h4_dir_info, h1_dir_info,
        daily_k=dk, h4_k=h4k
    )

    return {
        'score':          score,
        'grade':          grade,
        'breakdown':      breakdown,
        'daily_dir':      daily_dir,
        'h4_dir':         h4_dir,
        'h1_dir':         h1_dir,
        'daily_dir_info': daily_dir_info,
        'h4_dir_info':    h4_dir_info,
        'h1_dir_info':    h1_dir_info,
        'daily_k':        dk,
        'h4_k':           h4k,
        'h1_k':           h1k,
        'entry_level':    entry_level,
        'entry_label':    entry_strength_label(entry_level),
    }


# ── DEEP Watch 점수 ──────────────────────────────────────────
def calc_deep_score(daily_k: float, btc_change: float, coin_change: float,
                    days_at_bottom: int = 0, vol_ratio: float = 1.0,
                    weekly_k: float | None = None) -> dict:
    """
    DEEP Watch 전용 점수 (최대 100점)
      - K값 위치       : 30점
      - 상대 강도      : 30점
      - 바닥 유지 기간 : 20점
      - 거래량 소진    : 10점
      - 주봉 K 보너스  : 10점
    """
    score     = 0
    breakdown = {}

    # K값 위치 (30점)
    if daily_k <= 0:
        k_score = 30
    elif daily_k <= 1:
        k_score = 28
    elif daily_k <= 2:
        k_score = 25
    elif daily_k <= 3:
        k_score = 20
    elif daily_k <= 5:
        k_score = 15
    else:
        k_score = 0
    score += k_score
    breakdown['k_position'] = k_score

    # 상대 강도 (30점)
    relative = coin_change - btc_change
    if relative >= 6:
        rel_score = 30
    elif relative >= DEEP_RELATIVE_MIN:
        rel_score = 20
    elif relative >= 1:
        rel_score = 10
    else:
        rel_score = 0
    score += rel_score
    breakdown['relative_strength'] = rel_score
    breakdown['relative_value']    = round(relative, 2)

    # 바닥 유지 기간 (20점)
    if days_at_bottom >= 5:
        bottom_score = 20
    elif days_at_bottom >= 3:
        bottom_score = 15
    elif days_at_bottom >= 2:
        bottom_score = 10
    elif days_at_bottom >= 1:
        bottom_score = 5
    else:
        bottom_score = 0
    score += bottom_score
    breakdown['days_at_bottom'] = bottom_score

    # 거래량 소진 (10점) - 낮을수록 좋음
    vol = _safe_float(vol_ratio, 1.0)
    if vol <= 0.3:
        vol_score = 10
    elif vol <= 0.5:
        vol_score = 7
    elif vol <= 0.8:
        vol_score = 3
    else:
        vol_score = 0
    score += vol_score
    breakdown['volume_exhaustion'] = vol_score

    # 주봉 K 보너스 (10점)
    wk = _safe_float(weekly_k)
    if wk is not None:
        if wk <= 10:
            wk_bonus = 10
        elif wk <= 20:
            wk_bonus = 7
        elif wk <= 30:
            wk_bonus = 3
        else:
            wk_bonus = 0
    else:
        wk_bonus = 0
    score += wk_bonus
    breakdown['weekly_k_bonus'] = wk_bonus

    score = max(0, min(100, score))

    return {
        'deep_score': score,
        'breakdown':  breakdown,
    }


def get_deep_grade(deep_score: int) -> str:
    if deep_score >= DEEP_GRADE_S:
        return 'DEEP-S'
    elif deep_score >= DEEP_GRADE_A:
        return 'DEEP-A'
    elif deep_score >= DEEP_GRADE_B:
        return 'DEEP-B'
    return 'DEEP-C'


def evaluate_deep_condition(daily_k: float, btc_change: float,
                            coin_change: float) -> bool:
    """
    DEEP Watch 기본 조건:
      1) 일봉 K ≤ DEEP_K_THRESHOLD (기본 5.0)
      2) BTC 24h 변동 ≤ DEEP_BTC_DROP_MIN (기본 -2.0%)
      3) 코인 변동 > BTC 변동 + DEEP_RELATIVE_MIN (기본 +3.0%)
    """
    if daily_k > DEEP_K_THRESHOLD:
        return False
    if btc_change > DEEP_BTC_DROP_MIN:
        return False
    if coin_change <= btc_change + DEEP_RELATIVE_MIN:
        return False
    return True


# ── 매크로 필터 ──────────────────────────────────────────────
def evaluate_macro_filter(btc_closes: list) -> dict:
    """
    BTC 주간 MA20 필터
    Returns: {'pass': bool, 'btc_weekly_ma20': float|None, 'btc_current': float|None}
    """
    if not MACRO_ENABLED or len(btc_closes) < MACRO_MA_PERIOD:
        return {'pass': True, 'btc_weekly_ma20': None, 'btc_current': None}

    ma_series = sma(btc_closes, MACRO_MA_PERIOD)
    ma20      = ma_series[-1]
    current   = btc_closes[-1]

    if ma20 is None:
        return {'pass': True, 'btc_weekly_ma20': None, 'btc_current': current}

    return {
        'pass':            current >= ma20 * 0.95,  # 5% 여유 허용
        'btc_weekly_ma20': round(ma20, 0),
        'btc_current':     round(current, 0),
    }


# ── 모듈 설정 정보 ───────────────────────────────────────────
def get_module_config() -> dict:
    return {
        'version':              VERSION,
        'oversold_threshold':   OVERSOLD_THRESHOLD,
        'recovery_k':           RECOVERY_K,
        'macro_enabled':        MACRO_ENABLED,
        'macro_ma_period':      MACRO_MA_PERIOD,
        'grade_thresholds':     {'S': GRADE_S_THRESHOLD, 'A': GRADE_A_THRESHOLD, 'B': GRADE_B_THRESHOLD},
        'watch_expiry_days':    {'default': WATCH_EXPIRY_DAYS, 'B': WATCH_EXPIRY_DAYS_B, 'C': WATCH_EXPIRY_DAYS_C},
        'h4_overheat':          {'overheat': H4_OVERHEAT_THRESHOLD, 'warm': H4_WARM_THRESHOLD},
        'deep_thresholds':      {'k': DEEP_K_THRESHOLD, 'btc_drop': DEEP_BTC_DROP_MIN, 'relative': DEEP_RELATIVE_MIN},
        'presets':              {'short': _SHORT_PARAMS, 'mid': _MID_PARAMS, 'long': _LONG_PARAMS},
    }
