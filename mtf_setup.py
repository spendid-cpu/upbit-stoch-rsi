# -*- coding: utf-8 -*-
"""
mtf_setup.py — Multi-Timeframe Stochastic RSI Setup Module (v3.9.20)
====================================================================
순수 분석 로직만 담은 모듈. 외부 의존성 없음(텔레그램·파일·API 호출 없음).

변경사항 (v3.9.20):
  - evaluate_macro_filter: 주봉 MA20이 실제 필터 기준 (safe 여부 결정)
  - 일봉 MA20은 참고 정보로만 기록 (safe에 영향 없음)
"""

import os
import math
from datetime import datetime, timedelta, timezone

# ======================== 환경변수 ========================
def _env_bool(key, default=False):
    v = os.environ.get(key, str(default)).strip().lower()
    return v in ('1', 'true', 'yes', 'on')

def _env_float(key, default):
    try:
        return float(os.environ.get(key, default))
    except Exception:
        return float(default)

def _env_int(key, default):
    try:
        return int(os.environ.get(key, default))
    except Exception:
        return int(default)

# ── Stoch RSI 프리셋 ───────────────────────────────
STOCH_SHORT = (
    _env_int('STOCH_SHORT_RSI', 5),
    _env_int('STOCH_SHORT_STOCH', 5),
    _env_int('STOCH_SHORT_K', 3),
    _env_int('STOCH_SHORT_D', 3),
)
STOCH_MID = (
    _env_int('STOCH_MID_RSI', 10),
    _env_int('STOCH_MID_STOCH', 10),
    _env_int('STOCH_MID_K', 6),
    _env_int('STOCH_MID_D', 6),
)
STOCH_LONG = (
    _env_int('STOCH_LONG_RSI', 20),
    _env_int('STOCH_LONG_STOCH', 20),
    _env_int('STOCH_LONG_K', 12),
    _env_int('STOCH_LONG_D', 12),
)

# ── 과매도 임계값 ──────────────────────────────────
MTF_DAILY_SHORT_OVERSOLD = _env_float('MTF_DAILY_SHORT_OVERSOLD', 20.0)
MTF_4H_SHORT_OVERSOLD    = _env_float('MTF_4H_SHORT_OVERSOLD', 20.0)
MTF_1H_SHORT_OVERSOLD    = _env_float('MTF_1H_SHORT_OVERSOLD', 20.0)

# ── 트리거 옵션 ────────────────────────────────────
MTF_USE_GC_TRIGGER = _env_bool('MTF_USE_GC_TRIGGER', True)

# ── 받쳐줌 판정 파라미터 ───────────────────────────
K_FALL_THRESHOLD        = _env_float('K_FALL_THRESHOLD', 0.5)
SUPPORTIVE_LOOKBACK_DAILY = _env_int('SUPPORTIVE_LOOKBACK_DAILY', 3)
SUPPORTIVE_LOOKBACK_4H    = _env_int('SUPPORTIVE_LOOKBACK_4H', 3)
SUPPORTIVE_LOOKBACK_1H    = _env_int('SUPPORTIVE_LOOKBACK_1H', 3)

# ── Watch List 관리 ────────────────────────────────
WATCH_LIST_EXPIRY_DAYS  = _env_int('WATCH_LIST_EXPIRY_DAYS', 7)
WATCH_LIST_RECOVERY_K   = _env_float('WATCH_LIST_RECOVERY_K', 50.0)

# ── 매크로 안전장치 ────────────────────────────────
USE_MACRO_FILTER          = _env_bool('USE_MACRO_FILTER', True)
MACRO_CHECK_BTC           = _env_bool('MACRO_CHECK_BTC', True)
MACRO_CHECK_ALT           = _env_bool('MACRO_CHECK_ALT', True)
MACRO_DAILY_MA            = _env_int('MACRO_DAILY_MA', 20)
MACRO_WEEKLY_MA           = _env_int('MACRO_WEEKLY_MA', 20)
MACRO_REQUIRE_CLOSE       = _env_bool('MACRO_REQUIRE_CLOSE', True)
MACRO_RECOVERY_WAIT_DAYS  = _env_int('MACRO_RECOVERY_WAIT_DAYS', 2)
MACRO_RECOVERY_WAIT_WEEKS = _env_int('MACRO_RECOVERY_WAIT_WEEKS', 2)
MACRO_HOLD_ACTION         = os.environ.get('MACRO_HOLD_ACTION', 'keep')

# ── 무효화 조건 ────────────────────────────────────
INVALIDATE_DAILY_MID_DROP       = _env_bool('INVALIDATE_DAILY_MID_DROP', True)
INVALIDATE_4H_OVERBOUGHT_DROP   = _env_bool('INVALIDATE_4H_OVERBOUGHT_DROP', True)
INVALIDATE_1H_OVERBOUGHT_DROP   = _env_bool('INVALIDATE_1H_OVERBOUGHT_DROP', True)
INVALIDATE_OVERBOUGHT_HIGH      = _env_float('INVALIDATE_OVERBOUGHT_HIGH', 80.0)
INVALIDATE_OVERBOUGHT_LOW       = _env_float('INVALIDATE_OVERBOUGHT_LOW', 70.0)

# ======================== 기본 지표 ========================
def calc_sma(values, period):
    if not values or len(values) < period:
        return None
    return sum(values[-period:]) / period

def calc_rsi(closes, period=14):
    if not closes or len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    if len(gains) < period:
        return []
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = []
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100.0 - 100.0 / (1.0 + rs))
    return rsis

def calc_stoch_rsi(closes, rsi_period, stoch_period, k_smooth, d_smooth):
    rsis = calc_rsi(closes, rsi_period)
    if not rsis or len(rsis) < stoch_period + k_smooth + d_smooth:
        return [], []
    raw_k = []
    for i in range(stoch_period - 1, len(rsis)):
        window = rsis[i - stoch_period + 1:i + 1]
        mn, mx = min(window), max(window)
        if mx == mn:
            raw_k.append(50.0)
        else:
            raw_k.append((rsis[i] - mn) / (mx - mn) * 100.0)
    if len(raw_k) < k_smooth:
        return [], []
    k_line = []
    for i in range(k_smooth - 1, len(raw_k)):
        k_line.append(sum(raw_k[i - k_smooth + 1:i + 1]) / k_smooth)
    if len(k_line) < d_smooth:
        return k_line, []
    d_line = []
    for i in range(d_smooth - 1, len(k_line)):
        d_line.append(sum(k_line[i - d_smooth + 1:i + 1]) / d_smooth)
    return k_line, d_line

# ======================== 프리셋 일괄 계산 ========================
def calc_stoch_preset(closes, preset):
    rsi_p, stoch_p, k_s, d_s = preset
    return calc_stoch_rsi(closes, rsi_p, stoch_p, k_s, d_s)

def calc_all_presets(closes):
    result = {}
    for name, preset in [('short', STOCH_SHORT), ('mid', STOCH_MID), ('long', STOCH_LONG)]:
        k_line, d_line = calc_stoch_preset(closes, preset)
        if not k_line:
            result[name] = {'k': None, 'd': None, 'k_line': [], 'd_line': []}
            continue
        result[name] = {
            'k'     : round(k_line[-1], 2),
            'd'     : round(d_line[-1], 2) if d_line else None,
            'k_prev': round(k_line[-2], 2) if len(k_line) >= 2 else None,
            'd_prev': round(d_line[-2], 2) if len(d_line) >= 2 else None,
            'k_line': k_line,
            'd_line': d_line,
        }
    return result

# ======================== 받쳐줌 판정 ========================
def evaluate_supportive(k_line, closes, lookback=3, oversold_zone=30.0):
    if not k_line or len(k_line) < 2 or not closes or len(closes) < lookback + 1:
        return {
            'supportive': False,
            'state'     : 'insufficient_data',
            'reason'    : '데이터 부족',
            'k_change'  : 0.0,
        }
    k_cur  = k_line[-1]
    k_prev = k_line[-2]
    k_change = k_cur - k_prev

    if k_change > K_FALL_THRESHOLD:
        return {'supportive': True,  'state': 'rising',
                'reason': f'K 상승 ({k_change:+.2f})', 'k_change': round(k_change, 2)}

    if abs(k_change) <= K_FALL_THRESHOLD:
        return {'supportive': True,  'state': 'sideways',
                'reason': f'K 횡보 ({k_change:+.2f})', 'k_change': round(k_change, 2)}

    price_now          = closes[-1]
    price_lookback_min = min(closes[-(lookback + 1):-1])  # 현재가 제외
    price_lookback_start = closes[-(lookback + 1)]
    price_change_pct   = (price_now - price_lookback_start) / price_lookback_start * 100 if price_lookback_start else 0

    is_new_low = price_now <= price_lookback_min

    if is_new_low:
        return {'supportive': False, 'state': 'falling',
                'reason': f'K 하락({k_change:+.2f}) + 가격 신저점({price_change_pct:+.2f}%) 동조',
                'k_change': round(k_change, 2)}

    if price_change_pct > 0:
        return {'supportive': True,  'state': 'hidden_bullish_divergence',
                'reason': f'K 하락({k_change:+.2f}) but 가격 상승({price_change_pct:+.2f}%) — 히든 다이버전스',
                'k_change': round(k_change, 2)}

    return {'supportive': True, 'state': 'pulling_back',
            'reason': f'K 정상 조정 ({k_change:+.2f}, 가격 {price_change_pct:+.2f}%)',
            'k_change': round(k_change, 2)}

# ======================== 매크로 안전장치 ========================
def evaluate_macro_filter(daily_closes, weekly_closes, ticker='?', last_break_info=None):
    """
    매크로 안전 상태 판정.

    ★ v3.9.20 변경:
      - 주봉 MA20 위 = safe (실제 필터 기준)
      - 일봉 MA20은 참고 정보만 기록 (safe 여부에 영향 없음)
    """
    result = {
        'safe'                 : True,
        'state'                : 'active',
        'reason'               : '',
        'ticker'               : ticker,
        # 일봉 (참고용)
        'daily_close'          : None,
        'daily_ma'             : None,
        'daily_distance_pct'   : None,
        # 주봉 (실제 필터)
        'weekly_close'         : None,
        'weekly_ma'            : None,
        'weekly_distance_pct'  : None,
    }

    # ── 일봉 MA20 (참고용 — safe 판정에 영향 없음) ──────────────
    if daily_closes and len(daily_closes) >= MACRO_DAILY_MA + 1:
        d_close = daily_closes[-1]
        d_ma    = calc_sma(daily_closes, MACRO_DAILY_MA)
        if d_ma:
            d_dist = (d_close - d_ma) / d_ma * 100
            result['daily_close']        = round(d_close, 6)
            result['daily_ma']           = round(d_ma, 6)
            result['daily_distance_pct'] = round(d_dist, 3)
            # 참고용 state 기록 (safe는 변경하지 않음)
            if d_close < d_ma:
                result['daily_ref_state'] = 'below_daily_ma'
            else:
                result['daily_ref_state'] = 'above_daily_ma'

    # ── 주봉 MA20 (실제 필터 기준) ─────────────────────────────
    if weekly_closes and len(weekly_closes) >= MACRO_WEEKLY_MA + 1:
        w_close = weekly_closes[-1]
        w_ma    = calc_sma(weekly_closes, MACRO_WEEKLY_MA)
        if w_ma:
            w_dist = (w_close - w_ma) / w_ma * 100
            result['weekly_close']        = round(w_close, 6)
            result['weekly_ma']           = round(w_ma, 6)
            result['weekly_distance_pct'] = round(w_dist, 3)
            if w_close < w_ma:
                result['safe']   = False
                result['state']  = 'weekly_break'
                result['reason'] = (f'{ticker} 주봉 MA{MACRO_WEEKLY_MA} 이탈 '
                                    f'({w_dist:+.2f}%)')
    else:
        # 주봉 데이터 부족 시 safe=False로 보수적 처리
        result['safe']   = False
        result['state']  = 'weekly_data_insufficient'
        result['reason'] = f'{ticker} 주봉 데이터 부족'

    # ── 복구 후 관망 기간 체크 ──────────────────────────────────
    if result['safe'] and last_break_info:
        w_recovered_at = last_break_info.get('weekly_recovered_at')
        if w_recovered_at:
            try:
                rec_dt = datetime.fromisoformat(w_recovered_at)
                if rec_dt.tzinfo is None:
                    rec_dt = rec_dt.replace(tzinfo=timezone.utc)
                now_utc    = datetime.now(timezone.utc)
                weeks_since = (now_utc - rec_dt).total_seconds() / 604800
                if weeks_since < MACRO_RECOVERY_WAIT_WEEKS:
                    result['safe']   = False
                    result['state']  = 'recovering'
                    result['reason'] = (f'{ticker} 주봉 복구 후 관망 중 '
                                        f'({weeks_since:.1f}/{MACRO_RECOVERY_WAIT_WEEKS}주)')
                    return result
            except Exception:
                pass

    if result['safe']:
        result['reason'] = (f'{ticker} 주봉 MA{MACRO_WEEKLY_MA} 위 '
                            f'({result.get("weekly_distance_pct", 0):+.2f}%)')

    return result

# ======================== Watch List 등록 조건 ========================
def evaluate_watch_list_entry(daily_closes, ticker='?'):
    result = {
        'should_register'      : False,
        'reason'               : '',
        'ticker'               : ticker,
        'daily_short_k'        : None,
        'daily_mid_supportive' : None,
        'daily_long_supportive': None,
        'details'              : {},
    }

    if not daily_closes or len(daily_closes) < 60:
        result['reason'] = '일봉 데이터 부족 (60개 미만)'
        return result

    presets = calc_all_presets(daily_closes)
    short   = presets.get('short', {})
    mid     = presets.get('mid', {})
    long_   = presets.get('long', {})

    result['details']['daily_short'] = {'k': short.get('k'), 'd': short.get('d')}
    result['details']['daily_mid']   = {'k': mid.get('k'),   'd': mid.get('d')}
    result['details']['daily_long']  = {'k': long_.get('k'), 'd': long_.get('d')}

    short_k = short.get('k')
    if short_k is None:
        result['reason'] = '일봉 단기 Stoch RSI 계산 불가'
        return result

    result['daily_short_k'] = short_k

    if short_k > MTF_DAILY_SHORT_OVERSOLD:
        result['reason'] = (f'일봉 단기 K {short_k:.2f} > {MTF_DAILY_SHORT_OVERSOLD} '
                            f'(과매도 아님)')
        return result

    mid_k_line  = mid.get('k_line', [])
    mid_support = evaluate_supportive(mid_k_line, daily_closes,
                                       lookback=SUPPORTIVE_LOOKBACK_DAILY)
    result['daily_mid_supportive'] = mid_support
    if not mid_support['supportive']:
        result['reason'] = f'일봉 중기 비받쳐줌: {mid_support["reason"]}'
        return result

    long_k_line  = long_.get('k_line', [])
    long_support = evaluate_supportive(long_k_line, daily_closes,
                                        lookback=SUPPORTIVE_LOOKBACK_DAILY)
    result['daily_long_supportive'] = long_support
    if not long_support['supportive']:
        result['reason'] = f'일봉 장기 비받쳐줌: {long_support["reason"]}'
        return result

    result['should_register'] = True
    result['reason'] = (f'일봉 단기 K {short_k:.2f} 과매도 + '
                        f'중기 {mid_support["state"]} + 장기 {long_support["state"]}')
    return result

# ======================== 진입 트리거 ========================
def evaluate_entry_trigger(h4_closes, h1_closes, ticker='?'):
    result = {
        'should_enter'     : False,
        'reason'           : '',
        'ticker'           : ticker,
        'h4_short_k'       : None,
        'h4_mid_supportive': None,
        'h1_short_k'       : None,
        'h1_trigger_type'  : None,
        'progress': {
            'h4_short_oversold': False,
            'h4_mid_supportive': False,
            'h1_short_oversold': False,
            'h1_trigger'       : False,
        },
        'details': {},
    }

    if not h4_closes or len(h4_closes) < 60:
        result['reason'] = '4시간 데이터 부족'
        return result
    if not h1_closes or len(h1_closes) < 60:
        result['reason'] = '1시간 데이터 부족'
        return result

    h4_presets = calc_all_presets(h4_closes)
    h1_presets = calc_all_presets(h1_closes)

    h4_short = h4_presets.get('short', {})
    h4_mid   = h4_presets.get('mid', {})
    h1_short = h1_presets.get('short', {})

    result['details']['h4_short'] = {'k': h4_short.get('k'), 'd': h4_short.get('d')}
    result['details']['h4_mid']   = {'k': h4_mid.get('k'),   'd': h4_mid.get('d')}
    result['details']['h1_short'] = {'k': h1_short.get('k'), 'd': h1_short.get('d')}

    h4_short_k = h4_short.get('k')
    if h4_short_k is None:
        result['reason'] = '4시간 단기 K 계산 불가'
        return result
    result['h4_short_k'] = h4_short_k
    if h4_short_k > MTF_4H_SHORT_OVERSOLD:
        result['reason'] = f'4시간 단기 K {h4_short_k:.2f} > {MTF_4H_SHORT_OVERSOLD}'
        return result
    result['progress']['h4_short_oversold'] = True

    h4_mid_line  = h4_mid.get('k_line', [])
    h4_mid_support = evaluate_supportive(h4_mid_line, h4_closes,
                                          lookback=SUPPORTIVE_LOOKBACK_4H)
    result['h4_mid_supportive'] = h4_mid_support
    if not h4_mid_support['supportive']:
        result['reason'] = f'4시간 중기 비받쳐줌: {h4_mid_support["reason"]}'
        return result
    result['progress']['h4_mid_supportive'] = True

    h1_short_k = h1_short.get('k')
    if h1_short_k is None:
        result['reason'] = '1시간 단기 K 계산 불가'
        return result
    result['h1_short_k'] = h1_short_k
    if h1_short_k > MTF_1H_SHORT_OVERSOLD:
        result['reason'] = f'1시간 단기 K {h1_short_k:.2f} > {MTF_1H_SHORT_OVERSOLD}'
        return result
    result['progress']['h1_short_oversold'] = True

    h1_k_line = h1_short.get('k_line', [])
    h1_d_line = h1_short.get('d_line', [])

    if len(h1_k_line) < 2:
        result['reason'] = '1시간 K 시계열 부족'
        return result

    k_now  = h1_k_line[-1]
    k_prev = h1_k_line[-2]

    if MTF_USE_GC_TRIGGER:
        if len(h1_d_line) < 2:
            result['reason'] = '1시간 D 시계열 부족'
            return result
        d_now  = h1_d_line[-1]
        d_prev = h1_d_line[-2]
        is_gc  = (k_prev <= d_prev) and (k_now > d_now)
        if not is_gc:
            result['reason'] = (f'1시간 GC 미발생 '
                                f'(K {k_prev:.2f}→{k_now:.2f}, D {d_prev:.2f}→{d_now:.2f})')
            return result
        result['h1_trigger_type'] = 'GC'
    else:
        is_rising = k_now > k_prev
        if not is_rising:
            result['reason'] = f'1시간 K 하락 ({k_prev:.2f} → {k_now:.2f})'
            return result
        result['h1_trigger_type'] = 'K_RISE'

    result['progress']['h1_trigger'] = True
    result['should_enter'] = True
    result['reason'] = (f'4h K {h4_short_k:.1f} + 1h K {h1_short_k:.1f} '
                        f'+ {result["h1_trigger_type"]}')
    return result

# ======================== Watch List 무효화 판정 ========================
def evaluate_watch_invalidation(daily_closes, watch_item, ticker='?'):
    result = {
        'should_remove': False,
        'reason'       : '',
        'removal_type' : None,
        'ticker'       : ticker,
    }

    registered_at = watch_item.get('registered_at')
    if registered_at:
        try:
            reg_dt = datetime.fromisoformat(registered_at)
            if reg_dt.tzinfo is None:
                reg_dt = reg_dt.replace(tzinfo=timezone.utc)
            now_utc      = datetime.now(timezone.utc)
            days_elapsed = (now_utc - reg_dt).total_seconds() / 86400
            if days_elapsed >= WATCH_LIST_EXPIRY_DAYS:
                result['should_remove'] = True
                result['removal_type']  = 'expired'
                result['reason']        = (f'등록 후 {days_elapsed:.1f}일 경과 '
                                           f'(만료: {WATCH_LIST_EXPIRY_DAYS}일)')
                return result
        except Exception:
            pass

    if not daily_closes or len(daily_closes) < 60:
        return result

    presets = calc_all_presets(daily_closes)
    short   = presets.get('short', {})
    mid     = presets.get('mid', {})
    long_   = presets.get('long', {})

    short_k = short.get('k')
    if short_k is not None and short_k >= WATCH_LIST_RECOVERY_K:
        result['should_remove'] = True
        result['removal_type']  = 'recovered'
        result['reason']        = f'일봉 단기 K {short_k:.2f} ≥ {WATCH_LIST_RECOVERY_K} 회복'
        return result

    mid_line   = mid.get('k_line', [])
    mid_support = evaluate_supportive(mid_line, daily_closes,
                                       lookback=SUPPORTIVE_LOOKBACK_DAILY)
    if not mid_support['supportive']:
        result['should_remove'] = True
        result['removal_type']  = 'support_broken'
        result['reason']        = f'일봉 중기 비받쳐줌: {mid_support["reason"]}'
        return result

    long_line   = long_.get('k_line', [])
    long_support = evaluate_supportive(long_line, daily_closes,
                                        lookback=SUPPORTIVE_LOOKBACK_DAILY)
    if not long_support['supportive']:
        result['should_remove'] = True
        result['removal_type']  = 'support_broken'
        result['reason']        = f'일봉 장기 비받쳐줌: {long_support["reason"]}'
        return result

    return result

# ======================== 보유 포지션 무효화 판정 ========================
def evaluate_signal_invalidation_for_holding(daily_closes, h4_closes, h1_closes,
                                              pool_item, ticker='?'):
    result = {
        'should_exit': False,
        'reason'     : '',
        'exit_type'  : None,
        'ticker'     : ticker,
    }

    if INVALIDATE_DAILY_MID_DROP and daily_closes and len(daily_closes) >= 60:
        d_presets  = calc_all_presets(daily_closes)
        d_mid      = d_presets.get('mid', {})
        d_mid_line = d_mid.get('k_line', [])
        if d_mid_line:
            d_mid_support = evaluate_supportive(d_mid_line, daily_closes,
                                                 lookback=SUPPORTIVE_LOOKBACK_DAILY)
            if not d_mid_support['supportive']:
                result['should_exit'] = True
                result['exit_type']   = 'daily_mid_broken'
                result['reason']      = f'일봉 중기 받쳐줌 깨짐: {d_mid_support["reason"]}'
                return result

    if INVALIDATE_4H_OVERBOUGHT_DROP and h4_closes and len(h4_closes) >= 60:
        h4_presets  = calc_all_presets(h4_closes)
        h4_mid_line = h4_presets.get('mid', {}).get('k_line', [])
        if len(h4_mid_line) >= 2:
            k_now   = h4_mid_line[-1]
            recent  = h4_mid_line[-10:] if len(h4_mid_line) >= 10 else h4_mid_line
            reached_high = any(k >= INVALIDATE_OVERBOUGHT_HIGH for k in recent[:-1])
            if reached_high and k_now <= INVALIDATE_OVERBOUGHT_LOW:
                result['should_exit'] = True
                result['exit_type']   = 'h4_mid_overbought_drop'
                result['reason']      = (f'4시간 중기 K 과매수({INVALIDATE_OVERBOUGHT_HIGH}+) 후 '
                                         f'{INVALIDATE_OVERBOUGHT_LOW} 이하 하락 (현재 {k_now:.2f})')
                return result

    if INVALIDATE_1H_OVERBOUGHT_DROP and h1_closes and len(h1_closes) >= 60:
        h1_presets   = calc_all_presets(h1_closes)
        h1_short_line = h1_presets.get('short', {}).get('k_line', [])
        if len(h1_short_line) >= 2:
            k_now   = h1_short_line[-1]
            recent  = h1_short_line[-10:] if len(h1_short_line) >= 10 else h1_short_line
            reached_high = any(k >= INVALIDATE_OVERBOUGHT_HIGH for k in recent[:-1])
            if reached_high and k_now <= INVALIDATE_OVERBOUGHT_LOW:
                result['should_exit'] = True
                result['exit_type']   = 'h1_overbought_drop'
                result['reason']      = (f'1시간 단기 K 과매수({INVALIDATE_OVERBOUGHT_HIGH}+) 후 '
                                         f'{INVALIDATE_OVERBOUGHT_LOW} 이하 하락 (현재 {k_now:.2f})')
                return result

    return result

# ======================== 통합 분석 ========================
def full_analysis(daily_closes, h4_closes, h1_closes, ticker='?'):
    out = {
        'ticker'        : ticker,
        'daily'         : None,
        'h4'            : None,
        'h1'            : None,
        'watch_eligible': None,
        'entry_trigger' : None,
    }
    if daily_closes and len(daily_closes) >= 60:
        out['daily']          = calc_all_presets(daily_closes)
        out['watch_eligible'] = evaluate_watch_list_entry(daily_closes, ticker)
    if h4_closes and len(h4_closes) >= 60:
        out['h4'] = calc_all_presets(h4_closes)
    if h1_closes and len(h1_closes) >= 60:
        out['h1'] = calc_all_presets(h1_closes)
    if h4_closes and h1_closes:
        out['entry_trigger'] = evaluate_entry_trigger(h4_closes, h1_closes, ticker)
    return out

# ======================== 모듈 정보 ========================
MODULE_VERSION = '3.9.20'
MODULE_NAME    = 'mtf_setup'

def get_module_config():
    return {
        'module_version'           : MODULE_VERSION,
        'stoch_short'              : STOCH_SHORT,
        'stoch_mid'                : STOCH_MID,
        'stoch_long'               : STOCH_LONG,
        'mtf_daily_short_oversold' : MTF_DAILY_SHORT_OVERSOLD,
        'mtf_4h_short_oversold'    : MTF_4H_SHORT_OVERSOLD,
        'mtf_1h_short_oversold'    : MTF_1H_SHORT_OVERSOLD,
        'mtf_use_gc_trigger'       : MTF_USE_GC_TRIGGER,
        'k_fall_threshold'         : K_FALL_THRESHOLD,
        'watch_list_expiry_days'   : WATCH_LIST_EXPIRY_DAYS,
        'watch_list_recovery_k'    : WATCH_LIST_RECOVERY_K,
        'use_macro_filter'         : USE_MACRO_FILTER,
        'macro_check_btc'          : MACRO_CHECK_BTC,
        'macro_check_alt'          : MACRO_CHECK_ALT,
        'macro_daily_ma'           : MACRO_DAILY_MA,
        'macro_weekly_ma'          : MACRO_WEEKLY_MA,
        'macro_recovery_wait_days' : MACRO_RECOVERY_WAIT_DAYS,
        'macro_recovery_wait_weeks': MACRO_RECOVERY_WAIT_WEEKS,
        'macro_hold_action'        : MACRO_HOLD_ACTION,
        'macro_filter_basis'       : 'weekly_ma20',  # 필터 기준 명시
    }

# ======================== 단독 테스트 ========================
if __name__ == '__main__':
    print(f'=== {MODULE_NAME} v{MODULE_VERSION} self-test ===')
    import random
    random.seed(42)
    mock_daily = []
    price = 100.0
    for i in range(100):
        if i < 60:
            price *= (1 + random.uniform(-0.02, 0.005))
        else:
            price *= (1 + random.uniform(-0.01, 0.02))
        mock_daily.append(price)

    mock_h4, mock_h1 = [], []
    p = mock_daily[-1]
    for i in range(100):
        p *= (1 + random.uniform(-0.01, 0.012))
        mock_h4.append(p)
    p = mock_h4[-1]
    for i in range(100):
        p *= (1 + random.uniform(-0.005, 0.006))
        mock_h1.append(p)

    res = full_analysis(mock_daily, mock_h4, mock_h1, ticker='MOCK')
    print(f"Daily short K : {res['daily']['short']['k']}")
    print(f"Watch eligible: {res['watch_eligible']['should_register']} — {res['watch_eligible']['reason']}")
    print(f"Entry trigger : {res['entry_trigger']['should_enter']} — {res['entry_trigger']['reason']}")

    macro = evaluate_macro_filter(mock_daily, mock_daily[::5], ticker='BTC-MOCK')
    print(f"\nMacro safe   : {macro['safe']}, state: {macro['state']}")
    print(f"Weekly dist  : {macro.get('weekly_distance_pct')}%  (실제 필터)")
    print(f"Daily dist   : {macro.get('daily_distance_pct')}%   (참고용)")
    print('\n✓ Self-test 완료')
