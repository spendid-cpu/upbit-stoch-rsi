import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import pyupbit
import streamlit as st


# ==================================================
# Streamlit 기본 설정
# ==================================================

st.set_page_config(
    page_title="Stoch RSI 다중조건 코인 추천기",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# Stochastic RSI 설정
# ==================================================

STOCH_RSI_SETTINGS = {
    "short": {
        "name": "단기",
        "rsi_period": 5,
        "stoch_period": 5,
        "k_smooth": 3,
        "d_smooth": 3,
        "oversold": 20,
        "overbought": 80,
        "weight": 1.0
    },
    "middle": {
        "name": "중기",
        "rsi_period": 10,
        "stoch_period": 10,
        "k_smooth": 6,
        "d_smooth": 6,
        "oversold": 20,
        "overbought": 80,
        "weight": 1.5
    },
    "long": {
        "name": "장기",
        "rsi_period": 20,
        "stoch_period": 20,
        "k_smooth": 12,
        "d_smooth": 12,
        "oversold": 20,
        "overbought": 80,
        "weight": 2.0
    }
}


TIMEFRAME_SETTINGS = {
    "day": {
        "name": "일봉",
        "interval": "day",
        "weight": 45
    },
    "minute240": {
        "name": "4시간봉",
        "interval": "minute240",
        "weight": 35
    },
    "minute60": {
        "name": "1시간봉",
        "interval": "minute60",
        "weight": 20
    }
}


TIMEFRAME_ALL_OVERSOLD_BONUS = {
    "day": 50,
    "minute240": 35,
    "minute60": 20
}

ALL_NINE_OVERSOLD_BONUS = 45


# ==================================================
# 기본 유틸 함수
# ==================================================

def format_price(price):
    if price is None or pd.isna(price):
        return "-"
    try:
        price = float(price)
        if price >= 1000:
            return f"{price:,.0f}"
        elif price >= 1:
            return f"{price:,.2f}"
        else:
            return f"{price:.6f}"
    except Exception:
        return "-"


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def grade_rank(grade):
    rank = {
        "S+": 1,
        "S": 2,
        "A": 3,
        "B": 4,
        "C": 5,
        "대기": 6
    }
    return rank.get(grade, 99)


def get_grade_by_score(score):
    if score >= 540:
        return "S+"
    elif score >= 460:
        return "S"
    elif score >= 360:
        return "A"
    elif score >= 260:
        return "B"
    elif score >= 160:
        return "C"
    else:
        return None


# ==================================================
# Upbit API 함수
# ==================================================

@st.cache_data(ttl=300)
def get_krw_markets():
    """
    업비트 KRW 마켓 목록 조회
    유의종목은 제외
    """
    try:
        url = "https://api.upbit.com/v1/market/all"
        params = {
            "isDetails": "true"
        }
        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        markets = []
        for item in data:
            market = item.get("market", "")
            warning = item.get("market_warning", "NONE")

            if market.startswith("KRW-") and warning == "NONE":
                markets.append(market)

        return markets

    except Exception:
        try:
            return pyupbit.get_tickers(fiat="KRW")
        except Exception:
            return []


@st.cache_data(ttl=180)
def get_top_trade_value_markets(limit=60):
    """
    24시간 거래대금 상위 코인 조회
    추천 조건이 아니라 스캔 대상 선정용
    """
    markets = get_krw_markets()
    markets = [m for m in markets if m != "KRW-BTC"]

    if not markets:
        return []

    tickers = []

    for i in range(0, len(markets), 100):
        batch = markets[i:i + 100]
        try:
            url = "https://api.upbit.com/v1/ticker"
            params = {
                "markets": ",".join(batch)
            }
            res = requests.get(url, params=params, timeout=10)
            data = res.json()

            if isinstance(data, list):
                tickers.extend(data)

            time.sleep(0.08)

        except Exception:
            time.sleep(0.2)

    if not tickers:
        return markets[:limit]

    tickers = sorted(
        tickers,
        key=lambda x: x.get("acc_trade_price_24h", 0),
        reverse=True
    )

    return [x["market"] for x in tickers[:limit]]


@st.cache_data(ttl=300)
def get_ohlcv_cached(ticker, interval, count):
    """
    OHLCV 캐시 조회
    """
    try:
        df = pyupbit.get_ohlcv(
            ticker=ticker,
            interval=interval,
            count=count
        )
        return df
    except Exception:
        return None


@st.cache_data(ttl=120)
def get_current_prices_batch(tickers):
    """
    현재가 배치 조회
    """
    if not tickers:
        return {}

    result = {}

    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            url = "https://api.upbit.com/v1/ticker"
            params = {
                "markets": ",".join(batch)
            }
            res = requests.get(url, params=params, timeout=10)
            data = res.json()

            if isinstance(data, list):
                for item in data:
                    result[item["market"]] = item.get("trade_price")

            time.sleep(0.05)

        except Exception:
            time.sleep(0.2)

    return result


# ==================================================
# RSI / Stochastic RSI 계산
# ==================================================

def calculate_rsi(close, period=14):
    """
    Wilder 방식 RSI 계산
    TradingView 계열 RSI와 유사한 RMA 방식
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_stoch_rsi(
    df,
    rsi_period=5,
    stoch_period=5,
    k_smooth=3,
    d_smooth=3
):
    """
    Stochastic RSI 계산
    """
    if df is None or len(df) == 0:
        return None

    df = df.copy()

    rsi = calculate_rsi(df["close"], period=rsi_period)

    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()

    denominator = (rsi_max - rsi_min).replace(0, np.nan)

    stoch_rsi_raw = 100 * ((rsi - rsi_min) / denominator)

    df["rsi"] = rsi
    df["stoch_rsi_raw"] = stoch_rsi_raw
    df["stoch_rsi_k"] = stoch_rsi_raw.rolling(k_smooth).mean()
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(d_smooth).mean()

    return df


def calculate_stoch_rsi_by_setting(df, setting):
    return calculate_stoch_rsi(
        df=df,
        rsi_period=setting["rsi_period"],
        stoch_period=setting["stoch_period"],
        k_smooth=setting["k_smooth"],
        d_smooth=setting["d_smooth"]
    )


# ==================================================
# BTC MA20 필터
# ==================================================

def get_btc_ma20_status(warning_gap=0.3):
    """
    BTC 일봉 MA20 상태 확인
    """
    try:
        df = get_ohlcv_cached("KRW-BTC", "day", 80)
        current_price = pyupbit.get_current_price("KRW-BTC")

        if df is None or len(df) < 30 or current_price is None:
            return {
                "ok": False,
                "status": "BTC 데이터 부족",
                "current_price": current_price,
                "live_ma20": None,
                "gap_rate": None,
                "reason": "BTC 데이터 조회 실패"
            }

        closes = df["close"].copy()

        prev_close = closes.iloc[-2]
        prev_ma20 = closes.iloc[-21:-1].mean()

        live_ma20 = pd.concat([
            closes.iloc[-20:-1],
            pd.Series([current_price])
        ]).mean()

        confirmed_above = prev_close > prev_ma20
        live_above = current_price > live_ma20

        gap_rate = ((current_price - live_ma20) / live_ma20) * 100

        if confirmed_above and live_above:
            if gap_rate >= warning_gap:
                status = "MA20 위 유지"
                ok = True
            else:
                status = "MA20 근접 주의"
                ok = True

        elif confirmed_above and not live_above:
            status = "MA20 이탈중, 미확정"
            ok = False

        elif not confirmed_above and live_above:
            status = "MA20 재돌파 시도중"
            ok = False

        else:
            status = "MA20 아래"
            ok = False

        return {
            "ok": ok,
            "status": status,
            "current_price": float(current_price),
            "live_ma20": float(live_ma20),
            "gap_rate": float(gap_rate),
            "prev_close": float(prev_close),
            "prev_ma20": float(prev_ma20),
            "reason": status
        }

    except Exception as e:
        return {
            "ok": False,
            "status": "BTC 필터 오류",
            "current_price": None,
            "live_ma20": None,
            "gap_rate": None,
            "reason": str(e)
        }


# ==================================================
# Stoch RSI 상태 판단
# ==================================================

def judge_stochrsi_state(df, setting, oversold_mode="both"):
    """
    특정 시간봉 + 특정 세팅의 현재 Stoch RSI 상태 판단

    oversold_mode:
    - both: K와 D 모두 20 이하일 때 과매도
    - either: K 또는 D 하나만 20 이하이어도 과매도
    """
    calc = calculate_stoch_rsi_by_setting(df, setting)

    if calc is None:
        return {
            "ok": False,
            "oversold": False,
            "overbought": False,
            "k": None,
            "d": None,
            "rsi": None,
            "reason": "데이터 없음"
        }

    calc = calc.dropna().copy()

    if len(calc) == 0:
        return {
            "ok": False,
            "oversold": False,
            "overbought": False,
            "k": None,
            "d": None,
            "rsi": None,
            "reason": "Stoch RSI 계산 데이터 부족"
        }

    last = calc.iloc[-1]

    k = safe_float(last["stoch_rsi_k"])
    d = safe_float(last["stoch_rsi_d"])
    rsi = safe_float(last["rsi"])

    if k is None or d is None:
        return {
            "ok": False,
            "oversold": False,
            "overbought": False,
            "k": k,
            "d": d,
            "rsi": rsi,
            "reason": "K/D 계산 불가"
        }

    oversold_value = setting["oversold"]
    overbought_value = setting["overbought"]

    if oversold_mode == "either":
        is_oversold = k <= oversold_value or d <= oversold_value
    else:
        is_oversold = k <= oversold_value and d <= oversold_value

    is_overbought = k >= overbought_value or d >= overbought_value

    if is_oversold:
        state_text = "과매도"
    elif is_overbought:
        state_text = "과매수"
    else:
        state_text = "중립"

    return {
        "ok": True,
        "oversold": bool(is_oversold),
        "overbought": bool(is_overbought),
        "k": k,
        "d": d,
        "rsi": rsi,
        "reason": f"{state_text} / K {k:.2f} / D {d:.2f} / RSI {rsi:.2f}"
    }


def analyze_timeframe_all_settings(df, oversold_mode="both"):
    """
    하나의 시간봉에 대해 단기/중기/장기 Stoch RSI 분석
    """
    result = {}

    for setting_key, setting in STOCH_RSI_SETTINGS.items():
        result[setting_key] = judge_stochrsi_state(
            df=df,
            setting=setting,
            oversold_mode=oversold_mode
        )

    return result


# ==================================================
# 점수 계산
# ==================================================

def calculate_multi_stochrsi_score(analysis):
    """
    일봉 / 4시간봉 / 1시간봉
    단기 / 중기 / 장기

    총 9개 조건을 점수화
    """
    score = 0
    oversold_count = 0
    total_count = 0

    timeframe_summary = {}

    for tf_key, tf_info in TIMEFRAME_SETTINGS.items():
        tf_weight = tf_info["weight"]

        tf_oversold_count = 0
        tf_total_count = 0

        for setting_key, setting in STOCH_RSI_SETTINGS.items():
            total_count += 1
            tf_total_count += 1

            state = analysis[tf_key][setting_key]

            if state["oversold"]:
                point = tf_weight * setting["weight"]
                score += point

                oversold_count += 1
                tf_oversold_count += 1

        # 해당 시간봉에서 단기/중기/장기 모두 과매도이면 보너스
        if tf_oversold_count == tf_total_count:
            score += TIMEFRAME_ALL_OVERSOLD_BONUS.get(tf_key, 0)

        timeframe_summary[tf_key] = {
            "oversold_count": tf_oversold_count,
            "total_count": tf_total_count
        }

    # 전체 9개 모두 과매도이면 최고 보너스
    if oversold_count == total_count:
        score += ALL_NINE_OVERSOLD_BONUS

    return {
        "score": round(score, 1),
        "oversold_count": oversold_count,
        "total_count": total_count,
        "timeframe_summary": timeframe_summary
    }


def is_recommendable_by_multi_stochrsi(analysis, mode="balanced"):
    """
    추천 최소 통과 조건

    balanced:
    - 일봉에서 단/중/장 중 최소 1개 과매도
    - 4시간봉 또는 1시간봉에서 최소 1개 과매도

    strict:
    - 일봉 최소 1개
    - 4시간봉 최소 1개
    - 1시간봉 최소 1개

    aggressive:
    - 전체 9개 중 최소 2개 이상 과매도
    """
    day_any = any(
        analysis["day"][key]["oversold"]
        for key in STOCH_RSI_SETTINGS.keys()
    )

    h4_any = any(
        analysis["minute240"][key]["oversold"]
        for key in STOCH_RSI_SETTINGS.keys()
    )

    h1_any = any(
        analysis["minute60"][key]["oversold"]
        for key in STOCH_RSI_SETTINGS.keys()
    )

    total_oversold = 0
    for tf_key in TIMEFRAME_SETTINGS.keys():
        for setting_key in STOCH_RSI_SETTINGS.keys():
            if analysis[tf_key][setting_key]["oversold"]:
                total_oversold += 1

    if mode == "strict":
        return day_any and h4_any and h1_any

    if mode == "aggressive":
        return total_oversold >= 2

    return day_any and (h4_any or h1_any)


# ==================================================
# 코인 분석
# ==================================================

def analyze_coin_multi_stochrsi(
    ticker,
    oversold_mode="both",
    recommend_mode="balanced",
    include_waiting=False
):
    """
    코인 1개 분석
    """
    try:
        day_df = get_ohlcv_cached(ticker, "day", 180)
        h4_df = get_ohlcv_cached(ticker, "minute240", 180)
        h1_df = get_ohlcv_cached(ticker, "minute60", 180)

        if day_df is None or h4_df is None or h1_df is None:
            return None

        if len(day_df) < 80 or len(h4_df) < 80 or len(h1_df) < 80:
            return None

        analysis = {
            "day": analyze_timeframe_all_settings(
                day_df,
                oversold_mode=oversold_mode
            ),
            "minute240": analyze_timeframe_all_settings(
                h4_df,
                oversold_mode=oversold_mode
            ),
            "minute60": analyze_timeframe_all_settings(
                h1_df,
                oversold_mode=oversold_mode
            )
        }

        score_info = calculate_multi_stochrsi_score(analysis)
        score = score_info["score"]
        grade = get_grade_by_score(score)

        recommendable = is_recommendable_by_multi_stochrsi(
            analysis,
            mode=recommend_mode
        )

        if not recommendable and not include_waiting:
            return None

        if grade is None and not include_waiting:
            return None

        if grade is None and include_waiting:
            grade = "대기"

        day_count = score_info["timeframe_summary"]["day"]["oversold_count"]
        h4_count = score_info["timeframe_summary"]["minute240"]["oversold_count"]
        h1_count = score_info["timeframe_summary"]["minute60"]["oversold_count"]

        reason = (
            f"일봉 {day_count}/3, "
            f"4H {h4_count}/3, "
            f"1H {h1_count}/3"
        )

        return {
            "등급": grade,
            "코인": ticker,
            "점수": score,
            "추천사유": reason,
            "전체과매도개수": score_info["oversold_count"],
            "전체조건개수": score_info["total_count"],

            "일봉과매도개수": day_count,
            "4H과매도개수": h4_count,
            "1H과매도개수": h1_count,

            "일봉단기": "YES" if analysis["day"]["short"]["oversold"] else "NO",
            "일봉중기": "YES" if analysis["day"]["middle"]["oversold"] else "NO",
            "일봉장기": "YES" if analysis["day"]["long"]["oversold"] else "NO",

            "4H단기": "YES" if analysis["minute240"]["short"]["oversold"] else "NO",
            "4H중기": "YES" if analysis["minute240"]["middle"]["oversold"] else "NO",
            "4H장기": "YES" if analysis["minute240"]["long"]["oversold"] else "NO",

            "1H단기": "YES" if analysis["minute60"]["short"]["oversold"] else "NO",
            "1H중기": "YES" if analysis["minute60"]["middle"]["oversold"] else "NO",
            "1H장기": "YES" if analysis["minute60"]["long"]["oversold"] else "NO",

            "일봉단기K": analysis["day"]["short"]["k"],
            "일봉단기D": analysis["day"]["short"]["d"],
            "일봉단기RSI": analysis["day"]["short"]["rsi"],

            "일봉중기K": analysis["day"]["middle"]["k"],
            "일봉중기D": analysis["day"]["middle"]["d"],
            "일봉중기RSI": analysis["day"]["middle"]["rsi"],

            "일봉장기K": analysis["day"]["long"]["k"],
            "일봉장기D": analysis["day"]["long"]["d"],
            "일봉장기RSI": analysis["day"]["long"]["rsi"],

            "4H단기K": analysis["minute240"]["short"]["k"],
            "4H단기D": analysis["minute240"]["short"]["d"],
            "4H단기RSI": analysis["minute240"]["short"]["rsi"],

            "4H중기K": analysis["minute240"]["middle"]["k"],
            "4H중기D": analysis["minute240"]["middle"]["d"],
            "4H중기RSI": analysis["minute240"]["middle"]["rsi"],

            "4H장기K": analysis["minute240"]["long"]["k"],
            "4H장기D": analysis["minute240"]["long"]["d"],
            "4H장기RSI": analysis["minute240"]["long"]["rsi"],

            "1H단기K": analysis["minute60"]["short"]["k"],
            "1H단기D": analysis["minute60"]["short"]["d"],
            "1H단기RSI": analysis["minute60"]["short"]["rsi"],

            "1H중기K": analysis["minute60"]["middle"]["k"],
            "1H중기D": analysis["minute60"]["middle"]["d"],
            "1H중기RSI": analysis["minute60"]["middle"]["rsi"],

            "1H장기K": analysis["minute60"]["long"]["k"],
            "1H장기D": analysis["minute60"]["long"]["d"],
            "1H장기RSI": analysis["minute60"]["long"]["rsi"],
        }

    except Exception as e:
        print(f"{ticker} 분석 오류:", e)
        return None


# ==================================================
# 화면 UI
# ==================================================

st.title("📊 Upbit Stoch RSI 다중조건 코인 추천기")

st.caption(
    "단기/중기/장기 Stochastic RSI를 일봉·4시간봉·1시간봉에서 분석하여 점수화합니다."
)

with st.expander("추천 로직 설명", expanded=False):
    st.markdown("""
## 지표 기준

이 앱은 **Stochastic RSI** 기준입니다.

### 단기
- %K: 3
- %D: 3
- Stochastic 기간: 5
- RSI 기간: 5
- 과매수: 80
- 과매도: 20

### 중기
- %K: 6
- %D: 6
- Stochastic 기간: 10
- RSI 기간: 10
- 과매수: 80
- 과매도: 20

### 장기
- %K: 12
- %D: 12
- Stochastic 기간: 20
- RSI 기간: 20
- 과매수: 80
- 과매도: 20

## 최고점 조건

아래 9개 조건이 모두 과매도이면 최고점입니다.

- 일봉 단기 / 중기 / 장기 과매도
- 4시간봉 단기 / 중기 / 장기 과매도
- 1시간봉 단기 / 중기 / 장기 과매도

## 점수 가중치

시간봉 가중치:

- 일봉: 45
- 4시간봉: 35
- 1시간봉: 20

세팅 가중치:

- 단기: 1.0
- 중기: 1.5
- 장기: 2.0

보너스:

- 일봉 3개 모두 과매도: +50
- 4시간봉 3개 모두 과매도: +35
- 1시간봉 3개 모두 과매도: +20
- 전체 9개 모두 과매도: +45
""")


# ==================================================
# 사이드바 설정
# ==================================================

st.sidebar.header("설정")

scan_mode = st.sidebar.selectbox(
    "스캔 대상",
    ["거래대금 상위", "전체 KRW", "수동 관심코인"],
    index=0
)

top_count = st.sidebar.number_input(
    "거래대금 상위 N개",
    min_value=10,
    max_value=150,
    value=60,
    step=10
)

max_scan_count = st.sidebar.number_input(
    "최대 스캔 개수",
    min_value=0,
    max_value=200,
    value=60,
    step=10,
    help="0이면 제한 없음"
)

manual_text = st.sidebar.text_area(
    "수동 관심코인",
    value="KRW-ETH\nKRW-XRP\nKRW-SOL\nKRW-DOGE\nKRW-ADA\nKRW-AVAX\nKRW-LINK\nKRW-DOT\nKRW-SUI\nKRW-APT",
    height=180,
    help="수동 관심코인 모드에서 사용. 한 줄에 하나씩 입력"
)

st.sidebar.divider()

recommend_mode_text = st.sidebar.selectbox(
    "추천 최소 조건",
    ["균형형", "보수형", "공격형"],
    index=0,
    help="균형형: 일봉 최소 1개 + 4H/1H 중 최소 1개"
)

if recommend_mode_text == "보수형":
    recommend_mode = "strict"
elif recommend_mode_text == "공격형":
    recommend_mode = "aggressive"
else:
    recommend_mode = "balanced"

oversold_mode_text = st.sidebar.selectbox(
    "과매도 판정",
    ["K와 D 모두 20 이하", "K 또는 D 하나만 20 이하"],
    index=0
)

oversold_mode = "both" if oversold_mode_text == "K와 D 모두 20 이하" else "either"

btc_warning_gap = st.sidebar.number_input(
    "BTC MA20 근접 주의 이격률 %",
    min_value=0.0,
    max_value=5.0,
    value=0.3,
    step=0.1
)

ignore_btc_filter = st.sidebar.checkbox(
    "테스트용: BTC MA20 필터 무시",
    value=False
)

include_waiting = st.sidebar.checkbox(
    "대기 코인도 표시",
    value=False,
    help="추천 최소 조건을 만족하지 않아도 점수 확인용으로 표시"
)

request_delay = st.sidebar.number_input(
    "요청 간격 초",
    min_value=0.02,
    max_value=1.0,
    value=0.08,
    step=0.02
)


# ==================================================
# BTC 상태 표시
# ==================================================

btc = get_btc_ma20_status(warning_gap=btc_warning_gap)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("BTC 상태", btc["status"])

with col2:
    st.metric("BTC 현재가", format_price(btc["current_price"]))

with col3:
    st.metric("BTC 실시간 MA20", format_price(btc["live_ma20"]))

with col4:
    gap = btc["gap_rate"]
    st.metric("MA20 이격률", "-" if gap is None else f"{gap:.2f}%")

if btc["ok"]:
    st.success("BTC MA20 필터 통과: 신규 추천 허용 상태입니다.")
else:
    st.warning(f"BTC MA20 필터 미통과: {btc['reason']}")


# ==================================================
# 스캔 대상 만들기
# ==================================================

def build_scan_list():
    all_krw = get_krw_markets()
    all_krw = [m for m in all_krw if m != "KRW-BTC"]

    if scan_mode == "전체 KRW":
        tickers = all_krw

    elif scan_mode == "수동 관심코인":
        manual = [
            x.strip().upper()
            for x in manual_text.splitlines()
            if x.strip()
        ]
        tickers = [x for x in manual if x in all_krw and x != "KRW-BTC"]

    else:
        tickers = get_top_trade_value_markets(limit=top_count)
        tickers = [x for x in tickers if x in all_krw and x != "KRW-BTC"]

    tickers = list(dict.fromkeys(tickers))

    if max_scan_count and max_scan_count > 0:
        tickers = tickers[:max_scan_count]

    return tickers


# ==================================================
# 추천 실행
# ==================================================

st.divider()

run = st.button("🚀 추천받기", type="primary", use_container_width=True)

if run:
    if not btc["ok"] and not ignore_btc_filter:
        st.error("BTC가 MA20 조건을 충족하지 않아 추천을 중단합니다.")
        st.stop()

    tickers = build_scan_list()

    if not tickers:
        st.error("스캔할 코인이 없습니다. 설정을 확인해주세요.")
        st.stop()

    st.info(f"총 {len(tickers)}개 코인을 분석합니다.")

    progress = st.progress(0)
    status_text = st.empty()

    results = []

    for idx, ticker in enumerate(tickers):
        status_text.write(f"분석 중: {ticker} ({idx + 1}/{len(tickers)})")

        item = analyze_coin_multi_stochrsi(
            ticker=ticker,
            oversold_mode=oversold_mode,
            recommend_mode=recommend_mode,
            include_waiting=include_waiting
        )

        if item is not None:
            results.append(item)

        progress.progress((idx + 1) / len(tickers))
        time.sleep(request_delay)

    status_text.write("분석 완료")

    if not results:
        st.warning("현재 조건에 맞는 추천 코인이 없습니다.")
        st.info("""
조건이 너무 강하면 아래 순서로 완화해보세요.

1. 과매도 판정을 'K 또는 D 하나만 20 이하'로 변경  
2. 추천 최소 조건을 '공격형'으로 변경  
3. 대기 코인도 표시 체크  
4. 스캔 대상을 거래대금 상위 60개에서 전체 KRW로 변경  
""")
        st.stop()

    result_tickers = [x["코인"] for x in results]
    prices = get_current_prices_batch(result_tickers)

    for item in results:
        price = prices.get(item["코인"])
        item["현재가"] = price
        item["현재가표시"] = format_price(price)

    df = pd.DataFrame(results)

    numeric_cols = [
        "일봉단기K", "일봉단기D", "일봉단기RSI",
        "일봉중기K", "일봉중기D", "일봉중기RSI",
        "일봉장기K", "일봉장기D", "일봉장기RSI",

        "4H단기K", "4H단기D", "4H단기RSI",
        "4H중기K", "4H중기D", "4H중기RSI",
        "4H장기K", "4H장기D", "4H장기RSI",

        "1H단기K", "1H단기D", "1H단기RSI",
        "1H중기K", "1H중기D", "1H중기RSI",
        "1H장기K", "1H장기D", "1H장기RSI",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    df["등급순서"] = df["등급"].apply(grade_rank)

    df = df.sort_values(
        by=["등급순서", "점수", "전체과매도개수", "일봉과매도개수", "4H과매도개수", "1H과매도개수"],
        ascending=[True, False, False, False, False, False]
    ).drop(columns=["등급순서"])

    st.subheader("추천 결과")

    view_cols = [
        "등급",
        "코인",
        "점수",
        "현재가표시",
        "추천사유",
        "전체과매도개수",
        "전체조건개수",

        "일봉과매도개수",
        "4H과매도개수",
        "1H과매도개수",

        "일봉단기",
        "일봉중기",
        "일봉장기",

        "4H단기",
        "4H중기",
        "4H장기",

        "1H단기",
        "1H중기",
        "1H장기",
    ]

    st.dataframe(
        df[view_cols],
        use_container_width=True,
        hide_index=True
    )

    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "CSV 다운로드",
        data=csv,
        file_name=f"multi_stochrsi_recommend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.subheader("코인별 상세 분석")

    for _, row in df.iterrows():
        title = f"{row['등급']}등급 | {row['코인']} | {row['점수']}점 | {row['추천사유']}"

        with st.expander(title):
            st.markdown("### 과매도 요약")
            st.write(f"전체 과매도: {row['전체과매도개수']} / {row['전체조건개수']}")
            st.write(f"일봉: {row['일봉과매도개수']} / 3")
            st.write(f"4시간봉: {row['4H과매도개수']} / 3")
            st.write(f"1시간봉: {row['1H과매도개수']} / 3")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("### 일봉")
                st.write(f"단기: {row['일봉단기']} / K {row['일봉단기K']} / D {row['일봉단기D']} / RSI {row['일봉단기RSI']}")
                st.write(f"중기: {row['일봉중기']} / K {row['일봉중기K']} / D {row['일봉중기D']} / RSI {row['일봉중기RSI']}")
                st.write(f"장기: {row['일봉장기']} / K {row['일봉장기K']} / D {row['일봉장기D']} / RSI {row['일봉장기RSI']}")

            with c2:
                st.markdown("### 4시간봉")
                st.write(f"단기: {row['4H단기']} / K {row['4H단기K']} / D {row['4H단기D']} / RSI {row['4H단기RSI']}")
                st.write(f"중기: {row['4H중기']} / K {row['4H중기K']} / D {row['4H중기D']} / RSI {row['4H중기RSI']}")
                st.write(f"장기: {row['4H장기']} / K {row['4H장기K']} / D {row['4H장기D']} / RSI {row['4H장기RSI']}")

            with c3:
                st.markdown("### 1시간봉")
                st.write(f"단기: {row['1H단기']} / K {row['1H단기K']} / D {row['1H단기D']} / RSI {row['1H단기RSI']}")
                st.write(f"중기: {row['1H중기']} / K {row['1H중기K']} / D {row['1H중기D']} / RSI {row['1H중기RSI']}")
                st.write(f"장기: {row['1H장기']} / K {row['1H장기K']} / D {row['1H장기D']} / RSI {row['1H장기RSI']}")


st.divider()

st.caption("주의: 본 프로그램은 투자 참고용 보조 도구이며, 매수/매도 판단의 책임은 사용자 본인에게 있습니다.")
