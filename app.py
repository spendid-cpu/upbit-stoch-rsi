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
# 환경변수 (Railway 등 서버 배포 시 주입)
# ==================================================

import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")


# ==================================================
# 텔레그램 전송 유틸
# ==================================================

def send_telegram_message(text: str) -> bool:
    """
    텔레그램으로 메시지를 전송합니다.
    3900자 초과 시 자동으로 분할 전송합니다.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    chunks = []
    while len(text) > 3900:
        cut = text.rfind("\n", 0, 3900)
        if cut <= 0:
            cut = 3900
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    chunks.append(text)

    ok = True
    for part in chunks:
        try:
            r = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": part,
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if r.status_code >= 400:
                ok = False
            time.sleep(0.3)
        except Exception:
            ok = False

    return ok


def build_telegram_result_message(results: list, btc_info: dict) -> str:
    """
    추천 결과를 텔레그램 메시지 문자열로 변환합니다.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("📊 Stoch RSI 다중조건 코인 추천")
    lines.append(f"스캔시간: {now_str}")
    lines.append("")

    # BTC 주봉 MA20 상태
    btc_status = btc_info.get("status", "-")
    btc_price  = btc_info.get("current_price")
    weekly_ma20 = btc_info.get("weekly_ma20")
    gap         = btc_info.get("gap_rate")

    lines.append(f"🔶 BTC 주봉 MA20 상태: {btc_status}")
    if btc_price:
        lines.append(f"BTC 현재가: {format_price(btc_price)}")
    if weekly_ma20:
        lines.append(f"주봉 MA20: {format_price(weekly_ma20)}")
    if gap is not None:
        lines.append(f"MA20 이격률: {gap:.2f}%")
    lines.append("")

    if not results:
        lines.append("현재 조건에 맞는 추천 코인이 없습니다.")
        return "\n".join(lines)

    lines.append(f"총 {len(results)}개 코인 추천")
    lines.append("")

    for i, row in enumerate(results[:20], start=1):  # 최대 20개
        grade  = row.get("등급", "-")
        coin   = row.get("코인", "-")
        score  = row.get("점수", 0)
        reason = row.get("추천사유", "-")
        price  = row.get("현재가표시", "-")

        day_cnt = row.get("일봉과매도개수", 0)
        h4_cnt  = row.get("4H과매도개수", 0)
        h1_cnt  = row.get("1H과매도개수", 0)

        lines.append(
            f"{i}. [{grade}] {coin} | {score}점 | {price}"
        )
        lines.append(
            f"   일봉 {day_cnt}/3 · 4H {h4_cnt}/3 · 1H {h1_cnt}/3"
        )
        lines.append(f"   {reason}")
        lines.append("")

    lines.append("※ 투자 판단의 책임은 사용자 본인에게 있습니다.")
    return "\n".join(lines)


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
    return {"S+": 1, "S": 2, "A": 3, "B": 4, "C": 5, "대기": 6}.get(grade, 99)


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
    return None


# ==================================================
# Upbit API 함수
# ==================================================

@st.cache_data(ttl=300)
def get_krw_markets():
    try:
        url = "https://api.upbit.com/v1/market/all"
        res = requests.get(url, params={"isDetails": "true"}, timeout=10)
        data = res.json()
        markets = []
        for item in data:
            market  = item.get("market", "")
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
    markets = [m for m in get_krw_markets() if m != "KRW-BTC"]
    if not markets:
        return []
    tickers = []
    for i in range(0, len(markets), 100):
        batch = markets[i:i + 100]
        try:
            res  = requests.get(
                "https://api.upbit.com/v1/ticker",
                params={"markets": ",".join(batch)},
                timeout=10
            )
            data = res.json()
            if isinstance(data, list):
                tickers.extend(data)
            time.sleep(0.08)
        except Exception:
            time.sleep(0.2)
    if not tickers:
        return markets[:limit]
    tickers = sorted(tickers, key=lambda x: x.get("acc_trade_price_24h", 0), reverse=True)
    return [x["market"] for x in tickers[:limit]]


@st.cache_data(ttl=300)
def get_ohlcv_cached(ticker, interval, count):
    try:
        return pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=count)
    except Exception:
        return None


@st.cache_data(ttl=120)
def get_current_prices_batch(tickers):
    if not tickers:
        return {}
    result = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            res  = requests.get(
                "https://api.upbit.com/v1/ticker",
                params={"markets": ",".join(batch)},
                timeout=10
            )
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
    close     = close.astype(float)
    delta     = close.diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss  = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_stoch_rsi(df, rsi_period=5, stoch_period=5, k_smooth=3, d_smooth=3):
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    rsi     = calculate_rsi(df["close"], period=rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    denom   = (rsi_max - rsi_min).replace(0, np.nan)
    raw     = 100 * ((rsi - rsi_min) / denom)
    df["rsi"]          = rsi
    df["stoch_rsi_raw"] = raw
    df["stoch_rsi_k"]  = raw.rolling(k_smooth).mean()
    df["stoch_rsi_d"]  = df["stoch_rsi_k"].rolling(d_smooth).mean()
    return df


def calculate_stoch_rsi_by_setting(df, setting):
    return calculate_stoch_rsi(
        df,
        rsi_period=setting["rsi_period"],
        stoch_period=setting["stoch_period"],
        k_smooth=setting["k_smooth"],
        d_smooth=setting["d_smooth"],
    )


# ==================================================
# ★ BTC 주봉 MA20 필터 (핵심 변경 부분)
# ==================================================

@st.cache_data(ttl=600)
def get_btc_weekly_ohlcv(count=60):
    """
    BTC 주봉 데이터를 가져옵니다.
    pyupbit의 week 인터벌을 사용합니다.
    """
    try:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="week", count=count)
        return df
    except Exception:
        return None


def get_btc_ma20_status(warning_gap: float = 0.3) -> dict:
    """
    BTC 주봉 MA20 상태를 반환합니다.

    반환 구조:
    {
        "ok": bool,               # 필터 통과 여부
        "status": str,            # 상태 텍스트
        "current_price": float,   # BTC 현재가
        "weekly_ma20": float,     # 주봉 MA20 값
        "daily_ma20": float,      # 일봉 MA20 값 (참고용 표시)
        "gap_rate": float,        # 주봉 MA20 대비 이격률 %
        "reason": str,
    }
    """
    result_base = {
        "ok": False,
        "status": "데이터 없음",
        "current_price": None,
        "weekly_ma20": None,
        "daily_ma20": None,
        "gap_rate": None,
        "reason": "BTC 데이터 조회 실패",
    }

    try:
        # ── 주봉 데이터 ──────────────────────────────────────
        weekly_df = get_btc_weekly_ohlcv(count=60)
        if weekly_df is None or len(weekly_df) < 22:
            return result_base

        # 완성된 직전 주봉 MA20 (현재 진행 중인 봉 제외)
        confirmed_closes  = weekly_df["close"].iloc[:-1]
        prev_weekly_ma20  = confirmed_closes.iloc[-20:].mean()
        prev_weekly_close = confirmed_closes.iloc[-1]

        # 현재 진행 중인 봉 포함 실시간 MA20
        current_price  = pyupbit.get_current_price("KRW-BTC")
        if current_price is None:
            return result_base
        current_price = float(current_price)

        live_closes    = pd.concat([
            weekly_df["close"].iloc[-20:-1],
            pd.Series([current_price])
        ])
        live_weekly_ma20 = live_closes.mean()

        gap_rate = ((current_price - live_weekly_ma20) / live_weekly_ma20) * 100

        confirmed_above = prev_weekly_close > prev_weekly_ma20
        live_above      = current_price > live_weekly_ma20

        # ── 일봉 MA20 (참고용) ───────────────────────────────
        daily_ma20 = None
        try:
            daily_df  = get_ohlcv_cached("KRW-BTC", "day", 30)
            if daily_df is not None and len(daily_df) >= 21:
                daily_ma20 = float(daily_df["close"].iloc[-21:-1].mean())
        except Exception:
            pass

        # ── 상태 판단 ────────────────────────────────────────
        if confirmed_above and live_above:
            if gap_rate >= warning_gap:
                status = "주봉 MA20 위 유지"
                ok     = True
            else:
                status = "주봉 MA20 근접 주의"
                ok     = True
        elif confirmed_above and not live_above:
            status = "주봉 MA20 이탈 중 (미확정)"
            ok     = False
        elif not confirmed_above and live_above:
            status = "주봉 MA20 재돌파 시도 중"
            ok     = False
        else:
            status = "주봉 MA20 아래"
            ok     = False

        return {
            "ok":            ok,
            "status":        status,
            "current_price": current_price,
            "weekly_ma20":   float(live_weekly_ma20),
            "daily_ma20":    daily_ma20,
            "gap_rate":      float(gap_rate),
            "prev_close":    float(prev_weekly_close),
            "prev_ma20":     float(prev_weekly_ma20),
            "reason":        status,
        }

    except Exception as e:
        result_base["reason"] = str(e)
        return result_base


# ==================================================
# Stoch RSI 상태 판단
# ==================================================

def judge_stochrsi_state(df, setting, oversold_mode="both"):
    calc = calculate_stoch_rsi_by_setting(df, setting)
    if calc is None:
        return {"ok": False, "oversold": False, "overbought": False,
                "k": None, "d": None, "rsi": None, "reason": "데이터 없음"}
    calc = calc.dropna().copy()
    if len(calc) == 0:
        return {"ok": False, "oversold": False, "overbought": False,
                "k": None, "d": None, "rsi": None, "reason": "데이터 부족"}
    last  = calc.iloc[-1]
    k     = safe_float(last["stoch_rsi_k"])
    d     = safe_float(last["stoch_rsi_d"])
    rsi   = safe_float(last["rsi"])
    if k is None or d is None:
        return {"ok": False, "oversold": False, "overbought": False,
                "k": k, "d": d, "rsi": rsi, "reason": "K/D 계산 불가"}
    oversold_val  = setting["oversold"]
    overbought_val = setting["overbought"]
    if oversold_mode == "either":
        is_oversold = k <= oversold_val or d <= oversold_val
    else:
        is_oversold = k <= oversold_val and d <= oversold_val
    is_overbought = k >= overbought_val or d >= overbought_val
    state_text = "과매도" if is_oversold else ("과매수" if is_overbought else "중립")
    return {
        "ok": True,
        "oversold":   bool(is_oversold),
        "overbought": bool(is_overbought),
        "k":   k,
        "d":   d,
        "rsi": rsi,
        "reason": f"{state_text} / K {k:.2f} / D {d:.2f} / RSI {rsi:.2f}",
    }


def analyze_timeframe_all_settings(df, oversold_mode="both"):
    return {
        key: judge_stochrsi_state(df, setting, oversold_mode=oversold_mode)
        for key, setting in STOCH_RSI_SETTINGS.items()
    }


# ==================================================
# 점수 계산
# ==================================================

def calculate_multi_stochrsi_score(analysis):
    score = 0
    oversold_count = 0
    total_count    = 0
    timeframe_summary = {}

    for tf_key, tf_info in TIMEFRAME_SETTINGS.items():
        tf_weight        = tf_info["weight"]
        tf_oversold_count = 0
        tf_total_count   = 0

        for setting_key, setting in STOCH_RSI_SETTINGS.items():
            total_count    += 1
            tf_total_count += 1
            state = analysis[tf_key][setting_key]
            if state["oversold"]:
                score          += tf_weight * setting["weight"]
                oversold_count += 1
                tf_oversold_count += 1

        if tf_oversold_count == tf_total_count:
            score += TIMEFRAME_ALL_OVERSOLD_BONUS.get(tf_key, 0)

        timeframe_summary[tf_key] = {
            "oversold_count": tf_oversold_count,
            "total_count":    tf_total_count,
        }

    if oversold_count == total_count:
        score += ALL_NINE_OVERSOLD_BONUS

    return {
        "score":            round(score, 1),
        "oversold_count":   oversold_count,
        "total_count":      total_count,
        "timeframe_summary": timeframe_summary,
    }


def is_recommendable_by_multi_stochrsi(analysis, mode="balanced"):
    day_any = any(analysis["day"][k]["oversold"]       for k in STOCH_RSI_SETTINGS)
    h4_any  = any(analysis["minute240"][k]["oversold"] for k in STOCH_RSI_SETTINGS)
    h1_any  = any(analysis["minute60"][k]["oversold"]  for k in STOCH_RSI_SETTINGS)
    total_oversold = sum(
        1 for tf in TIMEFRAME_SETTINGS
        for sk in STOCH_RSI_SETTINGS
        if analysis[tf][sk]["oversold"]
    )
    if mode == "strict":
        return day_any and h4_any and h1_any
    if mode == "aggressive":
        return total_oversold >= 2
    return day_any and (h4_any or h1_any)


# ==================================================
# 코인 분석
# ==================================================

def analyze_coin_multi_stochrsi(ticker, oversold_mode="both",
                                 recommend_mode="balanced", include_waiting=False):
    try:
        day_df = get_ohlcv_cached(ticker, "day",       180)
        h4_df  = get_ohlcv_cached(ticker, "minute240", 180)
        h1_df  = get_ohlcv_cached(ticker, "minute60",  180)

        if day_df is None or h4_df is None or h1_df is None:
            return None
        if len(day_df) < 80 or len(h4_df) < 80 or len(h1_df) < 80:
            return None

        analysis = {
            "day":       analyze_timeframe_all_settings(day_df, oversold_mode),
            "minute240": analyze_timeframe_all_settings(h4_df,  oversold_mode),
            "minute60":  analyze_timeframe_all_settings(h1_df,  oversold_mode),
        }

        score_info    = calculate_multi_stochrsi_score(analysis)
        score         = score_info["score"]
        grade         = get_grade_by_score(score)
        recommendable = is_recommendable_by_multi_stochrsi(analysis, mode=recommend_mode)

        if not recommendable and not include_waiting:
            return None
        if grade is None and not include_waiting:
            return None
        if grade is None:
            grade = "대기"

        day_count = score_info["timeframe_summary"]["day"]["oversold_count"]
        h4_count  = score_info["timeframe_summary"]["minute240"]["oversold_count"]
        h1_count  = score_info["timeframe_summary"]["minute60"]["oversold_count"]

        return {
            "등급": grade, "코인": ticker, "점수": score,
            "추천사유":      f"일봉 {day_count}/3, 4H {h4_count}/3, 1H {h1_count}/3",
            "전체과매도개수": score_info["oversold_count"],
            "전체조건개수":  score_info["total_count"],
            "일봉과매도개수": day_count,
            "4H과매도개수":  h4_count,
            "1H과매도개수":  h1_count,
            # 과매도 YES/NO
            "일봉단기": "YES" if analysis["day"]["short"]["oversold"]       else "NO",
            "일봉중기": "YES" if analysis["day"]["middle"]["oversold"]      else "NO",
            "일봉장기": "YES" if analysis["day"]["long"]["oversold"]        else "NO",
            "4H단기":   "YES" if analysis["minute240"]["short"]["oversold"] else "NO",
            "4H중기":   "YES" if analysis["minute240"]["middle"]["oversold"] else "NO",
            "4H장기":   "YES" if analysis["minute240"]["long"]["oversold"]  else "NO",
            "1H단기":   "YES" if analysis["minute60"]["short"]["oversold"]  else "NO",
            "1H중기":   "YES" if analysis["minute60"]["middle"]["oversold"] else "NO",
            "1H장기":   "YES" if analysis["minute60"]["long"]["oversold"]   else "NO",
            # K/D/RSI 수치
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
        print(f"{ticker} 분석 오류: {e}")
        return None


# ==================================================
# 화면 UI
# ==================================================

st.title("📊 Upbit Stoch RSI 다중조건 코인 추천기")
st.caption("단기/중기/장기 Stochastic RSI를 일봉·4시간봉·1시간봉에서 분석하여 점수화합니다.")

with st.expander("추천 로직 설명", expanded=False):
    st.markdown("""
## 지표 기준

이 앱은 **Stochastic RSI** 기준입니다.

### 단기 / 중기 / 장기 세팅 (생략 — 원본과 동일)

## BTC 필터 변경 (v2)
- 기존 **일봉 MA20** 기준 → **주봉 MA20** 기준으로 변경
- 단기 노이즈를 줄이고 매크로 추세를 반영합니다
- 일봉 MA20은 참고용으로 함께 표시됩니다

## 텔레그램 알림
- 추천받기 실행 후 결과를 텔레그램으로 즉시 전송할 수 있습니다
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` 환경변수가 설정된 경우 자동 활성화됩니다
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
    "거래대금 상위 N개", min_value=10, max_value=150, value=60, step=10
)

max_scan_count = st.sidebar.number_input(
    "최대 스캔 개수", min_value=0, max_value=200, value=60, step=10,
    help="0이면 제한 없음"
)

manual_text = st.sidebar.text_area(
    "수동 관심코인",
    value="KRW-ETH\nKRW-XRP\nKRW-SOL\nKRW-DOGE\nKRW-ADA\nKRW-AVAX\nKRW-LINK\nKRW-DOT\nKRW-SUI\nKRW-APT",
    height=180,
)

st.sidebar.divider()

recommend_mode_text = st.sidebar.selectbox(
    "추천 최소 조건",
    ["균형형", "보수형", "공격형"],
    index=0,
)
recommend_mode = {"보수형": "strict", "공격형": "aggressive"}.get(recommend_mode_text, "balanced")

oversold_mode_text = st.sidebar.selectbox(
    "과매도 판정",
    ["K와 D 모두 20 이하", "K 또는 D 하나만 20 이하"],
    index=0
)
oversold_mode = "both" if oversold_mode_text == "K와 D 모두 20 이하" else "either"

btc_warning_gap = st.sidebar.number_input(
    "BTC 주봉 MA20 근접 주의 이격률 %",
    min_value=0.0, max_value=5.0, value=0.3, step=0.1
)

ignore_btc_filter = st.sidebar.checkbox("테스트용: BTC 주봉 MA20 필터 무시", value=False)

include_waiting = st.sidebar.checkbox(
    "대기 코인도 표시", value=False,
    help="추천 최소 조건을 만족하지 않아도 점수 확인용으로 표시"
)

request_delay = st.sidebar.number_input(
    "요청 간격 초", min_value=0.02, max_value=1.0, value=0.08, step=0.02
)

# 텔레그램 설정 섹션
st.sidebar.divider()
st.sidebar.subheader("📬 텔레그램 알림 설정")

telegram_enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

if telegram_enabled:
    st.sidebar.success("✅ 텔레그램 환경변수 감지됨 (자동 활성화)")
else:
    st.sidebar.warning("⚠️ 환경변수 미설정 — 아래에 직접 입력 가능")

# 환경변수가 없으면 UI에서 직접 입력받기
if not telegram_enabled:
    manual_token   = st.sidebar.text_input(
        "BOT TOKEN", value="", type="password",
        help="환경변수 TELEGRAM_BOT_TOKEN을 설정하거나 여기에 입력"
    )
    manual_chat_id = st.sidebar.text_input(
        "CHAT ID", value="",
        help="환경변수 TELEGRAM_CHAT_ID를 설정하거나 여기에 입력"
    )
    if manual_token and manual_chat_id:
        TELEGRAM_BOT_TOKEN = manual_token
        TELEGRAM_CHAT_ID   = manual_chat_id
        telegram_enabled   = True

send_telegram_on_result = st.sidebar.checkbox(
    "추천 결과 텔레그램 전송",
    value=telegram_enabled,
    disabled=not telegram_enabled,
    help="추천받기 완료 후 자동으로 텔레그램에 전송합니다"
)


# ==================================================
# ★ BTC 주봉 MA20 + 일봉 MA20 상태 표시
# ==================================================

btc = get_btc_ma20_status(warning_gap=btc_warning_gap)

st.subheader("🔶 BTC 시장 상태")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("BTC 주봉 MA20 상태", btc["status"])

with col2:
    st.metric("BTC 현재가", format_price(btc["current_price"]))

with col3:
    st.metric("주봉 MA20", format_price(btc["weekly_ma20"]))

with col4:
    # ★ 일봉 MA20 참고용 표시
    st.metric(
        "일봉 MA20 (참고)",
        format_price(btc.get("daily_ma20")),
        help="주봉 MA20이 메인 필터이며, 일봉 MA20은 단기 추세 참고용입니다."
    )

with col5:
    gap = btc["gap_rate"]
    st.metric(
        "주봉 MA20 이격률",
        "-" if gap is None else f"{gap:.2f}%",
        delta=None if gap is None else f"{gap:.2f}%",
        delta_color="normal"
    )

if btc["ok"]:
    st.success(f"✅ BTC 주봉 MA20 필터 통과: 신규 추천 허용 상태입니다. ({btc['reason']})")
else:
    st.warning(f"⚠️ BTC 주봉 MA20 필터 미통과: {btc['reason']}")

# 일봉 vs 주봉 MA20 비교 정보 박스
with st.expander("📈 BTC MA20 상세 (일봉 vs 주봉 비교)", expanded=False):
    d_ma20 = btc.get("daily_ma20")
    w_ma20 = btc.get("weekly_ma20")
    cur    = btc.get("current_price")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 주봉 MA20 (필터 기준)")
        st.write(f"현재 주봉 MA20: **{format_price(w_ma20)}**")
        if cur and w_ma20:
            d = ((cur - w_ma20) / w_ma20) * 100
            st.write(f"이격률: **{d:.2f}%**")
        st.caption("주봉 MA20은 매크로 상승/하락 추세를 판단하는 핵심 기준입니다.")
    with col_b:
        st.markdown("#### 일봉 MA20 (참고용)")
        st.write(f"현재 일봉 MA20: **{format_price(d_ma20)}**")
        if cur and d_ma20:
            d2 = ((cur - d_ma20) / d_ma20) * 100
            st.write(f"이격률: **{d2:.2f}%**")
        st.caption("일봉 MA20은 단기 추세 확인용입니다. 필터로 사용되지 않습니다.")


# ==================================================
# 스캔 대상 만들기
# ==================================================

def build_scan_list():
    all_krw = [m for m in get_krw_markets() if m != "KRW-BTC"]
    if scan_mode == "전체 KRW":
        tickers = all_krw
    elif scan_mode == "수동 관심코인":
        manual = [x.strip().upper() for x in manual_text.splitlines() if x.strip()]
        tickers = [x for x in manual if x in all_krw]
    else:
        tickers = get_top_trade_value_markets(limit=top_count)
        tickers = [x for x in tickers if x in all_krw]
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
        st.error("⛔ BTC 주봉 MA20 조건을 충족하지 않아 추천을 중단합니다.")
        st.stop()

    tickers = build_scan_list()
    if not tickers:
        st.error("스캔할 코인이 없습니다.")
        st.stop()

    st.info(f"총 {len(tickers)}개 코인을 분석합니다.")
    progress    = st.progress(0)
    status_text = st.empty()
    results     = []

    for idx, ticker in enumerate(tickers):
        status_text.write(f"분석 중: {ticker} ({idx + 1}/{len(tickers)})")
        item = analyze_coin_multi_stochrsi(
            ticker=ticker,
            oversold_mode=oversold_mode,
            recommend_mode=recommend_mode,
            include_waiting=include_waiting,
        )
        if item is not None:
            results.append(item)
        progress.progress((idx + 1) / len(tickers))
        time.sleep(request_delay)

    status_text.write("분석 완료")

    if not results:
        st.warning("현재 조건에 맞는 추천 코인이 없습니다.")
        st.stop()

    # 현재가 조회 및 병합
    result_tickers = [x["코인"] for x in results]
    prices = get_current_prices_batch(tuple(result_tickers))
    for item in results:
        price = prices.get(item["코인"])
        item["현재가"]    = price
        item["현재가표시"] = format_price(price)

    df = pd.DataFrame(results)

    # 수치 컬럼 반올림
    numeric_cols = [c for c in df.columns if any(
        c.endswith(s) for s in ["K", "D", "RSI"]
    )]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    df["등급순서"] = df["등급"].apply(grade_rank)
    df = df.sort_values(
        by=["등급순서", "점수", "전체과매도개수", "일봉과매도개수", "4H과매도개수", "1H과매도개수"],
        ascending=[True, False, False, False, False, False]
    ).drop(columns=["등급순서"])

    # ── 결과 표시 ──────────────────────────────────────────
    st.subheader("📋 추천 결과")

    view_cols = [
        "등급", "코인", "점수", "현재가표시", "추천사유",
        "전체과매도개수", "전체조건개수",
        "일봉과매도개수", "4H과매도개수", "1H과매도개수",
        "일봉단기", "일봉중기", "일봉장기",
        "4H단기",   "4H중기",   "4H장기",
        "1H단기",   "1H중기",   "1H장기",
    ]
    st.dataframe(df[view_cols], use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 CSV 다운로드",
        data=csv,
        file_name=f"stochrsi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # ── 텔레그램 전송 ───────────────────────────────────────
    if send_telegram_on_result and telegram_enabled:
        with st.spinner("텔레그램으로 전송 중..."):
            msg = build_telegram_result_message(df.to_dict("records"), btc)
            ok  = send_telegram_message(msg)
        if ok:
            st.success("✅ 텔레그램 전송 완료!")
        else:
            st.error("❌ 텔레그램 전송 실패 — BOT TOKEN / CHAT ID를 확인해주세요.")

    # ── 코인별 상세 ─────────────────────────────────────────
    st.subheader("🔍 코인별 상세 분석")
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
