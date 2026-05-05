# alert_bot.py
# Telegram Coin Alert Bot V3.3
# - Upbit KRW 4H scanner
# - Preliminary / Confirmed alert auto-detection
# - BTC 4H filter
# - Candle quality filter
# - MA5 slope filter
# - Telegram A/B/WATCH grading system

import os
import time
import math
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np

try:
    import pyupbit
except Exception:
    pyupbit = None


# =========================
# 1. ENV CONFIG
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

ALERT_INTERVAL = os.getenv("ALERT_INTERVAL", "minute240").strip()
ALERT_CANDLE_COUNT = int(os.getenv("ALERT_CANDLE_COUNT", "700"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.12"))
MAX_ALERT_COUNT = int(os.getenv("MAX_ALERT_COUNT", "10"))
SEND_EMPTY_ALERT = os.getenv("SEND_EMPTY_ALERT", "true").strip().lower() == "true"

KST = ZoneInfo("Asia/Seoul")

UPBIT_MARKET_URL = "https://api.upbit.com/v1/market/all"
UPBIT_TICKER_URL = "https://api.upbit.com/v1/ticker"


# =========================
# 2. BASIC UTILS
# =========================

def now_kst():
    return datetime.now(KST)


def format_now():
    return now_kst().strftime("%Y-%m-%d %H:%M:%S KST")


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def get_value(candidate, keys, default=0.0):
    for key in keys:
        if key in candidate:
            return candidate[key]
    return default


def percent(a, b):
    try:
        if b == 0:
            return 0.0
        return (a / b - 1.0) * 100.0
    except Exception:
        return 0.0


def is_number(x):
    try:
        if x is None:
            return False
        v = float(x)
        return not math.isnan(v)
    except Exception:
        return False


# =========================
# 3. TELEGRAM
# =========================

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 비어 있습니다.")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Telegram message limit is 4096 chars.
    chunks = []
    max_len = 3900

    if len(message) <= max_len:
        chunks = [message]
    else:
        lines = message.split("\n")
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > max_len:
                chunks.append(current)
                current = line
            else:
                current += ("\n" if current else "") + line
        if current:
            chunks.append(current)

    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True
        }

        res = requests.post(url, data=payload, timeout=15)
        if res.status_code != 200:
            raise RuntimeError(f"Telegram 전송 실패: {res.status_code} / {res.text}")

        time.sleep(0.3)


# =========================
# 4. UPBIT DATA
# =========================

def get_krw_markets():
    """
    KRW 마켓 전체 조회.
    pyupbit 우선 사용, 실패 시 requests 사용.
    """
    try:
        if pyupbit is not None:
            markets = pyupbit.get_tickers(fiat="KRW")
            if markets:
                return sorted(markets)
    except Exception:
        pass

    try:
        res = requests.get(UPBIT_MARKET_URL, params={"isDetails": "false"}, timeout=15)
        res.raise_for_status()
        data = res.json()
        markets = [x["market"] for x in data if x.get("market", "").startswith("KRW-")]
        return sorted(markets)
    except Exception as e:
        raise RuntimeError(f"KRW 마켓 조회 실패: {e}")


def get_ohlcv(market, interval="minute240", count=700, retry=2):
    """
    pyupbit OHLCV 사용.
    pyupbit는 count가 커도 내부적으로 처리 가능.
    """
    if pyupbit is None:
        return None

    for attempt in range(retry + 1):
        try:
            df = pyupbit.get_ohlcv(market, interval=interval, count=count)
            if df is None or len(df) < 50:
                time.sleep(0.3)
                continue

            df = df.copy()

            # pyupbit columns:
            # open, high, low, close, volume, value
            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    return None

            if "value" not in df.columns:
                df["value"] = df["close"] * df["volume"]

            df = df.dropna()
            return df

        except Exception:
            if attempt >= retry:
                return None
            time.sleep(0.5)

    return None


def get_ticker_prices(markets):
    try:
        joined = ",".join(markets)
        res = requests.get(UPBIT_TICKER_URL, params={"markets": joined}, timeout=15)
        res.raise_for_status()
        return res.json()
    except Exception:
        return []


# =========================
# 5. INDICATORS
# =========================

def calc_rsi(close, period=14):
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    return rsi


def calc_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi = calc_rsi(close, rsi_period)

    min_rsi = rsi.rolling(stoch_period, min_periods=stoch_period).min()
    max_rsi = rsi.rolling(stoch_period, min_periods=stoch_period).max()

    stoch = (rsi - min_rsi) / (max_rsi - min_rsi)
    stoch = stoch.replace([np.inf, -np.inf], np.nan).fillna(0.5) * 100

    k = stoch.rolling(k_smooth, min_periods=1).mean()
    d = k.rolling(d_smooth, min_periods=1).mean()

    return k, d


def prepare_indicators(df):
    df = df.copy()

    df["ma5"] = df["close"].rolling(5, min_periods=5).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=20).mean()

    short_k, short_d = calc_stoch_rsi(
        df["close"],
        rsi_period=14,
        stoch_period=14,
        k_smooth=3,
        d_smooth=3
    )

    mid_k, mid_d = calc_stoch_rsi(
        df["close"],
        rsi_period=14,
        stoch_period=28,
        k_smooth=3,
        d_smooth=3
    )

    df["short_k"] = short_k
    df["short_d"] = short_d
    df["mid_k"] = mid_k
    df["mid_d"] = mid_d

    df["value_ma20_prev"] = df["value"].shift(1).rolling(20, min_periods=5).mean()

    return df


# =========================
# 6. ALERT MODE
# =========================

def detect_alert_mode():
    """
    GitHub Actions 스케줄 기준:
    - 예비 알림: 00:05, 08:05, 12:05, 16:05, 20:05 KST
    - 확정 알림: 09:10, 13:10, 17:10, 21:10 KST

    확정 시간대 근처면 confirmed.
    나머지는 preliminary.
    """
    now = now_kst()
    h = now.hour
    m = now.minute

    if h in [9, 13, 17, 21] and 0 <= m <= 45:
        return "confirmed"

    return "preliminary"


def get_signal_index(df, alert_mode):
    """
    pyupbit minute240은 현재 진행 중인 4H 캔들이 포함된다.

    confirmed:
      막 마감된 이전 4H 캔들을 평가해야 하므로 -2 사용.

    preliminary:
      현재 진행 중인 4H 캔들을 평가하므로 -1 사용.
    """
    if len(df) < 60:
        return None

    if alert_mode == "confirmed":
        return len(df) - 2

    return len(df) - 1


# =========================
# 7. BTC FILTER
# =========================

def get_btc_filter(alert_mode):
    """
    BTC 4H 필터.
    - 예비: BTC가 급락 중이면 차단
    - 확정: BTC close > MA20 조건 포함
    """
    df = get_ohlcv("KRW-BTC", interval=ALERT_INTERVAL, count=ALERT_CANDLE_COUNT)
    if df is None or len(df) < 60:
        return {
            "pass": False,
            "reason": "BTC 데이터 부족",
            "btc_close": 0,
            "btc_ma20": 0,
            "btc_3bar": 0
        }

    df = prepare_indicators(df)
    idx = get_signal_index(df, alert_mode)

    if idx is None or idx < 25:
        return {
            "pass": False,
            "reason": "BTC 인덱스 부족",
            "btc_close": 0,
            "btc_ma20": 0,
            "btc_3bar": 0
        }

    close = safe_float(df["close"].iloc[idx])
    ma20 = safe_float(df["ma20"].iloc[idx])
    close_3ago = safe_float(df["close"].iloc[idx - 3])
    btc_3bar = percent(close, close_3ago)

    if ma20 <= 0:
        return {
            "pass": False,
            "reason": "BTC MA20 없음",
            "btc_close": close,
            "btc_ma20": ma20,
            "btc_3bar": btc_3bar
        }

    if alert_mode == "confirmed":
        passed = close > ma20 and btc_3bar >= -3.0
        reason = "BTC close > MA20" if passed else "BTC confirmed filter fail"
    else:
        passed = close >= ma20 * 0.985 and btc_3bar >= -3.5
        reason = "BTC not crashing" if passed else "BTC preliminary filter fail"

    return {
        "pass": passed,
        "reason": reason,
        "btc_close": close,
        "btc_ma20": ma20,
        "btc_3bar": btc_3bar
    }


# =========================
# 8. SIGNAL SCORE
# =========================

def calculate_base_score(metrics, alert_mode):
    """
    기존 조건검색용 기본 점수.
    점수 자체는 후보 선별용이고,
    V3.3 등급은 별도 alert_grade에서 계산한다.
    """
    score = 0

    volume_ratio = metrics["volume_ratio"]
    short_k = metrics["short_k"]
    short_d = metrics["short_d"]
    mid_k = metrics["mid_k"]
    ma20_dev = metrics["ma20_dev"]
    recent_3bar = metrics["recent_3bar"]
    close_position = metrics["close_position"]
    upper_wick_ratio = metrics["upper_wick_ratio"]
    candle_change = metrics["candle_change"]
    ma5_up = metrics["ma5_up"]
    bullish_ok = metrics["bullish_ok"]

    # 거래대금
    if volume_ratio >= 2.0:
        score += 60
    elif volume_ratio >= 1.7:
        score += 50
    elif volume_ratio >= 1.5:
        score += 40
    elif volume_ratio >= 1.3:
        score += 30

    # 단기 Stoch RSI
    if 20 <= short_k <= 55 and short_d <= 50:
        score += 55
    elif short_k <= 65 and short_d <= 55:
        score += 45
    elif short_k <= 70 and short_d <= 60:
        score += 30

    # 중기 K
    if mid_k <= 45:
        score += 35
    elif mid_k <= 60:
        score += 25
    elif mid_k <= 65:
        score += 15

    # MA20 근처
    if -2.0 <= ma20_dev <= 2.0:
        score += 35
    elif -4.0 <= ma20_dev <= 4.0:
        score += 25
    elif -6.0 <= ma20_dev <= 6.0:
        score += 15

    # 최근 3봉
    if 0 <= recent_3bar <= 4.0:
        score += 30
    elif -2.0 <= recent_3bar <= 5.0:
        score += 20
    elif recent_3bar >= -5.0:
        score += 10

    # 캔들 품질
    if close_position >= 0.85:
        score += 35
    elif close_position >= 0.70:
        score += 25
    elif close_position >= 0.55:
        score += 15

    if upper_wick_ratio <= 0.15:
        score += 30
    elif upper_wick_ratio <= 0.30:
        score += 20
    elif upper_wick_ratio <= 0.45:
        score += 10

    # 캔들 상승률
    if 1.0 <= candle_change <= 3.5:
        score += 25
    elif 0.2 <= candle_change <= 4.0:
        score += 15
    elif candle_change <= 5.0:
        score += 5

    # MA5 상승
    if ma5_up:
        score += 20

    # 음봉 과도 방지
    if bullish_ok:
        score += 15

    return score


# =========================
# 9. CANDIDATE METRICS
# =========================

def calculate_candidate_metrics(market, df, alert_mode):
    df = prepare_indicators(df)

    idx = get_signal_index(df, alert_mode)
    if idx is None or idx < 30:
        return None

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]

    open_price = safe_float(row["open"])
    high_price = safe_float(row["high"])
    low_price = safe_float(row["low"])
    close_price = safe_float(row["close"])
    value = safe_float(row["value"])

    ma20 = safe_float(row["ma20"])
    ma5 = safe_float(row["ma5"])
    prev_ma5 = safe_float(prev["ma5"])

    if open_price <= 0 or high_price <= 0 or low_price <= 0 or close_price <= 0:
        return None

    if ma20 <= 0:
        return None

    value_ma20_prev = safe_float(row["value_ma20_prev"])
    if value_ma20_prev <= 0:
        return None

    volume_ratio = value / value_ma20_prev

    short_k = safe_float(row["short_k"])
    short_d = safe_float(row["short_d"])
    mid_k = safe_float(row["mid_k"])

    ma20_dev = percent(close_price, ma20)

    close_3ago = safe_float(df["close"].iloc[idx - 3])
    recent_3bar = percent(close_price, close_3ago)

    candle_change = percent(close_price, open_price)

    candle_range = high_price - low_price
    if candle_range <= 0:
        close_position = 0.5
        upper_wick_ratio = 0.5
    else:
        close_position = (close_price - low_price) / candle_range
        upper_wick_ratio = (high_price - close_price) / candle_range

    ma5_up = ma5 > prev_ma5 if is_number(ma5) and is_number(prev_ma5) else False

    # 강한 음봉 제외: close >= 0.995 * open
    bullish_ok = close_price >= open_price * 0.995

    metrics = {
        "market": market,
        "price": close_price,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "value": value,

        "volume_ratio": volume_ratio,
        "short_k": short_k,
        "short_d": short_d,
        "mid_k": mid_k,
        "ma20_dev": ma20_dev,
        "recent_3bar": recent_3bar,
        "close_position": close_position,
        "upper_wick_ratio": upper_wick_ratio,
        "candle_change": candle_change,
        "ma5_up": ma5_up,
        "bullish_ok": bullish_ok,
    }

    score = calculate_base_score(metrics, alert_mode)
    metrics["score"] = score

    return metrics


# =========================
# 10. FILTERS
# =========================

def pass_preliminary_filter(c):
    """
    예비 알림:
    - score >= 200
    - volume_ratio >= 1.3
    - K <= 70
    - D <= 60
    - MA20 -6% ~ +6%
    - 최근 3봉 >= -5%
    - 캔들 품질
    - MA5 상승
    """
    if c["score"] < 200:
        return False

    if c["volume_ratio"] < 1.3:
        return False

    if c["short_k"] > 70:
        return False

    if c["short_d"] > 60:
        return False

    if not (-6.0 <= c["ma20_dev"] <= 6.0):
        return False

    if c["recent_3bar"] < -5.0:
        return False

    if c["close_position"] < 0.55:
        return False

    if c["upper_wick_ratio"] > 0.45:
        return False

    if c["candle_change"] > 5.0:
        return False

    if not c["bullish_ok"]:
        return False

    if not c["ma5_up"]:
        return False

    return True


def pass_confirmed_filter(c):
    """
    확정 알림:
    - score >= 210
    - volume_ratio >= 1.5
    - K 20~65
    - D <= 55
    - MA20 -5% ~ +5%
    - 최근 3봉 >= -4%
    - 종가위치 >= 0.55
    - 윗꼬리 <= 0.45
    - 캔들 상승 <= 4%
    - MA5 상승
    """
    if c["score"] < 210:
        return False

    if c["volume_ratio"] < 1.5:
        return False

    if not (20 <= c["short_k"] <= 65):
        return False

    if c["short_d"] > 55:
        return False

    if not (-5.0 <= c["ma20_dev"] <= 5.0):
        return False

    if c["recent_3bar"] < -4.0:
        return False

    if c["close_position"] < 0.55:
        return False

    if c["upper_wick_ratio"] > 0.45:
        return False

    if c["candle_change"] > 4.0:
        return False

    if not c["bullish_ok"]:
        return False

    if not c["ma5_up"]:
        return False

    return True


# =========================
# 11. V3.3 ALERT GRADING
# =========================

def calculate_alert_grade(candidate):
    """
    V3.3 텔레그램 표시용 등급 계산.
    기존 매수 필터를 바꾸는 것이 아니라,
    알림 내 우선순위와 보기 편한 등급만 정리한다.
    """

    score = safe_float(get_value(candidate, ["score", "signal_score", "점수"], 0))
    volume_ratio = safe_float(get_value(candidate, ["volume_ratio", "vol_ratio", "turnover_ratio", "거래대금비율"], 0))
    short_k = safe_float(get_value(candidate, ["short_k", "k", "stoch_k", "K"], 100))
    short_d = safe_float(get_value(candidate, ["short_d", "d", "stoch_d", "D"], 100))
    mid_k = safe_float(get_value(candidate, ["mid_k", "middle_k", "중기K"], 100))
    ma20_dev = safe_float(get_value(candidate, ["ma20_dev", "ma20_deviation", "ma20_gap", "MA20"], 999))
    close_position = safe_float(get_value(candidate, ["close_position", "close_pos", "종가위치"], 0))
    upper_wick_ratio = safe_float(get_value(candidate, ["upper_wick_ratio", "upper_wick", "윗꼬리"], 1))
    candle_change = safe_float(get_value(candidate, ["candle_change", "candle_rise", "candle_pct", "캔들"], 0))

    quality_score = 0

    # 1) 종가위치
    if close_position >= 0.90:
        quality_score += 30
    elif close_position >= 0.80:
        quality_score += 20
    elif close_position >= 0.70:
        quality_score += 10
    elif close_position < 0.60:
        quality_score -= 15

    # 2) 윗꼬리
    if upper_wick_ratio <= 0.10:
        quality_score += 30
    elif upper_wick_ratio <= 0.20:
        quality_score += 20
    elif upper_wick_ratio <= 0.30:
        quality_score += 10
    elif upper_wick_ratio > 0.40:
        quality_score -= 20

    # 3) 거래대금 증가
    if volume_ratio >= 2.0:
        quality_score += 25
    elif volume_ratio >= 1.7:
        quality_score += 20
    elif volume_ratio >= 1.5:
        quality_score += 10
    elif volume_ratio < 1.3:
        quality_score -= 20

    # 4) MA20 이격
    if -2.0 <= ma20_dev <= 2.0:
        quality_score += 20
    elif -3.0 <= ma20_dev <= 3.0:
        quality_score += 10
    elif ma20_dev > 4.0:
        quality_score -= 20

    # 5) 캔들 상승률
    if 1.0 <= candle_change <= 3.5:
        quality_score += 20
    elif 0.3 <= candle_change < 1.0:
        quality_score += 8
    elif candle_change > 4.0:
        quality_score -= 20

    # 6) Stoch RSI
    if short_k <= 45 and short_d <= 45:
        quality_score += 15
    elif short_k <= 60 and short_d <= 55:
        quality_score += 8
    elif short_k > 70 or short_d > 60:
        quality_score -= 20

    # 7) 중기 K
    if mid_k <= 50:
        quality_score += 8
    elif mid_k > 65:
        quality_score -= 12

    alert_score = score + quality_score

    # A급 조건
    is_a_grade = (
        score >= 230
        and volume_ratio >= 1.70
        and close_position >= 0.85
        and upper_wick_ratio <= 0.15
        and -2.0 <= ma20_dev <= 2.0
        and 1.0 <= candle_change <= 3.5
        and short_k <= 55
    )

    # B급 조건
    is_b_grade = (
        score >= 220
        and volume_ratio >= 1.50
        and close_position >= 0.70
        and upper_wick_ratio <= 0.30
        and -3.0 <= ma20_dev <= 3.0
        and candle_change <= 4.0
    )

    if is_a_grade:
        grade = "A"
        grade_label = "🔥 A급"
    elif is_b_grade:
        grade = "B"
        grade_label = "🟡 B급"
    else:
        grade = "WATCH"
        grade_label = "👀 관찰"

    candidate["alert_quality_score"] = quality_score
    candidate["alert_score"] = alert_score
    candidate["alert_grade"] = grade
    candidate["alert_grade_label"] = grade_label

    return candidate


def apply_alert_grades(candidates):
    graded = []
    for candidate in candidates:
        graded.append(calculate_alert_grade(candidate))

    grade_order = {
        "A": 0,
        "B": 1,
        "WATCH": 2
    }

    graded.sort(
        key=lambda x: (
            grade_order.get(x.get("alert_grade", "WATCH"), 9),
            -safe_float(x.get("alert_score", 0)),
            -safe_float(x.get("score", 0)),
            -safe_float(x.get("volume_ratio", 0)),
            -safe_float(x.get("close_position", 0)),
            safe_float(x.get("upper_wick_ratio", 1)),
        )
    )

    return graded


# =========================
# 12. MESSAGE FORMAT
# =========================

def format_price(price):
    price = safe_float(price)
    if price < 1:
        return f"{price:.8f}".rstrip("0").rstrip(".")
    if price < 10:
        return f"{price:.4f}"
    if price < 100:
        return f"{price:.3f}"
    return f"{price:.0f}"


def format_candidate_line(index, candidate):
    market = candidate.get("market") or candidate.get("ticker") or candidate.get("symbol") or "UNKNOWN"

    score = safe_float(candidate.get("score", 0))
    volume_ratio = safe_float(candidate.get("volume_ratio", 0))
    short_k = safe_float(candidate.get("short_k", 0))
    short_d = safe_float(candidate.get("short_d", 0))
    mid_k = safe_float(candidate.get("mid_k", 0))
    ma20_dev = safe_float(candidate.get("ma20_dev", 0))
    recent_3bar = safe_float(candidate.get("recent_3bar", 0))
    close_position = safe_float(candidate.get("close_position", 0))
    upper_wick_ratio = safe_float(candidate.get("upper_wick_ratio", 0))
    candle_change = safe_float(candidate.get("candle_change", 0))
    price = safe_float(candidate.get("price", 0))

    grade_label = candidate.get("alert_grade_label", "👀 관찰")
    alert_score = safe_float(candidate.get("alert_score", score))
    quality_score = safe_float(candidate.get("alert_quality_score", 0))

    return (
        f"{index}. {market} {grade_label}\n"
        f"점수 {score:.0f} / 보정 {alert_score:.0f} / 품질 {quality_score:+.0f}\n"
        f"거래대금 x{volume_ratio:.2f} / K {short_k:.2f} D {short_d:.2f} / 중기K {mid_k:.2f}\n"
        f"MA20 {ma20_dev:.2f}% / 3봉 {recent_3bar:.2f}% / 종가위치 {close_position:.2f} / 윗꼬리 {upper_wick_ratio:.2f}\n"
        f"캔들 {candle_change:.2f}% / 가격 {format_price(price)}\n"
    )


def build_telegram_message(candidates, alert_mode, scan_count, btc_info):
    candidates = apply_alert_grades(candidates)

    if alert_mode == "preliminary":
        title = "🟡 4H 마감 1시간 전 상승전조 후보"
    else:
        title = "🟢 4H 마감 후 상승전환 확정 후보"

    lines = []
    lines.append(title)
    lines.append(f"시간: {format_now()}")
    lines.append(f"스캔: 전체 KRW {scan_count}개")
    lines.append(f"후보: {len(candidates)}개")

    if btc_info:
        btc_close = safe_float(btc_info.get("btc_close", 0))
        btc_ma20 = safe_float(btc_info.get("btc_ma20", 0))
        btc_3bar = safe_float(btc_info.get("btc_3bar", 0))
        btc_status = "통과" if btc_info.get("pass") else "차단"
        lines.append(
            f"BTC필터: {btc_status} / BTC {format_price(btc_close)} / MA20 {format_price(btc_ma20)} / 3봉 {btc_3bar:.2f}%"
        )

    lines.append("")

    a_candidates = [c for c in candidates if c.get("alert_grade") == "A"]
    b_candidates = [c for c in candidates if c.get("alert_grade") == "B"]
    watch_candidates = [c for c in candidates if c.get("alert_grade") == "WATCH"]

    index = 1

    if a_candidates:
        lines.append("🔥 A급 우선 후보")
        for candidate in a_candidates:
            lines.append(format_candidate_line(index, candidate))
            index += 1

    if b_candidates:
        lines.append("🟡 B급 후보")
        for candidate in b_candidates:
            lines.append(format_candidate_line(index, candidate))
            index += 1

    if watch_candidates:
        lines.append("👀 관찰 후보")
        for candidate in watch_candidates:
            lines.append(format_candidate_line(index, candidate))
            index += 1

    lines.append("※ 자동매매 아님. 진입 전 호가/거래대금/BTC 상태 확인 필요.")
    lines.append("※ A급은 우선 검토 대상일 뿐, 매수 확정 신호가 아닙니다.")
    lines.append("※ 손절 기준과 BTC 4H 흐름을 반드시 확인하세요.")

    return "\n".join(lines)


def build_empty_message(alert_mode, scan_count, btc_info):
    if alert_mode == "preliminary":
        title = "🟡 4H 마감 1시간 전 상승전조 후보"
    else:
        title = "🟢 4H 마감 후 상승전환 확정 후보"

    lines = []
    lines.append(title)
    lines.append(f"시간: {format_now()}")
    lines.append(f"스캔: 전체 KRW {scan_count}개")
    lines.append("후보: 0개")
    lines.append("")

    if btc_info:
        btc_close = safe_float(btc_info.get("btc_close", 0))
        btc_ma20 = safe_float(btc_info.get("btc_ma20", 0))
        btc_3bar = safe_float(btc_info.get("btc_3bar", 0))
        btc_status = "통과" if btc_info.get("pass") else "차단"
        lines.append(
            f"BTC필터: {btc_status} / BTC {format_price(btc_close)} / MA20 {format_price(btc_ma20)} / 3봉 {btc_3bar:.2f}%"
        )
        lines.append("")

    lines.append("조건 만족 종목 없음")
    lines.append("※ 현재 조건에서는 무리한 매수보다 관망이 우선입니다.")

    return "\n".join(lines)


def build_error_message(error_text):
    return (
        "🔴 Telegram Coin Alert 오류 발생\n"
        f"시간: {format_now()}\n"
        f"오류:\n{error_text}"
    )


# =========================
# 13. SCANNER
# =========================

def scan_markets(alert_mode, btc_info):
    markets = get_krw_markets()
    scan_count = len(markets)

    candidates = []

    # BTC 필터 실패 시 전체 차단.
    if not btc_info.get("pass", False):
        return candidates, scan_count

    for i, market in enumerate(markets, start=1):
        try:
            # BTC 자체는 시장 필터용으로만 사용하고 후보에서는 제외.
            if market == "KRW-BTC":
                continue

            # 스테이블/법정화폐성 종목 제외.
            if market in ["KRW-USDT", "KRW-USDC"]:
                continue

            df = get_ohlcv(market, interval=ALERT_INTERVAL, count=ALERT_CANDLE_COUNT)
            if df is None or len(df) < 60:
                time.sleep(REQUEST_DELAY)
                continue

            metrics = calculate_candidate_metrics(market, df, alert_mode)
            if metrics is None:
                time.sleep(REQUEST_DELAY)
                continue

            if alert_mode == "confirmed":
                passed = pass_confirmed_filter(metrics)
            else:
                passed = pass_preliminary_filter(metrics)

            if passed:
                candidates.append(metrics)

            time.sleep(REQUEST_DELAY)

        except Exception:
            # 개별 종목 오류는 전체 실행을 막지 않는다.
            time.sleep(REQUEST_DELAY)
            continue

    candidates = apply_alert_grades(candidates)

    if MAX_ALERT_COUNT > 0:
        candidates = candidates[:MAX_ALERT_COUNT]

    return candidates, scan_count


# =========================
# 14. MAIN
# =========================

def main():
    print("========================================")
    print("Telegram Coin Alert Bot V3.3 started")
    print(f"Time: {format_now()}")
    print(f"Interval: {ALERT_INTERVAL}")
    print(f"Candle count: {ALERT_CANDLE_COUNT}")
    print(f"Request delay: {REQUEST_DELAY}")
    print(f"Max alert count: {MAX_ALERT_COUNT}")
    print(f"Send empty alert: {SEND_EMPTY_ALERT}")
    print("========================================")

    alert_mode = detect_alert_mode()
    print(f"Alert mode: {alert_mode}")

    btc_info = get_btc_filter(alert_mode)
    print(f"BTC filter: {btc_info}")

    candidates, scan_count = scan_markets(alert_mode, btc_info)
    print(f"Scan count: {scan_count}")
    print(f"Candidate count: {len(candidates)}")

    if candidates:
        message = build_telegram_message(
            candidates=candidates,
            alert_mode=alert_mode,
            scan_count=scan_count,
            btc_info=btc_info
        )
        print(message)
        send_telegram_message(message)

    else:
        if SEND_EMPTY_ALERT:
            message = build_empty_message(
                alert_mode=alert_mode,
                scan_count=scan_count,
                btc_info=btc_info
            )
            print(message)
            send_telegram_message(message)
        else:
            print("No candidates and SEND_EMPTY_ALERT=false. Telegram message skipped.")

    print("Telegram Coin Alert Bot V3.3 finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_text = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        print(error_text)

        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                send_telegram_message(build_error_message(error_text[:3000]))
        except Exception as telegram_error:
            print(f"Telegram error message failed: {telegram_error}")

        raise
