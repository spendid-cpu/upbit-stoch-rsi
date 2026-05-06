import os
import time
import json
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# =========================================================
# Telegram Coin Alert Bot V3.7
# ---------------------------------------------------------
# 핵심:
# - 4H 확정봉만 분석
# - 진행 중인 최신봉 제거
# - A/B 등급만 알림 및 추적
# - WATCH 등급 제외
# - TP +5%, SL -4%, 48시간 추적
# - 분석 기준봉 시작/마감/실행시간 표시
# - 백테스트 V1.1과 최대한 동일한 필터/점수/정렬 기준 적용
# =========================================================

UPBIT_BASE_URL = "https://api.upbit.com/v1"

# =========================================================
# ENV CONFIG
# =========================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

ALERT_INTERVAL = os.getenv("ALERT_INTERVAL", "minute240").strip()
ALERT_CANDLE_COUNT = int(os.getenv("ALERT_CANDLE_COUNT", "500"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.08"))
MAX_ALERT_COUNT = int(os.getenv("MAX_ALERT_COUNT", "10"))
SEND_EMPTY_ALERT = os.getenv("SEND_EMPTY_ALERT", "true").strip().lower() == "true"

ENABLE_SIGNAL_TRACKING = os.getenv("ENABLE_SIGNAL_TRACKING", "true").strip().lower() == "true"
SIGNAL_STATE_FILE = os.getenv("SIGNAL_STATE_FILE", "active_signals.json").strip()

TAKE_PROFIT_RATE = float(os.getenv("TAKE_PROFIT_RATE", "0.05"))
STOP_LOSS_RATE = float(os.getenv("STOP_LOSS_RATE", "0.04"))
MAX_HOLD_HOURS = int(os.getenv("MAX_HOLD_HOURS", "48"))
SIGNAL_COOLDOWN_HOURS = int(os.getenv("SIGNAL_COOLDOWN_HOURS", "12"))

MIN_4H_TRADE_VALUE = float(os.getenv("MIN_4H_TRADE_VALUE", "150000000"))
MIN_CANDLE_CHANGE = float(os.getenv("MIN_CANDLE_CHANGE", "0.7"))

# 알림/추적 허용 등급
ALLOWED_ALERT_GRADES = ["A", "B"]

# 4H 기준
CANDLE_HOURS = 4


# =========================================================
# TIME UTILS
# =========================================================
def now_utc():
    return datetime.now(timezone.utc)


def now_kst():
    return now_utc() + timedelta(hours=9)


def to_kst(dt):
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt + timedelta(hours=9)


def fmt_kst(dt):
    return to_kst(dt).strftime("%Y-%m-%d %H:%M:%S KST")


def parse_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def pct(a, b):
    if b == 0:
        return 0.0
    return (a / b - 1.0) * 100.0


def price_fmt(v):
    v = safe_float(v)
    if v >= 100000:
        return f"{v:,.0f}"
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:.1f}"
    if v >= 10:
        return f"{v:.2f}"
    if v >= 1:
        return f"{v:.3f}"
    return f"{v:.6f}".rstrip("0").rstrip(".")


# =========================================================
# TELEGRAM
# =========================================================
def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/chat_id missing. Message not sent.")
        print(text)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 3900
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]

    ok = True

    for chunk in chunks:
        try:
            res = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if res.status_code != 200:
                print(f"Telegram send failed: {res.status_code} {res.text}")
                ok = False
            time.sleep(0.4)
        except Exception as e:
            print(f"Telegram exception: {e}")
            ok = False

    return ok


# =========================================================
# UPBIT API
# =========================================================
def upbit_get(path, params=None, retry=3):
    url = f"{UPBIT_BASE_URL}{path}"

    for attempt in range(retry):
        try:
            res = requests.get(url, params=params, timeout=15)
            if res.status_code == 200:
                return res.json()

            print(f"Upbit API error {res.status_code}: {res.text[:200]}")
            time.sleep(0.5 + attempt)
        except Exception as e:
            print(f"Upbit API exception: {e}")
            time.sleep(0.5 + attempt)

    return None


def get_krw_markets():
    data = upbit_get("/market/all", params={"isDetails": "false"})
    if not data:
        return []

    markets = []

    for item in data:
        market = item.get("market", "")
        if not market.startswith("KRW-"):
            continue

        if market in ["KRW-BTC", "KRW-USDT", "KRW-USDC"]:
            continue

        markets.append(market)

    return sorted(markets)


def fetch_4h_candles(market, count=ALERT_CANDLE_COUNT):
    data = upbit_get(
        "/candles/minutes/240",
        params={
            "market": market,
            "count": count,
        },
    )

    time.sleep(REQUEST_DELAY)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame()

    df["dt_utc"] = pd.to_datetime(df["candle_date_time_utc"], utc=True)

    df = df.rename(
        columns={
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_price": "value",
            "candle_acc_trade_volume": "volume",
        }
    )

    needed = ["dt_utc", "open", "high", "low", "close", "value", "volume"]

    for col in needed:
        if col not in df.columns:
            print(f"{market}: missing column {col}")
            return pd.DataFrame()

    for col in ["open", "high", "low", "close", "value", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("dt_utc").reset_index(drop=True)

    return df[needed]


def fetch_tickers(markets):
    if not markets:
        return {}

    result = {}

    # Upbit ticker는 쉼표로 여러 마켓 조회 가능
    chunk_size = 100

    for i in range(0, len(markets), chunk_size):
        chunk = markets[i:i + chunk_size]
        data = upbit_get("/ticker", params={"markets": ",".join(chunk)})
        time.sleep(0.1)

        if not data:
            continue

        for item in data:
            market = item.get("market")
            result[market] = item

    return result


def get_current_prices(markets):
    tickers = fetch_tickers(markets)
    prices = {}

    for market, item in tickers.items():
        prices[market] = safe_float(item.get("trade_price"))

    return prices


# =========================================================
# CONFIRMED CANDLE HANDLING
# =========================================================
def remove_incomplete_current_candle(df):
    """
    핵심:
    Upbit 4H API 최신봉은 진행 중인 봉일 수 있음.
    현재 시간이 최신봉 시작 + 4시간 이전이면 최신봉 제거.
    """
    if df.empty:
        return df, None

    latest_start = df.iloc[-1]["dt_utc"]

    if isinstance(latest_start, pd.Timestamp):
        latest_start_dt = latest_start.to_pydatetime()
    else:
        latest_start_dt = latest_start

    if latest_start_dt.tzinfo is None:
        latest_start_dt = latest_start_dt.replace(tzinfo=timezone.utc)

    latest_close_dt = latest_start_dt + timedelta(hours=CANDLE_HOURS)

    current = now_utc()

    removed = None

    if current < latest_close_dt:
        removed = {
            "start_utc": latest_start_dt,
            "close_utc": latest_close_dt,
            "start_kst": fmt_kst(latest_start_dt),
            "close_kst": fmt_kst(latest_close_dt),
        }
        df = df.iloc[:-1].copy()

    return df.reset_index(drop=True), removed


def get_analysis_candle_info_from_df(df):
    if df.empty:
        return {}

    target_start = df.iloc[-1]["dt_utc"]

    if isinstance(target_start, pd.Timestamp):
        target_start_dt = target_start.to_pydatetime()
    else:
        target_start_dt = target_start

    if target_start_dt.tzinfo is None:
        target_start_dt = target_start_dt.replace(tzinfo=timezone.utc)

    target_close_dt = target_start_dt + timedelta(hours=CANDLE_HOURS)

    return {
        "candle_start_utc": target_start_dt.isoformat(),
        "candle_close_utc": target_close_dt.isoformat(),
        "candle_start_kst": fmt_kst(target_start_dt),
        "candle_close_kst": fmt_kst(target_close_dt),
        "expected_alert_kst": (to_kst(target_close_dt) + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S KST"),
        "run_time_kst": now_kst().strftime("%Y-%m-%d %H:%M:%S KST"),
    }


def build_analysis_info_text(info):
    if not info:
        return "분석 기준봉: 확인 불가"

    lines = []
    lines.append("🕯 분석 기준봉")
    lines.append(f"시작: {info.get('candle_start_kst', '-')}")
    lines.append(f"마감: {info.get('candle_close_kst', '-')}")
    lines.append(f"실행: {info.get('run_time_kst', '-')}")
    lines.append(f"예상 알림 기준: {info.get('expected_alert_kst', '-')}")
    lines.append("※ 진행 중인 최신 4H 봉은 제외하고 직전 확정봉만 분석")
    return "\n".join(lines)


# =========================================================
# INDICATORS
# =========================================================
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def add_indicators(df):
    df = df.copy()

    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma5_up"] = df["ma5"] > df["ma5"].shift(1)

    df["rsi"] = calc_rsi(df["close"], 14)

    rsi_min = df["rsi"].rolling(14).min()
    rsi_max = df["rsi"].rolling(14).max()

    stoch_rsi = (df["rsi"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    df["short_k"] = stoch_rsi.rolling(3).mean().fillna(50)
    df["short_d"] = df["short_k"].rolling(3).mean().fillna(50)

    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()

    mid_k = (df["close"] - low14) / (high14 - low14).replace(0, np.nan) * 100
    df["mid_k"] = mid_k.rolling(3).mean().fillna(50)

    df["volume_avg20"] = df["value"].shift(1).rolling(20).mean()
    df["volume_ratio"] = df["value"] / df["volume_avg20"].replace(0, np.nan)

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)

    df["close_position"] = ((df["close"] - df["low"]) / candle_range).fillna(0.5)
    df["upper_wick_ratio"] = ((df["high"] - df["close"]) / candle_range).fillna(0.5)

    df["candle_change"] = (df["close"] / df["open"] - 1.0) * 100.0
    df["recent_3bar"] = (df["close"] / df["close"].shift(3) - 1.0) * 100.0
    df["ma20_dev"] = (df["close"] / df["ma20"] - 1.0) * 100.0

    return df


# =========================================================
# BTC FILTER
# =========================================================
def get_btc_bullish_ok():
    df = fetch_4h_candles("KRW-BTC", ALERT_CANDLE_COUNT)

    if df.empty:
        return True, {}, "BTC 데이터 없음 → 필터 통과 처리"

    df, removed = remove_incomplete_current_candle(df)

    if len(df) < 30:
        return True, get_analysis_candle_info_from_df(df), "BTC 캔들 부족 → 필터 통과 처리"

    df = add_indicators(df)
    row = df.iloc[-1]

    candle_change = safe_float(row.get("candle_change"))
    ma20_dev = safe_float(row.get("ma20_dev"))

    bullish_ok = True
    reasons = []

    if candle_change <= -2.0:
        bullish_ok = False
        reasons.append(f"BTC 4H 급락 {candle_change:.2f}%")

    if ma20_dev <= -4.0:
        bullish_ok = False
        reasons.append(f"BTC MA20 이탈 {ma20_dev:.2f}%")

    if bullish_ok:
        status = f"BTC 필터 통과 / 4H {candle_change:+.2f}% / MA20 {ma20_dev:+.2f}%"
    else:
        status = "BTC 필터 차단 / " + ", ".join(reasons)

    return bullish_ok, get_analysis_candle_info_from_df(df), status


# =========================================================
# SCORE / FILTER / GRADE
# =========================================================
def calculate_base_score(c):
    score = 100

    volume_ratio = safe_float(c.get("volume_ratio"))
    value = safe_float(c.get("value"))
    short_k = safe_float(c.get("short_k"), 50)
    short_d = safe_float(c.get("short_d"), 50)
    mid_k = safe_float(c.get("mid_k"), 50)
    ma20_dev = safe_float(c.get("ma20_dev"))
    recent_3bar = safe_float(c.get("recent_3bar"))
    candle_change = safe_float(c.get("candle_change"))
    close_position = safe_float(c.get("close_position"))
    upper_wick_ratio = safe_float(c.get("upper_wick_ratio"))
    ma5_up = bool(c.get("ma5_up"))
    bullish_ok = bool(c.get("bullish_ok", True))

    # 상대 거래대금
    if volume_ratio >= 3.0:
        score += 50
    elif volume_ratio >= 2.0:
        score += 40
    elif volume_ratio >= 1.7:
        score += 30
    elif volume_ratio >= 1.5:
        score += 20
    else:
        score -= 30

    # 절대 4H 거래대금
    if value >= 500_000_000:
        score += 30
    elif value >= 300_000_000:
        score += 20
    elif value >= 150_000_000:
        score += 10
    else:
        score -= 30

    # Stoch RSI
    if 20 <= short_k <= 55 and short_d <= 55:
        score += 30
    elif 20 <= short_k <= 70 and short_d <= 60:
        score += 15
    elif short_k > 75 or short_d > 70:
        score -= 25

    # 중기 K
    if mid_k <= 55:
        score += 10
    elif mid_k > 70:
        score -= 15

    # MA20 이격
    if -2.0 <= ma20_dev <= 2.0:
        score += 25
    elif -5.0 <= ma20_dev <= 5.0:
        score += 10
    else:
        score -= 25

    # 캔들 상승률
    if 1.0 <= candle_change <= 3.5:
        score += 25
    elif 0.7 <= candle_change <= 4.0:
        score += 15
    elif candle_change > 4.0:
        score -= 30

    # 최근 3봉
    if -1.0 <= recent_3bar <= 4.0:
        score += 15
    elif -4.0 <= recent_3bar <= 6.0:
        score += 5
    elif recent_3bar > 6.0:
        score -= 25

    # 캔들 품질
    if close_position >= 0.85:
        score += 20
    elif close_position >= 0.60:
        score += 10
    else:
        score -= 20

    if upper_wick_ratio <= 0.15:
        score += 20
    elif upper_wick_ratio <= 0.35:
        score += 10
    else:
        score -= 25

    if ma5_up:
        score += 10
    else:
        score -= 10

    if bullish_ok:
        score += 10
    else:
        score -= 40

    return int(score)


def calculate_alert_grade(c):
    score = safe_float(c.get("score"))
    value = safe_float(c.get("value"))
    volume_ratio = safe_float(c.get("volume_ratio"))
    close_position = safe_float(c.get("close_position"))
    upper_wick_ratio = safe_float(c.get("upper_wick_ratio"))
    ma20_dev = safe_float(c.get("ma20_dev"))
    candle_change = safe_float(c.get("candle_change"))
    short_k = safe_float(c.get("short_k"), 50)
    short_d = safe_float(c.get("short_d"), 50)

    # A-grade
    if (
        score >= 230
        and value >= 200_000_000
        and volume_ratio >= 1.70
        and close_position >= 0.85
        and upper_wick_ratio <= 0.15
        and -2.0 <= ma20_dev <= 2.0
        and 1.0 <= candle_change <= 3.5
        and short_k <= 55
        and short_d <= 55
    ):
        return "A", "🔥 A급"

    # B-grade
    if (
        score >= 220
        and value >= 150_000_000
        and volume_ratio >= 1.50
        and close_position >= 0.70
        and upper_wick_ratio <= 0.30
        and -3.0 <= ma20_dev <= 3.0
        and 0.7 <= candle_change <= 4.0
        and short_k <= 70
        and short_d <= 60
    ):
        return "B", "🟡 B급"

    return "WATCH", "👀 관찰"


def pass_confirmed_filter(c):
    score = safe_float(c.get("score"))
    value = safe_float(c.get("value"))
    volume_ratio = safe_float(c.get("volume_ratio"))
    short_k = safe_float(c.get("short_k"), 50)
    short_d = safe_float(c.get("short_d"), 50)
    mid_k = safe_float(c.get("mid_k"), 50)
    ma20_dev = safe_float(c.get("ma20_dev"))
    recent_3bar = safe_float(c.get("recent_3bar"))
    close_position = safe_float(c.get("close_position"))
    upper_wick_ratio = safe_float(c.get("upper_wick_ratio"))
    candle_change = safe_float(c.get("candle_change"))
    ma5_up = bool(c.get("ma5_up"))
    bullish_ok = bool(c.get("bullish_ok", True))

    if score < 210:
        return False

    if value < MIN_4H_TRADE_VALUE:
        return False

    if volume_ratio < 1.5:
        return False

    if not (20 <= short_k <= 70):
        return False

    if short_d > 60:
        return False

    if mid_k > 70:
        return False

    if not (-5.0 <= ma20_dev <= 5.0):
        return False

    if not (-4.0 <= recent_3bar <= 6.0):
        return False

    if close_position < 0.60:
        return False

    if upper_wick_ratio > 0.35:
        return False

    if not (MIN_CANDLE_CHANGE <= candle_change <= 4.0):
        return False

    if not ma5_up:
        return False

    if not bullish_ok:
        return False

    return True


def make_candidate_from_row(market, row, bullish_ok, analysis_info):
    c = {
        "market": market,
        "dt_utc": row["dt_utc"],
        "open": safe_float(row["open"]),
        "high": safe_float(row["high"]),
        "low": safe_float(row["low"]),
        "close": safe_float(row["close"]),
        "value": safe_float(row["value"]),
        "volume": safe_float(row["volume"]),
        "volume_ratio": safe_float(row.get("volume_ratio")),
        "short_k": safe_float(row.get("short_k"), 50),
        "short_d": safe_float(row.get("short_d"), 50),
        "mid_k": safe_float(row.get("mid_k"), 50),
        "ma20_dev": safe_float(row.get("ma20_dev")),
        "recent_3bar": safe_float(row.get("recent_3bar")),
        "close_position": safe_float(row.get("close_position")),
        "upper_wick_ratio": safe_float(row.get("upper_wick_ratio")),
        "candle_change": safe_float(row.get("candle_change")),
        "ma5_up": bool(row.get("ma5_up")),
        "bullish_ok": bool(bullish_ok),
        "analysis_info": analysis_info or {},
    }

    c["score"] = calculate_base_score(c)

    grade, grade_label = calculate_alert_grade(c)
    c["alert_grade"] = grade
    c["alert_grade_label"] = grade_label

    return c


def grade_order_value(grade):
    return {
        "A": 0,
        "B": 1,
        "WATCH": 2,
    }.get(grade, 9)


def sort_candidates(candidates):
    return sorted(
        candidates,
        key=lambda x: (
            grade_order_value(x.get("alert_grade")),
            -safe_float(x.get("score")),
            -safe_float(x.get("value")),
            -safe_float(x.get("volume_ratio")),
        ),
    )


# =========================================================
# SIGNAL STATE
# =========================================================
def default_signal_state():
    return {
        "active": [],
        "closed": [],
    }


def load_signal_state():
    if not ENABLE_SIGNAL_TRACKING:
        return default_signal_state()

    if not os.path.exists(SIGNAL_STATE_FILE):
        return default_signal_state()

    try:
        with open(SIGNAL_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 구버전 리스트 형태 호환
        if isinstance(data, list):
            return {
                "active": data,
                "closed": [],
            }

        if not isinstance(data, dict):
            return default_signal_state()

        data.setdefault("active", [])
        data.setdefault("closed", [])

        return data

    except Exception as e:
        print(f"load_signal_state error: {e}")
        return default_signal_state()


def save_signal_state(state):
    if not ENABLE_SIGNAL_TRACKING:
        return

    try:
        with open(SIGNAL_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"Saved signal state: {SIGNAL_STATE_FILE}")
    except Exception as e:
        print(f"save_signal_state error: {e}")


def parse_signal_time(signal):
    for key in ["alert_time_utc", "created_at_utc", "signal_time_utc"]:
        value = signal.get(key)
        if value:
            dt = parse_dt(value)
            if dt:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
    return now_utc()


def is_duplicate_or_cooldown(state, market):
    current = now_utc()

    for s in state.get("active", []):
        if s.get("market") == market and s.get("status", "active") == "active":
            return True

    # 최근 closed 포함 쿨다운
    all_signals = state.get("active", []) + state.get("closed", [])

    for s in all_signals:
        if s.get("market") != market:
            continue

        dt = parse_signal_time(s)
        diff_h = (current - dt).total_seconds() / 3600.0

        if diff_h < SIGNAL_COOLDOWN_HOURS:
            return True

    return False


def add_new_signals_to_state(state, candidates):
    if not ENABLE_SIGNAL_TRACKING:
        return 0

    added = 0

    for c in candidates:
        grade = c.get("alert_grade")

        # V3.7 핵심: A/B만 저장, WATCH 제외
        if grade not in ALLOWED_ALERT_GRADES:
            continue

        market = c.get("market")
        if not market:
            continue

        if is_duplicate_or_cooldown(state, market):
            continue

        entry_price = safe_float(c.get("close"))
        if entry_price <= 0:
            continue

        info = c.get("analysis_info") or {}

        signal = {
            "market": market,
            "entry_price": entry_price,
            "tp_price": entry_price * (1.0 + TAKE_PROFIT_RATE),
            "sl_price": entry_price * (1.0 - STOP_LOSS_RATE),
            "take_profit_rate": TAKE_PROFIT_RATE,
            "stop_loss_rate": STOP_LOSS_RATE,
            "max_hold_hours": MAX_HOLD_HOURS,
            "alert_time_utc": now_utc().isoformat(),
            "alert_time_kst": now_kst().strftime("%Y-%m-%d %H:%M:%S KST"),
            "signal_candle_start_kst": info.get("candle_start_kst"),
            "signal_candle_close_kst": info.get("candle_close_kst"),
            "grade": grade,
            "grade_label": c.get("alert_grade_label"),
            "score": c.get("score"),
            "value": c.get("value"),
            "volume_ratio": c.get("volume_ratio"),
            "candle_change": c.get("candle_change"),
            "ma20_dev": c.get("ma20_dev"),
            "recent_3bar": c.get("recent_3bar"),
            "status": "active",
        }

        state.setdefault("active", []).append(signal)
        added += 1

    return added


def check_active_signals(state):
    if not ENABLE_SIGNAL_TRACKING:
        return []

    active = state.get("active", [])
    if not active:
        return []

    markets = sorted(list({s.get("market") for s in active if s.get("market")}))
    prices = get_current_prices(markets)

    current = now_utc()
    closed_events = []
    still_active = []

    for s in active:
        if s.get("status", "active") != "active":
            continue

        market = s.get("market")
        entry_price = safe_float(s.get("entry_price"))
        current_price = prices.get(market, 0.0)

        if not market or entry_price <= 0 or current_price <= 0:
            still_active.append(s)
            continue

        tp_price = safe_float(s.get("tp_price"), entry_price * (1.0 + TAKE_PROFIT_RATE))
        sl_price = safe_float(s.get("sl_price"), entry_price * (1.0 - STOP_LOSS_RATE))

        created = parse_signal_time(s)
        hold_hours = (current - created).total_seconds() / 3600.0

        return_rate = pct(current_price, entry_price)

        event_type = None

        if current_price >= tp_price:
            event_type = "TP"
        elif current_price <= sl_price:
            event_type = "SL"
        elif hold_hours >= MAX_HOLD_HOURS:
            event_type = "EXPIRED"

        if event_type:
            s["status"] = "closed"
            s["result"] = event_type
            s["exit_price"] = current_price
            s["exit_time_utc"] = current.isoformat()
            s["exit_time_kst"] = now_kst().strftime("%Y-%m-%d %H:%M:%S KST")
            s["return_rate"] = return_rate
            s["hold_hours"] = hold_hours

            state.setdefault("closed", []).append(s)

            closed_events.append({
                "type": event_type,
                "market": market,
                "entry_price": entry_price,
                "exit_price": current_price,
                "return_rate": return_rate,
                "hold_hours": hold_hours,
                "grade_label": s.get("grade_label", ""),
                "alert_time_kst": s.get("alert_time_kst", ""),
                "tp_price": tp_price,
                "sl_price": sl_price,
            })
        else:
            s["last_price"] = current_price
            s["last_return_rate"] = return_rate
            s["last_checked_kst"] = now_kst().strftime("%Y-%m-%d %H:%M:%S KST")
            still_active.append(s)

    state["active"] = still_active

    return closed_events


def build_closed_events_message(events):
    if not events:
        return ""

    lines = []
    lines.append("📌 TP/SL/만료 알림")

    for idx, e in enumerate(events, 1):
        event_type = e["type"]

        if event_type == "TP":
            icon = "✅ 익절 도달"
        elif event_type == "SL":
            icon = "❌ 손절 도달"
        else:
            icon = "⏰ 48시간 만료"

        lines.append("")
        lines.append(f"{idx}. {icon} / {e['market']} {e.get('grade_label', '')}")
        lines.append(f"추천가: {price_fmt(e['entry_price'])} / 현재가: {price_fmt(e['exit_price'])}")
        lines.append(f"수익률: {e['return_rate']:+.2f}% / 경과: {e['hold_hours']:.1f}시간")
        lines.append(f"TP: {price_fmt(e['tp_price'])} / SL: {price_fmt(e['sl_price'])}")
        if e.get("alert_time_kst"):
            lines.append(f"추천시간: {e['alert_time_kst']}")

    lines.append("")
    lines.append("※ 실제 체결 기준이 아닌 현재가 기준 알림입니다.")

    return "\n".join(lines)


def build_active_signals_summary(state, max_items=10):
    if not ENABLE_SIGNAL_TRACKING:
        return ""

    active = state.get("active", [])
    if not active:
        return ""

    markets = sorted(list({s.get("market") for s in active if s.get("market")}))
    prices = get_current_prices(markets)

    rows = []
    current = now_utc()

    for s in active:
        market = s.get("market")
        entry_price = safe_float(s.get("entry_price"))
        current_price = prices.get(market, safe_float(s.get("last_price")))

        if not market or entry_price <= 0 or current_price <= 0:
            continue

        created = parse_signal_time(s)
        hold_hours = (current - created).total_seconds() / 3600.0
        return_rate = pct(current_price, entry_price)

        tp_price = safe_float(s.get("tp_price"), entry_price * (1.0 + TAKE_PROFIT_RATE))
        sl_price = safe_float(s.get("sl_price"), entry_price * (1.0 - STOP_LOSS_RATE))

        tp_gap = pct(tp_price, current_price)
        sl_gap = pct(sl_price, current_price)

        rows.append({
            "market": market,
            "entry_price": entry_price,
            "current_price": current_price,
            "return_rate": return_rate,
            "hold_hours": hold_hours,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "tp_gap": tp_gap,
            "sl_gap": sl_gap,
            "grade_label": s.get("grade_label", ""),
            "alert_time_kst": s.get("alert_time_kst", ""),
            "signal_candle_start_kst": s.get("signal_candle_start_kst", ""),
            "signal_candle_close_kst": s.get("signal_candle_close_kst", ""),
        })

    if not rows:
        return ""

    rows = sorted(rows, key=lambda x: x["return_rate"], reverse=True)[:max_items]

    lines = []
    lines.append("📊 48시간 추적 중 신호")
    lines.append(f"현재시간: {now_kst().strftime('%Y-%m-%d %H:%M:%S KST')}")

    for idx, r in enumerate(rows, 1):
        if r["return_rate"] >= 3:
            icon = "🟢"
        elif r["return_rate"] >= 0:
            icon = "⚪"
        elif r["return_rate"] <= -3:
            icon = "🔴"
        else:
            icon = "🟡"

        lines.append("")
        lines.append(f"{idx}. {r['market']} {icon} {r['grade_label']}")
        lines.append(f"추천가: {price_fmt(r['entry_price'])} / 현재가: {price_fmt(r['current_price'])}")
        lines.append(f"현재수익률: {r['return_rate']:+.2f}% / 경과: {r['hold_hours']:.1f}시간")
        lines.append(f"TP {price_fmt(r['tp_price'])}까지 {r['tp_gap']:+.2f}% / SL {price_fmt(r['sl_price'])}까지 {r['sl_gap']:+.2f}%")
        if r.get("alert_time_kst"):
            lines.append(f"추천시간: {r['alert_time_kst']}")
        if r.get("signal_candle_start_kst") and r.get("signal_candle_close_kst"):
            lines.append(f"추천 기준봉: {r['signal_candle_start_kst']} ~ {r['signal_candle_close_kst']}")

    return "\n".join(lines)


# =========================================================
# SCAN MARKETS
# =========================================================
def scan_markets():
    bullish_ok, btc_analysis_info, btc_status = get_btc_bullish_ok()

    markets = get_krw_markets()
    scan_count = 0

    all_passed = []
    watch_excluded_count = 0
    filter_pass_count = 0

    analysis_info = btc_analysis_info or {}

    print(f"BTC status: {btc_status}")
    print(f"Scan markets: {len(markets)}")
    print(build_analysis_info_text(analysis_info))

    for idx, market in enumerate(markets, 1):
        try:
            df = fetch_4h_candles(market, ALERT_CANDLE_COUNT)

            if df.empty or len(df) < 130:
                continue

            df, removed = remove_incomplete_current_candle(df)

            if df.empty or len(df) < 130:
                continue

            # 기준봉 정보는 첫 정상 종목 기준으로도 보정
            if not analysis_info:
                analysis_info = get_analysis_candle_info_from_df(df)

            df = add_indicators(df)
            row = df.iloc[-1]

            c = make_candidate_from_row(market, row, bullish_ok, analysis_info)
            scan_count += 1

            if not pass_confirmed_filter(c):
                continue

            filter_pass_count += 1

            # V3.7: WATCH는 최종 알림/추적 제외
            if c.get("alert_grade") not in ALLOWED_ALERT_GRADES:
                watch_excluded_count += 1
                continue

            all_passed.append(c)

        except Exception as e:
            print(f"[{market}] scan error: {e}")

    sorted_candidates = sort_candidates(all_passed)
    selected = sorted_candidates[:MAX_ALERT_COUNT]

    meta = {
        "scan_count": scan_count,
        "market_count": len(markets),
        "filter_pass_count": filter_pass_count,
        "watch_excluded_count": watch_excluded_count,
        "btc_status": btc_status,
        "btc_ok": bullish_ok,
        "analysis_info": analysis_info,
        "total_ab_count": len(sorted_candidates),
    }

    return selected, meta


# =========================================================
# MESSAGE FORMAT
# =========================================================
def format_candidate_line(idx, c):
    market = c.get("market")
    grade_label = c.get("alert_grade_label")
    score = safe_float(c.get("score"))
    price = safe_float(c.get("close"))
    value_eok = safe_float(c.get("value")) / 100_000_000
    volume_ratio = safe_float(c.get("volume_ratio"))
    short_k = safe_float(c.get("short_k"))
    short_d = safe_float(c.get("short_d"))
    mid_k = safe_float(c.get("mid_k"))
    ma20_dev = safe_float(c.get("ma20_dev"))
    recent_3bar = safe_float(c.get("recent_3bar"))
    candle_change = safe_float(c.get("candle_change"))
    close_position = safe_float(c.get("close_position"))
    upper_wick_ratio = safe_float(c.get("upper_wick_ratio"))

    tp_price = price * (1.0 + TAKE_PROFIT_RATE)
    sl_price = price * (1.0 - STOP_LOSS_RATE)

    lines = []
    lines.append(f"{idx}. {market} {grade_label} / score {score:.0f}")
    lines.append(
        f"가격 {price_fmt(price)} / TP {price_fmt(tp_price)}(+{TAKE_PROFIT_RATE * 100:.1f}%) / "
        f"SL {price_fmt(sl_price)}(-{STOP_LOSS_RATE * 100:.1f}%)"
    )
    lines.append(
        f"4H거래대금 {value_eok:.1f}억 / 거래비 {volume_ratio:.2f}x / "
        f"캔들 {candle_change:+.2f}% / 3봉 {recent_3bar:+.2f}%"
    )
    lines.append(
        f"K/D {short_k:.1f}/{short_d:.1f} / 중기K {mid_k:.1f} / "
        f"MA20 {ma20_dev:+.2f}%"
    )
    lines.append(
        f"종가위치 {close_position:.2f} / 윗꼬리 {upper_wick_ratio:.2f}"
    )

    return "\n".join(lines)


def build_telegram_message(candidates, meta, active_summary=""):
    analysis_info = meta.get("analysis_info") or {}

    lines = []
    lines.append("🟢 4H 마감 후 상승전환 확정 후보 V3.7")
    lines.append("A/B 등급만 표시 · WATCH 제외 · 확정봉 기준")
    lines.append("")
    lines.append(build_analysis_info_text(analysis_info))
    lines.append("")
    lines.append(f"BTC 상태: {meta.get('btc_status', '-')}")
    lines.append(f"스캔: {meta.get('scan_count', 0)}개 / 필터통과: {meta.get('filter_pass_count', 0)}개")
    lines.append(f"A/B 후보: {meta.get('total_ab_count', 0)}개 / WATCH 제외: {meta.get('watch_excluded_count', 0)}개")
    lines.append(f"표시: 상위 {len(candidates)}개 / 최대 {MAX_ALERT_COUNT}개")
    lines.append(f"전략 TP/SL: +{TAKE_PROFIT_RATE * 100:.1f}% / -{STOP_LOSS_RATE * 100:.1f}% / {MAX_HOLD_HOURS}시간")
    lines.append("")

    a_list = [c for c in candidates if c.get("alert_grade") == "A"]
    b_list = [c for c in candidates if c.get("alert_grade") == "B"]

    idx = 1

    if a_list:
        lines.append("🔥 A급 후보")
        for c in a_list:
            lines.append(format_candidate_line(idx, c))
            lines.append("")
            idx += 1

    if b_list:
        lines.append("🟡 B급 후보")
        for c in b_list:
            lines.append(format_candidate_line(idx, c))
            lines.append("")
            idx += 1

    if active_summary:
        lines.append("")
        lines.append(active_summary)
        lines.append("")

    lines.append("※ 백테스트 기준: TOP_10_A_B / TP +5%, SL -4% 조합을 우선 적용")
    lines.append("※ 알림은 매수 강요가 아니며, 진입 전 BTC 흐름/저항/호가를 반드시 확인하세요.")
    lines.append("※ 실제 체결, 슬리피지, 수수료는 반영되지 않습니다.")

    return "\n".join(lines)


def build_empty_message(meta, active_summary=""):
    analysis_info = meta.get("analysis_info") or {}

    lines = []
    lines.append("🟢 4H 마감 후 상승전환 확정 후보 V3.7")
    lines.append("조건 만족 A/B 후보 없음")
    lines.append("")
    lines.append(build_analysis_info_text(analysis_info))
    lines.append("")
    lines.append(f"BTC 상태: {meta.get('btc_status', '-')}")
    lines.append(f"스캔: {meta.get('scan_count', 0)}개 / 필터통과: {meta.get('filter_pass_count', 0)}개")
    lines.append(f"A/B 후보: {meta.get('total_ab_count', 0)}개 / WATCH 제외: {meta.get('watch_excluded_count', 0)}개")
    lines.append("")
    lines.append("※ 현재 조건에서는 무리한 진입보다 관망 우선입니다.")

    if active_summary:
        lines.append("")
        lines.append(active_summary)

    lines.append("")
    lines.append("※ V3.7은 진행 중인 최신 4H 봉을 제외하고 확정봉만 분석합니다.")

    return "\n".join(lines)


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("Telegram Coin Alert Bot V3.7 started")
    print(f"Now KST: {now_kst().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"ALERT_INTERVAL={ALERT_INTERVAL}")
    print(f"ALERT_CANDLE_COUNT={ALERT_CANDLE_COUNT}")
    print(f"MAX_ALERT_COUNT={MAX_ALERT_COUNT}")
    print(f"SEND_EMPTY_ALERT={SEND_EMPTY_ALERT}")
    print(f"ENABLE_SIGNAL_TRACKING={ENABLE_SIGNAL_TRACKING}")
    print(f"TP={TAKE_PROFIT_RATE}, SL={STOP_LOSS_RATE}, HOLD={MAX_HOLD_HOURS}h")
    print("=" * 80)

    state = load_signal_state()

    # 1) 기존 추적 신호 TP/SL/만료 체크
    closed_events = check_active_signals(state)

    if closed_events:
        msg = build_closed_events_message(closed_events)
        print(msg)
        send_telegram_message(msg)

    # 2) 현재 추적 중 신호 요약 생성
    active_summary = build_active_signals_summary(state)

    # 3) 신규 후보 스캔
    candidates, meta = scan_markets()

    # 4) 텔레그램 알림
if candidates:
    added = add_new_signals_to_state(state, candidates)
    print(f"Added new active signals: {added}")

    save_signal_state(state)

    msg = build_telegram_message(candidates, meta, active_summary=active_summary)
    print(msg)
    send_telegram_message(msg)


    else:
        if SEND_EMPTY_ALERT:
            msg = build_empty_message(meta, active_summary=active_summary)
            print(msg)
            send_telegram_message(msg)
        else:
            print("No candidates and SEND_EMPTY_ALERT=false. Telegram skipped.")

    # 5) 상태 저장
    save_signal_state(state)

    print("Telegram Coin Alert Bot V3.7 finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"🚨 Telegram Coin Alert Bot V3.7 error\n\n{type(e).__name__}: {e}"
        print(error_msg)
        send_telegram_message(error_msg)
        raise
