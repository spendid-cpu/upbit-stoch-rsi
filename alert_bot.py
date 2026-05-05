# alert_bot.py
# Telegram Coin Alert Bot V3.5
# - Confirmed 4H close scanner only
# - Preliminary alert disabled
# - Momentum/chasing filter disabled
# - Overheated coins excluded
# - BTC 4H filter
# - Candle quality filter
# - MA5 slope filter
# - Telegram A/B/WATCH grading
# - Active signal tracking: TP +5%, SL -4%, Expire 48h

import os
import time
import math
import json
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
ALERT_CANDLE_COUNT = int(os.getenv("ALERT_CANDLE_COUNT", "500"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.08"))
MAX_ALERT_COUNT = int(os.getenv("MAX_ALERT_COUNT", "10"))
SEND_EMPTY_ALERT = os.getenv("SEND_EMPTY_ALERT", "true").strip().lower() == "true"

# Signal tracking
ENABLE_SIGNAL_TRACKING = os.getenv("ENABLE_SIGNAL_TRACKING", "true").strip().lower() == "true"
SIGNAL_STATE_FILE = os.getenv("SIGNAL_STATE_FILE", "active_signals.json").strip()
TAKE_PROFIT_RATE = float(os.getenv("TAKE_PROFIT_RATE", "0.05"))   # +5%
STOP_LOSS_RATE = float(os.getenv("STOP_LOSS_RATE", "0.04"))       # -4%
MAX_HOLD_HOURS = float(os.getenv("MAX_HOLD_HOURS", "48"))         # 48h
SIGNAL_COOLDOWN_HOURS = float(os.getenv("SIGNAL_COOLDOWN_HOURS", "12"))

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


def percent(a, b):
    try:
        a = float(a)
        b = float(b)
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


def parse_datetime(value):
    try:
        if not value:
            return None
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt.astimezone(KST)
    except Exception:
        return None


def format_price(price):
    price = safe_float(price)

    if price < 1:
        return f"{price:.8f}".rstrip("0").rstrip(".")
    if price < 10:
        return f"{price:.4f}"
    if price < 100:
        return f"{price:.3f}"
    return f"{price:.0f}"


# =========================
# 3. TELEGRAM
# =========================

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 비어 있습니다.")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    max_len = 3900
    chunks = []

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
# 4. SIGNAL STATE / TP SL TRACKING
# =========================

def load_signal_state():
    if not os.path.exists(SIGNAL_STATE_FILE):
        return {
            "active": [],
            "closed": []
        }

    try:
        with open(SIGNAL_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            active = []
            closed = []

            for item in data:
                if item.get("status") == "closed":
                    closed.append(item)
                else:
                    active.append(item)

            return {
                "active": active,
                "closed": closed
            }

        if not isinstance(data, dict):
            return {
                "active": [],
                "closed": []
            }

        data.setdefault("active", [])
        data.setdefault("closed", [])

        return data

    except Exception:
        return {
            "active": [],
            "closed": []
        }


def save_signal_state(state):
    state.setdefault("active", [])
    state.setdefault("closed", [])

    # closed 기록이 너무 커지는 것 방지
    if len(state["closed"]) > 300:
        state["closed"] = state["closed"][-300:]

    with open(SIGNAL_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_current_price_map(markets):
    markets = sorted(list(set([m for m in markets if m])))
    result = {}

    if not markets:
        return result

    batch_size = 80

    for i in range(0, len(markets), batch_size):
        batch = markets[i:i + batch_size]

        try:
            res = requests.get(
                UPBIT_TICKER_URL,
                params={"markets": ",".join(batch)},
                timeout=15
            )
            res.raise_for_status()
            data = res.json()

            for item in data:
                market = item.get("market")
                price = safe_float(item.get("trade_price"), 0)

                if market and price > 0:
                    result[market] = price

            time.sleep(0.15)

        except Exception:
            continue

    return result


def is_recently_closed(state, market):
    if SIGNAL_COOLDOWN_HOURS <= 0:
        return False

    now = now_kst()

    for sig in reversed(state.get("closed", [])):
        if sig.get("market") != market:
            continue

        exit_time = parse_datetime(sig.get("exit_time_iso"))
        if exit_time is None:
            exit_time = parse_datetime(sig.get("exit_time"))

        if exit_time is None:
            continue

        hours = (now - exit_time).total_seconds() / 3600.0

        if hours <= SIGNAL_COOLDOWN_HOURS:
            return True

    return False


def evaluate_active_signals(state):
    active = state.get("active", [])

    if not active:
        return [], False

    markets = [
        sig.get("market")
        for sig in active
        if sig.get("status", "active") == "active" and sig.get("market")
    ]

    price_map = get_current_price_map(markets)

    now = now_kst()
    new_active = []
    newly_closed = []
    events = []
    changed = False

    for sig in active:
        if sig.get("status", "active") != "active":
            newly_closed.append(sig)
            continue

        market = sig.get("market")
        entry_price = safe_float(sig.get("entry_price"), 0)
        alert_time = parse_datetime(sig.get("alert_time_iso"))

        if not market or entry_price <= 0:
            continue

        current_price = safe_float(price_map.get(market), 0)

        if current_price <= 0:
            new_active.append(sig)
            continue

        return_rate = percent(current_price, entry_price)

        hold_hours = 0.0
        if alert_time is not None:
            hold_hours = (now - alert_time).total_seconds() / 3600.0

        result = None
        result_label = None

        if return_rate >= TAKE_PROFIT_RATE * 100:
            result = "TP"
            result_label = "✅ 익절"
        elif return_rate <= -STOP_LOSS_RATE * 100:
            result = "SL"
            result_label = "❌ 손절"
        elif hold_hours >= MAX_HOLD_HOURS:
            result = "EXPIRED"
            result_label = "⏰ 만료"

        if result:
            closed_sig = dict(sig)
            closed_sig["status"] = "closed"
            closed_sig["result"] = result
            closed_sig["result_label"] = result_label
            closed_sig["exit_price"] = current_price
            closed_sig["exit_time"] = format_now()
            closed_sig["exit_time_iso"] = now.isoformat()
            closed_sig["return_rate"] = return_rate
            closed_sig["hold_hours"] = hold_hours

            newly_closed.append(closed_sig)
            events.append(closed_sig)
            changed = True

        else:
            sig["last_price"] = current_price
            sig["last_return_rate"] = return_rate
            sig["last_checked_time"] = format_now()
            sig["last_checked_time_iso"] = now.isoformat()

            new_active.append(sig)
            changed = True

    state["active"] = new_active
    state.setdefault("closed", [])
    state["closed"].extend(newly_closed)

    return events, changed


def build_tracking_event_message(events):
    if not events:
        return ""

    lines = []
    lines.append("📌 전략 결과 알림")
    lines.append(f"시간: {format_now()}")
    lines.append(
        f"기준: 익절 +{TAKE_PROFIT_RATE * 100:.1f}% / "
        f"손절 -{STOP_LOSS_RATE * 100:.1f}% / "
        f"만료 {MAX_HOLD_HOURS:.0f}시간"
    )
    lines.append("")

    tp_events = [e for e in events if e.get("result") == "TP"]
    sl_events = [e for e in events if e.get("result") == "SL"]
    expired_events = [e for e in events if e.get("result") == "EXPIRED"]

    def add_section(title, items):
        if not items:
            return

        lines.append(title)

        for i, e in enumerate(items, start=1):
            market = e.get("market", "UNKNOWN")
            grade_label = e.get("grade_label", "")
            entry = safe_float(e.get("entry_price"), 0)
            exit_price = safe_float(e.get("exit_price"), 0)
            ret = safe_float(e.get("return_rate"), 0)
            hold_hours = safe_float(e.get("hold_hours"), 0)
            alert_time = e.get("alert_time", "")

            lines.append(
                f"{i}. {market} {grade_label}\n"
                f"추천시간: {alert_time}\n"
                f"추천가: {format_price(entry)} / 현재가: {format_price(exit_price)}\n"
                f"수익률: {ret:+.2f}% / 경과: {hold_hours:.1f}시간\n"
            )

    add_section("✅ 익절 도달", tp_events)
    add_section("❌ 손절 도달", sl_events)
    add_section("⏰ 관찰 만료", expired_events)

    lines.append("※ 실제 매매 여부와 무관한 전략 기준 결과입니다.")
    lines.append("※ GitHub Actions 실행 시점의 현재가 기준으로 판정합니다.")

    return "\n".join(lines)


def add_new_signals_to_state(state, candidates, alert_mode):
    state.setdefault("active", [])
    state.setdefault("closed", [])

    active_markets = set()

    for sig in state["active"]:
        if sig.get("status", "active") == "active" and sig.get("market"):
            active_markets.add(sig["market"])

    now = now_kst()
    added = 0

    for c in candidates:
        market = c.get("market")
        entry_price = safe_float(c.get("price"), 0)

        if not market or entry_price <= 0:
            continue

        if market in active_markets:
            continue

        if is_recently_closed(state, market):
            continue

        tp_price = entry_price * (1.0 + TAKE_PROFIT_RATE)
        sl_price = entry_price * (1.0 - STOP_LOSS_RATE)

        signal = {
            "market": market,
            "status": "active",

            "entry_price": entry_price,
            "tp_price": tp_price,
            "sl_price": sl_price,

            "take_profit_rate": TAKE_PROFIT_RATE,
            "stop_loss_rate": STOP_LOSS_RATE,

            "alert_time": format_now(),
            "alert_time_iso": now.isoformat(),
            "alert_mode": alert_mode,

            "score": safe_float(c.get("score"), 0),
            "alert_score": safe_float(c.get("alert_score"), 0),
            "quality_score": safe_float(c.get("alert_quality_score"), 0),

            "grade": c.get("alert_grade", "WATCH"),
            "grade_label": c.get("alert_grade_label", "👀 관찰"),

            "signal_type": "CONFIRMED_REVERSAL",
            "signal_type_label": "🟢 확정전환",

            "volume_ratio": safe_float(c.get("volume_ratio"), 0),
            "short_k": safe_float(c.get("short_k"), 0),
            "short_d": safe_float(c.get("short_d"), 0),
            "mid_k": safe_float(c.get("mid_k"), 0),
            "ma20_dev": safe_float(c.get("ma20_dev"), 0),
            "recent_3bar": safe_float(c.get("recent_3bar"), 0),
            "close_position": safe_float(c.get("close_position"), 0),
            "upper_wick_ratio": safe_float(c.get("upper_wick_ratio"), 0),
            "candle_change": safe_float(c.get("candle_change"), 0),
        }

        state["active"].append(signal)
        active_markets.add(market)
        added += 1

    return added


# =========================
# 5. UPBIT DATA
# =========================

def get_krw_markets():
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

        markets = [
            item["market"]
            for item in data
            if item.get("market", "").startswith("KRW-")
        ]

        return sorted(markets)

    except Exception as e:
        raise RuntimeError(f"KRW 마켓 조회 실패: {e}")


def get_ohlcv(market, interval="minute240", count=500, retry=2):
    if pyupbit is None:
        return None

    for attempt in range(retry + 1):
        try:
            df = pyupbit.get_ohlcv(market, interval=interval, count=count)

            if df is None or len(df) < 60:
                time.sleep(0.3)
                continue

            df = df.copy()

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


# =========================
# 6. INDICATORS
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

    # 현재봉 거래대금은 직전 20봉 평균과 비교
    df["value_ma20_prev"] = df["value"].shift(1).rolling(20, min_periods=5).mean()

    return df


# =========================
# 7. ALERT MODE
# =========================

def detect_alert_mode():
    """
    V3.5 확정 전용 운영.
    예비 알림 없이 모든 실행을 confirmed로 처리한다.
    """
    return "confirmed"


def get_signal_index(df, alert_mode):
    """
    pyupbit minute240은 현재 진행 중인 새 4H 캔들이 포함될 수 있다.

    confirmed:
      방금 마감된 이전 4H 캔들을 평가해야 하므로 -2 사용.
    """
    if len(df) < 60:
        return None

    return len(df) - 2


# =========================
# 8. BTC FILTER
# =========================

def get_btc_filter(alert_mode):
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

    # V3.5 확정 전용 BTC 필터
    passed = close > ma20 and btc_3bar >= -3.0
    reason = "BTC close > MA20" if passed else "BTC confirmed filter fail"

    return {
        "pass": passed,
        "reason": reason,
        "btc_close": close,
        "btc_ma20": ma20,
        "btc_3bar": btc_3bar
    }


# =========================
# 9. BASE SCORE
# =========================

def calculate_base_score(metrics):
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

    # 거래대금 증가
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
    elif mid_k <= 70:
        score += 10

    # MA20 근처
    if -2.0 <= ma20_dev <= 2.0:
        score += 35
    elif -4.0 <= ma20_dev <= 4.0:
        score += 25
    elif -5.0 <= ma20_dev <= 5.0:
        score += 15

    # 최근 3봉
    if 0 <= recent_3bar <= 4.0:
        score += 30
    elif -2.0 <= recent_3bar <= 5.0:
        score += 20
    elif -4.0 <= recent_3bar <= 6.0:
        score += 10

    # 종가 위치
    if close_position >= 0.85:
        score += 35
    elif close_position >= 0.70:
        score += 25
    elif close_position >= 0.60:
        score += 15

    # 윗꼬리
    if upper_wick_ratio <= 0.15:
        score += 30
    elif upper_wick_ratio <= 0.25:
        score += 20
    elif upper_wick_ratio <= 0.35:
        score += 10

    # 캔들 상승률
    if 1.0 <= candle_change <= 3.5:
        score += 25
    elif 0.2 <= candle_change <= 4.0:
        score += 15

    # MA5 상승
    if ma5_up:
        score += 20

    # 음봉 과도 방지
    if bullish_ok:
        score += 15

    return score


# =========================
# 10. CANDIDATE METRICS
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

    # 강한 음봉 배제 기준
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

    score = calculate_base_score(metrics)
    metrics["score"] = score

    return metrics


# =========================
# 11. V3.5 CONFIRMED FILTER
# =========================

def pass_confirmed_filter(c):
    """
    V3.5 확정 전용 필터.
    너무 오른 종목, 과열 종목, 윗꼬리 긴 종목을 배제한다.
    """

    # 기본 점수
    if c["score"] < 210:
        return False

    # 거래대금 증가
    if c["volume_ratio"] < 1.5:
        return False

    # Stoch RSI 과열 배제
    if not (20 <= c["short_k"] <= 70):
        return False

    if c["short_d"] > 60:
        return False

    if c["mid_k"] > 70:
        return False

    # MA20 이격 과열 배제
    if not (-5.0 <= c["ma20_dev"] <= 5.0):
        return False

    # 최근 3봉 급등/급락 배제
    if c["recent_3bar"] < -4.0:
        return False

    if c["recent_3bar"] > 6.0:
        return False

    # 캔들 품질
    if c["close_position"] < 0.60:
        return False

    if c["upper_wick_ratio"] > 0.35:
        return False

    # 현재 캔들 너무 약하거나 너무 오른 것 배제
    if c["candle_change"] < 0.2:
        return False

    if c["candle_change"] > 4.0:
        return False

    # 강한 음봉 배제
    if not c["bullish_ok"]:
        return False

    # MA5 상승 기울기
    if not c["ma5_up"]:
        return False

    return True


# =========================
# 12. ALERT GRADING
# =========================

def calculate_alert_grade(candidate):
    score = safe_float(candidate.get("score"), 0)
    volume_ratio = safe_float(candidate.get("volume_ratio"), 0)
    short_k = safe_float(candidate.get("short_k"), 100)
    short_d = safe_float(candidate.get("short_d"), 100)
    mid_k = safe_float(candidate.get("mid_k"), 100)
    ma20_dev = safe_float(candidate.get("ma20_dev"), 999)
    recent_3bar = safe_float(candidate.get("recent_3bar"), 0)
    close_position = safe_float(candidate.get("close_position"), 0)
    upper_wick_ratio = safe_float(candidate.get("upper_wick_ratio"), 1)
    candle_change = safe_float(candidate.get("candle_change"), 0)

    quality_score = 0

    # 종가위치
    if close_position >= 0.90:
        quality_score += 30
    elif close_position >= 0.80:
        quality_score += 20
    elif close_position >= 0.70:
        quality_score += 10
    elif close_position < 0.60:
        quality_score -= 15

    # 윗꼬리
    if upper_wick_ratio <= 0.10:
        quality_score += 30
    elif upper_wick_ratio <= 0.20:
        quality_score += 20
    elif upper_wick_ratio <= 0.30:
        quality_score += 10
    elif upper_wick_ratio > 0.35:
        quality_score -= 20

    # 거래대금
    if volume_ratio >= 2.0:
        quality_score += 25
    elif volume_ratio >= 1.7:
        quality_score += 20
    elif volume_ratio >= 1.5:
        quality_score += 10

    # MA20 이격
    if -2.0 <= ma20_dev <= 2.0:
        quality_score += 20
    elif -3.0 <= ma20_dev <= 3.0:
        quality_score += 10
    elif ma20_dev > 4.0:
        quality_score -= 10

    # 캔들 상승률
    if 1.0 <= candle_change <= 3.5:
        quality_score += 20
    elif 0.3 <= candle_change < 1.0:
        quality_score += 8
    elif candle_change > 4.0:
        quality_score -= 20

    # 최근 3봉
    if 0 <= recent_3bar <= 4:
        quality_score += 10
    elif recent_3bar > 6:
        quality_score -= 20

    # Stoch RSI
    if short_k <= 45 and short_d <= 45:
        quality_score += 15
    elif short_k <= 60 and short_d <= 55:
        quality_score += 8
    elif short_k > 70 or short_d > 60:
        quality_score -= 20

    # 중기 K
    if mid_k <= 50:
        quality_score += 8
    elif mid_k > 70:
        quality_score -= 15

    alert_score = score + quality_score

    is_a_grade = (
        score >= 230
        and volume_ratio >= 1.70
        and close_position >= 0.85
        and upper_wick_ratio <= 0.15
        and -2.0 <= ma20_dev <= 2.0
        and 1.0 <= candle_change <= 3.5
        and short_k <= 60
        and short_d <= 55
        and mid_k <= 65
        and recent_3bar <= 5.0
    )

    is_b_grade = (
        score >= 220
        and volume_ratio >= 1.50
        and close_position >= 0.70
        and upper_wick_ratio <= 0.30
        and -3.0 <= ma20_dev <= 3.0
        and 0.2 <= candle_change <= 4.0
        and recent_3bar <= 6.0
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
    candidate["signal_type"] = "CONFIRMED_REVERSAL"
    candidate["signal_type_label"] = "🟢 확정전환"

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
            -safe_float(x.get("alert_score"), 0),
            -safe_float(x.get("score"), 0),
            -safe_float(x.get("volume_ratio"), 0),
            -safe_float(x.get("close_position"), 0),
            safe_float(x.get("upper_wick_ratio"), 1),
            str(x.get("market", ""))
        )
    )

    return graded


# =========================
# 13. MESSAGE FORMAT
# =========================

def format_candidate_line(index, candidate):
    market = candidate.get("market", "UNKNOWN")

    score = safe_float(candidate.get("score"), 0)
    volume_ratio = safe_float(candidate.get("volume_ratio"), 0)
    short_k = safe_float(candidate.get("short_k"), 0)
    short_d = safe_float(candidate.get("short_d"), 0)
    mid_k = safe_float(candidate.get("mid_k"), 0)
    ma20_dev = safe_float(candidate.get("ma20_dev"), 0)
    recent_3bar = safe_float(candidate.get("recent_3bar"), 0)
    close_position = safe_float(candidate.get("close_position"), 0)
    upper_wick_ratio = safe_float(candidate.get("upper_wick_ratio"), 0)
    candle_change = safe_float(candidate.get("candle_change"), 0)
    price = safe_float(candidate.get("price"), 0)

    grade_label = candidate.get("alert_grade_label", "👀 관찰")
    alert_score = safe_float(candidate.get("alert_score"), score)
    quality_score = safe_float(candidate.get("alert_quality_score"), 0)

    tp_price = price * (1.0 + TAKE_PROFIT_RATE)
    sl_price = price * (1.0 - STOP_LOSS_RATE)

    return (
        f"{index}. {market} {grade_label} 🟢 확정전환\n"
        f"점수 {score:.0f} / 보정 {alert_score:.0f} / 품질 {quality_score:+.0f}\n"
        f"거래대금 x{volume_ratio:.2f} / K {short_k:.2f} D {short_d:.2f} / 중기K {mid_k:.2f}\n"
        f"MA20 {ma20_dev:.2f}% / 3봉 {recent_3bar:.2f}% / 종가위치 {close_position:.2f} / 윗꼬리 {upper_wick_ratio:.2f}\n"
        f"캔들 {candle_change:.2f}% / 가격 {format_price(price)}\n"
        f"전략기준 TP {format_price(tp_price)} / SL {format_price(sl_price)}\n"
    )


def build_telegram_message(candidates, alert_mode, scan_count, btc_info):
    candidates = apply_alert_grades(candidates)

    title = "🟢 4H 마감 후 상승전환 확정 후보"

    lines = []
    lines.append(title)
    lines.append(f"시간: {format_now()}")
    lines.append(f"스캔: 전체 KRW {scan_count}개")
    lines.append(f"후보: {len(candidates)}개")

    if btc_info:
        btc_close = safe_float(btc_info.get("btc_close"), 0)
        btc_ma20 = safe_float(btc_info.get("btc_ma20"), 0)
        btc_3bar = safe_float(btc_info.get("btc_3bar"), 0)
        btc_status = "통과" if btc_info.get("pass") else "차단"

        lines.append(
            f"BTC필터: {btc_status} / BTC {format_price(btc_close)} / "
            f"MA20 {format_price(btc_ma20)} / 3봉 {btc_3bar:.2f}%"
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
    lines.append(f"※ 전략 추적 기준: 익절 +{TAKE_PROFIT_RATE * 100:.1f}% / 손절 -{STOP_LOSS_RATE * 100:.1f}% / 만료 {MAX_HOLD_HOURS:.0f}시간")
    lines.append("※ V3.5는 확정봉 기준이며, 과열/급등/윗꼬리 종목은 배제합니다.")

    return "\n".join(lines)


def build_empty_message(alert_mode, scan_count, btc_info):
    title = "🟢 4H 마감 후 상승전환 확정 후보"

    lines = []
    lines.append(title)
    lines.append(f"시간: {format_now()}")
    lines.append(f"스캔: 전체 KRW {scan_count}개")
    lines.append("후보: 0개")
    lines.append("")

    if btc_info:
        btc_close = safe_float(btc_info.get("btc_close"), 0)
        btc_ma20 = safe_float(btc_info.get("btc_ma20"), 0)
        btc_3bar = safe_float(btc_info.get("btc_3bar"), 0)
        btc_status = "통과" if btc_info.get("pass") else "차단"

        lines.append(
            f"BTC필터: {btc_status} / BTC {format_price(btc_close)} / "
            f"MA20 {format_price(btc_ma20)} / 3봉 {btc_3bar:.2f}%"
        )
        lines.append("")

    lines.append("조건 만족 종목 없음")
    lines.append("※ 현재 조건에서는 무리한 매수보다 관망이 우선입니다.")
    lines.append("※ V3.5는 확정봉 기준이라 신호 수가 적을 수 있습니다.")

    return "\n".join(lines)


def build_error_message(error_text):
    return (
        "🔴 Telegram Coin Alert 오류 발생\n"
        f"시간: {format_now()}\n"
        f"오류:\n{error_text}"
    )


# =========================
# 14. SCANNER
# =========================

def scan_markets(alert_mode, btc_info):
    markets = get_krw_markets()
    scan_count = len(markets)

    candidates = []

    if not btc_info.get("pass", False):
        return candidates, scan_count

    for market in markets:
        try:
            if market == "KRW-BTC":
                continue

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

            passed = pass_confirmed_filter(metrics)

            if passed:
                metrics = calculate_alert_grade(metrics)
                candidates.append(metrics)

            time.sleep(REQUEST_DELAY)

        except Exception:
            time.sleep(REQUEST_DELAY)
            continue

    candidates = apply_alert_grades(candidates)

    if MAX_ALERT_COUNT > 0:
        candidates = candidates[:MAX_ALERT_COUNT]

    return candidates, scan_count


# =========================
# 15. MAIN
# =========================

def main():
    print("========================================")
    print("Telegram Coin Alert Bot V3.5 started")
    print(f"Time: {format_now()}")
    print("Mode: CONFIRMED ONLY")
    print(f"Interval: {ALERT_INTERVAL}")
    print(f"Candle count: {ALERT_CANDLE_COUNT}")
    print(f"Request delay: {REQUEST_DELAY}")
    print(f"Max alert count: {MAX_ALERT_COUNT}")
    print(f"Send empty alert: {SEND_EMPTY_ALERT}")
    print(f"Signal tracking: {ENABLE_SIGNAL_TRACKING}")
    print(f"TP: {TAKE_PROFIT_RATE * 100:.1f}% / SL: {STOP_LOSS_RATE * 100:.1f}% / Expire: {MAX_HOLD_HOURS:.0f}h")
    print("========================================")

    # 1) 기존 추천 종목 TP/SL/만료 체크
    state = load_signal_state()

    if ENABLE_SIGNAL_TRACKING:
        events, state_changed = evaluate_active_signals(state)

        if events:
            tracking_message = build_tracking_event_message(events)
            print(tracking_message)
            send_telegram_message(tracking_message)

        if state_changed:
            save_signal_state(state)
            print(f"Signal state updated after tracking. Active: {len(state.get('active', []))}")

    # 2) 신규 후보 스캔
    alert_mode = detect_alert_mode()
    print(f"Alert mode: {alert_mode}")

    btc_info = get_btc_filter(alert_mode)
    print(f"BTC filter: {btc_info}")

    candidates, scan_count = scan_markets(alert_mode, btc_info)

    print(f"Scan count: {scan_count}")
    print(f"Candidate count: {len(candidates)}")

    # 3) 신규 후보 알림 및 active 저장
    if candidates:
        message = build_telegram_message(
            candidates=candidates,
            alert_mode=alert_mode,
            scan_count=scan_count,
            btc_info=btc_info
        )

        print(message)
        send_telegram_message(message)

        if ENABLE_SIGNAL_TRACKING:
            added = add_new_signals_to_state(state, candidates, alert_mode)

            if added > 0:
                save_signal_state(state)
                print(f"Added new active signals: {added}")
            else:
                print("No new active signals added.")

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

    print("Telegram Coin Alert Bot V3.5 finished")


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
