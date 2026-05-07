import os
import time
import json
import math
import requests
import pandas as pd
import numpy as np

from datetime import datetime, timedelta, timezone


# =========================================================
# VERSION
# =========================================================
BOT_VERSION = "V3.8"


# =========================================================
# TIMEZONE
# =========================================================
KST = timezone(timedelta(hours=9))


def now_kst():
    return datetime.now(KST)


def parse_kst_datetime(value):
    if not value:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=KST)
        return value.astimezone(KST)

    try:
        s = str(value).replace(" KST", "").strip()
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt.astimezone(KST)
    except Exception:
        return None


def format_kst(dt):
    if not dt:
        return "-"
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S KST")


# =========================================================
# ENV CONFIG
# =========================================================
RUN_MODE = os.getenv("RUN_MODE", "full").strip().lower()
# full  = 신규 스캔 + 기존 추적
# track = active_signals.json 추적만

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

ALERT_INTERVAL = os.getenv("ALERT_INTERVAL", "minute240")
ALERT_CANDLE_COUNT = int(os.getenv("ALERT_CANDLE_COUNT", "200"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.08"))

MAX_ALERT_COUNT = int(os.getenv("MAX_ALERT_COUNT", "10"))
SEND_EMPTY_ALERT = os.getenv("SEND_EMPTY_ALERT", "true").lower() == "true"

ENABLE_SIGNAL_TRACKING = os.getenv("ENABLE_SIGNAL_TRACKING", "true").lower() == "true"
SIGNAL_STATE_FILE = os.getenv("SIGNAL_STATE_FILE", "active_signals.json")

TAKE_PROFIT_RATE = float(os.getenv("TAKE_PROFIT_RATE", "0.05"))
STOP_LOSS_RATE = float(os.getenv("STOP_LOSS_RATE", "0.04"))
MAX_HOLD_HOURS = float(os.getenv("MAX_HOLD_HOURS", "48"))
SIGNAL_COOLDOWN_HOURS = float(os.getenv("SIGNAL_COOLDOWN_HOURS", "12"))

MIN_4H_TRADE_VALUE = float(os.getenv("MIN_4H_TRADE_VALUE", "150000000"))
MIN_CANDLE_CHANGE = float(os.getenv("MIN_CANDLE_CHANGE", "0.7"))

ALLOWED_ALERT_GRADES = [
    x.strip().upper()
    for x in os.getenv("ALLOWED_ALERT_GRADES", "A,B").split(",")
    if x.strip()
]

ENABLE_LIVE_PRICE_CHECK = os.getenv("ENABLE_LIVE_PRICE_CHECK", "true").lower() == "true"

ENTRY_OK_MAX_GAP = float(os.getenv("ENTRY_OK_MAX_GAP", "1.0"))
ENTRY_CAUTION_MAX_GAP = float(os.getenv("ENTRY_CAUTION_MAX_GAP", "2.0"))
ENTRY_CHASE_MAX_GAP = float(os.getenv("ENTRY_CHASE_MAX_GAP", "3.0"))

TRACK_PROFIT_NOTICE_1 = float(os.getenv("TRACK_PROFIT_NOTICE_1", "0.02"))
TRACK_PROFIT_NOTICE_2 = float(os.getenv("TRACK_PROFIT_NOTICE_2", "0.03"))
TRACK_NEAR_TP_RATE = float(os.getenv("TRACK_NEAR_TP_RATE", "0.015"))
TRACK_NEAR_SL_RATE = float(os.getenv("TRACK_NEAR_SL_RATE", "0.015"))
TRACK_DRAWDOWN_FROM_HIGH = float(os.getenv("TRACK_DRAWDOWN_FROM_HIGH", "0.02"))


# =========================================================
# FORMAT HELPERS
# =========================================================
def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        if value == "":
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def format_price(value):
    try:
        value = float(value)
    except Exception:
        return str(value)

    if value >= 1000:
        return f"{value:,.0f}"
    elif value >= 100:
        return f"{value:,.1f}"
    elif value >= 10:
        return f"{value:,.2f}"
    elif value >= 1:
        return f"{value:,.3f}"
    else:
        return f"{value:,.6f}"


def format_pct(value, digits=2):
    try:
        return f"{float(value):+.{digits}f}%"
    except Exception:
        return "-"


def chunk_list(items, size=100):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# =========================================================
# TELEGRAM
# =========================================================
def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram secrets missing. Message not sent.")
        print(message)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # Telegram text limit 대응
    chunks = []
    text = str(message)

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
                print(f"[WARN] Telegram send failed: {r.status_code} {r.text}")
                ok = False
            time.sleep(0.2)
        except Exception as e:
            print(f"[WARN] Telegram exception: {type(e).__name__}: {e}")
            ok = False

    return ok


# =========================================================
# UPBIT API
# =========================================================
def upbit_get(url, params=None, timeout=10):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_krw_markets():
    url = "https://api.upbit.com/v1/market/all"
    data = upbit_get(url, params={"isDetails": "false"})

    markets = []
    for item in data:
        market = item.get("market", "")
        if not market.startswith("KRW-"):
            continue

        # 기준 마켓 제외
        if market in ["KRW-BTC", "KRW-USDT", "KRW-USDC"]:
            continue

        markets.append(market)

    return markets


def fetch_candles(market, interval=ALERT_INTERVAL, count=ALERT_CANDLE_COUNT):
    url = f"https://api.upbit.com/v1/candles/minutes/{interval.replace('minute', '')}"

    if interval == "minute240":
        url = "https://api.upbit.com/v1/candles/minutes/240"

    params = {
        "market": market,
        "count": min(int(count), 200),
    }

    data = upbit_get(url, params=params)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "candle_date_time_kst" not in df.columns:
        return pd.DataFrame()

    df["candle_time"] = pd.to_datetime(df["candle_date_time_kst"])
    df["candle_time"] = df["candle_time"].apply(lambda x: x.to_pydatetime().replace(tzinfo=KST))

    df = df.sort_values("candle_time").reset_index(drop=True)

    return df


def fetch_ticker_map(markets):
    if not markets:
        return {}

    result = {}

    unique_markets = list(dict.fromkeys([m for m in markets if m]))

    for chunk in chunk_list(unique_markets, 100):
        url = "https://api.upbit.com/v1/ticker"
        params = {"markets": ",".join(chunk)}

        try:
            data = upbit_get(url, params=params, timeout=10)

            for item in data:
                market = item.get("market")
                if market:
                    result[market] = item

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"[WARN] fetch_ticker_map failed: {chunk} / {type(e).__name__}: {e}")

    return result


def fetch_single_ticker(market):
    m = fetch_ticker_map([market])
    return m.get(market, {})


# =========================================================
# CANDLE HANDLING
# =========================================================
def remove_incomplete_current_candle(df):
    if df is None or df.empty:
        return df

    df = df.copy().sort_values("candle_time").reset_index(drop=True)

    latest_start = df.iloc[-1]["candle_time"]
    latest_close = latest_start + timedelta(hours=4)

    # 현재 시간이 최신봉 마감 + 1분 전이면 진행 중 봉으로 보고 제거
    if now_kst() < latest_close + timedelta(minutes=1):
        df = df.iloc[:-1].reset_index(drop=True)

    return df


def get_target_candle_meta_from_df(df):
    if df is None or df.empty:
        return {}

    latest = df.iloc[-1]
    start = latest["candle_time"]
    end = start + timedelta(hours=4)
    expected = end + timedelta(minutes=10)
    run_time = now_kst()

    return {
        "candle_start": start,
        "candle_end": end,
        "expected_alert_time": expected,
        "run_time": run_time,
    }


# =========================================================
# INDICATORS
# =========================================================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def add_indicators(df):
    if df is None or df.empty:
        return df

    df = df.copy()

    close = df["trade_price"].astype(float)
    high = df["high_price"].astype(float)
    low = df["low_price"].astype(float)
    open_ = df["opening_price"].astype(float)
    volume = df["candle_acc_trade_volume"].astype(float)
    trade_value = df["candle_acc_trade_price"].astype(float)

    df["rsi"] = calc_rsi(close, 14)

    rsi_min = df["rsi"].rolling(14, min_periods=14).min()
    rsi_max = df["rsi"].rolling(14, min_periods=14).max()
    stoch_rsi = (df["rsi"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    df["stoch_k"] = stoch_rsi.rolling(3, min_periods=1).mean().fillna(50)
    df["stoch_d"] = df["stoch_k"].rolling(3, min_periods=1).mean().fillna(50)

    rsi_min_mid = df["rsi"].rolling(28, min_periods=14).min()
    rsi_max_mid = df["rsi"].rolling(28, min_periods=14).max()
    stoch_mid = (df["rsi"] - rsi_min_mid) / (rsi_max_mid - rsi_min_mid).replace(0, np.nan) * 100
    df["mid_k"] = stoch_mid.rolling(3, min_periods=1).mean().fillna(50)

    df["ma20"] = close.rolling(20, min_periods=1).mean()
    df["ma20_gap"] = ((close / df["ma20"]) - 1) * 100

    df["volume_ma20"] = volume.rolling(20, min_periods=1).mean()
    df["volume_ratio"] = volume / df["volume_ma20"].replace(0, np.nan)

    df["trade_value"] = trade_value
    df["candle_change"] = ((close / open_) - 1) * 100
    df["change_3"] = ((close / close.shift(3)) - 1) * 100

    candle_range = (high - low).replace(0, np.nan)
    df["close_position"] = ((close - low) / candle_range).fillna(0.5)
    df["upper_wick"] = ((high - close) / candle_range).fillna(0.0)

    df["body_ratio"] = ((close - open_).abs() / candle_range).fillna(0.0)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


# =========================================================
# BTC FILTER
# =========================================================
def check_btc_filter():
    meta = {
        "btc_filter_pass": False,
        "btc_change_4h": 0.0,
        "btc_ma20_gap": 0.0,
        "btc_state_text": "BTC 데이터 없음",
    }

    try:
        df = fetch_candles("KRW-BTC", interval="minute240", count=120)
        df = remove_incomplete_current_candle(df)
        df = add_indicators(df)

        if df.empty:
            return meta

        latest = df.iloc[-1]

        btc_change = safe_float(latest.get("candle_change"))
        btc_ma20_gap = safe_float(latest.get("ma20_gap"))

        # 너무 심한 약세만 차단
        btc_pass = True
        if btc_change <= -2.0:
            btc_pass = False
        if btc_ma20_gap <= -3.5:
            btc_pass = False

        meta["btc_filter_pass"] = btc_pass
        meta["btc_change_4h"] = btc_change
        meta["btc_ma20_gap"] = btc_ma20_gap
        meta["btc_state_text"] = (
            f"BTC 필터 {'통과' if btc_pass else '차단'} / "
            f"4H {btc_change:+.2f}% / MA20 {btc_ma20_gap:+.2f}%"
        )

        return meta

    except Exception as e:
        print(f"[WARN] BTC filter failed: {type(e).__name__}: {e}")
        meta["btc_state_text"] = f"BTC 필터 확인 실패: {type(e).__name__}"
        return meta


# =========================================================
# SCORING / FILTER
# =========================================================
def calculate_score(row):
    score = 0

    trade_value = safe_float(row.get("trade_value"))
    volume_ratio = safe_float(row.get("volume_ratio"))
    candle_change = safe_float(row.get("candle_change"))
    change_3 = safe_float(row.get("change_3"))
    stoch_k = safe_float(row.get("stoch_k"))
    stoch_d = safe_float(row.get("stoch_d"))
    mid_k = safe_float(row.get("mid_k"))
    ma20_gap = safe_float(row.get("ma20_gap"))
    close_position = safe_float(row.get("close_position"))
    upper_wick = safe_float(row.get("upper_wick"))

    # 거래대금
    if trade_value >= 1_500_000_000:
        score += 45
    elif trade_value >= 700_000_000:
        score += 35
    elif trade_value >= 300_000_000:
        score += 25
    elif trade_value >= MIN_4H_TRADE_VALUE:
        score += 15

    # 거래량 증가
    if volume_ratio >= 5:
        score += 45
    elif volume_ratio >= 3:
        score += 35
    elif volume_ratio >= 2:
        score += 25
    elif volume_ratio >= 1.5:
        score += 15

    # 캔들 상승률
    if candle_change >= 5:
        score += 45
    elif candle_change >= 3:
        score += 35
    elif candle_change >= 2:
        score += 25
    elif candle_change >= MIN_CANDLE_CHANGE:
        score += 15

    # 3봉 흐름
    if change_3 >= 5:
        score += 35
    elif change_3 >= 3:
        score += 25
    elif change_3 >= 1:
        score += 15
    elif change_3 >= 0:
        score += 5

    # StochRSI
    if stoch_k > stoch_d and 20 <= stoch_k <= 80:
        score += 30
    elif stoch_k > stoch_d:
        score += 20
    elif abs(stoch_k - stoch_d) <= 3 and stoch_k >= 45:
        score += 10

    # 중기 K
    if 35 <= mid_k <= 75:
        score += 25
    elif 25 <= mid_k <= 85:
        score += 15

    # MA20 이격
    if 0 <= ma20_gap <= 4:
        score += 35
    elif -1 <= ma20_gap <= 6:
        score += 25
    elif 0 <= ma20_gap <= 8:
        score += 15

    # 종가 위치
    if close_position >= 0.85:
        score += 30
    elif close_position >= 0.70:
        score += 20
    elif close_position >= 0.55:
        score += 10

    # 윗꼬리
    if upper_wick <= 0.15:
        score += 25
    elif upper_wick <= 0.25:
        score += 15
    elif upper_wick <= 0.35:
        score += 5

    return int(score)


def determine_grade(row, score):
    trade_value = safe_float(row.get("trade_value"))
    candle_change = safe_float(row.get("candle_change"))
    volume_ratio = safe_float(row.get("volume_ratio"))
    ma20_gap = safe_float(row.get("ma20_gap"))
    close_position = safe_float(row.get("close_position"))
    upper_wick = safe_float(row.get("upper_wick"))
    stoch_k = safe_float(row.get("stoch_k"))
    stoch_d = safe_float(row.get("stoch_d"))

    # A급: 수량은 적지만 안정형
    if (
        score >= 320
        and trade_value >= 300_000_000
        and candle_change >= 1.2
        and volume_ratio >= 2.0
        and 0 <= ma20_gap <= 5.5
        and close_position >= 0.70
        and upper_wick <= 0.30
    ):
        return "A"

    # B급
    if (
        score >= 250
        and trade_value >= MIN_4H_TRADE_VALUE
        and candle_change >= MIN_CANDLE_CHANGE
        and close_position >= 0.55
        and upper_wick <= 0.45
        and ma20_gap <= 8.0
        and stoch_k >= 35
    ):
        return "B"

    return "WATCH"


def pass_confirmed_filter(row):
    trade_value = safe_float(row.get("trade_value"))
    candle_change = safe_float(row.get("candle_change"))
    ma20_gap = safe_float(row.get("ma20_gap"))
    close_position = safe_float(row.get("close_position"))
    upper_wick = safe_float(row.get("upper_wick"))
    stoch_k = safe_float(row.get("stoch_k"))

    if trade_value < MIN_4H_TRADE_VALUE:
        return False

    if candle_change < MIN_CANDLE_CHANGE:
        return False

    # 과열/위험 배제
    if candle_change >= 12:
        return False

    if ma20_gap >= 12:
        return False

    if upper_wick > 0.50:
        return False

    if close_position < 0.50:
        return False

    if stoch_k >= 95:
        return False

    return True


# =========================================================
# SIGNAL STATE
# =========================================================
def load_signal_state():
    if not ENABLE_SIGNAL_TRACKING:
        return {"active": [], "closed": []}

    if not os.path.exists(SIGNAL_STATE_FILE):
        state = {"active": [], "closed": []}
        save_signal_state(state)
        return state

    try:
        with open(SIGNAL_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        if not isinstance(state, dict):
            state = {"active": [], "closed": []}

        if "active" not in state or not isinstance(state["active"], list):
            state["active"] = []

        if "closed" not in state or not isinstance(state["closed"], list):
            state["closed"] = []

        return state

    except Exception as e:
        print(f"[WARN] load_signal_state failed: {type(e).__name__}: {e}")
        return {"active": [], "closed": []}


def save_signal_state(state):
    if not ENABLE_SIGNAL_TRACKING:
        return

    try:
        with open(SIGNAL_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"Saved signal state: {SIGNAL_STATE_FILE}")
    except Exception as e:
        print(f"[WARN] save_signal_state failed: {type(e).__name__}: {e}")


def is_duplicate_or_cooldown(state, market):
    now = now_kst()

    for s in state.get("active", []):
        if s.get("market") == market:
            return True, "active_duplicate"

    for s in reversed(state.get("closed", [])[-200:]):
        if s.get("market") != market:
            continue

        closed_at = parse_kst_datetime(s.get("closed_at"))
        if not closed_at:
            continue

        diff_hours = (now - closed_at).total_seconds() / 3600
        if diff_hours < SIGNAL_COOLDOWN_HOURS:
            return True, "cooldown"

    return False, ""


def get_candidate_entry_price(c):
    for key in ["entry_price", "price", "trade_price", "close", "signal_price"]:
        if key in c and c.get(key) is not None:
            return safe_float(c.get(key))
    return 0.0


def get_candidate_tp_price(c, entry_price):
    for key in ["tp_price", "take_profit_price", "target_price"]:
        if key in c and c.get(key) is not None:
            return safe_float(c.get(key))
    return entry_price * (1 + TAKE_PROFIT_RATE)


def get_candidate_sl_price(c, entry_price):
    for key in ["sl_price", "stop_loss_price"]:
        if key in c and c.get(key) is not None:
            return safe_float(c.get(key))
    return entry_price * (1 - STOP_LOSS_RATE)


def add_new_signals_to_state(state, candidates):
    if not ENABLE_SIGNAL_TRACKING:
        return 0

    added = 0

    for c in candidates:
        market = c.get("market")
        if not market:
            continue

        grade = (c.get("alert_grade") or c.get("grade") or "").upper()
        if grade not in ALLOWED_ALERT_GRADES:
            continue

        dup, reason = is_duplicate_or_cooldown(state, market)
        if dup:
            print(f"Skip {market}: {reason}")
            continue

        entry_price = get_candidate_entry_price(c)
        if entry_price <= 0:
            continue

        tp_price = get_candidate_tp_price(c, entry_price)
        sl_price = get_candidate_sl_price(c, entry_price)

        recommended_at = now_kst()
        expire_at = recommended_at + timedelta(hours=MAX_HOLD_HOURS)

        signal = {
            "market": market,
            "grade": grade,
            "score": int(safe_float(c.get("score"))),
            "status": "active",

            "entry_price": entry_price,
            "current_price": safe_float(c.get("current_price"), entry_price),
            "tp_price": tp_price,
            "sl_price": sl_price,

            "take_profit_rate": TAKE_PROFIT_RATE,
            "stop_loss_rate": STOP_LOSS_RATE,

            "recommended_at": format_kst(recommended_at),
            "expire_at": format_kst(expire_at),

            "signal_candle_start": format_kst(c.get("signal_candle_start")),
            "signal_candle_end": format_kst(c.get("signal_candle_end")),

            "max_profit_rate": 0.0,
            "min_profit_rate": 0.0,
            "current_profit_rate": 0.0,

            "notice_flags": {},

            "raw": {
                "trade_value": safe_float(c.get("trade_value")),
                "volume_ratio": safe_float(c.get("volume_ratio")),
                "candle_change": safe_float(c.get("candle_change")),
                "change_3": safe_float(c.get("change_3")),
                "stoch_k": safe_float(c.get("stoch_k")),
                "stoch_d": safe_float(c.get("stoch_d")),
                "mid_k": safe_float(c.get("mid_k")),
                "ma20_gap": safe_float(c.get("ma20_gap")),
                "close_position": safe_float(c.get("close_position")),
                "upper_wick": safe_float(c.get("upper_wick")),
            },
        }

        state.setdefault("active", []).append(signal)
        added += 1

    return added


# =========================================================
# ACTIVE SIGNAL CHECK
# =========================================================
def check_active_signals(state):
    active = state.get("active", [])

    if not active:
        return []

    markets = [s.get("market") for s in active if s.get("market")]
    ticker_map = fetch_ticker_map(markets)

    still_active = []
    closed_events = []
    closed_records = []

    now = now_kst()

    for s in active:
        market = s.get("market")
        ticker = ticker_map.get(market, {})

        current_price = safe_float(ticker.get("trade_price"))
        entry_price = safe_float(s.get("entry_price"))
        tp_price = safe_float(s.get("tp_price"))
        sl_price = safe_float(s.get("sl_price"))

        if current_price <= 0 or entry_price <= 0:
            still_active.append(s)
            continue

        profit_rate = (current_price / entry_price) - 1

        prev_max = safe_float(s.get("max_profit_rate"), profit_rate)
        prev_min = safe_float(s.get("min_profit_rate"), profit_rate)

        s["current_price"] = current_price
        s["current_profit_rate"] = profit_rate
        s["max_profit_rate"] = max(prev_max, profit_rate)
        s["min_profit_rate"] = min(prev_min, profit_rate)
        s["last_checked_at"] = format_kst(now)

        expire_at = parse_kst_datetime(s.get("expire_at"))

        result = None

        if current_price >= tp_price:
            result = "TP"
        elif current_price <= sl_price:
            result = "SL"
        elif expire_at and now >= expire_at:
            result = "EXPIRED"

        if result:
            event = {
                "market": market,
                "grade": s.get("grade", "-"),
                "result": result,
                "entry_price": entry_price,
                "current_price": current_price,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "profit_rate": profit_rate,
                "max_profit_rate": safe_float(s.get("max_profit_rate")),
                "min_profit_rate": safe_float(s.get("min_profit_rate")),
                "recommended_at": s.get("recommended_at"),
                "closed_at": format_kst(now),
            }

            closed_events.append(event)

            closed_record = dict(s)
            closed_record["status"] = "closed"
            closed_record["result"] = result
            closed_record["closed_at"] = format_kst(now)
            closed_record["final_price"] = current_price
            closed_record["final_profit_rate"] = profit_rate

            closed_records.append(closed_record)

        else:
            still_active.append(s)

    state["active"] = still_active
    state.setdefault("closed", []).extend(closed_records)

    # closed 너무 커지는 것 방지
    if len(state["closed"]) > 500:
        state["closed"] = state["closed"][-500:]

    return closed_events


def build_closed_events_message(events):
    if not events:
        return ""

    lines = []
    lines.append(f"📌 TP/SL/만료 결과 알림 {BOT_VERSION}")
    lines.append(f"현재시간: {format_kst(now_kst())}")
    lines.append("")

    for idx, e in enumerate(events, start=1):
        result = e.get("result")
        icon = "✅" if result == "TP" else "❌" if result == "SL" else "⏰"

        lines.append(f"{idx}. {e.get('market')} {icon} {result} / {e.get('grade')}급")
        lines.append(
            f"추천가 {format_price(e.get('entry_price'))} / "
            f"종료가 {format_price(e.get('current_price'))}"
        )
        lines.append(
            f"최종수익률 {e.get('profit_rate', 0) * 100:+.2f}% / "
            f"최고 {e.get('max_profit_rate', 0) * 100:+.2f}% / "
            f"최저 {e.get('min_profit_rate', 0) * 100:+.2f}%"
        )
        lines.append(
            f"TP {format_price(e.get('tp_price'))} / "
            f"SL {format_price(e.get('sl_price'))}"
        )
        lines.append(f"추천시간: {e.get('recommended_at')}")
        lines.append(f"종료시간: {e.get('closed_at')}")
        lines.append("")

    return "\n".join(lines)


# =========================================================
# V3.8 TRACKING NOTICE
# =========================================================
def ensure_signal_notice_flags(signal):
    if "notice_flags" not in signal or not isinstance(signal.get("notice_flags"), dict):
        signal["notice_flags"] = {}
    return signal["notice_flags"]


def check_tracking_notices(state):
    active = state.get("active", [])

    if not active:
        return []

    markets = [s.get("market") for s in active if s.get("market")]
    ticker_map = fetch_ticker_map(markets)

    notices = []

    for s in active:
        market = s.get("market")
        ticker = ticker_map.get(market, {})
        current_price = safe_float(ticker.get("trade_price"))

        if current_price <= 0:
            continue

        entry_price = safe_float(s.get("entry_price"))
        tp_price = safe_float(s.get("tp_price"), entry_price * (1 + TAKE_PROFIT_RATE))
        sl_price = safe_float(s.get("sl_price"), entry_price * (1 - STOP_LOSS_RATE))

        if entry_price <= 0:
            continue

        profit_rate = (current_price / entry_price) - 1
        tp_distance = (tp_price / current_price) - 1
        sl_distance = (sl_price / current_price) - 1

        prev_max_profit = safe_float(s.get("max_profit_rate"), profit_rate)
        prev_min_profit = safe_float(s.get("min_profit_rate"), profit_rate)

        max_profit_rate = max(prev_max_profit, profit_rate)
        min_profit_rate = min(prev_min_profit, profit_rate)

        s["current_price"] = current_price
        s["current_profit_rate"] = profit_rate
        s["max_profit_rate"] = max_profit_rate
        s["min_profit_rate"] = min_profit_rate
        s["last_checked_at"] = format_kst(now_kst())

        flags = ensure_signal_notice_flags(s)

        grade = s.get("grade") or s.get("alert_grade") or "-"

        base = {
            "market": market,
            "grade": grade,
            "entry_price": entry_price,
            "current_price": current_price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "profit_rate": profit_rate,
            "tp_distance": tp_distance,
            "sl_distance": sl_distance,
            "max_profit_rate": max_profit_rate,
            "min_profit_rate": min_profit_rate,
            "recommended_at": s.get("recommended_at"),
        }

        # +2% 도달
        if profit_rate >= TRACK_PROFIT_NOTICE_1 and not flags.get("profit_2"):
            notice = dict(base)
            notice["type"] = "profit_2"
            notice["title"] = "B급 일부익절 검토 구간"
            notices.append(notice)
            flags["profit_2"] = True

        # +3% 도달
        if profit_rate >= TRACK_PROFIT_NOTICE_2 and not flags.get("profit_3"):
            notice = dict(base)
            notice["type"] = "profit_3"
            notice["title"] = "수익 확대 / 본절스탑 검토"
            notices.append(notice)
            flags["profit_3"] = True

        # TP 근접
        if 0 <= tp_distance <= TRACK_NEAR_TP_RATE and not flags.get("near_tp"):
            notice = dict(base)
            notice["type"] = "near_tp"
            notice["title"] = "TP 근접"
            notices.append(notice)
            flags["near_tp"] = True

        # SL 근접
        if 0 <= abs(sl_distance) <= TRACK_NEAR_SL_RATE and current_price > sl_price and not flags.get("near_sl"):
            notice = dict(base)
            notice["type"] = "near_sl"
            notice["title"] = "SL 근접 주의"
            notices.append(notice)
            flags["near_sl"] = True

        # 고점 대비 수익 반납
        drawdown_from_high = max_profit_rate - profit_rate
        if (
            max_profit_rate >= TRACK_PROFIT_NOTICE_1
            and drawdown_from_high >= TRACK_DRAWDOWN_FROM_HIGH
            and not flags.get("drawdown_from_high")
        ):
            notice = dict(base)
            notice["type"] = "drawdown_from_high"
            notice["title"] = "상승분 반납 주의"
            notice["drawdown_from_high"] = drawdown_from_high
            notices.append(notice)
            flags["drawdown_from_high"] = True

    return notices


def build_tracking_notices_message(notices):
    if not notices:
        return ""

    lines = []
    lines.append(f"📡 30분 추적 알림 {BOT_VERSION}")
    lines.append(f"현재시간: {format_kst(now_kst())}")
    lines.append("")

    for idx, n in enumerate(notices, start=1):
        market = n.get("market")
        grade = n.get("grade", "-")
        title = n.get("title", "-")

        entry_price = n.get("entry_price")
        current_price = n.get("current_price")
        tp_price = n.get("tp_price")
        sl_price = n.get("sl_price")

        profit_pct = n.get("profit_rate", 0) * 100
        tp_distance_pct = n.get("tp_distance", 0) * 100
        sl_distance_pct = n.get("sl_distance", 0) * 100
        max_profit_pct = n.get("max_profit_rate", 0) * 100

        lines.append(f"{idx}. {market} 🟡 {grade}급")
        lines.append(f"알림: {title}")
        lines.append(
            f"추천가 {format_price(entry_price)} / 현재가 {format_price(current_price)}"
        )
        lines.append(
            f"현재수익률 {profit_pct:+.2f}% / 최고수익률 {max_profit_pct:+.2f}%"
        )
        lines.append(
            f"TP {format_price(tp_price)}까지 {tp_distance_pct:+.2f}% / "
            f"SL {format_price(sl_price)}까지 {sl_distance_pct:+.2f}%"
        )

        if n.get("type") == "profit_2":
            lines.append("판정: +2% 이상 도달. B급은 일부익절 검토 가능.")
        elif n.get("type") == "profit_3":
            lines.append("판정: +3% 이상 도달. 일부익절 또는 본절스탑 검토.")
        elif n.get("type") == "near_tp":
            lines.append("판정: TP 근접. 익절 체결 여부 확인.")
        elif n.get("type") == "near_sl":
            lines.append("판정: SL 근접. 리스크 관리 필요.")
        elif n.get("type") == "drawdown_from_high":
            dd = n.get("drawdown_from_high", 0) * 100
            lines.append(f"판정: 최고수익률 대비 {dd:.2f}%p 반납. 상승 힘 약화 주의.")

        lines.append("")

    lines.append("※ 30분 추적 알림은 신규 매수 신호가 아니라 기존 active 신호 관리용입니다.")

    return "\n".join(lines)


# =========================================================
# LIVE PRICE ENRICH
# =========================================================
def judge_entry_by_gap(gap_pct):
    if gap_pct <= -1.0:
        return "⚪ 기준가 이하/약세 확인 필요"
    elif gap_pct <= ENTRY_OK_MAX_GAP:
        return "🟢 진입 가능권"
    elif gap_pct <= ENTRY_CAUTION_MAX_GAP:
        return "🟡 소액 또는 눌림 대기"
    elif gap_pct <= ENTRY_CHASE_MAX_GAP:
        return "🟠 추격주의"
    else:
        return "🔴 신규진입 비추천/눌림 대기"


def enrich_candidates_with_live_prices(candidates):
    if not ENABLE_LIVE_PRICE_CHECK or not candidates:
        return candidates

    markets = [c.get("market") for c in candidates if c.get("market")]
    ticker_map = fetch_ticker_map(markets)

    enriched = []

    for c in candidates:
        market = c.get("market")
        ticker = ticker_map.get(market, {})

        entry_price = get_candidate_entry_price(c)
        current_price = safe_float(ticker.get("trade_price"), entry_price)

        tp_price = get_candidate_tp_price(c, entry_price)
        sl_price = get_candidate_sl_price(c, entry_price)

        if entry_price > 0 and current_price > 0:
            entry_gap_pct = ((current_price / entry_price) - 1) * 100
            current_to_tp_pct = ((tp_price / current_price) - 1) * 100
            current_to_sl_pct = ((sl_price / current_price) - 1) * 100
        else:
            entry_gap_pct = 0.0
            current_to_tp_pct = 0.0
            current_to_sl_pct = 0.0

        c["entry_price"] = entry_price
        c["current_price"] = current_price
        c["tp_price"] = tp_price
        c["sl_price"] = sl_price

        c["entry_gap_pct"] = entry_gap_pct
        c["current_to_tp_pct"] = current_to_tp_pct
        c["current_to_sl_pct"] = current_to_sl_pct
        c["entry_judgement"] = judge_entry_by_gap(entry_gap_pct)

        enriched.append(c)

    return enriched


# =========================================================
# MARKET SCAN
# =========================================================
def build_candidate_from_row(market, row):
    entry_price = safe_float(row.get("trade_price"))
    score = calculate_score(row)
    grade = determine_grade(row, score)

    signal_start = row.get("candle_time")
    signal_end = signal_start + timedelta(hours=4) if signal_start else None

    c = {
        "market": market,
        "alert_grade": grade,
        "grade": grade,
        "score": score,

        "entry_price": entry_price,
        "price": entry_price,
        "tp_price": entry_price * (1 + TAKE_PROFIT_RATE),
        "sl_price": entry_price * (1 - STOP_LOSS_RATE),

        "trade_value": safe_float(row.get("trade_value")),
        "volume_ratio": safe_float(row.get("volume_ratio")),
        "candle_change": safe_float(row.get("candle_change")),
        "change_3": safe_float(row.get("change_3")),
        "stoch_k": safe_float(row.get("stoch_k")),
        "stoch_d": safe_float(row.get("stoch_d")),
        "mid_k": safe_float(row.get("mid_k")),
        "ma20_gap": safe_float(row.get("ma20_gap")),
        "close_position": safe_float(row.get("close_position")),
        "upper_wick": safe_float(row.get("upper_wick")),

        "signal_candle_start": signal_start,
        "signal_candle_end": signal_end,
    }

    return c


def scan_markets():
    run_time = now_kst()

    meta = {
        "run_time": run_time,
        "candle_start": None,
        "candle_end": None,
        "expected_alert_time": None,
        "btc_state_text": "-",
        "btc_filter_pass": False,
        "btc_change_4h": 0.0,
        "btc_ma20_gap": 0.0,
        "scan_count": 0,
        "filter_pass_count": 0,
        "ab_count": 0,
        "watch_excluded_count": 0,
        "display_count": 0,
        "max_alert_count": MAX_ALERT_COUNT,
    }

    btc_meta = check_btc_filter()
    meta.update(btc_meta)

    markets = fetch_krw_markets()
    meta["scan_count"] = len(markets)

    candidates = []
    filter_pass_count = 0
    watch_excluded_count = 0

    print(f"Scanning {len(markets)} KRW markets...")

    for idx, market in enumerate(markets, start=1):
        try:
            df = fetch_candles(market, interval=ALERT_INTERVAL, count=ALERT_CANDLE_COUNT)
            df = remove_incomplete_current_candle(df)

            if df is None or len(df) < 30:
                time.sleep(REQUEST_DELAY)
                continue

            if meta["candle_start"] is None:
                m = get_target_candle_meta_from_df(df)
                meta.update(m)

            df = add_indicators(df)
            row = df.iloc[-1]

            if not meta.get("btc_filter_pass", False):
                time.sleep(REQUEST_DELAY)
                continue

            if not pass_confirmed_filter(row):
                time.sleep(REQUEST_DELAY)
                continue

            filter_pass_count += 1

            c = build_candidate_from_row(market, row)

            if c["alert_grade"] not in ALLOWED_ALERT_GRADES:
                watch_excluded_count += 1
                time.sleep(REQUEST_DELAY)
                continue

            candidates.append(c)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"[WARN] scan failed {market}: {type(e).__name__}: {e}")
            time.sleep(REQUEST_DELAY)

    grade_order = {"A": 0, "B": 1, "WATCH": 2}

    candidates = sorted(
        candidates,
        key=lambda x: (
            grade_order.get(x.get("alert_grade"), 9),
            -safe_float(x.get("score")),
            -safe_float(x.get("trade_value")),
            -safe_float(x.get("volume_ratio")),
        ),
    )

    meta["filter_pass_count"] = filter_pass_count
    meta["watch_excluded_count"] = watch_excluded_count
    meta["ab_count"] = len(candidates)

    candidates = candidates[:MAX_ALERT_COUNT]
    meta["display_count"] = len(candidates)

    return candidates, meta


# =========================================================
# MESSAGE BUILDERS
# =========================================================
def format_alert_delay(meta):
    try:
        expected = meta.get("expected_alert_time")
        run_time = meta.get("run_time")

        expected_dt = parse_kst_datetime(expected)
        run_dt = parse_kst_datetime(run_time)

        if not expected_dt or not run_dt:
            return ""

        delay_min = (run_dt - expected_dt).total_seconds() / 60
        return f"알림 지연: {delay_min:+.0f}분"

    except Exception:
        return ""


def format_candidate_message_v38(idx, c):
    market = c.get("market", "-")
    grade = c.get("alert_grade") or c.get("grade") or "-"
    score = int(safe_float(c.get("score")))

    entry_price = get_candidate_entry_price(c)
    current_price = safe_float(c.get("current_price"), entry_price)

    tp_price = get_candidate_tp_price(c, entry_price)
    sl_price = get_candidate_sl_price(c, entry_price)

    entry_gap_pct = safe_float(c.get("entry_gap_pct"), 0.0)
    current_to_tp_pct = safe_float(c.get("current_to_tp_pct"), 0.0)
    current_to_sl_pct = safe_float(c.get("current_to_sl_pct"), 0.0)
    entry_judgement = c.get("entry_judgement", "-")

    trade_value = safe_float(c.get("trade_value"))
    volume_ratio = safe_float(c.get("volume_ratio"))
    candle_change = safe_float(c.get("candle_change"))
    change_3 = safe_float(c.get("change_3"))
    k = safe_float(c.get("stoch_k"))
    d = safe_float(c.get("stoch_d"))
    mid_k = safe_float(c.get("mid_k"))
    ma20_gap = safe_float(c.get("ma20_gap"))
    close_position = safe_float(c.get("close_position"))
    upper_wick = safe_float(c.get("upper_wick"))

    if grade == "A":
        grade_icon = "🔥"
    elif grade == "B":
        grade_icon = "🟡"
    else:
        grade_icon = "⚪"

    lines = []
    lines.append(f"{idx}. {market} {grade_icon} {grade}급 / score {score}")
    lines.append(
        f"기준가 {format_price(entry_price)} / 현재가 {format_price(current_price)} "
        f"({entry_gap_pct:+.2f}%)"
    )
    lines.append(f"TP {format_price(tp_price)} / SL {format_price(sl_price)}")
    lines.append(
        f"현재가 기준: TP까지 {current_to_tp_pct:+.2f}% / "
        f"SL까지 {current_to_sl_pct:+.2f}%"
    )
    lines.append(f"진입판정: {entry_judgement}")
    lines.append(
        f"4H거래대금 {trade_value / 100000000:.1f}억 / "
        f"거래비 {volume_ratio:.2f}x / "
        f"캔들 {candle_change:+.2f}% / "
        f"3봉 {change_3:+.2f}%"
    )
    lines.append(
        f"K/D {k:.1f}/{d:.1f} / "
        f"중기K {mid_k:.1f} / "
        f"MA20 {ma20_gap:+.2f}%"
    )
    lines.append(
        f"종가위치 {close_position:.2f} / "
        f"윗꼬리 {upper_wick:.2f}"
    )

    return "\n".join(lines)


def build_active_signals_summary(state):
    active = state.get("active", [])

    if not active:
        return ""

    lines = []
    lines.append("📊 48시간 추적 중 신호")
    lines.append(f"현재시간: {format_kst(now_kst())}")
    lines.append("")

    sorted_active = sorted(
        active,
        key=lambda x: safe_float(x.get("current_profit_rate")),
        reverse=True,
    )

    for idx, s in enumerate(sorted_active, start=1):
        market = s.get("market")
        grade = s.get("grade", "-")

        entry_price = safe_float(s.get("entry_price"))
        current_price = safe_float(s.get("current_price"), entry_price)
        tp_price = safe_float(s.get("tp_price"))
        sl_price = safe_float(s.get("sl_price"))

        if entry_price > 0 and current_price > 0:
            profit_pct = ((current_price / entry_price) - 1) * 100
            tp_distance = ((tp_price / current_price) - 1) * 100 if tp_price > 0 else 0
            sl_distance = ((sl_price / current_price) - 1) * 100 if sl_price > 0 else 0
        else:
            profit_pct = 0
            tp_distance = 0
            sl_distance = 0

        recommended_at = parse_kst_datetime(s.get("recommended_at"))
        elapsed_hours = 0.0
        if recommended_at:
            elapsed_hours = (now_kst() - recommended_at).total_seconds() / 3600

        icon = "🟢" if profit_pct >= 2 else "🟡" if profit_pct < 0 else "⚪"

        lines.append(f"{idx}. {market} {icon} 🟡 {grade}급")
        lines.append(f"추천가: {format_price(entry_price)} / 현재가: {format_price(current_price)}")
        lines.append(f"현재수익률: {profit_pct:+.2f}% / 경과: {elapsed_hours:.1f}시간")
        lines.append(
            f"TP {format_price(tp_price)}까지 {tp_distance:+.2f}% / "
            f"SL {format_price(sl_price)}까지 {sl_distance:+.2f}%"
        )
        lines.append(f"추천시간: {s.get('recommended_at')}")
        lines.append(
            f"추천 기준봉: {s.get('signal_candle_start', '-')} ~ "
            f"{s.get('signal_candle_end', '-')}"
        )
        lines.append("")

    return "\n".join(lines).strip()


def build_telegram_message(candidates, meta, active_summary=""):
    lines = []

    lines.append(f"🟢 4H 마감 후 상승전환 확정 후보 {BOT_VERSION}")
    lines.append("A/B 등급만 표시 · WATCH 제외 · 확정봉 기준")
    lines.append("")

    lines.append("🕯 분석 기준봉")
    lines.append(f"시작: {format_kst(meta.get('candle_start'))}")
    lines.append(f"마감: {format_kst(meta.get('candle_end'))}")
    lines.append(f"실행: {format_kst(meta.get('run_time'))}")
    lines.append(f"예상 알림 기준: {format_kst(meta.get('expected_alert_time'))}")

    delay_text = format_alert_delay(meta)
    if delay_text:
        lines.append(delay_text)

    lines.append("※ 진행 중인 최신 4H 봉은 제외하고 직전 확정봉만 분석")
    lines.append("※ 기준가는 4H 확정봉 종가이며, 현재가는 알림 발송 시점 기준")
    lines.append("")

    lines.append(f"BTC 상태: {meta.get('btc_state_text', '-')}")
    lines.append(
        f"스캔: {meta.get('scan_count', 0)}개 / "
        f"필터통과: {meta.get('filter_pass_count', 0)}개"
    )
    lines.append(
        f"A/B 후보: {meta.get('ab_count', 0)}개 / "
        f"WATCH 제외: {meta.get('watch_excluded_count', 0)}개"
    )
    lines.append(
        f"표시: 상위 {meta.get('display_count', 0)}개 / "
        f"최대 {meta.get('max_alert_count', MAX_ALERT_COUNT)}개"
    )
    lines.append(
        f"전략 TP/SL: +{TAKE_PROFIT_RATE * 100:.1f}% / "
        f"-{STOP_LOSS_RATE * 100:.1f}% / {MAX_HOLD_HOURS:.0f}시간"
    )
    lines.append("")

    a_list = [c for c in candidates if c.get("alert_grade") == "A"]
    b_list = [c for c in candidates if c.get("alert_grade") == "B"]

    if a_list:
        lines.append("🔥 A급 후보")
        for idx, c in enumerate(a_list, start=1):
            lines.append(format_candidate_message_v38(idx, c))
            lines.append("")

    if b_list:
        lines.append("🟡 B급 후보")
        for idx, c in enumerate(b_list, start=1):
            lines.append(format_candidate_message_v38(idx, c))
            lines.append("")

    if active_summary:
        lines.append("")
        lines.append(active_summary)
        lines.append("")

    lines.append("※ 백테스트 기준: TOP_10_A_B / TP +5%, SL -4% 조합 우선 적용")
    lines.append("※ 알림은 매수 강요가 아니며, 진입 전 BTC 흐름/저항/호가를 반드시 확인하세요.")
    lines.append("※ 실제 체결, 슬리피지, 수수료는 반영되지 않습니다.")

    return "\n".join(lines)


def build_empty_message(meta, active_summary=""):
    lines = []

    lines.append(f"🟢 4H 마감 후 상승전환 확정 후보 {BOT_VERSION}")
    lines.append("조건 만족 A/B 후보 없음")
    lines.append("")

    lines.append("🕯 분석 기준봉")
    lines.append(f"시작: {format_kst(meta.get('candle_start'))}")
    lines.append(f"마감: {format_kst(meta.get('candle_end'))}")
    lines.append(f"실행: {format_kst(meta.get('run_time'))}")
    lines.append(f"예상 알림 기준: {format_kst(meta.get('expected_alert_time'))}")

    delay_text = format_alert_delay(meta)
    if delay_text:
        lines.append(delay_text)

    lines.append("※ 진행 중인 최신 4H 봉은 제외하고 직전 확정봉만 분석")
    lines.append("")

    lines.append(f"BTC 상태: {meta.get('btc_state_text', '-')}")
    lines.append(
        f"스캔: {meta.get('scan_count', 0)}개 / "
        f"필터통과: {meta.get('filter_pass_count', 0)}개"
    )
    lines.append(
        f"A/B 후보: {meta.get('ab_count', 0)}개 / "
        f"WATCH 제외: {meta.get('watch_excluded_count', 0)}개"
    )
    lines.append("현재 조건에서는 관망 권고.")
    lines.append("")

    if active_summary:
        lines.append(active_summary)
        lines.append("")

    lines.append("※ 알림은 매수 강요가 아니며, 진입 전 BTC 흐름/저항/호가를 반드시 확인하세요.")

    return "\n".join(lines)


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print(f"Telegram Coin Alert Bot {BOT_VERSION} started")
    print(f"RUN_MODE={RUN_MODE}")
    print(f"Now KST: {format_kst(now_kst())}")
    print(f"ALERT_INTERVAL={ALERT_INTERVAL}")
    print(f"ALERT_CANDLE_COUNT={ALERT_CANDLE_COUNT}")
    print(f"MAX_ALERT_COUNT={MAX_ALERT_COUNT}")
    print(f"SEND_EMPTY_ALERT={SEND_EMPTY_ALERT}")
    print(f"ENABLE_SIGNAL_TRACKING={ENABLE_SIGNAL_TRACKING}")
    print(f"ENABLE_LIVE_PRICE_CHECK={ENABLE_LIVE_PRICE_CHECK}")
    print(f"TP={TAKE_PROFIT_RATE}, SL={STOP_LOSS_RATE}, HOLD={MAX_HOLD_HOURS}h")
    print("=" * 80)

    state = load_signal_state()

    # 1) 기존 active 신호 TP/SL/만료 체크
    closed_events = check_active_signals(state)

    if closed_events:
        closed_msg = build_closed_events_message(closed_events)
        print(closed_msg)
        send_telegram_message(closed_msg)

    # 2) 30분 추적 알림 체크
    tracking_notices = check_tracking_notices(state)

    if tracking_notices:
        tracking_msg = build_tracking_notices_message(tracking_notices)
        print(tracking_msg)
        send_telegram_message(tracking_msg)

    # track 모드는 신규 스캔 없이 종료
    if RUN_MODE == "track":
        save_signal_state(state)
        print("RUN_MODE=track. Skip new market scan.")
        print(f"Telegram Coin Alert Bot {BOT_VERSION} finished")
        return

    # 3) full 모드: active 요약
    active_summary = build_active_signals_summary(state)

    # 4) 신규 후보 스캔
    candidates, meta = scan_markets()

    # 5) 신규 후보에 현재가/괴리율/현재가 기준 TP/SL 거리 추가
    candidates = enrich_candidates_with_live_prices(candidates)

    if candidates:
        added = add_new_signals_to_state(state, candidates)
        print(f"Added new active signals: {added}")

        # 텔레그램 발송 전 저장
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

        save_signal_state(state)

    save_signal_state(state)
    print(f"Telegram Coin Alert Bot {BOT_VERSION} finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = (
            f"🚨 Telegram Coin Alert Bot {BOT_VERSION} error\n\n"
            f"{type(e).__name__}: {e}"
        )
        print(error_msg)
        send_telegram_message(error_msg)
        raise
