import os
import time
import json
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# =========================================================
# Backtest V1.1 for Telegram Coin Alert Strategy V3.6
# ---------------------------------------------------------
# 추가 기능:
# - 실제 알림처럼 동일 4H 시간대별 상위 MAX_ALERT_COUNT만 반영
# - A+B / A only / B only / WATCH 성과 별도 계산
# - TP/SL 조합 비교
# - CSV / JSON / Telegram 요약 출력
# =========================================================

UPBIT_BASE_URL = "https://api.upbit.com/v1"

# -----------------------------
# ENV CONFIG
# -----------------------------
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "60"))
BACKTEST_MAX_MARKETS = int(os.getenv("BACKTEST_MAX_MARKETS", "999"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.09"))

MAX_ALERT_COUNT = int(os.getenv("MAX_ALERT_COUNT", "10"))

TAKE_PROFIT_RATE = float(os.getenv("TAKE_PROFIT_RATE", "0.05"))
STOP_LOSS_RATE = float(os.getenv("STOP_LOSS_RATE", "0.04"))

MAX_HOLD_HOURS = int(os.getenv("MAX_HOLD_HOURS", "48"))
SIGNAL_COOLDOWN_HOURS = int(os.getenv("SIGNAL_COOLDOWN_HOURS", "12"))

MIN_4H_TRADE_VALUE = float(os.getenv("MIN_4H_TRADE_VALUE", "150000000"))
MIN_CANDLE_CHANGE = float(os.getenv("MIN_CANDLE_CHANGE", "0.7"))

SEND_TELEGRAM_BACKTEST = os.getenv("SEND_TELEGRAM_BACKTEST", "false").strip().lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RESULT_RAW_CSV_FILE = os.getenv("RESULT_RAW_CSV_FILE", "backtest_raw_results.csv")
RESULT_SELECTED_CSV_FILE = os.getenv("RESULT_SELECTED_CSV_FILE", "backtest_selected_results.csv")
SUMMARY_JSON_FILE = os.getenv("SUMMARY_JSON_FILE", "backtest_summary.json")

# TP/SL 비교 조합
# 형식: "0.04:0.04,0.045:0.04,0.05:0.04,0.04:0.035"
TP_SL_SETS_TEXT = os.getenv(
    "TP_SL_SETS",
    "0.04:0.04,0.045:0.04,0.05:0.04,0.04:0.035"
)

CANDLES_PER_DAY = 6
WARMUP_CANDLES = 120
HOLD_CANDLES = int(MAX_HOLD_HOURS / 4)

NEED_CANDLES = BACKTEST_DAYS * CANDLES_PER_DAY + WARMUP_CANDLES + HOLD_CANDLES + 30


# -----------------------------
# TIME / SAFE UTILS
# -----------------------------
def now_kst():
    return datetime.now(timezone.utc) + timedelta(hours=9)


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


def to_kst_text(dt):
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S KST")


def parse_tp_sl_sets(text):
    combos = []

    for part in text.split(","):
        part = part.strip()
        if not part:
            continue

        try:
            tp_text, sl_text = part.split(":")
            tp = float(tp_text.strip())
            sl = float(sl_text.strip())
            combos.append((tp, sl))
        except Exception:
            continue

    main_combo = (TAKE_PROFIT_RATE, STOP_LOSS_RATE)

    if main_combo not in combos:
        combos.insert(0, main_combo)

    # 중복 제거
    unique = []
    seen = set()
    for tp, sl in combos:
        key = f"{tp:.4f}:{sl:.4f}"
        if key not in seen:
            unique.append((tp, sl))
            seen.add(key)

    return unique


TP_SL_COMBOS = parse_tp_sl_sets(TP_SL_SETS_TEXT)


def combo_key(tp, sl):
    return f"TP{tp * 100:.1f}_SL{sl * 100:.1f}"


# -----------------------------
# TELEGRAM
# -----------------------------
def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/chat_id missing. Skipping Telegram send.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 3900

    ok = True
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]

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
            time.sleep(0.5)
        except Exception as e:
            print(f"Telegram exception: {e}")
            ok = False

    return ok


# -----------------------------
# UPBIT API
# -----------------------------
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

    markets = sorted(markets)

    if BACKTEST_MAX_MARKETS > 0:
        markets = markets[:BACKTEST_MAX_MARKETS]

    return markets


def fetch_4h_candles(market, need_count=NEED_CANDLES):
    all_rows = []
    to_value = None

    while len(all_rows) < need_count:
        count = min(200, need_count - len(all_rows))

        params = {
            "market": market,
            "count": count,
        }

        if to_value:
            params["to"] = to_value

        batch = upbit_get("/candles/minutes/240", params=params)
        time.sleep(REQUEST_DELAY)

        if not batch:
            break

        all_rows.extend(batch)

        if len(batch) < count:
            break

        oldest_utc = batch[-1].get("candle_date_time_utc")
        if not oldest_utc:
            break

        oldest_dt = datetime.fromisoformat(oldest_utc)
        next_to = oldest_dt - timedelta(seconds=1)
        to_value = next_to.strftime("%Y-%m-%d %H:%M:%S")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["candle_date_time_utc"])
    df["dt_utc"] = pd.to_datetime(df["candle_date_time_utc"], utc=True)
    df = df.sort_values("dt_utc").reset_index(drop=True)

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

    required_cols = ["dt_utc", "open", "high", "low", "close", "value", "volume"]

    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    for col in ["open", "high", "low", "close", "value", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[required_cols].reset_index(drop=True)


# -----------------------------
# INDICATORS
# -----------------------------
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


# -----------------------------
# BTC FILTER
# -----------------------------
def build_btc_filter_map():
    btc = fetch_4h_candles("KRW-BTC", NEED_CANDLES)

    if btc.empty:
        print("BTC candles empty. BTC filter treated as OK.")
        return {}

    btc = add_indicators(btc)

    result = {}

    for _, row in btc.iterrows():
        dt = row["dt_utc"]

        candle_change = safe_float(row.get("candle_change"), 0.0)
        ma20_dev = safe_float(row.get("ma20_dev"), 0.0)

        bullish_ok = True

        if candle_change <= -2.0:
            bullish_ok = False

        if ma20_dev <= -4.0:
            bullish_ok = False

        result[dt] = bullish_ok

    return result


def get_btc_ok_for_time(btc_map, dt):
    if not btc_map:
        return True
    return btc_map.get(dt, True)


# -----------------------------
# SCORE / FILTER / GRADE
# -----------------------------
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

    if value >= 500_000_000:
        score += 30
    elif value >= 300_000_000:
        score += 20
    elif value >= 150_000_000:
        score += 10
    else:
        score -= 30

    if 20 <= short_k <= 55 and short_d <= 55:
        score += 30
    elif 20 <= short_k <= 70 and short_d <= 60:
        score += 15
    elif short_k > 75 or short_d > 70:
        score -= 25

    if mid_k <= 55:
        score += 10
    elif mid_k > 70:
        score -= 15

    if -2.0 <= ma20_dev <= 2.0:
        score += 25
    elif -5.0 <= ma20_dev <= 5.0:
        score += 10
    else:
        score -= 25

    if 1.0 <= candle_change <= 3.5:
        score += 25
    elif 0.7 <= candle_change <= 4.0:
        score += 15
    elif candle_change > 4.0:
        score -= 30

    if -1.0 <= recent_3bar <= 4.0:
        score += 15
    elif -4.0 <= recent_3bar <= 6.0:
        score += 5
    elif recent_3bar > 6.0:
        score -= 25

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


def make_candidate_from_row(market, row, btc_ok):
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
        "bullish_ok": bool(btc_ok),
    }

    c["score"] = calculate_base_score(c)

    grade, grade_label = calculate_alert_grade(c)
    c["grade"] = grade
    c["grade_label"] = grade_label

    return c


# -----------------------------
# TRADE EVALUATION
# -----------------------------
def evaluate_trade(df, signal_idx, entry_price, tp_rate, sl_rate):
    tp_price = entry_price * (1.0 + tp_rate)
    sl_price = entry_price * (1.0 - sl_rate)

    future = df.iloc[signal_idx + 1: signal_idx + 1 + HOLD_CANDLES].copy()

    if future.empty:
        return None

    max_high = safe_float(future["high"].max())
    min_low = safe_float(future["low"].min())

    result = "EXPIRED"
    exit_price = safe_float(future.iloc[-1]["close"])
    exit_dt = future.iloc[-1]["dt_utc"]

    for _, row in future.iterrows():
        high = safe_float(row["high"])
        low = safe_float(row["low"])

        hit_tp = high >= tp_price
        hit_sl = low <= sl_price

        if hit_tp and hit_sl:
            # 같은 4H 봉 안에서 TP/SL 둘 다 닿으면 보수적으로 SL 처리
            result = "SL"
            exit_price = sl_price
            exit_dt = row["dt_utc"]
            break

        if hit_sl:
            result = "SL"
            exit_price = sl_price
            exit_dt = row["dt_utc"]
            break

        if hit_tp:
            result = "TP"
            exit_price = tp_price
            exit_dt = row["dt_utc"]
            break

    return {
        "result": result,
        "exit_price": exit_price,
        "exit_dt": exit_dt,
        "final_return": pct(exit_price, entry_price),
        "max_up": pct(max_high, entry_price),
        "max_down": pct(min_low, entry_price),
        "tp_price": tp_price,
        "sl_price": sl_price,
    }


def build_result_row(c, trade, tp_rate, sl_rate):
    dt = c["dt_utc"]

    return {
        "market": c["market"],
        "signal_time_utc": dt.isoformat(),
        "signal_time_kst": to_kst_text(dt),
        "entry_price": c["close"],
        "exit_time_utc": trade["exit_dt"].isoformat(),
        "exit_time_kst": to_kst_text(trade["exit_dt"]),
        "exit_price": trade["exit_price"],
        "result": trade["result"],
        "final_return": trade["final_return"],
        "max_up": trade["max_up"],
        "max_down": trade["max_down"],
        "tp_rate": tp_rate,
        "sl_rate": sl_rate,
        "tp_price": trade["tp_price"],
        "sl_price": trade["sl_price"],
        "grade": c["grade"],
        "grade_label": c["grade_label"],
        "score": c["score"],
        "value": c["value"],
        "volume_ratio": c["volume_ratio"],
        "candle_change": c["candle_change"],
        "recent_3bar": c["recent_3bar"],
        "ma20_dev": c["ma20_dev"],
        "short_k": c["short_k"],
        "short_d": c["short_d"],
        "mid_k": c["mid_k"],
        "close_position": c["close_position"],
        "upper_wick_ratio": c["upper_wick_ratio"],
        "bullish_ok": c["bullish_ok"],
    }


def backtest_market(market, btc_map, start_dt):
    df = fetch_4h_candles(market, NEED_CANDLES)

    if df.empty or len(df) < WARMUP_CANDLES + HOLD_CANDLES + 20:
        return {combo_key(tp, sl): [] for tp, sl in TP_SL_COMBOS}

    df = add_indicators(df)

    results_by_combo = {combo_key(tp, sl): [] for tp, sl in TP_SL_COMBOS}

    last_signal_dt = None

    for i in range(WARMUP_CANDLES, len(df) - HOLD_CANDLES):
        row = df.iloc[i]
        dt = row["dt_utc"]

        if dt < start_dt:
            continue

        if last_signal_dt is not None:
            gap_hours = (dt - last_signal_dt).total_seconds() / 3600.0
            if gap_hours < SIGNAL_COOLDOWN_HOURS:
                continue

        btc_ok = get_btc_ok_for_time(btc_map, dt)
        c = make_candidate_from_row(market, row, btc_ok)

        if not pass_confirmed_filter(c):
            continue

        valid_any = False

        for tp_rate, sl_rate in TP_SL_COMBOS:
            trade = evaluate_trade(df, i, c["close"], tp_rate, sl_rate)

            if trade is None:
                continue

            key = combo_key(tp_rate, sl_rate)
            results_by_combo[key].append(build_result_row(c, trade, tp_rate, sl_rate))
            valid_any = True

        if valid_any:
            last_signal_dt = dt

    return results_by_combo


# -----------------------------
# SELECTION / SUMMARY
# -----------------------------
def grade_order_value(grade):
    order = {
        "A": 0,
        "B": 1,
        "WATCH": 2,
    }
    return order.get(grade, 9)


def apply_topn_selection(results, max_count=10, allowed_grades=None):
    if not results:
        return []

    if allowed_grades is not None:
        results = [r for r in results if r.get("grade") in allowed_grades]

    grouped = {}

    for r in results:
        t = r["signal_time_utc"]
        grouped.setdefault(t, []).append(r)

    selected = []

    for t, items in grouped.items():
        items_sorted = sorted(
            items,
            key=lambda x: (
                grade_order_value(x.get("grade")),
                -safe_float(x.get("score")),
                -safe_float(x.get("value")),
                -safe_float(x.get("volume_ratio")),
            ),
        )

        selected.extend(items_sorted[:max_count])

    selected = sorted(selected, key=lambda x: x["signal_time_utc"], reverse=True)
    return selected


def summarize_results(results):
    total = len(results)

    tp = sum(1 for r in results if r["result"] == "TP")
    sl = sum(1 for r in results if r["result"] == "SL")
    expired = sum(1 for r in results if r["result"] == "EXPIRED")

    avg_final_return = np.mean([r["final_return"] for r in results]) if results else 0
    avg_max_up = np.mean([r["max_up"] for r in results]) if results else 0
    avg_max_down = np.mean([r["max_down"] for r in results]) if results else 0

    return {
        "total": total,
        "tp": tp,
        "sl": sl,
        "expired": expired,
        "tp_rate": (tp / total * 100) if total else 0,
        "sl_rate": (sl / total * 100) if total else 0,
        "expired_rate": (expired / total * 100) if total else 0,
        "avg_final_return": float(avg_final_return),
        "avg_max_up": float(avg_max_up),
        "avg_max_down": float(avg_max_down),
    }


def build_group_summaries(results):
    return {
        "RAW_ALL": summarize_results(results),
        f"TOP_{MAX_ALERT_COUNT}_ALL": summarize_results(
            apply_topn_selection(results, MAX_ALERT_COUNT, allowed_grades=None)
        ),
        f"TOP_{MAX_ALERT_COUNT}_A_B": summarize_results(
            apply_topn_selection(results, MAX_ALERT_COUNT, allowed_grades=["A", "B"])
        ),
        f"TOP_{MAX_ALERT_COUNT}_A_ONLY": summarize_results(
            apply_topn_selection(results, MAX_ALERT_COUNT, allowed_grades=["A"])
        ),
        f"TOP_{MAX_ALERT_COUNT}_B_ONLY": summarize_results(
            apply_topn_selection(results, MAX_ALERT_COUNT, allowed_grades=["B"])
        ),
        f"TOP_{MAX_ALERT_COUNT}_WATCH_ONLY": summarize_results(
            apply_topn_selection(results, MAX_ALERT_COUNT, allowed_grades=["WATCH"])
        ),
    }


def format_one_summary(name, s):
    return (
        f"{name}: {s['total']}개 / "
        f"TP {s['tp']} ({s['tp_rate']:.1f}%) / "
        f"SL {s['sl']} ({s['sl_rate']:.1f}%) / "
        f"만료 {s['expired']} ({s['expired_rate']:.1f}%) / "
        f"평균 {s['avg_final_return']:+.2f}% / "
        f"최대상승 {s['avg_max_up']:+.2f}% / "
        f"최대하락 {s['avg_max_down']:+.2f}%"
    )


def format_summary_message(final_summary, main_key, selected_main):
    lines = []

    lines.append("📊 V3.6 백테스트 V1.1 결과")
    lines.append("")
    lines.append(f"실행시간: {now_kst().strftime('%Y-%m-%d %H:%M:%S KST')}")
    lines.append(f"기간: 최근 {BACKTEST_DAYS}일")
    lines.append(f"실전반영: 동일 4H 시간대별 상위 {MAX_ALERT_COUNT}개 제한")
    lines.append(f"기준: 4H 확정봉 / 과열 배제 / 거래대금 필터")
    lines.append(f"최소 4H 거래대금: {MIN_4H_TRADE_VALUE / 100_000_000:.1f}억")
    lines.append(f"최소 캔들상승률: +{MIN_CANDLE_CHANGE:.1f}%")
    lines.append(f"최대보유: {MAX_HOLD_HOURS}시간")
    lines.append("")

    lines.append("✅ 메인 조합 결과")
    lines.append(f"조합: {main_key}")
    main_groups = final_summary["combo_summaries"][main_key]

    for name in [
        "RAW_ALL",
        f"TOP_{MAX_ALERT_COUNT}_ALL",
        f"TOP_{MAX_ALERT_COUNT}_A_B",
        f"TOP_{MAX_ALERT_COUNT}_A_ONLY",
        f"TOP_{MAX_ALERT_COUNT}_B_ONLY",
        f"TOP_{MAX_ALERT_COUNT}_WATCH_ONLY",
    ]:
        lines.append(format_one_summary(name, main_groups[name]))

    lines.append("")
    lines.append("📌 TP/SL 조합 비교")
    lines.append(f"기준: 동일 시간대 상위 {MAX_ALERT_COUNT}개, A+B만 반영")

    for key, groups in final_summary["combo_summaries"].items():
        s = groups[f"TOP_{MAX_ALERT_COUNT}_A_B"]
        lines.append(
            f"{key}: {s['total']}개 / "
            f"TP {s['tp_rate']:.1f}% / "
            f"SL {s['sl_rate']:.1f}% / "
            f"만료 {s['expired_rate']:.1f}% / "
            f"평균 {s['avg_final_return']:+.2f}%"
        )

    if selected_main:
        lines.append("")
        lines.append("최근 선택 신호 샘플")
        for idx, r in enumerate(selected_main[:10], 1):
            value_eok = r["value"] / 100_000_000
            lines.append(
                f"{idx}. {r['market']} {r['grade_label']} {r['result']} "
                f"/ 진입 {r['entry_price']:.4g} "
                f"/ 최종 {r['final_return']:+.2f}% "
                f"/ 최대 {r['max_up']:+.2f}% "
                f"/ 거래대금 {value_eok:.1f}억 "
                f"/ {r['signal_time_kst']}"
            )

    lines.append("")
    lines.append("해석 가이드")
    lines.append("- RAW_ALL: 조건 만족 전체 후보")
    lines.append(f"- TOP_{MAX_ALERT_COUNT}_ALL: 실제 알림처럼 시간대별 상위 {MAX_ALERT_COUNT}개")
    lines.append(f"- TOP_{MAX_ALERT_COUNT}_A_B: 관찰 제외, A/B만 반영")
    lines.append("- 실전 적용은 A+B 결과를 우선 참고하는 것을 추천")
    lines.append("")
    lines.append("※ 과거 캔들 기준 검증이며 실제 체결가, 슬리피지, 호가 공백은 반영되지 않았습니다.")
    lines.append("※ 같은 4H 봉에서 TP/SL 동시 도달 시 보수적으로 SL 처리했습니다.")

    return "\n".join(lines)


def save_outputs(raw_main, selected_main, final_summary):
    if raw_main:
        pd.DataFrame(raw_main).to_csv(RESULT_RAW_CSV_FILE, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(RESULT_RAW_CSV_FILE, index=False, encoding="utf-8-sig")

    if selected_main:
        pd.DataFrame(selected_main).to_csv(RESULT_SELECTED_CSV_FILE, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(RESULT_SELECTED_CSV_FILE, index=False, encoding="utf-8-sig")

    with open(SUMMARY_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"Saved raw CSV: {RESULT_RAW_CSV_FILE}")
    print(f"Saved selected CSV: {RESULT_SELECTED_CSV_FILE}")
    print(f"Saved summary JSON: {SUMMARY_JSON_FILE}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=" * 80)
    print("Telegram Coin Alert Strategy Backtest V1.1")
    print(f"Now KST: {now_kst().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"BACKTEST_DAYS={BACKTEST_DAYS}")
    print(f"NEED_CANDLES={NEED_CANDLES}")
    print(f"MAX_ALERT_COUNT={MAX_ALERT_COUNT}")
    print(f"TP_SL_COMBOS={TP_SL_COMBOS}")
    print("=" * 80)

    start_dt = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=BACKTEST_DAYS))
    print(f"Backtest start UTC: {start_dt}")

    print("Fetching BTC filter candles...")
    btc_map = build_btc_filter_map()
    print(f"BTC filter map size: {len(btc_map)}")

    markets = get_krw_markets()
    print(f"KRW markets count: {len(markets)}")

    all_results_by_combo = {combo_key(tp, sl): [] for tp, sl in TP_SL_COMBOS}

    for idx, market in enumerate(markets, 1):
        print(f"[{idx}/{len(markets)}] Backtesting {market}...")

        try:
            market_results_by_combo = backtest_market(market, btc_map, start_dt)

            for key, rows in market_results_by_combo.items():
                all_results_by_combo.setdefault(key, [])
                all_results_by_combo[key].extend(rows)

            main_key_tmp = combo_key(TAKE_PROFIT_RATE, STOP_LOSS_RATE)
            print(f"  main combo signals: {len(market_results_by_combo.get(main_key_tmp, []))}")

        except Exception as e:
            print(f"  ERROR {market}: {e}")

    # 정렬
    for key in all_results_by_combo.keys():
        all_results_by_combo[key] = sorted(
            all_results_by_combo[key],
            key=lambda x: x["signal_time_utc"],
            reverse=True,
        )

    main_key = combo_key(TAKE_PROFIT_RATE, STOP_LOSS_RATE)

    if main_key not in all_results_by_combo:
        main_key = list(all_results_by_combo.keys())[0]

    raw_main = all_results_by_combo[main_key]

    # 실제 운영 기준: 동일 시간대 상위 MAX_ALERT_COUNT, A+B만
    selected_main = apply_topn_selection(
        raw_main,
        max_count=MAX_ALERT_COUNT,
        allowed_grades=["A", "B"],
    )

    combo_summaries = {}

    for key, rows in all_results_by_combo.items():
        combo_summaries[key] = build_group_summaries(rows)

    final_summary = {
        "version": "V1.1",
        "created_at_kst": now_kst().strftime("%Y-%m-%d %H:%M:%S KST"),
        "backtest_days": BACKTEST_DAYS,
        "max_alert_count": MAX_ALERT_COUNT,
        "max_hold_hours": MAX_HOLD_HOURS,
        "min_4h_trade_value": MIN_4H_TRADE_VALUE,
        "min_candle_change": MIN_CANDLE_CHANGE,
        "main_combo": main_key,
        "combo_summaries": combo_summaries,
    }

    save_outputs(raw_main, selected_main, final_summary)

    message = format_summary_message(final_summary, main_key, selected_main)

    print("")
    print("=" * 80)
    print(message)
    print("=" * 80)

    if SEND_TELEGRAM_BACKTEST:
        send_telegram_message(message)

    print("Backtest V1.1 finished.")


if __name__ == "__main__":
    main()
