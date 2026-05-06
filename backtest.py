import os
import time
import json
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# =========================================================
# Backtest V1.0 for Telegram Coin Alert Strategy V3.6
# - 4H confirmed candle only
# - Overheated coin exclusion
# - Absolute 4H trade value filter
# - TP +5%, SL -4%, Max hold 48h
# - A/B/WATCH grade performance summary
# =========================================================

# -----------------------------
# ENV CONFIG
# -----------------------------
UPBIT_BASE_URL = "https://api.upbit.com/v1"

BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "60"))
BACKTEST_CANDLE_INTERVAL = os.getenv("BACKTEST_CANDLE_INTERVAL", "minute240")
BACKTEST_MAX_MARKETS = int(os.getenv("BACKTEST_MAX_MARKETS", "999"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.09"))

TAKE_PROFIT_RATE = float(os.getenv("TAKE_PROFIT_RATE", "0.05"))
STOP_LOSS_RATE = float(os.getenv("STOP_LOSS_RATE", "0.04"))
MAX_HOLD_HOURS = int(os.getenv("MAX_HOLD_HOURS", "48"))
SIGNAL_COOLDOWN_HOURS = int(os.getenv("SIGNAL_COOLDOWN_HOURS", "12"))

MIN_4H_TRADE_VALUE = float(os.getenv("MIN_4H_TRADE_VALUE", "150000000"))
MIN_CANDLE_CHANGE = float(os.getenv("MIN_CANDLE_CHANGE", "0.7"))

SEND_TELEGRAM_BACKTEST = os.getenv("SEND_TELEGRAM_BACKTEST", "false").strip().lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RESULT_CSV_FILE = os.getenv("RESULT_CSV_FILE", "backtest_results.csv")
SUMMARY_JSON_FILE = os.getenv("SUMMARY_JSON_FILE", "backtest_summary.json")

# 4H candles per day = 6
CANDLES_PER_DAY = 6
WARMUP_CANDLES = 120
HOLD_CANDLES = int(MAX_HOLD_HOURS / 4)
NEED_CANDLES = BACKTEST_DAYS * CANDLES_PER_DAY + WARMUP_CANDLES + HOLD_CANDLES + 20


# -----------------------------
# TIME UTILS
# -----------------------------
def now_kst():
    return datetime.now(timezone.utc) + timedelta(hours=9)


def parse_utc_datetime(value):
    # Upbit returns e.g. "2026-05-06T00:00:00"
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def utc_to_kst_text(dt):
    if isinstance(dt, str):
        dt = parse_utc_datetime(dt)
    return (dt + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S KST")


# -----------------------------
# SAFE UTILS
# -----------------------------
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


# -----------------------------
# TELEGRAM
# -----------------------------
def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token/chat_id missing. Skipping Telegram send.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    chunks = []
    max_len = 3900
    for i in range(0, len(text), max_len):
        chunks.append(text[i:i + max_len])

    ok = True
    for chunk in chunks:
        try:
            res = requests.post(
                url,
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk,
                    "disable_web_page_preview": True
                },
                timeout=15
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
    """
    Fetch 4H candles from Upbit using pagination.
    Upbit returns newest first.
    """
    all_rows = []
    to_value = None

    while len(all_rows) < need_count:
        count = min(200, need_count - len(all_rows))
        params = {
            "market": market,
            "count": count
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

    rename_map = {
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_price": "value",
        "candle_acc_trade_volume": "volume"
    }
    df = df.rename(columns=rename_map)

    needed_cols = ["dt_utc", "open", "high", "low", "close", "value", "volume"]
    for col in needed_cols:
        if col not in df.columns:
            return pd.DataFrame()

    for col in ["open", "high", "low", "close", "value", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[needed_cols].reset_index(drop=True)


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
        print("BTC candles empty. BTC filter will be treated as OK.")
        return {}

    btc = add_indicators(btc)
    result = {}

    for _, row in btc.iterrows():
        dt = row["dt_utc"]
        candle_change = safe_float(row.get("candle_change"), 0.0)
        ma20_dev = safe_float(row.get("ma20_dev"), 0.0)

        # 보수적 BTC 필터:
        # BTC 4H가 급락 중이거나 MA20에서 크게 아래면 신규 신호 약화
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
# SCORING / FILTER / GRADE
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

    # Relative volume
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

    # Absolute 4H trade value
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

    # Mid K
    if mid_k <= 55:
        score += 10
    elif mid_k > 70:
        score -= 15

    # MA20 deviation
    if -2.0 <= ma20_dev <= 2.0:
        score += 25
    elif -5.0 <= ma20_dev <= 5.0:
        score += 10
    else:
        score -= 25

    # Candle change
    if 1.0 <= candle_change <= 3.5:
        score += 25
    elif 0.7 <= candle_change <= 4.0:
        score += 15
    elif candle_change > 4.0:
        score -= 30

    # Recent 3-bar
    if -1.0 <= recent_3bar <= 4.0:
        score += 15
    elif -4.0 <= recent_3bar <= 6.0:
        score += 5
    elif recent_3bar > 6.0:
        score -= 25

    # Candle quality
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


def pass_confirmed_filter(c):
    """
    V3.6 confirmed-only conservative filter.
    """
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

    # V3.6 핵심 추가: 절대 4H 거래대금
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

    # V3.6 핵심 추가: 너무 약한 캔들 배제
    if not (MIN_CANDLE_CHANGE <= candle_change <= 4.0):
        return False

    if not ma5_up:
        return False

    if not bullish_ok:
        return False

    return True


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


# -----------------------------
# BACKTEST CORE
# -----------------------------
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


def evaluate_trade(df, signal_idx, entry_price):
    """
    Entry at signal candle close.
    Future window: next 12 candles = 48h.
    If both TP and SL are touched in same candle, conservatively count SL first.
    """
    tp_price = entry_price * (1.0 + TAKE_PROFIT_RATE)
    sl_price = entry_price * (1.0 - STOP_LOSS_RATE)

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
            # Conservative assumption
            result = "SL"
            exit_price = sl_price
            exit_dt = row["dt_utc"]
            break
        elif hit_sl:
            result = "SL"
            exit_price = sl_price
            exit_dt = row["dt_utc"]
            break
        elif hit_tp:
            result = "TP"
            exit_price = tp_price
            exit_dt = row["dt_utc"]
            break

    final_return = pct(exit_price, entry_price)
    max_up = pct(max_high, entry_price)
    max_down = pct(min_low, entry_price)

    return {
        "result": result,
        "exit_price": exit_price,
        "exit_dt": exit_dt,
        "final_return": final_return,
        "max_up": max_up,
        "max_down": max_down,
        "tp_price": tp_price,
        "sl_price": sl_price,
    }


def backtest_market(market, btc_map, start_dt):
    df = fetch_4h_candles(market, NEED_CANDLES)
    if df.empty or len(df) < WARMUP_CANDLES + HOLD_CANDLES + 20:
        return []

    df = add_indicators(df)
    results = []

    last_signal_dt = None

    # Skip last HOLD_CANDLES because result cannot be fully known yet
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

        trade = evaluate_trade(df, i, c["close"])
        if trade is None:
            continue

        last_signal_dt = dt

        result = {
            "market": market,
            "signal_time_utc": dt.isoformat(),
            "signal_time_kst": utc_to_kst_text(dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt),
            "entry_price": c["close"],
            "exit_time_utc": trade["exit_dt"].isoformat(),
            "exit_time_kst": utc_to_kst_text(trade["exit_dt"].to_pydatetime() if hasattr(trade["exit_dt"], "to_pydatetime") else trade["exit_dt"]),
            "exit_price": trade["exit_price"],
            "result": trade["result"],
            "final_return": trade["final_return"],
            "max_up": trade["max_up"],
            "max_down": trade["max_down"],
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

        results.append(result)

    return results


# -----------------------------
# SUMMARY
# -----------------------------
def summarize_results(results):
    total = len(results)
    tp = sum(1 for r in results if r["result"] == "TP")
    sl = sum(1 for r in results if r["result"] == "SL")
    expired = sum(1 for r in results if r["result"] == "EXPIRED")

    avg_final_return = np.mean([r["final_return"] for r in results]) if results else 0
    avg_max_up = np.mean([r["max_up"] for r in results]) if results else 0
    avg_max_down = np.mean([r["max_down"] for r in results]) if results else 0

    summary = {
        "backtest_days": BACKTEST_DAYS,
        "total_signals": total,
        "tp_count": tp,
        "sl_count": sl,
        "expired_count": expired,
        "tp_rate": (tp / total * 100) if total else 0,
        "sl_rate": (sl / total * 100) if total else 0,
        "expired_rate": (expired / total * 100) if total else 0,
        "avg_final_return": avg_final_return,
        "avg_max_up": avg_max_up,
        "avg_max_down": avg_max_down,
        "by_grade": {}
    }

    for grade in ["A", "B", "WATCH"]:
        gr = [r for r in results if r["grade"] == grade]
        g_total = len(gr)
        g_tp = sum(1 for r in gr if r["result"] == "TP")
        g_sl = sum(1 for r in gr if r["result"] == "SL")
        g_expired = sum(1 for r in gr if r["result"] == "EXPIRED")

        summary["by_grade"][grade] = {
            "total": g_total,
            "tp": g_tp,
            "sl": g_sl,
            "expired": g_expired,
            "tp_rate": (g_tp / g_total * 100) if g_total else 0,
            "sl_rate": (g_sl / g_total * 100) if g_total else 0,
            "expired_rate": (g_expired / g_total * 100) if g_total else 0,
            "avg_final_return": np.mean([r["final_return"] for r in gr]) if gr else 0,
            "avg_max_up": np.mean([r["max_up"] for r in gr]) if gr else 0,
            "avg_max_down": np.mean([r["max_down"] for r in gr]) if gr else 0,
        }

    return summary


def format_summary_message(summary, top_items):
    lines = []
    lines.append("📊 V3.6 백테스트 결과")
    lines.append("")
    lines.append(f"실행시간: {now_kst().strftime('%Y-%m-%d %H:%M:%S KST')}")
    lines.append(f"기간: 최근 {summary['backtest_days']}일")
    lines.append("기준: 4H 확정봉 / 과열 배제 / 거래대금 필터")
    lines.append(f"TP: +{TAKE_PROFIT_RATE * 100:.1f}% / SL: -{STOP_LOSS_RATE * 100:.1f}% / 최대보유: {MAX_HOLD_HOURS}시간")
    lines.append(f"최소 4H 거래대금: {MIN_4H_TRADE_VALUE / 100_000_000:.1f}억")
    lines.append(f"최소 캔들상승률: +{MIN_CANDLE_CHANGE:.1f}%")
    lines.append("")
    lines.append("전체 결과")
    lines.append(f"총 신호: {summary['total_signals']}개")
    lines.append(f"익절: {summary['tp_count']}개 ({summary['tp_rate']:.1f}%)")
    lines.append(f"손절: {summary['sl_count']}개 ({summary['sl_rate']:.1f}%)")
    lines.append(f"만료: {summary['expired_count']}개 ({summary['expired_rate']:.1f}%)")
    lines.append(f"평균 최종수익률: {summary['avg_final_return']:+.2f}%")
    lines.append(f"평균 최대상승률: {summary['avg_max_up']:+.2f}%")
    lines.append(f"평균 최대하락률: {summary['avg_max_down']:+.2f}%")
    lines.append("")
    lines.append("등급별 결과")

    label_map = {
        "A": "🔥 A급",
        "B": "🟡 B급",
        "WATCH": "👀 관찰"
    }

    for grade in ["A", "B", "WATCH"]:
        g = summary["by_grade"][grade]
        lines.append(
            f"{label_map[grade]}: {g['total']}개 / "
            f"TP {g['tp']} ({g['tp_rate']:.1f}%) / "
            f"SL {g['sl']} ({g['sl_rate']:.1f}%) / "
            f"만료 {g['expired']} ({g['expired_rate']:.1f}%) / "
            f"평균 {g['avg_final_return']:+.2f}%"
        )

    if top_items:
        lines.append("")
        lines.append("최근 신호 샘플")
        for idx, r in enumerate(top_items[:10], 1):
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
    lines.append("※ 백테스트는 과거 캔들 기준 검증이며 실제 체결/슬리피지/동시 TP·SL 순서는 반영되지 않을 수 있습니다.")
    lines.append("※ 동시 TP·SL 발생 캔들은 보수적으로 SL 처리했습니다.")

    return "\n".join(lines)


def save_results(results, summary):
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULT_CSV_FILE, index=False, encoding="utf-8-sig")
        print(f"Saved CSV: {RESULT_CSV_FILE}")
    else:
        pd.DataFrame().to_csv(RESULT_CSV_FILE, index=False, encoding="utf-8-sig")
        print(f"Saved empty CSV: {RESULT_CSV_FILE}")

    with open(SUMMARY_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary JSON: {SUMMARY_JSON_FILE}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=" * 70)
    print("Telegram Coin Alert Strategy Backtest V1.0")
    print(f"Now KST: {now_kst().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"BACKTEST_DAYS={BACKTEST_DAYS}")
    print(f"NEED_CANDLES={NEED_CANDLES}")
    print(f"TP={TAKE_PROFIT_RATE}, SL={STOP_LOSS_RATE}, HOLD={MAX_HOLD_HOURS}h")
    print("=" * 70)

    start_dt = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=BACKTEST_DAYS))
    print(f"Backtest start UTC: {start_dt}")

    print("Fetching BTC filter candles...")
    btc_map = build_btc_filter_map()
    print(f"BTC filter map size: {len(btc_map)}")

    markets = get_krw_markets()
    print(f"KRW markets count: {len(markets)}")

    all_results = []

    for idx, market in enumerate(markets, 1):
        print(f"[{idx}/{len(markets)}] Backtesting {market}...")
        try:
            results = backtest_market(market, btc_map, start_dt)
            print(f"  signals: {len(results)}")
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR {market}: {e}")

    all_results = sorted(all_results, key=lambda x: x["signal_time_utc"], reverse=True)

    summary = summarize_results(all_results)
    save_results(all_results, summary)

    top_items = all_results[:10]
    message = format_summary_message(summary, top_items)

    print("")
    print("=" * 70)
    print(message)
    print("=" * 70)

    if SEND_TELEGRAM_BACKTEST:
        send_telegram_message(message)

    print("Backtest finished.")


if __name__ == "__main__":
    main()
