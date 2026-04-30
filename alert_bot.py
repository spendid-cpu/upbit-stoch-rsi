import os
import time
import requests
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyupbit


# ==================================================
# 기본 설정
# ==================================================

KST = timezone(timedelta(hours=9))

STOCH_RSI_SETTINGS = {
    "short": {
        "rsi_period": 5,
        "stoch_period": 5,
        "k_smooth": 3,
        "d_smooth": 3,
    },
    "middle": {
        "rsi_period": 10,
        "stoch_period": 10,
        "k_smooth": 6,
        "d_smooth": 6,
    },
    "long": {
        "rsi_period": 20,
        "stoch_period": 20,
        "k_smooth": 12,
        "d_smooth": 12,
    },
}


STABLE_EXCLUDES = {
    "KRW-USDT",
    "KRW-USDC",
    "KRW-USDE",
}


# ==================================================
# 유틸
# ==================================================

def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt(x, digits=2):
    try:
        if x is None or pd.isna(x):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def get_kst_now():
    return datetime.now(KST)


def detect_alert_mode():
    """
    GitHub Actions 실행 시간을 기준으로 알림 모드 자동 판별.
    - 08,12,16,20시 KST: PRE
    - 09,13,17,21시 KST: POST
    환경변수 ALERT_MODE가 있으면 그 값을 우선 사용.
    """
    env_mode = os.getenv("ALERT_MODE", "").strip().upper()
    if env_mode in ["PRE", "POST"]:
        return env_mode

    now = get_kst_now()
    if now.hour in [8, 12, 16, 20]:
        return "PRE"
    if now.hour in [9, 13, 17, 21]:
        return "POST"

    return "PRE"


# ==================================================
# 텔레그램
# ==================================================

def send_telegram_message(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 없습니다.")

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    res = requests.post(url, data=payload, timeout=20)

    if res.status_code != 200:
        raise RuntimeError(f"Telegram 전송 실패: {res.status_code} / {res.text}")

    return True


# ==================================================
# Upbit 데이터
# ==================================================

def get_krw_markets():
    try:
        url = "https://api.upbit.com/v1/market/all"
        params = {"isDetails": "true"}
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


def get_ohlcv(ticker, interval="minute240", count=700):
    try:
        df = pyupbit.get_ohlcv(
            ticker=ticker,
            interval=interval,
            count=count
        )

        if df is None or df.empty:
            return None

        df = df.copy()
        df = df.sort_index()

        if "value" not in df.columns:
            if "close" in df.columns and "volume" in df.columns:
                df["value"] = df["close"] * df["volume"]
            else:
                df["value"] = 0

        return df

    except Exception:
        return None


# ==================================================
# 지표 계산
# ==================================================

def calculate_rsi(close, period=14):
    close = close.astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_stoch_rsi(df, setting, prefix):
    result = df.copy()

    rsi = calculate_rsi(result["close"], period=setting["rsi_period"])

    rsi_min = rsi.rolling(setting["stoch_period"]).min()
    rsi_max = rsi.rolling(setting["stoch_period"]).max()

    denominator = (rsi_max - rsi_min).replace(0, np.nan)
    raw = 100 * ((rsi - rsi_min) / denominator)

    k = raw.rolling(setting["k_smooth"]).mean()
    d = k.rolling(setting["d_smooth"]).mean()

    result[f"{prefix}_rsi"] = rsi
    result[f"{prefix}_raw"] = raw
    result[f"{prefix}_k"] = k
    result[f"{prefix}_d"] = d

    return result


def calculate_macd(df, fast=12, slow=26, signal=9):
    result = df.copy()

    ema_fast = result["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = result["close"].ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal

    result["macd"] = macd
    result["macd_signal"] = macd_signal
    result["macd_hist"] = hist

    return result


def prepare_indicators(df):
    result = df.copy()

    for key, setting in STOCH_RSI_SETTINGS.items():
        result = calculate_stoch_rsi(result, setting, key)

    result = calculate_macd(result)

    result["ma5"] = result["close"].rolling(5).mean()
    result["ma10"] = result["close"].rolling(10).mean()
    result["ma20"] = result["close"].rolling(20).mean()

    result["ret_3"] = result["close"].pct_change(3)

    result["value_avg3"] = result["value"].rolling(3).mean()
    result["value_avg20"] = result["value"].rolling(20).mean()
    result["volume_ratio"] = result["value_avg3"] / result["value_avg20"]

    candle_range = (result["high"] - result["low"]).replace(0, np.nan)

    result["close_position"] = (result["close"] - result["low"]) / candle_range

    result["upper_wick_ratio"] = (
        result["high"] - result[["open", "close"]].max(axis=1)
    ) / candle_range

    result["candle_return_pct"] = (
        (result["close"] - result["open"]) / result["open"]
    ) * 100

    result["ma5_slope_up"] = result["ma5"] > result["ma5"].shift(1)

    return result


# ==================================================
# BTC 필터
# ==================================================

def get_btc_indicator_df(interval="minute240", count=750):
    df = get_ohlcv("KRW-BTC", interval=interval, count=count)

    if df is None or df.empty:
        return None

    result = df.copy().sort_index()

    result["ma5"] = result["close"].rolling(5).mean()
    result["ma10"] = result["close"].rolling(10).mean()
    result["ma20"] = result["close"].rolling(20).mean()
    result["ret_3"] = result["close"].pct_change(3)

    return result.dropna().copy()


def check_btc_filter_at(
    btc_df,
    signal_time,
    btc_min_3bar_rise=-2.0,
    btc_require_close_above_ma20=True,
    btc_require_ma5_above_ma10=False,
):
    if btc_df is None or btc_df.empty:
        return False, "BTC 데이터 없음"

    try:
        pos = btc_df.index.get_indexer([signal_time], method="ffill")[0]

        if pos < 0:
            return False, "BTC 매칭 실패"

        row = btc_df.iloc[pos]

        btc_close = safe_float(row["close"])
        btc_ma5 = safe_float(row["ma5"])
        btc_ma10 = safe_float(row["ma10"])
        btc_ma20 = safe_float(row["ma20"])
        btc_ret_3 = safe_float(row["ret_3"])

        if None in [btc_close, btc_ma5, btc_ma10, btc_ma20, btc_ret_3]:
            return False, "BTC 지표 부족"

        btc_ret_3_pct = btc_ret_3 * 100

        close_above_ma20 = btc_close > btc_ma20
        ma5_above_ma10 = btc_ma5 > btc_ma10
        not_crashing = btc_ret_3_pct > btc_min_3bar_rise

        if btc_require_close_above_ma20 and not close_above_ma20:
            return False, f"BTC MA20 아래 / 3봉 {btc_ret_3_pct:.2f}%"

        if btc_require_ma5_above_ma10 and not ma5_above_ma10:
            return False, "BTC MA5 <= MA10"

        if close_above_ma20 or not_crashing:
            return True, f"BTC 통과 / 3봉 {btc_ret_3_pct:.2f}%"

        return False, f"BTC 약세 / 3봉 {btc_ret_3_pct:.2f}%"

    except Exception as e:
        return False, f"BTC 필터 오류: {e}"


# ==================================================
# 알림 모드별 파라미터
# ==================================================

def get_params(alert_mode):
    """
    PRE: 예비 알림
    POST: 확정 알림
    """

    if alert_mode == "POST":
        return {
            "min_signal_score": 220,
            "min_required_volume_ratio": 1.6,
            "max_short_k": 60.0,
            "max_short_d": 50.0,
            "max_middle_k": 55.0,
            "min_ma20_gap": -4.0,
            "max_ma20_gap": 2.5,
            "min_3bar_rise": -3.5,
            "max_3bar_rise": 4.5,
            "min_close_position": 0.65,
            "max_upper_wick_ratio": 0.35,
            "max_bear_candle_pct": 0.3,
            "max_signal_candle_return_pct": 3.5,
            "btc_min_3bar_rise": -1.5,
            "btc_require_close_above_ma20": True,
            "btc_require_ma5_above_ma10": False,
        }

    return {
        "min_signal_score": 210,
        "min_required_volume_ratio": 1.5,
        "max_short_k": 65.0,
        "max_short_d": 55.0,
        "max_middle_k": 60.0,
        "min_ma20_gap": -5.0,
        "max_ma20_gap": 3.0,
        "min_3bar_rise": -4.0,
        "max_3bar_rise": 5.0,
        "min_close_position": 0.60,
        "max_upper_wick_ratio": 0.40,
        "max_bear_candle_pct": 0.3,
        "max_signal_candle_return_pct": 4.0,
        "btc_min_3bar_rise": -2.0,
        "btc_require_close_above_ma20": True,
        "btc_require_ma5_above_ma10": False,
    }


# ==================================================
# 신호 판단
# ==================================================

def judge_signal_at(
    df,
    idx,
    btc_df,
    params,
    oversold_lookback=12,
    volume_ratio_threshold=1.3,
):
    if idx < 80:
        return None

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]
    prev2 = df.iloc[idx - 2]
    prev3 = df.iloc[idx - 3]

    score = 0
    reasons = []

    # BTC 필터
    btc_ok, btc_reason = check_btc_filter_at(
        btc_df=btc_df,
        signal_time=df.index[idx],
        btc_min_3bar_rise=params["btc_min_3bar_rise"],
        btc_require_close_above_ma20=params["btc_require_close_above_ma20"],
        btc_require_ma5_above_ma10=params["btc_require_ma5_above_ma10"],
    )

    if not btc_ok:
        return None

    score += 10
    reasons.append(btc_reason)

    # 과매도 이력
    recent = df.iloc[max(0, idx - oversold_lookback + 1):idx + 1]

    oversold_recent_short = (
        (recent["short_k"] <= 20) | (recent["short_d"] <= 20)
    ).any()

    oversold_recent_middle = (
        (recent["middle_k"] <= 20) | (recent["middle_d"] <= 20)
    ).any()

    oversold_recent_long = (
        (recent["long_k"] <= 20) | (recent["long_d"] <= 20)
    ).any()

    oversold_count = sum([
        bool(oversold_recent_short),
        bool(oversold_recent_middle),
        bool(oversold_recent_long),
    ])

    if oversold_recent_short:
        score += 25
        reasons.append("단기 과매도 이력")

    if oversold_recent_middle:
        score += 30
        reasons.append("중기 과매도 이력")

    if oversold_recent_long:
        score += 35
        reasons.append("장기 과매도 이력")

    if oversold_count == 0:
        return None

    # 단기 Stoch RSI
    short_k = safe_float(row["short_k"])
    short_d = safe_float(row["short_d"])
    prev_short_k = safe_float(prev["short_k"])
    prev_short_d = safe_float(prev["short_d"])

    if short_k is None or short_d is None or prev_short_k is None or prev_short_d is None:
        return None

    if short_k > params["max_short_k"]:
        return None

    if short_d > params["max_short_d"]:
        return None

    kd_cross = short_k > short_d and prev_short_k <= prev_short_d
    kd_above = short_k > short_d
    k_recover_20 = prev_short_k <= 20 and short_k > 20
    k_rising = short_k > prev_short_k

    if kd_cross:
        score += 30
        reasons.append("단기 K>D 골든크로스")
    elif kd_above:
        score += 18
        reasons.append("단기 K>D 유지")

    if k_recover_20:
        score += 35
        reasons.append("단기 K 20 회복")
    elif k_rising:
        score += 15
        reasons.append("단기 K 상승")

    if not (kd_above or k_recover_20):
        return None

    # 중기 Stoch RSI
    middle_k = safe_float(row["middle_k"])
    prev_middle_k = safe_float(prev["middle_k"])

    if middle_k is not None and middle_k > params["max_middle_k"]:
        return None

    if middle_k is not None and prev_middle_k is not None:
        if middle_k > prev_middle_k and middle_k < 80:
            score += 20
            reasons.append("중기 K 상승")

    # 거래대금
    volume_ratio = safe_float(row["volume_ratio"])

    if volume_ratio is None or volume_ratio < params["min_required_volume_ratio"]:
        return None

    if volume_ratio >= volume_ratio_threshold:
        score += 35
        reasons.append(f"거래대금 증가 x{volume_ratio:.2f}")
    else:
        score += 15
        reasons.append(f"거래대금 완만 증가 x{volume_ratio:.2f}")

    # MA / 캔들 품질
    close = safe_float(row["close"])
    ma5 = safe_float(row["ma5"])
    ma10 = safe_float(row["ma10"])
    ma20 = safe_float(row["ma20"])

    if close is None or ma5 is None or ma10 is None or ma20 is None:
        return None

    close_position = safe_float(row.get("close_position"))
    upper_wick_ratio = safe_float(row.get("upper_wick_ratio"))
    candle_return_pct = safe_float(row.get("candle_return_pct"))
    ma5_slope_up = bool(row.get("ma5_slope_up"))

    if close_position is None or upper_wick_ratio is None or candle_return_pct is None:
        return None

    if close_position < params["min_close_position"]:
        return None

    if upper_wick_ratio > params["max_upper_wick_ratio"]:
        return None

    if candle_return_pct < -abs(params["max_bear_candle_pct"]):
        return None

    if candle_return_pct > params["max_signal_candle_return_pct"]:
        return None

    if not ma5_slope_up:
        return None

    score += 20
    reasons.append(
        f"캔들품질 통과 / 종가위치 {close_position:.2f} / 윗꼬리 {upper_wick_ratio:.2f} / 캔들 {candle_return_pct:.2f}%"
    )

    score += 10
    reasons.append("MA5 상승기울기")

    if close > ma5:
        score += 15
        reasons.append("MA5 회복")

    if close > ma10:
        score += 10
        reasons.append("MA10 회복")

    # MACD
    hist = safe_float(row["macd_hist"])
    hist1 = safe_float(prev["macd_hist"])
    hist2 = safe_float(prev2["macd_hist"])
    hist3 = safe_float(prev3["macd_hist"])

    if None not in [hist, hist1, hist2, hist3]:
        if hist > hist1 > hist2:
            score += 25
            reasons.append("MACD 3봉 개선")
        elif hist > hist1:
            score += 10
            reasons.append("MACD 개선")

    # 급락 / 과열 / MA20 이격
    ret_3 = safe_float(row["ret_3"])

    if ret_3 is None:
        return None

    three_bar_rise_pct = ret_3 * 100
    ma20_gap_pct = ((close - ma20) / ma20) * 100 if ma20 else 0

    if three_bar_rise_pct < params["min_3bar_rise"]:
        return None

    if three_bar_rise_pct > params["max_3bar_rise"]:
        return None

    if ma20_gap_pct < params["min_ma20_gap"]:
        return None

    if ma20_gap_pct > params["max_ma20_gap"]:
        return None

    if score < params["min_signal_score"]:
        return None

    return {
        "ticker": None,
        "signal_time": df.index[idx],
        "score": score,
        "close": close,
        "short_k": short_k,
        "short_d": short_d,
        "middle_k": middle_k,
        "volume_ratio": volume_ratio,
        "ma20_gap_pct": ma20_gap_pct,
        "three_bar_rise_pct": three_bar_rise_pct,
        "close_position": close_position,
        "upper_wick_ratio": upper_wick_ratio,
        "candle_return_pct": candle_return_pct,
        "reason": " / ".join(reasons),
    }


# ==================================================
# 스캔
# ==================================================

def scan_market():
    alert_mode = detect_alert_mode()
    params = get_params(alert_mode)

    interval = os.getenv("ALERT_INTERVAL", "minute240")
    count = int(os.getenv("ALERT_CANDLE_COUNT", "700"))
    request_delay = float(os.getenv("REQUEST_DELAY", "0.12"))

    max_alert_count = int(os.getenv("MAX_ALERT_COUNT", "10"))

    markets = get_krw_markets()
    markets = [
        x for x in markets
        if x not in STABLE_EXCLUDES and x != "KRW-BTC"
    ]

    btc_df = get_btc_indicator_df(interval=interval, count=count + 50)

    results = []

    for idx, ticker in enumerate(markets):
        try:
            df = get_ohlcv(ticker, interval=interval, count=count)

            if df is None or len(df) < 120:
                time.sleep(request_delay)
                continue

            df = prepare_indicators(df)
            df = df.dropna().copy()

            if len(df) < 120:
                time.sleep(request_delay)
                continue

            # PRE: 현재 진행 중인 4H 봉 기준
            # POST: 방금 마감된 4H 봉 기준
            if alert_mode == "POST":
                signal_idx = len(df) - 2
            else:
                signal_idx = len(df) - 1

            signal = judge_signal_at(
                df=df,
                idx=signal_idx,
                btc_df=btc_df,
                params=params,
                oversold_lookback=12,
                volume_ratio_threshold=params["min_required_volume_ratio"],
            )

            if signal:
                signal["ticker"] = ticker
                results.append(signal)

            time.sleep(request_delay)

        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
            time.sleep(request_delay)

    results = sorted(
        results,
        key=lambda x: (x["score"], x["volume_ratio"]),
        reverse=True
    )

    return alert_mode, results[:max_alert_count], len(markets)


# ==================================================
# 메시지 포맷
# ==================================================

def build_message(alert_mode, signals, market_count):
    now = get_kst_now().strftime("%Y-%m-%d %H:%M:%S")

    if alert_mode == "POST":
        title = "🟢 4H 마감 후 상승전환 확정 후보"
    else:
        title = "🟡 4H 마감 1시간 전 상승전조 후보"

    if not signals:
        return (
            f"{title}\n"
            f"시간: {now} KST\n"
            f"스캔: 전체 KRW {market_count}개\n\n"
            f"조건 만족 종목 없음"
        )

    lines = []
    lines.append(title)
    lines.append(f"시간: {now} KST")
    lines.append(f"스캔: 전체 KRW {market_count}개")
    lines.append(f"후보: {len(signals)}개")
    lines.append("")

    for i, s in enumerate(signals, start=1):
        lines.append(f"<b>{i}. {s['ticker']}</b>")
        lines.append(
            f"점수 {s['score']} / 거래대금 x{fmt(s['volume_ratio'])} / "
            f"K {fmt(s['short_k'])} D {fmt(s['short_d'])} / 중기K {fmt(s['middle_k'])}"
        )
        lines.append(
            f"MA20 {fmt(s['ma20_gap_pct'])}% / 3봉 {fmt(s['three_bar_rise_pct'])}% / "
            f"종가위치 {fmt(s['close_position'])} / 윗꼬리 {fmt(s['upper_wick_ratio'])}"
        )
        lines.append(
            f"캔들 {fmt(s['candle_return_pct'])}% / 가격 {fmt(s['close'], 4)}"
        )
        lines.append("")

    lines.append("※ 자동매매 아님. 진입 전 호가/거래대금/BTC 상태 확인 필요.")

    text = "\n".join(lines)

    # 텔레그램 메시지 길이 제한 대응
    if len(text) > 3900:
        text = text[:3900] + "\n\n...메시지 길이 제한으로 일부 생략"

    return text


# ==================================================
# main
# ==================================================

def main():
    alert_mode, signals, market_count = scan_market()

    send_empty = os.getenv("SEND_EMPTY_ALERT", "true").lower() == "true"

    if not signals and not send_empty:
        print("조건 만족 종목 없음. 메시지 전송 생략.")
        return

    message = build_message(alert_mode, signals, market_count)

    print(message)

    send_telegram_message(message)

    print("텔레그램 전송 완료")


if __name__ == "__main__":
    main()

