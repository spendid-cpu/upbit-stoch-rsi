import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyupbit
import requests
import streamlit as st


# ==================================================
# Streamlit 설정
# ==================================================

st.set_page_config(
    page_title="상승전환 백테스트 V3",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 상승전환 전략 백테스트 V3")
st.caption(
    "과매도 이력 + Stoch RSI 전환 + 거래대금 증가 + BTC 필터 + 캔들 품질 + MA5 기울기"
)


# ==================================================
# Stoch RSI 설정
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
    },
    "middle": {
        "name": "중기",
        "rsi_period": 10,
        "stoch_period": 10,
        "k_smooth": 6,
        "d_smooth": 6,
        "oversold": 20,
        "overbought": 80,
    },
    "long": {
        "name": "장기",
        "rsi_period": 20,
        "stoch_period": 20,
        "k_smooth": 12,
        "d_smooth": 12,
        "oversold": 20,
        "overbought": 80,
    },
}


# ==================================================
# 유틸 함수
# ==================================================

def format_pct(x):
    try:
        if x is None or pd.isna(x):
            return "-"
        return f"{float(x):.2f}%"
    except Exception:
        return "-"


def format_num(x):
    try:
        if x is None or pd.isna(x):
            return "-"
        return f"{float(x):,.2f}"
    except Exception:
        return "-"


def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


# ==================================================
# Upbit 데이터
# ==================================================

@st.cache_data(ttl=600)
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


@st.cache_data(ttl=300)
def get_top_trade_value_markets(limit=60):
    markets = get_krw_markets()
    markets = [m for m in markets if m != "KRW-BTC"]

    if not markets:
        return []

    tickers = []

    for i in range(0, len(markets), 100):
        batch = markets[i:i + 100]

        try:
            url = "https://api.upbit.com/v1/ticker"
            params = {"markets": ",".join(batch)}
            res = requests.get(url, params=params, timeout=10)
            data = res.json()

            if isinstance(data, list):
                tickers.extend(data)

            time.sleep(0.06)

        except Exception:
            time.sleep(0.15)

    if not tickers:
        return markets[:limit]

    tickers = sorted(
        tickers,
        key=lambda x: x.get("acc_trade_price_24h", 0),
        reverse=True
    )

    return [x["market"] for x in tickers[:limit]]


@st.cache_data(ttl=600)
def get_ohlcv(ticker, interval="minute240", count=600):
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

    result["ret_1"] = result["close"].pct_change()
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
# BTC 시장 필터
# ==================================================

@st.cache_data(ttl=600)
def get_btc_indicator_df(interval="minute240", count=800):
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
    btc_min_3bar_rise=-2.5,
    btc_require_close_above_ma20=False,
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
# 신호 판단 V3
# ==================================================

def judge_signal_at(
    df,
    idx,
    oversold_lookback=12,
    volume_ratio_threshold=1.3,
    max_3bar_rise=8.0,
    max_ma20_gap=6.0,
    min_signal_score=200,

    min_required_volume_ratio=1.3,
    max_short_k=70.0,
    max_short_d=60.0,
    max_middle_k=65.0,
    min_ma20_gap=-6.0,
    min_3bar_rise=-5.0,

    btc_df=None,
    use_btc_filter=True,
    btc_min_3bar_rise=-2.5,
    btc_require_close_above_ma20=False,
    btc_require_ma5_above_ma10=False,

    min_close_position=0.55,
    max_upper_wick_ratio=0.45,
    max_bear_candle_pct=0.5,
    require_ma5_slope=True,
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
    if use_btc_filter:
        btc_ok, btc_reason = check_btc_filter_at(
            btc_df=btc_df,
            signal_time=df.index[idx],
            btc_min_3bar_rise=btc_min_3bar_rise,
            btc_require_close_above_ma20=btc_require_close_above_ma20,
            btc_require_ma5_above_ma10=btc_require_ma5_above_ma10,
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

    if short_k > max_short_k:
        return None

    if short_d > max_short_d:
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

    if middle_k is not None and middle_k > max_middle_k:
        return None

    if middle_k is not None and prev_middle_k is not None:
        if middle_k > prev_middle_k and middle_k < 80:
            score += 20
            reasons.append("중기 K 상승")

    # 거래대금
    volume_ratio = safe_float(row["volume_ratio"])

    if volume_ratio is None or volume_ratio < min_required_volume_ratio:
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

    if close_position < min_close_position:
        return None

    if upper_wick_ratio > max_upper_wick_ratio:
        return None

    if candle_return_pct < -abs(max_bear_candle_pct):
        return None

    if require_ma5_slope and not ma5_slope_up:
        return None

    score += 20
    reasons.append(
        f"캔들품질 통과 / 종가위치 {close_position:.2f} / 윗꼬리 {upper_wick_ratio:.2f}"
    )

    if require_ma5_slope:
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

    if three_bar_rise_pct < min_3bar_rise:
        return None

    if three_bar_rise_pct > max_3bar_rise:
        return None

    if ma20_gap_pct < min_ma20_gap:
        return None

    if ma20_gap_pct > max_ma20_gap:
        return None

    if short_k >= 80 and middle_k is not None and middle_k >= 80:
        return None

    score += 10
    reasons.append("과열 아님")

    if score < min_signal_score:
        return None

    return {
        "signal_time": df.index[idx],
        "score": score,
        "reason": " / ".join(reasons),
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
        "ma5_slope_up": ma5_slope_up,
    }


# ==================================================
# 백테스트
# ==================================================

def backtest_ticker(
    ticker,
    interval="minute240",
    count=600,
    oversold_lookback=12,
    volume_ratio_threshold=1.3,
    min_signal_score=200,
    take_profit_pct=5.0,
    stop_loss_pct=4.0,
    max_hold_bars=12,
    fee_pct=0.05,
    slippage_pct=0.05,
    max_3bar_rise=8.0,
    max_ma20_gap=6.0,

    min_required_volume_ratio=1.3,
    max_short_k=70.0,
    max_short_d=60.0,
    max_middle_k=65.0,
    min_ma20_gap=-6.0,
    min_3bar_rise=-5.0,

    use_btc_filter=True,
    btc_min_3bar_rise=-2.5,
    btc_require_close_above_ma20=False,
    btc_require_ma5_above_ma10=False,

    min_close_position=0.55,
    max_upper_wick_ratio=0.45,
    max_bear_candle_pct=0.5,
    require_ma5_slope=True,
):
    df = get_ohlcv(ticker, interval=interval, count=count)

    if df is None or len(df) < 120:
        return []

    df = prepare_indicators(df)
    df = df.dropna().copy()

    if len(df) < 120:
        return []

    btc_df = None

    if use_btc_filter:
        btc_df = get_btc_indicator_df(interval=interval, count=count + 50)

        if btc_df is None or btc_df.empty:
            return []

    results = []
    last_exit_idx = -1

    for i in range(80, len(df) - max_hold_bars - 2):
        if i <= last_exit_idx:
            continue

        signal = judge_signal_at(
            df,
            idx=i,
            oversold_lookback=oversold_lookback,
            volume_ratio_threshold=volume_ratio_threshold,
            max_3bar_rise=max_3bar_rise,
            max_ma20_gap=max_ma20_gap,
            min_signal_score=min_signal_score,

            min_required_volume_ratio=min_required_volume_ratio,
            max_short_k=max_short_k,
            max_short_d=max_short_d,
            max_middle_k=max_middle_k,
            min_ma20_gap=min_ma20_gap,
            min_3bar_rise=min_3bar_rise,

            btc_df=btc_df,
            use_btc_filter=use_btc_filter,
            btc_min_3bar_rise=btc_min_3bar_rise,
            btc_require_close_above_ma20=btc_require_close_above_ma20,
            btc_require_ma5_above_ma10=btc_require_ma5_above_ma10,

            min_close_position=min_close_position,
            max_upper_wick_ratio=max_upper_wick_ratio,
            max_bear_candle_pct=max_bear_candle_pct,
            require_ma5_slope=require_ma5_slope,
        )

        if signal is None:
            continue

        entry_idx = i + 1

        if entry_idx >= len(df):
            continue

        entry_time = df.index[entry_idx]
        raw_entry_price = float(df["open"].iloc[entry_idx])
        entry_price = raw_entry_price * (1 + slippage_pct / 100)

        tp_price = entry_price * (1 + take_profit_pct / 100)
        sl_price = entry_price * (1 - stop_loss_pct / 100)

        exit_idx = None
        exit_time = None
        exit_price = None
        exit_reason = None

        max_gain = -999
        max_drawdown = 999

        final_exit_j = min(entry_idx + max_hold_bars, len(df) - 1)

        for j in range(entry_idx, final_exit_j + 1):
            high = float(df["high"].iloc[j])
            low = float(df["low"].iloc[j])
            close_j = float(df["close"].iloc[j])

            gain = (high - entry_price) / entry_price * 100
            drawdown = (low - entry_price) / entry_price * 100

            max_gain = max(max_gain, gain)
            max_drawdown = min(max_drawdown, drawdown)

            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

            if hit_tp and hit_sl:
                exit_idx = j
                exit_time = df.index[j]
                exit_price = sl_price * (1 - slippage_pct / 100)
                exit_reason = "SL_우선"
                break

            if hit_sl:
                exit_idx = j
                exit_time = df.index[j]
                exit_price = sl_price * (1 - slippage_pct / 100)
                exit_reason = "손절"
                break

            if hit_tp:
                exit_idx = j
                exit_time = df.index[j]
                exit_price = tp_price * (1 - slippage_pct / 100)
                exit_reason = "익절"
                break

            if j == final_exit_j:
                exit_idx = j
                exit_time = df.index[j]
                exit_price = close_j * (1 - slippage_pct / 100)
                exit_reason = "시간청산"
                break

        if exit_idx is None:
            continue

        gross_return = (exit_price - entry_price) / entry_price * 100
        net_return = gross_return - (fee_pct * 2)

        hold_bars = exit_idx - entry_idx + 1

        results.append({
            "코인": ticker,
            "신호시간": signal["signal_time"],
            "진입시간": entry_time,
            "청산시간": exit_time,
            "보유봉수": hold_bars,
            "신호점수": signal["score"],
            "진입가": entry_price,
            "청산가": exit_price,
            "수익률": net_return,
            "최대상승률": max_gain,
            "최대하락률": max_drawdown,
            "청산사유": exit_reason,
            "단기K": signal["short_k"],
            "단기D": signal["short_d"],
            "중기K": signal["middle_k"],
            "거래대금비율": signal["volume_ratio"],
            "MA20이격률": signal["ma20_gap_pct"],
            "3봉상승률": signal["three_bar_rise_pct"],
            "종가위치": signal["close_position"],
            "윗꼬리비율": signal["upper_wick_ratio"],
            "캔들등락률": signal["candle_return_pct"],
            "MA5상승": signal["ma5_slope_up"],
            "신호사유": signal["reason"],
        })

        last_exit_idx = exit_idx

    return results


def summarize_results(df):
    if df is None or df.empty:
        return {}

    wins = df[df["수익률"] > 0]
    losses = df[df["수익률"] <= 0]

    gross_profit = wins["수익률"].sum() if not wins.empty else 0
    gross_loss = abs(losses["수익률"].sum()) if not losses.empty else 0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "신호수": len(df),
        "승률": (df["수익률"] > 0).mean() * 100,
        "평균수익률": df["수익률"].mean(),
        "중앙수익률": df["수익률"].median(),
        "총합수익률": df["수익률"].sum(),
        "평균최대상승률": df["최대상승률"].mean(),
        "평균최대하락률": df["최대하락률"].mean(),
        "익절비율": (df["청산사유"] == "익절").mean() * 100,
        "손절비율": df["청산사유"].isin(["손절", "SL_우선"]).mean() * 100,
        "시간청산비율": (df["청산사유"] == "시간청산").mean() * 100,
        "평균보유봉수": df["보유봉수"].mean(),
        "Profit Factor": profit_factor,
    }


# ==================================================
# 설명
# ==================================================

with st.expander("백테스트 V3 방식 설명", expanded=False):
    st.markdown("""
### V3 핵심

V3는 V2보다 더 엄격하게 상승전환 후보를 걸러냅니다.

추가된 필터:

1. BTC 4시간봉 시장 필터
2. 신호 캔들 종가 위치
3. 긴 윗꼬리 제외
4. 강한 음봉 제외
5. MA5 상승 기울기
6. 거래대금비율 강화
7. MA20 이격률 강화

검증 방식:

- 신호 발생: 해당 봉 종가 기준
- 진입: 다음 봉 시가
- 청산: 익절, 손절, 시간청산
- 같은 봉에서 익절/손절 모두 닿으면 보수적으로 손절 우선
""")


# ==================================================
# 사이드바 설정
# ==================================================

st.sidebar.header("백테스트 V3 설정")

scan_mode = st.sidebar.selectbox(
    "스캔 대상",
    ["거래대금 상위", "수동 관심코인"],
    index=0
)

top_count = st.sidebar.number_input(
    "거래대금 상위 N개",
    min_value=5,
    max_value=100,
    value=30,
    step=5
)

manual_text = st.sidebar.text_area(
    "수동 관심코인",
    value="KRW-ETH\nKRW-XRP\nKRW-SOL\nKRW-DOGE\nKRW-ADA\nKRW-AVAX\nKRW-LINK\nKRW-SUI\nKRW-APT",
    height=160
)

interval = st.sidebar.selectbox(
    "기준 봉",
    ["minute240", "minute60"],
    index=0
)

count = st.sidebar.number_input(
    "조회 캔들 수",
    min_value=200,
    max_value=1500,
    value=700,
    step=100
)

st.sidebar.divider()

oversold_lookback = st.sidebar.number_input(
    "과매도 이력 확인 봉 수",
    min_value=3,
    max_value=50,
    value=12,
    step=1
)

volume_ratio_threshold = st.sidebar.number_input(
    "거래대금 증가 기준",
    min_value=1.0,
    max_value=5.0,
    value=1.3,
    step=0.1
)

min_signal_score = st.sidebar.number_input(
    "최소 신호 점수",
    min_value=50,
    max_value=350,
    value=200,
    step=5
)

st.sidebar.divider()
st.sidebar.subheader("강화 필터")

exclude_stable = st.sidebar.checkbox(
    "USDT/USDC 스테이블 코인 제외",
    value=True
)

min_required_volume_ratio = st.sidebar.number_input(
    "필수 최소 거래대금비율",
    min_value=0.0,
    max_value=5.0,
    value=1.3,
    step=0.1
)

max_short_k = st.sidebar.number_input(
    "최대 단기 K",
    min_value=20.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

max_short_d = st.sidebar.number_input(
    "최대 단기 D",
    min_value=20.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

max_middle_k = st.sidebar.number_input(
    "최대 중기 K",
    min_value=20.0,
    max_value=100.0,
    value=65.0,
    step=1.0
)

min_ma20_gap = st.sidebar.number_input(
    "최소 MA20 이격률 %",
    min_value=-50.0,
    max_value=0.0,
    value=-6.0,
    step=1.0
)

max_ma20_gap = st.sidebar.number_input(
    "최대 MA20 이격률 %",
    min_value=0.0,
    max_value=50.0,
    value=6.0,
    step=1.0
)

min_3bar_rise = st.sidebar.number_input(
    "최소 최근 3봉 상승률 %",
    min_value=-30.0,
    max_value=0.0,
    value=-5.0,
    step=1.0
)

max_3bar_rise = st.sidebar.number_input(
    "과열 제외 최근 3봉 상승률 %",
    min_value=3.0,
    max_value=50.0,
    value=8.0,
    step=1.0
)

st.sidebar.divider()
st.sidebar.subheader("BTC / 캔들 품질 필터")

use_btc_filter = st.sidebar.checkbox(
    "BTC 4H 필터 사용",
    value=True
)

btc_min_3bar_rise = st.sidebar.number_input(
    "BTC 최소 최근 3봉 상승률 %",
    min_value=-10.0,
    max_value=5.0,
    value=-2.5,
    step=0.5
)

btc_require_close_above_ma20 = st.sidebar.checkbox(
    "BTC 종가 > MA20 필수",
    value=False
)

btc_require_ma5_above_ma10 = st.sidebar.checkbox(
    "BTC MA5 > MA10 필수",
    value=False
)

min_close_position = st.sidebar.number_input(
    "최소 종가 위치",
    min_value=0.0,
    max_value=1.0,
    value=0.55,
    step=0.05
)

max_upper_wick_ratio = st.sidebar.number_input(
    "최대 윗꼬리 비율",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05
)

max_bear_candle_pct = st.sidebar.number_input(
    "허용 음봉 폭 %",
    min_value=0.0,
    max_value=5.0,
    value=0.5,
    step=0.1
)

require_ma5_slope = st.sidebar.checkbox(
    "MA5 상승 기울기 필수",
    value=True
)

st.sidebar.divider()

take_profit_pct = st.sidebar.number_input(
    "익절 %",
    min_value=1.0,
    max_value=30.0,
    value=5.0,
    step=0.5
)

stop_loss_pct = st.sidebar.number_input(
    "손절 %",
    min_value=1.0,
    max_value=20.0,
    value=4.0,
    step=0.5
)

max_hold_bars = st.sidebar.number_input(
    "최대 보유봉 수",
    min_value=1,
    max_value=50,
    value=12,
    step=1
)

fee_pct = st.sidebar.number_input(
    "편도 수수료 %",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01
)

slippage_pct = st.sidebar.number_input(
    "편도 슬리피지 %",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01
)

request_delay = st.sidebar.number_input(
    "요청 간격 초",
    min_value=0.02,
    max_value=1.0,
    value=0.08,
    step=0.02
)


# ==================================================
# 실행
# ==================================================

run = st.button("🧪 백테스트 V3 실행", type="primary", use_container_width=True)

if run:
    if scan_mode == "거래대금 상위":
        tickers = get_top_trade_value_markets(limit=int(top_count))
    else:
        all_krw = get_krw_markets()
        manual = [
            x.strip().upper()
            for x in manual_text.replace(",", "\n").splitlines()
            if x.strip()
        ]
        tickers = [x for x in manual if x in all_krw]

    tickers = [x for x in tickers if x != "KRW-BTC"]

    if exclude_stable:
        stable_excludes = ["KRW-USDT", "KRW-USDC"]
        tickers = [x for x in tickers if x not in stable_excludes]

    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        st.error("백테스트할 코인이 없습니다.")
        st.stop()

    st.info(f"총 {len(tickers)}개 코인을 백테스트합니다.")

    progress = st.progress(0)
    status = st.empty()

    all_results = []

    for idx, ticker in enumerate(tickers):
        status.write(f"백테스트 중: {ticker} ({idx + 1}/{len(tickers)})")

        results = backtest_ticker(
            ticker=ticker,
            interval=interval,
            count=int(count),
            oversold_lookback=int(oversold_lookback),
            volume_ratio_threshold=float(volume_ratio_threshold),
            min_signal_score=int(min_signal_score),
            take_profit_pct=float(take_profit_pct),
            stop_loss_pct=float(stop_loss_pct),
            max_hold_bars=int(max_hold_bars),
            fee_pct=float(fee_pct),
            slippage_pct=float(slippage_pct),
            max_3bar_rise=float(max_3bar_rise),
            max_ma20_gap=float(max_ma20_gap),

            min_required_volume_ratio=float(min_required_volume_ratio),
            max_short_k=float(max_short_k),
            max_short_d=float(max_short_d),
            max_middle_k=float(max_middle_k),
            min_ma20_gap=float(min_ma20_gap),
            min_3bar_rise=float(min_3bar_rise),

            use_btc_filter=bool(use_btc_filter),
            btc_min_3bar_rise=float(btc_min_3bar_rise),
            btc_require_close_above_ma20=bool(btc_require_close_above_ma20),
            btc_require_ma5_above_ma10=bool(btc_require_ma5_above_ma10),

            min_close_position=float(min_close_position),
            max_upper_wick_ratio=float(max_upper_wick_ratio),
            max_bear_candle_pct=float(max_bear_candle_pct),
            require_ma5_slope=bool(require_ma5_slope),
        )

        all_results.extend(results)

        progress.progress((idx + 1) / len(tickers))
        time.sleep(float(request_delay))

    status.write("백테스트 완료")

    if not all_results:
        st.warning("조건에 맞는 과거 신호가 없습니다.")
        st.info("""
조건이 너무 강할 수 있습니다.

완화 예시:
- 최소 신호 점수 200 → 180
- 필수 거래대금비율 1.3 → 1.1
- 최대 단기K 70 → 75
- 최대 단기D 60 → 65
- 최대 중기K 65 → 70
- 최소 종가 위치 0.55 → 0.50
- 최대 윗꼬리 비율 0.45 → 0.55
- MA5 상승 기울기 필수 해제
- 조회 캔들 수 700 → 1000
""")
        st.stop()

    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values("신호시간", ascending=False)

    summary = summarize_results(result_df)

    st.subheader("백테스트 V3 요약")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("신호수", summary.get("신호수", 0))
        st.metric("승률", format_pct(summary.get("승률")))

    with c2:
        st.metric("평균수익률", format_pct(summary.get("평균수익률")))
        st.metric("중앙수익률", format_pct(summary.get("중앙수익률")))

    with c3:
        st.metric("평균최대상승률", format_pct(summary.get("평균최대상승률")))
        st.metric("평균최대하락률", format_pct(summary.get("평균최대하락률")))

    with c4:
        st.metric("익절비율", format_pct(summary.get("익절비율")))
        st.metric("손절비율", format_pct(summary.get("손절비율")))

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.metric("시간청산비율", format_pct(summary.get("시간청산비율")))

    with c6:
        st.metric("총합수익률", format_pct(summary.get("총합수익률")))

    with c7:
        st.metric("평균보유봉수", format_num(summary.get("평균보유봉수")))

    with c8:
        pf = summary.get("Profit Factor")
        st.metric("Profit Factor", "-" if pd.isna(pf) else f"{pf:.2f}")

    st.subheader("결과 해석")

    avg_ret = summary.get("평균수익률", 0)
    win_rate = summary.get("승률", 0)
    pf = summary.get("Profit Factor", np.nan)
    signal_count = summary.get("신호수", 0)

    if avg_ret > 0 and win_rate >= 50 and (not pd.isna(pf) and pf >= 1.2):
        st.success("현재 V3 설정은 긍정적입니다. 텔레그램 알림 조건 후보로 검토할 수 있습니다.")
    elif avg_ret > 0:
        st.info("평균수익률은 양호하지만 승률 또는 손익비가 애매합니다. 추가 조정이 필요합니다.")
    else:
        st.warning("현재 설정은 수익성이 부족합니다. 필터 또는 익절/손절 조건을 조정하세요.")

    if signal_count < 10:
        st.warning("신호수가 너무 적습니다. 조건이 과도하게 강할 수 있습니다.")
    elif signal_count > 200:
        st.warning("신호수가 많습니다. 텔레그램 알림용으로는 조건을 더 강화하는 것이 좋습니다.")

    st.subheader("백테스트 V3 상세 결과")

    display_df = result_df.copy()

    round_cols = [
        "수익률",
        "최대상승률",
        "최대하락률",
        "단기K",
        "단기D",
        "중기K",
        "거래대금비율",
        "MA20이격률",
        "3봉상승률",
        "종가위치",
        "윗꼬리비율",
        "캔들등락률",
    ]

    for col in round_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)

    view_cols = [
        "코인",
        "신호시간",
        "진입시간",
        "청산시간",
        "보유봉수",
        "신호점수",
        "수익률",
        "최대상승률",
        "최대하락률",
        "청산사유",
        "단기K",
        "단기D",
        "중기K",
        "거래대금비율",
        "MA20이격률",
        "3봉상승률",
        "종가위치",
        "윗꼬리비율",
        "캔들등락률",
        "MA5상승",
        "신호사유",
    ]

    available_cols = [c for c in view_cols if c in display_df.columns]

    st.dataframe(
        display_df[available_cols],
        use_container_width=True,
        hide_index=True
    )

    csv = result_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "CSV 다운로드",
        data=csv,
        file_name=f"backtest_v3_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    st.info("왼쪽 설정을 확인한 뒤 백테스트 V3를 실행하세요.")

    st.markdown("""
## V3 추천 초기 설정

처음에는 기본값 그대로 실행해보세요.

```text
스캔 대상: 거래대금 상위 30개
기준 봉: minute240
조회 캔들 수: 700

과매도 이력 확인 봉 수: 12
거래대금 증가 기준: 1.3
최소 신호 점수: 200

필수 최소 거래대금비율: 1.3
최대 단기K: 70
최대 단기D: 60
최대 중기K: 65
MA20 이격률: -6% ~ +6%
최근 3봉 상승률: -5% ~ +8%

BTC 4H 필터 사용: 체크
BTC 최소 최근 3봉 상승률: -2.5%
BTC 종가 > MA20 필수: 미체크
BTC MA5 > MA10 필수: 미체크

최소 종가 위치: 0.55
최대 윗꼬리 비율: 0.45
허용 음봉 폭: 0.5%
MA5 상승 기울기 필수: 체크

익절: 5%
손절: 4%
최대 보유봉 수: 12
수수료: 0.05%
슬리피지: 0.05%
```
""")
