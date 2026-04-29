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
    page_title="상승전환 백테스트",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 상승전환 전략 백테스트")
st.caption("4시간봉 기준: 과매도 이력 + Stoch RSI 전환 + 거래대금 증가 + MA 회복")


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
# 기본 유틸
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
        df = pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=count)

        if df is None or df.empty:
            return None

        df = df.copy()
        df = df.sort_index()

        # 거래대금 컬럼 없으면 생성
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

    return result


# ==================================================
# 신호 판단
# ==================================================

def judge_signal_at(
    df,
    idx,
    oversold_lookback=12,
    volume_ratio_threshold=1.3,
    max_3bar_rise=12.0,
    max_ma20_gap=12.0,
    min_signal_score=100,
):
    """
    idx 위치의 4시간봉 종가 기준으로 신호 판단.
    진입은 다음 봉 시가로 별도 처리.
    """

    if idx < 80:
        return None

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]
    prev2 = df.iloc[idx - 2]
    prev3 = df.iloc[idx - 3]

    score = 0
    reasons = []

    # --------------------------------------------------
    # 1. 최근 과매도 이력
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2. 단기 Stoch RSI 전환
    # --------------------------------------------------
    short_k = safe_float(row["short_k"])
    short_d = safe_float(row["short_d"])
    prev_short_k = safe_float(prev["short_k"])
    prev_short_d = safe_float(prev["short_d"])

    if short_k is None or short_d is None or prev_short_k is None:
        return None

    kd_cross = short_k > short_d and prev_short_k <= prev_short_d
    kd_above = short_k > short_d
    k_recover_20 = prev_short_k <= 20 and short_k > 20
    k_rising = short_k > prev_short_k

    if kd_cross:
        score += 30
        reasons.append("4H 단기 K>D 골든크로스")
    elif kd_above:
        score += 18
        reasons.append("4H 단기 K>D 유지")

    if k_recover_20:
        score += 35
        reasons.append("4H 단기 K 20 회복")
    elif k_rising and short_k < 80:
        score += 15
        reasons.append("4H 단기 K 상승 중")

    # --------------------------------------------------
    # 3. 중기 Stoch RSI 개선
    # --------------------------------------------------
    middle_k = safe_float(row["middle_k"])
    prev_middle_k = safe_float(prev["middle_k"])

    if middle_k is not None and prev_middle_k is not None:
        if middle_k > prev_middle_k and middle_k < 80:
            score += 20
            reasons.append("4H 중기 K 상승")

    # --------------------------------------------------
    # 4. 거래대금 증가
    # --------------------------------------------------
    volume_ratio = safe_float(row["volume_ratio"])

    if volume_ratio is not None:
        if volume_ratio >= volume_ratio_threshold:
            score += 35
            reasons.append(f"거래대금 증가 x{volume_ratio:.2f}")
        elif volume_ratio >= 1.1:
            score += 15
            reasons.append(f"거래대금 완만 증가 x{volume_ratio:.2f}")

    # --------------------------------------------------
    # 5. MA 회복
    # --------------------------------------------------
    close = safe_float(row["close"])
    ma5 = safe_float(row["ma5"])
    ma10 = safe_float(row["ma10"])
    ma20 = safe_float(row["ma20"])

    if close is None or ma5 is None or ma10 is None or ma20 is None:
        return None

    if close > ma5:
        score += 15
        reasons.append("MA5 회복")

    if close > ma10:
        score += 10
        reasons.append("MA10 회복")

    # --------------------------------------------------
    # 6. MACD Histogram 개선
    # --------------------------------------------------
    hist = safe_float(row["macd_hist"])
    hist1 = safe_float(prev["macd_hist"])
    hist2 = safe_float(prev2["macd_hist"])
    hist3 = safe_float(prev3["macd_hist"])

    if None not in [hist, hist1, hist2, hist3]:
        if hist > hist1 > hist2:
            score += 25
            reasons.append("MACD 히스토그램 3봉 개선")
        elif hist > hist1:
            score += 10
            reasons.append("MACD 히스토그램 개선")

    # --------------------------------------------------
    # 7. 과열 방지
    # --------------------------------------------------
    ret_3 = safe_float(row["ret_3"])

    if ret_3 is None:
        return None

    three_bar_rise_pct = ret_3 * 100
    ma20_gap_pct = ((close - ma20) / ma20) * 100 if ma20 else 0

    overheat = False

    if three_bar_rise_pct >= max_3bar_rise:
        overheat = True
        reasons.append(f"과열 제외: 3봉 상승률 {three_bar_rise_pct:.2f}%")

    if ma20_gap_pct >= max_ma20_gap:
        overheat = True
        reasons.append(f"과열 제외: MA20 이격 {ma20_gap_pct:.2f}%")

    if short_k >= 80 and middle_k is not None and middle_k >= 80:
        overheat = True
        reasons.append("과열 제외: 단기/중기 Stoch RSI 과매수")

    if not overheat:
        score += 10
        reasons.append("과열 아님")

    # --------------------------------------------------
    # 최종 조건
    # --------------------------------------------------
    # 최소한 과매도 이력 1개는 있어야 함
    if oversold_count == 0:
        return None

    # 핵심 전환 조건: K>D 또는 K 20 회복
    if not (kd_above or k_recover_20):
        return None

    # 과열이면 제외
    if overheat:
        return None

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
    min_signal_score=100,
    take_profit_pct=5.0,
    stop_loss_pct=4.0,
    max_hold_bars=12,
    fee_pct=0.05,
    slippage_pct=0.05,
    max_3bar_rise=12.0,
    max_ma20_gap=12.0,
):
    df = get_ohlcv(ticker, interval=interval, count=count)

    if df is None or len(df) < 120:
        return []

    df = prepare_indicators(df)
    df = df.dropna().copy()

    if len(df) < 120:
        return []

    results = []
    last_exit_idx = -1

    for i in range(80, len(df) - max_hold_bars - 2):
        # 포지션 중복 방지
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

        for j in range(entry_idx, min(entry_idx + max_hold_bars, len(df) - 1) + 1):
            high = float(df["high"].iloc[j])
            low = float(df["low"].iloc[j])
            close = float(df["close"].iloc[j])

            gain = (high - entry_price) / entry_price * 100
            drawdown = (low - entry_price) / entry_price * 100

            max_gain = max(max_gain, gain)
            max_drawdown = min(max_drawdown, drawdown)

            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

            # 같은 봉에서 TP/SL 둘 다 닿으면 보수적으로 SL 먼저 처리
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

            # 마지막 보유봉 청산
            if j == min(entry_idx + max_hold_bars, len(df) - 1):
                exit_idx = j
                exit_time = df.index[j]
                exit_price = close * (1 - slippage_pct / 100)
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
        "평균보유봉수": df["보유봉수"].mean(),
        "Profit Factor": profit_factor,
    }


# ==================================================
# UI
# ==================================================

with st.expander("백테스트 방식 설명", expanded=False):
    st.markdown("""
### 검증 방식

- 기준 시간봉: 기본 4시간봉
- 신호 발생: 해당 4시간봉 종가 기준으로 판단
- 진입 가격: 다음 4시간봉 시가
- 청산 방식:
  - 목표수익 도달
  - 손절 도달
  - 최대 보유봉수 도달 시 시간청산
- 같은 봉에서 익절/손절 모두 닿으면 보수적으로 손절 우선 처리
- 수수료와 슬리피지를 반영

### 현재 검증하는 핵심 아이디어

과매도 상태에서 바로 매수하는 것이 아니라,

1. 과매도 이력이 있고
2. Stoch RSI가 전환되고
3. 거래대금이 증가하고
4. MA5/MA10을 회복하고
5. 과열은 아닌 경우

를 검증합니다.
""")


st.sidebar.header("백테스트 설정")

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
    index=0,
    help="우선 4시간봉 검증을 추천합니다."
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
    step=1,
    help="4시간봉 기준 12개는 약 2일입니다."
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
    max_value=250,
    value=110,
    step=5
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
    step=1,
    help="4시간봉 기준 12개는 약 2일입니다."
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

st.sidebar.divider()

max_3bar_rise = st.sidebar.number_input(
    "과열 제외: 최근 3봉 상승률 %",
    min_value=3.0,
    max_value=50.0,
    value=12.0,
    step=1.0
)

max_ma20_gap = st.sidebar.number_input(
    "과열 제외: MA20 이격률 %",
    min_value=3.0,
    max_value=50.0,
    value=12.0,
    step=1.0
)

request_delay = st.sidebar.number_input(
    "요청 간격 초",
    min_value=0.02,
    max_value=1.0,
    value=0.08,
    step=0.02
)


run = st.button("🧪 백테스트 실행", type="primary", use_container_width=True)

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
        )

        all_results.extend(results)

        progress.progress((idx + 1) / len(tickers))
        time.sleep(float(request_delay))

    status.write("백테스트 완료")

    if not all_results:
        st.warning("조건에 맞는 과거 신호가 없습니다.")
        st.info("최소 신호 점수를 낮추거나, 거래대금 증가 기준을 완화하거나, 조회 캔들 수를 늘려보세요.")
        st.stop()

    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values("신호시간", ascending=False)

    summary = summarize_results(result_df)

    st.subheader("백테스트 요약")

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

    c5, c6, c7 = st.columns(3)

    with c5:
        st.metric("총합수익률", format_pct(summary.get("총합수익률")))

    with c6:
        st.metric("평균보유봉수", format_num(summary.get("평균보유봉수")))

    with c7:
        pf = summary.get("Profit Factor")
        st.metric("Profit Factor", "-" if pd.isna(pf) else f"{pf:.2f}")

    st.subheader("결과 해석")

    avg_ret = summary.get("평균수익률", 0)
    win_rate = summary.get("승률", 0)
    avg_dd = summary.get("평균최대하락률", 0)
    pf = summary.get("Profit Factor", np.nan)

    if avg_ret > 0 and win_rate >= 50 and (not pd.isna(pf) and pf >= 1.2):
        st.success("현재 설정은 기본적으로 긍정적인 결과입니다. 다만 기간과 대상 코인을 바꿔 추가 검증하세요.")
    elif avg_ret > 0:
        st.info("평균수익률은 양호하지만 승률 또는 손익비가 애매합니다. 조건 조정이 필요합니다.")
    else:
        st.warning("현재 설정은 수익성이 부족합니다. 거래대금 기준, 신호점수, 익절/손절 비율을 조정해보세요.")

    st.write("""
### 체크할 핵심 기준

- 평균수익률이 양수인가?
- 승률이 50% 이상인가?
- 평균최대하락률이 손절폭보다 과도하지 않은가?
- Profit Factor가 1.2 이상인가?
- 신호수가 너무 적거나 너무 많지 않은가?
""")

    st.subheader("백테스트 상세 결과")

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
        "신호사유",
    ]

    st.dataframe(
        display_df[view_cols],
        use_container_width=True,
        hide_index=True
    )

    csv = result_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "CSV 다운로드",
        data=csv,
        file_name=f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    st.info("왼쪽 설정을 확인한 뒤 백테스트를 실행하세요.")

    st.markdown("""
## 추천 초기 설정

처음에는 아래 설정으로 검증해보세요.

```text
스캔 대상: 거래대금 상위 30개
기준 봉: minute240
조회 캔들 수: 700
과매도 이력 확인 봉 수: 12
거래대금 증가 기준: 1.3
최소 신호 점수: 110
익절: 5%
손절: 4%
최대 보유봉 수: 12
수수료: 0.05%
슬리피지: 0.05%
```

## 좋은 결과 기준

```text
승률: 50% 이상
평균수익률: 양수
Profit Factor: 1.2 이상
평균최대하락률: -4%~-6% 이내
익절비율 > 손절비율이면 좋음
```

## 결과가 안 좋으면

1. 최소 신호 점수 110 → 120으로 높이기
2. 거래대금 증가 기준 1.3 → 1.5로 높이기
3. 손절 4% → 3% 또는 3.5%로 낮추기
4. 익절 5% → 4% 또는 6%로 비교하기
5. 과열 제외 MA20 이격 12% → 8%로 낮추기

## 이번 백테스트의 목적

```text
과매도 이력
+ 4시간봉 Stoch RSI 상승전환
+ 거래대금 증가
+ MA5/MA10 회복
+ 과열 종목 제외
```

위 조건이 실제로 수익성이 있는지 확인하는 1차 검증입니다.
""")


