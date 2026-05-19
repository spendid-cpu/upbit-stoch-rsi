# ==================================================
# ★ BTC 주봉 MA20 필터 (백테스트용)
# ==================================================

@st.cache_data(ttl=600)
def get_btc_weekly_df(count=120):
    """
    BTC 주봉 OHLCV. 백테스트 히스토리에 충분한 개수를 확보합니다.
    """
    try:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="week", count=count)
        if df is None or df.empty:
            return None
        return df.sort_index()
    except Exception:
        return None


def get_btc_indicator_df(interval="week", count=120):
    """
    백테스트 내부에서 호출되는 BTC 지표 DF.
    interval 인자는 무시하고 항상 주봉을 사용합니다.
    """
    df = get_btc_weekly_df(count=count)
    if df is None or df.empty:
        return None

    result = df.copy()
    result["ma20"]  = result["close"].rolling(20).mean()
    result["ret_3"] = result["close"].pct_change(3)   # 주봉 기준 3봉 = 3주
    return result.dropna().copy()


def check_btc_filter_at(
    btc_df,
    signal_time,
    btc_min_3bar_rise=-2.0,
    btc_require_close_above_ma20=True,
    btc_require_ma5_above_ma10=False,   # 주봉에서는 미사용 (호환용 파라미터 유지)
):
    """
    특정 signal_time 시점에서 BTC 주봉 MA20 필터를 체크합니다.
    signal_time 이전에 완성된 가장 최근 주봉을 기준으로 판단합니다.
    """
    if btc_df is None or btc_df.empty:
        return False, "BTC 데이터 없음"

    try:
        # signal_time 이전에 완성된 주봉만 사용 (미래 데이터 방지)
        available = btc_df[btc_df.index < signal_time]
        if available.empty:
            return False, "BTC 매칭 실패"

        row = available.iloc[-1]

        btc_close   = safe_float(row["close"])
        btc_ma20    = safe_float(row["ma20"])
        btc_ret_3   = safe_float(row["ret_3"])   # 주봉 3봉 전 대비

        if None in [btc_close, btc_ma20, btc_ret_3]:
            return False, "BTC 지표 부족"

        btc_ret_3_pct   = btc_ret_3 * 100
        close_above_ma20 = btc_close > btc_ma20
        not_crashing    = btc_ret_3_pct > btc_min_3bar_rise

        if btc_require_close_above_ma20 and not close_above_ma20:
            return False, f"BTC 주봉 MA20 아래 / 3주 {btc_ret_3_pct:.2f}%"

        if close_above_ma20 or not_crashing:
            return True, f"BTC 주봉 통과 / 3주 {btc_ret_3_pct:.2f}%"

        return False, f"BTC 주봉 약세 / 3주 {btc_ret_3_pct:.2f}%"

    except Exception as e:
        return False, f"BTC 필터 오류: {e}"
