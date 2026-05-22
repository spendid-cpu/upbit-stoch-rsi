"""
scanner.py v2.4.3
- VERSION 상수 추가
- _migrate_watch_item 강화 (GX! 완전 수정)
- manual_activate_watch 추가
- run_single_scan 추가 (수동 스캔 API용)
"""

VERSION = 'v2.4.3'

# ── 기존 import/상수 아래에 추가 ──

def _migrate_watch_item(item: dict) -> dict:
    """레거시 문자열 dir_info를 dict로 변환"""
    for key in ['daily_dir_info', 'h4_dir_info', 'h1_dir_info']:
        v = item.get(key)
        if isinstance(v, str):
            # 문자열에서 방향/GX 파싱
            gx  = 'GX' in v or '✨' in v
            raw = v.replace('!','').replace('✨GX','').replace('GX','').strip()
            dir_map = {'↑':'상승','↗':'반등','→':'횡보','↓':'하락'}
            direction = dir_map.get(raw, '횡보')
            item[key] = {
                'direction':    direction,
                'golden_cross': gx,
                'gx':           gx,
                'raw':          v,
            }
        elif v is None:
            item[key] = {'direction':'횡보','golden_cross':False,'gx':False}
    return item

def _load_watch_list() -> list:
    """watch_list.json 로드 + 자동 마이그레이션"""
    if not os.path.exists(WATCH_LIST_FILE):
        return []
    try:
        with open(WATCH_LIST_FILE, 'r', encoding='utf-8') as f:
            items = json.load(f)
        return [_migrate_watch_item(i) for i in items]
    except Exception as e:
        logger.error(f'watch_list 로드 오류: {e}')
        return []

def run_single_scan():
    """수동 스캔 트리거 (API용)"""
    try:
        logger.info('수동 스캔 시작')
        scan_market()
    except Exception as e:
        logger.error(f'수동 스캔 오류: {e}')

def manual_activate_watch(ticker: str) -> dict:
    """Watch → Active 수동 진입"""
    ticker = ticker.upper()
    if not ticker.startswith('KRW-'):
        ticker = 'KRW-' + ticker
    try:
        watch_list     = _load_watch_list()
        active_trades  = _load_active_trades()

        # 이미 활성 체크
        if any(t['ticker']==ticker for t in active_trades):
            return {'success': False, 'message': f'{ticker} 이미 활성 트레이드 존재'}

        # Watch 목록에서 찾기
        item = next((w for w in watch_list if w['ticker']==ticker), None)
        if not item:
            return {'success': False, 'message': f'{ticker} Watch 목록에 없음'}

        # 현재가 조회
        prices = get_current_prices([ticker])
        price  = prices.get(ticker, 0)
        if not price:
            return {'success': False, 'message': '현재가 조회 실패'}

        # Active 항목 생성
        now  = time.time()
        tp   = round(price * 1.05, 8)
        sl   = round(price * 0.97, 8)
        trade = {
            'ticker':        ticker,
            'grade':         item.get('grade','C'),
            'score':         item.get('score', 0),
            'entry_price':   price,
            'current_price': price,
            'tp_price':      tp,
            'sl_price':      sl,
            'trade_type':    'manual',
            'entry_time':    now,
            'daily_k':       item.get('daily_k'),
            'h4_k':          item.get('h4_k'),
            'h1_k':          item.get('h1_k'),
            'entry_strength':item.get('entry_strength', 0),
        }
        active_trades.append(trade)

        # Watch에서 제거
        watch_list = [w for w in watch_list if w['ticker']!=ticker]
        _save_watch_list(watch_list)
        _save_active_trades(active_trades)

        # 상태 업데이트
        with _state_lock:
            scanner_state['watch_list']    = watch_list
            scanner_state['watch_count']   = len(watch_list)
            scanner_state['active_trades'] = active_trades
            scanner_state['active_count']  = len(active_trades)

        # 텔레그램 알림
        sym = ticker.replace('KRW-','')
        msg = (f"👆 수동진입\n"
               f"티커: {sym} ({item.get('grade','C')}등급)\n"
               f"점수: {item.get('score',0)}\n"
               f"진입가: {price:,.0f}\n"
               f"TP: {tp:,.0f} (+5%)\n"
               f"SL: {sl:,.0f} (-3%)\n"
               f"일K: {item.get('daily_k','?')} | 4hK: {item.get('h4_k','?')}")
        send_telegram(msg)

        logger.info(f'수동 진입: {ticker} @ {price}')
        return {'success': True, 'message': f'{ticker} 수동 진입 완료', 'price': price}

    except Exception as e:
        logger.error(f'manual_activate_watch 오류: {e}')
        return {'success': False, 'message': str(e)}
