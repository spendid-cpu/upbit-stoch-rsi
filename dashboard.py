# -*- coding: utf-8 -*-
"""
dashboard.py — Upbit MTF 스캐너 Flask 대시보드 (v2.0)
변경사항:
  - Watch 점수/등급 표시 (등록점수 vs 현재점수)
  - Active 트레이드 패널
  - 승률 통계 카드
  - 수동 청산 버튼
  - 히스토리 페이지
"""

import threading
import json
from datetime import datetime, timezone
from flask import Flask, jsonify, request, render_template_string
import scanner
import mtf_setup

app = Flask(__name__)

def _start_threads():
    t1 = threading.Thread(target=scanner.scanner_loop,     daemon=True)
    t2 = threading.Thread(target=scanner.price_check_loop, daemon=True)
    t1.start()
    t2.start()

_start_threads()

# ════════════════════════════════════════════════
# HTML 템플릿
# ════════════════════════════════════════════════

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Upbit MTF Scanner</title>
<style>
  :root {
    --bg:#0f1117; --card:#1a1d27; --border:#2a2d3e;
    --text:#e2e8f0; --sub:#94a3b8; --green:#22c55e;
    --red:#ef4444; --yellow:#f59e0b; --orange:#f97316;
    --blue:#3b82f6; --purple:#a855f7;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;font-size:14px}
  .wrap{max-width:1400px;margin:0 auto;padding:16px}
  h1{font-size:20px;font-weight:700;margin-bottom:16px}
  h2{font-size:15px;font-weight:600;margin:20px 0 10px;color:var(--sub)}

  /* 카드 그리드 */
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:20px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px}
  .card .label{font-size:11px;color:var(--sub);margin-bottom:6px}
  .card .value{font-size:20px;font-weight:700}

  /* 테이블 */
  .tbl-wrap{overflow-x:auto;margin-bottom:24px}
  table{width:100%;border-collapse:collapse;background:var(--card);border-radius:10px;overflow:hidden}
  th{background:#12151f;color:var(--sub);font-weight:500;padding:10px 12px;text-align:left;font-size:12px}
  td{padding:10px 12px;border-top:1px solid var(--border);vertical-align:middle}
  tr:hover td{background:#1e2235}

  /* 등급 뱃지 */
  .badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700}
  .g-S{background:#ef444433;color:#ef4444}
  .g-A{background:#f9731633;color:#f97316}
  .g-B{background:#f59e0b33;color:#f59e0b}
  .g-C{background:#94a3b833;color:#94a3b8}
  .g-manual{background:#a855f733;color:#a855f7}
  .g-auto{background:#22c55e33;color:#22c55e}

  /* 점수 변화 */
  .score-up{color:var(--green)}
  .score-dn{color:var(--red)}
  .score-eq{color:var(--sub)}

  /* 진행바 */
  .bar-wrap{background:#12151f;border-radius:4px;height:6px;min-width:60px}
  .bar{height:6px;border-radius:4px;transition:width .3s}
  .bar-green{background:var(--green)}
  .bar-red{background:var(--red)}
  .bar-yellow{background:var(--yellow)}

  /* 버튼 */
  .btn{padding:5px 12px;border-radius:6px;border:none;cursor:pointer;font-size:12px;font-weight:600}
  .btn-scan{background:var(--blue);color:#fff}
  .btn-close{background:var(--red);color:#fff}
  .btn-remove{background:#374151;color:var(--sub)}

  /* 수동 입력 폼 */
  .form-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
  input[type=text],input[type=number]{
    background:#12151f;border:1px solid var(--border);color:var(--text);
    padding:6px 10px;border-radius:6px;font-size:13px;width:140px
  }

  /* 탭 */
  .tabs{display:flex;gap:4px;margin-bottom:12px}
  .tab{padding:6px 14px;border-radius:6px;cursor:pointer;font-size:12px;background:#12151f;color:var(--sub)}
  .tab.active{background:var(--blue);color:#fff}

  /* 통계 섹션 */
  .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:16px}
  .stat-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px}
  .stat-card .title{font-size:11px;color:var(--sub);margin-bottom:8px}
  .stat-row{display:flex;justify-content:space-between;margin-bottom:4px;font-size:13px}

  /* 모바일 */
  @media(max-width:600px){
    .cards{grid-template-columns:repeat(2,1fr)}
    td,th{padding:8px 8px;font-size:12px}
    .hide-mobile{display:none}
  }

  #msg{margin:8px 0;font-size:13px;color:var(--green)}
  .countdown{font-size:12px;color:var(--sub)}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
  .dot-green{background:var(--green)}
  .dot-yellow{background:var(--yellow);animation:blink 1s infinite}
  .dot-red{background:var(--red)}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
</style>
</head>
<body>
<div class="wrap">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:16px">
    <h1>🔍 Upbit MTF Scanner</h1>
    <div style="display:flex;align-items:center;gap:12px">
      <span id="status-dot"></span>
      <span class="countdown" id="countdown"></span>
      <button class="btn btn-scan" onclick="manualScan()">🔄 수동 스캔</button>
    </div>
  </div>

  <!-- 상태 카드 -->
  <div class="cards">
    <div class="card">
      <div class="label">스캔 종목</div>
      <div class="value" id="total-scanned">-</div>
    </div>
    <div class="card">
      <div class="label">Watch</div>
      <div class="value" id="watch-count" style="color:var(--yellow)">-</div>
    </div>
    <div class="card">
      <div class="label">Active</div>
      <div class="value" id="active-count" style="color:var(--orange)">-</div>
    </div>
    <div class="card">
      <div class="label">TP 승률</div>
      <div class="value" id="tp-rate" style="color:var(--green)">-</div>
    </div>
    <div class="card">
      <div class="label">BTC 주봉MA20</div>
      <div class="value" id="macro-weekly">-</div>
    </div>
    <div class="card">
      <div class="label">BTC 일봉MA20</div>
      <div class="value" id="macro-daily">-</div>
    </div>
  </div>

  <!-- 통계 -->
  <h2>📊 통계</h2>
  <div class="stat-grid" id="stat-grid"></div>

  <!-- Active 트레이드 -->
  <h2>🔥 Active 트레이드</h2>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>종목</th><th>등급</th><th>점수</th>
        <th>진입가</th><th>현재가</th><th>수익률</th>
        <th>진행</th><th class="hide-mobile">진입시간</th><th>청산</th>
      </tr></thead>
      <tbody id="active-tbody"></tbody>
    </table>
  </div>

  <!-- Watch List -->
  <h2>👁 Watch List</h2>
  <div class="form-row">
    <input type="text" id="w-ticker" placeholder="종목 (예: BTC)">
    <input type="number" id="w-price" placeholder="진입가">
    <button class="btn btn-scan" onclick="addWatch()">+ 수동 등록</button>
    <span id="msg"></span>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>종목</th><th>등급</th>
        <th>등록점수</th><th>현재점수</th><th>변화</th>
        <th class="hide-mobile">일봉K</th>
        <th class="hide-mobile">4hK</th>
        <th class="hide-mobile">1hK</th>
        <th>등록가</th><th>현재가</th><th>수익률</th>
        <th>구분</th><th>삭제</th>
      </tr></thead>
      <tbody id="watch-tbody"></tbody>
    </table>
  </div>

  <!-- 신규 등록 / 삭제 -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div>
      <h2>🆕 신규 Watch</h2>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>종목</th><th>등급</th><th>점수</th><th>일봉K</th></tr></thead>
          <tbody id="new-tbody"></tbody>
        </table>
      </div>
    </div>
    <div>
      <h2>🗑 만료/제거</h2>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>종목</th><th>등급</th><th>사유</th></tr></thead>
          <tbody id="removed-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <p style="color:var(--sub);font-size:11px;margin-top:20px;text-align:center">
    마지막 스캔: <span id="last-scan">-</span> |
    가격 업데이트: <span id="last-price">-</span>
  </p>
</div>

<script>
const GRADE_COLOR = {S:'#ef4444',A:'#f97316',B:'#f59e0b',C:'#94a3b8'};
const TP  = {{ tp_pct }};
const SL  = {{ sl_pct }};

function fmt(v,dec=0){ return v==null?'-':Number(v).toLocaleString('ko-KR',{minimumFractionDigits:dec,maximumFractionDigits:dec}); }
function fmtPct(v){ if(v==null)return'-'; let s=v>=0?'+':''; return `<span style="color:${v>=0?'#22c55e':'#ef4444'}">${s}${Number(v).toFixed(2)}%</span>`; }
function gradeBadge(g){ return `<span class="badge g-${g}">${g}</span>`; }
function scoreChange(init,cur){
  let d=cur-init;
  if(d>0) return `<span class="score-up">+${d}↑</span>`;
  if(d<0) return `<span class="score-dn">${d}↓</span>`;
  return `<span class="score-eq">→</span>`;
}

async function fetchState(){
  try{
    const r = await fetch('/api/state');
    const s = await r.json();
    updateUI(s);
  }catch(e){ console.error(e); }
}

function updateUI(s){
  // 상태 dot
  const dot = document.getElementById('status-dot');
  if(s.status==='scanning'){
    dot.innerHTML='<span class="dot dot-yellow"></span>스캔 중...';
  } else if(s.status==='done'){
    dot.innerHTML='<span class="dot dot-green"></span>정상';
  } else {
    dot.innerHTML='<span class="dot dot-red"></span>오류';
  }

  document.getElementById('total-scanned').textContent = s.total_scanned||'-';
  document.getElementById('watch-count').textContent   = (s.watch_list||[]).length;
  document.getElementById('active-count').textContent  = (s.active_trades||[]).length;

  // 매크로
  const m = s.macro||{};
  const wd = m.weekly_distance_pct;
  document.getElementById('macro-weekly').innerHTML =
    wd!=null ? `<span style="color:${wd>=0?'#22c55e':'#ef4444'}">${wd>=0?'+':''}${wd}%</span>` : '-';
  const dd = m.daily_distance_pct;
  document.getElementById('macro-daily').innerHTML =
    dd!=null ? `<span style="color:${dd>=0?'#22c55e':'#ef4444'}">${dd>=0?'+':''}${dd}%</span>` : '-';

  // 통계
  const st = s.stats||{};
  document.getElementById('tp-rate').textContent = st.tp_rate!=null ? st.tp_rate+'%' : '-';
  renderStats(st);

  // Active
  renderActive(s.active_trades||[]);

  // Watch
  renderWatch(s.watch_list||[]);

  // 신규/삭제
  const newTb = document.getElementById('new-tbody');
  newTb.innerHTML = (s.new_entries||[]).map(w=>`
    <tr>
      <td><b>${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(w.current?.grade||'?')}</td>
      <td>${w.current?.score||0}</td>
      <td>${w.current?.daily_k??'-'}</td>
    </tr>`).join('') || '<tr><td colspan="4" style="color:var(--sub);text-align:center">없음</td></tr>';

  const rmTb = document.getElementById('removed-tbody');
  rmTb.innerHTML = (s.removed_items||[]).map(w=>`
    <tr>
      <td><b>${(w.ticker||'').replace('KRW-','')}</b></td>
      <td>${gradeBadge(w.current?.grade||w.grade||'?')}</td>
      <td style="color:var(--sub);font-size:11px">만료</td>
    </tr>`).join('') || '<tr><td colspan="3" style="color:var(--sub);text-align:center">없음</td></tr>';

  // 시간
  if(s.last_scan_at) document.getElementById('last-scan').textContent = new Date(s.last_scan_at).toLocaleString('ko-KR');
  if(s.last_price_check_at) document.getElementById('last-price').textContent = new Date(s.last_price_check_at).toLocaleString('ko-KR');
}

function renderStats(st){
  const gs = st.grade_stats||{};
  document.getElementById('stat-grid').innerHTML = `
    <div class="stat-card">
      <div class="title">📈 Watch 전체</div>
      <div class="stat-row"><span>총 등록</span><span>${st.total||0}건</span></div>
      <div class="stat-row"><span>Active 전환율</span><span style="color:var(--green)">${st.watch_to_active_rate||0}%</span></div>
      <div class="stat-row"><span>만료</span><span style="color:var(--sub)">${st.expired||0}건</span></div>
    </div>
    <div class="stat-card">
      <div class="title">💰 Active 청산</div>
      <div class="stat-row"><span>TP</span><span style="color:var(--green)">${st.tp||0}건 (${st.tp_rate||0}%)</span></div>
      <div class="stat-row"><span>SL</span><span style="color:var(--red)">${st.sl||0}건</span></div>
      <div class="stat-row"><span>Timeout</span><span style="color:var(--sub)">${st.timeout||0}건</span></div>
      <div class="stat-row"><span>평균 수익률</span><span style="color:${(st.avg_pnl||0)>=0?'var(--green)':'var(--red)'}">
        ${st.avg_pnl!=null?(st.avg_pnl>=0?'+':'')+st.avg_pnl+'%':'-'}</span></div>
    </div>
    ${['S','A','B','C'].map(g=>{
      const d = gs[g]||{};
      return `<div class="stat-card">
        <div class="title">${g}등급 통계</div>
        <div class="stat-row"><span>청산</span><span>${d.total||0}건</span></div>
        <div class="stat-row"><span>TP 승률</span><span style="color:var(--green)">${d.tp_rate||0}%</span></div>
        <div class="stat-row"><span>평균 수익</span><span style="color:${(d.avg_pnl||0)>=0?'var(--green)':'var(--red)'}">
          ${d.avg_pnl!=null?(d.avg_pnl>=0?'+':'')+d.avg_pnl+'%':'-'}</span></div>
      </div>`;
    }).join('')}
  `;
}

function renderActive(trades){
  const tb = document.getElementById('active-tbody');
  if(!trades.length){
    tb.innerHTML='<tr><td colspan="9" style="color:var(--sub);text-align:center;padding:20px">Active 트레이드 없음</td></tr>';
    return;
  }
  tb.innerHTML = trades.map(t=>{
    const pnl = t.pnl_pct||0;
    const barW = Math.min(Math.abs(pnl)/Math.max(TP,SL)*100, 100);
    const barColor = pnl>=0?'bar-green':'bar-red';
    const actAt = t.activated_at ? new Date(t.activated_at).toLocaleString('ko-KR',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}) : '-';
    return `<tr>
      <td><b>${t.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(t.grade||'?')}</td>
      <td>${t.entry_score||0}</td>
      <td>${fmt(t.entry_price)}</td>
      <td>${fmt(t.current_price)}</td>
      <td>${fmtPct(pnl)}</td>
      <td>
        <div class="bar-wrap"><div class="bar ${barColor}" style="width:${barW}%"></div></div>
        <div style="font-size:10px;color:var(--sub);margin-top:2px">TP:+${TP}% SL:-${SL}%</div>
      </td>
      <td class="hide-mobile" style="font-size:11px;color:var(--sub)">${actAt}</td>
      <td><button class="btn btn-close" onclick="closeTrade('${t.ticker}')">청산</button></td>
    </tr>`;
  }).join('');
}

function renderWatch(list){
  const tb = document.getElementById('watch-tbody');
  if(!list.length){
    tb.innerHTML='<tr><td colspan="13" style="color:var(--sub);text-align:center;padding:20px">Watch 종목 없음</td></tr>';
    return;
  }
  tb.innerHTML = list.map(w=>{
    const snap = w.snapshot||{};
    const cur  = w.current||{};
    const initScore = snap.score||0;
    const curScore  = cur.score||0;
    const ep  = snap.entry_price||0;
    const cp  = cur.price||0;
    const profit = ep>0 ? ((cp-ep)/ep*100) : 0;
    const isManual = w.manual;
    return `<tr>
      <td><b>${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(cur.grade||snap.grade||'C')}</td>
      <td style="color:var(--sub)">${initScore}</td>
      <td style="color:${GRADE_COLOR[cur.grade]||'#94a3b8'};font-weight:700">${curScore}</td>
      <td>${scoreChange(initScore, curScore)}</td>
      <td class="hide-mobile" style="font-size:12px">${cur.daily_k??snap.daily_k??'-'}</td>
      <td class="hide-mobile" style="font-size:12px">${cur.h4_k??snap.h4_k??'-'}</td>
      <td class="hide-mobile" style="font-size:12px">${cur.h1_k??snap.h1_k??'-'}</td>
      <td>${fmt(ep)}</td>
      <td>${fmt(cp)}</td>
      <td>${fmtPct(profit)}</td>
      <td>
        <span class="badge ${isManual?'g-manual':'g-auto'}">${isManual?'수동':'자동'}</span>
      </td>
      <td><button class="btn btn-remove" onclick="removeWatch('${w.ticker}')">×</button></td>
    </tr>`;
  }).join('');
}

// 카운트다운
let nextScanAt = null;
function updateCountdown(){
  if(!nextScanAt){ document.getElementById('countdown').textContent=''; return; }
  const diff = Math.max(0, Math.floor((new Date(nextScanAt)-Date.now())/1000));
  const m = Math.floor(diff/60), s = diff%60;
  document.getElementById('countdown').textContent = `다음 스캔 ${m}:${String(s).padStart(2,'0')}`;
}

async function manualScan(){
  await fetch('/api/scan', {method:'POST'});
  document.getElementById('msg').textContent='스캔 요청됨';
  setTimeout(()=>document.getElementById('msg').textContent='', 3000);
}

async function addWatch(){
  const ticker = document.getElementById('w-ticker').value.trim();
  const price  = document.getElementById('w-price').value.trim();
  if(!ticker||!price){ alert('종목과 진입가를 입력하세요'); return; }
  const r = await fetch('/api/watch/add', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ticker, entry_price: parseFloat(price)})
  });
  const d = await r.json();
  document.getElementById('msg').textContent = d.message||'';
  if(d.ok){ document.getElementById('w-ticker').value=''; document.getElementById('w-price').value=''; fetchState(); }
}

async function removeWatch(ticker){
  if(!confirm(`${ticker} Watch 삭제?`)) return;
  await fetch('/api/watch/remove', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ticker})
  });
  fetchState();
}

async function closeTrade(ticker){
  if(!confirm(`${ticker} 수동 청산?`)) return;
  await fetch('/api/trade/close', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ticker})
  });
  fetchState();
}

// 주기적 갱신
async function poll(){
  const r = await fetch('/api/state');
  const s = await r.json();
  nextScanAt = s.next_scan_at;
  updateUI(s);
}
setInterval(poll, 15000);
setInterval(updateCountdown, 1000);
poll();
</script>
</body>
</html>
"""

# ════════════════════════════════════════════════
# Routes
# ════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(
        HTML,
        tp_pct=scanner.TRADE_TP_PCT,
        sl_pct=scanner.TRADE_SL_PCT,
    )

@app.route('/api/state')
def api_state():
    with scanner._state_lock:
        return jsonify(scanner.scanner_state)

@app.route('/api/config')
def api_config():
    return jsonify(mtf_setup.get_module_config())

@app.route('/api/scan', methods=['POST'])
def api_scan():
    scanner.manual_scan()
    return jsonify({'ok': True, 'message': '스캔 요청됨'})

@app.route('/api/watch/add', methods=['POST'])
def api_watch_add():
    data = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    price  = data.get('entry_price')
    if not ticker or price is None:
        return jsonify({'ok': False, 'message': '종목과 진입가 필요'}), 400
    try:
        price = float(price)
    except:
        return jsonify({'ok': False, 'message': '진입가 형식 오류'}), 400
    ok, msg = scanner.add_manual_watch(ticker, price)
    return jsonify({'ok': ok, 'message': msg})

@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    if not ticker:
        return jsonify({'ok': False, 'message': '종목 필요'}), 400
    ok, msg = scanner.remove_watch(ticker)
    return jsonify({'ok': ok, 'message': msg})

@app.route('/api/trade/close', methods=['POST'])
def api_trade_close():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    price  = data.get('price')
    if not ticker:
        return jsonify({'ok': False, 'message': '종목 필요'}), 400
    ok, msg = scanner.manual_close_trade(ticker, price)
    return jsonify({'ok': ok, 'message': msg})

@app.route('/api/history')
def api_history():
    return jsonify(scanner.load_trade_history())

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
