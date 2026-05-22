"""
dashboard.py v2.4.5
- 상단 버전 뱃지 추가
- /api/version 엔드포인트
- GX! 완전 수정
- 진입/제거 버튼 UI 개선
- 버전 클릭 시 변경이력 툴팁
- scanner 루프 함수 직접 스레드 시작 (start_background_tasks 제거)
"""

from flask import Flask, jsonify, request, render_template_string
import threading, json, os, time
import scanner
import mtf_setup

app = Flask(__name__)

DASHBOARD_VERSION = 'v2.4.5'

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Upbit MTF Scanner</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #7d8590; --green: #3fb950;
    --red: #f85149; --yellow: #d29922; --orange: #db6d28;
    --blue: #58a6ff; --purple: #bc8cff; --pink: #ff7b72;
    --teal: #39d353;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; font-size: 13px; }

  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px; background: var(--card);
    border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 100;
  }
  .header-left { display: flex; align-items: center; gap: 10px; }
  .header-title { font-size: 16px; font-weight: 700; color: var(--blue); }
  .version-badge {
    background: #1f2937; border: 1px solid var(--border);
    color: var(--muted); font-size: 10px; padding: 2px 7px;
    border-radius: 10px; cursor: pointer; position: relative;
    transition: all 0.2s;
  }
  .version-badge:hover { color: var(--blue); border-color: var(--blue); }
  .version-tooltip {
    display: none; position: absolute; top: 24px; left: 0;
    background: #1c2333; border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; width: 260px;
    font-size: 11px; line-height: 1.7; z-index: 200;
    color: var(--text); white-space: pre-wrap;
  }
  .version-badge:hover .version-tooltip { display: block; }
  .idle-badge {
    background: #1a2a1a; border: 1px solid var(--green);
    color: var(--green); font-size: 10px; padding: 2px 8px; border-radius: 10px;
  }
  .header-right { display: flex; align-items: center; gap: 12px; font-size: 12px; color: var(--muted); }
  .btn-scan {
    background: #1f3a5f; border: 1px solid var(--blue);
    color: var(--blue); padding: 4px 12px; border-radius: 6px;
    cursor: pointer; font-size: 12px; transition: all 0.2s;
  }
  .btn-scan:hover { background: var(--blue); color: #000; }

  .stats-row { display: flex; gap: 10px; padding: 14px 20px; flex-wrap: wrap; }
  .stat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 16px; flex: 1; min-width: 130px;
  }
  .stat-label { font-size: 10px; color: var(--muted); margin-bottom: 4px; }
  .stat-value { font-size: 20px; font-weight: 700; }
  .stat-sub { font-size: 10px; color: var(--muted); margin-top: 2px; }
  .green { color: var(--green); } .red { color: var(--red); }
  .yellow { color: var(--yellow); } .blue { color: var(--blue); }
  .orange { color: var(--orange); }

  .grade-bars { display: flex; gap: 8px; padding: 0 20px 14px; flex-wrap: wrap; }
  .grade-bar-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 14px; flex: 1; min-width: 120px;
  }
  .grade-bar-label { font-size: 10px; color: var(--muted); margin-bottom: 6px; }
  .grade-bar-track {
    height: 4px; background: #21262d; border-radius: 2px; overflow: hidden; margin-bottom: 4px;
  }
  .grade-bar-fill { height: 100%; border-radius: 2px; transition: width 0.5s; }
  .grade-bar-stats { font-size: 10px; color: var(--muted); }

  .tabs { display: flex; padding: 0 20px; border-bottom: 1px solid var(--border); }
  .tab {
    padding: 8px 16px; cursor: pointer; font-size: 12px;
    color: var(--muted); border-bottom: 2px solid transparent;
    transition: all 0.2s; display: flex; align-items: center; gap: 5px;
  }
  .tab.active { color: var(--blue); border-bottom-color: var(--blue); }
  .tab:hover { color: var(--text); }
  .tab-count {
    background: #21262d; color: var(--muted);
    font-size: 10px; padding: 1px 6px; border-radius: 8px;
  }
  .tab.active .tab-count { background: var(--blue); color: #000; }

  .entry-guide {
    background: #0d1f0d; border: 1px solid #1a3a1a;
    border-radius: 8px; margin: 12px 20px; padding: 10px 14px;
  }
  .entry-guide-title { font-size: 11px; color: var(--green); margin-bottom: 8px; font-weight: 600; }
  .entry-guide-items { display: flex; flex-wrap: wrap; gap: 12px; }
  .eg-item { font-size: 10px; color: var(--muted); line-height: 1.5; }

  .table-wrap { padding: 0 20px 20px; overflow-x: auto; }
  .search-row { display: flex; gap: 8px; margin-bottom: 10px; align-items: center; }
  .search-input {
    background: var(--card); border: 1px solid var(--border);
    color: var(--text); padding: 5px 10px; border-radius: 6px; font-size: 12px; width: 180px;
  }
  .btn-add {
    background: #1a3a1a; border: 1px solid var(--green);
    color: var(--green); padding: 5px 12px; border-radius: 6px;
    cursor: pointer; font-size: 12px;
  }
  table { width: 100%; border-collapse: collapse; }
  th {
    text-align: left; padding: 8px 10px; font-size: 10px;
    color: var(--muted); border-bottom: 1px solid var(--border);
    white-space: nowrap; font-weight: 500;
  }
  td { padding: 7px 10px; border-bottom: 1px solid #21262d; vertical-align: middle; }
  tr:hover td { background: #161b22; }
  .ticker-name { font-weight: 700; font-size: 13px; color: var(--text); }

  .k-extreme { color: var(--red); font-weight: 700; }
  .k-low     { color: var(--orange); }
  .k-mid     { color: var(--yellow); }
  .k-high    { color: var(--green); }
  .k-over    { color: var(--red); font-weight: 700; }

  .grade-s { background:#3d1a6e; color:var(--purple); border:1px solid var(--purple); padding:1px 6px; border-radius:4px; font-size:10px; font-weight:700; }
  .grade-a { background:#1a2a4a; color:var(--blue);   border:1px solid var(--blue);   padding:1px 6px; border-radius:4px; font-size:10px; font-weight:700; }
  .grade-b { background:#1a3a1a; color:var(--green);  border:1px solid var(--green);  padding:1px 6px; border-radius:4px; font-size:10px; font-weight:700; }
  .grade-c { background:#2a2a1a; color:var(--yellow); border:1px solid var(--yellow); padding:1px 6px; border-radius:4px; font-size:10px; font-weight:700; }

  .es-strong  { color:var(--green);  font-weight:700; }
  .es-target  { color:var(--blue);   font-weight:600; }
  .es-watch   { color:var(--yellow); }
  .es-wait    { color:var(--muted);  }
  .es-overheat { color:var(--orange); font-size:10px; }

  .dir-up   { color:var(--green); }
  .dir-down { color:var(--red);   }
  .dir-side { color:var(--muted); }
  .dir-gx   { color:var(--yellow); font-weight:700; }

  .type-auto   { background:#1a2a4a; color:var(--blue);   border:1px solid var(--blue);   font-size:9px; padding:1px 5px; border-radius:3px; }
  .type-manual { background:#1a3a1a; color:var(--green);  border:1px solid var(--green);  font-size:9px; padding:1px 5px; border-radius:3px; }
  .type-deep   { background:#3a1a2a; color:var(--pink);   border:1px solid var(--pink);   font-size:9px; padding:1px 5px; border-radius:3px; }

  .btn-entry {
    background:#1a3a1a; border:1px solid var(--green);
    color:var(--green); padding:3px 8px; border-radius:5px;
    cursor:pointer; font-size:11px; transition:all 0.2s; margin-right:4px;
  }
  .btn-entry:hover { background:var(--green); color:#000; }
  .btn-remove {
    background:#3a1a1a; border:1px solid var(--red);
    color:var(--red); padding:3px 8px; border-radius:5px;
    cursor:pointer; font-size:11px; transition:all 0.2s;
  }
  .btn-remove:hover { background:var(--red); color:#fff; }
  .btn-close {
    background:#3a1a1a; border:1px solid var(--red);
    color:var(--red); padding:3px 8px; border-radius:5px;
    cursor:pointer; font-size:11px;
  }

  .modal-overlay {
    display:none; position:fixed; top:0; left:0; width:100%; height:100%;
    background:rgba(0,0,0,0.7); z-index:1000; align-items:center; justify-content:center;
  }
  .modal-overlay.show { display:flex; }
  .modal {
    background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:24px; width:320px;
  }
  .modal-title { font-size:15px; font-weight:700; margin-bottom:16px; color:var(--blue); }
  .modal-row { display:flex; justify-content:space-between; margin-bottom:8px; font-size:12px; }
  .modal-row span:first-child { color:var(--muted); }
  .modal-btns { display:flex; gap:8px; margin-top:16px; }
  .modal-btns button { flex:1; padding:8px; border-radius:7px; cursor:pointer; font-size:12px; font-weight:600; }
  .btn-confirm { background:var(--green); border:none; color:#000; }
  .btn-cancel  { background:#21262d; border:1px solid var(--border); color:var(--text); }

  .countdown { font-size:10px; color:var(--muted); }
  .empty-state { text-align:center; padding:40px; color:var(--muted); font-size:13px; }
  ::-webkit-scrollbar { width:6px; height:6px; }
  ::-webkit-scrollbar-track { background:var(--bg); }
  ::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <span class="header-title">📊 Upbit MTF Scanner</span>
    <div class="version-badge">
      v2.4.5
      <div class="version-tooltip" id="versionTooltip">📦 dashboard  v2.4.5
🔧 scanner    v2.4.2
⚙️ mtf_setup  v4.2.3

변경이력:
• 상단 버전 뱃지 추가
• GX! 완전 수정
• 진입/제거 버튼 분리
• 레거시 dir_info 자동 마이그레이션
• scanner 루프 직접 스레드 시작</div>
    </div>
    <span class="idle-badge" id="idleBadge">● idle</span>
  </div>
  <div class="header-right">
    <span id="lastScanTime">마지막 스캔: --</span>
    <button class="btn-scan" onclick="manualScan()">🔄 수동 스캔</button>
  </div>
</div>

<div class="stats-row">
  <div class="stat-card">
    <div class="stat-label">Watch</div>
    <div class="stat-value blue" id="statWatch">0</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Active</div>
    <div class="stat-value green" id="statActive">0</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">DEEP Active</div>
    <div class="stat-value" id="statDeep">0</div>
    <div class="stat-sub">BTC 하락 중 비버</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">승률</div>
    <div class="stat-value" id="statWinrate">0.0%</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">평균 PnL</div>
    <div class="stat-value" id="statPnl">+0.00%</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">BTC MA20</div>
    <div class="stat-value blue" id="statBtcMa">-</div>
    <div class="stat-sub" id="statBtcSub">현재 - | -</div>
  </div>
</div>

<div class="grade-bars" id="gradeBars"></div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('watch')" id="tab-watch">
    👁 Watch <span class="tab-count" id="tc-watch">0</span>
  </div>
  <div class="tab" onclick="switchTab('active')" id="tab-active">
    ⚡ Active <span class="tab-count" id="tc-active">0</span>
  </div>
  <div class="tab" onclick="switchTab('deep')" id="tab-deep">
    🔴 DEEP <span class="tab-count" id="tc-deep">0</span>
  </div>
  <div class="tab" onclick="switchTab('new')" id="tab-new">
    🆕 신규 <span class="tab-count" id="tc-new">0</span>
  </div>
  <div class="tab" onclick="switchTab('history')" id="tab-history">
    📋 히스토리
  </div>
</div>

<div class="entry-guide">
  <div class="entry-guide-title">📌 진입강도 매매 가이드</div>
  <div class="entry-guide-items">
    <div class="eg-item">🚀 <b>강한신호</b> – S/A등급 → 적극 진입 고려</div>
    <div class="eg-item">🎯 <b>준비</b> – B등급 → 차트 확인 후 진입</div>
    <div class="eg-item">👀 <b>관찰</b> – 방향 형성 중 / 일봉K≤2 최소 보장</div>
    <div class="eg-item">⏳ <b>대기</b> – 신호 없음</div>
    <div class="eg-item">⚠ <b>4hK과열</b> – 4hK≥80 → 진입강도 1단계 하향</div>
  </div>
</div>

<div id="tabContent">
  <div id="pane-watch" class="tab-pane">
    <div class="table-wrap">
      <div class="search-row">
        <input class="search-input" id="watchSearch" placeholder="KRW-BTC or BTC" oninput="renderWatch()">
        <button class="btn-add" onclick="addWatch()">+ Watch 추가</button>
      </div>
      <table>
        <thead><tr>
          <th>티커</th><th>등급</th><th>점수</th><th>Δ</th>
          <th>일봉K</th><th>4hK</th><th>1hK</th>
          <th>진입강도</th><th>방향(일/4h/1h)</th><th>추세</th>
          <th>거래량</th><th>현재가</th><th>등록</th><th>만료</th><th>관리</th>
        </tr></thead>
        <tbody id="watchBody"></tbody>
      </table>
    </div>
  </div>

  <div id="pane-active" class="tab-pane" style="display:none">
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>티커</th><th>등급</th><th>타입</th><th>진입가</th>
          <th>현재가</th><th>PnL</th><th>TP</th><th>SL</th>
          <th>진입시각</th><th>관리</th>
        </tr></thead>
        <tbody id="activeBody"></tbody>
      </table>
    </div>
  </div>

  <div id="pane-deep" class="tab-pane" style="display:none">
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>티커</th><th>DEEP점수</th><th>일봉K</th><th>BTC상대강도</th>
          <th>바닥일수</th><th>거래량비율</th><th>진입시각</th><th>관리</th>
        </tr></thead>
        <tbody id="deepBody"></tbody>
      </table>
    </div>
  </div>

  <div id="pane-new" class="tab-pane" style="display:none">
    <div class="table-wrap">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <span style="font-size:11px; color:var(--muted)">최근 신규 감지 항목</span>
        <span class="countdown" id="nextScanCountdown">다음 스캔: --</span>
      </div>
      <table>
        <thead><tr>
          <th>티커</th><th>등급</th><th>점수</th><th>일봉K</th><th>4hK</th>
          <th>진입강도</th><th>방향</th><th>감지시각</th>
        </tr></thead>
        <tbody id="newBody"></tbody>
      </table>
    </div>
  </div>

  <div id="pane-history" class="tab-pane" style="display:none">
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>티커</th><th>등급</th><th>타입</th><th>진입가</th><th>종료가</th>
          <th>PnL</th><th>결과</th><th>진입시각</th><th>종료시각</th>
        </tr></thead>
        <tbody id="histBody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="modal-overlay" id="entryModal">
  <div class="modal">
    <div class="modal-title">📈 수동 진입 확인</div>
    <div class="modal-row"><span>티커</span><span id="m-ticker" style="font-weight:700"></span></div>
    <div class="modal-row"><span>등급</span><span id="m-grade"></span></div>
    <div class="modal-row"><span>현재가</span><span id="m-price" style="color:var(--blue)"></span></div>
    <div class="modal-row"><span>목표가 (TP +5%)</span><span id="m-tp" style="color:var(--green)"></span></div>
    <div class="modal-row"><span>손절가 (SL -3%)</span><span id="m-sl" style="color:var(--red)"></span></div>
    <div class="modal-row"><span>진입강도</span><span id="m-es"></span></div>
    <div class="modal-btns">
      <button class="btn-confirm" onclick="confirmEntry()">✅ 진입</button>
      <button class="btn-cancel"  onclick="closeModal()">취소</button>
    </div>
  </div>
</div>

<script>
let state = {};
let currentTab = 'watch';
let pendingEntry = null;
let nextScanSec = 300;
let countdownTimer = null;

async function fetchState() {
  try {
    const r = await fetch('/api/state');
    state = await r.json();
    render();
  } catch(e) { console.error('fetchState error:', e); }
}

async function manualScan() {
  document.getElementById('idleBadge').textContent = '● scanning...';
  document.getElementById('idleBadge').style.color = 'var(--yellow)';
  try {
    await fetch('/api/scan', {method:'POST'});
    setTimeout(fetchState, 3000);
  } finally {
    setTimeout(()=>{
      document.getElementById('idleBadge').textContent = '● idle';
      document.getElementById('idleBadge').style.color = 'var(--green)';
    }, 3000);
  }
}

async function addWatch() {
  const q = document.getElementById('watchSearch').value.trim().toUpperCase();
  if (!q) return;
  const ticker = q.includes('-') ? q : 'KRW-' + q;
  const r = await fetch('/api/watch/add', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ticker})
  });
  const d = await r.json();
  alert(d.success ? '✅ ' + ticker + ' Watch 추가 완료' : '❌ ' + (d.message||'실패'));
  if (d.success) fetchState();
}

async function removeWatch(ticker) {
  if (!confirm(ticker + ' Watch 제거?')) return;
  await fetch('/api/watch/remove', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ticker})
  });
  fetchState();
}

async function closeActive(ticker) {
  if (!confirm(ticker + ' 포지션 종료?')) return;
  await fetch('/api/active/close', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ticker})
  });
  fetchState();
}

function openEntryModal(ticker) {
  const item = (state.watch_list||[]).find(w=>w.ticker===ticker);
  if (!item) return;
  const price = item.current_price || 0;
  pendingEntry = {ticker, item};
  document.getElementById('m-ticker').textContent = ticker.replace('KRW-','');
  document.getElementById('m-grade').innerHTML    = gradeHtml(item.grade);
  document.getElementById('m-price').textContent  = fmtPrice(price);
  document.getElementById('m-tp').textContent     = fmtPrice(price * 1.05);
  document.getElementById('m-sl').textContent     = fmtPrice(price * 0.97);
  document.getElementById('m-es').innerHTML       = esHtml(item.entry_level, item.h4_k);
  document.getElementById('entryModal').classList.add('show');
}

async function confirmEntry() {
  if (!pendingEntry) return;
  closeModal();
  const r = await fetch('/api/watch/activate', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ticker: pendingEntry.ticker})
  });
  const d = await r.json();
  if (d.success) { switchTab('active'); fetchState(); }
  else alert('❌ ' + (d.message||'진입 실패'));
  pendingEntry = null;
}

function closeModal() {
  document.getElementById('entryModal').classList.remove('show');
}

function render() {
  renderStats();
  renderGradeBars();
  renderWatch();
  renderActive();
  renderDeep();
  renderNew();
  renderHistory();
  updateTabCounts();
  if (state.last_scan_at) {
    document.getElementById('lastScanTime').textContent =
      '마지막 스캔: ' + new Date(state.last_scan_at).toLocaleTimeString('ko-KR');
  }
  if (state.versions) {
    document.getElementById('versionTooltip').textContent =
      '📦 dashboard  ' + (state.versions.dashboard||'v2.4.5') + '\n' +
      '🔧 scanner    ' + (state.versions.scanner||'-') + '\n' +
      '⚙️ mtf_setup  ' + (state.versions.mtf_setup||'-') + '\n\n변경이력:\n' +
      '• 상단 버전 뱃지 추가\n• GX! 완전 수정\n• 진입/제거 버튼 분리\n• scanner 루프 직접 스레드 시작';
  }
}

function renderStats() {
  const s = state;
  setText('statWatch',  s.watch_count  || 0);
  setText('statActive', s.active_count || 0);
  const deepCnt = (s.active_trades||[]).filter(t=>t.trade_type==='deep').length;
  setText('statDeep', deepCnt);
  document.getElementById('statDeep').className = 'stat-value ' + (deepCnt > 0 ? 'red' : '');

  const stats = s.stats || {};
  const wr    = stats.win_rate != null ? stats.win_rate.toFixed(1) : '0.0';
  setText('statWinrate', wr + '%');

  const pnl   = stats.avg_pnl || 0;
  const pnlEl = document.getElementById('statPnl');
  pnlEl.textContent = (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%';
  pnlEl.className   = 'stat-value ' + (pnl >= 0 ? 'green' : 'red');

  const macro = s.macro || {};
  if (macro.ma20) {
    setText('statBtcMa', fmtPrice(macro.ma20));
    const cur  = macro.current_price || 0;
    const diff = cur > 0 ? ((cur - macro.ma20) / macro.ma20 * 100).toFixed(1) : 0;
    document.getElementById('statBtcSub').innerHTML =
      '현재 ' + fmtPrice(cur) + ' | <span class="' + (diff >= 0 ? 'green' : 'red') + '">' +
      (diff >= 0 ? '+' : '') + diff + '%</span>';
  }
}

function renderGradeBars() {
  const el     = document.getElementById('gradeBars');
  const grades = ['S','A','B','C'];
  const colors = {S:'var(--purple)',A:'var(--blue)',B:'var(--green)',C:'var(--yellow)'};
  const stats  = (state.stats || {}).grade_stats || {};
  const total  = Object.values(stats).reduce((a,v) => a + (v.total||0), 0) || 1;
  el.innerHTML = grades.map(g => {
    const d   = stats[g] || {total:0, avg_pnl:0, win_rate:0};
    const pct = Math.round((d.total||0) / total * 100);
    return `<div class="grade-bar-card">
      <div class="grade-bar-label">${g}등급</div>
      <div class="grade-bar-track">
        <div class="grade-bar-fill" style="width:${pct}%;background:${colors[g]}"></div>
      </div>
      <div class="grade-bar-stats">${d.win_rate||0}% | avg ${d.avg_pnl>=0?'+':''}${(d.avg_pnl||0).toFixed(1)}% | ${d.total||0}건</div>
    </div>`;
  }).join('');
}

function updateTabCounts() {
  const wl   = state.watch_list    || [];
  const at   = state.active_trades || [];
  const deep = at.filter(t => t.trade_type === 'deep');
  const ne   = state.new_entries   || [];
  setText('tc-watch',  wl.length);
  setText('tc-active', at.length);
  setText('tc-deep',   deep.length);
  setText('tc-new',    ne.length);
}

function renderWatch() {
  const q    = (document.getElementById('watchSearch').value || '').toUpperCase();
  const list = (state.watch_list || []).filter(w =>
    !q || w.ticker.includes(q) || w.ticker.replace('KRW-','').includes(q));
  const tb = document.getElementById('watchBody');
  if (!list.length) {
    tb.innerHTML = '<tr><td colspan="15" class="empty-state">Watch 항목 없음</td></tr>';
    return;
  }
  tb.innerHTML = list.map(w => {
    const sym   = w.ticker.replace('KRW-','');
    const delta = deltaCell(w.score_history);
    const exp   = w.expire_at  ? timeLeft(w.expire_at)     : '-';
    const reg   = w.registered_at ? fmtTimeISO(w.registered_at) : '-';
    return `<tr>
      <td><span class="ticker-name">${sym}</span></td>
      <td>${gradeHtml(w.grade)}</td>
      <td>${w.score||0}</td>
      <td>${delta}</td>
      <td>${kCell(w.daily_k)}</td>
      <td>${h4kCell(w.h4_k)}</td>
      <td>${kCell(w.h1_k)}</td>
      <td>${esHtml(w.entry_level, w.h4_k)}</td>
      <td>${dirCell(w.daily_dir_info, w.h4_dir_info, w.h1_dir_info)}</td>
      <td>${sparkline(w.score_history)}</td>
      <td>${(w.vol_ratio||1).toFixed(2)}x</td>
      <td>${fmtPrice(w.current_price)}</td>
      <td>${reg}</td>
      <td>${exp}</td>
      <td>
        <button class="btn-entry"  onclick="openEntryModal('${w.ticker}')">진입</button>
        <button class="btn-remove" onclick="removeWatch('${w.ticker}')">제거</button>
      </td>
    </tr>`;
  }).join('');
}

function renderActive() {
  const list = (state.active_trades || []).filter(t => t.trade_type !== 'deep');
  const tb   = document.getElementById('activeBody');
  if (!list.length) {
    tb.innerHTML = '<tr><td colspan="10" class="empty-state">활성 트레이드 없음</td></tr>';
    return;
  }
  tb.innerHTML = list.map(t => {
    const pnl = t.pnl_pct || 0;
    return `<tr>
      <td><span class="ticker-name">${t.ticker.replace('KRW-','')}</span></td>
      <td>${gradeHtml(t.grade)}</td>
      <td><span class="type-${t.trade_type||'normal'}">${typeLabel(t.trade_type)}</span></td>
      <td>${fmtPrice(t.entry_price)}</td>
      <td>${fmtPrice(t.current_price)}</td>
      <td class="${pnl>=0?'green':'red'}">${(pnl>=0?'+':'')+pnl.toFixed(2)}%</td>
      <td class="green">${fmtPrice(t.tp_price)}</td>
      <td class="red">${fmtPrice(t.sl_price)}</td>
      <td>${t.activated_at ? fmtTimeISO(t.activated_at) : '-'}</td>
      <td><button class="btn-close" onclick="closeActive('${t.ticker}')">종료</button></td>
    </tr>`;
  }).join('');
}

function renderDeep() {
  const list = (state.active_trades || []).filter(t => t.trade_type === 'deep');
  const tb   = document.getElementById('deepBody');
  if (!list.length) {
    tb.innerHTML = '<tr><td colspan="8" class="empty-state">DEEP 진입 없음</td></tr>';
    return;
  }
  tb.innerHTML = list.map(t => `<tr>
    <td><span class="ticker-name">${t.ticker.replace('KRW-','')}</span></td>
    <td>${t.deep_score||0}</td>
    <td>${kCell(t.daily_k)}</td>
    <td>${(t.relative_strength||0).toFixed(1)}%</td>
    <td>${t.days_at_bottom||0}일</td>
    <td>${(t.vol_ratio||1).toFixed(2)}x</td>
    <td>${t.activated_at ? fmtTimeISO(t.activated_at) : '-'}</td>
    <td><button class="btn-close" onclick="closeActive('${t.ticker}')">종료</button></td>
  </tr>`).join('');
}

function renderNew() {
  const list = state.new_entries || [];
  const tb   = document.getElementById('newBody');
  if (!list.length) {
    tb.innerHTML = '<tr><td colspan="8" class="empty-state">최근 신규 감지 없음</td></tr>';
    return;
  }
  tb.innerHTML = list.map(n => `<tr>
    <td><span class="ticker-name">${n.ticker.replace('KRW-','')}</span></td>
    <td>${gradeHtml(n.grade)}</td>
    <td>${n.score||0}</td>
    <td>${kCell(n.daily_k)}</td>
    <td>${h4kCell(n.h4_k)}</td>
    <td>${esHtml(n.entry_level, n.h4_k)}</td>
    <td>${dirCell(n.daily_dir_info, n.h4_dir_info, n.h1_dir_info)}</td>
    <td>${n.registered_at ? fmtTimeISO(n.registered_at) : '-'}</td>
  </tr>`).join('');
}

function renderHistory() {
  const list = state.trade_history || [];
  const tb   = document.getElementById('histBody');
  if (!list.length) {
    tb.innerHTML = '<tr><td colspan="9" class="empty-state">히스토리 없음</td></tr>';
    return;
  }
  tb.innerHTML = [...list].filter(h=>h.type==='close').reverse().slice(0,50).map(h => {
    const pnl = h.pnl_pct || 0;
    return `<tr>
      <td><span class="ticker-name">${h.ticker.replace('KRW-','')}</span></td>
      <td>${gradeHtml(h.grade)}</td>
      <td><span class="type-${h.trade_type||'normal'}">${typeLabel(h.trade_type)}</span></td>
      <td>${fmtPrice(h.entry_price)}</td>
      <td>${fmtPrice(h.close_price)}</td>
      <td class="${pnl>=0?'green':'red'}">${(pnl>=0?'+':'')+pnl.toFixed(2)}%</td>
      <td>${pnl>=0?'✅':'❌'} ${h.close_reason||''}</td>
      <td>${h.activated_at ? fmtTimeISO(h.activated_at) : '-'}</td>
      <td>${h.closed_at   ? fmtTimeISO(h.closed_at)    : '-'}</td>
    </tr>`;
  }).join('');
}

// ── 셀 헬퍼 ──
function kCell(v) {
  if (v == null) return '<span style="color:var(--muted)">-</span>';
  const n = parseFloat(v);
  let cls = 'k-high';
  if      (n <= 2)  cls = 'k-extreme';
  else if (n <= 20) cls = 'k-low';
  else if (n <= 50) cls = 'k-mid';
  else if (n >= 80) cls = 'k-over';
  return `<span class="${cls}">${n.toFixed(1)}</span>`;
}

function h4kCell(v) {
  if (v == null) return '<span style="color:var(--muted)">-</span>';
  const n    = parseFloat(v);
  const icon = n >= 80 ? '🔥' : n >= 60 ? '⚠' : '';
  return kCell(v) + (icon ? `<span style="font-size:11px">${icon}</span>` : '');
}

function dirIcon(di) {
  if (!di) return '<span style="color:var(--muted)">-</span>';
  if (typeof di === 'string') {
    const s = di.replace('!', '');
    if (s.includes('GX') || s.includes('✨')) return '<span class="dir-gx">✨GX</span>';
    if (s.includes('↑')) return `<span class="dir-up">${s}</span>`;
    if (s.includes('↓')) return `<span class="dir-down">${s}</span>`;
    return `<span class="dir-side">${s}</span>`;
  }
  const dir = di.direction || di.dir || '';
  const gx  = di.golden_cross || di.gx || false;
  if (gx) return '<span class="dir-gx">✨GX</span>';
  const map = {'상승':'↑','반등':'↗','횡보':'→','하락':'↓',
               'up':'↑','recovery':'↗','sideways':'→','down':'↓'};
  const icon = map[dir] || dir || '-';
  const cls  = (dir==='상승'||dir==='반등'||dir==='up'||dir==='recovery') ? 'dir-up'
             : (dir==='하락'||dir==='down') ? 'dir-down' : 'dir-side';
  return `<span class="${cls}">${icon}</span>`;
}

function dirCell(d, h4, h1) {
  return dirIcon(d) + dirIcon(h4) + dirIcon(h1);
}

function esHtml(lvl, h4k) {
  const level    = parseInt(lvl) || 0;
  const labels   = ['⏳ 대기','👀 관찰','🎯 준비','🚀 강신호'];
  const clss     = ['es-wait','es-watch','es-target','es-strong'];
  const idx      = Math.max(0, Math.min(3, level));
  const overheat = (parseFloat(h4k)||0) >= 80
    ? '<br><span class="es-overheat">⚠ 4hK과열</span>' : '';
  return `<span class="${clss[idx]}">${labels[idx]}</span>${overheat}`;
}

function gradeHtml(g) {
  const c = {S:'grade-s',A:'grade-a',B:'grade-b',C:'grade-c'};
  return `<span class="${c[g]||'grade-c'}">${g||'C'}</span>`;
}

function deltaCell(hist) {
  if (!hist || hist.length < 2) return '<span style="color:var(--muted)">-</span>';
  const d = hist[hist.length-1] - hist[hist.length-2];
  if (d > 0) return `<span class="green">▲${d}</span>`;
  if (d < 0) return `<span class="red">▼${Math.abs(d)}</span>`;
  return '<span style="color:var(--muted)">–</span>';
}

function sparkline(data, w=60, h=20) {
  if (!data || data.length === 0) return '<span style="color:var(--muted);font-size:10px">-</span>';
  if (data.length === 1)          return '<span style="color:var(--muted);font-size:14px">○</span>';
  const mn  = Math.min(...data), mx = Math.max(...data);
  const rng = mx - mn || 1;
  const pts = data.map((v,i) => {
    const x = (i / (data.length-1)) * w;
    const y = h - ((v - mn) / rng) * (h - 2) - 1;
    return `${x},${y}`;
  }).join(' ');
  return `<svg width="${w}" height="${h}"><polyline points="${pts}" fill="none" stroke="var(--blue)" stroke-width="1.5"/></svg>`;
}

function typeLabel(t) {
  return {normal:'자동', manual:'👆 수동', deep:'🔴 DEEP'}[t] || t || '자동';
}

function fmtPrice(v) {
  if (!v) return '-';
  return Number(v).toLocaleString('ko-KR');
}

function fmtTimeISO(iso) {
  if (!iso) return '-';
  const d = new Date(iso);
  return (d.getMonth()+1).toString().padStart(2,'0') + '/' +
          d.getDate().toString().padStart(2,'0') + ' ' +
          d.getHours().toString().padStart(2,'0') + ':' +
          d.getMinutes().toString().padStart(2,'0');
}

function timeLeft(iso) {
  if (!iso) return '-';
  const diff = new Date(iso) - Date.now();
  if (diff <= 0) return '<span class="red">만료</span>';
  const h = Math.floor(diff / 3600000);
  const m = Math.floor((diff % 3600000) / 60000);
  return h > 0 ? `${h}h${m}m` : `${m}분`;
}

function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}

function switchTab(tab) {
  currentTab = tab;
  ['watch','active','deep','new','history'].forEach(t => {
    document.getElementById('tab-' + t).classList.toggle('active', t === tab);
    const p = document.getElementById('pane-' + t);
    if (p) p.style.display = t === tab ? '' : 'none';
  });
  if (tab === 'new') startCountdown();
  else stopCountdown();
}

function startCountdown() {
  stopCountdown();
  nextScanSec = 300;
  countdownTimer = setInterval(() => {
    nextScanSec--;
    if (nextScanSec < 0) nextScanSec = 300;
    const m = Math.floor(nextScanSec / 60), s = nextScanSec % 60;
    setText('nextScanCountdown', `다음 스캔: ${m}:${s.toString().padStart(2,'0')}`);
  }, 1000);
}
function stopCountdown() {
  if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null; }
}

fetchState();
setInterval(fetchState, 15000);
</script>
</body>
</html>
"""

# ── Flask Routes ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/state')
def api_state():
    with scanner._state_lock:
        s = dict(scanner.scanner_state)
    # 히스토리는 파일에서 직접 읽기
    s['trade_history'] = scanner.load_trade_history()
    s['versions'] = {
        'dashboard': DASHBOARD_VERSION,
        'scanner':   getattr(scanner,   'VERSION', '-'),
        'mtf_setup': getattr(mtf_setup, 'VERSION', '-'),
    }
    return jsonify(s)

@app.route('/api/version')
def api_version():
    return jsonify({
        'dashboard': DASHBOARD_VERSION,
        'scanner':   getattr(scanner,   'VERSION', '-'),
        'mtf_setup': getattr(mtf_setup, 'VERSION', '-'),
    })

@app.route('/api/scan', methods=['POST'])
def api_scan():
    # scanner.manual_scan() → _scan_trigger.set() 호출
    scanner.manual_scan()
    return jsonify({'success': True})

@app.route('/api/watch/add', methods=['POST'])
def api_watch_add():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').upper()
    if not ticker:
        return jsonify({'success': False, 'message': '티커 없음'})
    return jsonify(scanner.add_manual_watch(ticker))

@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').upper()
    return jsonify(scanner.remove_watch(ticker))

@app.route('/api/watch/activate', methods=['POST'])
def api_watch_activate():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').upper()
    return jsonify(scanner.manual_activate_watch(ticker))

@app.route('/api/active/close', methods=['POST'])
def api_active_close():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').upper()
    return jsonify(scanner.manual_close_trade(ticker))

@app.route('/api/reset/watchlist', methods=['POST'])
def api_reset_watchlist():
    try:
        if os.path.exists(scanner.WATCH_LIST_FILE):
            os.remove(scanner.WATCH_LIST_FILE)
        with scanner._state_lock:
            scanner.scanner_state['watch_list']  = []
            scanner.scanner_state['watch_count'] = 0
            scanner.scanner_state['new_entries'] = []
        return jsonify({'success': True, 'message': 'watch_list 초기화 완료'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/config')
def api_config():
    cfg = getattr(mtf_setup, 'get_module_config', lambda: {})()
    return jsonify(cfg)

# ── 실행 진입점 ───────────────────────────────────────────────
if __name__ == '__main__':
    threads = [
        threading.Thread(target=scanner.scanner_loop,        daemon=True, name='scanner_loop'),
        threading.Thread(target=scanner.watch_rescan_loop,   daemon=True, name='watch_rescan'),
        threading.Thread(target=scanner.price_check_loop,    daemon=True, name='price_check'),
        threading.Thread(target=scanner.active_monitor_loop, daemon=True, name='active_monitor'),
        threading.Thread(target=scanner.daily_summary_loop,  daemon=True, name='daily_summary'),
    ]
    for t in threads:
        t.start()
    print(f'✅ Dashboard {DASHBOARD_VERSION} + Scanner {scanner.VERSION} 시작')

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
