"""
dashboard.py  v3.0.3
─────────────────────────────────────────────
변경사항:
  v3.0.1  BTC KRW/USD 표기, 일봉/주봉 MA20, 이벤트 패널, 상태바
  v3.0.2  C등급 차단, 등록가/현재가+변동률 컬럼
  v3.0.3  ⚠️ 타이밍주의 / 🔴 과매수 배지
          B등급 강화 기준 반영 (aligned 표시)
          등급별 색상 개선
─────────────────────────────────────────────
"""

DASHBOARD_VERSION = 'v3.0.3'

import threading
from flask import Flask, jsonify, request, render_template_string
import scanner as sc

app = Flask(__name__)

# ── HTML 템플릿 ───────────────────────────────────────
TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MTF Scanner {{ dashboard_version }}</title>
<style>
  :root {
    --bg:      #0d1117;
    --card:    #161b22;
    --border:  #30363d;
    --text:    #e6edf3;
    --sub:     #8b949e;
    --green:   #3fb950;
    --red:     #f85149;
    --yellow:  #d29922;
    --blue:    #58a6ff;
    --purple:  #bc8cff;
    --orange:  #ffa657;
    --s-color: #ffd700;
    --a-color: #58a6ff;
    --b-color: #3fb950;
    --x-color: #f85149;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', sans-serif; font-size: 13px; }

  /* 헤더 */
  .header {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 8px;
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .logo { font-size: 18px; font-weight: 700; color: var(--blue); }
  .version-badge {
    background: #1f2937; border: 1px solid var(--border);
    border-radius: 12px; padding: 3px 10px;
    font-size: 11px; color: var(--sub); cursor: pointer;
    position: relative;
  }
  .version-badge:hover .version-tooltip { display: block; }
  .version-tooltip {
    display: none; position: absolute; top: 28px; left: 0;
    background: #1f2937; border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px;
    min-width: 280px; z-index: 100; font-size: 11px;
    line-height: 1.8; color: var(--text); white-space: nowrap;
  }

  /* BTC 블록 */
  .btc-block {
    display: flex; flex-direction: column; gap: 4px; align-items: flex-end;
  }
  .btc-price-row {
    display: flex; align-items: center; gap: 8px;
  }
  .btc-price { font-size: 16px; font-weight: 700; color: var(--orange); }
  .btc-usd   { font-size: 12px; color: var(--sub); }
  .btc-ma-row {
    display: flex; gap: 16px; font-size: 11px;
  }
  .btc-ma-item { display: flex; align-items: center; gap: 4px; }
  .ma-label  { color: var(--sub); }
  .ma-value  { color: var(--text); }
  .ma-signal-above { color: var(--green); font-weight: 600; }
  .ma-signal-below { color: var(--red);   font-weight: 600; }
  .ma-pct    { font-size: 10px; }

  /* 상태바 */
  .status-bar {
    background: #0d1117;
    border-bottom: 1px solid var(--border);
    padding: 6px 20px;
    display: flex; gap: 16px; align-items: center;
    flex-wrap: wrap;
  }
  .status-item {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: var(--sub);
  }
  .status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--sub);
  }
  .status-dot.active   { background: var(--green); animation: pulse 1.2s infinite; }
  .status-dot.warning  { background: var(--yellow); animation: pulse 1.2s infinite; }
  .status-dot.idle     { background: var(--sub); }
  @keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.3; }
  }
  .status-text.running { color: var(--green); font-weight: 600; }
  .status-text.idle    { color: var(--sub); }
  .last-scan-info { margin-left: auto; font-size: 11px; color: var(--sub); }

  /* 메인 */
  .main { padding: 16px 20px; }

  /* 통계 카드 */
  .stats-row {
    display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap;
  }
  .stat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 16px;
    min-width: 110px; flex: 1;
  }
  .stat-label { font-size: 11px; color: var(--sub); margin-bottom: 4px; }
  .stat-value { font-size: 22px; font-weight: 700; color: var(--text); }
  .stat-sub   { font-size: 10px; color: var(--sub); margin-top: 2px; }

  /* 탭 */
  .tabs {
    display: flex; gap: 4px; margin-bottom: 12px;
    border-bottom: 1px solid var(--border); padding-bottom: 0;
  }
  .tab-btn {
    background: none; border: none; border-bottom: 2px solid transparent;
    color: var(--sub); padding: 8px 16px; cursor: pointer;
    font-size: 13px; font-weight: 500; transition: all 0.2s;
  }
  .tab-btn:hover  { color: var(--text); }
  .tab-btn.active { color: var(--blue); border-bottom-color: var(--blue); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* 테이블 */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    background: var(--card); color: var(--sub);
    padding: 8px 10px; text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap; font-weight: 500;
  }
  td {
    padding: 7px 10px; border-bottom: 1px solid #21262d;
    vertical-align: middle; white-space: nowrap;
  }
  tr:hover td { background: #1c2128; }

  /* 배지 */
  .badge {
    display: inline-block; padding: 2px 7px;
    border-radius: 10px; font-size: 11px; font-weight: 600;
  }
  .badge-S { background: #3d2e00; color: var(--s-color); border: 1px solid var(--s-color); }
  .badge-A { background: #0d2137; color: var(--a-color); border: 1px solid var(--a-color); }
  .badge-B { background: #0d2e15; color: var(--b-color); border: 1px solid var(--b-color); }
  .badge-X { background: #2d1010; color: var(--x-color); border: 1px solid var(--x-color); }

  .badge-gc   { background: #0d2e15; color: var(--green); font-size: 10px; padding: 1px 5px; }
  .badge-warn { background: #2e2200; color: var(--yellow); font-size: 10px; padding: 1px 5px; }
  .badge-over { background: #2d1010; color: var(--red);    font-size: 10px; padding: 1px 5px; }
  .badge-vol  { background: #1a1f2e; color: var(--blue);   font-size: 10px; padding: 1px 5px; }

  .score-bar {
    display: inline-block; height: 5px; border-radius: 3px;
    vertical-align: middle; margin-left: 5px;
  }

  /* 가격 셀 */
  .price-cell { line-height: 1.6; }
  .price-main { font-weight: 500; }
  .price-current { font-size: 11px; }
  .price-up   { color: var(--green); }
  .price-down { color: var(--red); }
  .price-flat { color: var(--sub); }

  /* KD 셀 */
  .kd-cell { font-size: 11px; line-height: 1.5; }
  .kd-up   { color: var(--green); }
  .kd-down { color: var(--red); }
  .kd-os   { color: var(--orange); font-weight: 600; }
  .kd-ob   { color: var(--red);    font-weight: 600; }

  /* 이벤트 패널 */
  .event-panel {
    position: fixed; bottom: 20px; right: 20px;
    width: 300px; max-height: 380px;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden; z-index: 200;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  }
  .event-header {
    padding: 10px 14px; background: #1c2128;
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; font-weight: 600;
  }
  .event-toggle {
    background: none; border: none; color: var(--sub);
    cursor: pointer; font-size: 14px; padding: 0 4px;
  }
  .event-list {
    overflow-y: auto; max-height: 320px; padding: 8px 0;
  }
  .event-item {
    padding: 6px 14px; display: flex; gap: 8px;
    border-bottom: 1px solid #21262d; font-size: 11px;
  }
  .event-time { color: var(--sub); min-width: 48px; flex-shrink: 0; }
  .event-msg  { color: var(--text); line-height: 1.4; }
  .event-item.watch  { border-left: 3px solid var(--blue); }
  .event-item.active { border-left: 3px solid var(--green); }
  .event-item.close  { border-left: 3px solid var(--red); }
  .event-item.deep   { border-left: 3px solid var(--orange); }
  .event-item.system { border-left: 3px solid var(--sub); }
  .event-item.error  { border-left: 3px solid var(--red); }

  /* 토스트 */
  .toast-container {
    position: fixed; top: 60px; right: 20px; z-index: 300;
    display: flex; flex-direction: column; gap: 8px;
  }
  .toast {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 16px;
    font-size: 12px; color: var(--text);
    animation: slideIn 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  @keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to   { transform: translateX(0);    opacity: 1; }
  }

  /* 버튼 */
  .btn {
    border: none; border-radius: 6px; padding: 5px 12px;
    font-size: 11px; cursor: pointer; font-weight: 500;
  }
  .btn-blue   { background: #1f4b8e; color: var(--blue); }
  .btn-green  { background: #0d2e15; color: var(--green); }
  .btn-red    { background: #2d1010; color: var(--red); }
  .btn-yellow { background: #2e2200; color: var(--yellow); }
  .btn:hover  { opacity: 0.8; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .scan-btn {
    background: var(--blue); color: #fff;
    border: none; border-radius: 8px;
    padding: 7px 16px; font-size: 12px;
    cursor: pointer; font-weight: 600;
  }
  .scan-btn:hover    { background: #4493f8; }
  .scan-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .empty-msg {
    text-align: center; color: var(--sub);
    padding: 40px; font-size: 13px;
  }

  /* 반응형 */
  @media (max-width: 768px) {
    .event-panel { width: 260px; }
    .btc-ma-row  { flex-direction: column; gap: 4px; }
  }
</style>
</head>
<body>

<!-- 헤더 -->
<div class="header">
  <div class="header-left">
    <span class="logo">📊 MTF Scanner</span>
    <div class="version-badge">
      {{ dashboard_version }}
      <div class="version-tooltip">
        <b>📋 변경사항</b><br>
        v3.0.3 타이밍경고(⚠️4hK≥70 / 🔴1hK≥80)<br>
        v3.0.3 B등급 강화: score≥55 + aligned≥2<br>
        v3.0.3 A등급: score≥70 + 4h/1h GC<br>
        v3.0.3 S등급: score≥85 + 일봉GC + 4h/1hGC<br>
        v3.0.2 C등급 Watch 차단<br>
        v3.0.2 등록가/현재가+변동률 표시<br>
        v3.0.1 BTC 일봉/주봉 MA20 KRW/USD<br>
        v3.0.1 실시간 이벤트 패널<br>
        v3.0.1 스캔 상태바<br>
      </div>
    </div>
  </div>

  <!-- BTC 블록 -->
  <div class="btc-block">
    <div class="btc-price-row">
      <span class="btc-price" id="btcPrice">–</span>
      <span class="btc-usd"   id="btcUsd">–</span>
      <button class="scan-btn" onclick="triggerScan()">🔄 즉시스캔</button>
    </div>
    <div class="btc-ma-row">
      <div class="btc-ma-item">
        <span class="ma-label">일봉MA20</span>
        <span class="ma-value" id="btcDailyMa">–</span>
        <span id="btcDailySignal">–</span>
        <span class="ma-pct"  id="btcDailyPct"></span>
      </div>
      <div class="btc-ma-item">
        <span class="ma-label">주봉MA20</span>
        <span class="ma-value" id="btcWeeklyMa">–</span>
        <span id="btcWeeklySignal">–</span>
        <span class="ma-pct"   id="btcWeeklyPct"></span>
      </div>
    </div>
  </div>
</div>

<!-- 상태바 -->
<div class="status-bar">
  <div class="status-item">
    <div class="status-dot" id="dotScanner"></div>
    <span class="status-text" id="txtScanner">대기중</span>
  </div>
  <div class="status-item">
    <div class="status-dot" id="dotRescan"></div>
    <span class="status-text" id="txtRescan">Watch재스캔 대기</span>
  </div>
  <div class="status-item">
    <div class="status-dot" id="dotPrice"></div>
    <span class="status-text" id="txtPrice">가격체크 대기</span>
  </div>
  <div class="status-item">
    <div class="status-dot" id="dotDeep"></div>
    <span class="status-text" id="txtDeep">DEEP 대기</span>
  </div>
  <div class="last-scan-info">
    마지막 스캔: <span id="lastScan">–</span> |
    다음 스캔: <span id="nextScan">–</span> |
    <span id="countdown"></span>
  </div>
</div>

<!-- 메인 -->
<div class="main">

  <!-- 통계 카드 -->
  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-label">📋 Watch</div>
      <div class="stat-value" id="statWatch">–</div>
      <div class="stat-sub">감시 종목</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">✅ Active</div>
      <div class="stat-value" id="statActive">–</div>
      <div class="stat-sub">진입 종목</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">🔥 DEEP</div>
      <div class="stat-value" id="statDeep">–</div>
      <div class="stat-sub">상대강도</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">📡 스캔</div>
      <div class="stat-value" id="statScan">–</div>
      <div class="stat-sub" id="statScanSub">총 스캔 횟수</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">🏆 승률</div>
      <div class="stat-value" id="statWin">–</div>
      <div class="stat-sub" id="statWinSub">–</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">💰 누적PnL</div>
      <div class="stat-value" id="statPnl">–</div>
      <div class="stat-sub">종료 트레이드</div>
    </div>
  </div>

  <!-- 탭 -->
  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('watch')">📋 Watch</button>
    <button class="tab-btn"        onclick="switchTab('active')">✅ Active</button>
    <button class="tab-btn"        onclick="switchTab('deep')">🔥 DEEP</button>
    <button class="tab-btn"        onclick="switchTab('history')">📊 History</button>
  </div>

  <!-- Watch 탭 -->
  <div id="tab-watch" class="tab-content active">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th>
            <th>등급</th>
            <th>점수</th>
            <th>등록가 / 현재가</th>
            <th>일봉 장기 K/D</th>
            <th>일봉 중기 K/D</th>
            <th>일봉 단기 K/D</th>
            <th>4h K/D</th>
            <th>1h K/D</th>
            <th>GC</th>
            <th>거래량</th>
            <th>바닥일수</th>
            <th>등록일</th>
            <th>만료</th>
            <th>관리</th>
          </tr>
        </thead>
        <tbody id="watchTbody">
          <tr><td colspan="15" class="empty-msg">스캔 대기 중...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Active 탭 -->
  <div id="tab-active" class="tab-content">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th>
            <th>등급</th>
            <th>점수</th>
            <th>진입가</th>
            <th>현재가</th>
            <th>PnL</th>
            <th>TP</th>
            <th>SL</th>
            <th>거래량</th>
            <th>진입일</th>
            <th>관리</th>
          </tr>
        </thead>
        <tbody id="activeTbody">
          <tr><td colspan="11" class="empty-msg">진입 종목 없음</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- DEEP 탭 -->
  <div id="tab-deep" class="tab-content">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th>
            <th>상대강도</th>
            <th>등급</th>
            <th>현재가</th>
            <th>스캔시간</th>
          </tr>
        </thead>
        <tbody id="deepTbody">
          <tr><td colspan="5" class="empty-msg">BTC 급락 시 자동 스캔</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- History 탭 -->
  <div id="tab-history" class="tab-content">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th>
            <th>등급</th>
            <th>진입가</th>
            <th>종료가</th>
            <th>PnL</th>
            <th>종료사유</th>
            <th>진입일</th>
            <th>종료일</th>
          </tr>
        </thead>
        <tbody id="historyTbody">
          <tr><td colspan="8" class="empty-msg">종료된 트레이드 없음</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</div><!-- /main -->

<!-- 이벤트 패널 -->
<div class="event-panel" id="eventPanel">
  <div class="event-header">
    <span>📡 실시간 이벤트</span>
    <button class="event-toggle" onclick="toggleEventPanel()">▼</button>
  </div>
  <div class="event-list" id="eventList">
    <div class="empty-msg" style="padding:20px">이벤트 없음</div>
  </div>
</div>

<!-- 토스트 -->
<div class="toast-container" id="toastContainer"></div>

<script>
// ── 전역 ───────────────────────────────────────────────
let _eventPanelOpen = true;
let _countdown      = 0;
let _countdownTimer = null;

// ── 탭 전환 ───────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

// ── 이벤트 패널 토글 ──────────────────────────────────
function toggleEventPanel() {
  const list = document.getElementById('eventList');
  const btn  = document.querySelector('.event-toggle');
  _eventPanelOpen = !_eventPanelOpen;
  list.style.display = _eventPanelOpen ? 'block' : 'none';
  btn.textContent    = _eventPanelOpen ? '▼' : '▲';
}

// ── 토스트 ─────────────────────────────────────────────
function showToast(msg, duration = 3000) {
  const c = document.getElementById('toastContainer');
  const t = document.createElement('div');
  t.className   = 'toast';
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

// ── 헬퍼: 숫자 포맷 ───────────────────────────────────
function fmtPrice(p) {
  if (p == null) return '–';
  if (p >= 1000) return Number(p).toLocaleString('ko-KR', {maximumFractionDigits: 0});
  if (p >= 1)    return Number(p).toFixed(2);
  if (p >= 0.01) return Number(p).toFixed(4);
  return Number(p).toFixed(6);
}
function fmtUsd(krw, rate) {
  if (!krw || !rate) return '';
  return '$' + (krw / rate).toLocaleString('en-US', {maximumFractionDigits: 0});
}
function fmtPct(v) {
  if (v == null) return '–';
  const sign = v >= 0 ? '+' : '';
  return sign + Number(v).toFixed(2) + '%';
}
function fmtDate(s) {
  if (!s) return '–';
  return s.replace('T', ' ').substring(0, 16);
}

// ── 등급 배지 ─────────────────────────────────────────
function gradeBadge(g) {
  if (!g || g === '-') return '<span style="color:var(--sub)">–</span>';
  const cls = { S: 'badge-S', A: 'badge-A', B: 'badge-B', X: 'badge-X' };
  return `<span class="badge ${cls[g] || 'badge-B'}">${g}</span>`;
}

// ── 점수 바 ───────────────────────────────────────────
function scorebar(score, grade) {
  const colors = { S: 'var(--s-color)', A: 'var(--a-color)', B: 'var(--b-color)', X: 'var(--x-color)' };
  const color  = colors[grade] || 'var(--sub)';
  const w      = Math.min(score, 100);
  return `<span style="color:${color};font-weight:600">${score}</span>
          <span class="score-bar" style="width:${w * 0.5}px;background:${color}"></span>`;
}

// ── KD 셀 ────────────────────────────────────────────
function kdCell(k, d, signal) {
  const kClass = k <= 20 ? 'kd-os' : (k >= 80 ? 'kd-ob' : '');
  const dir    = signal === 'BUY_OK' ? '↑' : (signal === 'BUY_NO' ? '↓' : '');
  const dirCls = signal === 'BUY_OK' ? 'kd-up' : (signal === 'BUY_NO' ? 'kd-down' : '');
  return `<div class="kd-cell">
    <span class="${kClass}">${k != null ? k.toFixed(1) : '–'}</span>
    <span class="kd-cell" style="color:var(--sub)">/${d != null ? d.toFixed(1) : '–'}</span>
    <span class="${dirCls}">${dir}</span>
  </div>`;
}

// ── GC 배지 ───────────────────────────────────────────
function gcBadge(item) {
  const parts = [];
  if (item.daily_gc) parts.push('<span class="badge badge-gc">일봉✨</span>');
  if (item.h4_gc)    parts.push('<span class="badge badge-gc">4h✨</span>');
  if (item.h1_gc)    parts.push('<span class="badge badge-gc">1h✨</span>');
  return parts.join(' ') || '<span style="color:var(--sub)">–</span>';
}

// ── 타이밍 경고 배지 ──────────────────────────────────
function warningBadge(item) {
  const parts = [];
  if (item.overbought_warning) parts.push('<span class="badge badge-over">🔴과매수</span>');
  else if (item.timing_warning) parts.push('<span class="badge badge-warn">⚠️타이밍</span>');
  return parts.join('');
}

// ── 거래량 배지 ───────────────────────────────────────
function volBadge(v) {
  if (v == null) return '–';
  const icon = v >= 2 ? '🔥' : (v >= 1 ? '📈' : '');
  return `<span class="badge badge-vol">${icon}${v.toFixed(1)}x</span>`;
}

// ── 등록가/현재가 셀 ──────────────────────────────────
function priceCell(entryPrice, currentPrice, pricePct) {
  if (!currentPrice) currentPrice = entryPrice;
  const pct    = pricePct != null ? pricePct : 0;
  const pClass = pct > 0 ? 'price-up' : (pct < 0 ? 'price-down' : 'price-flat');
  const sign   = pct >= 0 ? '+' : '';
  return `<div class="price-cell">
    <div class="price-main">${fmtPrice(entryPrice)}</div>
    <div class="price-current ${pClass}">${fmtPrice(currentPrice)} (${sign}${pct.toFixed(2)}%)</div>
  </div>`;
}

// ── BTC 업데이트 ──────────────────────────────────────
function updateBtc(s) {
  const rate = s.usdt_rate || 1350;
  // 가격
  const priceEl = document.getElementById('btcPrice');
  const usdEl   = document.getElementById('btcUsd');
  if (s.btc_price) {
    priceEl.textContent = '₩' + Number(s.btc_price).toLocaleString('ko-KR');
    usdEl.textContent   = '(' + fmtUsd(s.btc_price, rate) + ')';
  }
  // 일봉 MA20
  const dMa  = document.getElementById('btcDailyMa');
  const dSig = document.getElementById('btcDailySignal');
  const dPct = document.getElementById('btcDailyPct');
  if (s.btc_daily_ma20) {
    dMa.textContent = '₩' + Number(s.btc_daily_ma20).toLocaleString('ko-KR') +
                      ' (' + fmtUsd(s.btc_daily_ma20, rate) + ')';
    const isAbove = s.btc_daily_signal === 'ABOVE';
    dSig.className   = isAbove ? 'ma-signal-above' : 'ma-signal-below';
    dSig.textContent = isAbove ? '▲' : '▼';
    if (s.btc_price && s.btc_daily_ma20) {
      const pct = ((s.btc_price - s.btc_daily_ma20) / s.btc_daily_ma20 * 100).toFixed(2);
      dPct.textContent = (pct >= 0 ? '+' : '') + pct + '%';
      dPct.style.color = pct >= 0 ? 'var(--green)' : 'var(--red)';
    }
  }
  // 주봉 MA20
  const wMa  = document.getElementById('btcWeeklyMa');
  const wSig = document.getElementById('btcWeeklySignal');
  const wPct = document.getElementById('btcWeeklyPct');
  if (s.btc_weekly_ma20) {
    wMa.textContent = '₩' + Number(s.btc_weekly_ma20).toLocaleString('ko-KR') +
                      ' (' + fmtUsd(s.btc_weekly_ma20, rate) + ')';
    const isAbove = s.btc_weekly_signal === 'ABOVE';
    wSig.className   = isAbove ? 'ma-signal-above' : 'ma-signal-below';
    wSig.textContent = isAbove ? '▲' : '▼';
    if (s.btc_price && s.btc_weekly_ma20) {
      const pct = ((s.btc_price - s.btc_weekly_ma20) / s.btc_weekly_ma20 * 100).toFixed(2);
      wPct.textContent = (pct >= 0 ? '+' : '') + pct + '%';
      wPct.style.color = pct >= 0 ? 'var(--green)' : 'var(--red)';
    }
  }
}

// ── 상태바 업데이트 ───────────────────────────────────
function updateStatusBar(s) {
  function setStatus(dotId, txtId, active, label) {
    const dot = document.getElementById(dotId);
    const txt = document.getElementById(txtId);
    dot.className  = 'status-dot ' + (active ? 'active' : 'idle');
    txt.className  = 'status-text ' + (active ? 'running' : 'idle');
    txt.textContent = label;
  }
  setStatus('dotScanner', 'txtScanner', s.running,
    s.running ? '⏳ 전체 스캐닝 중...' : '✅ 스캐너 대기');
  setStatus('dotRescan', 'txtRescan', s.watch_rescanning,
    s.watch_rescanning ? '⏳ Watch 재스캔 중...' : 'Watch 재스캔 대기');
  setStatus('dotPrice', 'txtPrice', s.price_checking,
    s.price_checking ? '⏳ 가격 체크 중...' : '가격 체크 대기');
  setStatus('dotDeep', 'txtDeep', s.deep_scanning,
    s.deep_scanning ? '🔥 DEEP 스캔 중...' : 'DEEP 대기');

  document.getElementById('lastScan').textContent = s.last_scan || '–';
  document.getElementById('nextScan').textContent = s.next_scan || '–';

  // 카운트다운
  if (s.next_scan) {
    const next = new Date(s.next_scan.replace(' ', 'T'));
    const diff = Math.max(0, Math.floor((next - new Date()) / 1000));
    startCountdown(diff);
  }
}

// ── 카운트다운 ────────────────────────────────────────
function startCountdown(sec) {
  if (_countdownTimer) clearInterval(_countdownTimer);
  _countdown = sec;
  function tick() {
    const el = document.getElementById('countdown');
    if (!el) return;
    if (_countdown <= 0) {
      el.textContent = '⏳ 스캔 예정';
      return;
    }
    const m = Math.floor(_countdown / 60);
    const s = _countdown % 60;
    el.textContent = `다음스캔 ${m}분 ${String(s).padStart(2,'0')}초`;
    _countdown--;
  }
  tick();
  _countdownTimer = setInterval(tick, 1000);
}

// ── 통계 카드 ─────────────────────────────────────────
function updateStats(data) {
  const s = data.state || {};
  document.getElementById('statWatch').textContent  = data.watch?.length ?? 0;
  document.getElementById('statActive').textContent = data.active?.length ?? 0;
  document.getElementById('statDeep').textContent   = data.deep?.length ?? 0;
  document.getElementById('statScan').textContent   = s.scan_count ?? 0;

  const sub = document.getElementById('statScanSub');
  sub.textContent = s.running ? '⏳ 스캔 중...' : '총 스캔 횟수';
  sub.style.color = s.running ? 'var(--yellow)' : 'var(--sub)';

  const total = s.total_trades || 0;
  const wins  = s.win_trades   || 0;
  const wr    = total > 0 ? Math.round(wins / total * 100) : 0;
  document.getElementById('statWin').textContent    = total > 0 ? wr + '%' : '–';
  document.getElementById('statWinSub').textContent = `${wins}/${total}건`;

  const pnl    = s.total_pnl || 0;
  const pnlEl  = document.getElementById('statPnl');
  pnlEl.textContent = total > 0 ? (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%' : '–';
  pnlEl.style.color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
}

// ── Watch 테이블 ──────────────────────────────────────
function renderWatch(watch) {
  const tbody = document.getElementById('watchTbody');
  if (!watch || watch.length === 0) {
    tbody.innerHTML = '<tr><td colspan="15" class="empty-msg">Watch 종목 없음 (B등급 이상만 표시)</td></tr>';
    return;
  }
  // 점수 내림차순 정렬
  const sorted = [...watch].sort((a, b) => (b.score || 0) - (a.score || 0));
  tbody.innerHTML = sorted.map(item => {
    const symbol = item.market.replace('KRW-', '');
    const warn   = warningBadge(item);
    const dLongSig = item.d_long_k <= 20 ? 'BUY_OK' : 'NEUTRAL';
    return `<tr>
      <td><b>${symbol}</b>${warn ? '<br>' + warn : ''}</td>
      <td>${gradeBadge(item.grade)}</td>
      <td>${scorebar(item.score || 0, item.grade)}</td>
      <td>${priceCell(item.entry_price, item.current_price, item.price_change)}</td>
      <td>${kdCell(item.d_long_k, item.d_long_d, dLongSig)}</td>
      <td>${kdCell(item.d_mid_k,  item.d_mid_d,  'NEUTRAL')}</td>
      <td>${kdCell(item.d_short_k,item.d_short_d,'NEUTRAL')}</td>
      <td>${kdCell(item.h4_k_val, item.h4_d_val, item.h4_gc ? 'BUY_OK' : 'NEUTRAL')}</td>
      <td>${kdCell(item.h1_k_val, item.h1_d_val, item.h1_gc ? 'BUY_OK' : 'NEUTRAL')}</td>
      <td>${gcBadge(item)}</td>
      <td>${volBadge(item.volume_ratio)}</td>
      <td>${item.bottom_days ?? 0}일</td>
      <td>${(item.registered_at || '').substring(5, 16)}</td>
      <td>${item.expire_at || '–'}</td>
      <td>
        <button class="btn btn-green" onclick="activateWatch('${item.market}')">진입</button>
        <button class="btn btn-red"   onclick="removeWatch('${item.market}')">제거</button>
      </td>
    </tr>`;
  }).join('');
}

// ── Active 테이블 ─────────────────────────────────────
function renderActive(active) {
  const tbody = document.getElementById('activeTbody');
  if (!active || active.length === 0) {
    tbody.innerHTML = '<tr><td colspan="11" class="empty-msg">진입 종목 없음</td></tr>';
    return;
  }
  tbody.innerHTML = active.map(item => {
    const symbol = item.market.replace('KRW-', '');
    const cur    = item.current_price || item.entry_price;
    const pnl    = ((cur - item.entry_price) / item.entry_price * 100);
    const pClass = pnl >= 0 ? 'price-up' : 'price-down';
    return `<tr>
      <td><b>${symbol}</b></td>
      <td>${gradeBadge(item.grade)}</td>
      <td>${scorebar(item.score || 0, item.grade)}</td>
      <td>${fmtPrice(item.entry_price)}</td>
      <td>${fmtPrice(cur)}</td>
      <td class="${pClass}" style="font-weight:600">${fmtPct(pnl)}</td>
      <td style="color:var(--green)">+${item.tp_pct}% (${fmtPrice(item.tp_price)})</td>
      <td style="color:var(--red)">-${item.sl_pct}% (${fmtPrice(item.sl_price)})</td>
      <td>${volBadge(item.volume_ratio)}</td>
      <td>${(item.entered_at || '').substring(5, 16)}</td>
      <td>
        <button class="btn btn-red" onclick="closeActive('${item.market}')">종료</button>
      </td>
    </tr>`;
  }).join('');
}

// ── DEEP 테이블 ───────────────────────────────────────
function renderDeep(deep) {
  const tbody = document.getElementById('deepTbody');
  if (!deep || deep.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-msg">BTC 급락(-1% 이상) 시 자동 스캔</td></tr>';
    return;
  }
  tbody.innerHTML = deep.map(item => {
    const symbol = item.market.replace('KRW-', '');
    const rsCol  = item.rs >= 2 ? 'var(--green)' : (item.rs >= 0 ? 'var(--yellow)' : 'var(--red)');
    return `<tr>
      <td><b>${symbol}</b></td>
      <td style="color:${rsCol};font-weight:600">${item.rs >= 0 ? '+' : ''}${item.rs}%</td>
      <td>${gradeBadge(item.rs_grade)}</td>
      <td>${fmtPrice(item.price)}</td>
      <td>${(item.scanned_at || '').substring(5, 16)}</td>
    </tr>`;
  }).join('');
}

// ── History 테이블 ────────────────────────────────────
function renderHistory(history) {
  const tbody = document.getElementById('historyTbody');
  if (!history || history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-msg">종료된 트레이드 없음</td></tr>';
    return;
  }
  const sorted = [...history].reverse();
  tbody.innerHTML = sorted.map(item => {
    const symbol = (item.market || '').replace('KRW-', '');
    const pnl    = item.pnl || 0;
    const pClass = pnl >= 0 ? 'price-up' : 'price-down';
    const rColor = item.reason === 'TP' ? 'var(--green)' : (item.reason === 'SL' ? 'var(--red)' : 'var(--sub)');
    return `<tr>
      <td><b>${symbol}</b></td>
      <td>${gradeBadge(item.grade)}</td>
      <td>${fmtPrice(item.entry_price)}</td>
      <td>${fmtPrice(item.close_price)}</td>
      <td class="${pClass}" style="font-weight:600">${fmtPct(pnl)}</td>
      <td style="color:${rColor};font-weight:600">${item.reason || '–'}</td>
      <td>${(item.entered_at || '').substring(5, 16)}</td>
      <td>${(item.closed_at  || '').substring(5, 16)}</td>
    </tr>`;
  }).join('');
}

// ── 이벤트 패널 ───────────────────────────────────────
async function fetchEvents() {
  try {
    const r = await fetch('/api/events');
    const d = await r.json();
    const list = document.getElementById('eventList');
    if (!d.events || d.events.length === 0) {
      list.innerHTML = '<div class="empty-msg" style="padding:20px">이벤트 없음</div>';
      return;
    }
    list.innerHTML = [...d.events].reverse().map(e =>
      `<div class="event-item ${e.type || 'system'}">
        <span class="event-time">${e.time || ''}</span>
        <span class="event-msg">${e.message || ''}</span>
      </div>`
    ).join('');
  } catch(e) {}
}

// ── 상태 조회 ─────────────────────────────────────────
async function fetchState() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();
    updateBtc(d.state || {});
    updateStatusBar(d.state || {});
    updateStats(d);
    renderWatch(d.watch);
    renderActive(d.active);
    renderDeep(d.deep);
    renderHistory(d.history);
  } catch(e) {
    console.error('fetchState error:', e);
  }
}

// ── 즉시 스캔 ────────────────────────────────────────
async function triggerScan() {
  const btn = document.querySelector('.scan-btn');
  btn.textContent = '⏳ 스캔 중...';
  btn.disabled    = true;
  showToast('🔄 스캔 요청 중...');
  try {
    const r = await fetch('/api/scan', { method: 'POST' });
    const d = await r.json();
    if (d.success) {
      showToast('✅ 스캔 시작됨! 결과 업데이트 중...');
      let cnt = 0;
      const poll = setInterval(() => {
        fetchState();
        fetchEvents();
        cnt++;
        if (cnt >= 8) {
          clearInterval(poll);
          btn.textContent = '🔄 즉시스캔';
          btn.disabled    = false;
        }
      }, 5000);
    } else {
      showToast('❌ ' + d.message);
      btn.textContent = '🔄 즉시스캔';
      btn.disabled    = false;
    }
  } catch(e) {
    showToast('❌ 요청 실패');
    btn.textContent = '🔄 즉시스캔';
    btn.disabled    = false;
  }
}

// ── Watch 관리 ────────────────────────────────────────
async function activateWatch(market) {
  if (!confirm(market.replace('KRW-','') + ' 즉시 진입하시겠습니까?')) return;
  const r = await fetch('/api/watch/activate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ market })
  });
  const d = await r.json();
  showToast(d.success ? '✅ ' + d.message : '❌ ' + d.message);
  fetchState(); fetchEvents();
}

async function removeWatch(market) {
  if (!confirm(market.replace('KRW-','') + ' Watch에서 제거하시겠습니까?')) return;
  const r = await fetch('/api/watch/remove', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ market })
  });
  const d = await r.json();
  showToast(d.success ? '🗑️ ' + d.message : '❌ ' + d.message);
  fetchState();
}

async function closeActive(market) {
  if (!confirm(market.replace('KRW-','') + ' 포지션을 종료하시겠습니까?')) return;
  const r = await fetch('/api/active/close', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ market })
  });
  const d = await r.json();
  showToast(d.success ? '🔴 ' + d.message : '❌ ' + d.message);
  fetchState(); fetchEvents();
}

// ── 초기화 & 폴링 ─────────────────────────────────────
fetchState();
fetchEvents();
setInterval(fetchState,  15000);
setInterval(fetchEvents, 10000);
</script>
</body>
</html>
"""

# ── Flask 라우트 ──────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(TEMPLATE, dashboard_version=DASHBOARD_VERSION)

@app.route('/api/version')
def api_version():
    return jsonify({
        'dashboard': DASHBOARD_VERSION,
        'scanner':   sc.VERSION,
        'mtf_setup': sc.MTF_VERSION,
    })

@app.route('/api/state')
def api_state():
    state   = sc.get_scanner_state()
    watch   = sc._load_json(sc.WATCH_FILE,   [])
    active  = sc._load_json(sc.ACTIVE_FILE,  [])
    deep    = sc._load_json(sc.DEEP_FILE,    [])
    history = sc._load_json(sc.HISTORY_FILE, [])

    # 트레이드 통계
    total_trades = len(history)
    win_trades   = len([h for h in history if h.get('pnl', 0) > 0])
    total_pnl    = round(sum(h.get('pnl', 0) for h in history), 2)

    # next_scan 계산
    next_scan = state.get('next_scan')

    return jsonify({
        'state':   {**state,
                    'total_trades': total_trades,
                    'win_trades':   win_trades,
                    'total_pnl':    total_pnl},
        'watch':   watch,
        'active':  active,
        'deep':    deep,
        'history': history[-50:],
        'next_scan': next_scan,
    })

@app.route('/api/events')
def api_events():
    events = sc._load_json(sc.EVENT_FILE, [])
    return jsonify({'events': events})

@app.route('/api/scan', methods=['POST'])
def api_scan():
    t = threading.Thread(target=sc.run_single_scan, daemon=True)
    t.start()
    return jsonify({'success': True, 'message': '스캔 시작됨'})

@app.route('/api/watch/add', methods=['POST'])
def api_watch_add():
    data   = request.get_json() or {}
    market = data.get('market', '')
    if not market:
        return jsonify({'success': False, 'message': 'market 필요'})
    return jsonify(sc.add_watch(market))

@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data   = request.get_json() or {}
    market = data.get('market', '')
    return jsonify(sc.remove_watch(market))

@app.route('/api/watch/activate', methods=['POST'])
def api_watch_activate():
    data   = request.get_json() or {}
    market = data.get('market', '')
    return jsonify(sc.activate_watch(market))

@app.route('/api/active/close', methods=['POST'])
def api_active_close():
    data   = request.get_json() or {}
    market = data.get('market', '')
    reason = data.get('reason', 'manual')
    return jsonify(sc.close_active(market, reason))

@app.route('/api/watch/reset', methods=['POST'])
def api_watch_reset():
    return jsonify(sc.reset_watch_list())

@app.route('/api/config')
def api_config():
    return jsonify({
        'scan_interval_min':   sc.SCAN_INTERVAL_MIN,
        'rescan_interval_min': sc.RESCAN_INTERVAL_MIN,
        'price_check_sec':     sc.PRICE_CHECK_SEC,
        'tp_pct':              sc.TP_PCT,
        'sl_pct':              sc.SL_PCT,
        'watch_expire_days':   sc.WATCH_EXPIRE_DAYS,
        'max_workers':         sc.MAX_WORKERS,
        'request_delay':       sc.REQUEST_DELAY,
        'allowed_grades':      ['S', 'A', 'B'],
    })


# ── 앱 시작 ───────────────────────────────────────────
if __name__ == '__main__':
    print(f'✅ Dashboard {DASHBOARD_VERSION} + Scanner {sc.VERSION} 시작')
    print(f'   MTF Setup: {sc.MTF_VERSION}')
    print(f'   Watch 허용 등급: S / A / B (score≥55 + aligned≥2)')
    print(f'   타이밍경고: 4h K≥70 ⚠️ | 과매수경고: 1h K≥80 🔴')
    print(f'   자동진입: watch_eligible + GC + 경고없음')

    # 루프 스레드 시작
    loops = [
        ('scanner',      sc.scanner_loop),
        ('watch_rescan', sc.watch_rescan_loop),
        ('price_check',  sc.price_check_loop),
        ('active_monitor', sc.active_monitor_loop),
        ('deep_scan',    sc.deep_scan_loop),
        ('daily_summary',sc.daily_summary_loop),
    ]
    for name, fn in loops:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
    print(f'   루프 {len(loops)}개 시작: {" / ".join(n for n,_ in loops)}')
    print(f'🚀 http://0.0.0.0:{sc.PORT}')

    app.run(host='0.0.0.0', port=sc.PORT, debug=False)
