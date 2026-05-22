"""
dashboard.py v2.4.2
Upbit MTF 스캐너 대시보드

변경사항 (v2.4.2):
- dirIcon: golden_cross 필드 방어 처리 ('golden_cross' / 'gx' 둘 다 허용)
- dirIcon: direction null/undefined/빈값 안전 처리
- GX 아이콘 ✨GX 통일 (GX! 포맷 제거)
- STEEM 케이스: 일봉K≤2 횡보/하락도 최소 👀관찰 표시
- DEEP 탭: active_trades 필터 방식으로 항상 정상 렌더링
- sparkline: 데이터 1개일 때 점으로 표시 (- 대신)
- 신규(🆕) 탭: 5분 카운트다운 타이머 추가
- Watch 테이블: 점수 변화 delta 컬럼 추가
"""

import os
from flask import Flask, jsonify, render_template_string, request
import scanner
import mtf_setup

app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upbit MTF Scanner {{ version }}</title>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3e;
    --text: #e2e8f0; --muted: #64748b; --accent: #6366f1;
    --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
    --orange: #f97316; --blue: #3b82f6; --purple: #a855f7;
    --deep: #ec4899;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', sans-serif; font-size: 13px; }
  a { color: var(--accent); text-decoration: none; }

  /* 헤더 */
  .header { background: var(--card); border-bottom: 1px solid var(--border);
            padding: 12px 20px; display: flex; align-items: center; gap: 16px; }
  .header h1 { font-size: 18px; font-weight: 700; }
  .status-badge { padding: 3px 10px; border-radius: 12px; font-size: 11px;
                  font-weight: 600; background: #22c55e22; color: var(--green); }
  .status-badge.scanning { background: #f59e0b22; color: var(--yellow); }
  .status-badge.error    { background: #ef444422; color: var(--red); }
  .btn { padding: 5px 14px; border-radius: 6px; border: 1px solid var(--border);
         background: var(--card); color: var(--text); cursor: pointer; font-size: 12px; }
  .btn:hover { background: var(--accent); border-color: var(--accent); color: #fff; }
  .spacer { flex: 1; }

  /* 카드 */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
           gap: 12px; padding: 16px 20px; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 10px; padding: 14px 16px; }
  .card .label { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
  .card .value { font-size: 22px; font-weight: 700; }
  .card .sub   { font-size: 11px; color: var(--muted); margin-top: 4px; }
  .card.deep-card .value { color: var(--deep); }

  /* 등급 승률 바 */
  .grade-bars { display: flex; gap: 10px; padding: 0 20px 16px; flex-wrap: wrap; }
  .grade-bar-item { background: var(--card); border: 1px solid var(--border);
                    border-radius: 8px; padding: 10px 14px; min-width: 130px; }
  .grade-bar-item .g-label { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
  .grade-bar-item .g-bar-bg   { background: #ffffff15; border-radius: 4px; height: 6px; margin: 4px 0; }
  .grade-bar-item .g-bar-fill { height: 6px; border-radius: 4px; background: var(--accent); }
  .grade-bar-item .g-stats    { font-size: 11px; color: var(--muted); }

  /* DEEP 패널 */
  .deep-info-panel { margin: 0 20px 16px;
                     background: #ec489915; border: 1px solid #ec489950;
                     border-radius: 10px; padding: 14px 18px; }
  .deep-info-panel h3 { color: var(--deep); font-size: 14px; margin-bottom: 10px; }
  .deep-info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }
  .deep-info-item .d-label { font-size: 11px; color: var(--muted); }
  .deep-info-item .d-value { font-size: 16px; font-weight: 700; color: var(--deep); }

  /* 탭 */
  .tabs { display: flex; padding: 0 20px;
          border-bottom: 1px solid var(--border); }
  .tab  { padding: 10px 18px; cursor: pointer;
          border-bottom: 2px solid transparent;
          font-size: 13px; color: var(--muted); transition: all .15s; }
  .tab.active   { color: var(--text); border-bottom-color: var(--accent); }
  .tab:hover    { color: var(--text); }
  .tab.deep-tab { color: var(--deep); }
  .tab.deep-tab.active { border-bottom-color: var(--deep); }

  .tab-content        { display: none; padding: 16px 20px; }
  .tab-content.active { display: block; }

  /* 범례 */
  .legend { background: var(--card); border: 1px solid var(--border);
            border-radius: 8px; padding: 10px 16px; margin-bottom: 14px;
            font-size: 11px; color: var(--muted); }
  .legend .legend-title { font-weight: 600; color: var(--text); margin-bottom: 8px; }
  .legend-grid { display: grid;
                 grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                 gap: 4px 16px; }
  .legend-row      { display: flex; gap: 8px; align-items: center; }
  .legend-row .es  { min-width: 90px; }
  .legend-row .combo { color: var(--text); }

  /* 테이블 */
  .tbl-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  thead th { background: #ffffff08; padding: 8px 10px; text-align: left;
             border-bottom: 1px solid var(--border); white-space: nowrap;
             font-size: 11px; color: var(--muted); font-weight: 600; }
  tbody tr { border-bottom: 1px solid #ffffff08; transition: background .1s; }
  tbody tr:hover { background: #ffffff05; }
  tbody td { padding: 8px 10px; vertical-align: middle; white-space: nowrap; }

  /* 배지 */
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 11px; font-weight: 700; }
  .badge-S      { background: #f59e0b22; color: #f59e0b; }
  .badge-A      { background: #6366f122; color: #818cf8; }
  .badge-B      { background: #22c55e22; color: #4ade80; }
  .badge-C      { background: #ffffff10; color: var(--muted); }
  .badge-deep-s { background: #ec489930; color: var(--deep); }
  .badge-deep-a { background: #ec489920; color: #f472b6; }
  .badge-deep-b { background: #ec489910; color: #fda4af; }

  /* K값 색상 */
  .k-red    { color: #ef4444; font-weight: 700; }
  .k-orange { color: #f97316; font-weight: 600; }
  .k-yellow { color: #f59e0b; }
  .k-white  { color: var(--text); }

  /* 진입강도 */
  .es-3 { color: #22c55e; font-weight: 700; }
  .es-2 { color: #6366f1; }
  .es-1 { color: #94a3b8; }
  .es-0 { color: #374151; }

  /* PnL */
  .pnl-pos { color: var(--green); font-weight: 600; }
  .pnl-neg { color: var(--red);   font-weight: 600; }

  /* 점수 delta */
  .delta-pos { color: var(--green); font-size: 11px; }
  .delta-neg { color: var(--red);   font-size: 11px; }
  .delta-neu { color: var(--muted); font-size: 11px; }

  /* 스파크라인 */
  .sparkline-wrap svg { display: block; }

  /* 방향 */
  .dir-up   { color: var(--green); }
  .dir-mid  { color: var(--yellow); }
  .dir-down { color: var(--red); }
  .dir-gx   { color: var(--yellow); font-weight: 700; font-size: 11px; }

  /* 수동 입력 */
  .add-form { display: flex; gap: 8px; margin-bottom: 14px; }
  .add-form input { background: var(--card); border: 1px solid var(--border);
                    border-radius: 6px; padding: 7px 12px; color: var(--text);
                    font-size: 13px; width: 200px; }
  .add-form input:focus { outline: none; border-color: var(--accent); }

  /* 신규 카운트다운 */
  .new-countdown { font-size: 11px; color: var(--muted); margin-bottom: 10px; }

  /* 히스토리 */
  .hist-activate { color: var(--blue); }
  .hist-close    { color: var(--muted); }
  .hist-deep     { color: var(--deep); }

  @media (max-width: 768px) {
    .cards { grid-template-columns: repeat(2, 1fr); }
    .header h1 { font-size: 15px; }
  }
</style>
</head>
<body>

<!-- ── 헤더 ─────────────────────────────────────────────── -->
<div class="header">
  <h1>📊 Upbit MTF Scanner</h1>
  <span id="statusBadge" class="status-badge">● idle</span>
  <span class="spacer"></span>
  <span id="lastScan" style="font-size:11px;color:var(--muted)"></span>
  <button class="btn" onclick="triggerScan()">🔄 수동 스캔</button>
</div>

<!-- ── 요약 카드 ──────────────────────────────────────────── -->
<div class="cards">
  <div class="card">
    <div class="label">Watch</div>
    <div class="value" id="cardWatch">-</div>
    <div class="sub"   id="cardWatchSub"></div>
  </div>
  <div class="card">
    <div class="label">Active</div>
    <div class="value" id="cardActive">-</div>
    <div class="sub"   id="cardActiveSub"></div>
  </div>
  <div class="card deep-card">
    <div class="label">DEEP Active</div>
    <div class="value" id="cardDeep">-</div>
    <div class="sub">BTC 하락 중 버팀</div>
  </div>
  <div class="card">
    <div class="label">승률</div>
    <div class="value" id="cardWinRate">-</div>
    <div class="sub"   id="cardWinSub"></div>
  </div>
  <div class="card">
    <div class="label">평균 PnL</div>
    <div class="value" id="cardAvgPnl">-</div>
    <div class="sub"   id="cardPnlSub"></div>
  </div>
  <div class="card">
    <div class="label">BTC MA20</div>
    <div class="value" id="cardBtcMa" style="font-size:15px">-</div>
    <div class="sub"   id="cardBtcSub"></div>
  </div>
</div>

<!-- ── 등급별 승률 바 ─────────────────────────────────────── -->
<div class="grade-bars" id="gradeBars"></div>

<!-- ── DEEP 패널 (DEEP Active 있을 때만) ─────────────────── -->
<div id="deepInfoPanel" class="deep-info-panel" style="display:none">
  <h3>🔴 DEEP Watch 현황</h3>
  <div class="deep-info-grid" id="deepInfoGrid"></div>
</div>

<!-- ── 탭 ────────────────────────────────────────────────── -->
<div class="tabs">
  <div class="tab active"   onclick="switchTab('watch')">
    👁 Watch <span id="tabWatchCnt"></span>
  </div>
  <div class="tab"          onclick="switchTab('active')">
    ⚡ Active <span id="tabActiveCnt"></span>
  </div>
  <div class="tab deep-tab" onclick="switchTab('deep')">
    🔴 DEEP <span id="tabDeepCnt"></span>
  </div>
  <div class="tab"          onclick="switchTab('new')">
    🆕 신규 <span id="tabNewCnt"></span>
  </div>
  <div class="tab"          onclick="switchTab('history')">
    📜 히스토리
  </div>
</div>

<!-- ── Watch 탭 ──────────────────────────────────────────── -->
<div id="tab-watch" class="tab-content active">
  <div class="legend">
    <div class="legend-title">📖 진입강도 해석 가이드</div>
    <div class="legend-grid">
      <div class="legend-row">
        <span class="es es-3">🚀강한신호</span>
        <span class="combo">S/A등급 → 적극 진입 고려</span>
      </div>
      <div class="legend-row">
        <span class="es es-3">🚀강한신호</span>
        <span class="combo">B등급 → 차트 강반등, 다음 재스캔 주목</span>
      </div>
      <div class="legend-row">
        <span class="es es-2">🎯진입고려</span>
        <span class="combo">S/A등급 → 진입 준비 / 일봉K≤2이면 자동 보장</span>
      </div>
      <div class="legend-row">
        <span class="es es-1">👀관찰</span>
        <span class="combo">방향 형성 중 / 일봉K≤2 횡보여도 최소 보장</span>
      </div>
      <div class="legend-row">
        <span class="es es-0">⏳대기</span>
        <span class="combo">신호 없음 → 완전 대기</span>
      </div>
      <div class="legend-row">
        <span style="color:var(--orange)">⚠ 4hK 과열</span>
        <span class="combo">4hK&gt;80 → 진입강도 1단계 자동 하향</span>
      </div>
    </div>
  </div>

  <div class="add-form">
    <input id="addInput" type="text" placeholder="KRW-BTC or BTC"
           onkeydown="if(event.key==='Enter')addWatch()">
    <button class="btn" onclick="addWatch()">+ Watch 추가</button>
  </div>

  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>티커</th><th>등급</th><th>점수</th><th>Δ</th>
          <th>일봉K</th><th>4hK</th><th>1hK</th>
          <th>진입강도</th><th>방향(일/4h/1h)</th><th>추세</th>
          <th>거래량</th><th>현재가</th><th>등록</th><th>만료</th><th>관리</th>
        </tr>
      </thead>
      <tbody id="watchBody"></tbody>
    </table>
  </div>
</div>

<!-- ── Active 탭 ─────────────────────────────────────────── -->
<div id="tab-active" class="tab-content">
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>티커</th><th>등급</th><th>점수</th><th>유형</th>
          <th>진입가</th><th>현재가</th><th>PnL</th>
          <th>TP</th><th>SL</th>
          <th>진입강도</th><th>방향</th>
          <th>진입일시</th><th>타임아웃</th><th>관리</th>
        </tr>
      </thead>
      <tbody id="activeBody"></tbody>
    </table>
  </div>
</div>

<!-- ── DEEP 탭 ───────────────────────────────────────────── -->
<div id="tab-deep" class="tab-content">
  <div class="legend" style="border-color:#ec489950">
    <div class="legend-title" style="color:var(--deep)">🔴 DEEP Watch 란?</div>
    <div style="margin-top:6px;line-height:1.8">
      일봉K ≤ 5 (극단 과매도) + BTC 하락 중 + 해당 코인이 BTC보다 버티는 종목<br>
      BTC 회복 시 강한 반등 가능성이 높은 선진입 전략<br>
      <span style="color:var(--deep)">DEEP-S/A → 자동 Active 전환</span>
      &nbsp;|&nbsp;
      <span style="color:#f472b6">DEEP-B → Watch 유지</span>
    </div>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>티커</th><th>DEEP등급</th><th>DEEP점수</th>
          <th>일봉K</th><th>BTC 24h</th><th>코인 24h</th><th>상대강도</th>
          <th>바닥유지</th><th>거래량비율</th><th>주봉K</th>
          <th>현재가</th><th>PnL</th><th>진입일시</th><th>관리</th>
        </tr>
      </thead>
      <tbody id="deepBody"></tbody>
    </table>
  </div>
</div>

<!-- ── 신규 탭 ───────────────────────────────────────────── -->
<div id="tab-new" class="tab-content">
  <div class="new-countdown" id="newCountdown"></div>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>티커</th><th>등급</th><th>점수</th>
          <th>일봉K</th><th>4hK</th><th>1hK</th>
          <th>진입강도</th><th>방향</th>
          <th>현재가</th><th>등록일시</th>
        </tr>
      </thead>
      <tbody id="newBody"></tbody>
    </table>
  </div>
</div>

<!-- ── 히스토리 탭 ────────────────────────────────────────── -->
<div id="tab-history" class="tab-content">
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>유형</th><th>티커</th><th>등급</th><th>거래유형</th>
          <th>진입가</th><th>청산가</th><th>PnL</th><th>사유</th><th>일시</th>
        </tr>
      </thead>
      <tbody id="histBody"></tbody>
    </table>
  </div>
</div>

<!-- ──────────────────────────────────────────────────────── -->
<script>
// ── 탭 전환 ────────────────────────────────────────────────
const TAB_NAMES = ['watch','active','deep','new','history'];
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', TAB_NAMES[i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => {
    c.classList.toggle('active', c.id === 'tab-' + name);
  });
}

// ── 포맷 헬퍼 ──────────────────────────────────────────────
function fmtPrice(v) {
  if (v == null || v === 0) return '-';
  if (v >= 1000) return Number(v).toLocaleString('ko-KR', {maximumFractionDigits: 0});
  if (v >= 1)    return Number(v).toFixed(2);
  return Number(v).toFixed(4);
}

function fmtPct(v) {
  if (v == null) return '-';
  const cls = v >= 0 ? 'pnl-pos' : 'pnl-neg';
  return `<span class="${cls}">${v >= 0 ? '+' : ''}${v.toFixed(2)}%</span>`;
}

function fmtTime(iso) {
  if (!iso) return '-';
  try {
    const d   = new Date(iso);
    const kst = new Date(d.getTime() + 9 * 3600 * 1000);
    const mo  = String(kst.getUTCMonth() + 1).padStart(2, '0');
    const da  = String(kst.getUTCDate()).padStart(2, '0');
    const hh  = String(kst.getUTCHours()).padStart(2, '0');
    const mm  = String(kst.getUTCMinutes()).padStart(2, '0');
    return `${mo}/${da} ${hh}:${mm}`;
  } catch(e) { return '-'; }
}

function timeAgo(iso) {
  if (!iso) return '-';
  try {
    const diff = Date.now() - new Date(iso).getTime();
    const m    = Math.floor(diff / 60000);
    if (m < 60)  return m + '분전';
    const h = Math.floor(m / 60);
    if (h < 24)  return h + '시간전';
    return Math.floor(h / 24) + '일전';
  } catch(e) { return '-'; }
}

// ── K값 셀 ────────────────────────────────────────────────
function kCell(v) {
  if (v == null) return '<span class="k-white">-</span>';
  const f = parseFloat(v);
  if (isNaN(f))  return '<span class="k-white">-</span>';
  const cls = f <= 5  ? 'k-red'
            : f <= 10 ? 'k-orange'
            : f <= 20 ? 'k-yellow'
            : 'k-white';
  return `<span class="${cls}">${f.toFixed(1)}</span>`;
}

function h4KCell(v) {
  if (v == null) return '<span class="k-white">-</span>';
  const f = parseFloat(v);
  if (isNaN(f))  return '<span class="k-white">-</span>';
  const cls  = f <= 5  ? 'k-red'
             : f <= 20 ? 'k-orange'
             : 'k-white';
  const icon = f > 80 ? '🔥' : f > 50 ? '⚠' : '';
  return `<span class="${cls}">${f.toFixed(1)}${icon}</span>`;
}

// ── 방향 아이콘 (v2.4.2 핵심 수정) ──────────────────────────
function dirIcon(dirInfo) {
  // null / undefined / 비-object 방어
  if (!dirInfo || typeof dirInfo !== 'object') {
    return '<span class="dir-icon dir-mid">→</span>';
  }

  const dir = dirInfo.direction || '횡보';

  // golden_cross: 'golden_cross' 키 우선, 없으면 'gx' 키 확인
  const gx = (dirInfo.golden_cross === true) || (dirInfo.gx === true);

  if (gx) {
    return '<span class="dir-icon dir-gx">✨GX</span>';
  }

  const iconMap = { '상승': '↑', '반등': '↗', '횡보': '→', '하락': '↓' };
  const clsMap  = {
    '상승': 'dir-up', '반등': 'dir-up',
    '횡보': 'dir-mid', '하락': 'dir-down',
  };
  const icon = iconMap[dir] || '→';
  const cls  = clsMap[dir]  || 'dir-mid';
  return `<span class="dir-icon ${cls}">${icon}</span>`;
}

function dirCell(dInfo, h4Info, h1Info) {
  return dirIcon(dInfo) + dirIcon(h4Info) + dirIcon(h1Info);
}

// ── 진입강도 셀 ───────────────────────────────────────────
function esCell(level, label) {
  if (level == null) return '-';
  const lv   = parseInt(level, 10);
  const cls  = ['es-0','es-1','es-2','es-3'][lv] || 'es-0';
  const dflt = ['⏳대기','👀관찰','🎯진입고려','🚀강한신호'][lv] || '⏳대기';
  const lbl  = label || dflt;
  return `<span class="${cls}">${lbl}</span>`;
}

// ── 등급 배지 ─────────────────────────────────────────────
function gradeBadge(g) {
  if (!g) return '-';
  return `<span class="badge badge-${g}">${g}</span>`;
}

function deepGradeBadge(g) {
  if (!g) return '-';
  const sub = g.includes('S') ? 's' : g.includes('A') ? 'a' : 'b';
  return `<span class="badge badge-deep-${sub}">${g}</span>`;
}

// ── 점수 delta ────────────────────────────────────────────
function deltaCell(hist) {
  if (!hist || hist.length < 2) return '<span class="delta-neu">-</span>';
  const d = hist[hist.length - 1] - hist[hist.length - 2];
  if (d > 0)  return `<span class="delta-pos">▲${d}</span>`;
  if (d < 0)  return `<span class="delta-neg">▼${Math.abs(d)}</span>`;
  return '<span class="delta-neu">→</span>';
}

// ── 스파크라인 (1개 데이터도 점으로 표시) ─────────────────
function sparkline(data, w = 60, h = 20) {
  if (!data || data.length === 0) {
    return '<span style="color:var(--muted);font-size:10px">-</span>';
  }
  if (data.length === 1) {
    // 점 하나 표시
    return `<svg width="${w}" height="${h}">
      <circle cx="${w/2}" cy="${h/2}" r="2" fill="var(--muted)"/>
    </svg>`;
  }

  const pts   = data.slice(-20);
  const mn    = Math.min(...pts);
  const mx    = Math.max(...pts);
  const range = mx - mn || 1;
  const step  = w / (pts.length - 1);

  const points = pts.map((v, i) => {
    const x = i * step;
    const y = h - ((v - mn) / range) * (h - 2) - 1;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const last  = pts[pts.length - 1];
  const first = pts[0];
  const color = last >= first ? '#22c55e' : '#ef4444';

  return `<svg width="${w}" height="${h}">
    <polyline points="${points}" fill="none"
      stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>
  </svg>`;
}

// ── Watch 테이블 렌더 ──────────────────────────────────────
function renderWatch(list) {
  const tbody = document.getElementById('watchBody');
  if (!list || list.length === 0) {
    tbody.innerHTML =
      '<tr><td colspan="15" style="text-align:center;padding:24px;color:var(--muted)">데이터 없음</td></tr>';
    return;
  }
  tbody.innerHTML = list.map(w => {
    const ticker  = w.ticker || '-';
    const name    = ticker.replace('KRW-', '');
    const manual  = w.manual ? '👤' : '';
    const expireStr = w.expire_at ? fmtTime(w.expire_at) : '∞';
    return `<tr>
      <td>
        <a href="https://upbit.com/exchange?code=CRIX.UPBIT.${ticker}" target="_blank">
          ${manual}${name}
        </a>
      </td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${w.score ?? '-'}</td>
      <td>${deltaCell(w.score_history)}</td>
      <td>${kCell(w.daily_k)}</td>
      <td>${h4KCell(w.h4_k)}</td>
      <td>${kCell(w.h1_k)}</td>
      <td>${esCell(w.entry_level, w.entry_label)}</td>
      <td>${dirCell(w.daily_dir_info, w.h4_dir_info, w.h1_dir_info)}</td>
      <td>${sparkline(w.score_history)}</td>
      <td>${w.vol_ratio != null ? w.vol_ratio.toFixed(2) + 'x' : '-'}</td>
      <td>${fmtPrice(w.current_price)}</td>
      <td>${timeAgo(w.registered_at)}</td>
      <td>${expireStr}</td>
      <td>
        <button class="btn" style="font-size:11px;padding:2px 8px"
          onclick="removeWatch('${ticker}')">제거</button>
      </td>
    </tr>`;
  }).join('');
}

// ── Active 테이블 렌더 ────────────────────────────────────
function renderActive(list) {
  const tbody = document.getElementById('activeBody');
  const items = (list || []).filter(a => (a.trade_type || 'normal') === 'normal');
  if (items.length === 0) {
    tbody.innerHTML =
      '<tr><td colspan="14" style="text-align:center;padding:24px;color:var(--muted)">Active 없음</td></tr>';
    return;
  }
  tbody.innerHTML = items.map(a => {
    const ticker = a.ticker || '-';
    const name   = ticker.replace('KRW-', '');
    return `<tr>
      <td>
        <a href="https://upbit.com/exchange?code=CRIX.UPBIT.${ticker}" target="_blank">
          ${name}
        </a>
      </td>
      <td>${gradeBadge(a.grade)}</td>
      <td>${a.score ?? '-'}</td>
      <td><span style="color:var(--blue)">${a.trade_type || 'normal'}</span></td>
      <td>${fmtPrice(a.entry_price)}</td>
      <td>${fmtPrice(a.current_price)}</td>
      <td>${fmtPct(a.pnl_pct)}</td>
      <td style="color:var(--green)">${fmtPrice(a.tp_price)}</td>
      <td style="color:var(--red)">${fmtPrice(a.sl_price)}</td>
      <td>${esCell(a.entry_level, a.entry_label)}</td>
      <td>${dirCell(a.daily_dir_info, a.h4_dir_info, a.h1_dir_info)}</td>
      <td>${fmtTime(a.activated_at)}</td>
      <td>${fmtTime(a.timeout_at)}</td>
      <td>
        <button class="btn" style="font-size:11px;padding:2px 8px;color:var(--red)"
          onclick="closeTrade('${ticker}')">청산</button>
      </td>
    </tr>`;
  }).join('');
}

// ── DEEP 테이블 렌더 ──────────────────────────────────────
function renderDeep(activeList) {
  const tbody = document.getElementById('deepBody');
  const list  = (activeList || []).filter(a => a.trade_type === 'deep');

  if (list.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="14" style="text-align:center;padding:24px;color:var(--muted)">
          DEEP Active 없음<br>
          <span style="font-size:11px">
            일봉K≤5 + BTC 하락 + 코인 버팀 조건 충족 시 자동 등록
          </span>
        </td>
      </tr>`;
    return;
  }

  tbody.innerHTML = list.map(a => {
    const ticker = a.ticker || '-';
    const name   = ticker.replace('KRW-', '');
    const rel    = a.relative_strength;
    const relStr = rel != null
      ? `<span style="color:var(--green)">+${rel.toFixed(2)}%p</span>` : '-';
    return `<tr>
      <td>
        <a href="https://upbit.com/exchange?code=CRIX.UPBIT.${ticker}" target="_blank">
          <span style="color:var(--deep)">🔴</span>${name}
        </a>
      </td>
      <td>${deepGradeBadge(a.deep_grade)}</td>
      <td>${a.deep_score ?? '-'}</td>
      <td>${kCell(a.daily_k)}</td>
      <td><span style="color:var(--red)">
        ${a.btc_change != null ? a.btc_change.toFixed(2) + '%' : '-'}
      </span></td>
      <td>${a.coin_change != null ? a.coin_change.toFixed(2) + '%' : '-'}</td>
      <td>${relStr}</td>
      <td>${a.days_at_bottom ?? '-'}일</td>
      <td>${a.vol_ratio != null ? a.vol_ratio.toFixed(2) + 'x' : '-'}</td>
      <td>${kCell(a.weekly_k)}</td>
      <td>${fmtPrice(a.current_price)}</td>
      <td>${fmtPct(a.pnl_pct)}</td>
      <td>${fmtTime(a.activated_at)}</td>
      <td>
        <button class="btn" style="font-size:11px;padding:2px 8px;color:var(--red)"
          onclick="closeTrade('${ticker}')">청산</button>
      </td>
    </tr>`;
  }).join('');
}

// ── 신규 테이블 렌더 ─────────────────────────────────────
let _newRegisteredAt = null;

function renderNew(list) {
  const tbody = document.getElementById('newBody');
  const cntEl = document.getElementById('tabNewCnt');
  const cdEl  = document.getElementById('newCountdown');

  if (!list || list.length === 0) {
    tbody.innerHTML =
      '<tr><td colspan="10" style="text-align:center;padding:24px;color:var(--muted)">신규 항목 없음 (5분 후 자동 초기화)</td></tr>';
    cntEl.textContent   = '';
    cdEl.textContent    = '';
    _newRegisteredAt    = null;
    return;
  }

  cntEl.textContent = `(${list.length})`;

  // 가장 최근 등록 시각 기준으로 카운트다운
  if (!_newRegisteredAt && list[0]?.registered_at) {
    _newRegisteredAt = new Date(list[0].registered_at).getTime();
  }
  if (_newRegisteredAt) {
    const remain = 300000 - (Date.now() - _newRegisteredAt);
    if (remain > 0) {
      const m = Math.floor(remain / 60000);
      const s = Math.floor((remain % 60000) / 1000);
      cdEl.textContent = `⏱ ${m}분 ${s}초 후 초기화`;
    } else {
      cdEl.textContent = '초기화 대기 중...';
    }
  }

  tbody.innerHTML = list.map(w => {
    const ticker = w.ticker || '-';
    const name   = ticker.replace('KRW-', '');
    return `<tr style="background:#22c55e08">
      <td>
        <a href="https://upbit.com/exchange?code=CRIX.UPBIT.${ticker}" target="_blank">
          🆕${name}
        </a>
      </td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${w.score ?? '-'}</td>
      <td>${kCell(w.daily_k)}</td>
      <td>${h4KCell(w.h4_k)}</td>
      <td>${kCell(w.h1_k)}</td>
      <td>${esCell(w.entry_level, w.entry_label)}</td>
      <td>${dirCell(w.daily_dir_info, w.h4_dir_info, w.h1_dir_info)}</td>
      <td>${fmtPrice(w.current_price)}</td>
      <td>${fmtTime(w.registered_at)}</td>
    </tr>`;
  }).join('');
}

// ── 히스토리 테이블 렌더 ──────────────────────────────────
function renderHistory(list) {
  const tbody = document.getElementById('histBody');
  if (!list || list.length === 0) {
    tbody.innerHTML =
      '<tr><td colspan="9" style="text-align:center;padding:24px;color:var(--muted)">히스토리 없음</td></tr>';
    return;
  }
  const sorted = [...list].reverse().slice(0, 100);
  tbody.innerHTML = sorted.map(h => {
    const typeMap   = { activate: '진입', close: '청산' };
    const typeLabel = typeMap[h.type] || h.type;
    const cls = h.trade_type === 'deep' ? 'hist-deep'
              : h.type === 'activate'   ? 'hist-activate'
              : 'hist-close';
    const pnl = h.pnl_pct != null ? fmtPct(h.pnl_pct) : '-';
    const dt  = h.closed_at || h.activated_at;
    return `<tr>
      <td class="${cls}">${typeLabel}</td>
      <td>${(h.ticker || '-').replace('KRW-', '')}</td>
      <td>${gradeBadge(h.grade)}</td>
      <td>${h.trade_type || 'normal'}</td>
      <td>${fmtPrice(h.entry_price)}</td>
      <td>${fmtPrice(h.close_price)}</td>
      <td>${pnl}</td>
      <td>${h.close_reason || '-'}</td>
      <td>${fmtTime(dt)}</td>
    </tr>`;
  }).join('');
}

// ── 등급 승률 바 렌더 ────────────────────────────────────
function renderGradeBars(gradeStats) {
  const el = document.getElementById('gradeBars');
  ['S', 'A', 'B', 'C'].forEach(g => {
    const s   = (gradeStats || {})[g] || {};
    const wr  = s.win_rate ?? 0;
    const avg = s.avg_pnl  ?? 0;
    const tot = s.total    ?? 0;
    const item = document.createElement('div');
    item.className = 'grade-bar-item';
    item.innerHTML = `
      <div class="g-label">${g}등급 승률</div>
      <div class="g-bar-bg">
        <div class="g-bar-fill" style="width:${wr}%"></div>
      </div>
      <div class="g-stats">
        ${wr}%&nbsp;|&nbsp;avg ${avg >= 0 ? '+' : ''}${avg}%&nbsp;|&nbsp;${tot}건
      </div>`;
    el.appendChild(item);
  });
}

// ── DEEP 패널 렌더 ────────────────────────────────────────
function renderDeepPanel(activeList) {
  const panel = document.getElementById('deepInfoPanel');
  const grid  = document.getElementById('deepInfoGrid');
  const dList = (activeList || []).filter(a => a.trade_type === 'deep');

  if (dList.length === 0) {
    panel.style.display = 'none';
    return;
  }

  panel.style.display = 'block';
  const totalPnl = dList.reduce((s, a) => s + (a.pnl_pct || 0), 0);
  const avgPnl   = totalPnl / dList.length;
  const winCnt   = dList.filter(a => (a.pnl_pct || 0) >= 0).length;
  const maxRel   = Math.max(...dList.map(a => a.relative_strength || 0));

  grid.innerHTML = `
    <div class="deep-info-item">
      <div class="d-label">DEEP Active 수</div>
      <div class="d-value">${dList.length}개</div>
    </div>
    <div class="deep-info-item">
      <div class="d-label">DEEP 승률</div>
      <div class="d-value">${Math.round(winCnt / dList.length * 100)}%</div>
    </div>
    <div class="deep-info-item">
      <div class="d-label">DEEP 평균 PnL</div>
      <div class="d-value" style="color:${avgPnl >= 0 ? 'var(--green)' : 'var(--red)'}">
        ${avgPnl >= 0 ? '+' : ''}${avgPnl.toFixed(2)}%
      </div>
    </div>
    <div class="deep-info-item">
      <div class="d-label">최고 상대강도</div>
      <div class="d-value">${maxRel.toFixed(2)}%p</div>
    </div>`;
}

// ── 전체 상태 업데이트 ────────────────────────────────────
function updateState(data) {
  // 헤더
  const badge = document.getElementById('statusBadge');
  const st    = data.status || 'idle';
  badge.textContent = '● ' + st;
  badge.className   = 'status-badge' + (st === 'scanning' ? ' scanning' : st === 'error' ? ' error' : '');

  const lastEl = document.getElementById('lastScan');
  lastEl.textContent = data.last_scan_at
    ? '마지막 스캔: ' + fmtTime(data.last_scan_at) : '';

  // 요약 카드
  const stats = data.stats || {};
  document.getElementById('cardWatch').textContent   = data.watch_count  ?? '-';
  document.getElementById('cardActive').textContent  = data.active_count ?? '-';
  document.getElementById('cardDeep').textContent    = data.deep_count   ?? 0;
  document.getElementById('cardWinRate').textContent =
    stats.win_rate != null ? stats.win_rate.toFixed(1) + '%' : '-';
  document.getElementById('cardWinSub').textContent  =
    stats.total_trades ? `총 ${stats.total_trades}건` : '';
  document.getElementById('cardAvgPnl').textContent  =
    stats.avg_pnl != null
      ? (stats.avg_pnl >= 0 ? '+' : '') + stats.avg_pnl.toFixed(2) + '%' : '-';

  // BTC MA20
  const macro = data.macro || {};
  if (macro.btc_weekly_ma20) {
    document.getElementById('cardBtcMa').textContent = Number(macro.btc_weekly_ma20).toLocaleString();
    const pass = macro.pass;
    document.getElementById('cardBtcSub').innerHTML =
      `현재 ${Number(macro.btc_current || 0).toLocaleString()} `
      + `<span style="color:${pass ? 'var(--green)' : 'var(--red)'}">`
      + `${pass ? '✅통과' : '❌미통과'}</span>`;
  }

  // 탭 카운트
  document.getElementById('tabWatchCnt').textContent  = `(${data.watch_count  || 0})`;
  document.getElementById('tabActiveCnt').textContent = `(${data.active_count || 0})`;
  document.getElementById('tabDeepCnt').textContent   = `(${data.deep_count   || 0})`;

  // 테이블 렌더
  renderWatch(data.watch_list);
  renderActive(data.active_trades);
  renderDeep(data.active_trades);
  renderNew(data.new_entries);
  renderGradeBars(stats.grade_stats);
  renderDeepPanel(data.active_trades);
}

// ── 폴링 ─────────────────────────────────────────────────
async function fetchState() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) throw new Error(r.status);
    updateState(await r.json());
  } catch(e) { console.warn('fetchState:', e); }
}

async function fetchHistory() {
  try {
    const r = await fetch('/api/history');
    if (!r.ok) throw new Error(r.status);
    renderHistory(await r.json());
  } catch(e) { console.warn('fetchHistory:', e); }
}

// ── 액션 ─────────────────────────────────────────────────
async function triggerScan() {
  await fetch('/api/scan', { method: 'POST' });
  setTimeout(fetchState, 1500);
}

async function addWatch() {
  const val = document.getElementById('addInput').value.trim();
  if (!val) return;
  const r = await fetch('/api/watch/add', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ ticker: val }),
  });
  const d = await r.json();
  alert(d.message);
  document.getElementById('addInput').value = '';
  fetchState();
}

async function removeWatch(ticker) {
  if (!confirm(`${ticker} Watch에서 제거?`)) return;
  await fetch('/api/watch/remove', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ ticker }),
  });
  fetchState();
}

async function closeTrade(ticker) {
  if (!confirm(`${ticker} 수동 청산?`)) return;
  const r = await fetch('/api/active/close', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ ticker }),
  });
  const d = await r.json();
  alert(d.message);
  fetchState();
  fetchHistory();
}

// ── 초기 실행 ─────────────────────────────────────────────
fetchState();
fetchHistory();
setInterval(fetchState,   15000);   // 15초 폴링
setInterval(fetchHistory, 60000);   // 60초 히스토리 갱신
setInterval(() => {                 // 1초 카운트다운 갱신
  const cdEl = document.getElementById('newCountdown');
  if (_newRegisteredAt && cdEl) {
    const remain = 300000 - (Date.now() - _newRegisteredAt);
    if (remain > 0) {
      const m = Math.floor(remain / 60000);
      const s = Math.floor((remain % 60000) / 1000);
      cdEl.textContent = `⏱ ${m}분 ${s}초 후 초기화`;
    } else {
      cdEl.textContent = '초기화 대기 중...';
    }
  }
}, 1000);
</script>
</body>
</html>
"""


# ── Flask 라우트 ──────────────────────────────────────────────
@app.route('/')
def index():
    with scanner._state_lock:
        version = scanner.scanner_state.get('version', scanner.VERSION)
    return render_template_string(HTML, version=version)


@app.route('/api/state')
def api_state():
    with scanner._state_lock:
        return jsonify(dict(scanner.scanner_state))


@app.route('/api/history')
def api_history():
    return jsonify(scanner.load_trade_history())


@app.route('/api/scan', methods=['POST'])
def api_scan():
    scanner.manual_scan()
    return jsonify({'success': True, 'message': '수동 스캔 요청됨'})


@app.route('/api/watch/add', methods=['POST'])
def api_watch_add():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    if not ticker:
        return jsonify({'success': False, 'message': '티커를 입력하세요'})
    return jsonify(scanner.add_manual_watch(ticker))


@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    if not ticker:
        return jsonify({'success': False, 'message': '티커를 입력하세요'})
    return jsonify(scanner.remove_watch(ticker))


@app.route('/api/active/close', methods=['POST'])
def api_active_close():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    if not ticker:
        return jsonify({'success': False, 'message': '티커를 입력하세요'})
    return jsonify(scanner.manual_close_trade(ticker))


@app.route('/api/config')
def api_config():
    return jsonify(mtf_setup.get_module_config())


# ── 실행 ─────────────────────────────────────────────────────
if __name__ == '__main__':
    import threading

    threads = [
        threading.Thread(target=scanner.scanner_loop,
                         daemon=True, name='scanner_loop'),
        threading.Thread(target=scanner.watch_rescan_loop,
                         daemon=True, name='watch_rescan'),
        threading.Thread(target=scanner.price_check_loop,
                         daemon=True, name='price_check'),
        threading.Thread(target=scanner.active_monitor_loop,
                         daemon=True, name='active_monitor'),
        threading.Thread(target=scanner.daily_summary_loop,
                         daemon=True, name='daily_summary'),
    ]
    for t in threads:
        t.start()

    app.run(
        host  = '0.0.0.0',
        port  = int(os.environ.get('PORT', 5000)),
        debug = False,
    )
