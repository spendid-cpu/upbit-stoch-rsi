# -*- coding: utf-8 -*-
"""
dashboard.py — Upbit MTF 스캐너 대시보드
Version : v2.3.0
Changelog:
  v2.3.0 - K값 색상 (≤5:빨강, ≤10:주황, ≤20:노랑, >20:기본)
           4hK 과열 경고 (>80: 🔥, >50: ⚠)
           방향 아이콘 컬럼 (일봉/4h/1h)
           진입강도 컬럼 추가
           score_history 미니 스파크라인
           통계 카드 확장 (평균Watch시간, 등급별 승률 바)
           일일 요약 루프 스레드 추가
  v2.2.0 - 4단계 루프 구조
"""

import threading
import logging
import os
from flask import Flask, jsonify, request, render_template_string
import scanner
import mtf_setup

log = logging.getLogger(__name__)
app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════
# HTML 템플릿
# ═══════════════════════════════════════════════════════════════
HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Upbit MTF Scanner v2.3.0</title>
<style>
  :root {
    --bg:      #0d1117;
    --card:    #161b22;
    --border:  #30363d;
    --text:    #e6edf3;
    --muted:   #8b949e;
    --green:   #3fb950;
    --red:     #f85149;
    --orange:  #f0883e;
    --yellow:  #d29922;
    --blue:    #58a6ff;
    --lime:    #7ee787;
    --purple:  #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; font-size: 13px; }
  .header { padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; }
  .header h1 { font-size: 16px; font-weight: 700; }
  .version-badge { background: var(--blue); color: #000; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
  .container { padding: 16px 20px; }

  /* 카드 그리드 */
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-bottom: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  .card-label { color: var(--muted); font-size: 11px; margin-bottom: 4px; }
  .card-value { font-size: 22px; font-weight: 700; }
  .card-sub   { color: var(--muted); font-size: 11px; margin-top: 2px; }

  /* 등급별 승률 바 */
  .grade-bar-wrap { display: flex; flex-direction: column; gap: 4px; margin-top: 4px; }
  .grade-bar-row  { display: flex; align-items: center; gap: 6px; font-size: 11px; }
  .grade-bar-bg   { flex: 1; background: var(--border); border-radius: 3px; height: 6px; }
  .grade-bar-fill { height: 6px; border-radius: 3px; transition: width .4s; }

  /* 섹션 */
  .section { margin-bottom: 20px; }
  .section-title { font-size: 13px; font-weight: 700; margin-bottom: 8px; color: var(--muted); display: flex; align-items: center; gap: 8px; }
  .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
  .badge-blue   { background: #1f3a5f; color: var(--blue); }
  .badge-green  { background: #1a3d2b; color: var(--green); }
  .badge-orange { background: #3d2a0f; color: var(--orange); }
  .badge-red    { background: #3d1515; color: var(--red); }
  .badge-gray   { background: #21262d; color: var(--muted); }

  /* 테이블 */
  .tbl-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { background: #0d1117; color: var(--muted); padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; font-weight: 600; }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d; white-space: nowrap; vertical-align: middle; }
  tr:hover td { background: #1c2128; }

  /* K값 색상 */
  .k-extreme { color: var(--red);    font-weight: 700; }
  .k-low     { color: var(--orange); font-weight: 600; }
  .k-mid     { color: var(--yellow); }
  .k-normal  { color: var(--muted);  }
  .k-hot     { color: var(--red);    font-weight: 700; }

  /* 과열 아이콘 */
  .overheat  { font-size: 12px; }

  /* 방향 아이콘 */
  .dir-up    { color: var(--green);  }
  .dir-rise  { color: var(--lime);   }
  .dir-side  { color: var(--muted);  }
  .dir-down  { color: var(--red);    }
  .dir-gold  { color: var(--yellow); }

  /* 진입강도 */
  .es-3 { color: var(--green);  font-weight: 700; }
  .es-2 { color: var(--lime);   }
  .es-1 { color: var(--yellow); }
  .es-0 { color: var(--muted);  }

  /* 수익률 */
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red);   }
  .pnl-neu { color: var(--muted); }

  /* 점수 변화 */
  .sc-up   { color: var(--green); }
  .sc-down { color: var(--red);   }
  .sc-neu  { color: var(--muted); }

  /* 스파크라인 */
  .spark { display: inline-flex; align-items: flex-end; gap: 1px; height: 18px; }
  .spark-bar { width: 4px; background: var(--blue); border-radius: 1px; min-height: 2px; opacity: .8; }

  /* 버튼 */
  .btn { padding: 4px 10px; border-radius: 5px; border: none; cursor: pointer; font-size: 11px; font-weight: 600; }
  .btn-scan   { background: var(--blue);   color: #000; }
  .btn-del    { background: #3d1515;       color: var(--red); }
  .btn-close  { background: #1a3d2b;       color: var(--green); }
  .btn:hover  { opacity: .85; }

  /* 입력 */
  .input-row { display: flex; gap: 8px; margin-bottom: 10px; }
  input[type=text] { background: var(--card); border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 5px; font-size: 12px; }

  /* 상태 점 */
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot-green  { background: var(--green);  }
  .dot-orange { background: var(--orange); }
  .dot-red    { background: var(--red);    }

  /* 메시지 */
  #msg { padding: 6px 12px; border-radius: 5px; margin-bottom: 10px; font-size: 12px; display: none; }
  .msg-ok  { background: #1a3d2b; color: var(--green); }
  .msg-err { background: #3d1515; color: var(--red);   }

  /* 푸터 타임스탬프 */
  .timestamps { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 6px; font-size: 11px; color: var(--muted); }

  /* 탭 */
  .tabs { display: flex; gap: 4px; margin-bottom: 12px; }
  .tab { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); color: var(--muted); cursor: pointer; font-size: 12px; }
  .tab.active { background: var(--blue); color: #000; border-color: var(--blue); font-weight: 700; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
</style>
</head>
<body>

<div class="header">
  <h1>📡 Upbit MTF Scanner</h1>
  <span class="version-badge">v2.3.0</span>
  <span id="statusDot" class="dot dot-green" style="margin-left:auto"></span>
  <span id="statusTxt" style="font-size:11px;color:var(--muted)">idle</span>
  <button class="btn btn-scan" onclick="manualScan()">🔄 수동스캔</button>
</div>

<div class="container">
  <div id="msg"></div>

  <!-- 요약 카드 -->
  <div class="cards" id="summaryCards">
    <div class="card">
      <div class="card-label">Watch 종목</div>
      <div class="card-value" id="cWatchCnt">-</div>
      <div class="card-sub" id="cWatchSub">-</div>
    </div>
    <div class="card">
      <div class="card-label">Active 종목</div>
      <div class="card-value" id="cActiveCnt">-</div>
      <div class="card-sub" id="cActiveSub">-</div>
    </div>
    <div class="card">
      <div class="card-label">승률 (TP)</div>
      <div class="card-value" id="cWinRate">-</div>
      <div class="card-sub" id="cWinSub">-</div>
    </div>
    <div class="card">
      <div class="card-label">평균 PnL</div>
      <div class="card-value" id="cAvgPnl">-</div>
      <div class="card-sub" id="cPnlSub">-</div>
    </div>
    <div class="card">
      <div class="card-label">BTC 주봉MA20</div>
      <div class="card-value" id="cBtcWeekly" style="font-size:15px">-</div>
      <div class="card-sub" id="cBtcDaily">-</div>
    </div>
    <div class="card">
      <div class="card-label">Watch 전환율</div>
      <div class="card-value" id="cConvRate">-</div>
      <div class="card-sub" id="cAvgWatch">-</div>
    </div>
  </div>

  <!-- 등급별 승률 바 -->
  <div class="card" style="margin-bottom:16px">
    <div class="card-label" style="margin-bottom:8px">등급별 승률</div>
    <div class="grade-bar-wrap" id="gradeBarWrap">
      <div class="grade-bar-row"><span style="width:12px">S</span><div class="grade-bar-bg"><div class="grade-bar-fill" id="barS" style="width:0%;background:#f85149"></div></div><span id="lblS">0%</span></div>
      <div class="grade-bar-row"><span style="width:12px">A</span><div class="grade-bar-bg"><div class="grade-bar-fill" id="barA" style="width:0%;background:#f0883e"></div></div><span id="lblA">0%</span></div>
      <div class="grade-bar-row"><span style="width:12px">B</span><div class="grade-bar-bg"><div class="grade-bar-fill" id="barB" style="width:0%;background:#d29922"></div></div><span id="lblB">0%</span></div>
      <div class="grade-bar-row"><span style="width:12px">C</span><div class="grade-bar-bg"><div class="grade-bar-fill" id="barC" style="width:0%;background:#8b949e"></div></div><span id="lblC">0%</span></div>
    </div>
  </div>

  <!-- 탭 -->
  <div class="tabs">
    <div class="tab active" onclick="switchTab('watch')">📋 Watch (<span id="tabWatchCnt">0</span>)</div>
    <div class="tab" onclick="switchTab('active')">🔵 Active (<span id="tabActiveCnt">0</span>)</div>
    <div class="tab" onclick="switchTab('new')">🆕 신규 (<span id="tabNewCnt">0</span>)</div>
    <div class="tab" onclick="switchTab('removed')">🗑️ 만료 (<span id="tabRemovedCnt">0</span>)</div>
    <div class="tab" onclick="switchTab('history')">📜 히스토리</div>
  </div>

  <!-- Watch 탭 -->
  <div class="tab-content active" id="tabWatch">
    <div class="input-row">
      <input type="text" id="addInput" placeholder="종목 입력 (예: CHZ)" onkeydown="if(event.key==='Enter')addWatch()">
      <button class="btn btn-scan" onclick="addWatch()">+ 추가</button>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th><th>등급</th><th>등록점수</th><th>현재점수</th><th>변화</th>
            <th>추세</th><th>일봉K</th><th>4hK</th><th>1hK</th>
            <th>진입강도</th><th>등록가</th><th>현재가</th><th>수익률</th>
            <th>등록시간</th><th>Watch시간</th><th>구분</th><th>삭제</th>
          </tr>
        </thead>
        <tbody id="watchTbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Active 탭 -->
  <div class="tab-content" id="tabActive">
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th><th>등급</th><th>진입점수</th><th>진입강도</th>
            <th>진입가</th><th>현재가</th><th>수익률</th><th>TP</th><th>SL</th>
            <th>진입시간</th><th>보유시간</th><th>청산</th>
          </tr>
        </thead>
        <tbody id="activeTbody"></tbody>
      </table>
    </div>
  </div>

  <!-- 신규 탭 -->
  <div class="tab-content" id="tabNew">
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th><th>등급</th><th>점수</th><th>추세</th>
            <th>일봉K</th><th>4hK</th><th>1hK</th><th>진입강도</th><th>등록시간</th>
          </tr>
        </thead>
        <tbody id="newTbody"></tbody>
      </table>
    </div>
  </div>

  <!-- 만료 탭 -->
  <div class="tab-content" id="tabRemoved">
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th><th>등급</th><th>등록점수</th><th>등록가</th><th>최종가</th><th>수익률</th>
          </tr>
        </thead>
        <tbody id="removedTbody"></tbody>
      </table>
    </div>
  </div>

  <!-- 히스토리 탭 -->
  <div class="tab-content" id="tabHistory">
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th><th>결과</th><th>등급</th><th>진입가</th><th>청산가</th>
            <th>수익률</th><th>보유시간</th><th>청산시각</th>
          </tr>
        </thead>
        <tbody id="historyTbody"></tbody>
      </table>
    </div>
  </div>

  <!-- 타임스탬프 -->
  <div class="timestamps" style="margin-top:16px">
    <span>전체스캔: <span id="tsLastScan">-</span></span>
    <span>Watch재스캔: <span id="tsWatchRescan">-</span></span>
    <span>가격체크: <span id="tsPrice">-</span></span>
    <span>Active체크: <span id="tsActive">-</span></span>
    <span>다음스캔: <span id="tsNextScan" style="color:var(--blue)">-</span></span>
    <span style="margin-left:auto">스캔횟수: <span id="tsScanCnt">0</span></span>
  </div>
</div>

<script>
// ── 유틸 ─────────────────────────────────────────────────────
function fmt(v, dec=null) {
  if (v == null) return '-';
  const n = Number(v);
  if (isNaN(n)) return '-';
  if (dec !== null) return n.toLocaleString('ko-KR', {minimumFractionDigits: dec, maximumFractionDigits: dec});
  if (n >= 100)   return n.toLocaleString('ko-KR', {maximumFractionDigits: 0});
  if (n >= 10)    return n.toLocaleString('ko-KR', {maximumFractionDigits: 1});
  if (n >= 1)     return n.toLocaleString('ko-KR', {maximumFractionDigits: 2});
  if (n >= 0.1)   return n.toLocaleString('ko-KR', {maximumFractionDigits: 3});
  if (n >= 0.01)  return n.toLocaleString('ko-KR', {maximumFractionDigits: 4});
  return n.toLocaleString('ko-KR', {maximumFractionDigits: 6});
}

function fmtPct(v) {
  if (v == null) return '-';
  const n = Number(v);
  const cls = n > 0 ? 'pnl-pos' : n < 0 ? 'pnl-neg' : 'pnl-neu';
  return `<span class="${cls}">${n >= 0 ? '+' : ''}${n.toFixed(2)}%</span>`;
}

function fmtTime(iso) {
  if (!iso) return '-';
  try {
    const d = new Date(iso);
    return `${String(d.getMonth()+1).padStart(2,'0')}/${String(d.getDate()).padStart(2,'0')} `
         + `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
  } catch { return '-'; }
}

function fmtElapsed(iso) {
  if (!iso) return '-';
  try {
    const diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (diff < 3600)  return `${Math.floor(diff/60)}분`;
    if (diff < 86400) return `${Math.floor(diff/3600)}h ${Math.floor((diff%3600)/60)}m`;
    return `${Math.floor(diff/86400)}일 ${Math.floor((diff%86400)/3600)}h`;
  } catch { return '-'; }
}

function gradeBadge(g) {
  const map = {S:'badge-red',A:'badge-orange',B:'badge-blue',C:'badge-gray'};
  return `<span class="badge ${map[g]||'badge-gray'}">${g||'?'}</span>`;
}

function scoreChangeBadge(snap_score, cur_score) {
  const diff = (cur_score||0) - (snap_score||0);
  if (diff > 0)  return `<span class="sc-up">+${diff}↑</span>`;
  if (diff < 0)  return `<span class="sc-down">${diff}↓</span>`;
  return `<span class="sc-neu">→</span>`;
}

// K값 색상
function kCell(val) {
  if (val == null) return '<span class="k-normal">-</span>';
  const n = Number(val);
  if (n <= 5)  return `<span class="k-extreme">${n.toFixed(1)}</span>`;
  if (n <= 10) return `<span class="k-low">${n.toFixed(1)}</span>`;
  if (n <= 20) return `<span class="k-mid">${n.toFixed(1)}</span>`;
  return `<span class="k-normal">${n.toFixed(1)}</span>`;
}

// 4hK 과열 아이콘
function h4KCell(val) {
  if (val == null) return '<span class="k-normal">-</span>';
  const n = Number(val);
  let icon = '';
  if (n > 80) icon = '<span class="overheat">🔥</span>';
  else if (n > 50) icon = '<span class="overheat">⚠️</span>';
  if (n <= 5)  return `<span class="k-extreme">${n.toFixed(1)}${icon}</span>`;
  if (n <= 10) return `<span class="k-low">${n.toFixed(1)}${icon}</span>`;
  if (n <= 20) return `<span class="k-mid">${n.toFixed(1)}${icon}</span>`;
  return `<span class="k-normal">${n.toFixed(1)}${icon}</span>`;
}

// 방향 아이콘
function dirIcon(dir, golden=false) {
  if (golden) return '<span class="dir-gold">✨GX</span>';
  const map = {
    '상승': '<span class="dir-up">↑</span>',
    '반등': '<span class="dir-rise">↗</span>',
    '횡보': '<span class="dir-side">→</span>',
    '하락': '<span class="dir-down">↓</span>',
  };
  return map[dir] || '<span class="dir-side">?</span>';
}

// 방향 3개 합쳐서 표시
function dirCell(cur) {
  const d = dirIcon(cur.daily_dir, false);
  const h = dirIcon(cur.h4_dir,   cur.h4_golden);
  const l = dirIcon(cur.h1_dir,   cur.h1_golden);
  return `${d}${h}${l}`;
}

// 진입강도
function esCell(es) {
  if (!es) return '-';
  const cls = ['es-0','es-1','es-2','es-3'][es.level||0];
  return `<span class="${cls}">${es.icon||''} ${es.label||''}</span>`;
}

// 스파크라인
function sparkline(history) {
  if (!history || history.length < 2) return '';
  const scores = history.slice(-10).map(h => h.score||0);
  const maxS = Math.max(...scores, 1);
  const bars  = scores.map(s => {
    const h = Math.max(2, Math.round(s / maxS * 18));
    return `<div class="spark-bar" style="height:${h}px"></div>`;
  }).join('');
  return `<div class="spark">${bars}</div>`;
}

// ── 탭 전환 ──────────────────────────────────────────────────
let currentTab = 'watch';
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    const names = ['watch','active','new','removed','history'];
    t.classList.toggle('active', names[i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  const map = {watch:'tabWatch',active:'tabActive',new:'tabNew',removed:'tabRemoved',history:'tabHistory'};
  document.getElementById(map[name]).classList.add('active');
  currentTab = name;
  if (name === 'history') loadHistory();
}

// ── 렌더링 ───────────────────────────────────────────────────
function updateUI(state) {
  const stats = state.stats || {};
  const macro = state.macro || {};

  // 상태 표시
  const dotEl = document.getElementById('statusDot');
  const txtEl = document.getElementById('statusTxt');
  dotEl.className = 'dot ' + (state.status==='scanning'?'dot-orange':state.status==='error'?'dot-red':'dot-green');
  txtEl.textContent = state.status || 'idle';

  // 카드
  document.getElementById('cWatchCnt').textContent  = state.watch_count  || 0;
  document.getElementById('cActiveCnt').textContent = state.active_count || 0;
  document.getElementById('tabWatchCnt').textContent  = state.watch_count  || 0;
  document.getElementById('tabActiveCnt').textContent = state.active_count || 0;
  document.getElementById('tabNewCnt').textContent     = (state.new_entries||[]).length;
  document.getElementById('tabRemovedCnt').textContent = (state.removed_items||[]).length;

  const wr = stats.win_rate || 0;
  document.getElementById('cWinRate').textContent = wr + '%';
  document.getElementById('cWinRate').style.color = wr >= 50 ? 'var(--green)' : wr >= 30 ? 'var(--orange)' : 'var(--red)';
  document.getElementById('cWinSub').textContent = `TP:${stats.tp||0} SL:${stats.sl||0} TO:${stats.timeout||0}`;

  const ap = stats.avg_pnl || 0;
  document.getElementById('cAvgPnl').textContent = (ap>=0?'+':'')+ap.toFixed(2)+'%';
  document.getElementById('cAvgPnl').style.color = ap >= 0 ? 'var(--green)' : 'var(--red)';
  document.getElementById('cPnlSub').textContent = `최고:${stats.best_pnl||0}% 최저:${stats.worst_pnl||0}%`;

  if (macro.btc_price && macro.btc_weekly_ma20) {
    const ok = macro.macro_ok;
    document.getElementById('cBtcWeekly').textContent = fmt(macro.btc_weekly_ma20);
    document.getElementById('cBtcWeekly').style.color = ok ? 'var(--green)' : 'var(--red)';
    document.getElementById('cBtcDaily').textContent  = `일봉MA20: ${fmt(macro.btc_daily_ma20)}`;
  }

  // 전환율 / 평균Watch시간
  const activated = stats.activated || 0;
  const total     = stats.total     || 0;
  const convRate  = total ? Math.round(activated / total * 100) : 0;
  document.getElementById('cConvRate').textContent = convRate + '%';
  document.getElementById('cAvgWatch').textContent = `평균Watch: ${stats.avg_watch_hours||0}h`;

  // 등급별 승률 바
  const gs = stats.grade_stats || {};
  ['S','A','B','C'].forEach(g => {
    const wr = gs[g]?.win_rate || 0;
    document.getElementById('bar'+g).style.width = wr + '%';
    document.getElementById('lbl'+g).textContent = wr + '%';
  });

  // 테이블 렌더
  renderWatch(state.watch_list || []);
  renderActive(state.active_trades || []);
  renderNew(state.new_entries || []);
  renderRemoved(state.removed_items || []);

  // 타임스탬프
  document.getElementById('tsLastScan').textContent    = fmtTime(state.last_scan_at);
  document.getElementById('tsWatchRescan').textContent = fmtTime(state.last_watch_rescan_at);
  document.getElementById('tsPrice').textContent       = fmtTime(state.last_price_check_at);
  document.getElementById('tsActive').textContent      = fmtTime(state.last_active_check_at);
  document.getElementById('tsScanCnt').textContent     = state.scan_count || 0;

  // 다음스캔 카운트다운
  updateCountdown(state.next_scan_at);
}

function renderWatch(list) {
  const tbody = document.getElementById('watchTbody');
  if (!list.length) { tbody.innerHTML = '<tr><td colspan="17" style="text-align:center;color:var(--muted)">Watch 종목 없음</td></tr>'; return; }
  tbody.innerHTML = list.map(w => {
    const snap = w.snapshot || {};
    const cur  = w.current  || {};
    const diff = (cur.price - snap.entry_price) / (snap.entry_price||1) * 100;
    const es   = cur.entry_strength || {};
    return `<tr>
      <td><b>${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${snap.score||0}</td>
      <td><b>${cur.score||0}</b></td>
      <td>${scoreChangeBadge(snap.score, cur.score)}</td>
      <td>${dirCell(cur)}</td>
      <td>${kCell(cur.daily_k)}</td>
      <td>${h4KCell(cur.h4_k)}</td>
      <td>${kCell(cur.h1_k)}</td>
      <td>${esCell(es)}</td>
      <td>${fmt(snap.entry_price)}</td>
      <td>${fmt(cur.price)}</td>
      <td>${fmtPct(diff)}</td>
      <td>${fmtTime(snap.registered_at)}</td>
      <td>${fmtElapsed(snap.registered_at)}</td>
      <td><span class="badge ${w.manual?'badge-blue':'badge-gray'}">${w.manual?'수동':'자동'}</span></td>
      <td><button class="btn btn-del" onclick="removeWatch('${w.ticker}')">✕</button></td>
    </tr>`;
  }).join('');
}

function renderActive(list) {
  const tbody = document.getElementById('activeTbody');
  if (!list.length) { tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;color:var(--muted)">Active 종목 없음</td></tr>'; return; }
  tbody.innerHTML = list.map(t => {
    const es = t.entry_strength || {};
    return `<tr>
      <td><b>${t.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(t.grade)}</td>
      <td>${t.entry_score||0}</td>
      <td>${esCell(es)}</td>
      <td>${fmt(t.entry_price)}</td>
      <td>${fmt(t.current_price)}</td>
      <td>${fmtPct(t.pnl_pct)}</td>
      <td style="color:var(--green)">${fmt(t.tp_price)}</td>
      <td style="color:var(--red)">${fmt(t.sl_price)}</td>
      <td>${fmtTime(t.activated_at)}</td>
      <td>${fmtElapsed(t.activated_at)}</td>
      <td><button class="btn btn-close" onclick="closeTrade('${t.ticker}')">청산</button></td>
    </tr>`;
  }).join('');
}

function renderNew(list) {
  const tbody = document.getElementById('newTbody');
  if (!list.length) { tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--muted)">신규 등록 없음</td></tr>'; return; }
  tbody.innerHTML = list.map(w => {
    const snap = w.snapshot || {};
    const cur  = w.current  || {};
    const es   = cur.entry_strength || {};
    return `<tr>
      <td><b>${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${snap.score||0}</td>
      <td>${dirCell(cur)}</td>
      <td>${kCell(snap.daily_k)}</td>
      <td>${h4KCell(snap.h4_k)}</td>
      <td>${kCell(snap.h1_k)}</td>
      <td>${esCell(es)}</td>
      <td>${fmtTime(snap.registered_at)}</td>
    </tr>`;
  }).join('');
}

function renderRemoved(list) {
  const tbody = document.getElementById('removedTbody');
  if (!list.length) { tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--muted)">만료 종목 없음</td></tr>'; return; }
  tbody.innerHTML = list.map(w => {
    const snap = w.snapshot || {};
    const cur  = w.current  || {};
    const diff = snap.entry_price ? (cur.price - snap.entry_price) / snap.entry_price * 100 : 0;
    return `<tr>
      <td><b>${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${snap.score||0}</td>
      <td>${fmt(snap.entry_price)}</td>
      <td>${fmt(cur.price)}</td>
      <td>${fmtPct(diff)}</td>
    </tr>`;
  }).join('');
}

async function loadHistory() {
  try {
    const res  = await fetch('/api/history');
    const data = await res.json();
    const list = (data.history || []).slice().reverse().slice(0, 100);
    const tbody = document.getElementById('historyTbody');
    if (!list.length) { tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--muted)">히스토리 없음</td></tr>'; return; }
    const resultMap = {tp:'✅TP', sl:'❌SL', timeout:'⏰TO', manual:'🖐️수동', activated:'🔵Active', expired:'🗑️만료', manual_remove:'🗑️삭제'};
    tbody.innerHTML = list.map(h => `<tr>
      <td><b>${(h.ticker||'').replace('KRW-','')}</b></td>
      <td>${resultMap[h.result]||h.result}</td>
      <td>${gradeBadge(h.grade)}</td>
      <td>${fmt(h.entry_price)}</td>
      <td>${fmt(h.close_price)}</td>
      <td>${h.pnl_pct != null ? fmtPct(h.pnl_pct) : '-'}</td>
      <td>${h.hours_held != null ? h.hours_held.toFixed(1)+'h' : '-'}</td>
      <td>${fmtTime(h.closed_at || h.activated_at)}</td>
    </tr>`).join('');
  } catch(e) { console.error(e); }
}

// ── 카운트다운 ────────────────────────────────────────────────
let _nextAt = null;
function updateCountdown(iso) {
  _nextAt = iso;
}
setInterval(() => {
  if (!_nextAt) return;
  const diff = Math.max(0, Math.floor((new Date(_nextAt) - Date.now()) / 1000));
  const m = Math.floor(diff / 60), s = diff % 60;
  document.getElementById('tsNextScan').textContent =
    diff > 0 ? `${m}분 ${String(s).padStart(2,'0')}초 후` : '스캔 중...';
}, 1000);

// ── 보유시간 실시간 갱신 ──────────────────────────────────────
setInterval(() => {
  document.querySelectorAll('[data-elapsed]').forEach(el => {
    el.textContent = fmtElapsed(el.dataset.elapsed);
  });
}, 30000);

// ── API 액션 ─────────────────────────────────────────────────
function showMsg(txt, ok=true) {
  const el = document.getElementById('msg');
  el.textContent = txt;
  el.className   = ok ? 'msg-ok' : 'msg-err';
  el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', 3000);
}

async function manualScan() {
  try {
    const r = await fetch('/api/scan', {method:'POST'});
    const d = await r.json();
    showMsg(d.msg || '스캔 트리거 완료');
  } catch { showMsg('오류 발생', false); }
}

async function addWatch() {
  const ticker = document.getElementById('addInput').value.trim();
  if (!ticker) return;
  try {
    const r = await fetch('/api/watch/add', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ticker})});
    const d = await r.json();
    showMsg(d.msg, d.ok);
    if (d.ok) { document.getElementById('addInput').value = ''; poll(); }
  } catch { showMsg('오류 발생', false); }
}

async function removeWatch(ticker) {
  if (!confirm(`${ticker} Watch에서 삭제하시겠습니까?`)) return;
  try {
    const r = await fetch('/api/watch/remove', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ticker})});
    const d = await r.json();
    showMsg(d.msg, d.ok);
    if (d.ok) poll();
  } catch { showMsg('오류 발생', false); }
}

async function closeTrade(ticker) {
  if (!confirm(`${ticker} 수동 청산하시겠습니까?`)) return;
  try {
    const r = await fetch('/api/trade/close', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ticker})});
    const d = await r.json();
    showMsg(d.msg, d.ok);
    if (d.ok) poll();
  } catch { showMsg('오류 발생', false); }
}

// ── 폴링 ─────────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();
    updateUI(d);
  } catch(e) { console.error('poll 오류:', e); }
}

poll();
setInterval(poll, 15000);
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════
# Flask 라우트
# ═══════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template_string(HTML,
        tp_pct=scanner.TRADE_TP_PCT,
        sl_pct=scanner.TRADE_SL_PCT,
    )

@app.route('/api/state')
def api_state():
    with scanner._state_lock:
        return jsonify(dict(scanner.scanner_state))

@app.route('/api/config')
def api_config():
    return jsonify(mtf_setup.get_module_config())

@app.route('/api/scan', methods=['POST'])
def api_scan():
    scanner.manual_scan()
    return jsonify({'ok': True, 'msg': '수동 스캔 트리거 완료'})

@app.route('/api/watch/add', methods=['POST'])
def api_watch_add():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    if not ticker:
        return jsonify({'ok': False, 'msg': '종목명 필요'}), 400
    return jsonify(scanner.add_manual_watch(ticker))

@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    if not ticker:
        return jsonify({'ok': False, 'msg': '종목명 필요'}), 400
    return jsonify(scanner.remove_watch(ticker))

@app.route('/api/trade/close', methods=['POST'])
def api_trade_close():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    if not ticker:
        return jsonify({'ok': False, 'msg': '종목명 필요'}), 400
    return jsonify(scanner.manual_close_trade(ticker))

@app.route('/api/history')
def api_history():
    return jsonify({'history': scanner.load_trade_history()})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'version': scanner.VERSION})

# ═══════════════════════════════════════════════════════════════
# 스레드 시작
# ═══════════════════════════════════════════════════════════════
def _start_threads():
    try:
        watch_list    = scanner.load_watch_list()
        active_trades = scanner.load_active_trades()
        stats         = scanner.calc_stats()
        with scanner._state_lock:
            scanner.scanner_state['watch_list']    = watch_list
            scanner.scanner_state['active_trades'] = active_trades
            scanner.scanner_state['stats']         = stats
        log.info(f"초기 로드: Watch {len(watch_list)}개, Active {len(active_trades)}개")
    except Exception as e:
        log.warning(f"초기 로드 실패: {e}")

    for target in [
        scanner.scanner_loop,
        scanner.watch_rescan_loop,
        scanner.price_check_loop,
        scanner.active_monitor_loop,
        scanner.daily_summary_loop,
    ]:
        threading.Thread(target=target, daemon=True).start()

_start_threads()

if __name__ == '__main__':
    port = int(os.getenv('PORT', '8080'))
    app.run(host='0.0.0.0', port=port, debug=False)
