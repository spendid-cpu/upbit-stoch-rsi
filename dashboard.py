# -*- coding: utf-8 -*-
"""
dashboard.py — Upbit MTF 스캐너 대시보드
Version : v2.4.0
Changelog:
  v2.4.0 - DEEP Watch 전용 UI 추가
           DEEP 배지 (💎), DEEP 카드, DEEP 통계
           Active 테이블에 DEEP 구분 표시
           BTC 24h 변화율 표시
           진입강도 등급 상한 완전 제거 (순수 방향 기반)
           해석표 범례 업데이트
  v2.3.1 - dirIcon null 방어, 스파크라인, MED 버그 수정
"""

import threading
import logging
import os
from flask import Flask, jsonify, request, render_template_string
import scanner
import mtf_setup

log = logging.getLogger(__name__)
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Upbit MTF Scanner v2.4.0</title>
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
    --deep:    #58a6ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; font-size: 13px; }

  .header { padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .header h1 { font-size: 16px; font-weight: 700; }
  .version-badge { background: var(--blue); color: #000; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
  .deep-badge    { background: #1a2a4a; color: var(--purple); padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; border: 1px solid var(--purple); }

  .container { padding: 16px 20px; }

  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-bottom: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  .card.deep-card { border-color: var(--purple); background: #0f111a; }
  .card-label { color: var(--muted); font-size: 11px; margin-bottom: 4px; }
  .card-value { font-size: 22px; font-weight: 700; }
  .card-sub   { color: var(--muted); font-size: 11px; margin-top: 2px; }

  .grade-bar-wrap { display: flex; flex-direction: column; gap: 5px; margin-top: 6px; }
  .grade-bar-row  { display: flex; align-items: center; gap: 6px; font-size: 11px; }
  .grade-bar-bg   { flex: 1; background: var(--border); border-radius: 3px; height: 7px; }
  .grade-bar-fill { height: 7px; border-radius: 3px; transition: width .5s; }

  .badge         { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
  .badge-red     { background: #3d1515; color: var(--red); }
  .badge-orange  { background: #3d2a0f; color: var(--orange); }
  .badge-blue    { background: #1f3a5f; color: var(--blue); }
  .badge-gray    { background: #21262d; color: var(--muted); }
  .badge-green   { background: #1a3d2b; color: var(--green); }
  .badge-purple  { background: #1a1a3a; color: var(--purple); border: 1px solid var(--purple); }
  .badge-deep-s  { background: #2a0f3d; color: #e879f9; border: 1px solid #e879f9; }
  .badge-deep-a  { background: #0f1f3d; color: var(--blue); border: 1px solid var(--blue); }
  .badge-deep-b  { background: #1a1a3d; color: var(--purple); border: 1px solid var(--purple); }

  .tbl-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { background: #0d1117; color: var(--muted); padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; font-weight: 600; }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d; white-space: nowrap; vertical-align: middle; }
  tr:hover td  { background: #1c2128; }
  tr.deep-row  { background: #0a0d14; }
  tr.deep-row:hover td { background: #101520; }

  .k-extreme { color: var(--red);    font-weight: 700; }
  .k-low     { color: var(--orange); font-weight: 600; }
  .k-mid     { color: var(--yellow); }
  .k-normal  { color: var(--muted);  }

  .dir-up   { color: var(--green);  font-weight: 700; }
  .dir-rise { color: var(--lime);   }
  .dir-side { color: var(--muted);  }
  .dir-down { color: var(--red);    }
  .dir-gold { color: var(--yellow); font-weight: 700; }

  .es-3 { color: var(--green);  font-weight: 700; }
  .es-2 { color: var(--lime);   font-weight: 600; }
  .es-1 { color: var(--yellow); }
  .es-0 { color: var(--muted);  }

  .pnl-pos { color: var(--green); font-weight: 600; }
  .pnl-neg { color: var(--red);   font-weight: 600; }
  .pnl-neu { color: var(--muted); }

  .sc-up   { color: var(--green); }
  .sc-down { color: var(--red);   }
  .sc-neu  { color: var(--muted); }

  .deep-score { color: var(--purple); font-weight: 700; font-size: 13px; }
  .deep-rel   { color: var(--lime);   font-weight: 600; }
  .deep-vol   { color: var(--blue);   }

  .spark { display: inline-flex; align-items: flex-end; gap: 1px; height: 16px; vertical-align: middle; margin-left: 4px; }
  .spark-bar { width: 3px; background: var(--blue); border-radius: 1px; min-height: 2px; opacity: .75; }

  .btn       { padding: 4px 10px; border-radius: 5px; border: none; cursor: pointer; font-size: 11px; font-weight: 600; }
  .btn-scan  { background: var(--blue);   color: #000; }
  .btn-del   { background: #3d1515;       color: var(--red); }
  .btn-close { background: #1a3d2b;       color: var(--green); }
  .btn:hover { opacity: .85; }

  .input-row { display: flex; gap: 8px; margin-bottom: 10px; }
  input[type=text] { background: var(--card); border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 5px; font-size: 12px; width: 180px; }

  .dot        { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot-green  { background: var(--green); }
  .dot-orange { background: var(--orange); animation: pulse .8s infinite; }
  .dot-red    { background: var(--red); }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  #msg { padding: 6px 12px; border-radius: 5px; margin-bottom: 10px; font-size: 12px; display: none; }
  .msg-ok  { background: #1a3d2b; color: var(--green); }
  .msg-err { background: #3d1515; color: var(--red);   }

  .timestamps { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; font-size: 11px; color: var(--muted); }

  .tabs { display: flex; gap: 4px; margin-bottom: 12px; flex-wrap: wrap; }
  .tab { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); color: var(--muted); cursor: pointer; font-size: 12px; transition: all .2s; }
  .tab.active { background: var(--blue); color: #000; border-color: var(--blue); font-weight: 700; }
  .tab.deep-tab { border-color: var(--purple); color: var(--purple); }
  .tab.deep-tab.active { background: var(--purple); color: #000; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .legend { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; font-size: 11px; color: var(--muted); background: var(--card); padding: 10px 14px; border-radius: 6px; border: 1px solid var(--border); }
  .legend-title { color: var(--text); font-weight: 600; }

  /* DEEP 전용 정보 패널 */
  .deep-info { background: #0a0d14; border: 1px solid var(--purple); border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 11px; }
  .deep-info-title { color: var(--purple); font-weight: 700; margin-bottom: 6px; font-size: 12px; }
  .deep-info-row { display: flex; gap: 16px; flex-wrap: wrap; }
  .deep-info-item { display: flex; flex-direction: column; gap: 2px; }
  .deep-info-label { color: var(--muted); }
  .deep-info-value { color: var(--purple); font-weight: 600; }
</style>
</head>
<body>

<div class="header">
  <h1>📡 Upbit MTF Scanner</h1>
  <span class="version-badge">v2.4.0</span>
  <span class="deep-badge">💎 DEEP</span>
  <span id="statusDot" class="dot dot-green" style="margin-left:auto"></span>
  <span id="statusTxt" style="font-size:11px;color:var(--muted);margin-right:8px">idle</span>
  <button class="btn btn-scan" onclick="manualScan()">🔄 수동스캔</button>
</div>

<div class="container">
  <div id="msg"></div>

  <!-- 요약 카드 -->
  <div class="cards">
    <div class="card">
      <div class="card-label">Watch 종목</div>
      <div class="card-value" id="cWatchCnt">-</div>
      <div class="card-sub"  id="cWatchSub">-</div>
    </div>
    <div class="card deep-card">
      <div class="card-label">💎 DEEP Watch</div>
      <div class="card-value" id="cDeepCnt" style="color:var(--purple)">-</div>
      <div class="card-sub"  id="cDeepSub">-</div>
    </div>
    <div class="card">
      <div class="card-label">Active 종목</div>
      <div class="card-value" id="cActiveCnt">-</div>
      <div class="card-sub"  id="cActiveSub">-</div>
    </div>
    <div class="card">
      <div class="card-label">승률 (전체)</div>
      <div class="card-value" id="cWinRate">-</div>
      <div class="card-sub"  id="cWinSub">-</div>
    </div>
    <div class="card deep-card">
      <div class="card-label">💎 DEEP 승률</div>
      <div class="card-value" id="cDeepWinRate" style="color:var(--purple)">-</div>
      <div class="card-sub"  id="cDeepWinSub">-</div>
    </div>
    <div class="card">
      <div class="card-label">평균 PnL</div>
      <div class="card-value" id="cAvgPnl">-</div>
      <div class="card-sub"  id="cPnlSub">-</div>
    </div>
    <div class="card">
      <div class="card-label">BTC 주봉MA20</div>
      <div class="card-value" id="cBtcWeekly" style="font-size:15px">-</div>
      <div class="card-sub"  id="cBtcInfo">-</div>
    </div>
    <div class="card">
      <div class="card-label">Watch 전환율</div>
      <div class="card-value" id="cConvRate">-</div>
      <div class="card-sub"  id="cAvgWatch">-</div>
    </div>
  </div>

  <!-- 등급별 승률 바 -->
  <div class="card" style="margin-bottom:12px">
    <div class="card-label" style="margin-bottom:8px">등급별 승률</div>
    <div class="grade-bar-wrap">
      <div class="grade-bar-row">
        <span style="width:50px;color:#e879f9">DEEP-S</span>
        <div class="grade-bar-bg"><div class="grade-bar-fill" id="barDS" style="width:0%;background:#e879f9"></div></div>
        <span id="lblDS" style="min-width:32px">0%</span>
        <span id="cntDS" style="color:var(--muted);font-size:10px"></span>
      </div>
      <div class="grade-bar-row">
        <span style="width:50px;color:var(--blue)">DEEP-A</span>
        <div class="grade-bar-bg"><div class="grade-bar-fill" id="barDA" style="width:0%;background:var(--blue)"></div></div>
        <span id="lblDA" style="min-width:32px">0%</span>
        <span id="cntDA" style="color:var(--muted);font-size:10px"></span>
      </div>
      <div class="grade-bar-row">
        <span style="width:50px;color:var(--red)">S</span>
        <div class="grade-bar-bg"><div class="grade-bar-fill" id="barS" style="width:0%;background:var(--red)"></div></div>
        <span id="lblS" style="min-width:32px">0%</span>
        <span id="cntS" style="color:var(--muted);font-size:10px"></span>
      </div>
      <div class="grade-bar-row">
        <span style="width:50px;color:var(--orange)">A</span>
        <div class="grade-bar-bg"><div class="grade-bar-fill" id="barA" style="width:0%;background:var(--orange)"></div></div>
        <span id="lblA" style="min-width:32px">0%</span>
        <span id="cntA" style="color:var(--muted);font-size:10px"></span>
      </div>
      <div class="grade-bar-row">
        <span style="width:50px;color:var(--yellow)">B</span>
        <div class="grade-bar-bg"><div class="grade-bar-fill" id="barB" style="width:0%;background:var(--yellow)"></div></div>
        <span id="lblB" style="min-width:32px">0%</span>
        <span id="cntB" style="color:var(--muted);font-size:10px"></span>
      </div>
    </div>
  </div>

  <!-- 진입강도 해석표 -->
  <div class="legend">
    <span class="legend-title">진입강도 + 등급 조합 해석:</span>
    <span>S/A + 🚀 → <b style="color:var(--green)">적극 진입</b></span>
    <span>S/A + 🎯 → <b style="color:var(--lime)">진입 준비</b></span>
    <span>S/A + ⏳ → <b style="color:var(--yellow)">방향 대기</b></span>
    <span>B + 🚀 → <b style="color:var(--orange)">주목 (곧 등급상승 가능)</b></span>
    <span>B/C + ⏳ → <b style="color:var(--muted)">완전 대기</b></span>
    <span>💎DEEP → <b style="color:var(--purple)">선진입 (반등 신호 불필요)</b></span>
  </div>

  <!-- 탭 -->
  <div class="tabs">
    <div class="tab active" onclick="switchTab('watch')">📋 Watch (<span id="tabWatchCnt">0</span>)</div>
    <div class="tab deep-tab" onclick="switchTab('deep')">💎 DEEP (<span id="tabDeepCnt">0</span>)</div>
    <div class="tab" onclick="switchTab('active')">🔵 Active (<span id="tabActiveCnt">0</span>)</div>
    <div class="tab" onclick="switchTab('new')">🆕 신규 (<span id="tabNewCnt">0</span>)</div>
    <div class="tab" onclick="switchTab('removed')">🗑️ 만료 (<span id="tabRemovedCnt">0</span>)</div>
    <div class="tab" onclick="switchTab('history')">📜 히스토리</div>
  </div>

  <!-- Watch 탭 -->
  <div class="tab-content active" id="tabWatch">
    <div class="input-row">
      <input type="text" id="addInput" placeholder="종목 입력 (예: CHZ)"
             onkeydown="if(event.key==='Enter')addWatch()">
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

  <!-- DEEP 탭 -->
  <div class="tab-content" id="tabDeep">
    <div class="deep-info">
      <div class="deep-info-title">💎 DEEP Watch 조건: 일봉K≤5 + BTC 24h≤-2% + 종목이 BTC보다 3% 이상 선방</div>
      <div class="deep-info-row" id="deepInfoRow">
        <div class="deep-info-item">
          <span class="deep-info-label">BTC 24h</span>
          <span class="deep-info-value" id="deepBtcChange">-</span>
        </div>
        <div class="deep-info-item">
          <span class="deep-info-label">DEEP Active (누적)</span>
          <span class="deep-info-value" id="deepTotalCnt">-</span>
        </div>
        <div class="deep-info-item">
          <span class="deep-info-label">DEEP 승률</span>
          <span class="deep-info-value" id="deepWinRateInfo">-</span>
        </div>
        <div class="deep-info-item">
          <span class="deep-info-label">DEEP 평균PnL</span>
          <span class="deep-info-value" id="deepAvgPnl">-</span>
        </div>
      </div>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th><th>DEEP등급</th><th>DEEP점수</th>
            <th>일봉K</th><th>BTC 24h</th><th>종목 24h</th><th>상대강도</th>
            <th>바닥일수</th><th>거래량비율</th><th>주봉K</th>
            <th>등록가</th><th>현재가</th><th>수익률</th>
            <th>등록시간</th><th>삭제</th>
          </tr>
        </thead>
        <tbody id="deepTbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Active 탭 -->
  <div class="tab-content" id="tabActive">
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>종목</th><th>타입</th><th>등급</th><th>진입점수</th><th>진입강도</th>
            <th>진입가</th><th>현재가</th><th>수익률</th>
            <th>TP</th><th>SL</th><th>진입시간</th><th>보유시간</th><th>청산</th>
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
            <th>종목</th><th>등급</th><th>등록점수</th>
            <th>등록가</th><th>최종가</th><th>수익률</th>
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
            <th>종목</th><th>타입</th><th>결과</th><th>등급</th>
            <th>진입가</th><th>청산가</th><th>수익률</th>
            <th>보유시간</th><th>청산시각</th>
          </tr>
        </thead>
        <tbody id="historyTbody"></tbody>
      </table>
    </div>
  </div>

  <div class="timestamps">
    <span>전체스캔: <span id="tsLastScan">-</span></span>
    <span>Watch재스캔: <span id="tsWatchRescan">-</span></span>
    <span>가격체크: <span id="tsPrice">-</span></span>
    <span>Active체크: <span id="tsActive">-</span></span>
    <span>다음스캔: <span id="tsNextScan" style="color:var(--blue)">-</span></span>
    <span style="margin-left:auto">스캔횟수: <span id="tsScanCnt">0</span></span>
  </div>
</div>

<script>
// ── 포맷 유틸 ─────────────────────────────────────────────────
function fmt(v, dec=null) {
  if (v == null || v === '') return '-';
  const n = Number(v);
  if (isNaN(n)) return '-';
  if (dec !== null)
    return n.toLocaleString('ko-KR', {minimumFractionDigits:dec, maximumFractionDigits:dec});
  if (n >= 100)  return n.toLocaleString('ko-KR', {maximumFractionDigits:0});
  if (n >= 10)   return n.toLocaleString('ko-KR', {maximumFractionDigits:1});
  if (n >= 1)    return n.toLocaleString('ko-KR', {maximumFractionDigits:2});
  if (n >= 0.1)  return n.toLocaleString('ko-KR', {maximumFractionDigits:3});
  if (n >= 0.01) return n.toLocaleString('ko-KR', {maximumFractionDigits:4});
  return n.toLocaleString('ko-KR', {maximumFractionDigits:6});
}

function fmtPct(v) {
  if (v == null) return '-';
  const n = Number(v);
  if (isNaN(n)) return '-';
  const cls = n > 0 ? 'pnl-pos' : n < 0 ? 'pnl-neg' : 'pnl-neu';
  return `<span class="${cls}">${n>=0?'+':''}${n.toFixed(2)}%</span>`;
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
    if (diff < 60)    return `${Math.floor(diff)}초`;
    if (diff < 3600)  return `${Math.floor(diff/60)}분`;
    if (diff < 86400) return `${Math.floor(diff/3600)}h ${Math.floor((diff%3600)/60)}m`;
    return `${Math.floor(diff/86400)}일 ${Math.floor((diff%86400)/3600)}h`;
  } catch { return '-'; }
}

function gradeBadge(g) {
  const map = {
    'S':      'badge-red',
    'A':      'badge-orange',
    'B':      'badge-blue',
    'C':      'badge-gray',
    'DEEP-S': 'badge-deep-s',
    'DEEP-A': 'badge-deep-a',
    'DEEP-B': 'badge-deep-b',
  };
  return `<span class="badge ${map[g]||'badge-gray'}">${g||'?'}</span>`;
}

function scoreChangeBadge(s, c) {
  const diff = (c||0) - (s||0);
  if (diff > 0) return `<span class="sc-up">+${diff}↑</span>`;
  if (diff < 0) return `<span class="sc-down">${diff}↓</span>`;
  return `<span class="sc-neu">→</span>`;
}

function kCell(val) {
  if (val == null) return `<span class="k-normal">-</span>`;
  const n = Number(val);
  if (isNaN(n)) return `<span class="k-normal">-</span>`;
  if (n <= 5)  return `<span class="k-extreme">${n.toFixed(1)}</span>`;
  if (n <= 10) return `<span class="k-low">${n.toFixed(1)}</span>`;
  if (n <= 20) return `<span class="k-mid">${n.toFixed(1)}</span>`;
  return `<span class="k-normal">${n.toFixed(1)}</span>`;
}

function h4KCell(val) {
  if (val == null) return `<span class="k-normal">-</span>`;
  const n = Number(val);
  if (isNaN(n)) return `<span class="k-normal">-</span>`;
  const icon = n > 80 ? '🔥' : n > 50 ? '⚠️' : '';
  if (n <= 5)  return `<span class="k-extreme">${n.toFixed(1)}${icon}</span>`;
  if (n <= 10) return `<span class="k-low">${n.toFixed(1)}${icon}</span>`;
  if (n <= 20) return `<span class="k-mid">${n.toFixed(1)}${icon}</span>`;
  return `<span class="k-normal">${n.toFixed(1)}${icon}</span>`;
}

function dirIcon(dir, golden) {
  if (!dir || dir==='' || dir==='null') return `<span class="dir-side">-</span>`;
  if (golden) return `<span class="dir-gold">✨GX</span>`;
  const map = {
    '상승':    `<span class="dir-up">↑</span>`,
    '반등':    `<span class="dir-rise">↗</span>`,
    '횡보':    `<span class="dir-side">→</span>`,
    '하락':    `<span class="dir-down">↓</span>`,
    '알수없음':`<span class="dir-side">-</span>`,
  };
  return map[dir] || `<span class="dir-side">-</span>`;
}

function dirCell(cur, history) {
  if (!cur) return '---';
  const d = dirIcon(cur.daily_dir, false);
  const h = dirIcon(cur.h4_dir,   cur.h4_golden||false);
  const l = dirIcon(cur.h1_dir,   cur.h1_golden||false);
  return `${d}${h}${l}${sparkline(history||[])}`;
}

function esCell(es) {
  if (!es) return `<span class="es-0">⏳ 대기</span>`;
  const cls = ['es-0','es-1','es-2','es-3'][es.level||0] || 'es-0';
  return `<span class="${cls}">${es.icon||'⏳'} ${es.label||'대기'}</span>`;
}

function sparkline(history) {
  if (!history || history.length < 2) return '';
  const scores = history.slice(-10).map(h => h.score||0);
  const maxS   = Math.max(...scores, 1);
  return `<div class="spark">${scores.map(s=>{
    const h = Math.max(2, Math.round(s/maxS*16));
    return `<div class="spark-bar" style="height:${h}px"></div>`;
  }).join('')}</div>`;
}

function typeBadge(type) {
  if (type === 'deep')
    return `<span class="badge badge-deep-s">💎DEEP</span>`;
  return `<span class="badge badge-gray">일반</span>`;
}

// ── 탭 전환 ──────────────────────────────────────────────────
const TAB_NAMES = ['watch','deep','active','new','removed','history'];
const TAB_IDS   = {
  watch:'tabWatch', deep:'tabDeep', active:'tabActive',
  new:'tabNew', removed:'tabRemoved', history:'tabHistory'
};

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', TAB_NAMES[i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(TAB_IDS[name]).classList.add('active');
  if (name === 'history') loadHistory();
}

// ── UI 업데이트 ───────────────────────────────────────────────
function updateUI(state) {
  const stats = state.stats || {};
  const macro = state.macro || {};

  const statusMap = {scanning:'dot-orange', error:'dot-red', idle:'dot-green'};
  document.getElementById('statusDot').className =
    'dot ' + (statusMap[state.status] || 'dot-green');
  document.getElementById('statusTxt').textContent = state.status || 'idle';

  const watchList  = state.watch_list    || [];
  const deepList   = watchList.filter(w => w.type === 'deep');
  const normalList = watchList.filter(w => w.type !== 'deep');

  document.getElementById('cWatchCnt').textContent  = normalList.length;
  document.getElementById('cDeepCnt').textContent   = deepList.length;
  document.getElementById('cActiveCnt').textContent = state.active_count || 0;
  document.getElementById('tabWatchCnt').textContent  = normalList.length;
  document.getElementById('tabDeepCnt').textContent   = deepList.length;
  document.getElementById('tabActiveCnt').textContent = state.active_count || 0;
  document.getElementById('tabNewCnt').textContent     = (state.new_entries   || []).length;
  document.getElementById('tabRemovedCnt').textContent = (state.removed_items || []).length;

  // DEEP 카드 서브
  const deepActive = (state.active_trades||[]).filter(t=>t.type==='deep');
  document.getElementById('cDeepSub').textContent = `Active: ${deepActive.length}개`;

  // 승률
  const wr = stats.win_rate || 0;
  const wrEl = document.getElementById('cWinRate');
  wrEl.textContent = wr + '%';
  wrEl.style.color = wr>=50?'var(--green)':wr>=30?'var(--orange)':'var(--red)';
  document.getElementById('cWinSub').textContent =
    `TP:${stats.tp||0} SL:${stats.sl||0} TO:${stats.timeout||0}`;

  // DEEP 승률
  const dwr = stats.deep_win_rate || 0;
  document.getElementById('cDeepWinRate').textContent = dwr + '%';
  document.getElementById('cDeepWinSub').textContent =
    `총 ${stats.deep_total||0}건 | avg ${stats.deep_avg_pnl||0}%`;

  // 평균 PnL
  const ap = stats.avg_pnl || 0;
  const apEl = document.getElementById('cAvgPnl');
  apEl.textContent = (ap>=0?'+':'')+ap.toFixed(2)+'%';
  apEl.style.color = ap>=0?'var(--green)':'var(--red)';
  document.getElementById('cPnlSub').textContent =
    `최고:${stats.best_pnl||0}% 최저:${stats.worst_pnl||0}%`;

  // BTC
  if (macro.btc_price != null) {
    const ok = macro.macro_ok;
    const btcEl = document.getElementById('cBtcWeekly');
    btcEl.textContent = fmt(macro.btc_weekly_ma20);
    btcEl.style.color = ok ? 'var(--green)' : 'var(--red)';
    const ch24 = macro.btc_change_24h;
    document.getElementById('cBtcInfo').textContent =
      `24h: ${ch24!=null?(ch24>=0?'+':'')+ch24.toFixed(2)+'%':'-'} | 일봉MA:${fmt(macro.btc_daily_ma20)}`;
  }

  // 전환율
  const total    = stats.total     || 0;
  const activated= stats.activated || 0;
  document.getElementById('cConvRate').textContent =
    total ? Math.round(activated/total*100)+'%' : '0%';
  document.getElementById('cAvgWatch').textContent = `평균Watch: ${stats.avg_watch_hours||0}h`;

  // 승률 바
  const gs = stats.grade_stats || {};
  [['DS','DEEP-S'],['DA','DEEP-A'],['S','S'],['A','A'],['B','B']].forEach(([id,g]) => {
    const gd = gs[g] || {};
    const wr = gd.win_rate || 0;
    document.getElementById('bar'+id).style.width = wr+'%';
    document.getElementById('lbl'+id).textContent = wr+'%';
    document.getElementById('cnt'+id).textContent = gd.total?`(${gd.tp||0}/${gd.total})`:'';
  });

  // DEEP 탭 info
  const btc24 = macro.btc_change_24h;
  document.getElementById('deepBtcChange').textContent =
    btc24 != null ? (btc24>=0?'+':'')+btc24.toFixed(2)+'%' : '-';
  document.getElementById('deepBtcChange').style.color =
    btc24 != null && btc24 < -2 ? 'var(--red)' : 'var(--muted)';
  document.getElementById('deepTotalCnt').textContent  = `${stats.deep_total||0}건`;
  document.getElementById('deepWinRateInfo').textContent = `${stats.deep_win_rate||0}%`;
  document.getElementById('deepAvgPnl').textContent    = `${stats.deep_avg_pnl||0}%`;

  // 테이블
  renderWatch(normalList);
  renderDeep(deepList);
  renderActive(state.active_trades || []);
  renderNew(state.new_entries       || []);
  renderRemoved(state.removed_items || []);

  // 타임스탬프
  document.getElementById('tsLastScan').textContent    = fmtTime(state.last_scan_at);
  document.getElementById('tsWatchRescan').textContent = fmtTime(state.last_watch_rescan_at);
  document.getElementById('tsPrice').textContent       = fmtTime(state.last_price_check_at);
  document.getElementById('tsActive').textContent      = fmtTime(state.last_active_check_at);
  document.getElementById('tsScanCnt').textContent     = state.scan_count || 0;
  updateCountdown(state.next_scan_at);
}

// ── 테이블 렌더 ───────────────────────────────────────────────
function renderWatch(list) {
  const tbody = document.getElementById('watchTbody');
  if (!list.length) {
    tbody.innerHTML = '<tr><td colspan="17" style="text-align:center;color:var(--muted);padding:20px">Watch 종목 없음</td></tr>';
    return;
  }
  const sorted = [...list].sort((a,b) => (b.current?.score||0)-(a.current?.score||0));
  tbody.innerHTML = sorted.map(w => {
    const snap = w.snapshot || {};
    const cur  = w.current  || {};
    const es   = cur.entry_strength || {};
    const diff = snap.entry_price
      ? ((cur.price||0)-snap.entry_price)/snap.entry_price*100 : 0;
    return `<tr>
      <td><b>${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${snap.score||0}</td>
      <td><b>${cur.score||0}</b></td>
      <td>${scoreChangeBadge(snap.score, cur.score)}</td>
      <td>${dirCell(cur, w.score_history)}</td>
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

function renderDeep(list) {
  const tbody = document.getElementById('deepTbody');
  if (!list.length) {
    tbody.innerHTML = '<tr><td colspan="15" style="text-align:center;color:var(--muted);padding:20px">DEEP Watch 종목 없음 (BTC 하락 + K≤5 + 상대강도 조건 충족 시 자동 등록)</td></tr>';
    return;
  }
  tbody.innerHTML = list.map(w => {
    const snap = w.snapshot || {};
    const cur  = w.current  || {};
    const dd   = w.deep_data || {};
    const bd   = dd.breakdown || {};
    const diff = snap.entry_price
      ? ((cur.price||0)-snap.entry_price)/snap.entry_price*100 : 0;
    const btcCh  = bd.btc_change  != null ? bd.btc_change.toFixed(1)+'%'  : '-';
    const coinCh = bd.coin_change != null ? bd.coin_change.toFixed(1)+'%' : '-';
    const rel    = bd.relative    != null ? '+'+bd.relative.toFixed(1)+'%': '-';
    const volR   = bd.vol_ratio   != null ? bd.vol_ratio.toFixed(2)+'x'   : '-';
    const wkK    = bd.weekly_k    != null ? bd.weekly_k.toFixed(1)        : '-';
    return `<tr class="deep-row">
      <td><b style="color:var(--purple)">${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(w.grade)}</td>
      <td><span class="deep-score">${dd.score||0}</span></td>
      <td>${kCell(cur.daily_k)}</td>
      <td style="color:var(--red)">${btcCh}</td>
      <td>${coinCh}</td>
      <td><span class="deep-rel">${rel}</span></td>
      <td>${bd.bottom_days||1}일</td>
      <td><span class="deep-vol">${volR}</span></td>
      <td>${kCell(wkK==='−'?null:parseFloat(wkK))}</td>
      <td>${fmt(snap.entry_price)}</td>
      <td>${fmt(cur.price)}</td>
      <td>${fmtPct(diff)}</td>
      <td>${fmtTime(snap.registered_at)}</td>
      <td><button class="btn btn-del" onclick="removeWatch('${w.ticker}')">✕</button></td>
    </tr>`;
  }).join('');
}

function renderActive(list) {
  const tbody = document.getElementById('activeTbody');
  if (!list.length) {
    tbody.innerHTML = '<tr><td colspan="13" style="text-align:center;color:var(--muted);padding:20px">Active 종목 없음</td></tr>';
    return;
  }
  tbody.innerHTML = list.map(t => {
    const es = t.entry_strength || {};
    return `<tr class="${t.type==='deep'?'deep-row':''}">
      <td><b>${t.ticker.replace('KRW-','')}</b></td>
      <td>${typeBadge(t.type)}</td>
      <td>${gradeBadge(t.grade)}</td>
      <td>${t.entry_score||0}</td>
      <td>${t.type==='deep'?'<span style="color:var(--purple)">💎 선진입</span>':esCell(es)}</td>
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
  if (!list.length) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:20px">신규 없음</td></tr>';
    return;
  }
  tbody.innerHTML = list.map(w => {
    const snap = w.snapshot || {};
    const cur  = w.current  || {};
    const es   = cur.entry_strength || {};
    return `<tr>
      <td><b>${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${snap.score||0}</td>
      <td>${dirCell(cur,[])}</td>
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
  if (!list.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:20px">만료 없음</td></tr>';
    return;
  }
  tbody.innerHTML = list.map(w => {
    const snap = w.snapshot || {};
    const cur  = w.current  || {};
    const ep   = snap.entry_price || 0;
    const cp   = cur.price || 0;
    return `<tr>
      <td><b>${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${snap.score||0}</td>
      <td>${fmt(ep)}</td>
      <td>${fmt(cp)}</td>
      <td>${fmtPct(ep?(cp-ep)/ep*100:0)}</td>
    </tr>`;
  }).join('');
}

async function loadHistory() {
  try {
    const res  = await fetch('/api/history');
    const data = await res.json();
    const list = (data.history||[]).slice().reverse().slice(0,100);
    const tbody = document.getElementById('historyTbody');
    if (!list.length) {
      tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:20px">히스토리 없음</td></tr>';
      return;
    }
    const rMap = {
      tp:'✅TP', sl:'❌SL', timeout:'⏰TO', manual:'🖐️수동',
      activated:'🔵Active', expired:'🗑️만료', manual_remove:'🗑️삭제'
    };
    tbody.innerHTML = list.map(h => `<tr class="${h.type==='deep'?'deep-row':''}">
      <td><b>${(h.ticker||'').replace('KRW-','')}</b></td>
      <td>${typeBadge(h.type||'normal')}</td>
      <td>${rMap[h.result]||h.result}</td>
      <td>${gradeBadge(h.grade)}</td>
      <td>${fmt(h.entry_price)}</td>
      <td>${fmt(h.close_price)}</td>
      <td>${h.pnl_pct!=null?fmtPct(h.pnl_pct):'-'}</td>
      <td>${h.hours_held!=null?h.hours_held.toFixed(1)+'h':'-'}</td>
      <td>${fmtTime(h.closed_at||h.activated_at)}</td>
    </tr>`).join('');
  } catch(e) { console.error(e); }
}

// ── 카운트다운 ────────────────────────────────────────────────
let _nextAt = null;
function updateCountdown(iso) { _nextAt = iso; }
setInterval(() => {
  if (!_nextAt) return;
  const diff = Math.max(0, Math.floor((new Date(_nextAt)-Date.now())/1000));
  const m = Math.floor(diff/60), s = diff%60;
  document.getElementById('tsNextScan').textContent =
    diff>0 ? `${m}분 ${String(s).padStart(2,'0')}초 후` : '스캔 중...';
}, 1000);

// ── API 액션 ─────────────────────────────────────────────────
function showMsg(txt, ok=true) {
  const el = document.getElementById('msg');
  el.textContent = txt;
  el.className   = ok?'msg-ok':'msg-err';
  el.style.display = 'block';
  setTimeout(()=>{ el.style.display='none'; }, 3000);
}

async function manualScan() {
  try {
    const r = await fetch('/api/scan',{method:'POST'});
    const d = await r.json();
    showMsg(d.msg||'스캔 트리거 완료');
  } catch { showMsg('오류 발생',false); }
}

async function addWatch() {
  const ticker = document.getElementById('addInput').value.trim();
  if (!ticker) return;
  try {
    const r = await fetch('/api/watch/add',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ticker})
    });
    const d = await r.json();
    showMsg(d.msg, d.ok);
    if (d.ok) { document.getElementById('addInput').value=''; poll(); }
  } catch { showMsg('오류 발생',false); }
}

async function removeWatch(ticker) {
  if (!confirm(`${ticker.replace('KRW-','')} 삭제하시겠습니까?`)) return;
  try {
    const r = await fetch('/api/watch/remove',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ticker})
    });
    const d = await r.json();
    showMsg(d.msg, d.ok);
    if (d.ok) poll();
  } catch { showMsg('오류 발생',false); }
}

async function closeTrade(ticker) {
  if (!confirm(`${ticker.replace('KRW-','')} 수동 청산하시겠습니까?`)) return;
  try {
    const r = await fetch('/api/trade/close',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ticker})
    });
    const d = await r.json();
    showMsg(d.msg, d.ok);
    if (d.ok) poll();
  } catch { showMsg('오류 발생',false); }
}

async function poll() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();
    updateUI(d);
  } catch(e) { console.error('poll 오류:',e); }
}

poll();
setInterval(poll, 15000);
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML,
        tp_pct=scanner.TRADE_TP_PCT,
        sl_pct=scanner.TRADE_SL_PCT)

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
    ticker = data.get('ticker','').strip()
    if not ticker:
        return jsonify({'ok':False,'msg':'종목명 필요'}), 400
    return jsonify(scanner.add_manual_watch(ticker))

@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data   = request.get_json() or {}
    ticker = data.get('ticker','').strip()
    if not ticker:
        return jsonify({'ok':False,'msg':'종목명 필요'}), 400
    return jsonify(scanner.remove_watch(ticker))

@app.route('/api/trade/close', methods=['POST'])
def api_trade_close():
    data   = request.get_json() or {}
    ticker = data.get('ticker','').strip()
    if not ticker:
        return jsonify({'ok':False,'msg':'종목명 필요'}), 400
    return jsonify(scanner.manual_close_trade(ticker))

@app.route('/api/history')
def api_history():
    return jsonify({'history': scanner.load_trade_history()})

@app.route('/health')
def health():
    return jsonify({'status':'ok','version':scanner.VERSION})

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
    port = int(os.getenv('PORT','8080'))
    app.run(host='0.0.0.0', port=port, debug=False)
