"""
dashboard.py  v3.0.5
─────────────────────────────────────────────
변경사항:
  v3.0.1  BTC KRW/USD, 일봉/주봉 MA20, 이벤트 패널, 상태바
  v3.0.2  C등급 차단, 등록가/현재가+변동률
  v3.0.3  ⚠️타이밍 / 🔴과매수 배지, B등급 강화
  v3.0.4  등록가 대비 변동률 표시 개선
  v3.0.5  일봉 단기/중기 K/D 컬럼 ↑↓ 방향 표시
          K>D → 초록 ↑ / K<D → 빨강 ↓
─────────────────────────────────────────────
"""

DASHBOARD_VERSION = 'v3.0.5'

import threading, logging
from flask import Flask, jsonify, request, render_template_string
import scanner as sc

log = logging.getLogger(__name__)
app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MTF Scanner {{ dv }}</title>
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
  --orange:  #ffa657;
  --s-color: #ffd700;
  --a-color: #58a6ff;
  --b-color: #3fb950;
  --x-color: #f85149;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Segoe UI', sans-serif; font-size: 13px;
}

/* 헤더 */
.header {
  background: var(--card); border-bottom: 1px solid var(--border);
  padding: 12px 20px;
  display: flex; align-items: center;
  justify-content: space-between; flex-wrap: wrap; gap: 8px;
}
.header-left { display: flex; align-items: center; gap: 12px; }
.logo { font-size: 18px; font-weight: 700; color: var(--blue); }
.version-badge {
  background: #1f2937; border: 1px solid var(--border);
  border-radius: 12px; padding: 3px 10px;
  font-size: 11px; color: var(--sub);
  cursor: pointer; position: relative;
}
.version-badge:hover .vtip { display: block; }
.vtip {
  display: none; position: absolute; top: 28px; left: 0;
  background: #1f2937; border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 14px;
  min-width: 310px; z-index: 100;
  font-size: 11px; line-height: 1.9;
  color: var(--text); white-space: nowrap;
}

/* BTC 블록 */
.btc-block { display: flex; flex-direction: column; gap: 4px; align-items: flex-end; }
.btc-price-row { display: flex; align-items: center; gap: 10px; }
.btc-price { font-size: 16px; font-weight: 700; color: var(--orange); }
.btc-usd   { font-size: 12px; color: var(--sub); }
.btc-ma-row { display: flex; gap: 18px; font-size: 11px; }
.btc-ma-item { display: flex; align-items: center; gap: 5px; }
.ma-label { color: var(--sub); }
.ma-value { color: var(--text); }
.ma-above { color: var(--green); font-weight: 600; }
.ma-below { color: var(--red);   font-weight: 600; }
.ma-pct   { font-size: 10px; }

/* 상태바 */
.status-bar {
  background: #0a0e14; border-bottom: 1px solid var(--border);
  padding: 5px 20px;
  display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
}
.si { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--sub); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--sub); flex-shrink: 0; }
.dot.on  { background: var(--green); animation: blink 1.2s infinite; }
.dot.off { background: var(--sub); }
.st.on  { color: var(--green); font-weight: 600; }
.st.off { color: var(--sub); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.last-info { margin-left: auto; font-size: 11px; color: var(--sub); }

/* 메인 */
.main { padding: 14px 20px; }

/* 통계 카드 */
.stats-row { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
.stat-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px;
  min-width: 100px; flex: 1;
}
.stat-label { font-size: 11px; color: var(--sub); margin-bottom: 3px; }
.stat-value { font-size: 22px; font-weight: 700; }
.stat-sub   { font-size: 10px; color: var(--sub); margin-top: 2px; }

/* 탭 */
.tabs { display: flex; gap: 2px; margin-bottom: 10px; border-bottom: 1px solid var(--border); }
.tab-btn {
  background: none; border: none; border-bottom: 2px solid transparent;
  color: var(--sub); padding: 8px 16px; cursor: pointer;
  font-size: 13px; font-weight: 500; transition: all .2s;
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
  white-space: nowrap; font-weight: 500; position: sticky; top: 0;
}
td {
  padding: 7px 10px; border-bottom: 1px solid #21262d;
  vertical-align: middle; white-space: nowrap;
}
tr:hover td { background: #1c2128; }

/* 배지 */
.badge { display: inline-block; padding: 2px 7px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-S { background:#3d2e00; color:var(--s-color); border:1px solid var(--s-color); }
.badge-A { background:#0d2137; color:var(--a-color); border:1px solid var(--a-color); }
.badge-B { background:#0d2e15; color:var(--b-color); border:1px solid var(--b-color); }
.badge-X { background:#2d1010; color:var(--x-color); border:1px solid var(--x-color); }
.badge-gc   { background:#0d2e15; color:var(--green);  font-size:10px; padding:1px 5px; }
.badge-warn { background:#2e2200; color:var(--yellow); font-size:10px; padding:1px 5px; }
.badge-over { background:#2d1010; color:var(--red);    font-size:10px; padding:1px 5px; }
.badge-vol  { background:#1a1f2e; color:var(--blue);   font-size:10px; padding:1px 5px; }

.score-bar {
  display: inline-block; height: 4px; border-radius: 2px;
  vertical-align: middle; margin-left: 4px;
}

/* 가격 셀 */
.pc { line-height: 1.7; }
.pc-reg { font-size: 11px; color: var(--sub); }
.pc-cur { font-weight: 500; }
.pc-pct { font-size: 11px; margin-left: 4px; }
.up   { color: var(--green); }
.down { color: var(--red); }
.flat { color: var(--sub); }

/* KD 셀 */
.kd  { font-size: 11px; line-height: 1.6; }
.kos { color: var(--orange); font-weight: 600; }
.kob { color: var(--red);    font-weight: 600; }
.kup { color: var(--green);  font-weight: 600; }
.kdn { color: var(--red);    font-weight: 600; }

/* K방향 화살표 */
.k-rising  { color: var(--green); font-size: 11px; font-weight: 700; }
.k-falling { color: var(--red);   font-size: 11px; font-weight: 700; }

/* 이벤트 패널 */
.evt-panel {
  position: fixed; bottom: 20px; right: 20px;
  width: 300px; max-height: 380px;
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; overflow: hidden; z-index: 200;
  box-shadow: 0 8px 24px rgba(0,0,0,.5);
}
.evt-hdr {
  padding: 9px 14px; background: #1c2128;
  border-bottom: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center;
  font-size: 12px; font-weight: 600;
}
.evt-toggle { background: none; border: none; color: var(--sub); cursor: pointer; font-size: 14px; }
.evt-list   { overflow-y: auto; max-height: 320px; padding: 6px 0; }
.evt-item {
  padding: 5px 14px; display: flex; gap: 8px;
  border-bottom: 1px solid #21262d; font-size: 11px;
}
.evt-time { color: var(--sub); min-width: 48px; flex-shrink: 0; }
.evt-msg  { color: var(--text); line-height: 1.4; }
.evt-item.watch  { border-left: 3px solid var(--blue); }
.evt-item.active { border-left: 3px solid var(--green); }
.evt-item.close  { border-left: 3px solid var(--red); }
.evt-item.deep   { border-left: 3px solid var(--orange); }
.evt-item.system { border-left: 3px solid var(--sub); }
.evt-item.error  { border-left: 3px solid var(--red); }

/* 토스트 */
.toast-wrap { position: fixed; top: 60px; right: 20px; z-index: 300; display: flex; flex-direction: column; gap: 8px; }
.toast {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 16px;
  font-size: 12px; color: var(--text);
  animation: slideIn .3s ease;
  box-shadow: 0 4px 12px rgba(0,0,0,.4);
}
@keyframes slideIn { from{transform:translateX(100%);opacity:0} to{transform:translateX(0);opacity:1} }

/* 버튼 */
.btn { border: none; border-radius: 6px; padding: 4px 10px; font-size: 11px; cursor: pointer; font-weight: 500; }
.btn-green { background:#0d2e15; color:var(--green); }
.btn-red   { background:#2d1010; color:var(--red); }
.btn:hover    { opacity: .8; }
.btn:disabled { opacity: .4; cursor: not-allowed; }
.scan-btn {
  background: var(--blue); color: #fff;
  border: none; border-radius: 8px;
  padding: 7px 16px; font-size: 12px;
  cursor: pointer; font-weight: 600;
}
.scan-btn:hover    { background: #4493f8; }
.scan-btn:disabled { opacity: .5; cursor: not-allowed; }
.empty-msg { text-align: center; color: var(--sub); padding: 40px; font-size: 13px; }

@media (max-width: 768px) {
  .evt-panel  { width: 260px; }
  .btc-ma-row { flex-direction: column; gap: 4px; }
}
</style>
</head>
<body>

<!-- 헤더 -->
<div class="header">
  <div class="header-left">
    <span class="logo">📊 MTF Scanner</span>
    <div class="version-badge">
      {{ dv }}
      <div class="vtip">
        <b>📋 변경사항</b><br>
        v3.0.5 일봉 단기/중기 K방향 점수 반영 (+5/-5)<br>
        v3.0.5 K>D → 초록↑ / K&lt;D → 빨강↓ 시각 표시<br>
        v3.0.4 등록가 대비 변동률 표시 (entry_price 고정)<br>
        v3.0.3 ⚠️타이밍(4hK≥70) / 🔴과매수(1hK≥80)<br>
        v3.0.3 B등급: score≥55 + aligned≥2<br>
        v3.0.2 C등급 Watch 차단<br>
        v3.0.1 BTC 일봉/주봉 MA20 KRW/USD<br>
      </div>
    </div>
  </div>
  <div class="btc-block">
    <div class="btc-price-row">
      <span class="btc-price" id="btcPrice">–</span>
      <span class="btc-usd"   id="btcUsd">–</span>
      <button class="scan-btn" onclick="triggerScan()">🔄 즉시스캔</button>
    </div>
    <div class="btc-ma-row">
      <div class="btc-ma-item">
        <span class="ma-label">일봉MA20</span>
        <span class="ma-value" id="dMaVal">–</span>
        <span id="dMaSig">–</span>
        <span class="ma-pct" id="dMaPct"></span>
      </div>
      <div class="btc-ma-item">
        <span class="ma-label">주봉MA20</span>
        <span class="ma-value" id="wMaVal">–</span>
        <span id="wMaSig">–</span>
        <span class="ma-pct" id="wMaPct"></span>
      </div>
    </div>
  </div>
</div>

<!-- 상태바 -->
<div class="status-bar">
  <div class="si"><div class="dot" id="d0"></div><span class="st" id="t0">스캐너 대기</span></div>
  <div class="si"><div class="dot" id="d1"></div><span class="st" id="t1">Watch재스캔 대기</span></div>
  <div class="si"><div class="dot" id="d2"></div><span class="st" id="t2">가격체크 대기</span></div>
  <div class="si"><div class="dot" id="d3"></div><span class="st" id="t3">DEEP 대기</span></div>
  <div class="last-info">
    마지막: <span id="lastScan">–</span> |
    다음: <span id="nextScan">–</span> |
    <span id="countdown" style="color:var(--yellow)"></span>
  </div>
</div>

<!-- 메인 -->
<div class="main">
  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-label">📋 Watch</div>
      <div class="stat-value" id="sWatch">–</div>
      <div class="stat-sub">감시 종목</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">✅ Active</div>
      <div class="stat-value" id="sActive">–</div>
      <div class="stat-sub">진입 종목</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">🔥 DEEP</div>
      <div class="stat-value" id="sDeep">–</div>
      <div class="stat-sub">상대강도</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">📡 스캔</div>
      <div class="stat-value" id="sScan">–</div>
      <div class="stat-sub" id="sScanSub">총 스캔 횟수</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">🏆 승률</div>
      <div class="stat-value" id="sWin">–</div>
      <div class="stat-sub" id="sWinSub">–</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">💰 누적PnL</div>
      <div class="stat-value" id="sPnl">–</div>
      <div class="stat-sub">종료 트레이드</div>
    </div>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="sw('watch',this)">📋 Watch</button>
    <button class="tab-btn"        onclick="sw('active',this)">✅ Active</button>
    <button class="tab-btn"        onclick="sw('deep',this)">🔥 DEEP</button>
    <button class="tab-btn"        onclick="sw('history',this)">📊 History</button>
  </div>

  <!-- Watch 탭 -->
  <div id="tab-watch" class="tab-content active">
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>종목</th><th>등급</th><th>점수</th>
          <th>등록가</th><th>현재가 (등록가대비)</th>
          <th>일봉 장기 K/D</th>
          <th>일봉 중기 K/D ↕</th>
          <th>일봉 단기 K/D ↕</th>
          <th>4h K/D</th><th>1h K/D</th><th>GC</th>
          <th>거래량</th><th>바닥일수</th><th>등록일</th><th>만료</th><th>관리</th>
        </tr></thead>
        <tbody id="watchBody">
          <tr><td colspan="16" class="empty-msg">스캔 대기 중...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Active 탭 -->
  <div id="tab-active" class="tab-content">
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>종목</th><th>등급</th><th>점수</th>
          <th>진입가</th><th>현재가 (진입가대비)</th>
          <th>PnL</th><th>TP</th><th>SL</th>
          <th>거래량</th><th>진입일</th><th>관리</th>
        </tr></thead>
        <tbody id="activeBody">
          <tr><td colspan="11" class="empty-msg">진입 종목 없음</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- DEEP 탭 -->
  <div id="tab-deep" class="tab-content">
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>종목</th><th>상대강도</th><th>등급</th><th>현재가</th><th>스캔시간</th>
        </tr></thead>
        <tbody id="deepBody">
          <tr><td colspan="5" class="empty-msg">BTC 급락 시 자동 스캔</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- History 탭 -->
  <div id="tab-history" class="tab-content">
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>종목</th><th>등급</th><th>진입가</th><th>종료가</th>
          <th>PnL</th><th>종료사유</th><th>진입일</th><th>종료일</th>
        </tr></thead>
        <tbody id="historyBody">
          <tr><td colspan="8" class="empty-msg">종료된 트레이드 없음</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- 이벤트 패널 -->
<div class="evt-panel">
  <div class="evt-hdr">
    <span>📡 실시간 이벤트</span>
    <button class="evt-toggle" onclick="toggleEvt()">▼</button>
  </div>
  <div class="evt-list" id="evtList">
    <div class="empty-msg" style="padding:16px">이벤트 없음</div>
  </div>
</div>

<div class="toast-wrap" id="toastWrap"></div>

<script>
let _evtOpen = true;
let _cdSec   = 0;
let _cdTimer = null;

function sw(name, btn) {
  document.querySelectorAll('.tab-content').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(e => e.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}
function toggleEvt() {
  const l = document.getElementById('evtList');
  const b = document.querySelector('.evt-toggle');
  _evtOpen = !_evtOpen;
  l.style.display = _evtOpen ? 'block' : 'none';
  b.textContent   = _evtOpen ? '▼' : '▲';
}
function toast(msg, ms=3500) {
  const c = document.getElementById('toastWrap');
  const t = document.createElement('div');
  t.className = 'toast'; t.textContent = msg;
  c.appendChild(t); setTimeout(()=>t.remove(), ms);
}

// 숫자 포맷
function fp(p) {
  if (p == null) return '–';
  if (p >= 100000) return Number(p).toLocaleString('ko-KR',{maximumFractionDigits:0});
  if (p >= 1)      return Number(p).toFixed(2);
  if (p >= 0.01)   return Number(p).toFixed(4);
  return Number(p).toFixed(6);
}
function fusd(krw, rate) {
  if (!krw||!rate) return '';
  return '($'+Number(krw/rate).toLocaleString('en-US',{maximumFractionDigits:0})+')';
}
function fPct(v) {
  if (v==null) return '–';
  return (v>=0?'+':'')+Number(v).toFixed(2)+'%';
}

// 배지
function gb(g) {
  if (!g||g==='-') return '<span style="color:var(--sub)">–</span>';
  const m={S:'badge-S',A:'badge-A',B:'badge-B',X:'badge-X'};
  return `<span class="badge ${m[g]||'badge-B'}">${g}</span>`;
}
function sb(score, grade) {
  const c={S:'var(--s-color)',A:'var(--a-color)',B:'var(--b-color)',X:'var(--x-color)'};
  const col=c[grade]||'var(--sub)';
  return `<span style="color:${col};font-weight:600">${score}</span>`
       + `<span class="score-bar" style="width:${Math.min(score,100)*.5}px;background:${col}"></span>`;
}

// ── KD 셀 (방향 표시 포함) ───────────────────────────
function kdCell(k, d, sig, showDirection=false) {
  const kc  = k <= 20 ? 'kos' : (k >= 80 ? 'kob' : '');
  // GC/DC 화살표
  const dir = sig==='BUY_OK' ? '↑' : (sig==='BUY_NO' ? '↓' : '');
  const dc  = sig==='BUY_OK' ? 'kup' : (sig==='BUY_NO' ? 'kdn' : '');
  // K방향 화살표 (K>D 여부)
  let dirArrow = '';
  if (showDirection && k != null && d != null) {
    if (k > d) {
      dirArrow = '<span class="k-rising"> ↑</span>';
    } else if (k < d) {
      dirArrow = '<span class="k-falling"> ↓</span>';
    } else {
      dirArrow = '<span style="color:var(--sub)"> –</span>';
    }
  }
  return `<div class="kd">
    <span class="${kc}">${k!=null?k.toFixed(1):'–'}</span>
    <span style="color:var(--sub)">/${d!=null?d.toFixed(1):'–'}</span>
    <span class="${dc}">${dir}</span>${dirArrow}
  </div>`;
}

function gcb(item) {
  const p=[];
  if (item.daily_gc) p.push('<span class="badge badge-gc">일봉✨</span>');
  if (item.h4_gc)    p.push('<span class="badge badge-gc">4h✨</span>');
  if (item.h1_gc)    p.push('<span class="badge badge-gc">1h✨</span>');
  return p.join(' ')||'<span style="color:var(--sub)">–</span>';
}
function wb(item) {
  if (item.overbought_warning) return '<span class="badge badge-over">🔴과매수</span>';
  if (item.timing_warning)     return '<span class="badge badge-warn">⚠️타이밍</span>';
  return '';
}
function vb(v) {
  if (v==null) return '–';
  const ico = v>=2?'🔥':(v>=1?'📈':'');
  return `<span class="badge badge-vol">${ico}${v.toFixed(1)}x</span>`;
}

// 가격 셀
function regCell(p) {
  return `<div style="color:var(--sub);font-size:12px">${fp(p)}</div>`;
}
function curCell(cur, pct) {
  if (!cur) return '–';
  pct = pct||0;
  const pc = pct>0.05?'up':(pct<-0.05?'down':'flat');
  const ar = pct>0.05?'▲':(pct<-0.05?'▼':'');
  const sg = pct>=0?'+':'';
  return `<div class="pc">
    <span class="pc-cur">${fp(cur)}</span>
    <span class="pc-pct ${pc}">${ar}${sg}${pct.toFixed(2)}%</span>
  </div>`;
}

// BTC
function updateBtc(s) {
  const rate = s.usdt_rate||1350;
  if (s.btc_price) {
    document.getElementById('btcPrice').textContent =
      '₩'+Number(s.btc_price).toLocaleString('ko-KR');
    document.getElementById('btcUsd').textContent = fusd(s.btc_price,rate);
  }
  function setMa(vi,si,pi,ma,sig,price) {
    if (!ma) return;
    document.getElementById(vi).textContent =
      '₩'+Number(ma).toLocaleString('ko-KR')+' '+fusd(ma,rate);
    const el=document.getElementById(si);
    const ia=sig==='ABOVE';
    el.className=ia?'ma-above':'ma-below';
    el.textContent=ia?'▲':'▼';
    if (price) {
      const pct=((price-ma)/ma*100).toFixed(2);
      const pe=document.getElementById(pi);
      pe.textContent=(pct>=0?'+':'')+pct+'%';
      pe.style.color=pct>=0?'var(--green)':'var(--red)';
    }
  }
  setMa('dMaVal','dMaSig','dMaPct',s.btc_daily_ma20, s.btc_daily_signal, s.btc_price);
  setMa('wMaVal','wMaSig','wMaPct',s.btc_weekly_ma20,s.btc_weekly_signal,s.btc_price);
}

// 상태바
function updateStatus(s) {
  function set(di,ti,on,label) {
    document.getElementById(di).className='dot '+(on?'on':'off');
    const t=document.getElementById(ti);
    t.className='st '+(on?'on':'off');
    t.textContent=label;
  }
  set('d0','t0',s.running,           s.running?'⏳ 전체 스캐닝 중...':'✅ 스캐너 대기');
  set('d1','t1',s.watch_rescanning,  s.watch_rescanning?'⏳ Watch 재스캔 중...':'Watch 재스캔 대기');
  set('d2','t2',s.price_checking,    s.price_checking?'⏳ 가격 체크 중...':'가격 체크 대기');
  set('d3','t3',s.deep_scanning,     s.deep_scanning?'🔥 DEEP 스캔 중...':'DEEP 대기');
  document.getElementById('lastScan').textContent=s.last_scan||'–';
  document.getElementById('nextScan').textContent=s.next_scan||'–';
  if (s.next_scan) {
    const diff=Math.max(0,Math.floor(
      (new Date(s.next_scan.replace(' ','T'))-new Date())/1000));
    startCd(diff);
  }
}

function startCd(sec) {
  if (_cdTimer) clearInterval(_cdTimer);
  _cdSec=sec;
  function tick() {
    const el=document.getElementById('countdown');
    if (!el) return;
    if (_cdSec<=0){el.textContent='⏳ 스캔 예정';return;}
    const m=Math.floor(_cdSec/60), s=_cdSec%60;
    el.textContent=`다음 ${m}분 ${String(s).padStart(2,'0')}초`;
    _cdSec--;
  }
  tick(); _cdTimer=setInterval(tick,1000);
}

function updateStats(d) {
  const s=d.state||{};
  document.getElementById('sWatch').textContent  = d.watch?.length??0;
  document.getElementById('sActive').textContent = d.active?.length??0;
  document.getElementById('sDeep').textContent   = d.deep?.length??0;
  document.getElementById('sScan').textContent   = s.scan_count??0;
  const sub=document.getElementById('sScanSub');
  sub.textContent=s.running?'⏳ 스캔 중...':'총 스캔 횟수';
  sub.style.color=s.running?'var(--yellow)':'var(--sub)';
  const total=s.total_trades||0, wins=s.win_trades||0;
  const wr=total>0?Math.round(wins/total*100):0;
  document.getElementById('sWin').textContent    = total>0?wr+'%':'–';
  document.getElementById('sWinSub').textContent = `${wins}/${total}건`;
  const pnl=s.total_pnl||0;
  const pe=document.getElementById('sPnl');
  pe.textContent=total>0?(pnl>=0?'+':'')+pnl.toFixed(2)+'%':'–';
  pe.style.color=pnl>=0?'var(--green)':'var(--red)';
}

// ── Watch 테이블 ──────────────────────────────────────
function renderWatch(watch) {
  const tb=document.getElementById('watchBody');
  if (!watch||!watch.length) {
    tb.innerHTML='<tr><td colspan="16" class="empty-msg">Watch 종목 없음 (B등급 이상)</td></tr>';
    return;
  }
  const sorted=[...watch].sort((a,b)=>(b.score||0)-(a.score||0));
  tb.innerHTML=sorted.map(w=>{
    const sym=w.market.replace('KRW-','');
    const warn=wb(w);
    const dLSig=w.d_long_k<=20?'BUY_OK':'NEUTRAL';
    return `<tr>
      <td><b>${sym}</b>${warn?'<br>'+warn:''}</td>
      <td>${gb(w.grade)}</td>
      <td>${sb(w.score||0,w.grade)}</td>
      <td>${regCell(w.entry_price)}</td>
      <td>${curCell(w.current_price,w.price_change)}</td>
      <td>${kdCell(w.d_long_k,  w.d_long_d,  dLSig,        false)}</td>
      <td>${kdCell(w.d_mid_k,   w.d_mid_d,   'NEUTRAL',    true)}</td>
      <td>${kdCell(w.d_short_k, w.d_short_d, 'NEUTRAL',    true)}</td>
      <td>${kdCell(w.h4_k_val,  w.h4_d_val,  w.h4_gc?'BUY_OK':'NEUTRAL', false)}</td>
      <td>${kdCell(w.h1_k_val,  w.h1_d_val,  w.h1_gc?'BUY_OK':'NEUTRAL', false)}</td>
      <td>${gcb(w)}</td>
      <td>${vb(w.volume_ratio)}</td>
      <td>${w.bottom_days??0}일</td>
      <td>${(w.registered_at||'').substring(5,16)}</td>
      <td>${w.expire_at||'–'}</td>
      <td>
        <button class="btn btn-green" onclick="doActivate('${w.market}')">진입</button>
        <button class="btn btn-red"   onclick="doRemove('${w.market}')">제거</button>
      </td>
    </tr>`;
  }).join('');
}

// Active
function renderActive(active) {
  const tb=document.getElementById('activeBody');
  if (!active||!active.length) {
    tb.innerHTML='<tr><td colspan="11" class="empty-msg">진입 종목 없음</td></tr>';
    return;
  }
  tb.innerHTML=active.map(a=>{
    const sym=a.market.replace('KRW-','');
    const cur=a.current_price||a.entry_price;
    const pnl=(cur-a.entry_price)/a.entry_price*100;
    return `<tr>
      <td><b>${sym}</b></td>
      <td>${gb(a.grade)}</td>
      <td>${sb(a.score||0,a.grade)}</td>
      <td>${regCell(a.entry_price)}</td>
      <td>${curCell(cur,pnl)}</td>
      <td class="${pnl>=0?'up':'down'}" style="font-weight:600">${fPct(pnl)}</td>
      <td style="color:var(--green)">+${a.tp_pct}% (${fp(a.tp_price)})</td>
      <td style="color:var(--red)">-${a.sl_pct}% (${fp(a.sl_price)})</td>
      <td>${vb(a.volume_ratio)}</td>
      <td>${(a.entered_at||'').substring(5,16)}</td>
      <td><button class="btn btn-red" onclick="doClose('${a.market}')">종료</button></td>
    </tr>`;
  }).join('');
}

// DEEP
function renderDeep(deep) {
  const tb=document.getElementById('deepBody');
  if (!deep||!deep.length) {
    tb.innerHTML='<tr><td colspan="5" class="empty-msg">BTC 급락(-1% 이상) 시 자동 스캔</td></tr>';
    return;
  }
  tb.innerHTML=deep.map(d=>{
    const sym=d.market.replace('KRW-','');
    const c=d.rs>=2?'var(--green)':(d.rs>=0?'var(--yellow)':'var(--red)');
    return `<tr>
      <td><b>${sym}</b></td>
      <td style="color:${c};font-weight:600">${d.rs>=0?'+':''}${d.rs}%</td>
      <td>${gb(d.rs_grade)}</td>
      <td>${fp(d.price)}</td>
      <td>${(d.scanned_at||'').substring(5,16)}</td>
    </tr>`;
  }).join('');
}

// History
function renderHistory(history) {
  const tb=document.getElementById('historyBody');
  if (!history||!history.length) {
    tb.innerHTML='<tr><td colspan="8" class="empty-msg">종료된 트레이드 없음</td></tr>';
    return;
  }
  tb.innerHTML=[...history].reverse().map(h=>{
    const sym=(h.market||'').replace('KRW-','');
    const pnl=h.pnl||0;
    const rc=h.reason==='TP'?'var(--green)':h.reason==='SL'?'var(--red)':'var(--sub)';
    return `<tr>
      <td><b>${sym}</b></td>
      <td>${gb(h.grade)}</td>
      <td>${fp(h.entry_price)}</td>
      <td>${fp(h.close_price)}</td>
      <td class="${pnl>=0?'up':'down'}" style="font-weight:600">${fPct(pnl)}</td>
      <td style="color:${rc};font-weight:600">${h.reason||'–'}</td>
      <td>${(h.entered_at||'').substring(5,16)}</td>
      <td>${(h.closed_at||'').substring(5,16)}</td>
    </tr>`;
  }).join('');
}

// 이벤트
async function fetchEvents() {
  try {
    const r=await fetch('/api/events');
    const d=await r.json();
    const el=document.getElementById('evtList');
    if (!d.events||!d.events.length) {
      el.innerHTML='<div class="empty-msg" style="padding:16px">이벤트 없음</div>';
      return;
    }
    el.innerHTML=[...d.events].reverse().map(e=>{
      if (!e||typeof e!=='object') return '';
      return `<div class="evt-item ${e.type||'system'}">
        <span class="evt-time">${e.time||''}</span>
        <span class="evt-msg">${e.message||''}</span>
      </div>`;
    }).join('');
  } catch(e){}
}

async function fetchState() {
  try {
    const r=await fetch('/api/state');
    const d=await r.json();
    if (d.error){console.error('/api/state:',d.error);return;}
    updateBtc(d.state||{});
    updateStatus(d.state||{});
    updateStats(d);
    renderWatch(d.watch);
    renderActive(d.active);
    renderDeep(d.deep);
    renderHistory(d.history);
  } catch(e){console.error('fetchState:',e);}
}

async function triggerScan() {
  const btn=document.querySelector('.scan-btn');
  btn.textContent='⏳ 스캔 중...'; btn.disabled=true;
  toast('🔄 스캔 요청 중...');
  try {
    const r=await fetch('/api/scan',{method:'POST'});
    const d=await r.json();
    if (d.success) {
      toast('✅ 스캔 시작! 결과 업데이트 중...');
      let cnt=0;
      const p=setInterval(()=>{
        fetchState(); fetchEvents(); cnt++;
        if (cnt>=8){clearInterval(p);btn.textContent='🔄 즉시스캔';btn.disabled=false;}
      },5000);
    } else {
      toast('❌ '+d.message);
      btn.textContent='🔄 즉시스캔'; btn.disabled=false;
    }
  } catch(e){
    toast('❌ 요청 실패');
    btn.textContent='🔄 즉시스캔'; btn.disabled=false;
  }
}

async function doActivate(market) {
  if (!confirm(market.replace('KRW-','')+' 즉시 진입하시겠습니까?')) return;
  const r=await fetch('/api/watch/activate',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({market})
  });
  const d=await r.json();
  toast(d.success?'✅ '+d.message:'❌ '+d.message);
  fetchState(); fetchEvents();
}

async function doRemove(market) {
  if (!confirm(market.replace('KRW-','')+' Watch에서 제거하시겠습니까?')) return;
  const r=await fetch('/api/watch/remove',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({market})
  });
  const d=await r.json();
  toast(d.success?'🗑️ '+d.message:'❌ '+d.message);
  fetchState();
}

async function doClose(market) {
  if (!confirm(market.replace('KRW-','')+' 포지션을 종료하시겠습니까?')) return;
  const r=await fetch('/api/active/close',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({market})
  });
  const d=await r.json();
  toast(d.success?'🔴 '+d.message:'❌ '+d.message);
  fetchState(); fetchEvents();
}

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
    return render_template_string(TEMPLATE, dv=DASHBOARD_VERSION)

@app.route('/api/version')
def api_version():
    return jsonify({
        'dashboard': DASHBOARD_VERSION,
        'scanner':   sc.VERSION,
        'mtf_setup': sc.MTF_VERSION,
    })

@app.route('/api/state')
def api_state():
    try:
        state   = sc.get_scanner_state()
        watch   = sc._load_json(sc.WATCH_FILE,   [])
        active  = sc._load_json(sc.ACTIVE_FILE,  [])
        deep    = sc._load_json(sc.DEEP_FILE,    [])
        history = sc._load_json(sc.HISTORY_FILE, [])
        total_trades = len(history)
        win_trades   = len([h for h in history if h.get('pnl', 0) > 0])
        total_pnl    = round(sum(h.get('pnl', 0) for h in history), 2)
        return jsonify({
            'state': {**state,
                      'total_trades': total_trades,
                      'win_trades':   win_trades,
                      'total_pnl':    total_pnl},
            'watch':     watch,
            'active':    active,
            'deep':      deep,
            'history':   history[-50:],
            'next_scan': state.get('next_scan'),
        })
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log.error(f'/api/state error: {err}')
        return jsonify({'error': str(e), 'trace': err}), 500

@app.route('/api/events')
def api_events():
    events = sc._load_json(sc.EVENT_FILE, [])
    return jsonify({'events': events if isinstance(events, list) else []})

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

if __name__ == '__main__':
    print(f'✅ Dashboard {DASHBOARD_VERSION} + Scanner {sc.VERSION} 시작')
    print(f'   MTF Setup  : {sc.MTF_VERSION}')
    print(f'   v3.0.5: 일봉 단기/중기 K방향 점수+시각 반영')

    loops = [
        ('scanner',        sc.scanner_loop),
        ('watch_rescan',   sc.watch_rescan_loop),
        ('price_check',    sc.price_check_loop),
        ('active_monitor', sc.active_monitor_loop),
        ('deep_scan',      sc.deep_scan_loop),
        ('daily_summary',  sc.daily_summary_loop),
    ]
    for name, fn in loops:
        threading.Thread(target=fn, name=name, daemon=True).start()
    print(f'   루프 {len(loops)}개: {" / ".join(n for n,_ in loops)}')
    print(f'🚀 http://0.0.0.0:{sc.PORT}')
    app.run(host='0.0.0.0', port=sc.PORT, debug=False)
