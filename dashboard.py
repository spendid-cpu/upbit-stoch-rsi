"""
dashboard.py v3.1.6
변경사항:
- v3.1.4: 🔥RS 배지, 경과일/D-N, active_price_loop
- v3.1.5: BTC 5분봉/15분봉 사이클 배지, ⚡ GOOD+ 신호, /api/btc 엔드포인트
- v3.1.6: 다이버전스 배지 추가 (🔼BULL DIV / 🔽BEAR DIV / ↗HID BULL)
           divBadge() JS 함수 추가
           Watch/Active 테이블 종목명 셀에 divBadge 삽입
"""

import threading
import traceback
from flask import Flask, jsonify, request, render_template_string
import scanner as sc

DASHBOARD_VERSION = 'v3.1.6'
app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Upbit StochRSI Scanner {{ version }}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; font-size: 13px; }

  .header {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 10px 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    position: sticky; top: 0; z-index: 100;
  }
  .header h1 { font-size: 15px; color: #58a6ff; white-space: nowrap; }
  .version-badge { background: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 2px 8px; font-size: 11px; color: #8b949e; }

  .btc-info { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .btc-price { color: #f0883e; font-weight: bold; font-size: 14px; }
  .btc-signal { padding: 2px 7px; border-radius: 4px; font-size: 11px; }
  .btc-signal.ABOVE   { background: #1a4731; color: #3fb950; }
  .btc-signal.BELOW   { background: #4d1c1c; color: #f85149; }
  .btc-signal.UNKNOWN { background: #21262d; color: #8b949e; }

  .btc-cycles { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-left: auto; }
  .btc-cycle-label { font-size: 11px; color: #8b949e; margin-right: 2px; }

  .entry-signal { padding: 3px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; white-space: nowrap; }
  .entry-signal.GOOD     { background: #1a4731; color: #3fb950; }
  .entry-signal.GOODPLUS { background: #0d3d1a; color: #39d353; border: 1px solid #39d353; animation: pulse 0.8s infinite; }
  .entry-signal.CAUTION  { background: #3d2e0a; color: #d29922; }
  .entry-signal.BLOCK    { background: #4d1c1c; color: #f85149; animation: pulse 1s infinite; }

  .status-bar {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 6px 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
  }
  .status-item { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #8b949e; }
  .status-dot  { width: 8px; height: 8px; border-radius: 50%; background: #30363d; }
  .status-dot.active { background: #3fb950; animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  .block-banner {
    display: none; background: #4d1c1c; border: 1px solid #f85149;
    padding: 8px 16px; font-size: 13px; color: #f85149;
    text-align: center; font-weight: bold;
  }
  .goodplus-banner {
    display: none; background: #0d3d1a; border: 1px solid #39d353;
    padding: 8px 16px; font-size: 13px; color: #39d353;
    text-align: center; font-weight: bold;
  }

  .main { display: grid; grid-template-columns: 1fr 300px; gap: 0; height: calc(100vh - 120px); }
  .left-panel  { overflow-y: auto; padding: 12px; min-width: 0; }
  .right-panel { border-left: 1px solid #30363d; overflow-y: auto; padding: 12px; background: #0d1117; }

  @media (max-width: 768px) {
    .main { grid-template-columns: 1fr; height: auto; overflow: visible; }
    .left-panel  { height: auto; overflow: visible; padding: 8px; }
    .right-panel { border-left: none; border-top: 1px solid #30363d; height: 300px; overflow-y: auto; padding: 8px; }
    .btc-cycles  { margin-left: 0; }
    .header h1   { font-size: 13px; }
  }

  .panel-title { font-size: 12px; color: #8b949e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px; }
  .count-badge { background: #21262d; border-radius: 10px; padding: 1px 7px; font-size: 11px; color: #58a6ff; }

  .btn { padding: 5px 12px; border-radius: 4px; border: none; cursor: pointer; font-size: 12px; transition: opacity 0.2s; }
  .btn:hover { opacity: 0.8; }
  .btn-primary { background: #238636; color: #fff; }
  .btn-danger  { background: #da3633; color: #fff; }
  .btn-warning { background: #d29922; color: #fff; }
  .btn-sm { padding: 3px 8px; font-size: 11px; }

  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { background: #161b22; color: #8b949e; padding: 6px 8px; text-align: left; white-space: nowrap; position: sticky; top: 0; border-bottom: 1px solid #30363d; }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d; white-space: nowrap; vertical-align: middle; }
  tr:hover td { background: #161b22; }

  .grade-S { color: #f0883e; font-weight: bold; }
  .grade-A { color: #3fb950; font-weight: bold; }
  .grade-B { color: #58a6ff; }
  .grade-X { color: #f85149; }
  .grade-C { color: #8b949e; }

  .score-bar { display: flex; align-items: center; gap: 6px; }
  .score-bar-bg   { width: 60px; height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; }
  .score-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }

  .price-cell    { display: flex; flex-direction: column; gap: 1px; }
  .price-entry   { color: #8b949e; font-size: 11px; }
  .price-current { font-weight: bold; }
  .price-change  { font-size: 11px; }
  .price-up   { color: #3fb950; }
  .price-down { color: #f85149; }
  .price-flat { color: #8b949e; }

  .reg-at-cell    { display: flex; flex-direction: column; gap: 2px; }
  .reg-at-time    { font-size: 11px; color: #8b949e; }
  .reg-at-elapsed { font-size: 10px; color: #58a6ff; }
  .reg-at-expire  { font-size: 10px; }
  .expire-ok   { color: #3fb950; }
  .expire-warn { color: #d29922; }
  .expire-soon { color: #f85149; }

  .badge-rs-S { background: #4d3219; color: #f0883e; }
  .badge-rs-A { background: #1a4731; color: #3fb950; }
  .badge-rs-B { background: #1c2d4a; color: #58a6ff; }

  .cycle-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }
  .cycle-BOTTOM  { background: #1a4731; color: #3fb950; }
  .cycle-RISING  { background: #1c2d4a; color: #58a6ff; }
  .cycle-PEAK    { background: #4d3219; color: #f0883e; }
  .cycle-FALLING { background: #4d1c1c; color: #f85149; }

  .badge-gc-active { background: #39d353; color: #0d1117; font-weight: bold; }

  .badge { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px; margin-left: 3px; }
  .badge-warning { background: #3d2e0a; color: #d29922; }
  .badge-danger  { background: #4d1c1c; color: #f85149; }
  .badge-gc      { background: #1c2d4a; color: #79c0ff; }

  /* v3.1.6: 다이버전스 배지 스타일 */
  .badge-bull-strong { background: #0d3d1a; color: #39d353; border: 1px solid #39d353; }
  .badge-bull        { background: #1a4731; color: #3fb950; }
  .badge-bear-strong { background: #4d1c1c; color: #f85149; border: 1px solid #f85149; }
  .badge-bear        { background: #3d1c1c; color: #f85149; }
  .badge-hid-bull    { background: #1c2d4a; color: #79c0ff; }

  .event-item { padding: 6px 8px; border-bottom: 1px solid #21262d; font-size: 11px; }
  .event-time { color: #8b949e; font-size: 10px; }
  .event-DEFAULT      { border-left: 3px solid #30363d; }
  .event-WATCH_ADD    { border-left: 3px solid #3fb950; }
  .event-WATCH_REMOVE { border-left: 3px solid #f85149; }
  .event-ACTIVE_ENTER { border-left: 3px solid #f0883e; }
  .event-ACTIVE_CLOSE { border-left: 3px solid #79c0ff; }
  .event-DEEP_SCAN    { border-left: 3px solid #d2992a; }

  .stat-cards { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin-bottom: 12px; }
  .stat-card  { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 8px 12px; }
  .stat-label { font-size: 10px; color: #8b949e; }
  .stat-value { font-size: 18px; font-weight: bold; color: #c9d1d9; }

  .toolbar { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; align-items: center; }
  .tab-btn { padding: 4px 12px; border-radius: 4px; border: 1px solid #30363d; background: #21262d; color: #8b949e; cursor: pointer; font-size: 12px; }
  .tab-btn.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }

  .loading { color: #8b949e; padding: 20px; text-align: center; }
  .empty   { color: #8b949e; padding: 20px; text-align: center; font-size: 12px; }
</style>
</head>
<body>

<div class="header">
  <h1>📊 Upbit StochRSI Scanner</h1>
  <span class="version-badge">{{ version }}</span>
  <div class="btc-info">
    <span class="btc-price" id="btcPrice">로딩중...</span>
    <span id="btcDailySignal"  class="btc-signal UNKNOWN">Daily -</span>
    <span id="btcWeeklySignal" class="btc-signal UNKNOWN">Weekly -</span>
    <span id="btcMa" style="font-size:11px;color:#8b949e;"></span>
  </div>
  <div class="btc-cycles">
    <span class="btc-cycle-label">BTC</span>
    <span id="btcDShort" class="cycle-badge cycle-RISING">일단기 -</span>
    <span id="btcDMid"   class="cycle-badge cycle-RISING">일중기 -</span>
    <span id="btcH4"     class="cycle-badge cycle-RISING">4h -</span>
    <span id="btcH1"     class="cycle-badge cycle-RISING">1h -</span>
    <span id="btcM15"    class="cycle-badge cycle-RISING">15m -</span>
    <span id="btcM5"     class="cycle-badge cycle-RISING">5m -</span>
    <span id="entrySignal" class="entry-signal CAUTION">🟡 관망</span>
  </div>
</div>

<div class="goodplus-banner" id="goodplusBanner">
  ⚡ GOOD+ — 1h 바닥 + 15분 GC + 5분 GC 동시 확인! 최적 진입 타이밍
</div>
<div class="block-banner" id="blockBanner">
  🚫 BTC 단기+중기 PEAK/FALLING — 신규 진입 자제 (auto_entry 차단 중)
</div>

<div class="status-bar">
  <div class="status-item">
    <div class="status-dot" id="scanDot"></div>
    <span id="scanStatus">대기 중</span>
  </div>
  <div class="status-item">⏱ 다음 스캔: <span id="nextScan">-</span></div>
  <div class="status-item">📦 스캔횟수: <span id="totalSymbols">-</span></div>
  <div class="status-item">💱 USDT: <span id="usdtRate">-</span>원</div>
  <div class="status-item" style="margin-left:auto;">
    <button class="btn btn-primary btn-sm" onclick="triggerScan()">🔍 즉시 스캔</button>
    <button class="btn btn-warning btn-sm" onclick="resetWatch()" style="margin-left:6px;">🔄 Watch 초기화</button>
  </div>
</div>

<div class="main">
  <div class="left-panel">
    <div class="stat-cards">
      <div class="stat-card">
        <div class="stat-label">Watch</div>
        <div class="stat-value" id="watchCount">0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Active</div>
        <div class="stat-value" id="activeCount">0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">누적 PnL</div>
        <div class="stat-value" id="totalPnl">0%</div>
      </div>
    </div>

    <div class="toolbar">
      <button class="tab-btn active" onclick="switchTab('watch')"   id="tabWatch">👁 Watch</button>
      <button class="tab-btn"        onclick="switchTab('active')"  id="tabActive">⚡ Active</button>
      <button class="tab-btn"        onclick="switchTab('history')" id="tabHistory">📜 History</button>
    </div>

    <div id="watchPanel">
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>종목</th>
            <th>등급</th>
            <th>점수</th>
            <th>등록가/현재가</th>
            <th>등록시각</th>
            <th>단기사이클</th>
            <th>중기사이클</th>
            <th>4h</th>
            <th>1h</th>
            <th>GC</th>
            <th>거래량</th>
            <th>바닥일</th>
            <th>관리</th>
          </tr></thead>
          <tbody id="watchBody"><tr><td colspan="13" class="loading">로딩중...</td></tr></tbody>
        </table>
      </div>
    </div>

    <div id="activePanel" style="display:none;">
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>종목</th><th>등급</th><th>진입가/현재가</th><th>수익률</th>
            <th>TP</th><th>SL</th><th>단기사이클</th><th>진입시각</th><th>관리</th>
          </tr></thead>
          <tbody id="activeBody"><tr><td colspan="9" class="empty">Active 종목 없음</td></tr></tbody>
        </table>
      </div>
    </div>

    <div id="historyPanel" style="display:none;">
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>종목</th><th>등급</th><th>진입가</th><th>청산가</th>
            <th>수익률</th><th>사유</th><th>청산시각</th>
          </tr></thead>
          <tbody id="historyBody"><tr><td colspan="7" class="empty">거래 내역 없음</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="right-panel">
    <div class="panel-title">
      📡 실시간 이벤트
      <span class="count-badge" id="eventCount">0</span>
    </div>
    <div id="eventList"></div>
  </div>
</div>

<script>
let currentTab = 'watch';

function switchTab(tab) {
  currentTab = tab;
  ['watch','active','history'].forEach(t => {
    document.getElementById(t+'Panel').style.display = t===tab ? '' : 'none';
    document.getElementById('tab'+t.charAt(0).toUpperCase()+t.slice(1))
            .classList.toggle('active', t===tab);
  });
}

// ── 포맷 유틸 ──────────────────────────────────────────────────────
function fmtPrice(p) {
  if (!p && p!==0) return '-';
  if (p>=1000000) return (p/1000000).toFixed(2)+'M';
  if (p>=1000)    return p.toLocaleString('ko-KR');
  if (p>=1)       return p.toFixed(2);
  if (p>=0.01)    return p.toFixed(4);
  return p.toFixed(6);
}
function fmtPct(v) {
  if (v==null) return '-';
  return (v>=0?'+':'')+v.toFixed(2)+'%';
}
function gradeHtml(g) {
  return `<span class="grade-${g||'-'}">${g||'-'}</span>`;
}
function scoreBarHtml(score) {
  const pct=Math.min(100,score);
  const color = pct>=70?'#3fb950':pct>=55?'#58a6ff':pct>=40?'#d29922':'#f85149';
  return `<div class="score-bar"><span>${score}</span>
    <div class="score-bar-bg"><div class="score-bar-fill" style="width:${pct}%;background:${color};"></div></div>
  </div>`;
}
function priceCell(entry, current, pct) {
  const p=pct!=null?pct:0;
  const cls=p>0.1?'price-up':p<-0.1?'price-down':'price-flat';
  const arrow=p>0.1?'▲':p<-0.1?'▼':'–';
  return `<div class="price-cell">
    <span class="price-entry">등록 ${fmtPrice(entry)}</span>
    <span class="price-current">${fmtPrice(current)}</span>
    <span class="price-change ${cls}">${arrow} ${fmtPct(p)}</span>
  </div>`;
}
function fmtRegAtCell(addedAt, expireAt) {
  if (!addedAt) return '<span class="reg-at-time">-</span>';
  const timeStr = addedAt.slice(5,16).replace('T',' ');
  const now=new Date(), added=new Date(addedAt);
  const elapsedMs=now-added;
  const elapsedHrs=Math.floor(elapsedMs/3600000);
  const elapsedDays=Math.floor(elapsedMs/86400000);
  let elapsedStr='';
  if(elapsedDays>=1)     elapsedStr=`+${elapsedDays}일`;
  else if(elapsedHrs>=1) elapsedStr=`+${elapsedHrs}시간`;
  else                   elapsedStr='+방금';
  let expireHtml='';
  if(expireAt){
    const exp=new Date(expireAt);
    const remainD=Math.ceil((exp-now)/86400000);
    let expCls='expire-ok';
    if(remainD<=1)      expCls='expire-soon';
    else if(remainD<=2) expCls='expire-warn';
    expireHtml=`<span class="reg-at-expire ${expCls}">D-${remainD}</span>`;
  }
  return `<div class="reg-at-cell">
    <span class="reg-at-time">${timeStr}</span>
    <span class="reg-at-elapsed">${elapsedStr}</span>
    ${expireHtml}
  </div>`;
}
function cycleBadge(cycle) {
  const label={BOTTOM:'🟢바닥',RISING:'🔵상승',PEAK:'🔴고점',FALLING:'⚫하락'};
  const c=cycle||'RISING';
  return `<span class="cycle-badge cycle-${c}">${label[c]||c}</span>`;
}
function gcBadge(h4gc,h1gc,dailyGc) {
  let b='';
  if(dailyGc) b+=`<span class="badge badge-gc">일GC</span>`;
  if(h4gc)    b+=`<span class="badge badge-gc">4hGC</span>`;
  if(h1gc)    b+=`<span class="badge badge-gc">1hGC</span>`;
  return b||'-';
}
function deepRsBadge(grade) {
  if(!grade||grade==='-') return '';
  return `<span class="badge badge-rs-${grade}">🔥RS-${grade}</span>`;
}
function warningBadges(item) {
  let b='';
  if(item.timing_warning)     b+=`<span class="badge badge-warning">⚠️4h</span>`;
  if(item.overbought_warning) b+=`<span class="badge badge-danger">🔴1h</span>`;
  return b;
}

// ── v3.1.6: 다이버전스 배지 ─────────────────────────────────────
function divBadge(item) {
  if(!item) return '';
  if(item.bull_div && item.div_strength==='STRONG')
    return `<span class="badge badge-bull-strong">🔼BULL★</span>`;
  if(item.bull_div)
    return `<span class="badge badge-bull">🔼BULL DIV</span>`;
  if(item.bear_div && item.div_strength==='STRONG')
    return `<span class="badge badge-bear-strong">🔽BEAR★</span>`;
  if(item.bear_div)
    return `<span class="badge badge-bear">🔽BEAR DIV</span>`;
  if(item.hidden_bull)
    return `<span class="badge badge-hid-bull">↗HID BULL</span>`;
  return '';
}

// ── BTC 사이클 업데이트 (v3.1.5: 15m/5m 포함) ───────────────────
function updateBtcCycles(s) {
  const cycleLabel={BOTTOM:'🟢바닥',RISING:'🔵상승',PEAK:'🔴고점',FALLING:'⚫하락'};

  const cycles = {
    btcDShort: s.btc_d_short_cycle || 'RISING',
    btcDMid:   s.btc_d_mid_cycle   || 'RISING',
    btcH4:     s.btc_h4_cycle      || 'RISING',
    btcH1:     s.btc_h1_cycle      || 'RISING',
  };
  const labels = {btcDShort:'일단기',btcDMid:'일중기',btcH4:'4h',btcH1:'1h'};
  Object.entries(cycles).forEach(([id, cycle]) => {
    const el=document.getElementById(id);
    if(!el) return;
    el.className=`cycle-badge cycle-${cycle}`;
    el.textContent=labels[id]+' '+(cycleLabel[cycle]||cycle);
  });

  // 15분봉
  const m15Cycle = s.btc_m15_cycle || 'RISING';
  const m15gc    = s.btc_m15_gc    || false;
  const m15El    = document.getElementById('btcM15');
  if(m15El){
    m15El.className = `cycle-badge cycle-${m15Cycle}${m15gc?' badge-gc-active':''}`;
    m15El.textContent = '15m '+(cycleLabel[m15Cycle]||m15Cycle)+(m15gc?' GC':'');
  }

  // 5분봉
  const m5Cycle = s.btc_m5_cycle || 'RISING';
  const m5gc    = s.btc_m5_gc    || false;
  const m5El    = document.getElementById('btcM5');
  if(m5El){
    m5El.className = `cycle-badge cycle-${m5Cycle}${m5gc?' badge-gc-active':''}`;
    m5El.textContent = '5m '+(cycleLabel[m5Cycle]||m5Cycle)+(m5gc?' GC':'');
  }

  // 진입 신호
  const sig = s.btc_entry_signal || 'CAUTION';
  const sigEl = document.getElementById('entrySignal');
  const sigMap = {
    'GOOD+':  { cls:'GOODPLUS', text:'⚡ GOOD+ 최적타이밍' },
    GOOD:     { cls:'GOOD',     text:'🟢 매수가능'         },
    CAUTION:  { cls:'CAUTION',  text:'🟡 관망'             },
    BLOCK:    { cls:'BLOCK',    text:'🔴 진입금지'         },
  };
  const sm = sigMap[sig] || sigMap.CAUTION;
  sigEl.className   = `entry-signal ${sm.cls}`;
  sigEl.textContent = sm.text;

  document.getElementById('blockBanner').style.display    = sig==='BLOCK'  ? 'block' : 'none';
  document.getElementById('goodplusBanner').style.display = sig==='GOOD+'  ? 'block' : 'none';
}

// ── Watch 렌더링 (v3.1.6: divBadge 추가) ────────────────────────
function renderWatch(items) {
  const tbody=document.getElementById('watchBody');
  if(!items||!items.length){
    tbody.innerHTML='<tr><td colspan="13" class="empty">Watch 종목 없음</td></tr>';
    return;
  }
  items=[...items].filter(w=>['S','A','B'].includes(w.grade)).sort((a,b)=>b.score-a.score);
  tbody.innerHTML=items.map(w=>{
    const ticker=(w.market||'').replace('KRW-','');
    return `<tr>
      <td><b>${ticker}</b>${warningBadges(w)}${deepRsBadge(w.deep_rs_grade)}${divBadge(w)}</td>
      <td>${gradeHtml(w.grade)}</td>
      <td>${scoreBarHtml(w.score)}</td>
      <td>${priceCell(w.reg_price,w.current_price,w.price_change)}</td>
      <td>${fmtRegAtCell(w.added_at,w.expire_at)}</td>
      <td>${cycleBadge(w.d_short_cycle)}</td>
      <td>${cycleBadge(w.d_mid_cycle)}</td>
      <td>${cycleBadge(w.h4_cycle)}</td>
      <td>${cycleBadge(w.h1_cycle)}</td>
      <td>${gcBadge(w.h4_gc,w.h1_gc,w.daily_gc)}</td>
      <td>${w.vol_ratio!=null?w.vol_ratio.toFixed(1)+'x':'-'}</td>
      <td>${w.bottom_days||0}일</td>
      <td>
        <button class="btn btn-primary btn-sm" onclick="activateWatch('${w.market}')">진입</button>
        <button class="btn btn-danger btn-sm"  onclick="removeWatch('${w.market}')" style="margin-left:4px;">제거</button>
      </td>
    </tr>`;
  }).join('');
}

// ── Active 렌더링 (v3.1.6: divBadge 추가) ───────────────────────
function renderActive(items) {
  const tbody=document.getElementById('activeBody');
  if(!items||!items.length){
    tbody.innerHTML='<tr><td colspan="9" class="empty">Active 종목 없음</td></tr>';
    return;
  }
  tbody.innerHTML=items.map(a=>{
    const ticker=(a.market||'').replace('KRW-','');
    const pnl=a.pnl_pct||0;
    const pc=pnl>0?'price-up':pnl<0?'price-down':'price-flat';
    return `<tr>
      <td><b>${ticker}</b>${deepRsBadge(a.deep_rs_grade)}${divBadge(a)}</td>
      <td>${gradeHtml(a.grade)}</td>
      <td>${priceCell(a.entry_price,a.current_price,a.pnl_pct)}</td>
      <td class="${pc}">${fmtPct(pnl)}</td>
      <td class="price-up">${fmtPrice(a.tp_price)}</td>
      <td class="price-down">${fmtPrice(a.sl_price)}</td>
      <td>${cycleBadge(a.d_short_cycle)}</td>
      <td>${(a.entry_at||'-').slice(5,16).replace('T',' ')}</td>
      <td><button class="btn btn-danger btn-sm" onclick="closeActive('${a.market}')">청산</button></td>
    </tr>`;
  }).join('');
}

// ── History 렌더링 ────────────────────────────────────────────────
function renderHistory(items) {
  const tbody=document.getElementById('historyBody');
  if(!items||!items.length){
    tbody.innerHTML='<tr><td colspan="7" class="empty">거래 내역 없음</td></tr>';
    return;
  }
  const sorted=[...items].sort((a,b)=>(b.close_at||'').localeCompare(a.close_at||''));
  tbody.innerHTML=sorted.slice(0,50).map(h=>{
    const pnl=h.pnl_pct||0;
    const pc=pnl>0?'price-up':pnl<0?'price-down':'price-flat';
    return `<tr>
      <td><b>${(h.market||'').replace('KRW-','')}</b></td>
      <td>${gradeHtml(h.grade||'-')}</td>
      <td>${fmtPrice(h.entry_price)}</td>
      <td>${fmtPrice(h.close_price)}</td>
      <td class="${pc}">${fmtPct(pnl)}</td>
      <td>${h.close_reason||'-'}</td>
      <td>${(h.close_at||'-').slice(5,16).replace('T',' ')}</td>
    </tr>`;
  }).join('');
}

// ── 이벤트 렌더링 ─────────────────────────────────────────────────
function renderEvents(events) {
  const el=document.getElementById('eventList');
  document.getElementById('eventCount').textContent=events.length;
  if(!events.length){el.innerHTML='<div class="empty">이벤트 없음</div>';return;}
  const typeMap={'📋':'WATCH_ADD','🗑️':'WATCH_REMOVE','✅':'ACTIVE_ENTER','🟢':'ACTIVE_CLOSE','🔴':'ACTIVE_CLOSE','🔥':'DEEP_SCAN'};
  el.innerHTML=[...events].reverse().slice(0,50).map(e=>{
    const cls='event-'+(typeMap[e.emoji]||'DEFAULT');
    return `<div class="event-item ${cls}">
      <div class="event-time">${e.time||''}</div>
      <div>${e.emoji||''} ${e.msg||''}</div>
    </div>`;
  }).join('');
}

// ── 전체 상태 업데이트 ────────────────────────────────────────────
function updateState(data) {
  const s=data.state||{};
  const btcKrw=s.btc_price||0;
  const usdRate=s.usdt_rate||1450;
  document.getElementById('btcPrice').textContent=
    btcKrw.toLocaleString('ko-KR')+'원 ($'+Math.round(btcKrw/usdRate).toLocaleString()+')';

  const dSig=s.btc_daily_above===true?'ABOVE':s.btc_daily_above===false?'BELOW':'UNKNOWN';
  const wSig=s.btc_weekly_above===true?'ABOVE':s.btc_weekly_above===false?'BELOW':'UNKNOWN';
  const dEl=document.getElementById('btcDailySignal');
  const wEl=document.getElementById('btcWeeklySignal');
  dEl.textContent='Daily '+dSig; dEl.className='btc-signal '+dSig;
  wEl.textContent='Weekly '+wSig; wEl.className='btc-signal '+wSig;

  const dMa=s.btc_daily_ma20||0, wMa=s.btc_weekly_ma20||0;
  document.getElementById('btcMa').textContent=
    '일MA20:'+dMa.toLocaleString('ko-KR')+' / 주MA20:'+wMa.toLocaleString('ko-KR');

  updateBtcCycles(s);

  const running=s.running||s.watch_rescanning||s.price_checking;
  document.getElementById('scanDot').className='status-dot'+(running?' active':'');
  document.getElementById('scanStatus').textContent=running?'스캔 중...':'대기 중';
  document.getElementById('nextScan').textContent=(s.next_scan||'-').slice(11,16);
  document.getElementById('totalSymbols').textContent=s.scan_count||0;
  document.getElementById('usdtRate').textContent=(s.usdt_rate||0).toLocaleString();

  const watch=data.watch||[], active=data.active||[], history=data.history||[];
  document.getElementById('watchCount').textContent=watch.filter(w=>['S','A','B'].includes(w.grade)).length;
  document.getElementById('activeCount').textContent=active.length;
  const pnl=s.total_pnl||0;
  const pnlEl=document.getElementById('totalPnl');
  pnlEl.textContent=(pnl>=0?'+':'')+pnl.toFixed(2)+'%';
  pnlEl.style.color=pnl>0?'#3fb950':pnl<0?'#f85149':'#c9d1d9';

  renderWatch(watch);
  renderActive(active);
  renderHistory(history);
}

// ── BTC 전용 경량 갱신 (10초) ─────────────────────────────────────
async function fetchBtc() {
  try {
    const r=await fetch('/api/btc');
    if(!r.ok) return;
    const d=await r.json();
    updateBtcCycles(d);
    const usdRate=1450;
    if(d.btc_price){
      document.getElementById('btcPrice').textContent=
        d.btc_price.toLocaleString('ko-KR')+'원 ($'+
        Math.round(d.btc_price/usdRate).toLocaleString()+')';
    }
  } catch(e){console.error('fetchBtc:',e);}
}

async function fetchState() {
  try {
    const r=await fetch('/api/state');
    if(!r.ok) throw new Error('status '+r.status);
    updateState(await r.json());
  } catch(e){console.error('fetchState:',e);}
}
async function fetchEvents() {
  try {
    const r=await fetch('/api/events');
    if(!r.ok) return;
    renderEvents((await r.json()).events||[]);
  } catch(e){console.error('fetchEvents:',e);}
}
async function triggerScan() {
  await fetch('/api/scan',{method:'POST'});
  setTimeout(fetchState,2000);
}
async function resetWatch() {
  if(!confirm('Watch 목록을 초기화할까요?')) return;
  await fetch('/api/watch/reset',{method:'POST'});
  fetchState();
}
async function removeWatch(market) {
  await fetch('/api/watch/remove',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({market})});
  fetchState();
}
async function activateWatch(market) {
  await fetch('/api/watch/activate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({market})});
  fetchState(); switchTab('active');
}
async function closeActive(market) {
  if(!confirm(market+' 청산할까요?')) return;
  await fetch('/api/active/close',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({market})});
  fetchState();
}

// 초기 로드
fetchState();
fetchEvents();
fetchBtc();

// 주기 설정
setInterval(fetchBtc,    10000);   // 10초: BTC 사이클 전용
setInterval(fetchState,  10000);   // 10초: 전체 상태
setInterval(fetchEvents, 15000);   // 15초: 이벤트
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(TEMPLATE, version=DASHBOARD_VERSION)

@app.route('/api/version')
def api_version():
    return jsonify({'dashboard':DASHBOARD_VERSION,'scanner':sc.VERSION,'mtf':sc.MTF_VERSION})

@app.route('/api/state')
def api_state():
    try:
        return jsonify({
            'state':   sc.get_scanner_state(),
            'watch':   sc._load_json(sc.WATCH_FILE,   []),
            'active':  sc._load_json(sc.ACTIVE_FILE,  []),
            'history': sc._load_json(sc.HISTORY_FILE, []),
            'deep':    sc._load_json(sc.DEEP_FILE,    []),
        })
    except Exception as e:
        return jsonify({'error':str(e),'trace':traceback.format_exc()}),500

@app.route('/api/btc')
def api_btc():
    try:
        s = sc.get_scanner_state()
        return jsonify({
            'btc_price':         s.get('btc_price'),
            'btc_d_short_cycle': s.get('btc_d_short_cycle'),
            'btc_d_mid_cycle':   s.get('btc_d_mid_cycle'),
            'btc_h4_cycle':      s.get('btc_h4_cycle'),
            'btc_h1_cycle':      s.get('btc_h1_cycle'),
            'btc_m15_cycle':     s.get('btc_m15_cycle'),
            'btc_m15_gc':        s.get('btc_m15_gc'),
            'btc_m5_cycle':      s.get('btc_m5_cycle'),
            'btc_m5_gc':         s.get('btc_m5_gc'),
            'btc_entry_signal':  s.get('btc_entry_signal'),
            'btc_daily_above':   s.get('btc_daily_above'),
            'btc_weekly_above':  s.get('btc_weekly_above'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/events')
def api_events():
    try:
        return jsonify({'events': sc._load_json(sc.EVENT_FILE, [])})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/scan', methods=['POST'])
def api_scan():
    return jsonify(sc.run_single_scan())

@app.route('/api/watch/add', methods=['POST'])
def api_watch_add():
    data=request.get_json() or {}
    market=data.get('market','').upper()
    if not market.startswith('KRW-'): market='KRW-'+market
    return jsonify(sc.manual_add_watch(market))

@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data=request.get_json() or {}
    return jsonify(sc.manual_remove_watch(data.get('market','').replace('KRW-','')))

@app.route('/api/watch/activate', methods=['POST'])
def api_watch_activate():
    data=request.get_json() or {}
    return jsonify(sc.manual_activate_watch(data.get('market','').replace('KRW-','')))

@app.route('/api/watch/reset', methods=['POST'])
def api_watch_reset():
    return jsonify(sc.reset_watch_list())

@app.route('/api/active/close', methods=['POST'])
def api_active_close():
    data=request.get_json() or {}
    return jsonify(sc.manual_close_active(data.get('market','').replace('KRW-',''), reason='수동종료'))

@app.route('/api/config')
def api_config():
    return jsonify({
        'scan_interval_min':   sc.SCAN_INTERVAL_MIN,
        'rescan_interval_min': sc.WATCH_RESCAN_INTERVAL_MIN,
        'price_check_min':     sc.PRICE_CHECK_INTERVAL_MIN,
        'tp_pct':              sc.TRADE_TP_PCT,
        'sl_pct':              sc.TRADE_SL_PCT,
        'watch_expire_days':   sc.WATCH_EXPIRE_DAYS,
        'allowed_grades':      ['S','A','B'],
        'watch_drop_pct':      sc.WATCH_DROP_PCT,
        'watch_rise_pct':      sc.WATCH_RISE_PCT,
    })

# ─────────────────────────────────────────────
# BACKGROUND THREADS
# ─────────────────────────────────────────────
def start_background_threads():
    for fn in [
        sc.scanner_loop,
        sc.watch_rescan_loop,
        sc.price_check_loop,
        sc.active_monitor_loop,
        sc.active_price_loop,    # v3.1.4
        sc.btc_fast_loop,        # v3.1.5
        sc.deep_scan_loop,
        sc.daily_summary_loop,
    ]:
        threading.Thread(target=fn, daemon=True).start()

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print(f'✅ Dashboard {DASHBOARD_VERSION} + Scanner {sc.VERSION} 시작')
    print(f'   MTF: {sc.MTF_VERSION}')
    print(f'   BTC 전용 빠른 루프 10초 (5m/15m) ✅')
    print(f'   ⚡ GOOD+ 신호 (1h바닥 + 15mGC + 5mGC) ✅')
    print(f'   /api/btc 경량 엔드포인트 ✅')
    print(f'   fetchState/fetchBtc 10초 갱신 ✅')
    print(f'   Watch 등록시각 + 경과일 + D-N 만료 표시 ✅')
    print(f'   🔥RS 배지 (DEEP 상대강도) ✅')
    print(f'   Active 30초 가격 업데이트 루프 ✅')
    print(f'   🔼🔽 다이버전스 배지 (BULL★/BULL/BEAR★/BEAR/HID BULL) ✅')
    start_background_threads()
    app.run(host='0.0.0.0', port=sc.PORT, debug=False)
