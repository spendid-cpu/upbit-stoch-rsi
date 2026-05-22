"""
dashboard.py v2.4.3

변경사항:
- renderGradeBars: appendChild 방식 → innerHTML 직접 할당으로 완전 교체 (증식 완전 차단)
- dirIcon: string 타입 레거시 데이터 안전 처리, 'GX' 문자 포함 여부 2차 폴백
- sparkline: 1개 데이터 → SVG 대신 '○' 텍스트 (렌더링 안정)
- 전체 null/undefined 방어 강화
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
    --bg:#0f1117; --card:#1a1d27; --border:#2a2d3e;
    --text:#e2e8f0; --muted:#64748b; --accent:#6366f1;
    --green:#22c55e; --red:#ef4444; --yellow:#f59e0b;
    --orange:#f97316; --blue:#3b82f6; --deep:#ec4899;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;font-size:13px;}
  a{color:var(--accent);text-decoration:none;}

  /* 헤더 */
  .header{background:var(--card);border-bottom:1px solid var(--border);
          padding:12px 20px;display:flex;align-items:center;gap:16px;}
  .header h1{font-size:18px;font-weight:700;}
  .status-badge{padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;
                background:#22c55e22;color:var(--green);}
  .status-badge.scanning{background:#f59e0b22;color:var(--yellow);}
  .status-badge.error   {background:#ef444422;color:var(--red);}
  .btn{padding:5px 14px;border-radius:6px;border:1px solid var(--border);
       background:var(--card);color:var(--text);cursor:pointer;font-size:12px;}
  .btn:hover{background:var(--accent);border-color:var(--accent);color:#fff;}
  .spacer{flex:1;}

  /* 카드 */
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
         gap:12px;padding:16px 20px;}
  .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;}
  .card .label{font-size:11px;color:var(--muted);margin-bottom:6px;}
  .card .value{font-size:22px;font-weight:700;}
  .card .sub  {font-size:11px;color:var(--muted);margin-top:4px;}
  .card.deep-card .value{color:var(--deep);}

  /* 등급 승률 바 - 고정 4개 */
  .grade-bars{display:flex;gap:10px;padding:0 20px 16px;flex-wrap:wrap;}
  .grade-bar-item{background:var(--card);border:1px solid var(--border);
                  border-radius:8px;padding:10px 14px;min-width:130px;}
  .grade-bar-item .g-label  {font-size:11px;color:var(--muted);margin-bottom:6px;}
  .grade-bar-item .g-bar-bg {background:#ffffff15;border-radius:4px;height:6px;margin:4px 0;}
  .grade-bar-item .g-bar-fill{height:6px;border-radius:4px;background:var(--accent);}
  .grade-bar-item .g-stats  {font-size:11px;color:var(--muted);}

  /* DEEP 패널 */
  .deep-info-panel{margin:0 20px 16px;background:#ec489915;
                   border:1px solid #ec489950;border-radius:10px;padding:14px 18px;}
  .deep-info-panel h3{color:var(--deep);font-size:14px;margin-bottom:10px;}
  .deep-info-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;}
  .deep-info-item .d-label{font-size:11px;color:var(--muted);}
  .deep-info-item .d-value{font-size:16px;font-weight:700;color:var(--deep);}

  /* 탭 */
  .tabs{display:flex;padding:0 20px;border-bottom:1px solid var(--border);}
  .tab{padding:10px 18px;cursor:pointer;border-bottom:2px solid transparent;
       font-size:13px;color:var(--muted);transition:all .15s;}
  .tab.active  {color:var(--text);border-bottom-color:var(--accent);}
  .tab:hover   {color:var(--text);}
  .tab.deep-tab{color:var(--deep);}
  .tab.deep-tab.active{border-bottom-color:var(--deep);}
  .tab-content       {display:none;padding:16px 20px;}
  .tab-content.active{display:block;}

  /* 범례 */
  .legend{background:var(--card);border:1px solid var(--border);border-radius:8px;
          padding:10px 16px;margin-bottom:14px;font-size:11px;color:var(--muted);}
  .legend .legend-title{font-weight:600;color:var(--text);margin-bottom:8px;}
  .legend-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:4px 16px;}
  .legend-row{display:flex;gap:8px;align-items:center;}
  .legend-row .es{min-width:90px;}
  .legend-row .combo{color:var(--text);}

  /* 테이블 */
  .tbl-wrap{overflow-x:auto;}
  table{width:100%;border-collapse:collapse;font-size:12px;}
  thead th{background:#ffffff08;padding:8px 10px;text-align:left;
           border-bottom:1px solid var(--border);white-space:nowrap;
           font-size:11px;color:var(--muted);font-weight:600;}
  tbody tr{border-bottom:1px solid #ffffff08;transition:background .1s;}
  tbody tr:hover{background:#ffffff05;}
  tbody td{padding:8px 10px;vertical-align:middle;white-space:nowrap;}

  /* 배지 */
  .badge{display:inline-block;padding:2px 8px;border-radius:10px;
         font-size:11px;font-weight:700;}
  .badge-S     {background:#f59e0b22;color:#f59e0b;}
  .badge-A     {background:#6366f122;color:#818cf8;}
  .badge-B     {background:#22c55e22;color:#4ade80;}
  .badge-C     {background:#ffffff10;color:var(--muted);}
  .badge-deep-s{background:#ec489930;color:var(--deep);}
  .badge-deep-a{background:#ec489920;color:#f472b6;}
  .badge-deep-b{background:#ec489910;color:#fda4af;}

  /* K값 */
  .k-red   {color:#ef4444;font-weight:700;}
  .k-orange{color:#f97316;font-weight:600;}
  .k-yellow{color:#f59e0b;}
  .k-white {color:var(--text);}

  /* 진입강도 */
  .es-3{color:#22c55e;font-weight:700;}
  .es-2{color:#6366f1;}
  .es-1{color:#94a3b8;}
  .es-0{color:#374151;}

  /* PnL */
  .pnl-pos{color:var(--green);font-weight:600;}
  .pnl-neg{color:var(--red);  font-weight:600;}

  /* delta */
  .delta-pos{color:var(--green);font-size:11px;}
  .delta-neg{color:var(--red);  font-size:11px;}
  .delta-neu{color:var(--muted);font-size:11px;}

  /* 방향 */
  .dir-up {color:var(--green);}
  .dir-mid{color:var(--yellow);}
  .dir-dn {color:var(--red);}
  .dir-gx {color:var(--yellow);font-weight:700;font-size:11px;}

  /* 수동 입력 */
  .add-form{display:flex;gap:8px;margin-bottom:14px;}
  .add-form input{background:var(--card);border:1px solid var(--border);
                  border-radius:6px;padding:7px 12px;color:var(--text);
                  font-size:13px;width:200px;}
  .add-form input:focus{outline:none;border-color:var(--accent);}

  /* 신규 카운트다운 */
  .new-cd{font-size:11px;color:var(--muted);margin-bottom:10px;}

  /* 히스토리 */
  .h-act {color:var(--blue);}
  .h-cls {color:var(--muted);}
  .h-deep{color:var(--deep);}

  @media(max-width:768px){
    .cards{grid-template-columns:repeat(2,1fr);}
    .header h1{font-size:15px;}
  }
</style>
</head>
<body>

<!-- 헤더 -->
<div class="header">
  <h1>📊 Upbit MTF Scanner</h1>
  <span id="statusBadge" class="status-badge">● idle</span>
  <span class="spacer"></span>
  <span id="lastScan" style="font-size:11px;color:var(--muted)"></span>
  <button class="btn" onclick="triggerScan()">🔄 수동 스캔</button>
</div>

<!-- 요약 카드 -->
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

<!-- 등급 승률 바 (고정 4개 div, JS가 내용만 교체) -->
<div class="grade-bars">
  <div class="grade-bar-item" id="gbar-S"></div>
  <div class="grade-bar-item" id="gbar-A"></div>
  <div class="grade-bar-item" id="gbar-B"></div>
  <div class="grade-bar-item" id="gbar-C"></div>
</div>

<!-- DEEP 패널 -->
<div id="deepInfoPanel" class="deep-info-panel" style="display:none">
  <h3>🔴 DEEP Watch 현황</h3>
  <div class="deep-info-grid" id="deepInfoGrid"></div>
</div>

<!-- 탭 -->
<div class="tabs">
  <div class="tab active"   onclick="switchTab('watch')"  >👁 Watch  <span id="tabWatchCnt" ></span></div>
  <div class="tab"          onclick="switchTab('active')" >⚡ Active <span id="tabActiveCnt"></span></div>
  <div class="tab deep-tab" onclick="switchTab('deep')"   >🔴 DEEP   <span id="tabDeepCnt"  ></span></div>
  <div class="tab"          onclick="switchTab('new')"    >🆕 신규   <span id="tabNewCnt"   ></span></div>
  <div class="tab"          onclick="switchTab('history')">📜 히스토리</div>
</div>

<!-- Watch 탭 -->
<div id="tab-watch" class="tab-content active">
  <div class="legend">
    <div class="legend-title">📖 진입강도 해석 가이드</div>
    <div class="legend-grid">
      <div class="legend-row"><span class="es es-3">🚀강한신호</span><span class="combo">S/A등급 → 적극 진입 고려</span></div>
      <div class="legend-row"><span class="es es-3">🚀강한신호</span><span class="combo">B등급 → 차트 강반등, 다음 재스캔 주목</span></div>
      <div class="legend-row"><span class="es es-2">🎯진입고려</span><span class="combo">S/A등급 → 진입 준비 / 일봉K≤2 자동 보장</span></div>
      <div class="legend-row"><span class="es es-1">👀관찰</span>    <span class="combo">방향 형성 중 / 일봉K≤2 횡보여도 최소 보장</span></div>
      <div class="legend-row"><span class="es es-0">⏳대기</span>    <span class="combo">신호 없음 → 완전 대기</span></div>
      <div class="legend-row"><span style="color:var(--orange)">⚠ 4hK과열</span><span class="combo">4hK&gt;80 → 진입강도 1단계 자동 하향</span></div>
    </div>
  </div>
  <div class="add-form">
    <input id="addInput" type="text" placeholder="KRW-BTC or BTC"
           onkeydown="if(event.key==='Enter')addWatch()">
    <button class="btn" onclick="addWatch()">+ Watch 추가</button>
  </div>
  <div class="tbl-wrap">
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

<!-- Active 탭 -->
<div id="tab-active" class="tab-content">
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>티커</th><th>등급</th><th>점수</th><th>유형</th>
        <th>진입가</th><th>현재가</th><th>PnL</th>
        <th>TP</th><th>SL</th><th>진입강도</th><th>방향</th>
        <th>진입일시</th><th>타임아웃</th><th>관리</th>
      </tr></thead>
      <tbody id="activeBody"></tbody>
    </table>
  </div>
</div>

<!-- DEEP 탭 -->
<div id="tab-deep" class="tab-content">
  <div class="legend" style="border-color:#ec489950">
    <div class="legend-title" style="color:var(--deep)">🔴 DEEP Watch 란?</div>
    <div style="margin-top:6px;line-height:1.8">
      일봉K ≤ 5 (극단 과매도) + BTC 하락 중 + 해당 코인이 BTC보다 버티는 종목<br>
      <span style="color:var(--deep)">DEEP-S/A → 자동 Active 전환</span>
      &nbsp;|&nbsp;
      <span style="color:#f472b6">DEEP-B → Watch 유지</span>
    </div>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>티커</th><th>DEEP등급</th><th>DEEP점수</th>
        <th>일봉K</th><th>BTC 24h</th><th>코인 24h</th><th>상대강도</th>
        <th>바닥유지</th><th>거래량비율</th><th>주봉K</th>
        <th>현재가</th><th>PnL</th><th>진입일시</th><th>관리</th>
      </tr></thead>
      <tbody id="deepBody"></tbody>
    </table>
  </div>
</div>

<!-- 신규 탭 -->
<div id="tab-new" class="tab-content">
  <div class="new-cd" id="newCd"></div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>티커</th><th>등급</th><th>점수</th>
        <th>일봉K</th><th>4hK</th><th>1hK</th>
        <th>진입강도</th><th>방향</th><th>현재가</th><th>등록일시</th>
      </tr></thead>
      <tbody id="newBody"></tbody>
    </table>
  </div>
</div>

<!-- 히스토리 탭 -->
<div id="tab-history" class="tab-content">
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>유형</th><th>티커</th><th>등급</th><th>거래유형</th>
        <th>진입가</th><th>청산가</th><th>PnL</th><th>사유</th><th>일시</th>
      </tr></thead>
      <tbody id="histBody"></tbody>
    </table>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════
//  유틸
// ═══════════════════════════════════════════════════════════
const TAB_NAMES = ['watch','active','deep','new','history'];

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i)=>
    t.classList.toggle('active', TAB_NAMES[i]===name));
  document.querySelectorAll('.tab-content').forEach(c=>
    c.classList.toggle('active', c.id==='tab-'+name));
}

function fmtPrice(v) {
  if(v==null||v===0) return '-';
  if(v>=1000) return Number(v).toLocaleString('ko-KR',{maximumFractionDigits:0});
  if(v>=1)    return Number(v).toFixed(2);
  return Number(v).toFixed(4);
}

function fmtPct(v) {
  if(v==null) return '-';
  const c = v>=0?'pnl-pos':'pnl-neg';
  return `<span class="${c}">${v>=0?'+':''}${v.toFixed(2)}%</span>`;
}

function fmtTime(iso) {
  if(!iso) return '-';
  try {
    const d   = new Date(iso);
    const kst = new Date(d.getTime()+9*3600000);
    const p   = n=>String(n).padStart(2,'0');
    return `${p(kst.getUTCMonth()+1)}/${p(kst.getUTCDate())} ${p(kst.getUTCHours())}:${p(kst.getUTCMinutes())}`;
  } catch(e){ return '-'; }
}

function timeAgo(iso) {
  if(!iso) return '-';
  try {
    const m = Math.floor((Date.now()-new Date(iso).getTime())/60000);
    if(m<60)  return m+'분전';
    const h = Math.floor(m/60);
    if(h<24)  return h+'시간전';
    return Math.floor(h/24)+'일전';
  } catch(e){ return '-'; }
}

// ═══════════════════════════════════════════════════════════
//  셀 렌더러
// ═══════════════════════════════════════════════════════════
function kCell(v) {
  if(v==null) return '<span class="k-white">-</span>';
  const f = parseFloat(v);
  if(isNaN(f)) return '<span class="k-white">-</span>';
  const c = f<=5?'k-red':f<=10?'k-orange':f<=20?'k-yellow':'k-white';
  return `<span class="${c}">${f.toFixed(1)}</span>`;
}

function h4KCell(v) {
  if(v==null) return '<span class="k-white">-</span>';
  const f = parseFloat(v);
  if(isNaN(f)) return '<span class="k-white">-</span>';
  const c    = f<=5?'k-red':f<=20?'k-orange':'k-white';
  const icon = f>80?'🔥':f>50?'⚠':'';
  return `<span class="${c}">${f.toFixed(1)}${icon}</span>`;
}

// ── 방향 아이콘 (v2.4.3 핵심 수정) ──────────────────────────
function dirIcon(info) {
  // 1) null / undefined
  if(info==null) return '<span class="dir-mid">→</span>';

  // 2) string 타입 레거시 데이터 처리
  if(typeof info === 'string') {
    if(info.includes('GX') || info.includes('✨'))
      return '<span class="dir-gx">✨GX</span>';
    const smap={'상승':'↑','반등':'↗','횡보':'→','하락':'↓'};
    const scls={'상승':'dir-up','반등':'dir-up','횡보':'dir-mid','하락':'dir-dn'};
    return `<span class="${scls[info]||'dir-mid'}">${smap[info]||'→'}</span>`;
  }

  // 3) object 타입 정상 처리
  if(typeof info !== 'object') return '<span class="dir-mid">→</span>';

  const dir = info.direction || '횡보';
  // golden_cross 필드 다양한 키 이름 대응
  const gx  = info.golden_cross===true
           || info.gx===true
           || info.goldenCross===true;

  if(gx) return '<span class="dir-gx">✨GX</span>';

  const imap={'상승':'↑','반등':'↗','횡보':'→','하락':'↓'};
  const cmap={'상승':'dir-up','반등':'dir-up','횡보':'dir-mid','하락':'dir-dn'};
  return `<span class="${cmap[dir]||'dir-mid'}">${imap[dir]||'→'}</span>`;
}

function dirCell(d,h4,h1) {
  return dirIcon(d)+dirIcon(h4)+dirIcon(h1);
}

function esCell(level, label) {
  if(level==null) return '-';
  const lv  = parseInt(level,10)||0;
  const cls = ['es-0','es-1','es-2','es-3'][lv]||'es-0';
  const lbl = label||['⏳대기','👀관찰','🎯진입고려','🚀강한신호'][lv]||'⏳대기';
  return `<span class="${cls}">${lbl}</span>`;
}

function gradeBadge(g) {
  if(!g) return '-';
  return `<span class="badge badge-${g}">${g}</span>`;
}

function deepGradeBadge(g) {
  if(!g) return '-';
  const s = g.includes('S')?'s':g.includes('A')?'a':'b';
  return `<span class="badge badge-deep-${s}">${g}</span>`;
}

function deltaCell(hist) {
  if(!hist||hist.length<2) return '<span class="delta-neu">-</span>';
  const d = hist[hist.length-1]-hist[hist.length-2];
  if(d>0)  return `<span class="delta-pos">▲${d}</span>`;
  if(d<0)  return `<span class="delta-neg">▼${Math.abs(d)}</span>`;
  return '<span class="delta-neu">→</span>';
}

// sparkline: 1개 데이터 → '○' 텍스트 (SVG 렌더 불안정 방지)
function sparkline(data, w=60, h=20) {
  if(!data||data.length===0)
    return '<span style="color:var(--muted);font-size:10px">-</span>';
  if(data.length===1)
    return '<span style="color:var(--muted);font-size:14px">○</span>';

  const pts   = data.slice(-20);
  const mn    = Math.min(...pts);
  const mx    = Math.max(...pts);
  const range = mx-mn||1;
  const step  = w/(pts.length-1);
  const points= pts.map((v,i)=>{
    const x = i*step;
    const y = h-((v-mn)/range)*(h-2)-1;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const color = pts[pts.length-1]>=pts[0]?'#22c55e':'#ef4444';
  return `<svg width="${w}" height="${h}" style="display:block">
    <polyline points="${points}" fill="none" stroke="${color}"
      stroke-width="1.5" stroke-linejoin="round"/>
  </svg>`;
}

// ═══════════════════════════════════════════════════════════
//  등급 승률 바 (v2.4.3 핵심 수정: 고정 div에 innerHTML 교체)
// ═══════════════════════════════════════════════════════════
function renderGradeBars(gs) {
  ['S','A','B','C'].forEach(g => {
    const el = document.getElementById('gbar-'+g);
    if(!el) return;                          // div 없으면 스킵
    const s   = (gs||{})[g]||{};
    const wr  = s.win_rate??0;
    const avg = s.avg_pnl ??0;
    const tot = s.total   ??0;
    // innerHTML 직접 교체 → 절대 증식 없음
    el.innerHTML = `
      <div class="g-label">${g}등급 승률</div>
      <div class="g-bar-bg">
        <div class="g-bar-fill" style="width:${wr}%"></div>
      </div>
      <div class="g-stats">
        ${wr}% | avg ${avg>=0?'+':''}${avg}% | ${tot}건
      </div>`;
  });
}

// ═══════════════════════════════════════════════════════════
//  테이블 렌더러
// ═══════════════════════════════════════════════════════════
function renderWatch(list) {
  const tb = document.getElementById('watchBody');
  if(!list||!list.length){
    tb.innerHTML='<tr><td colspan="15" style="text-align:center;padding:24px;color:var(--muted)">데이터 없음</td></tr>';
    return;
  }
  tb.innerHTML = list.map(w=>{
    const tk  = w.ticker||'-';
    const nm  = tk.replace('KRW-','');
    const exp = w.expire_at?fmtTime(w.expire_at):'∞';
    return `<tr>
      <td><a href="https://upbit.com/exchange?code=CRIX.UPBIT.${tk}" target="_blank">${w.manual?'👤':''}${nm}</a></td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${w.score??'-'}</td>
      <td>${deltaCell(w.score_history)}</td>
      <td>${kCell(w.daily_k)}</td>
      <td>${h4KCell(w.h4_k)}</td>
      <td>${kCell(w.h1_k)}</td>
      <td>${esCell(w.entry_level,w.entry_label)}</td>
      <td>${dirCell(w.daily_dir_info,w.h4_dir_info,w.h1_dir_info)}</td>
      <td>${sparkline(w.score_history)}</td>
      <td>${w.vol_ratio!=null?w.vol_ratio.toFixed(2)+'x':'-'}</td>
      <td>${fmtPrice(w.current_price)}</td>
      <td>${timeAgo(w.registered_at)}</td>
      <td>${exp}</td>
      <td><button class="btn" style="font-size:11px;padding:2px 8px"
            onclick="removeWatch('${tk}')">제거</button></td>
    </tr>`;
  }).join('');
}

function renderActive(list) {
  const tb    = document.getElementById('activeBody');
  const items = (list||[]).filter(a=>(a.trade_type||'normal')==='normal');
  if(!items.length){
    tb.innerHTML='<tr><td colspan="14" style="text-align:center;padding:24px;color:var(--muted)">Active 없음</td></tr>';
    return;
  }
  tb.innerHTML = items.map(a=>{
    const tk = a.ticker||'-';
    const nm = tk.replace('KRW-','');
    return `<tr>
      <td><a href="https://upbit.com/exchange?code=CRIX.UPBIT.${tk}" target="_blank">${nm}</a></td>
      <td>${gradeBadge(a.grade)}</td>
      <td>${a.score??'-'}</td>
      <td><span style="color:var(--blue)">${a.trade_type||'normal'}</span></td>
      <td>${fmtPrice(a.entry_price)}</td>
      <td>${fmtPrice(a.current_price)}</td>
      <td>${fmtPct(a.pnl_pct)}</td>
      <td style="color:var(--green)">${fmtPrice(a.tp_price)}</td>
      <td style="color:var(--red)">${fmtPrice(a.sl_price)}</td>
      <td>${esCell(a.entry_level,a.entry_label)}</td>
      <td>${dirCell(a.daily_dir_info,a.h4_dir_info,a.h1_dir_info)}</td>
      <td>${fmtTime(a.activated_at)}</td>
      <td>${fmtTime(a.timeout_at)}</td>
      <td><button class="btn" style="font-size:11px;padding:2px 8px;color:var(--red)"
            onclick="closeTrade('${tk}')">청산</button></td>
    </tr>`;
  }).join('');
}

function renderDeep(list) {
  const tb   = document.getElementById('deepBody');
  const deep = (list||[]).filter(a=>a.trade_type==='deep');
  if(!deep.length){
    tb.innerHTML=`<tr><td colspan="14" style="text-align:center;padding:24px;color:var(--muted)">
      DEEP Active 없음<br>
      <span style="font-size:11px">일봉K≤5 + BTC 하락 + 코인 버팀 조건 충족 시 자동 등록</span>
    </td></tr>`;
    return;
  }
  tb.innerHTML = deep.map(a=>{
    const tk  = a.ticker||'-';
    const nm  = tk.replace('KRW-','');
    const rel = a.relative_strength;
    const rs  = rel!=null?`<span style="color:var(--green)">+${rel.toFixed(2)}%p</span>`:'-';
    return `<tr>
      <td><a href="https://upbit.com/exchange?code=CRIX.UPBIT.${tk}" target="_blank">
        <span style="color:var(--deep)">🔴</span>${nm}</a></td>
      <td>${deepGradeBadge(a.deep_grade)}</td>
      <td>${a.deep_score??'-'}</td>
      <td>${kCell(a.daily_k)}</td>
      <td><span style="color:var(--red)">${a.btc_change!=null?a.btc_change.toFixed(2)+'%':'-'}</span></td>
      <td>${a.coin_change!=null?a.coin_change.toFixed(2)+'%':'-'}</td>
      <td>${rs}</td>
      <td>${a.days_at_bottom??'-'}일</td>
      <td>${a.vol_ratio!=null?a.vol_ratio.toFixed(2)+'x':'-'}</td>
      <td>${kCell(a.weekly_k)}</td>
      <td>${fmtPrice(a.current_price)}</td>
      <td>${fmtPct(a.pnl_pct)}</td>
      <td>${fmtTime(a.activated_at)}</td>
      <td><button class="btn" style="font-size:11px;padding:2px 8px;color:var(--red)"
            onclick="closeTrade('${tk}')">청산</button></td>
    </tr>`;
  }).join('');
}

let _newRegAt = null;
function renderNew(list) {
  const tb  = document.getElementById('newBody');
  const cnt = document.getElementById('tabNewCnt');
  const cd  = document.getElementById('newCd');
  if(!list||!list.length){
    tb.innerHTML='<tr><td colspan="10" style="text-align:center;padding:24px;color:var(--muted)">신규 없음 (5분 후 자동 초기화)</td></tr>';
    cnt.textContent='';
    cd.textContent='';
    _newRegAt=null;
    return;
  }
  cnt.textContent=`(${list.length})`;
  if(!_newRegAt && list[0]?.registered_at)
    _newRegAt = new Date(list[0].registered_at).getTime();

  tb.innerHTML = list.map(w=>{
    const tk = w.ticker||'-';
    const nm = tk.replace('KRW-','');
    return `<tr style="background:#22c55e08">
      <td><a href="https://upbit.com/exchange?code=CRIX.UPBIT.${tk}" target="_blank">🆕${nm}</a></td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${w.score??'-'}</td>
      <td>${kCell(w.daily_k)}</td>
      <td>${h4KCell(w.h4_k)}</td>
      <td>${kCell(w.h1_k)}</td>
      <td>${esCell(w.entry_level,w.entry_label)}</td>
      <td>${dirCell(w.daily_dir_info,w.h4_dir_info,w.h1_dir_info)}</td>
      <td>${fmtPrice(w.current_price)}</td>
      <td>${fmtTime(w.registered_at)}</td>
    </tr>`;
  }).join('');
}

function renderHistory(list) {
  const tb = document.getElementById('histBody');
  if(!list||!list.length){
    tb.innerHTML='<tr><td colspan="9" style="text-align:center;padding:24px;color:var(--muted)">히스토리 없음</td></tr>';
    return;
  }
  tb.innerHTML = [...list].reverse().slice(0,100).map(h=>{
    const lbl = {activate:'진입',close:'청산'}[h.type]||h.type;
    const cls = h.trade_type==='deep'?'h-deep':h.type==='activate'?'h-act':'h-cls';
    const dt  = h.closed_at||h.activated_at;
    return `<tr>
      <td class="${cls}">${lbl}</td>
      <td>${(h.ticker||'-').replace('KRW-','')}</td>
      <td>${gradeBadge(h.grade)}</td>
      <td>${h.trade_type||'normal'}</td>
      <td>${fmtPrice(h.entry_price)}</td>
      <td>${fmtPrice(h.close_price)}</td>
      <td>${h.pnl_pct!=null?fmtPct(h.pnl_pct):'-'}</td>
      <td>${h.close_reason||'-'}</td>
      <td>${fmtTime(dt)}</td>
    </tr>`;
  }).join('');
}

// ── DEEP 패널 ────────────────────────────────────────────────
function renderDeepPanel(list) {
  const panel = document.getElementById('deepInfoPanel');
  const grid  = document.getElementById('deepInfoGrid');
  const dl    = (list||[]).filter(a=>a.trade_type==='deep');
  if(!dl.length){ panel.style.display='none'; return; }
  panel.style.display='block';
  const avgPnl = dl.reduce((s,a)=>s+(a.pnl_pct||0),0)/dl.length;
  const winCnt = dl.filter(a=>(a.pnl_pct||0)>=0).length;
  const maxRel = Math.max(...dl.map(a=>a.relative_strength||0));
  grid.innerHTML=`
    <div class="deep-info-item"><div class="d-label">DEEP Active</div>
      <div class="d-value">${dl.length}개</div></div>
    <div class="deep-info-item"><div class="d-label">DEEP 승률</div>
      <div class="d-value">${Math.round(winCnt/dl.length*100)}%</div></div>
    <div class="deep-info-item"><div class="d-label">평균 PnL</div>
      <div class="d-value" style="color:${avgPnl>=0?'var(--green)':'var(--red)'}">
        ${avgPnl>=0?'+':''}${avgPnl.toFixed(2)}%</div></div>
    <div class="deep-info-item"><div class="d-label">최고 상대강도</div>
      <div class="d-value">${maxRel.toFixed(2)}%p</div></div>`;
}

// ═══════════════════════════════════════════════════════════
//  상태 업데이트
// ═══════════════════════════════════════════════════════════
function updateState(data) {
  // 헤더
  const badge = document.getElementById('statusBadge');
  const st    = data.status||'idle';
  badge.textContent = '● '+st;
  badge.className   = 'status-badge'+(st==='scanning'?' scanning':st==='error'?' error':'');
  const ls = document.getElementById('lastScan');
  ls.textContent = data.last_scan_at?'마지막 스캔: '+fmtTime(data.last_scan_at):'';

  // 카드
  const s = data.stats||{};
  document.getElementById('cardWatch').textContent   = data.watch_count ??'-';
  document.getElementById('cardActive').textContent  = data.active_count??'-';
  document.getElementById('cardDeep').textContent    = data.deep_count  ??0;
  document.getElementById('cardWinRate').textContent =
    s.win_rate!=null?s.win_rate.toFixed(1)+'%':'-';
  document.getElementById('cardWinSub').textContent  =
    s.total_trades?`총 ${s.total_trades}건`:'';
  document.getElementById('cardAvgPnl').textContent  =
    s.avg_pnl!=null?(s.avg_pnl>=0?'+':'')+s.avg_pnl.toFixed(2)+'%':'-';

  // BTC MA20
  const m = data.macro||{};
  if(m.btc_weekly_ma20){
    document.getElementById('cardBtcMa').textContent = Number(m.btc_weekly_ma20).toLocaleString();
    document.getElementById('cardBtcSub').innerHTML  =
      `현재 ${Number(m.btc_current||0).toLocaleString()} `+
      `<span style="color:${m.pass?'var(--green)':'var(--red)'}">${m.pass?'✅통과':'❌미통과'}</span>`;
  }

  // 탭 카운트
  document.getElementById('tabWatchCnt').textContent  = `(${data.watch_count ||0})`;
  document.getElementById('tabActiveCnt').textContent = `(${data.active_count||0})`;
  document.getElementById('tabDeepCnt').textContent   = `(${data.deep_count  ||0})`;

  // 렌더
  renderWatch(data.watch_list);
  renderActive(data.active_trades);
  renderDeep(data.active_trades);
  renderNew(data.new_entries);
  renderGradeBars(s.grade_stats);   // ← 고정 div에 innerHTML 교체
  renderDeepPanel(data.active_trades);
}

// ═══════════════════════════════════════════════════════════
//  폴링 & 액션
// ═══════════════════════════════════════════════════════════
async function fetchState() {
  try {
    const r = await fetch('/api/state');
    if(!r.ok) throw r.status;
    updateState(await r.json());
  } catch(e){ console.warn('fetchState:',e); }
}
async function fetchHistory() {
  try {
    const r = await fetch('/api/history');
    if(!r.ok) throw r.status;
    renderHistory(await r.json());
  } catch(e){ console.warn('fetchHistory:',e); }
}
async function triggerScan() {
  await fetch('/api/scan',{method:'POST'});
  setTimeout(fetchState,1500);
}
async function addWatch() {
  const v = document.getElementById('addInput').value.trim();
  if(!v) return;
  const r = await fetch('/api/watch/add',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ticker:v})
  });
  alert((await r.json()).message);
  document.getElementById('addInput').value='';
  fetchState();
}
async function removeWatch(tk) {
  if(!confirm(`${tk} 제거?`)) return;
  await fetch('/api/watch/remove',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ticker:tk})
  });
  fetchState();
}
async function closeTrade(tk) {
  if(!confirm(`${tk} 수동 청산?`)) return;
  const r = await fetch('/api/active/close',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ticker:tk})
  });
  alert((await r.json()).message);
  fetchState(); fetchHistory();
}

// 초기 실행
fetchState();
fetchHistory();
setInterval(fetchState,   15000);
setInterval(fetchHistory, 60000);

// 신규 카운트다운 (1초)
setInterval(()=>{
  const cd = document.getElementById('newCd');
  if(!_newRegAt||!cd) return;
  const rem = 300000-(Date.now()-_newRegAt);
  if(rem>0){
    const mm=Math.floor(rem/60000), ss=Math.floor((rem%60000)/1000);
    cd.textContent=`⏱ ${mm}분 ${ss}초 후 초기화`;
  } else {
    cd.textContent='초기화 대기 중...';
  }
},1000);
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
    ticker = data.get('ticker','').strip()
    if not ticker:
        return jsonify({'success': False, 'message': '티커를 입력하세요'})
    return jsonify(scanner.add_manual_watch(ticker))

@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data   = request.get_json() or {}
    ticker = data.get('ticker','').strip()
    if not ticker:
        return jsonify({'success': False, 'message': '티커를 입력하세요'})
    return jsonify(scanner.remove_watch(ticker))

@app.route('/api/active/close', methods=['POST'])
def api_active_close():
    data   = request.get_json() or {}
    ticker = data.get('ticker','').strip()
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
        threading.Thread(target=scanner.scanner_loop,        daemon=True, name='scanner_loop'),
        threading.Thread(target=scanner.watch_rescan_loop,   daemon=True, name='watch_rescan'),
        threading.Thread(target=scanner.price_check_loop,    daemon=True, name='price_check'),
        threading.Thread(target=scanner.active_monitor_loop, daemon=True, name='active_monitor'),
        threading.Thread(target=scanner.daily_summary_loop,  daemon=True, name='daily_summary'),
    ]
    for t in threads:
        t.start()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)), debug=False)
