# -*- coding: utf-8 -*-
"""
dashboard.py — Upbit MTF 스캐너 Flask 대시보드 (v2.1)
변경사항:
  - 시작 시 기존 watch_list / active_trades 즉시 로드
  - poll() 에러 핸들링 강화
  - fmt() 소수점 자동 처리
"""

import threading
import logging
import os
from flask import Flask, jsonify, request, render_template_string
import scanner
import mtf_setup

log = logging.getLogger(__name__)
app = Flask(__name__)


def _start_threads():
    # 시작 시 기존 데이터 즉시 로드
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
  h1{font-size:20px;font-weight:700}
  h2{font-size:15px;font-weight:600;margin:20px 0 10px;color:var(--sub)}

  /* 상단 카드 */
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:20px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px}
  .card .label{font-size:11px;color:var(--sub);margin-bottom:6px}
  .card .value{font-size:22px;font-weight:700}

  /* 통계 그리드 */
  .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:20px}
  .stat-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px}
  .stat-card .title{font-size:11px;color:var(--sub);margin-bottom:8px;font-weight:600}
  .stat-row{display:flex;justify-content:space-between;margin-bottom:4px;font-size:13px}

  /* 테이블 */
  .tbl-wrap{overflow-x:auto;margin-bottom:24px}
  table{width:100%;border-collapse:collapse;background:var(--card);border-radius:10px;overflow:hidden;min-width:600px}
  th{background:#12151f;color:var(--sub);font-weight:500;padding:10px 12px;text-align:left;font-size:12px;white-space:nowrap}
  td{padding:10px 12px;border-top:1px solid var(--border);vertical-align:middle;white-space:nowrap}
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
  .score-up{color:var(--green);font-weight:600}
  .score-dn{color:var(--red);font-weight:600}
  .score-eq{color:var(--sub)}

  /* 진행바 */
  .bar-wrap{background:#12151f;border-radius:4px;height:6px;min-width:80px}
  .bar{height:6px;border-radius:4px;transition:width .3s}
  .bar-green{background:var(--green)}
  .bar-red{background:var(--red)}

  /* 버튼 */
  .btn{padding:5px 12px;border-radius:6px;border:none;cursor:pointer;font-size:12px;font-weight:600;transition:opacity .2s}
  .btn:hover{opacity:.8}
  .btn-scan{background:var(--blue);color:#fff}
  .btn-close{background:var(--red);color:#fff}
  .btn-remove{background:#374151;color:var(--sub)}

  /* 수동 입력 */
  .form-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
  input[type=text],input[type=number]{
    background:#12151f;border:1px solid var(--border);color:var(--text);
    padding:6px 10px;border-radius:6px;font-size:13px;width:140px
  }
  input[type=text]:focus,input[type=number]:focus{
    outline:none;border-color:var(--blue)
  }

  /* 2컬럼 그리드 */
  .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}

  /* 상태 dot */
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle}
  .dot-green{background:var(--green)}
  .dot-yellow{background:var(--yellow);animation:blink 1s infinite}
  .dot-red{background:var(--red)}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

  .countdown{font-size:12px;color:var(--sub)}
  #msg{font-size:12px;color:var(--green);margin-left:8px}

  /* 모바일 */
  @media(max-width:768px){
    .cards{grid-template-columns:repeat(2,1fr)}
    .two-col{grid-template-columns:1fr}
    .hide-mobile{display:none}
    td,th{padding:8px 8px;font-size:12px}
    table{min-width:400px}
  }
  @media(max-width:480px){
    .stat-grid{grid-template-columns:1fr 1fr}
    .card .value{font-size:18px}
  }
</style>
</head>
<body>
<div class="wrap">

  <!-- 헤더 -->
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:16px">
    <h1>🔍 Upbit MTF Scanner</h1>
    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
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
  <div class="stat-grid" id="stat-grid">
    <div class="stat-card"><div class="title">로딩 중...</div></div>
  </div>

  <!-- Active 트레이드 -->
  <h2>🔥 Active 트레이드</h2>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>종목</th>
        <th>등급</th>
        <th>점수</th>
        <th>진입가</th>
        <th>현재가</th>
        <th>수익률</th>
        <th>진행</th>
        <th class="hide-mobile">진입시간</th>
        <th>청산</th>
      </tr></thead>
      <tbody id="active-tbody">
        <tr><td colspan="9" style="color:var(--sub);text-align:center;padding:20px">로딩 중...</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Watch List -->
  <h2>👁 Watch List</h2>
  <div class="form-row">
    <input type="text"   id="w-ticker" placeholder="종목 (예: BTC)">
    <input type="number" id="w-price"  placeholder="진입가">
    <button class="btn btn-scan" onclick="addWatch()">+ 수동 등록</button>
    <span id="msg"></span>
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>종목</th>
        <th>등급</th>
        <th>등록점수</th>
        <th>현재점수</th>
        <th>변화</th>
        <th class="hide-mobile">일봉K</th>
        <th class="hide-mobile">4hK</th>
        <th class="hide-mobile">1hK</th>
        <th>등록가</th>
        <th>현재가</th>
        <th>수익률</th>
        <th>구분</th>
        <th>삭제</th>
      </tr></thead>
      <tbody id="watch-tbody">
        <tr><td colspan="13" style="color:var(--sub);text-align:center;padding:20px">로딩 중...</td></tr>
      </tbody>
    </table>
  </div>

  <!-- 신규 / 만료 -->
  <div class="two-col">
    <div>
      <h2>🆕 신규 Watch</h2>
      <div class="tbl-wrap">
        <table>
          <thead><tr>
            <th>종목</th><th>등급</th><th>점수</th><th>일봉K</th>
          </tr></thead>
          <tbody id="new-tbody">
            <tr><td colspan="4" style="color:var(--sub);text-align:center;padding:16px">없음</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    <div>
      <h2>🗑 만료/제거</h2>
      <div class="tbl-wrap">
        <table>
          <thead><tr>
            <th>종목</th><th>등급</th><th>사유</th>
          </tr></thead>
          <tbody id="removed-tbody">
            <tr><td colspan="3" style="color:var(--sub);text-align:center;padding:16px">없음</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <p style="color:var(--sub);font-size:11px;margin-top:20px;text-align:center">
    마지막 스캔: <span id="last-scan">-</span> &nbsp;|&nbsp;
    가격 업데이트: <span id="last-price">-</span>
  </p>

</div><!-- /wrap -->

<script>
const GRADE_COLOR = {S:'#ef4444', A:'#f97316', B:'#f59e0b', C:'#94a3b8'};
const TP = {{ tp_pct }};
const SL = {{ sl_pct }};

// ── 포맷 헬퍼 ─────────────────────────────────
function fmt(v, dec=null){
  if(v == null || v === '' || isNaN(Number(v))) return '-';
  const n = Number(v);
  if(dec !== null){
    return n.toLocaleString('ko-KR',{minimumFractionDigits:dec, maximumFractionDigits:dec});
  }
  if(n >= 1000000) return n.toLocaleString('ko-KR', {maximumFractionDigits:0});
  if(n >= 100)     return n.toLocaleString('ko-KR', {maximumFractionDigits:0});
  if(n >= 10)      return n.toLocaleString('ko-KR', {maximumFractionDigits:1});
  if(n >= 1)       return n.toLocaleString('ko-KR', {maximumFractionDigits:2});
  if(n >= 0.1)     return n.toLocaleString('ko-KR', {maximumFractionDigits:3});
  if(n >= 0.01)    return n.toLocaleString('ko-KR', {maximumFractionDigits:4});
  return n.toLocaleString('ko-KR', {maximumFractionDigits:6});
}

function fmtPct(v){
  if(v == null) return '-';
  const n = Number(v);
  const s = n >= 0 ? '+' : '';
  const c = n >= 0 ? '#22c55e' : '#ef4444';
  return `<span style="color:${c}">${s}${n.toFixed(2)}%</span>`;
}

function gradeBadge(g){
  return `<span class="badge g-${g||'C'}">${g||'?'}</span>`;
}

function scoreChange(init, cur){
  const d = (cur||0) - (init||0);
  if(d > 0) return `<span class="score-up">+${d}↑</span>`;
  if(d < 0) return `<span class="score-dn">${d}↓</span>`;
  return `<span class="score-eq">→</span>`;
}

function fmtTime(iso){
  if(!iso) return '-';
  return new Date(iso).toLocaleString('ko-KR',{
    month:'2-digit', day:'2-digit',
    hour:'2-digit',  minute:'2-digit'
  });
}

// ── UI 업데이트 ───────────────────────────────
function updateUI(s){
  // 상태 dot
  const dot = document.getElementById('status-dot');
  if(s.status === 'scanning'){
    dot.innerHTML = '<span class="dot dot-yellow"></span>스캔 중...';
  } else if(s.status === 'done' || s.status === 'idle'){
    dot.innerHTML = '<span class="dot dot-green"></span>정상';
  } else {
    dot.innerHTML = `<span class="dot dot-red"></span>${s.error||'오류'}`;
  }

  document.getElementById('total-scanned').textContent = s.total_scanned || '-';
  document.getElementById('watch-count').textContent   = (s.watch_list||[]).length;
  document.getElementById('active-count').textContent  = (s.active_trades||[]).length;

  // 매크로
  const m  = s.macro || {};
  const wd = m.weekly_distance_pct;
  const dd = m.daily_distance_pct;
  document.getElementById('macro-weekly').innerHTML =
    wd != null ? `<span style="color:${wd>=0?'#22c55e':'#ef4444'}">${wd>=0?'+':''}${wd}%</span>` : '-';
  document.getElementById('macro-daily').innerHTML =
    dd != null ? `<span style="color:${dd>=0?'#22c55e':'#ef4444'}">${dd>=0?'+':''}${dd}%</span>` : '-';

  // 통계
  const st = s.stats || {};
  document.getElementById('tp-rate').textContent =
    st.tp_rate != null ? st.tp_rate + '%' : '-';
  renderStats(st);

  // 테이블
  renderActive(s.active_trades || []);
  renderWatch(s.watch_list     || []);
  renderNew(s.new_entries      || []);
  renderRemoved(s.removed_items|| []);

  // 시간
  if(s.last_scan_at)
    document.getElementById('last-scan').textContent  = fmtTime(s.last_scan_at);
  if(s.last_price_check_at)
    document.getElementById('last-price').textContent = fmtTime(s.last_price_check_at);
}

function renderStats(st){
  const gs = st.grade_stats || {};
  const gradeCards = ['S','A','B','C'].map(g => {
    const d = gs[g] || {};
    return `
      <div class="stat-card">
        <div class="title">${g}등급 통계</div>
        <div class="stat-row"><span>청산</span><span>${d.total||0}건</span></div>
        <div class="stat-row">
          <span>TP 승률</span>
          <span style="color:var(--green)">${d.tp_rate||0}%</span>
        </div>
        <div class="stat-row">
          <span>평균 수익</span>
          <span style="color:${(d.avg_pnl||0)>=0?'var(--green)':'var(--red)'}">
            ${d.avg_pnl!=null?(d.avg_pnl>=0?'+':'')+d.avg_pnl+'%':'-'}
          </span>
        </div>
      </div>`;
  }).join('');

  document.getElementById('stat-grid').innerHTML = `
    <div class="stat-card">
      <div class="title">📈 Watch 전체</div>
      <div class="stat-row"><span>총 등록</span><span>${st.total||0}건</span></div>
      <div class="stat-row">
        <span>Active 전환율</span>
        <span style="color:var(--green)">${st.watch_to_active_rate||0}%</span>
      </div>
      <div class="stat-row">
        <span>만료</span>
        <span style="color:var(--sub)">${st.expired||0}건</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="title">💰 Active 청산</div>
      <div class="stat-row">
        <span>TP</span>
        <span style="color:var(--green)">${st.tp||0}건 (${st.tp_rate||0}%)</span>
      </div>
      <div class="stat-row">
        <span>SL</span>
        <span style="color:var(--red)">${st.sl||0}건</span>
      </div>
      <div class="stat-row">
        <span>Timeout</span>
        <span style="color:var(--sub)">${st.timeout||0}건</span>
      </div>
      <div class="stat-row">
        <span>평균 수익률</span>
        <span style="color:${(st.avg_pnl||0)>=0?'var(--green)':'var(--red)'}">
          ${st.avg_pnl!=null?(st.avg_pnl>=0?'+':'')+st.avg_pnl+'%':'-'}
        </span>
      </div>
    </div>
    ${gradeCards}
  `;
}

function renderActive(trades){
  const tb = document.getElementById('active-tbody');
  if(!trades.length){
    tb.innerHTML = '<tr><td colspan="9" style="color:var(--sub);text-align:center;padding:20px">Active 트레이드 없음</td></tr>';
    return;
  }
  tb.innerHTML = trades.map(t => {
    const pnl   = t.pnl_pct || 0;
    const barW  = Math.min(Math.abs(pnl) / Math.max(TP, SL) * 100, 100);
    const barCl = pnl >= 0 ? 'bar-green' : 'bar-red';
    return `<tr>
      <td><b>${t.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(t.grade)}</td>
      <td>${t.entry_score||0}</td>
      <td>${fmt(t.entry_price)}</td>
      <td>${fmt(t.current_price)}</td>
      <td>${fmtPct(pnl)}</td>
      <td>
        <div class="bar-wrap">
          <div class="bar ${barCl}" style="width:${barW}%"></div>
        </div>
        <div style="font-size:10px;color:var(--sub);margin-top:2px">
          TP:+${TP}% / SL:-${SL}%
        </div>
      </td>
      <td class="hide-mobile" style="font-size:11px;color:var(--sub)">
        ${fmtTime(t.activated_at)}
      </td>
      <td>
        <button class="btn btn-close" onclick="closeTrade('${t.ticker}')">청산</button>
      </td>
    </tr>`;
  }).join('');
}

function renderWatch(list){
  const tb = document.getElementById('watch-tbody');
  if(!list.length){
    tb.innerHTML = '<tr><td colspan="13" style="color:var(--sub);text-align:center;padding:20px">Watch 종목 없음</td></tr>';
    return;
  }
  tb.innerHTML = list.map(w => {
    const snap      = w.snapshot || {};
    const cur       = w.current  || {};
    const initScore = snap.score  || 0;
    const curScore  = cur.score   || 0;
    const ep        = snap.entry_price || 0;
    const cp        = cur.price        || 0;
    const profit    = ep > 0 ? (cp - ep) / ep * 100 : 0;
    const curGrade  = cur.grade || snap.grade || 'C';
    const isManual  = w.manual;
    const gradeColor = GRADE_COLOR[curGrade] || '#94a3b8';

    return `<tr>
      <td><b>${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(curGrade)}</td>
      <td style="color:var(--sub)">${initScore}</td>
      <td style="color:${gradeColor};font-weight:700">${curScore}</td>
      <td>${scoreChange(initScore, curScore)}</td>
      <td class="hide-mobile" style="font-size:12px;color:var(--sub)">
        ${cur.daily_k ?? snap.daily_k ?? '-'}
      </td>
      <td class="hide-mobile" style="font-size:12px;color:var(--sub)">
        ${cur.h4_k ?? snap.h4_k ?? '-'}
      </td>
      <td class="hide-mobile" style="font-size:12px;color:var(--sub)">
        ${cur.h1_k ?? snap.h1_k ?? '-'}
      </td>
      <td>${fmt(ep)}</td>
      <td>${fmt(cp)}</td>
      <td>${fmtPct(profit)}</td>
      <td>
        <span class="badge ${isManual?'g-manual':'g-auto'}">
          ${isManual?'수동':'자동'}
        </span>
      </td>
      <td>
        <button class="btn btn-remove" onclick="removeWatch('${w.ticker}')">×</button>
      </td>
    </tr>`;
  }).join('');
}

function renderNew(list){
  const tb = document.getElementById('new-tbody');
  if(!list.length){
    tb.innerHTML = '<tr><td colspan="4" style="color:var(--sub);text-align:center;padding:16px">없음</td></tr>';
    return;
  }
  tb.innerHTML = list.map(w => {
    const cur  = w.current  || {};
    const snap = w.snapshot || {};
    return `<tr>
      <td><b>${w.ticker.replace('KRW-','')}</b></td>
      <td>${gradeBadge(cur.grade || snap.grade)}</td>
      <td>${cur.score || snap.score || 0}</td>
      <td style="font-size:12px">${cur.daily_k ?? snap.daily_k ?? '-'}</td>
    </tr>`;
  }).join('');
}

function renderRemoved(list){
  const tb = document.getElementById('removed-tbody');
  if(!list.length){
    tb.innerHTML = '<tr><td colspan="3" style="color:var(--sub);text-align:center;padding:16px">없음</td></tr>';
    return;
  }
  tb.innerHTML = list.map(w => {
    const cur  = w.current  || {};
    const snap = w.snapshot || {};
    const grade = cur.grade || snap.grade || w.grade || '?';
    return `<tr>
      <td><b>${(w.ticker||'').replace('KRW-','')}</b></td>
      <td>${gradeBadge(grade)}</td>
      <td style="color:var(--sub);font-size:11px">만료</td>
    </tr>`;
  }).join('');
}

// ── 카운트다운 ────────────────────────────────
let nextScanAt = null;
function updateCountdown(){
  if(!nextScanAt){
    document.getElementById('countdown').textContent = '';
    return;
  }
  const diff = Math.max(0, Math.floor((new Date(nextScanAt) - Date.now()) / 1000));
  const m    = Math.floor(diff / 60);
  const s    = diff % 60;
  document.getElementById('countdown').textContent =
    `다음 스캔 ${m}:${String(s).padStart(2,'0')}`;
}

// ── API 액션 ──────────────────────────────────
async function manualScan(){
  try {
    await fetch('/api/scan', {method:'POST'});
    showMsg('스캔 요청됨');
  } catch(e) { showMsg('요청 실패', true); }
}

async function addWatch(){
  const ticker = document.getElementById('w-ticker').value.trim();
  const price  = document.getElementById('w-price').value.trim();
  if(!ticker || !price){ showMsg('종목과 진입가를 입력하세요', true); return; }
  try {
    const r = await fetch('/api/watch/add', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ticker, entry_price: parseFloat(price)})
    });
    const d = await r.json();
    showMsg(d.message || (d.ok ? '등록 완료' : '실패'), !d.ok);
    if(d.ok){
      document.getElementById('w-ticker').value = '';
      document.getElementById('w-price').value  = '';
      await poll();
    }
  } catch(e) { showMsg('오류 발생', true); }
}

async function removeWatch(ticker){
  if(!confirm(`${ticker.replace('KRW-','')} Watch 삭제?`)) return;
  try {
    await fetch('/api/watch/remove', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ticker})
    });
    await poll();
  } catch(e) { showMsg('삭제 실패', true); }
}

async function closeTrade(ticker){
  if(!confirm(`${ticker.replace('KRW-','')} 수동 청산?`)) return;
  try {
    await fetch('/api/trade/close', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ticker})
    });
    await poll();
  } catch(e) { showMsg('청산 실패', true); }
}

function showMsg(text, isError=false){
  const el = document.getElementById('msg');
  el.style.color = isError ? '#ef4444' : '#22c55e';
  el.textContent = text;
  setTimeout(() => el.textContent = '', 3000);
}

// ── 폴링 ─────────────────────────────────────
async function poll(){
  try {
    const r = await fetch('/api/state');
    if(!r.ok) throw new Error(`HTTP ${r.status}`);
    const s = await r.json();
    nextScanAt = s.next_scan_at;
    updateUI(s);
  } catch(e) {
    console.error('poll error:', e);
  }
}

setInterval(poll,            15000);
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
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    price  = data.get('entry_price')
    if not ticker or price is None:
        return jsonify({'ok': False, 'message': '종목과 진입가 필요'}), 400
    try:
        price = float(price)
    except Exception:
        return jsonify({'ok': False, 'message': '진입가 형식 오류'}), 400
    ok, msg = scanner.add_manual_watch(ticker, price)
    return jsonify({'ok': ok, 'message': msg})

@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data   = request.get_json() or {}
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
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
