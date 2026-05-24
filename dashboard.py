"""
dashboard.py v3.0.0
Upbit MTF 스캐너 대시보드
- Watch / Active / DEEP / History 통합 탭
- StochRSI K+D 단기/중기/장기 표시
- DEEP 상대강도 탭
- 버전 배지 + 다음 스캔 카운트다운
"""

import os
import json
import threading
import logging
from datetime import datetime
from flask import Flask, jsonify, request, render_template_string

import scanner
import mtf_setup

DASHBOARD_VERSION = 'v3.0.0'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
# HTML 템플릿
# ══════════════════════════════════════════════════════════════

HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MTF Scanner Dashboard</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --sub: #8b949e; --green: #3fb950;
    --red: #f85149; --yellow: #d29922; --blue: #58a6ff;
    --purple: #bc8cff; --orange: #ffa657;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', sans-serif; font-size: 13px; }

  /* ── 헤더 ── */
  .header {
    background: var(--card); border-bottom: 1px solid var(--border);
    padding: 12px 20px; display: flex; align-items: center;
    justify-content: space-between; position: sticky; top: 0; z-index: 100;
  }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .header h1 { font-size: 16px; font-weight: 700; color: var(--blue); }
  .version-badge {
    background: #1f2937; border: 1px solid var(--border);
    border-radius: 12px; padding: 3px 10px; font-size: 11px;
    color: var(--sub); cursor: pointer; position: relative;
  }
  .version-badge:hover .tooltip { display: block; }
  .tooltip {
    display: none; position: absolute; top: 28px; left: 0;
    background: #1f2937; border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; width: 260px;
    font-size: 11px; color: var(--text); z-index: 200; white-space: pre-line;
  }
  .header-right { display: flex; align-items: center; gap: 12px; }
  .btc-info { font-size: 12px; color: var(--sub); }
  .btc-info span { color: var(--text); font-weight: 600; }
  .countdown { font-size: 12px; color: var(--sub); }
  .countdown span { color: var(--yellow); font-weight: 700; }
  .scan-btn {
    background: var(--blue); color: #000; border: none;
    border-radius: 6px; padding: 6px 14px; font-size: 12px;
    font-weight: 700; cursor: pointer;
  }
  .scan-btn:hover { opacity: 0.85; }

  /* ── 통계 카드 ── */
  .stats-grid {
    display: grid; grid-template-columns: repeat(6, 1fr);
    gap: 10px; padding: 14px 20px;
  }
  @media (max-width: 900px) {
    .stats-grid { grid-template-columns: repeat(3, 1fr); }
  }
  .stat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
  }
  .stat-label { font-size: 11px; color: var(--sub); margin-bottom: 4px; }
  .stat-value { font-size: 22px; font-weight: 700; }
  .stat-sub { font-size: 11px; color: var(--sub); margin-top: 2px; }

  /* ── 탭 ── */
  .tabs {
    display: flex; gap: 4px; padding: 0 20px;
    border-bottom: 1px solid var(--border);
  }
  .tab {
    padding: 8px 16px; cursor: pointer; border-radius: 6px 6px 0 0;
    font-size: 13px; color: var(--sub); border: 1px solid transparent;
    border-bottom: none; transition: all 0.15s;
  }
  .tab.active {
    background: var(--card); color: var(--text);
    border-color: var(--border);
  }
  .tab-content { display: none; padding: 16px 20px; }
  .tab-content.active { display: block; }

  /* ── 테이블 ── */
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    background: #1c2128; color: var(--sub); font-weight: 600;
    padding: 8px 10px; text-align: left; white-space: nowrap;
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 8px 10px; border-bottom: 1px solid #1c2128;
    white-space: nowrap;
  }
  tr:hover td { background: #1c2128; }

  /* ── 등급 배지 ── */
  .grade {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-weight: 700; font-size: 12px;
  }
  .g-S { background: #1a3a2a; color: var(--green); }
  .g-A { background: #1a2a3a; color: var(--blue); }
  .g-B { background: #2a2a1a; color: var(--yellow); }
  .g-C { background: #2a1a1a; color: var(--sub); }
  .g-X { background: #3a1a1a; color: var(--red); }

  /* ── 시그널 ── */
  .sig-BUY_OK  { color: var(--green); font-weight: 700; }
  .sig-BUY_NO  { color: var(--red);   font-weight: 700; }
  .sig-WATCH   { color: var(--yellow);}
  .sig-NEUTRAL { color: var(--sub);   }

  /* ── K/D 값 표시 ── */
  .kd-cell { font-size: 11px; }
  .kd-os  { color: var(--green); }
  .kd-ob  { color: var(--red);   }
  .kd-neu { color: var(--sub);   }
  .kd-gc  { color: var(--green); font-weight: 700; }
  .kd-dc  { color: var(--red);   font-weight: 700; }

  /* ── PnL ── */
  .pnl-pos { color: var(--green); font-weight: 700; }
  .pnl-neg { color: var(--red);   font-weight: 700; }

  /* ── 버튼 ── */
  .btn {
    border: none; border-radius: 5px; padding: 4px 10px;
    font-size: 11px; font-weight: 600; cursor: pointer;
  }
  .btn-entry  { background: #1a3a2a; color: var(--green); }
  .btn-remove { background: #3a1a1a; color: var(--red);   }
  .btn-close  { background: #2a1a1a; color: var(--orange);}
  .btn:hover  { opacity: 0.8; }

  /* ── 진입 가이드 ── */
  .guide-box {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;
    font-size: 12px;
  }
  .guide-box h3 { font-size: 13px; margin-bottom: 8px; color: var(--blue); }
  .guide-row { display: flex; gap: 20px; flex-wrap: wrap; }
  .guide-item { color: var(--sub); }
  .guide-item span { color: var(--text); }

  /* ── DEEP 카드 ── */
  .deep-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px;
  }
  .deep-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
  }
  .deep-card .ticker { font-size: 15px; font-weight: 700; margin-bottom: 6px; }
  .deep-card .rs-val { font-size: 20px; font-weight: 700; color: var(--orange); }
  .deep-card .deep-info { font-size: 11px; color: var(--sub); margin-top: 4px; }

  /* ── 빈 상태 ── */
  .empty-state {
    text-align: center; padding: 50px 20px; color: var(--sub);
  }
  .empty-state .icon { font-size: 36px; margin-bottom: 10px; }

  /* ── 로딩 ── */
  .loading { text-align: center; padding: 30px; color: var(--sub); }

  /* ── 알림 토스트 ── */
  #toast {
    position: fixed; bottom: 24px; right: 24px;
    background: #1f2937; border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 18px; font-size: 13px;
    display: none; z-index: 9999; max-width: 300px;
  }
</style>
</head>
<body>

<!-- ── 헤더 ── -->
<div class="header">
  <div class="header-left">
    <h1>📊 MTF Scanner</h1>
    <div class="version-badge">
      {{ dashboard_version }}
      <div class="tooltip">
Dashboard {{ dashboard_version }}
Scanner  {{ scanner_version }}
MTF Setup {{ mtf_version }}

📋 변경이력 (v3.0.0)
• StochRSI K+D 단기/중기/장기 도입
• DEEP 상대강도 탭 추가
• 골든크로스 기반 자동 Active 전환
• BUY_NO 진입 차단 로직
• 일봉 장기 K≤20 Watch 등록
      </div>
    </div>
  </div>
  <div class="header-right">
    <div class="btc-info">
      BTC <span id="btcPrice">-</span>
      <span id="btcMa20Badge" style="margin-left:6px;">-</span>
    </div>
    <div class="countdown">다음스캔 <span id="countdown">--:--</span></div>
    <button class="scan-btn" onclick="triggerScan()">🔄 즉시스캔</button>
  </div>
</div>

<!-- ── 통계 카드 ── -->
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-label">📋 Watch</div>
    <div class="stat-value" id="statWatch">-</div>
    <div class="stat-sub">감시 중</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">✅ Active</div>
    <div class="stat-value" id="statActive">-</div>
    <div class="stat-sub">진입 중</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">🔥 DEEP</div>
    <div class="stat-value" id="statDeep">-</div>
    <div class="stat-sub">상대강도</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">🎯 승률</div>
    <div class="stat-value" id="statWinrate">-</div>
    <div class="stat-sub" id="statTradeCount">-</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">💰 평균수익</div>
    <div class="stat-value" id="statAvgPnl">-</div>
    <div class="stat-sub">누적 평균</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">📡 스캔</div>
    <div class="stat-value" id="statScanCount">-</div>
    <div class="stat-sub">총 스캔 횟수</div>
  </div>
</div>

<!-- ── 탭 ── -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('watch')">📋 Watch</div>
  <div class="tab" onclick="switchTab('active')">✅ Active</div>
  <div class="tab" onclick="switchTab('deep')">🔥 DEEP</div>
  <div class="tab" onclick="switchTab('history')">📊 History</div>
</div>

<!-- ── Watch 탭 ── -->
<div id="tab-watch" class="tab-content active">
  <div class="guide-box">
    <h3>📌 Watch 진입 기준</h3>
    <div class="guide-row">
      <div class="guide-item">등록조건 <span>일봉 장기K ≤ 20</span></div>
      <div class="guide-item">자동진입 <span>일봉장기 과매도 + 4h/1h 골든크로스</span></div>
      <div class="guide-item">진입차단 <span>과매수(K≥80) + 데드크로스</span></div>
    </div>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>종목</th>
          <th>등급</th>
          <th>점수</th>
          <th>등록가</th>
          <th>일봉 장기 K/D</th>
          <th>일봉 중기 K/D</th>
          <th>일봉 단기 K/D</th>
          <th>4h K/D</th>
          <th>1h K/D</th>
          <th>4h GC</th>
          <th>1h GC</th>
          <th>거래량</th>
          <th>바닥일수</th>
          <th>등록일</th>
          <th>만료</th>
          <th>관리</th>
        </tr>
      </thead>
      <tbody id="watchTable">
        <tr><td colspan="16" class="loading">로딩 중...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ── Active 탭 ── -->
<div id="tab-active" class="tab-content">
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>종목</th>
          <th>등급</th>
          <th>진입가</th>
          <th>현재가</th>
          <th>수익률</th>
          <th>TP</th>
          <th>SL</th>
          <th>진입유형</th>
          <th>일봉장기K</th>
          <th>4h K/D</th>
          <th>1h K/D</th>
          <th>거래량</th>
          <th>진입일</th>
          <th>만료</th>
          <th>관리</th>
        </tr>
      </thead>
      <tbody id="activeTable">
        <tr><td colspan="15" class="loading">로딩 중...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ── DEEP 탭 ── -->
<div id="tab-deep" class="tab-content">
  <div class="guide-box">
    <h3>🔥 DEEP 상대강도 기준</h3>
    <div class="guide-row">
      <div class="guide-item">발동조건 <span>BTC 1h ≤ -1% 또는 4h ≤ -2%</span></div>
      <div class="guide-item">S등급 <span>RS ≥ +5% + 거래량 ≥ 2배</span></div>
      <div class="guide-item">A등급 <span>RS ≥ +3% + 거래량 ≥ 1.5배</span></div>
      <div class="guide-item">B등급 <span>RS ≥ +2% + 거래량 ≥ 1.3배</span></div>
    </div>
  </div>
  <div id="deepBtcInfo" style="padding:8px 0 12px; font-size:12px; color:var(--sub);"></div>
  <div id="deepGrid" class="deep-grid">
    <div class="loading">로딩 중...</div>
  </div>
</div>

<!-- ── History 탭 ── -->
<div id="tab-history" class="tab-content">
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>종목</th>
          <th>등급</th>
          <th>진입가</th>
          <th>종료가</th>
          <th>수익률</th>
          <th>종료사유</th>
          <th>진입유형</th>
          <th>진입일</th>
          <th>종료일</th>
        </tr>
      </thead>
      <tbody id="historyTable">
        <tr><td colspan="9" class="loading">로딩 중...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ── 토스트 ── -->
<div id="toast"></div>

<script>
// ── 전역 상태 ──────────────────────────────────────────────
let _state     = {};
let _nextScan  = null;
let _countdownTimer = null;

// ── 탭 전환 ────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    const names = ['watch','active','deep','history'];
    t.classList.toggle('active', names[i] === name);
  });
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
}

// ── 상태 갱신 ───────────────────────────────────────────────
async function fetchState() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();
    _state = d;
    renderAll(d);
  } catch(e) {
    console.error('fetchState 오류:', e);
  }
}

function renderAll(d) {
  renderStats(d);
  renderBtc(d);
  renderWatch(d.watch_list    || []);
  renderActive(d.active_list  || []);
  renderDeep(d.deep_list      || [], d);
  renderHistory(d.history     || []);
  updateCountdown(d.next_scan);
}

// ── 통계 카드 ───────────────────────────────────────────────
function renderStats(d) {
  const s = d.scanner_state || {};
  set('statWatch',      s.watch_count  ?? '-');
  set('statActive',     s.active_count ?? '-');
  set('statDeep',       s.deep_count   ?? '-');
  set('statScanCount',  s.scan_count   ?? '-');

  const total = s.total_trades || 0;
  const wins  = s.win_trades   || 0;
  const wr    = total > 0 ? Math.round(wins/total*100) : 0;
  const avg   = total > 0 ? (s.total_pnl / total).toFixed(2) : '0.00';

  const wrEl = document.getElementById('statWinrate');
  wrEl.textContent = total > 0 ? wr + '%' : '-';
  wrEl.style.color = wr >= 50 ? 'var(--green)' : 'var(--red)';

  const avgEl = document.getElementById('statAvgPnl');
  avgEl.textContent = total > 0 ? (avg > 0 ? '+' : '') + avg + '%' : '-';
  avgEl.style.color = avg >= 0 ? 'var(--green)' : 'var(--red)';

  set('statTradeCount', `${wins}승 / ${total}건`);
}

// ── BTC 정보 ────────────────────────────────────────────────
function renderBtc(d) {
  const s = d.scanner_state || {};
  const price = s.btc_price;
  set('btcPrice', price ? Number(price).toLocaleString() + ' KRW' : '-');

  const badge = document.getElementById('btcMa20Badge');
  if (s.btc_above_ma20 === true) {
    badge.textContent = '▲MA20';
    badge.style.color = 'var(--green)';
  } else if (s.btc_above_ma20 === false) {
    badge.textContent = '▼MA20';
    badge.style.color = 'var(--red)';
  } else {
    badge.textContent = '';
  }
}

// ── Watch 테이블 ────────────────────────────────────────────
function renderWatch(list) {
  const tb = document.getElementById('watchTable');
  if (!list.length) {
    tb.innerHTML = `<tr><td colspan="16"><div class="empty-state">
      <div class="icon">📋</div>일봉 장기K ≤ 20 조건 대기 중</div></td></tr>`;
    return;
  }

  list.sort((a,b) => (b.score||0) - (a.score||0));

  tb.innerHTML = list.map(w => {
    const regPrice = w.reg_price ? Number(w.reg_price).toLocaleString() : '-';
    const expDays  = daysLeft(w.expire_at);

    return `<tr>
      <td><b>${w.ticker}</b></td>
      <td>${gradeBadge(w.grade)}</td>
      <td>${scoreBar(w.score)}</td>
      <td style="font-size:11px;">${regPrice}</td>
      <td class="kd-cell">${kdCell(w.daily_long_k,  w.daily_long_d)}</td>
      <td class="kd-cell">${kdCell(w.daily_mid_k,   w.daily_mid_d)}</td>
      <td class="kd-cell">${kdCell(w.daily_short_k, w.daily_short_d)}</td>
      <td class="kd-cell">${kdCell(w.h4_short_k,    w.h4_short_d)}</td>
      <td class="kd-cell">${kdCell(w.h1_short_k,    w.h1_short_d)}</td>
      <td>${gcBadge(w.h4_gc)}</td>
      <td>${gcBadge(w.h1_gc)}</td>
      <td>${volBadge(w.vol_ratio)}</td>
      <td>${w.bottom_days ?? '-'}일</td>
      <td style="font-size:11px;">${fmtDate(w.added_at)}</td>
      <td style="font-size:11px;color:${expDays<=1?'var(--red)':'var(--sub)'}">
        ${expDays}일</td>
      <td>
        <button class="btn btn-entry"  onclick="activateWatch('${w.ticker}')">진입</button>
        <button class="btn btn-remove" onclick="removeWatch('${w.ticker}')">제거</button>
      </td>
    </tr>`;
  }).join('');
}

// ── Active 테이블 ───────────────────────────────────────────
function renderActive(list) {
  const tb = document.getElementById('activeTable');
  if (!list.length) {
    tb.innerHTML = `<tr><td colspan="15"><div class="empty-state">
      <div class="icon">✅</div>진입된 종목 없음</div></td></tr>`;
    return;
  }

  tb.innerHTML = list.map(a => {
    const pnl     = a.pnl_pct ?? 0;
    const pnlCls  = pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    const pnlTxt  = (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%';
    const typeLabel = a.trade_type === 'manual' ? '👤수동' : '🤖자동';
    const expDays  = daysLeft(a.expire_at);

    return `<tr>
      <td><b>${a.ticker}</b></td>
      <td>${gradeBadge(a.grade)}</td>
      <td>${Number(a.entry_price).toLocaleString()}</td>
      <td>${a.current_price ? Number(a.current_price).toLocaleString() : '-'}</td>
      <td class="${pnlCls}">${pnlTxt}</td>
      <td style="color:var(--green);font-size:11px;">
        ${Number(a.tp_price).toLocaleString()}</td>
      <td style="color:var(--red);font-size:11px;">
        ${Number(a.sl_price).toLocaleString()}</td>
      <td>${typeLabel}</td>
      <td class="kd-cell">${kdVal(a.daily_long_k)}</td>
      <td class="kd-cell">${kdCell(a.h4_short_k, a.h4_short_d)}</td>
      <td class="kd-cell">${kdCell(a.h1_short_k, a.h1_short_d)}</td>
      <td>${volBadge(a.vol_ratio)}</td>
      <td style="font-size:11px;">${fmtDate(a.entry_at)}</td>
      <td style="font-size:11px;color:${expDays<=4?'var(--yellow)':'var(--sub)'}">
        ${expDays}일</td>
      <td>
        <button class="btn btn-close" onclick="closeActive('${a.ticker}')">종료</button>
      </td>
    </tr>`;
  }).join('');
}

// ── DEEP 그리드 ─────────────────────────────────────────────
function renderDeep(list, d) {
  const s = d.scanner_state || {};
  const btcInfo = document.getElementById('deepBtcInfo');
  const p1h = s.btc_1h_pct;
  const p4h = s.btc_4h_pct;
  btcInfo.textContent = `BTC 변화율: 1h ${p1h != null ? (p1h>0?'+':'')+p1h+'%' : '-'} / 4h ${p4h != null ? (p4h>0?'+':'')+p4h+'%' : '-'} | 마지막 DEEP 스캔: ${s.last_deep_scan ? fmtDate(s.last_deep_scan) : '대기 중'}`;

  const grid = document.getElementById('deepGrid');
  if (!list.length) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1;">
      <div class="icon">🔥</div>BTC 하락 감지 시 자동 스캔 실행</div>`;
    return;
  }

  grid.innerHTML = list.map(item => {
    const gradeColor = {S:'var(--green)',A:'var(--blue)',B:'var(--yellow)',C:'var(--sub)'}[item.deep_grade] || 'var(--sub)';
    return `<div class="deep-card">
      <div class="ticker" style="color:${gradeColor}">
        ${gradeBadge(item.deep_grade)} ${item.ticker}
      </div>
      <div class="rs-val">+${item.rs}%</div>
      <div class="deep-info">
        코인변화: ${item.coin_pct > 0 ? '+' : ''}${item.coin_pct}%<br>
        BTC변화: ${item.btc_pct > 0 ? '+' : ''}${item.btc_pct}%<br>
        거래량: ${volBadge(item.vol_ratio)}<br>
        스캔: ${fmtDate(item.scanned_at)}
      </div>
    </div>`;
  }).join('');
}

// ── History 테이블 ──────────────────────────────────────────
function renderHistory(list) {
  const tb = document.getElementById('historyTable');
  if (!list.length) {
    tb.innerHTML = `<tr><td colspan="9"><div class="empty-state">
      <div class="icon">📊</div>종료된 거래 없음</div></td></tr>`;
    return;
  }

  const sorted = [...list].reverse();
  tb.innerHTML = sorted.map(h => {
    const pnl    = h.pnl_pct ?? 0;
    const pnlCls = pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    const pnlTxt = (pnl >= 0 ? '+' : '') + pnl.toFixed(2) + '%';
    const typeLabel = h.trade_type === 'manual' ? '👤수동' : '🤖자동';
    const reasonColor = {
      'TP':'var(--green)', 'SL':'var(--red)',
      '시간만료':'var(--yellow)', '수동종료':'var(--sub)'
    }[h.close_reason] || 'var(--sub)';

    return `<tr>
      <td><b>${h.ticker}</b></td>
      <td>${gradeBadge(h.grade)}</td>
      <td>${Number(h.entry_price).toLocaleString()}</td>
      <td>${h.close_price ? Number(h.close_price).toLocaleString() : '-'}</td>
      <td class="${pnlCls}">${pnlTxt}</td>
      <td style="color:${reasonColor}">${h.close_reason ?? '-'}</td>
      <td>${typeLabel}</td>
      <td style="font-size:11px;">${fmtDate(h.entry_at)}</td>
      <td style="font-size:11px;">${fmtDate(h.close_at)}</td>
    </tr>`;
  }).join('');
}

// ── 카운트다운 ───────────────────────────────────────────────
function updateCountdown(nextScan) {
  if (!nextScan) return;
  _nextScan = new Date(nextScan);
  if (_countdownTimer) clearInterval(_countdownTimer);
  _countdownTimer = setInterval(() => {
    const diff = Math.max(0, Math.floor((_nextScan - Date.now()) / 1000));
    const m    = String(Math.floor(diff / 60)).padStart(2, '0');
    const s    = String(diff % 60).padStart(2, '0');
    const el   = document.getElementById('countdown');
    if (el) el.textContent = `${m}:${s}`;
    if (diff === 0) clearInterval(_countdownTimer);
  }, 1000);
}

// ── API 액션 ────────────────────────────────────────────────
async function triggerScan() {
  showToast('🔄 스캔 요청 중...');
  const r = await fetch('/api/scan', {method:'POST'});
  const d = await r.json();
  showToast(d.success ? '✅ 스캔 시작됨' : '❌ ' + d.message);
  setTimeout(fetchState, 3000);
}

async function activateWatch(ticker) {
  if (!confirm(`${ticker} 를 Active로 전환하시겠습니까?`)) return;
  const r = await fetch('/api/watch/activate', {
    method:  'POST',
    headers: {'Content-Type':'application/json'},
    body:    JSON.stringify({ticker})
  });
  const d = await r.json();
  showToast(d.success ? `✅ ${ticker} Active 전환` : '❌ ' + d.message);
  if (d.success) fetchState();
}

async function removeWatch(ticker) {
  if (!confirm(`${ticker} 를 Watch에서 제거하시겠습니까?`)) return;
  const r = await fetch('/api/watch/remove', {
    method:  'POST',
    headers: {'Content-Type':'application/json'},
    body:    JSON.stringify({ticker})
  });
  const d = await r.json();
  showToast(d.success ? `🗑️ ${ticker} 제거됨` : '❌ ' + d.message);
  if (d.success) fetchState();
}

async function closeActive(ticker) {
  if (!confirm(`${ticker} 포지션을 수동 종료하시겠습니까?`)) return;
  const r = await fetch('/api/active/close', {
    method:  'POST',
    headers: {'Content-Type':'application/json'},
    body:    JSON.stringify({ticker, reason:'수동종료'})
  });
  const d = await r.json();
  showToast(d.success ? `🔴 ${ticker} 종료됨` : '❌ ' + d.message);
  if (d.success) fetchState();
}

// ── 렌더 헬퍼 ──────────────────────────────────────────────
function set(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function gradeBadge(g) {
  const cls = {'S':'g-S','A':'g-A','B':'g-B','C':'g-C','X':'g-X'}[g] || 'g-C';
  return `<span class="grade ${cls}">${g || '-'}</span>`;
}

function scoreBar(score) {
  const s   = score ?? 0;
  const pct = Math.min(s, 100);
  const col = s >= 70 ? 'var(--green)' : s >= 40 ? 'var(--yellow)' : 'var(--sub)';
  return `<div style="display:flex;align-items:center;gap:5px;">
    <div style="width:50px;height:6px;background:#1c2128;border-radius:3px;">
      <div style="width:${pct}%;height:100%;background:${col};border-radius:3px;"></div>
    </div>
    <span style="color:${col};font-size:11px;">${s}</span>
  </div>`;
}

function kdCell(k, d) {
  if (k == null) return '<span class="kd-neu">-</span>';
  const kRound = Math.round(k * 10) / 10;
  const dRound = d != null ? Math.round(d * 10) / 10 : null;

  let kCls = 'kd-neu';
  if (k <= 20)  kCls = 'kd-os';
  if (k >= 80)  kCls = 'kd-ob';

  let gcMark = '';
  if (d != null) {
    if (k > d)  gcMark = '<span class="kd-gc"> ↑</span>';
    if (k < d)  gcMark = '<span class="kd-dc"> ↓</span>';
  }

  const dStr = dRound != null
    ? `<span class="kd-neu">/ ${dRound}</span>`
    : '';

  return `<span class="${kCls}">${kRound}</span>${dStr}${gcMark}`;
}

function kdVal(k) {
  if (k == null) return '<span class="kd-neu">-</span>';
  const r    = Math.round(k * 10) / 10;
  let   cls  = 'kd-neu';
  if (k <= 20) cls = 'kd-os';
  if (k >= 80) cls = 'kd-ob';
  return `<span class="${cls}">${r}</span>`;
}

function gcBadge(gc) {
  return gc
    ? '<span style="color:var(--green);font-weight:700;">✨GC</span>'
    : '<span style="color:var(--sub);">-</span>';
}

function volBadge(ratio) {
  if (ratio == null) return '<span style="color:var(--sub);">-</span>';
  const r   = ratio.toFixed(1);
  const col = ratio >= 2.0 ? 'var(--orange)'
            : ratio >= 1.5 ? 'var(--yellow)'
            : 'var(--sub)';
  const icon = ratio >= 2.0 ? '🔥' : ratio >= 1.5 ? '⚡' : '';
  return `<span style="color:${col};">${icon}${r}x</span>`;
}

function fmtDate(iso) {
  if (!iso) return '-';
  try {
    const d = new Date(iso);
    const M = String(d.getMonth()+1).padStart(2,'0');
    const D = String(d.getDate()).padStart(2,'0');
    const h = String(d.getHours()).padStart(2,'0');
    const m = String(d.getMinutes()).padStart(2,'0');
    return `${M}/${D} ${h}:${m}`;
  } catch { return '-'; }
}

function daysLeft(iso) {
  if (!iso) return 0;
  try {
    const diff = new Date(iso) - Date.now();
    return Math.max(0, Math.ceil(diff / 86400000));
  } catch { return 0; }
}

function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.display = 'block';
  setTimeout(() => { el.style.display = 'none'; }, 3000);
}

// ── 초기화 ──────────────────────────────────────────────────
fetchState();
setInterval(fetchState, 15000);
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════
# Flask 라우트
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(
        HTML,
        dashboard_version = DASHBOARD_VERSION,
        scanner_version   = scanner.VERSION,
        mtf_version       = mtf_setup.VERSION,
    )


@app.route('/api/version')
def api_version():
    return jsonify({
        'dashboard': DASHBOARD_VERSION,
        'scanner':   scanner.VERSION,
        'mtf_setup': mtf_setup.VERSION,
    })


@app.route('/api/state')
def api_state():
    with scanner._state_lock:
        state = dict(scanner._scanner_state)

    history = scanner.load_history()

    # 누적 통계 보정
    total = len(history)
    wins  = sum(1 for h in history if h.get('pnl_pct', 0) > 0)
    pnl   = sum(h.get('pnl_pct', 0) for h in history)

    state['total_trades'] = total
    state['win_trades']   = wins
    state['total_pnl']    = round(pnl, 2)

    return jsonify({
        'scanner_state': state,
        'watch_list':    scanner.load_watch_list(),
        'active_list':   scanner.load_active_list(),
        'deep_list':     scanner.load_deep_list(),
        'history':       history[-50:],
        'next_scan':     state.get('next_scan'),
    })


@app.route('/api/scan', methods=['POST'])
def api_scan():
    t = threading.Thread(
        target=scanner.run_single_scan, daemon=True, name='manual_scan'
    )
    t.start()
    return jsonify({'success': True, 'message': '스캔 시작됨'})


@app.route('/api/watch/add', methods=['POST'])
def api_watch_add():
    data   = request.get_json(force=True) or {}
    ticker = data.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({'success': False, 'message': 'ticker 필요'})
    result = scanner.manual_add_watch(ticker)
    return jsonify(result)


@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data   = request.get_json(force=True) or {}
    ticker = data.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({'success': False, 'message': 'ticker 필요'})
    result = scanner.manual_remove_watch(ticker)
    return jsonify(result)


@app.route('/api/watch/activate', methods=['POST'])
def api_watch_activate():
    data   = request.get_json(force=True) or {}
    ticker = data.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({'success': False, 'message': 'ticker 필요'})
    result = scanner.manual_activate_watch(ticker)
    return jsonify(result)


@app.route('/api/active/close', methods=['POST'])
def api_active_close():
    data   = request.get_json(force=True) or {}
    ticker = data.get('ticker', '').strip().upper()
    reason = data.get('reason', '수동종료')
    if not ticker:
        return jsonify({'success': False, 'message': 'ticker 필요'})
    result = scanner.manual_close_active(ticker, reason)
    return jsonify(result)


@app.route('/api/watch/reset', methods=['POST'])
def api_watch_reset():
    result = scanner.reset_watch_list()
    return jsonify(result)


@app.route('/api/config')
def api_config():
    return jsonify({
        'scan_interval_min':         scanner.SCAN_INTERVAL_MIN,
        'watch_rescan_interval_min': scanner.WATCH_RESCAN_INTERVAL_MIN,
        'price_check_interval_min':  scanner.PRICE_CHECK_INTERVAL_MIN,
        'deep_scan_interval_min':    scanner.DEEP_SCAN_INTERVAL_MIN,
        'trade_tp_pct':              scanner.TRADE_TP_PCT,
        'trade_sl_pct':              scanner.TRADE_SL_PCT,
        'trade_timeout_h':           scanner.TRADE_TIMEOUT_H,
        'btc_drop_1h_pct':           scanner.BTC_DROP_1H_PCT,
        'btc_drop_4h_pct':           scanner.BTC_DROP_4H_PCT,
        'max_workers':               scanner.MAX_WORKERS,
        'candle_count':              scanner.CANDLE_COUNT,
    })


# ══════════════════════════════════════════════════════════════
# 실행 진입점
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    threads = [
        threading.Thread(target=scanner.scanner_loop,        daemon=True, name='scanner_loop'),
        threading.Thread(target=scanner.watch_rescan_loop,   daemon=True, name='watch_rescan'),
        threading.Thread(target=scanner.price_check_loop,    daemon=True, name='price_check'),
        threading.Thread(target=scanner.active_monitor_loop, daemon=True, name='active_monitor'),
        threading.Thread(target=scanner.deep_scan_loop,      daemon=True, name='deep_scan'),
        threading.Thread(target=scanner.daily_summary_loop,  daemon=True, name='daily_summary'),
    ]
    for t in threads:
        t.start()

    print(f'✅ Dashboard {DASHBOARD_VERSION} + Scanner {scanner.VERSION} 시작')
    print(f'   MTF Setup: {mtf_setup.VERSION}')
    print(f'   루프 6개 시작: scanner / watch_rescan / price_check / active_monitor / deep_scan / daily_summary')

    port = int(os.environ.get('PORT', 5000))
    print(f'🚀 http://0.0.0.0:{port}')
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
