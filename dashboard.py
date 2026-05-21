# -*- coding: utf-8 -*-
"""
dashboard.py — Upbit MTF 스캐너 Flask 대시보드 (v1.2)
변경사항:
  - 수동 스캔 버튼 (/api/scan POST)
  - 다음 스캔 카운트다운 (JS)
  - 현재가 / 등락률 / 등록가 대비 수익률 표시
  - Chart.js Stoch RSI 차트 (일봉/4h/1h)
  - 신호 히스토리 페이지 (/history)
  - 모바일 완전 최적화
"""

import os
import threading
from flask import Flask, jsonify, render_template_string, request

import scanner
import mtf_setup

app = Flask(__name__)

_scan_thread = threading.Thread(target=scanner.scanner_loop, daemon=True)
_scan_thread.start()

# ═══════════════════════════════════════════════════════════
#  메인 대시보드 HTML
# ═══════════════════════════════════════════════════════════
HTML = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Upbit MTF Scanner</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
/* ── 기본 ── */
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0e1117;--surface:#1a1d27;--border:#2a2d3a;
  --green:#4caf50;--red:#f44336;--orange:#ff9800;--blue:#90caf9;
  --text:#e0e0e0;--muted:#888;--dim:#555;
}
body{font-family:"Segoe UI",sans-serif;background:var(--bg);color:var(--text);font-size:14px}

/* ── 헤더 ── */
header{background:var(--surface);padding:12px 16px;border-bottom:1px solid var(--border);
       display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;
       position:sticky;top:0;z-index:100}
header h1{font-size:1.1rem;font-weight:700}
.header-right{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.btn-scan{background:#1e3a5f;color:#64b5f6;border:1px solid #2a5298;padding:6px 14px;
          border-radius:6px;cursor:pointer;font-size:0.82rem;font-weight:600;transition:.2s}
.btn-scan:hover{background:#2a5298}
.btn-scan:disabled{opacity:.5;cursor:not-allowed}
.countdown{font-size:0.78rem;color:var(--muted)}

/* ── 레이아웃 ── */
.container{max-width:1200px;margin:0 auto;padding:16px}

/* ── 카드 그리드 ── */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:20px}
.card{background:var(--surface);border-radius:8px;padding:14px;border:1px solid var(--border)}
.card .lbl{font-size:0.7rem;color:var(--muted);margin-bottom:4px}
.card .val{font-size:1.25rem;font-weight:700;line-height:1.2}
.card .sub{font-size:0.68rem;color:var(--dim);margin-top:3px}

/* ── 색상 ── */
.c-green{color:var(--green)} .c-red{color:var(--red)}
.c-orange{color:var(--orange)} .c-blue{color:var(--blue)}
.c-muted{color:var(--muted)}

/* ── 섹션 타이틀 ── */
.sec-title{font-size:0.95rem;font-weight:700;margin:20px 0 10px;color:#aaa;
           display:flex;align-items:center;gap:6px}

/* ── 테이블 ── */
.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border-radius:8px}
table{width:100%;border-collapse:collapse;background:var(--surface);min-width:500px}
th{background:#252836;padding:9px 12px;text-align:left;font-size:0.72rem;color:var(--muted);white-space:nowrap}
td{padding:9px 12px;font-size:0.82rem;border-top:1px solid var(--border);vertical-align:middle}

/* ── 배지 ── */
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:0.68rem;font-weight:700;margin-left:4px}
.b-new    {background:#1e3a5f;color:#64b5f6}
.b-signal {background:#1b3a1b;color:#66bb6a}
.b-removed{background:#3a1f1f;color:#ef9a9a}
.b-chart  {background:#2a2020;color:#ffcc02;cursor:pointer;border:none;font-size:0.68rem;
           font-weight:700;padding:2px 7px;border-radius:4px}

/* ── 매크로 블록 ── */
.macro-box{background:#131720;border:1px solid var(--border);border-radius:8px;
           padding:12px 14px;margin-bottom:18px}
.macro-box .title{font-size:0.75rem;color:#666;margin-bottom:10px}

/* ── 차트 모달 ── */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:200;
          align-items:center;justify-content:center;padding:16px}
.modal-bg.open{display:flex}
.modal{background:var(--surface);border-radius:10px;padding:18px;width:100%;
       max-width:700px;max-height:90vh;overflow-y:auto;border:1px solid var(--border)}
.modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.modal-header h3{font-size:1rem;font-weight:700}
.modal-close{background:none;border:none;color:var(--muted);font-size:1.4rem;cursor:pointer;line-height:1}
.chart-tabs{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.chart-tab{padding:5px 14px;border-radius:5px;border:1px solid var(--border);background:var(--bg);
           color:var(--muted);cursor:pointer;font-size:0.78rem;transition:.15s}
.chart-tab.active{background:#1e3a5f;color:#64b5f6;border-color:#2a5298}
.chart-wrap{position:relative;height:220px}

/* ── 상태 색상 ── */
.st-idle    {color:var(--muted)} .st-scanning{color:var(--orange)}
.st-done    {color:var(--green)} .st-error   {color:var(--red)}

/* ── 가격 표시 ── */
.price-up  {color:var(--green)} .price-dn{color:var(--red)} .price-flat{color:var(--muted)}

/* ── 수익률 바 ── */
.pnl-bar{display:inline-block;padding:1px 6px;border-radius:3px;font-size:0.72rem;font-weight:700}
.pnl-pos{background:#1b3a1b;color:#66bb6a} .pnl-neg{background:#3a1f1f;color:#ef9a9a}
.pnl-neu{background:#252836;color:#aaa}

/* ── 반응형 ── */
@media(max-width:600px){
  .grid{grid-template-columns:repeat(2,1fr)}
  .card .val{font-size:1.05rem}
  header h1{font-size:1rem}
  .btn-scan{padding:5px 10px;font-size:0.76rem}
  th,td{padding:7px 9px;font-size:0.76rem}
}
@media(max-width:360px){
  .grid{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>
<header>
  <h1>📡 Upbit MTF Scanner</h1>
  <div class="header-right">
    <span class="countdown" id="cd">—</span>
    <button class="btn-scan" id="scanBtn" onclick="triggerScan()">🔄 수동 스캔</button>
  </div>
</header>

<div class="container">

<!-- ── 상태 카드 ── -->
<div class="grid">
  <div class="card">
    <div class="lbl">스캐너 상태</div>
    <div class="val st-{{ state.status }}">{{ state.status.upper() }}</div>
  </div>
  <div class="card">
    <div class="lbl">마지막 스캔</div>
    <div class="val" style="font-size:.85rem">{{ (state.last_scan_at or '—')[:16] }}</div>
  </div>
  <div class="card">
    <div class="lbl">누적 스캔</div>
    <div class="val c-blue">{{ state.scan_count }}</div>
  </div>
  <div class="card">
    <div class="lbl">스캔 종목</div>
    <div class="val c-blue">{{ state.total_scanned }}</div>
  </div>
  <div class="card">
    <div class="lbl">Watch List</div>
    <div class="val c-green">{{ state.watch_list | length }}</div>
  </div>
  <div class="card">
    <div class="lbl">진입 신호</div>
    <div class="val {% if state.entry_signals %}c-green{% else %}c-muted{% endif %}">
      {{ state.entry_signals | length }}
    </div>
  </div>
</div>

<!-- ── 매크로 ── -->
{% set macro = state.macro or {} %}
<div class="macro-box">
  <div class="title">📊 BTC 매크로 필터</div>
  <div class="grid" style="margin-bottom:0">
    <div class="card" style="border-color:{% if macro.safe %}#2e7d32{% else %}#b71c1c{% endif %}">
      <div class="lbl">BTC 주봉 MA20 <span style="color:#ff9800">★필터</span></div>
      {% set w = macro.weekly_distance_pct or 0 %}
      <div class="val {% if macro.safe %}c-green{% else %}c-red{% endif %}">{{ '%+.2f'|format(w) }}%</div>
      <div class="sub">{% if macro.safe %}▲ MA20 위 — 허용{% else %}▼ MA20 아래 — 차단{% endif %}</div>
    </div>
    <div class="card">
      <div class="lbl">BTC 일봉 MA20 <span style="color:#555">(참고)</span></div>
      {% set d = macro.daily_distance_pct or 0 %}
      <div class="val {% if d>0 %}c-green{% elif d<-3 %}c-red{% else %}c-orange{% endif %}">{{ '%+.2f'|format(d) }}%</div>
      <div class="sub">{% if d>0 %}▲ 위{% else %}▼ 아래{% endif %} (필터 미적용)</div>
    </div>
    <div class="card">
      <div class="lbl">매크로 상태</div>
      <div class="val {% if macro.safe %}c-green{% else %}c-red{% endif %}" style="font-size:.9rem">{{ macro.state or '—' }}</div>
      <div class="sub" style="font-size:.65rem">{{ macro.reason or '' }}</div>
    </div>
  </div>
</div>

<!-- ── Watch List ── -->
<div class="sec-title">📋 Watch List ({{ state.watch_list | length }})</div>
{% if state.watch_list %}
<div class="tbl-wrap">
<table>
  <thead>
    <tr>
      <th>티커</th>
      <th>일봉K</th>
      <th>등록가</th>
      <th>현재가</th>
      <th>등락률</th>
      <th>수익률</th>
      <th>차트</th>
      <th>등록일</th>
    </tr>
  </thead>
  <tbody>
    {% for item in state.watch_list %}
    {% set cp = item.change_pct or 0 %}
    {% set ep = item.entry_price %}
    {% set cur = item.current_price %}
    {% if ep and cur and ep > 0 %}
      {% set pnl = (cur - ep) / ep * 100 %}
    {% else %}
      {% set pnl = none %}
    {% endif %}
    <tr>
      <td>
        <b>{{ item.ticker.replace('KRW-','') }}</b>
        <a href="https://upbit.com/exchange?code=CRIX.UPBIT.{{ item.ticker }}"
           target="_blank" style="color:#555;font-size:.7rem;margin-left:4px">↗</a>
      </td>
      <td><b>{{ '%.1f'|format(item.daily_short_k or 0) }}</b></td>
      <td style="color:var(--muted)">
        {% if ep %}{{ '{:,.0f}'.format(ep) if ep >= 100 else '{:.4f}'.format(ep) }}{% else %}—{% endif %}
      </td>
      <td class="{% if cp>0 %}price-up{% elif cp<0 %}price-dn{% else %}price-flat{% endif %}">
        {% if cur %}{{ '{:,.0f}'.format(cur) if cur >= 100 else '{:.4f}'.format(cur) }}{% else %}—{% endif %}
      </td>
      <td class="{% if cp>0 %}price-up{% elif cp<0 %}price-dn{% else %}price-flat{% endif %}">
        {% if cur %}{{ '%+.2f'|format(cp) }}%{% else %}—{% endif %}
      </td>
      <td>
        {% if pnl is not none %}
        <span class="pnl-bar {% if pnl>0 %}pnl-pos{% elif pnl<0 %}pnl-neg{% else %}pnl-neu{% endif %}">
          {{ '%+.2f'|format(pnl) }}%
        </span>
        {% else %}<span class="c-muted">—</span>{% endif %}
      </td>
      <td>
        <button class="b-chart badge"
          onclick="openChart('{{ item.ticker }}')">📈 차트</button>
      </td>
      <td style="color:var(--dim);font-size:.72rem">{{ item.registered_at[:10] }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% else %}
<p style="color:var(--dim);padding:16px 0">Watch List가 비어있습니다.</p>
{% endif %}

<!-- ── 진입 신호 ── -->
{% if state.entry_signals %}
<div class="sec-title">🚀 진입 트리거 (이번 스캔)</div>
<div class="tbl-wrap">
<table>
  <thead>
    <tr><th>티커</th><th>4h K</th><th>1h K</th><th>트리거</th><th>등록가</th><th>현재가</th><th>등락률</th></tr>
  </thead>
  <tbody>
    {% for sig in state.entry_signals %}
    {% set t  = sig.trigger %}
    {% set p  = state.current_prices.get(sig.ticker, {}) %}
    {% set cp = p.get('change_pct', 0) %}
    {% set cur = p.get('price') %}
    <tr>
      <td><b>{{ sig.ticker.replace('KRW-','') }}</b>
        <span class="badge b-signal">SIGNAL</span>
        <a href="https://upbit.com/exchange?code=CRIX.UPBIT.{{ sig.ticker }}"
           target="_blank" style="color:#555;font-size:.7rem;margin-left:2px">↗</a>
      </td>
      <td>{{ '%.1f'|format(t.h4_short_k or 0) }}</td>
      <td>{{ '%.1f'|format(t.h1_short_k or 0) }}</td>
      <td style="font-size:.75rem">{{ t.h1_trigger_type }}</td>
      <td style="color:var(--muted);font-size:.8rem">
        {% if sig.entry_price %}{{ '{:,.0f}'.format(sig.entry_price) if sig.entry_price>=100 else '{:.4f}'.format(sig.entry_price) }}{% else %}—{% endif %}
      </td>
      <td class="{% if cp>0 %}price-up{% elif cp<0 %}price-dn{% else %}price-flat{% endif %}">
        {% if cur %}{{ '{:,.0f}'.format(cur) if cur>=100 else '{:.4f}'.format(cur) }}{% else %}—{% endif %}
      </td>
      <td class="{% if cp>0 %}price-up{% elif cp<0 %}price-dn{% else %}price-flat{% endif %}">
        {% if cur %}{{ '%+.2f'|format(cp) }}%{% else %}—{% endif %}
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% endif %}

<!-- ── 신규 등록 ── -->
{% if state.new_entries %}
<div class="sec-title">✨ 신규 Watch 등록 (이번 스캔)</div>
<div class="tbl-wrap">
<table>
  <thead><tr><th>티커</th><th>일봉K</th><th>등록가</th><th>사유</th></tr></thead>
  <tbody>
    {% for e in state.new_entries %}
    <tr>
      <td><b>{{ e.ticker.replace('KRW-','') }}</b><span class="badge b-new">NEW</span></td>
      <td>{{ '%.1f'|format(e.daily_short_k or 0) }}</td>
      <td style="color:var(--muted)">
        {% if e.entry_price %}{{ '{:,.0f}'.format(e.entry_price) if e.entry_price>=100 else '{:.4f}'.format(e.entry_price) }}{% else %}—{% endif %}
      </td>
      <td style="font-size:.75rem;color:var(--muted)">{{ e.reason }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% endif %}

<!-- ── 제거 목록 ── -->
{% if state.removed %}
<div class="sec-title">🗑️ Watch 제거 (이번 스캔)</div>
<div class="tbl-wrap">
<table>
  <thead><tr><th>티커</th><th>유형</th><th>사유</th></tr></thead>
  <tbody>
    {% for r in state.removed %}
    <tr>
      <td><b>{{ r.ticker.replace('KRW-','') }}</b><span class="badge b-removed">REMOVED</span></td>
      <td style="font-size:.78rem">{{ r.removal_type }}</td>
      <td style="font-size:.75rem;color:var(--muted)">{{ r.reason }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% endif %}

<!-- ── 오류 ── -->
{% if state.error %}
<div class="sec-title" style="color:var(--red)">⚠️ 오류</div>
<div style="background:var(--surface);padding:12px;border-radius:6px;color:var(--red);font-size:.82rem">
  {{ state.error }}
</div>
{% endif %}

<!-- ── 하단 링크 ── -->
<div style="display:flex;justify-content:space-between;align-items:center;margin-top:20px;flex-wrap:wrap;gap:8px">
  <a href="/history" style="color:#64b5f6;font-size:.8rem;text-decoration:none">📜 신호 히스토리 →</a>
  <span style="font-size:.72rem;color:var(--dim)">MTF Scanner v1.2 | 60s 자동갱신</span>
</div>

</div><!-- /container -->

<!-- ═══ 차트 모달 ═══ -->
<div class="modal-bg" id="chartModal">
  <div class="modal">
    <div class="modal-header">
      <h3 id="chartTitle">Stoch RSI 차트</h3>
      <button class="modal-close" onclick="closeChart()">✕</button>
    </div>
    <div class="chart-tabs">
      <button class="chart-tab active" onclick="switchTab('daily',this)">일봉</button>
      <button class="chart-tab"        onclick="switchTab('h4',this)">4시간</button>
      <button class="chart-tab"        onclick="switchTab('h1',this)">1시간</button>
    </div>
    <div class="chart-wrap">
      <canvas id="stochChart"></canvas>
    </div>
    <div style="font-size:.7rem;color:var(--dim);margin-top:8px">
      🟡 K선 &nbsp; 🔵 D선 &nbsp; — 과매도(20) / 과매수(80) 기준선 표시
    </div>
  </div>
</div>

<script>
// ── 차트 데이터 (서버에서 주입) ──
const CHART_DATA = {{ chart_data_json }};
const NEXT_SCAN  = "{{ state.next_scan_at or '' }}";

// ── 카운트다운 ──
function updateCountdown(){
  if(!NEXT_SCAN) return;
  const diff = Math.max(0, Math.floor((new Date(NEXT_SCAN) - Date.now()) / 1000));
  const m = Math.floor(diff/60), s = diff%60;
  document.getElementById('cd').textContent =
    `다음 스캔 ${m}:${s.toString().padStart(2,'0')}`;
  if(diff > 0) setTimeout(updateCountdown, 1000);
}
updateCountdown();

// ── 수동 스캔 ──
function triggerScan(){
  const btn = document.getElementById('scanBtn');
  btn.disabled = true;
  btn.textContent = '⏳ 스캔 중...';
  fetch('/api/scan', {method:'POST'})
    .then(r => r.json())
    .then(d => {
      btn.textContent = '✅ 요청됨';
      setTimeout(() => location.reload(), 3000);
    })
    .catch(() => {
      btn.disabled = false;
      btn.textContent = '🔄 수동 스캔';
    });
}

// ── Stoch RSI 차트 ──
let currentTicker = null;
let currentTab    = 'daily';
let chartInstance = null;

function openChart(ticker){
  currentTicker = ticker;
  currentTab    = 'daily';
  document.querySelectorAll('.chart-tab').forEach((t,i) => t.classList.toggle('active', i===0));
  document.getElementById('chartTitle').textContent = ticker.replace('KRW-','') + ' · Stoch RSI';
  document.getElementById('chartModal').classList.add('open');
  renderChart();
}

function closeChart(){
  document.getElementById('chartModal').classList.remove('open');
  if(chartInstance){ chartInstance.destroy(); chartInstance = null; }
}

function switchTab(tab, el){
  currentTab = tab;
  document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  renderChart();
}

function renderChart(){
  const d = CHART_DATA[currentTicker];
  if(!d){ return; }
  const kKey = currentTab + '_k';
  const dKey = currentTab + '_d';
  const kArr = d[kKey] || [];
  const dArr = d[dKey] || [];
  const labels = kArr.map((_,i) => i+1);

  if(chartInstance) chartInstance.destroy();

  const ctx = document.getElementById('stochChart').getContext('2d');
  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'K',
          data: kArr,
          borderColor: '#ffcc02',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: 'D',
          data: dArr,
          borderColor: '#64b5f6',
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      scales: {
        x: { display: false },
        y: {
          min: 0, max: 100,
          ticks: { color:'#666', font:{size:10} },
          grid: { color:'#1e2130' },
        }
      },
      plugins: {
        legend: { labels: { color:'#aaa', font:{size:11}, boxWidth:12 } },
        annotation: {},
        tooltip: { mode:'index', intersect:false }
      }
    },
    plugins: [{
      id: 'refLines',
      beforeDraw(chart){
        const {ctx, scales:{y}} = chart;
        [20, 80].forEach(val => {
          const yPos = y.getPixelForValue(val);
          ctx.save();
          ctx.strokeStyle = val===20 ? 'rgba(76,175,80,.4)' : 'rgba(244,67,54,.4)';
          ctx.setLineDash([4,4]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(chart.chartArea.left,  yPos);
          ctx.lineTo(chart.chartArea.right, yPos);
          ctx.stroke();
          ctx.restore();
        });
      }
    }]
  });
}

// 모달 바깥 클릭 시 닫기
document.getElementById('chartModal').addEventListener('click', function(e){
  if(e.target === this) closeChart();
});
</script>
</body>
</html>'''

# ═══════════════════════════════════════════════════════════
#  신호 히스토리 페이지
# ═══════════════════════════════════════════════════════════
HISTORY_HTML = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>신호 히스토리 | MTF Scanner</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Segoe UI",sans-serif;background:#0e1117;color:#e0e0e0;font-size:14px}
header{background:#1a1d27;padding:12px 16px;border-bottom:1px solid #2a2d3a;
       display:flex;align-items:center;gap:12px}
header a{color:#64b5f6;text-decoration:none;font-size:.85rem}
header h1{font-size:1.05rem;font-weight:700}
.container{max-width:900px;margin:0 auto;padding:16px}
.sec-title{font-size:.95rem;font-weight:700;margin:16px 0 10px;color:#aaa}
.tbl-wrap{overflow-x:auto;border-radius:8px}
table{width:100%;border-collapse:collapse;background:#1a1d27;min-width:480px}
th{background:#252836;padding:9px 12px;text-align:left;font-size:.72rem;color:#888}
td{padding:9px 12px;font-size:.8rem;border-top:1px solid #2a2d3a}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:.68rem;font-weight:700}
.b-signal{background:#1b3a1b;color:#66bb6a}
.empty{color:#555;padding:20px 0;text-align:center}
a.back{display:inline-block;margin-bottom:16px;color:#64b5f6;text-decoration:none;font-size:.85rem}
</style>
</head>
<body>
<header>
  <a href="/">← 대시보드</a>
  <h1>📜 신호 히스토리</h1>
</header>
<div class="container">
  <div class="sec-title">진입 트리거 히스토리 (최근 {{ history|length }}건)</div>
  {% if history %}
  <div class="tbl-wrap">
  <table>
    <thead>
      <tr><th>#</th><th>티커</th><th>발생 시각</th><th>4h K</th><th>1h K</th><th>트리거</th><th>등록가</th></tr>
    </thead>
    <tbody>
      {% for sig in history|reverse %}
      {% set t = sig.trigger %}
      <tr>
        <td style="color:#555">{{ loop.index }}</td>
        <td>
          <b>{{ sig.ticker.replace('KRW-','') }}</b>
          <span class="badge b-signal">SIGNAL</span>
        </td>
        <td style="color:#888;font-size:.75rem">{{ sig.triggered_at[:16] }}</td>
        <td>{{ '%.1f'|format(t.h4_short_k or 0) }}</td>
        <td>{{ '%.1f'|format(t.h1_short_k or 0) }}</td>
        <td style="font-size:.75rem">{{ t.h1_trigger_type }}</td>
        <td style="color:#666;font-size:.78rem">
          {% if sig.entry_price %}
            {{ '{:,.0f}'.format(sig.entry_price) if sig.entry_price >= 100 else '{:.4f}'.format(sig.entry_price) }}
          {% else %}—{% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  </div>
  {% else %}
  <p class="empty">아직 진입 신호가 없습니다.</p>
  {% endif %}
</div>
</body>
</html>'''

# ═══════════════════════════════════════════════════════════
#  Flask 라우트
# ═══════════════════════════════════════════════════════════
import json as _json

@app.route('/')
def index():
    with scanner._state_lock:
        state = dict(scanner.scanner_state)
    # chart_data를 JSON 문자열로 변환 (템플릿에 직접 주입)
    chart_data_json = _json.dumps(state.get('chart_data', {}))
    return render_template_string(HTML, state=state, chart_data_json=chart_data_json)

@app.route('/history')
def history():
    hist = scanner.load_signal_history()
    return render_template_string(HISTORY_HTML, history=hist)

@app.route('/api/scan', methods=['POST'])
def api_scan():
    """수동 스캔 트리거."""
    with scanner._state_lock:
        status = scanner.scanner_state['status']
    if status == 'scanning':
        return jsonify({'ok': False, 'msg': '이미 스캔 중입니다.'})
    scanner.manual_scan()
    return jsonify({'ok': True, 'msg': '수동 스캔이 요청되었습니다.'})

@app.route('/api/state')
def api_state():
    with scanner._state_lock:
        return jsonify(scanner.scanner_state)

@app.route('/api/config')
def api_config():
    return jsonify(mtf_setup.get_module_config())

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
