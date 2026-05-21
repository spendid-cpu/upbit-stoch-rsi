# -*- coding: utf-8 -*-
"""
dashboard.py — Upbit MTF 스캐너 Flask 대시보드 (v1.4)
변경사항:
  - 수동 Watch 등록 폼 (티커 + 진입가)
  - Watch List 각 행에 삭제 버튼
  - 수동 등록 종목 MANUAL 배지 표시
  - 가격 모니터링 루프 스레드 추가
"""

import os
import json as _json
import threading
from flask import Flask, jsonify, render_template_string, request

import scanner
import mtf_setup

app = Flask(__name__)

# 메인 스캔 루프
_scan_thread = threading.Thread(target=scanner.scanner_loop, daemon=True)
_scan_thread.start()

# 가격 모니터링 루프 (5분)
_price_thread = threading.Thread(target=scanner.price_check_loop, daemon=True)
_price_thread.start()

HTML = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Upbit MTF Scanner</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0e1117;--surface:#1a1d27;--border:#2a2d3a;
  --green:#4caf50;--red:#f44336;--orange:#ff9800;--blue:#90caf9;--yellow:#ffcc02;
  --text:#e0e0e0;--muted:#888;--dim:#555;
}
body{font-family:"Segoe UI",sans-serif;background:var(--bg);color:var(--text);font-size:14px}
header{background:var(--surface);padding:12px 16px;border-bottom:1px solid var(--border);
       display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;
       position:sticky;top:0;z-index:100}
header h1{font-size:1.1rem;font-weight:700}
.header-right{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.btn{padding:6px 14px;border-radius:6px;cursor:pointer;font-size:.82rem;font-weight:600;
     border:1px solid;transition:.2s;line-height:1}
.btn-scan{background:#1e3a5f;color:#64b5f6;border-color:#2a5298}
.btn-scan:hover{background:#2a5298}
.btn:disabled{opacity:.5;cursor:not-allowed}
.countdown{font-size:.78rem;color:var(--muted)}
.container{max-width:1200px;margin:0 auto;padding:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.card{background:var(--surface);border-radius:8px;padding:13px;border:1px solid var(--border)}
.card .lbl{font-size:.68rem;color:var(--muted);margin-bottom:4px}
.card .val{font-size:1.2rem;font-weight:700;line-height:1.2}
.card .sub{font-size:.65rem;color:var(--dim);margin-top:3px}
.c-green{color:var(--green)}.c-red{color:var(--red)}
.c-orange{color:var(--orange)}.c-blue{color:var(--blue)}
.c-yellow{color:var(--yellow)}.c-muted{color:var(--muted)}
.sec-title{font-size:.92rem;font-weight:700;margin:16px 0 8px;color:#aaa}
.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border-radius:8px}
table{width:100%;border-collapse:collapse;background:var(--surface);min-width:460px}
th{background:#252836;padding:8px 11px;text-align:left;font-size:.7rem;color:var(--muted);white-space:nowrap}
td{padding:8px 11px;font-size:.8rem;border-top:1px solid var(--border);vertical-align:middle}
.badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:.63rem;font-weight:700;margin-left:3px}
.b-new    {background:#1e3a5f;color:#64b5f6}
.b-signal {background:#1b3a1b;color:#66bb6a}
.b-removed{background:#3a1f1f;color:#ef9a9a}
.b-manual {background:#2a1a40;color:#ce93d8}
.b-auto   {background:#1a2a1a;color:#a5d6a7}
.b-tp     {background:#1b3a1b;color:#66bb6a}
.b-sl     {background:#3a1f1f;color:#ef9a9a}
.b-timeout{background:#2a2010;color:#ffcc02}
.b-active {background:#1e2a3a;color:#90caf9}
.b-chart  {background:#2a2020;color:#ffcc02;cursor:pointer;border:none;
           font-size:.63rem;font-weight:700;padding:2px 6px;border-radius:3px}
.btn-del  {background:#3a1f1f;color:#ef9a9a;border:none;cursor:pointer;
           font-size:.63rem;font-weight:700;padding:2px 7px;border-radius:3px;
           transition:.15s}
.btn-del:hover{background:#5a2f2f}
.macro-box{background:#131720;border:1px solid var(--border);border-radius:8px;
           padding:11px 13px;margin-bottom:14px}
.macro-box .title{font-size:.72rem;color:#666;margin-bottom:8px}

/* ── 수동 등록 폼 ── */
.add-form{background:var(--surface);border:1px solid var(--border);border-radius:8px;
          padding:14px 16px;margin-bottom:16px}
.add-form .form-title{font-size:.82rem;font-weight:700;color:#aaa;margin-bottom:10px}
.form-row{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end}
.form-group{display:flex;flex-direction:column;gap:4px}
.form-group label{font-size:.68rem;color:var(--muted)}
.form-input{background:#252836;border:1px solid var(--border);color:var(--text);
            padding:7px 10px;border-radius:6px;font-size:.82rem;width:160px;outline:none;
            transition:.2s}
.form-input:focus{border-color:#2a5298}
.form-input::placeholder{color:#555}
.btn-add{background:#1b3a1b;color:#66bb6a;border:1px solid #2e5a2e;
         padding:7px 16px;border-radius:6px;cursor:pointer;font-size:.82rem;
         font-weight:600;white-space:nowrap;transition:.2s}
.btn-add:hover{background:#2e5a2e}
.form-msg{font-size:.76rem;margin-top:6px;padding:4px 8px;border-radius:4px}
.form-msg.ok {background:#1b3a1b;color:#66bb6a}
.form-msg.err{background:#3a1f1f;color:#ef9a9a}

/* TP/SL 진행바 */
.progress-wrap{background:#252836;border-radius:3px;height:5px;margin-top:3px;overflow:hidden}
.progress-fill{height:100%;border-radius:3px}
.fill-green{background:var(--green)}.fill-red{background:var(--red)}

/* 승률 바 */
.stat-bar{background:#252836;border-radius:3px;height:7px;margin-top:5px;overflow:hidden;display:flex}
.sw{background:var(--green)}.sl{background:var(--red)}.st{background:var(--yellow)}

/* 차트 모달 */
.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:200;
          align-items:center;justify-content:center;padding:16px}
.modal-bg.open{display:flex}
.modal{background:var(--surface);border-radius:10px;padding:18px;width:100%;
       max-width:700px;max-height:90vh;overflow-y:auto;border:1px solid var(--border)}
.modal-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.modal-header h3{font-size:.98rem;font-weight:700}
.modal-close{background:none;border:none;color:var(--muted);font-size:1.4rem;cursor:pointer}
.chart-tabs{display:flex;gap:7px;margin-bottom:10px;flex-wrap:wrap}
.chart-tab{padding:4px 13px;border-radius:5px;border:1px solid var(--border);
           background:var(--bg);color:var(--muted);cursor:pointer;font-size:.76rem}
.chart-tab.active{background:#1e3a5f;color:#64b5f6;border-color:#2a5298}
.chart-wrap{position:relative;height:210px}

.st-idle{color:var(--muted)}.st-scanning{color:var(--orange)}
.st-done{color:var(--green)}.st-error{color:var(--red)}
.price-up{color:var(--green)}.price-dn{color:var(--red)}.price-flat{color:var(--muted)}
.pnl-bar{display:inline-block;padding:1px 6px;border-radius:3px;font-size:.7rem;font-weight:700}
.pnl-pos{background:#1b3a1b;color:#66bb6a}
.pnl-neg{background:#3a1f1f;color:#ef9a9a}
.pnl-neu{background:#252836;color:#aaa}

@media(max-width:600px){
  .grid{grid-template-columns:repeat(2,1fr)}
  .card .val{font-size:1rem}
  header h1{font-size:.95rem}
  th,td{padding:6px 8px;font-size:.73rem}
  .form-input{width:130px}
}
</style>
</head>
<body>
<header>
  <h1>📡 Upbit MTF Scanner</h1>
  <div class="header-right">
    <span class="countdown" id="cd">—</span>
    <button class="btn btn-scan" id="scanBtn" onclick="triggerScan()">🔄 수동 스캔</button>
  </div>
</header>

<div class="container">

<!-- 상태 카드 -->
<div class="grid">
  <div class="card">
    <div class="lbl">스캐너 상태</div>
    <div class="val st-{{ state.status }}">{{ state.status.upper() }}</div>
    <div class="sub">가격체크 {{ state.last_price_check_at[:16] if state.last_price_check_at else '—' }}</div>
  </div>
  <div class="card">
    <div class="lbl">마지막 스캔</div>
    <div class="val" style="font-size:.82rem">{{ (state.last_scan_at or '—')[:16] }}</div>
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
    <div class="lbl">활성 트레이드</div>
    <div class="val c-yellow">{{ state.active_trades | length }}</div>
  </div>
  <div class="card">
    <div class="lbl">진입 신호 (이번)</div>
    <div class="val {% if state.entry_signals %}c-green{% else %}c-muted{% endif %}">
      {{ state.entry_signals | length }}
    </div>
  </div>
</div>

<!-- 승률 카드 -->
{% set stats = state.trade_stats or {} %}
{% if stats.total %}
<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(140px,1fr));margin-bottom:14px">
  <div class="card" style="border-color:#2e7d32">
    <div class="lbl">누적 승률</div>
    <div class="val c-green">{{ '%.1f'|format(stats.win_rate or 0) }}%</div>
    <div class="sub">{{ stats.win }}승 {{ stats.loss }}패 {{ stats.timeout }}타임아웃</div>
    {% set t=stats.total %}
    <div class="stat-bar">
      <div class="sw" style="width:{{ (stats.win/t*100)|round }}%"></div>
      <div class="sl" style="width:{{ (stats.loss/t*100)|round }}%"></div>
      <div class="st" style="width:{{ (stats.timeout/t*100)|round }}%"></div>
    </div>
  </div>
  <div class="card">
    <div class="lbl">평균 PnL</div>
    <div class="val {% if (stats.avg_pnl or 0)>0 %}c-green{% else %}c-red{% endif %}">
      {{ '%+.2f'|format(stats.avg_pnl or 0) }}%
    </div>
    <div class="sub">총 {{ stats.total }}건</div>
  </div>
  <div class="card">
    <div class="lbl">최고 / 최저</div>
    <div class="val" style="font-size:.9rem">
      <span class="c-green">{{ '%+.2f'|format(stats.best_pnl or 0) }}%</span> /
      <span class="c-red">{{ '%+.2f'|format(stats.worst_pnl or 0) }}%</span>
    </div>
    <div class="sub">TP+{{ tp_pct }}% SL-{{ sl_pct }}% {{ timeout_h }}h</div>
  </div>
</div>
{% endif %}

<!-- 매크로 -->
{% set macro = state.macro or {} %}
<div class="macro-box">
  <div class="title">📊 BTC 매크로 필터</div>
  <div class="grid" style="margin-bottom:0">
    <div class="card" style="border-color:{% if macro.safe %}#2e7d32{% else %}#b71c1c{% endif %}">
      <div class="lbl">BTC 주봉 MA20 <span style="color:#ff9800">★필터</span></div>
      {% set w = macro.weekly_distance_pct or 0 %}
      <div class="val {% if macro.safe %}c-green{% else %}c-red{% endif %}">{{ '%+.2f'|format(w) }}%</div>
      <div class="sub">{% if macro.safe %}▲ 허용{% else %}▼ 차단{% endif %}</div>
    </div>
    <div class="card">
      <div class="lbl">BTC 일봉 MA20 <span style="color:#555">(참고)</span></div>
      {% set d = macro.daily_distance_pct or 0 %}
      <div class="val {% if d>0 %}c-green{% elif d<-3 %}c-red{% else %}c-orange{% endif %}">{{ '%+.2f'|format(d) }}%</div>
      <div class="sub">{% if d>0 %}▲ 위{% else %}▼ 아래{% endif %} (미적용)</div>
    </div>
    <div class="card">
      <div class="lbl">매크로 상태</div>
      <div class="val {% if macro.safe %}c-green{% else %}c-red{% endif %}" style="font-size:.88rem">{{ macro.state or '—' }}</div>
      <div class="sub" style="font-size:.63rem">{{ macro.reason or '' }}</div>
    </div>
  </div>
</div>

<!-- ★ 수동 Watch 등록 폼 -->
<div class="add-form">
  <div class="form-title">➕ 수동 Watch 등록</div>
  <div class="form-row">
    <div class="form-group">
      <label>티커 (예: XRP, KRW-XRP)</label>
      <input class="form-input" id="addTicker" placeholder="XRP" maxlength="20"
             onkeydown="if(event.key==='Enter') addWatch()">
    </div>
    <div class="form-group">
      <label>진입가 (원)</label>
      <input class="form-input" id="addPrice" placeholder="2,039" type="text"
             onkeydown="if(event.key==='Enter') addWatch()">
    </div>
    <button class="btn-add" onclick="addWatch()">등록</button>
  </div>
  <div id="addMsg" class="form-msg" style="display:none"></div>
</div>

<!-- 활성 트레이드 -->
{% if state.active_trades %}
<div class="sec-title">⚡ 활성 트레이드 ({{ state.active_trades|length }}) — TP+{{ tp_pct }}% / SL-{{ sl_pct }}% / {{ timeout_h }}h</div>
<div class="tbl-wrap">
<table>
  <thead>
    <tr><th>티커</th><th>진입가</th><th>현재가</th><th>PnL</th>
        <th>TP</th><th>SL</th><th>경과</th><th>진행</th></tr>
  </thead>
  <tbody>
    {% for trade in state.active_trades %}
    {% set pnl=trade.pnl_pct or 0 %}
    {% set ep=trade.entry_price or 0 %}
    {% set cp=trade.current_price or 0 %}
    {% set held=trade.hours_held or 0 %}
    <tr>
      <td>
        <b>{{ trade.ticker.replace('KRW-','') }}</b>
        <span class="badge b-active">ACTIVE</span>
        <a href="https://upbit.com/exchange?code=CRIX.UPBIT.{{ trade.ticker }}"
           target="_blank" style="color:#555;font-size:.65rem;margin-left:2px">↗</a>
      </td>
      <td style="color:var(--muted);font-size:.78rem">{{ '{:,.0f}'.format(ep) if ep>=100 else '{:.4f}'.format(ep) }}</td>
      <td class="{% if pnl>0 %}price-up{% elif pnl<0 %}price-dn{% else %}price-flat{% endif %}">
        {{ '{:,.0f}'.format(cp) if cp>=100 else '{:.4f}'.format(cp) }}
      </td>
      <td>
        <span class="pnl-bar {% if pnl>0 %}pnl-pos{% elif pnl<0 %}pnl-neg{% else %}pnl-neu{% endif %}">
          {{ '%+.2f'|format(pnl) }}%
        </span>
      </td>
      <td style="color:var(--green);font-size:.76rem">
        {% set tp=trade.tp_price or 0 %}{{ '{:,.0f}'.format(tp) if tp>=100 else '{:.4f}'.format(tp) }}
      </td>
      <td style="color:var(--red);font-size:.76rem">
        {% set sl=trade.sl_price or 0 %}{{ '{:,.0f}'.format(sl) if sl>=100 else '{:.4f}'.format(sl) }}
      </td>
      <td style="color:var(--muted);font-size:.74rem">{{ '%.1f'|format(held) }}h</td>
      <td style="min-width:65px">
        {% set bar=[(pnl+sl_pct)/(tp_pct+sl_pct)*100,0]|max %}
        {% set bar=[bar,100]|min %}
        <div class="progress-wrap">
          <div class="progress-fill {% if pnl>=0 %}fill-green{% else %}fill-red{% endif %}"
               style="width:{{ bar|round }}%"></div>
        </div>
        <div style="font-size:.6rem;color:#444;text-align:center;margin-top:1px">SL←→TP</div>
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% endif %}

<!-- 이번 스캔 청산 -->
{% if state.closed_trades %}
<div class="sec-title">📌 청산 완료 (이번)</div>
<div class="tbl-wrap">
<table>
  <thead><tr><th>티커</th><th>결과</th><th>진입가</th><th>청산가</th><th>PnL</th><th>보유</th></tr></thead>
  <tbody>
    {% for t in state.closed_trades %}
    <tr>
      <td><b>{{ t.ticker.replace('KRW-','') }}</b></td>
      <td>{% if t.result=='TP' %}<span class="badge b-tp">🎯 TP</span>
          {% elif t.result=='SL' %}<span class="badge b-sl">🛑 SL</span>
          {% else %}<span class="badge b-timeout">⏰ TIMEOUT</span>{% endif %}</td>
      <td style="color:var(--muted);font-size:.76rem">
        {% set ep=t.entry_price or 0 %}{{ '{:,.0f}'.format(ep) if ep>=100 else '{:.4f}'.format(ep) }}</td>
      <td style="font-size:.76rem">
        {% set xp=t.exit_price or 0 %}{{ '{:,.0f}'.format(xp) if xp>=100 else '{:.4f}'.format(xp) }}</td>
      <td><span class="pnl-bar {% if (t.pnl_pct or 0)>0 %}pnl-pos{% elif (t.pnl_pct or 0)<0 %}pnl-neg{% else %}pnl-neu{% endif %}">
        {{ '%+.2f'|format(t.pnl_pct or 0) }}%</span></td>
      <td style="color:var(--muted);font-size:.74rem">{{ '%.1f'|format(t.hours_held or 0) }}h</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% endif %}

<!-- Watch List -->
<div class="sec-title">📋 Watch List ({{ state.watch_list|length }})</div>
{% if state.watch_list %}
<div class="tbl-wrap">
<table>
  <thead>
    <tr><th>티커</th><th>구분</th><th>일봉K</th><th>등록가</th>
        <th>현재가</th><th>등락률</th><th>수익률</th><th>차트</th><th>삭제</th><th>등록일</th></tr>
  </thead>
  <tbody>
    {% for item in state.watch_list %}
    {% set cp=item.change_pct or 0 %}
    {% set ep=item.entry_price %}
    {% set cur=item.current_price %}
    {% if ep and cur and ep>0 %}{% set pnl=(cur-ep)/ep*100 %}{% else %}{% set pnl=none %}{% endif %}
    <tr>
      <td>
        <b>{{ item.ticker.replace('KRW-','') }}</b>
        <a href="https://upbit.com/exchange?code=CRIX.UPBIT.{{ item.ticker }}"
           target="_blank" style="color:#555;font-size:.63rem;margin-left:2px">↗</a>
      </td>
      <td>
        {% if item.manual %}
          <span class="badge b-manual">수동</span>
        {% else %}
          <span class="badge b-auto">자동</span>
        {% endif %}
      </td>
      <td><b>{{ '%.1f'|format(item.daily_short_k or 0) }}</b></td>
      <td style="color:var(--muted);font-size:.76rem">
        {% if ep %}{{ '{:,.0f}'.format(ep) if ep>=100 else '{:.4f}'.format(ep) }}{% else %}—{% endif %}
      </td>
      <td class="{% if cp>0 %}price-up{% elif cp<0 %}price-dn{% else %}price-flat{% endif %}">
        {% if cur %}{{ '{:,.0f}'.format(cur) if cur>=100 else '{:.4f}'.format(cur) }}{% else %}—{% endif %}
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
      <td><button class="b-chart badge" onclick="openChart('{{ item.ticker }}')">📈 차트</button></td>
      <td>
        <button class="btn-del"
          onclick="removeWatch('{{ item.ticker }}', this)">✕</button>
      </td>
      <td style="color:var(--dim);font-size:.68rem">{{ item.registered_at[:10] }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% else %}
<p style="color:var(--dim);padding:14px 0">Watch List가 비어있습니다.</p>
{% endif %}

<!-- 진입 신호 -->
{% if state.entry_signals %}
<div class="sec-title">🚀 진입 트리거 (이번 스캔)</div>
<div class="tbl-wrap">
<table>
  <thead><tr><th>티커</th><th>4h K</th><th>1h K</th><th>트리거</th><th>진입가</th><th>TP</th><th>SL</th></tr></thead>
  <tbody>
    {% for sig in state.entry_signals %}{% set tr=sig.trigger %}{% set ep=sig.entry_price %}
    <tr>
      <td><b>{{ sig.ticker.replace('KRW-','') }}</b><span class="badge b-signal">SIGNAL</span>
        <a href="https://upbit.com/exchange?code=CRIX.UPBIT.{{ sig.ticker }}"
           target="_blank" style="color:#555;font-size:.63rem;margin-left:2px">↗</a></td>
      <td>{{ '%.1f'|format(tr.h4_short_k or 0) }}</td>
      <td>{{ '%.1f'|format(tr.h1_short_k or 0) }}</td>
      <td style="font-size:.73rem">{{ tr.h1_trigger_type }}</td>
      <td style="color:var(--muted);font-size:.76rem">{% if ep %}{{ '{:,.0f}'.format(ep) if ep>=100 else '{:.4f}'.format(ep) }}{% else %}—{% endif %}</td>
      <td style="color:var(--green);font-size:.76rem">{% if ep %}{{ '{:,.0f}'.format(ep*(1+tp_pct/100)) if ep>=100 else '{:.4f}'.format(ep*(1+tp_pct/100)) }}{% else %}—{% endif %}</td>
      <td style="color:var(--red);font-size:.76rem">{% if ep %}{{ '{:,.0f}'.format(ep*(1-sl_pct/100)) if ep>=100 else '{:.4f}'.format(ep*(1-sl_pct/100)) }}{% else %}—{% endif %}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% endif %}

<!-- 신규 등록 -->
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
      <td style="color:var(--muted)">{% if e.entry_price %}{{ '{:,.0f}'.format(e.entry_price) if e.entry_price>=100 else '{:.4f}'.format(e.entry_price) }}{% else %}—{% endif %}</td>
      <td style="font-size:.73rem;color:var(--muted)">{{ e.reason }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% endif %}

{% if state.removed %}
<div class="sec-title">🗑️ Watch 제거 (이번 스캔)</div>
<div class="tbl-wrap">
<table>
  <thead><tr><th>티커</th><th>유형</th><th>사유</th></tr></thead>
  <tbody>
    {% for r in state.removed %}
    <tr>
      <td><b>{{ r.ticker.replace('KRW-','') }}</b><span class="badge b-removed">REMOVED</span></td>
      <td style="font-size:.75rem">{{ r.removal_type }}</td>
      <td style="font-size:.72rem;color:var(--muted)">{{ r.reason }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
{% endif %}

{% if state.error %}
<div class="sec-title" style="color:var(--red)">⚠️ 오류</div>
<div style="background:var(--surface);padding:11px;border-radius:6px;color:var(--red);font-size:.8rem">{{ state.error }}</div>
{% endif %}

<div style="display:flex;justify-content:space-between;align-items:center;margin-top:16px;flex-wrap:wrap;gap:8px">
  <div style="display:flex;gap:14px">
    <a href="/history" style="color:#64b5f6;font-size:.76rem;text-decoration:none">📜 신호 히스토리</a>
    <a href="/trades"  style="color:#ffcc02;font-size:.76rem;text-decoration:none">📊 트레이드 히스토리</a>
  </div>
  <span style="font-size:.68rem;color:var(--dim)">MTF Scanner v1.4 | 60s 자동갱신</span>
</div>
</div>

<!-- 차트 모달 -->
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
    <div class="chart-wrap"><canvas id="stochChart"></canvas></div>
    <div style="font-size:.67rem;color:var(--dim);margin-top:7px">
      🟡 K선 &nbsp;🔵 D선 &nbsp;|&nbsp; 녹색 점선 과매도(20) / 적색 점선 과매수(80)
    </div>
  </div>
</div>

<script>
const CHART_DATA = {{ chart_data_json }};
const NEXT_SCAN  = "{{ state.next_scan_at or '' }}";

// 카운트다운
function updateCountdown(){
  if(!NEXT_SCAN) return;
  const diff=Math.max(0,Math.floor((new Date(NEXT_SCAN)-Date.now())/1000));
  const m=Math.floor(diff/60),s=diff%60;
  document.getElementById('cd').textContent=`다음 스캔 ${m}:${s.toString().padStart(2,'0')}`;
  if(diff>0) setTimeout(updateCountdown,1000);
}
updateCountdown();

// 수동 스캔
function triggerScan(){
  const btn=document.getElementById('scanBtn');
  btn.disabled=true; btn.textContent='⏳ 스캔 중...';
  fetch('/api/scan',{method:'POST'})
    .then(r=>r.json())
    .then(()=>{btn.textContent='✅ 요청됨';setTimeout(()=>location.reload(),3000);})
    .catch(()=>{btn.disabled=false;btn.textContent='🔄 수동 스캔';});
}

// ★ 수동 Watch 등록
function addWatch(){
  const ticker = document.getElementById('addTicker').value.trim();
  const price  = document.getElementById('addPrice').value.trim().replace(/,/g,'');
  const msgEl  = document.getElementById('addMsg');

  if(!ticker || !price){
    showMsg('티커와 진입가를 입력하세요.', false); return;
  }

  fetch('/api/watch/add', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ticker, entry_price: parseFloat(price)})
  })
  .then(r=>r.json())
  .then(d=>{
    showMsg(d.msg, d.ok);
    if(d.ok){
      document.getElementById('addTicker').value='';
      document.getElementById('addPrice').value='';
      setTimeout(()=>location.reload(), 1500);
    }
  })
  .catch(()=>showMsg('오류가 발생했습니다.', false));
}

function showMsg(msg, ok){
  const el=document.getElementById('addMsg');
  el.textContent=msg;
  el.className='form-msg '+(ok?'ok':'err');
  el.style.display='block';
  if(ok) setTimeout(()=>el.style.display='none', 3000);
}

// ★ Watch 삭제
function removeWatch(ticker, btn){
  if(!confirm(`${ticker.replace('KRW-','')} 를 Watch List에서 제거할까요?`)) return;
  btn.disabled=true; btn.textContent='...';
  fetch('/api/watch/remove',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ticker})
  })
  .then(r=>r.json())
  .then(d=>{
    if(d.ok) location.reload();
    else{ btn.disabled=false; btn.textContent='✕'; alert(d.msg); }
  })
  .catch(()=>{btn.disabled=false; btn.textContent='✕';});
}

// 차트
let currentTicker=null,currentTab='daily',chartInstance=null;
function openChart(ticker){
  currentTicker=ticker; currentTab='daily';
  document.querySelectorAll('.chart-tab').forEach((t,i)=>t.classList.toggle('active',i===0));
  document.getElementById('chartTitle').textContent=ticker.replace('KRW-','')+' · Stoch RSI';
  document.getElementById('chartModal').classList.add('open');
  renderChart();
}
function closeChart(){
  document.getElementById('chartModal').classList.remove('open');
  if(chartInstance){chartInstance.destroy();chartInstance=null;}
}
function switchTab(tab,el){
  currentTab=tab;
  document.querySelectorAll('.chart-tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active'); renderChart();
}
function renderChart(){
  const d=CHART_DATA[currentTicker]; if(!d) return;
  const kArr=d[currentTab+'_k']||[],dArr=d[currentTab+'_d']||[];
  if(chartInstance) chartInstance.destroy();
  chartInstance=new Chart(document.getElementById('stochChart').getContext('2d'),{
    type:'line',
    data:{labels:kArr.map((_,i)=>i+1),datasets:[
      {label:'K',data:kArr,borderColor:'#ffcc02',backgroundColor:'transparent',borderWidth:1.5,pointRadius:0,tension:0.3},
      {label:'D',data:dArr,borderColor:'#64b5f6',backgroundColor:'transparent',borderWidth:1.5,pointRadius:0,tension:0.3},
    ]},
    options:{
      responsive:true,maintainAspectRatio:false,animation:{duration:180},
      scales:{x:{display:false},y:{min:0,max:100,ticks:{color:'#666',font:{size:10}},grid:{color:'#1e2130'}}},
      plugins:{legend:{labels:{color:'#aaa',font:{size:11},boxWidth:12}},tooltip:{mode:'index',intersect:false}}
    },
    plugins:[{id:'ref',beforeDraw(c){
      const{ctx,scales:{y}}=c;
      [{v:20,col:'rgba(76,175,80,.4)'},{v:80,col:'rgba(244,67,54,.4)'}].forEach(({v,col})=>{
        const yp=y.getPixelForValue(v);
        ctx.save();ctx.strokeStyle=col;ctx.setLineDash([4,4]);ctx.lineWidth=1;
        ctx.beginPath();ctx.moveTo(c.chartArea.left,yp);ctx.lineTo(c.chartArea.right,yp);
        ctx.stroke();ctx.restore();
      });
    }}]
  });
}
document.getElementById('chartModal').addEventListener('click',function(e){
  if(e.target===this) closeChart();
});
</script>
</body>
</html>'''

# ── 히스토리 페이지들은 이전과 동일 ──
HISTORY_HTML = r'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>신호 히스토리</title>
<style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:"Segoe UI",sans-serif;background:#0e1117;color:#e0e0e0;font-size:14px}
header{background:#1a1d27;padding:12px 16px;border-bottom:1px solid #2a2d3a;display:flex;align-items:center;gap:12px}
header a{color:#64b5f6;text-decoration:none;font-size:.85rem}header h1{font-size:1rem;font-weight:700}
.container{max-width:900px;margin:0 auto;padding:16px}.sec{font-size:.9rem;font-weight:700;margin:14px 0 9px;color:#aaa}
.tw{overflow-x:auto;border-radius:8px}table{width:100%;border-collapse:collapse;background:#1a1d27;min-width:460px}
th{background:#252836;padding:8px 11px;text-align:left;font-size:.7rem;color:#888}
td{padding:8px 11px;font-size:.78rem;border-top:1px solid #2a2d3a}
.badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:.65rem;font-weight:700}
.b-s{background:#1b3a1b;color:#66bb6a}.empty{color:#555;padding:20px 0;text-align:center}
</style></head><body>
<header><a href="/">← 대시보드</a><h1>📜 신호 히스토리</h1></header>
<div class="container">
  <div class="sec">진입 트리거 히스토리 ({{ history|length }}건)</div>
  {% if history %}<div class="tw"><table>
    <thead><tr><th>#</th><th>티커</th><th>시각</th><th>4h K</th><th>1h K</th><th>트리거</th><th>진입가</th></tr></thead>
    <tbody>{% for sig in history|reverse %}{% set t=sig.trigger %}
    <tr>
      <td style="color:#555">{{ loop.index }}</td>
      <td><b>{{ sig.ticker.replace('KRW-','') }}</b><span class="badge b-s">SIG</span></td>
      <td style="color:#888;font-size:.72rem">{{ sig.triggered_at[:16] }}</td>
      <td>{{ '%.1f'|format(t.h4_short_k or 0) }}</td>
      <td>{{ '%.1f'|format(t.h1_short_k or 0) }}</td>
      <td style="font-size:.72rem">{{ t.h1_trigger_type }}</td>
      <td style="color:#666;font-size:.75rem">{% if sig.entry_price %}{{ '{:,.0f}'.format(sig.entry_price) if sig.entry_price>=100 else '{:.4f}'.format(sig.entry_price) }}{% else %}—{% endif %}</td>
    </tr>{% endfor %}</tbody>
  </table></div>
  {% else %}<p class="empty">아직 신호가 없습니다.</p>{% endif %}
</div></body></html>'''

TRADES_HTML = r'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>트레이드 히스토리</title>
<style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:"Segoe UI",sans-serif;background:#0e1117;color:#e0e0e0;font-size:14px}
header{background:#1a1d27;padding:12px 16px;border-bottom:1px solid #2a2d3a;display:flex;align-items:center;gap:12px}
header a{color:#64b5f6;text-decoration:none;font-size:.85rem}header h1{font-size:1rem;font-weight:700}
.container{max-width:960px;margin:0 auto;padding:16px}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:16px}
.card{background:#1a1d27;border-radius:8px;padding:13px;border:1px solid #2a2d3a}
.card .lbl{font-size:.68rem;color:#888;margin-bottom:4px}.card .val{font-size:1.2rem;font-weight:700}
.c-g{color:#4caf50}.c-r{color:#f44336}.c-y{color:#ffcc02}.c-b{color:#90caf9}
.sec{font-size:.9rem;font-weight:700;margin:14px 0 9px;color:#aaa}
.tw{overflow-x:auto;border-radius:8px}table{width:100%;border-collapse:collapse;background:#1a1d27;min-width:500px}
th{background:#252836;padding:8px 11px;text-align:left;font-size:.7rem;color:#888}
td{padding:8px 11px;font-size:.78rem;border-top:1px solid #2a2d3a}
.badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:.65rem;font-weight:700}
.b-tp{background:#1b3a1b;color:#66bb6a}.b-sl{background:#3a1f1f;color:#ef9a9a}.b-to{background:#2a2010;color:#ffcc02}
.pp{color:#4caf50;font-weight:700}.pn{color:#f44336;font-weight:700}
.empty{color:#555;padding:20px 0;text-align:center}
.bar{background:#252836;border-radius:3px;height:6px;margin-top:5px;overflow:hidden;display:flex}
.bw{background:#4caf50}.bl{background:#f44336}.bt{background:#ffcc02}
</style></head><body>
<header><a href="/">← 대시보드</a><h1>📊 트레이드 히스토리</h1></header>
<div class="container">
{% if stats.total %}
<div class="sg">
  <div class="card" style="border-color:#2e7d32">
    <div class="lbl">승률</div><div class="val c-g">{{ '%.1f'|format(stats.win_rate or 0) }}%</div>
    {% set t=stats.total %}
    <div class="bar"><div class="bw" style="width:{{ (stats.win/t*100)|round }}%"></div>
    <div class="bl" style="width:{{ (stats.loss/t*100)|round }}%"></div>
    <div class="bt" style="width:{{ (stats.timeout/t*100)|round }}%"></div></div>
  </div>
  <div class="card"><div class="lbl">총 / 승 / 패 / 타임아웃</div>
    <div class="val c-b">{{ stats.total }}건</div>
    <div style="font-size:.72rem;margin-top:4px"><span class="c-g">{{ stats.win }}승</span> / <span class="c-r">{{ stats.loss }}패</span> / <span class="c-y">{{ stats.timeout }}타임아웃</span></div>
  </div>
  <div class="card"><div class="lbl">평균 PnL</div>
    <div class="val {% if (stats.avg_pnl or 0)>=0 %}c-g{% else %}c-r{% endif %}">{{ '%+.2f'|format(stats.avg_pnl or 0) }}%</div>
  </div>
  <div class="card"><div class="lbl">최고 / 최저</div>
    <div class="val" style="font-size:.9rem"><span class="c-g">{{ '%+.2f'|format(stats.best_pnl or 0) }}%</span> / <span class="c-r">{{ '%+.2f'|format(stats.worst_pnl or 0) }}%</span></div>
  </div>
</div>
{% endif %}
<div class="sec">청산 기록 ({{ history|length }}건)</div>
{% if history %}<div class="tw"><table>
  <thead><tr><th>#</th><th>티커</th><th>결과</th><th>진입가</th><th>청산가</th><th>PnL</th><th>보유</th><th>청산일시</th></tr></thead>
  <tbody>{% for t in history|reverse %}
  <tr>
    <td style="color:#555">{{ loop.index }}</td>
    <td><b>{{ t.ticker.replace('KRW-','') }}</b></td>
    <td>{% if t.result=='TP' %}<span class="badge b-tp">🎯 TP</span>
        {% elif t.result=='SL' %}<span class="badge b-sl">🛑 SL</span>
        {% else %}<span class="badge b-to">⏰ TIMEOUT</span>{% endif %}</td>
    <td style="color:#666;font-size:.75rem">{% set ep=t.entry_price or 0 %}{{ '{:,.0f}'.format(ep) if ep>=100 else '{:.4f}'.format(ep) }}</td>
    <td style="font-size:.75rem">{% set xp=t.exit_price or 0 %}{{ '{:,.0f}'.format(xp) if xp>=100 else '{:.4f}'.format(xp) }}</td>
    <td class="{% if (t.pnl_pct or 0)>=0 %}pp{% else %}pn{% endif %}">{{ '%+.2f'|format(t.pnl_pct or 0) }}%</td>
    <td style="color:#888;font-size:.75rem">{{ '%.1f'|format(t.hours_held or 0) }}h</td>
    <td style="color:#555;font-size:.72rem">{{ t.closed_at[:16] if t.closed_at else '—' }}</td>
  </tr>{% endfor %}</tbody>
</table></div>
{% else %}<p class="empty">아직 청산된 거래가 없습니다.</p>{% endif %}
</div></body></html>'''

# ===================== Flask 라우트 =====================
@app.route('/')
def index():
    with scanner._state_lock:
        state = dict(scanner.scanner_state)
    chart_data_json = _json.dumps(state.get('chart_data', {}))
    return render_template_string(
        HTML, state=state,
        chart_data_json=chart_data_json,
        tp_pct=scanner.TRADE_TP_PCT,
        sl_pct=scanner.TRADE_SL_PCT,
        timeout_h=scanner.TRADE_TIMEOUT_H,
    )

@app.route('/history')
def history():
    return render_template_string(HISTORY_HTML, history=scanner.load_signal_history())

@app.route('/trades')
def trades():
    return render_template_string(
        TRADES_HTML,
        history=scanner.load_trade_history(),
        stats=scanner.calc_trade_stats()
    )

@app.route('/api/scan', methods=['POST'])
def api_scan():
    with scanner._state_lock:
        status = scanner.scanner_state['status']
    if status == 'scanning':
        return jsonify({'ok': False, 'msg': '이미 스캔 중입니다.'})
    scanner.manual_scan()
    return jsonify({'ok': True, 'msg': '수동 스캔 요청됨'})

@app.route('/api/watch/add', methods=['POST'])
def api_watch_add():
    data        = request.get_json() or {}
    ticker      = data.get('ticker', '').strip()
    entry_price = data.get('entry_price')
    if not ticker:
        return jsonify({'ok': False, 'msg': '티커를 입력하세요.'})
    if not entry_price:
        return jsonify({'ok': False, 'msg': '진입가를 입력하세요.'})
    try:
        entry_price = float(str(entry_price).replace(',', ''))
        if entry_price <= 0:
            raise ValueError
    except ValueError:
        return jsonify({'ok': False, 'msg': '진입가가 올바르지 않습니다.'})
    ok, msg = scanner.add_manual_watch(ticker, entry_price)
    return jsonify({'ok': ok, 'msg': msg})

@app.route('/api/watch/remove', methods=['POST'])
def api_watch_remove():
    data   = request.get_json() or {}
    ticker = data.get('ticker', '').strip()
    if not ticker:
        return jsonify({'ok': False, 'msg': '티커를 입력하세요.'})
    ok, msg = scanner.remove_watch(ticker)
    return jsonify({'ok': ok, 'msg': msg})

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
