# -*- coding: utf-8 -*-
"""
dashboard.py — Upbit MTF 스캐너 Flask 대시보드 (v1.1)
변경사항:
  - BTC 주봉 MA20 카드 추가 (실제 필터 기준)
  - BTC 일봉 MA20 카드 추가 (참고용)
  - 스캔 대상 종목 수 표시
"""

import os
import threading
from flask import Flask, jsonify, render_template_string

import scanner
import mtf_setup

app = Flask(__name__)

# ── 백그라운드 스캔 스레드 시작 ──────────────────
_scan_thread = threading.Thread(target=scanner.scanner_loop, daemon=True)
_scan_thread.start()

HTML = '''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Upbit MTF Scanner</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "Segoe UI", sans-serif; background: #0e1117; color: #e0e0e0; }
  header { background: #1a1d27; padding: 16px 24px; border-bottom: 1px solid #2a2d3a; }
  header h1 { font-size: 1.3rem; font-weight: 700; }
  header span { font-size: 0.8rem; color: #888; margin-left: 12px; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: #1a1d27; border-radius: 8px; padding: 16px; border: 1px solid #2a2d3a; }
  .card .label { font-size: 0.72rem; color: #888; margin-bottom: 6px; }
  .card .value { font-size: 1.3rem; font-weight: 700; }
  .card .sub   { font-size: 0.7rem; color: #555; margin-top: 4px; }
  .safe   { color: #4caf50; }
  .danger { color: #f44336; }
  .warn   { color: #ff9800; }
  .neutral{ color: #90caf9; }
  .section-title { font-size: 1rem; font-weight: 700; margin: 20px 0 10px; color: #aaa; }
  table { width: 100%; border-collapse: collapse; background: #1a1d27; border-radius: 8px; overflow: hidden; }
  th { background: #252836; padding: 10px 14px; text-align: left; font-size: 0.78rem; color: #888; }
  td { padding: 10px 14px; font-size: 0.85rem; border-top: 1px solid #2a2d3a; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; }
  .badge-new     { background: #1e3a5f; color: #64b5f6; }
  .badge-signal  { background: #1b3a1b; color: #66bb6a; }
  .badge-removed { background: #3a1f1f; color: #ef9a9a; }
  .status-idle     { color: #888; }
  .status-scanning { color: #ff9800; }
  .status-done     { color: #4caf50; }
  .status-error    { color: #f44336; }
  .refresh-note { font-size: 0.75rem; color: #555; text-align: right; margin-top: 16px; }
  .macro-section { background: #131720; border: 1px solid #2a2d3a; border-radius: 8px;
                   padding: 12px 16px; margin-bottom: 20px; }
  .macro-section .title { font-size: 0.8rem; color: #666; margin-bottom: 8px; }
</style>
</head>
<body>
<header>
  <h1>📡 Upbit MTF Scanner</h1>
  <span>v1.1 | 60초마다 자동 새로고침</span>
</header>
<div class="container">

  <!-- 상태 카드 -->
  <div class="grid">
    <div class="card">
      <div class="label">스캐너 상태</div>
      <div class="value status-{{ state.status }}">{{ state.status.upper() }}</div>
    </div>
    <div class="card">
      <div class="label">마지막 스캔</div>
      <div class="value" style="font-size:0.9rem">{{ (state.last_scan_at or "—")[:16] }}</div>
    </div>
    <div class="card">
      <div class="label">누적 스캔</div>
      <div class="value">{{ state.scan_count }}</div>
    </div>
    <div class="card">
      <div class="label">스캔 종목 수</div>
      <div class="value neutral">{{ state.total_scanned }}</div>
    </div>
    <div class="card">
      <div class="label">Watch List</div>
      <div class="value safe">{{ state.watch_list | length }}</div>
    </div>
    <div class="card">
      <div class="label">진입 신호 (이번)</div>
      <div class="value {% if state.entry_signals %}safe{% else %}neutral{% endif %}">
        {{ state.entry_signals | length }}
      </div>
    </div>
  </div>

  <!-- BTC 매크로 카드 -->
  {% set macro = state.macro or {} %}
  <div class="macro-section">
    <div class="title">📊 BTC 매크로 필터</div>
    <div class="grid" style="margin-bottom:0">

      <!-- 주봉 MA20 (실제 필터) -->
      <div class="card" style="border-color: {% if macro.safe %}#2e7d32{% else %}#b71c1c{% endif %}">
        <div class="label">BTC 주봉 MA20 <span style="color:#ff9800">★ 필터 기준</span></div>
        {% set w_dist = macro.weekly_distance_pct or 0 %}
        <div class="value {% if macro.safe %}safe{% else %}danger{% endif %}">
          {{ '%+.2f'|format(w_dist) }}%
        </div>
        <div class="sub">
          {% if macro.safe %}▲ MA20 위 — 스캔 허용{% else %}▼ MA20 아래 — 신규 등록 차단{% endif %}
        </div>
      </div>

      <!-- 일봉 MA20 (참고) -->
      <div class="card">
        <div class="label">BTC 일봉 MA20 <span style="color:#555">(참고)</span></div>
        {% set d_dist = macro.daily_distance_pct or 0 %}
        <div class="value {% if d_dist > 0 %}safe{% elif d_dist < -3 %}danger{% else %}warn{% endif %}">
          {{ '%+.2f'|format(d_dist) }}%
        </div>
        <div class="sub">
          {% if d_dist > 0 %}▲ 일봉 MA20 위{% else %}▼ 일봉 MA20 아래{% endif %}
          (필터 미적용)
        </div>
      </div>

      <!-- 매크로 상태 -->
      <div class="card">
        <div class="label">매크로 상태</div>
        <div class="value {% if macro.safe %}safe{% else %}danger{% endif %}" style="font-size:1rem">
          {{ macro.state or "—" }}
        </div>
        <div class="sub" style="font-size:0.68rem">{{ macro.reason or "" }}</div>
      </div>

    </div>
  </div>

  <!-- Watch List -->
  <div class="section-title">📋 Watch List ({{ state.watch_list | length }})</div>
  {% if state.watch_list %}
  <table>
    <thead>
      <tr><th>티커</th><th>일봉 단기 K</th><th>등록 사유</th><th>등록 시각</th></tr>
    </thead>
    <tbody>
      {% for item in state.watch_list %}
      <tr>
        <td><b>{{ item.ticker }}</b></td>
        <td>{{ "%.1f"|format(item.daily_short_k or 0) }}</td>
        <td style="font-size:0.78rem; color:#888">{{ item.reason }}</td>
        <td style="font-size:0.78rem; color:#555">{{ item.registered_at[:16] }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p style="color:#555; padding: 16px 0">Watch List가 비어있습니다.</p>
  {% endif %}

  <!-- 진입 신호 -->
  {% if state.entry_signals %}
  <div class="section-title">🚀 진입 트리거 (이번 스캔)</div>
  <table>
    <thead>
      <tr><th>티커</th><th>4h K</th><th>1h K</th><th>트리거</th><th>사유</th></tr>
    </thead>
    <tbody>
      {% for sig in state.entry_signals %}
      {% set t = sig.trigger %}
      <tr>
        <td><b>{{ sig.ticker }}</b> <span class="badge badge-signal">SIGNAL</span></td>
        <td>{{ "%.1f"|format(t.h4_short_k or 0) }}</td>
        <td>{{ "%.1f"|format(t.h1_short_k or 0) }}</td>
        <td>{{ t.h1_trigger_type }}</td>
        <td style="font-size:0.78rem; color:#888">{{ t.reason }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  <!-- 신규 Watch 등록 -->
  {% if state.new_entries %}
  <div class="section-title">✨ 신규 Watch 등록 (이번 스캔)</div>
  <table>
    <thead><tr><th>티커</th><th>일봉 K</th><th>사유</th></tr></thead>
    <tbody>
      {% for e in state.new_entries %}
      <tr>
        <td><b>{{ e.ticker }}</b> <span class="badge badge-new">NEW</span></td>
        <td>{{ "%.1f"|format(e.daily_short_k or 0) }}</td>
        <td style="font-size:0.78rem; color:#888">{{ e.reason }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  <!-- 제거 목록 -->
  {% if state.removed %}
  <div class="section-title">🗑️ Watch 제거 (이번 스캔)</div>
  <table>
    <thead><tr><th>티커</th><th>제거 유형</th><th>사유</th></tr></thead>
    <tbody>
      {% for r in state.removed %}
      <tr>
        <td><b>{{ r.ticker }}</b> <span class="badge badge-removed">REMOVED</span></td>
        <td>{{ r.removal_type }}</td>
        <td style="font-size:0.78rem; color:#888">{{ r.reason }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% endif %}

  <!-- 오류 -->
  {% if state.error %}
  <div class="section-title" style="color:#f44336">⚠️ 오류</div>
  <div style="background:#1a1d27; padding:12px; border-radius:6px;
              color:#f44336; font-size:0.85rem">{{ state.error }}</div>
  {% endif %}

  <p class="refresh-note">60초마다 자동 새로고침 | MTF Scanner v1.1</p>
</div>
</body>
</html>'''

@app.route('/')
def index():
    with scanner._state_lock:
        state = dict(scanner.scanner_state)
    return render_template_string(HTML, state=state)

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
