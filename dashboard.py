# -*- coding: utf-8 -*-
"""
dashboard.py — Upbit MTF 스캐너 대시보드
Version : v2.3.1
Changelog:
  v2.3.1 - dirIcon null/undefined 방어 처리
           dirCell null 방어 처리
           MED '??' 버그 수정
           스파크라인 추세 컬럼에 통합
           진입강도 등급 상한 반영 (B→최대👀관찰, A→최대🎯진입고려)
           esCell 툴팁에 raw_level 표시 (실제 신호강도 vs 등급 제한 표시)
  v2.3.0 - K값 색상, 4hK 과열, 방향 아이콘, 진입강도, 등급별 승률바
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
<title>Upbit MTF Scanner v2.3.1</title>
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

  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(155px, 1fr)); gap: 10px; margin-bottom: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  .card-label { color: var(--muted); font-size: 11px; margin-bottom: 4px; }
  .card-value { font-size: 22px; font-weight: 700; }
  .card-sub   { color: var(--muted); font-size: 11px; margin-top: 2px; }

  .grade-bar-wrap { display: flex; flex-direction: column; gap: 5px; margin-top: 6px; }
  .grade-bar-row  { display: flex; align-items: center; gap: 6px; font-size: 11px; }
  .grade-bar-bg   { flex: 1; background: var(--border); border-radius: 3px; height: 7px; }
  .grade-bar-fill { height: 7px; border-radius: 3px; transition: width .5s; }

  .section { margin-bottom: 20px; }
  .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
  .badge-red    { background: #3d1515; color: var(--red); }
  .badge-orange { background: #3d2a0f; color: var(--orange); }
  .badge-blue   { background: #1f3a5f; color: var(--blue); }
  .badge-gray   { background: #21262d; color: var(--muted); }
  .badge-green  { background: #1a3d2b; color: var(--green); }

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

  /* 방향 */
  .dir-up   { color: var(--green);  font-weight: 700; }
  .dir-rise { color: var(--lime);   }
  .dir-side { color: var(--muted);  }
  .dir-down { color: var(--red);    }
  .dir-gold { color: var(--yellow); font-weight: 700; }

  /* 진입강도 */
  .es-3 { color: var(--green);  font-weight: 700; }
  .es-2 { color: var(--lime);   font-weight: 600; }
  .es-1 { color: var(--yellow); }
  .es-0 { color: var(--muted);  }
  /* 등급 제한 표시 */
  .es-capped { opacity: 0.6; font-size: 10px; color: var(--muted); margin-left: 3px; }

  /* 수익률 */
  .pnl-pos { color: var(--green); font-weight: 600; }
  .pnl-neg { color: var(--red);   font-weight: 600; }
  .pnl-neu { color: var(--muted); }

  /* 점수 변화 */
  .sc-up   { color: var(--green); }
  .sc-down { color: var(--red);   }
  .sc-neu  { color: var(--muted); }

  /* 스파크라인 */
  .spark { display: inline-flex; align-items: flex-end; gap: 1px; height: 16px; vertical-align: middle; margin-left: 4px; }
  .spark-bar { width: 3px; background: var(--blue); border-radius: 1px; min-height: 2px; opacity: .75; }

  /* 버튼 */
  .btn       { padding: 4px 10px; border-radius: 5px; border: none; cursor: pointer; font-size: 11px; font-weight: 600; }
  .btn-scan  { background: var(--blue);   color: #000; }
  .btn-del   { background: #3d1515;       color: var(--red); }
  .btn-close { background: #1a3d2b;       color: var(--green); }
  .btn:hover { opacity: .85; }

  .input-row { display: flex; gap: 8px; margin-bottom: 10px; }
  input[type=text] { background: var(--card); border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 5px; font-size: 12px; width: 180px; }

  .dot         { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .dot-green   { background: var(--green);  }
  .dot-orange  { background: var(--orange); animation: pulse .8s infinite; }
  .dot-red     { background: var(--red);    }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  #msg { padding: 6px 12px; border-radius: 5px; margin-bottom: 10px; font-size: 12px; display: none; }
  .msg-ok  { background: #1a3d2b; color: var(--green); }
  .msg-err { background: #3d1515; color: var(--red);   }

  .timestamps { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; font-size: 11px; color: var(--muted); }

  .tabs { display: flex; gap: 4px; margin-bottom: 12px; flex-wrap: wrap; }
  .tab { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); color: var(--muted); cursor: pointer; font-size: 12px; transition: all .2s; }
  .tab.active { background: var(--blue); color: #000; border-color: var(--blue); font-weight: 700; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* 등급별 진입강도 범례 */
  .legend { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; font-size: 11px; color: var(--muted); background: var(--card); padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border); }
  .legend-item { display: flex; align-items: center; gap: 4px; }
</style>
</head>
<body>

<div class="header">
  <h1>📡 Upbit MTF Scanner</h1>
  <span class="version-badge">v2.3.1</span>
  <span id="statusDot" class="dot dot-green" style="margin-left:auto"></span>
  <span id="statusTxt" style="font-size:11px;color:var(--muted);margin-right:8px">idle</span>
  <button class="btn btn-scan" onclick="manualScan()">🔄 수동스캔</button>
</div>

<div class="container">
  <div id<span class="cursor">█</span>
