"""5단계: 활동량 대시보드(HTML) 생성 - Plotly.js 사용, 영상별 1개 + 전체 요약 1개."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .render import id_color

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

CSS = """
:root{--bg:#f6f5f1;--card:#fff;--ink:#1f2328;--mut:#5c6370;--line:#e3e0d8;--acc:#b5562b}
@media (prefers-color-scheme: dark){:root{--bg:#16171a;--card:#1f2125;--ink:#e8e6e1;--mut:#9aa0a8;--line:#30333a;--acc:#e08a5a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Pretendard","Noto Sans KR","Malgun Gothic",system-ui,sans-serif}
header{padding:20px 24px;border-bottom:1px solid var(--line)}h1{margin:0;font-size:22px}header p{margin:4px 0 0;color:var(--mut);font-size:14px}
main{max-width:1280px;margin:0 auto;padding:20px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}.kpi b{display:block;font-size:26px;margin-top:4px}.kpi span{color:var(--mut);font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}.card h2{font-size:15px;margin:2px 4px 8px}
.full{grid-column:1/-1}video{width:100%;border-radius:8px;background:#000}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.tblwrap{overflow-x:auto}a{color:var(--acc)}
"""


def _hex(tid):
    b, g, r = id_color(int(tid))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_video_dashboard(out_dir: Path, name: str, summary: pd.DataFrame, ts: pd.DataFrame,
                          act: pd.DataFrame, meta: dict, info: dict):
    out_dir = Path(out_dir)
    s = summary.sort_values("id")
    # 궤적: 0.5초 간격으로 다운샘플
    step = max(1, int(meta["fps"] / 2))
    traj = {}
    for tid, g in act[act.frame % step == 0].groupby("tid"):
        traj[int(tid)] = dict(x=g.cx.round(1).tolist(), y=g.cy.round(1).tolist())
    data = dict(
        name=name, meta=meta, info=info,
        summary=s.to_dict(orient="records"),
        colors={int(t): _hex(t) for t in s.id},
        ts={int(t): dict(t=g.bin.tolist(), a=g.activity.round(1).tolist())
            for t, g in ts.groupby("tid")},
        traj=traj,
    )
    rows = "".join(
        f"<tr><td><span class='sw' style='background:{_hex(r.id)}'></span>ID {r.id}</td>"
        f"<td>{r.tracked_s}</td><td>{r.observed_ratio*100:.0f}%</td><td>{r.distance_m}</td><td>{r.mean_speed}</td>"
        f"<td>{r.rest_pct}</td><td>{r.active_pct}</td><td>{r.walk_pct}</td><td><b>{r.activity_index}</b></td></tr>"
        for r in s.itertuples())
    most = s.sort_values("activity_index", ascending=False).iloc[0] if len(s) else None
    kpis = [
        ("식별된 개체 수", f"{len(s)} 두"),
        ("영상 길이", f"{meta['n_frames']/meta['fps']:.0f} 초"),
        ("총 이동거리(무리)", f"{s.distance_m.sum():.1f} m"),
        ("평균 활동지수", f"{s.activity_index.mean():.1f}" if len(s) else "-"),
        ("최고 활동 개체", f"ID {int(most.id)} ({most.activity_index})" if most is not None else "-"),
    ]
    kpi_html = "".join(f"<div class='kpi'><span>{k}</span><b>{v}</b></div>" for k, v in kpis)
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{name} 활동량 대시보드</title>
<style>{CSS}</style><script src="{PLOTLY}"></script></head><body>
<header><h1>🐂 {name} · 한우 개체별 활동량 대시보드</h1>
<p>검출 모델: {info.get('model','')} · 추적: BoT-SORT + {('ReID(' + info['reid'] + ') 고정 개체군 배정 N=' + str((info.get('identity') or {}).get('herd_size'))) if info.get('reid') else '오프라인 tracklet 정합'} · 원근보정 체고 {info.get('cow_height_m')} m 기준 · <a href="../index.html">전체 요약</a></p></header>
<main>
<div class="kpis">{kpi_html}</div>
<div class="grid">
<div class="card full"><h2>ID 부여 결과 영상</h2><video src="{name}_tracked.mp4" controls muted preload="metadata"></video></div>
<div class="card"><h2>개체별 이동거리 (m)</h2><div id="dist" style="height:320px"></div></div>
<div class="card"><h2>개체별 행동 상태 비율 (%)</h2><div id="state" style="height:320px"></div></div>
<div class="card full"><h2>시간대별 활동지수 (5초 평균)</h2><div id="ts" style="height:360px"></div></div>
<div class="card full"><h2>이동 궤적 (화면 좌표)</h2><div id="traj" style="height:460px"></div></div>
<div class="card full"><h2>개체별 요약표</h2><div class="tblwrap"><table>
<tr><th>ID</th><th>추적시간(s)</th><th>실관측비율</th><th>이동거리(m)</th><th>평균속도(m/s)</th><th>휴식%</th><th>국소활동%</th><th>보행%</th><th>활동지수</th></tr>
{rows}</table></div></div>
</div></main>
<script>
const D={json.dumps(data, ensure_ascii=False)};
const dark=matchMedia('(prefers-color-scheme: dark)').matches;
const ink=dark?'#e8e6e1':'#1f2328', grid=dark?'#30333a':'#e3e0d8';
const L=(o)=>Object.assign({{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:ink,size:12}},
  margin:{{l:50,r:10,t:10,b:40}},xaxis:{{gridcolor:grid}},yaxis:{{gridcolor:grid}}}},o||{{}});
const cfg={{displaylogo:false,responsive:true}};
const ids=D.summary.map(r=>'ID '+r.id), col=D.summary.map(r=>D.colors[r.id]);
Plotly.newPlot('dist',[{{type:'bar',x:ids,y:D.summary.map(r=>r.distance_m),marker:{{color:col}}}}],L(),cfg);
Plotly.newPlot('state',[
 {{type:'bar',name:'휴식',x:ids,y:D.summary.map(r=>r.rest_pct),marker:{{color:'#9aa0a8'}}}},
 {{type:'bar',name:'국소활동',x:ids,y:D.summary.map(r=>r.active_pct),marker:{{color:'#e8a33d'}}}},
 {{type:'bar',name:'보행',x:ids,y:D.summary.map(r=>r.walk_pct),marker:{{color:'#3fa45b'}}}}],
 L({{barmode:'stack',legend:{{orientation:'h',y:1.1}}}}),cfg);
Plotly.newPlot('ts',Object.entries(D.ts).map(([k,v])=>({{type:'scatter',mode:'lines',name:'ID '+k,x:v.t,y:v.a,line:{{color:D.colors[k],width:1.6}}}})),
 L({{xaxis:{{title:'시간(s)',gridcolor:grid}},yaxis:{{title:'활동지수',gridcolor:grid}}}}),cfg);
Plotly.newPlot('traj',Object.entries(D.traj).map(([k,v])=>({{type:'scatter',mode:'lines',name:'ID '+k,x:v.x,y:v.y,line:{{color:D.colors[k],width:1.5}}}})),
 L({{images:[{{source:'{name}_bg.jpg',xref:'x',yref:'y',x:0,y:0,sizex:D.meta.width,sizey:D.meta.height,sizing:'stretch',layer:'below',opacity:0.55}}],
   xaxis:{{range:[0,D.meta.width],showgrid:false}},yaxis:{{range:[D.meta.height,0],showgrid:false,scaleanchor:'x'}}}}),cfg);
</script></body></html>"""
    (out_dir / "dashboard.html").write_text(html, encoding="utf-8")


def write_index(root: Path, results: list[dict]):
    rows = "".join(
        f"<tr><td><a href='{r['name']}/dashboard.html'>{r['name']}</a></td><td>{r['duration_s']:.0f}</td>"
        f"<td>{r['n_ids']}</td><td>{r['raw_ids']}</td><td>{r['total_distance']:.1f}</td><td>{r['mean_activity']:.1f}</td>"
        f"<td>{r['proc_fps']:.0f}</td></tr>" for r in results)
    data = json.dumps(results, ensure_ascii=False)
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>한우 활동량 전체 요약</title><style>{CSS}</style><script src="{PLOTLY}"></script></head><body>
<header><h1>🐂 한우 활동량 분석 · 전체 영상 요약</h1><p>영상별 상세 대시보드는 표의 이름을 누르세요.</p></header>
<main><div class="grid">
<div class="card full"><div class="tblwrap"><table><tr><th>영상</th><th>길이(s)</th><th>최종 ID 수</th><th>정합 전 ID 수</th><th>총 이동거리(m)</th><th>평균 활동지수</th><th>처리속도(fps)</th></tr>{rows}</table></div></div>
<div class="card"><h2>영상별 평균 활동지수</h2><div id="a" style="height:320px"></div></div>
<div class="card"><h2>ID 정합 효과 (정합 전 → 후)</h2><div id="b" style="height:320px"></div></div>
</div></main><script>
const R={data};const dark=matchMedia('(prefers-color-scheme: dark)').matches;const ink=dark?'#e8e6e1':'#1f2328',g=dark?'#30333a':'#e3e0d8';
const L=o=>Object.assign({{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:ink}},margin:{{l:45,r:10,t:10,b:40}},xaxis:{{gridcolor:g}},yaxis:{{gridcolor:g}}}},o||{{}});
Plotly.newPlot('a',[{{type:'bar',x:R.map(r=>r.name),y:R.map(r=>r.mean_activity),marker:{{color:'#b5562b'}}}}],L(),{{displaylogo:false,responsive:true}});
Plotly.newPlot('b',[{{type:'bar',name:'정합 전',x:R.map(r=>r.name),y:R.map(r=>r.raw_ids),marker:{{color:'#9aa0a8'}}}},{{type:'bar',name:'정합 후',x:R.map(r=>r.name),y:R.map(r=>r.n_ids),marker:{{color:'#3fa45b'}}}}],L({{barmode:'group',legend:{{orientation:'h',y:1.12}}}}),{{displaylogo:false,responsive:true}});
</script></body></html>"""
    (Path(root) / "index.html").write_text(html, encoding="utf-8")
