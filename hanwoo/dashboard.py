"""Stage 5: activity dashboards (HTML) with Plotly.js - one per video plus one overall summary."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .render import id_color

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

CSS = """
:root{--bg:#f6f5f1;--card:#fff;--ink:#1f2328;--mut:#5c6370;--line:#e3e0d8;--acc:#b5562b}
@media (prefers-color-scheme: dark){:root{--bg:#16171a;--card:#1f2125;--ink:#e8e6e1;--mut:#9aa0a8;--line:#30333a;--acc:#e08a5a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
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
    # trajectories: downsampled to 0.5 s steps
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
        ("Animals identified", f"{len(s)}"),
        ("Video length", f"{meta['n_frames']/meta['fps']:.0f} s"),
        ("Total distance (herd)", f"{s.distance_m.sum():.1f} m"),
        ("Mean activity index", f"{s.activity_index.mean():.1f}" if len(s) else "-"),
        ("Most active animal", f"ID {int(most.id)} ({most.activity_index})" if most is not None else "-"),
    ]
    kpi_html = "".join(f"<div class='kpi'><span>{k}</span><b>{v}</b></div>" for k, v in kpis)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{name} Activity Dashboard</title>
<style>{CSS}</style><script src="{PLOTLY}"></script></head><body>
<header><h1>🐂 {name} · Hanwoo Per-Animal Activity Dashboard</h1>
<p>Detector: {info.get('model','')} · Tracking: BoT-SORT + {('ReID(' + info['reid'] + ') fixed-herd assignment N=' + str((info.get('identity') or {}).get('herd_size'))) if info.get('reid') else 'offline tracklet linking'} · perspective correction with {info.get('cow_height_m')} m cow height · <a href="../index.html">Overall summary</a></p></header>
<main>
<div class="kpis">{kpi_html}</div>
<div class="grid">
<div class="card full"><h2>Video with assigned IDs</h2><video src="{name}_tracked.mp4" controls muted preload="metadata"></video></div>
<div class="card"><h2>Distance per animal (m)</h2><div id="dist" style="height:320px"></div></div>
<div class="card"><h2>Behaviour state share per animal (%)</h2><div id="state" style="height:320px"></div></div>
<div class="card full"><h2>Activity index over time (5 s mean)</h2><div id="ts" style="height:360px"></div></div>
<div class="card full"><h2>Trajectories (image coordinates)</h2><div id="traj" style="height:460px"></div></div>
<div class="card full"><h2>Per-animal summary</h2><div class="tblwrap"><table>
<tr><th>ID</th><th>Tracked (s)</th><th>Observed ratio</th><th>Distance (m)</th><th>Mean speed (m/s)</th><th>Rest %</th><th>Active %</th><th>Walk %</th><th>Activity index</th></tr>
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
 {{type:'bar',name:'Rest',x:ids,y:D.summary.map(r=>r.rest_pct),marker:{{color:'#9aa0a8'}}}},
 {{type:'bar',name:'Active (local)',x:ids,y:D.summary.map(r=>r.active_pct),marker:{{color:'#e8a33d'}}}},
 {{type:'bar',name:'Walk',x:ids,y:D.summary.map(r=>r.walk_pct),marker:{{color:'#3fa45b'}}}}],
 L({{barmode:'stack',legend:{{orientation:'h',y:1.1}}}}),cfg);
Plotly.newPlot('ts',Object.entries(D.ts).map(([k,v])=>({{type:'scatter',mode:'lines',name:'ID '+k,x:v.t,y:v.a,line:{{color:D.colors[k],width:1.6}}}})),
 L({{xaxis:{{title:'Time (s)',gridcolor:grid}},yaxis:{{title:'Activity index',gridcolor:grid}}}}),cfg);
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
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hanwoo Activity Summary</title><style>{CSS}</style><script src="{PLOTLY}"></script></head><body>
<header><h1>🐂 Hanwoo Activity Analysis · All Videos</h1><p>Click a name in the table for that video's detailed dashboard.</p></header>
<main><div class="grid">
<div class="card full"><div class="tblwrap"><table><tr><th>Video</th><th>Length (s)</th><th>Final IDs</th><th>IDs before linking</th><th>Total distance (m)</th><th>Mean activity index</th><th>Processing speed (fps)</th></tr>{rows}</table></div></div>
<div class="card"><h2>Mean activity index per video</h2><div id="a" style="height:320px"></div></div>
<div class="card"><h2>Effect of ID linking (before → after)</h2><div id="b" style="height:320px"></div></div>
</div></main><script>
const R={data};const dark=matchMedia('(prefers-color-scheme: dark)').matches;const ink=dark?'#e8e6e1':'#1f2328',g=dark?'#30333a':'#e3e0d8';
const L=o=>Object.assign({{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{{color:ink}},margin:{{l:45,r:10,t:10,b:40}},xaxis:{{gridcolor:g}},yaxis:{{gridcolor:g}}}},o||{{}});
Plotly.newPlot('a',[{{type:'bar',x:R.map(r=>r.name),y:R.map(r=>r.mean_activity),marker:{{color:'#b5562b'}}}}],L(),{{displaylogo:false,responsive:true}});
Plotly.newPlot('b',[{{type:'bar',name:'Before linking',x:R.map(r=>r.name),y:R.map(r=>r.raw_ids),marker:{{color:'#9aa0a8'}}}},{{type:'bar',name:'After linking',x:R.map(r=>r.name),y:R.map(r=>r.n_ids),marker:{{color:'#3fa45b'}}}}],L({{barmode:'group',legend:{{orientation:'h',y:1.12}}}}),{{displaylogo:false,responsive:true}});
</script></body></html>"""
    (Path(root) / "index.html").write_text(html, encoding="utf-8")
