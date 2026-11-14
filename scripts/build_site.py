"""Course site builder: site_src/lessons/*.html (body fragments) → site/*.html (shared layout + navigation).

Special tags inside body fragments
  <!--CODE:path/to/file.py-->            : insert the whole real source file from the repo as a code block
  <!--CODE:path/to/file.py:start-end-->  : insert only that line range
  <!--RESULTS-->                         : insert a results table read from outputs/*/result.json
  <!--TRAIN_METRICS-->                   : insert metrics of the best epoch from runs/hanwoo/train/results.csv
  <!--TRACKER_COMPARE-->                 : insert the tracker comparison table (outputs/tracker_comparison.json)
  <!--VIDEOS-->                          : insert the tracked result videos
usage: python scripts/build_site.py
"""
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "site_src"
OUT = ROOT / "site"

LESSONS = [
    ("00_intro", "Course Introduction and Overview"),
    ("01_env", "Setting Up the Environment"),
    ("02_data", "Understanding the Data (Barn Videos)"),
    ("03_detect", "Object Detection: Trying Pretrained YOLO"),
    ("04_dataset", "Building Training Data Automatically (Pseudo-labeling)"),
    ("05_train", "Fine-tuning and Evaluating YOLO"),
    ("06_tracking", "Multi-Object Tracking (BoT-SORT)"),
    ("07_stitching", "ID Consistency: Offline Tracklet Stitching"),
    ("07b_reid", "ReID Training and Fixed-Herd ID Keeping"),
    ("08_activity", "Activity Measurement Algorithm"),
    ("09_visualize", "Result Video and Dashboard"),
    ("10_results", "Running All Videos and Analysing Results"),
    ("11_apply", "Applying to New Videos and Extending"),
    ("12_live", "Real-Time Tracking App (RTSP Cameras)"),
    ("13_paper", "Paper: Fixed-Herd Constraint and Self-Supervised ReID"),
]

LAYOUT = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Hanwoo Activity AI Course</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<link rel="stylesheet" href="assets/style.css"></head><body>
<button class="menu" onclick="document.body.classList.toggle('open')" aria-label="Menu">☰</button>
<nav class="side"><a class="brand" href="index.html">🐂 Hanwoo Activity AI</a>
<div class="sub">YOLO · Tracking · Activity · Dashboard</div><ol>{nav}</ol>
<div class="navfoot"><a href="../outputs/index.html">▶ Result dashboards</a></div></nav>
<main class="content"><div class="crumb">Lesson {num} / {total}</div><h1>{title}</h1>
{body}
<div class="pager">{prev}{next}</div></main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>hljs.highlightAll();document.querySelectorAll('pre').forEach(p=>{{const b=document.createElement('button');b.className='copy';b.textContent='Copy';
b.onclick=()=>{{navigator.clipboard.writeText(p.innerText.replace(/Copy$/,''));b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1200)}};p.appendChild(b)}});</script>
</body></html>"""


def code_block(spec: str) -> str:
    m = re.match(r"(.+?)(?::(\d+)-(\d+))?$", spec.strip())
    path, a, b = m.group(1), m.group(2), m.group(3)
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    if a:
        lines = lines[int(a) - 1:int(b)]
    lang = {"py": "python", "yaml": "yaml", "html": "html", "txt": "plaintext"}.get(path.rsplit(".", 1)[-1], "plaintext")
    rng = f" (L{a}–{b})" if a else ""
    return (f'<div class="file">📄 {path}{rng}</div><pre><code class="language-{lang}">'
            f'{html.escape(chr(10).join(lines))}</code></pre>')


def results_table() -> str:
    rows = []
    for p in sorted((ROOT / "outputs").glob("*/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))["result"]
        n = r["name"]
        u = r.get("unassigned_det_ratio")
        unas = "-" if u is None else f"{100 * u:.1f}%"
        rows.append(f"<tr><td><a href='../outputs/{n}/dashboard.html'>{n}</a></td><td>{r['duration_s']:.0f}</td>"
                    f"<td>{r['raw_ids']}</td><td><b>{r['n_ids']}</b></td><td>{r.get('herd_size') or '-'}</td>"
                    f"<td>{r['n_links']}</td><td>{r.get('n_splits', 0)}</td>"
                    f"<td>{unas}</td>"
                    f"<td>{r['total_distance']:.1f}</td><td>{r['mean_activity']:.1f}</td><td>{r['proc_fps']:.0f}</td></tr>")
    if not rows:
        return "<p class='note'>No results yet. Run <code>python run_activity.py E:/Video/cow</code> and rebuild.</p>"
    return ("<div class='tbl'><table><tr><th>Video</th><th>Length (s)</th><th>Tracker IDs</th><th>Final IDs</th>"
            "<th>Estimated herd N</th><th>Merges</th><th>Swap splits</th><th>Dropped detections</th><th>Total distance (m)</th><th>Mean activity index</th><th>Processing speed (fps)</th></tr>"
            + "".join(rows) + "</table></div>")


def tracker_compare() -> str:
    p = ROOT / "outputs/tracker_comparison.json"
    if not p.exists():
        return "<p class='note'>No comparison results</p>"
    rows = json.loads(p.read_text(encoding="utf-8"))
    body = "".join(
        f"<tr><td>{r['video']}</td><td>{r['variant']}</td><td>{r['herd']}</td><td>{r['ids']}</td>"
        f"<td><b>{r['ratio']}</b></td><td>{r['mean_len_s']}</td><td>{r['full']}</td></tr>" for r in rows)
    return ("<div class='tbl'><table><tr><th>Video</th><th>Method</th><th>Estimated herd</th><th>IDs</th>"
            "<th>IDs/herd</th><th>Mean duration per ID (s)</th><th>Full-length IDs</th></tr>" + body + "</table></div>")


def videos() -> str:
    items = []
    for p in sorted((ROOT / "outputs").glob("*/*_tracked.mp4")):
        n = p.parent.name
        items.append(f"<figure><video src='../outputs/{n}/{p.name}' controls muted preload='none' "
                     f"poster='../outputs/{n}/{n}_bg.jpg'></video><figcaption>{n} · "
                     f"<a href='../outputs/{n}/dashboard.html'>Dashboard</a></figcaption></figure>")
    return "<div class='two'>" + "".join(items) + "</div>"


def train_metrics() -> str:
    p = ROOT / "runs/hanwoo/train/results.csv"
    if not p.exists():
        return "<p class='note'>No training results</p>"
    import csv
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    return (f"<div class='kpis'><div><span>Epochs trained</span><b>{len(rows)}</b></div>"
            f"<div><span>best epoch</span><b>{int(float(best['epoch']))}</b></div>"
            f"<div><span>Precision</span><b>{float(best['metrics/precision(B)']):.3f}</b></div>"
            f"<div><span>Recall</span><b>{float(best['metrics/recall(B)']):.3f}</b></div>"
            f"<div><span>mAP50</span><b>{float(best['metrics/mAP50(B)']):.3f}</b></div>"
            f"<div><span>mAP50-95</span><b>{float(best['metrics/mAP50-95(B)']):.3f}</b></div></div>")


def main():
    OUT.mkdir(exist_ok=True)
    shutil.copytree(SRC / "assets", OUT / "assets", dirs_exist_ok=True)
    total = len(LESSONS)
    for i, (slug, title) in enumerate(LESSONS):
        body = (SRC / "lessons" / f"{slug}.html").read_text(encoding="utf-8")
        body = re.sub(r"<!--CODE:(.+?)-->", lambda m: code_block(m.group(1)), body)
        body = body.replace("<!--RESULTS-->", results_table()).replace("<!--TRAIN_METRICS-->", train_metrics())
        body = body.replace("<!--TRACKER_COMPARE-->", tracker_compare()).replace("<!--VIDEOS-->", videos())
        fname = "index.html" if i == 0 else f"{slug}.html"
        nav = "".join(
            f"<li class='{'on' if j == i else ''}'><a href='{'index.html' if j == 0 else s + '.html'}'>"
            f"<span>{j:02d}</span>{t}</a></li>" for j, (s, t) in enumerate(LESSONS))
        prev = (f"<a href='{'index.html' if i == 1 else LESSONS[i-1][0] + '.html'}'>← {LESSONS[i-1][1]}</a>"
                if i > 0 else "<span></span>")
        nxt = f"<a class='nx' href='{LESSONS[i+1][0]}.html'>{LESSONS[i+1][1]} →</a>" if i < total - 1 else ""
        page = LAYOUT.format(title=title, nav=nav, num=f"{i:02d}", total=f"{total-1:02d}", body=body, prev=prev, next=nxt)
        (OUT / fname).write_text(page, encoding="utf-8")
    print("built", total, "pages →", OUT)


if __name__ == "__main__":
    main()
