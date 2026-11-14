"""강좌 사이트 빌더: site_src/lessons/*.html (본문 조각) → site/*.html (공통 레이아웃 + 내비게이션).

본문 조각 안의 특수 태그
  <!--CODE:path/to/file.py-->            : 저장소의 실제 소스 전체를 코드 블록으로 삽입
  <!--CODE:path/to/file.py:start-end-->  : 해당 줄 범위만 삽입
  <!--RESULTS-->                         : outputs/*/result.json 을 읽어 결과 표 삽입
  <!--TRAIN_METRICS-->                   : runs/hanwoo/train/results.csv 마지막 epoch 지표 삽입
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
    ("00_intro", "강좌 소개와 전체 구조"),
    ("01_env", "개발 환경 구축"),
    ("02_data", "데이터(축사 영상) 이해"),
    ("03_detect", "객체 검출: 사전학습 YOLO 시험"),
    ("04_dataset", "학습 데이터 자동 구축 (Pseudo-labeling)"),
    ("05_train", "YOLO 미세조정 학습과 평가"),
    ("06_tracking", "다중 객체 추적 (BoT-SORT)"),
    ("07_stitching", "ID 일관성: 오프라인 Tracklet 정합"),
    ("07b_reid", "ReID 학습과 고정 개체군 ID 유지"),
    ("08_activity", "활동량 측정 알고리즘"),
    ("09_visualize", "결과 영상과 대시보드"),
    ("10_results", "전체 영상 실행과 결과 분석"),
    ("11_apply", "새 영상 적용과 확장"),
]

LAYOUT = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · 한우 활동량 AI 강좌</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<link rel="stylesheet" href="assets/style.css"></head><body>
<button class="menu" onclick="document.body.classList.toggle('open')" aria-label="메뉴">☰</button>
<nav class="side"><a class="brand" href="index.html">🐂 한우 활동량 AI</a>
<div class="sub">YOLO · 추적 · 활동량 · 대시보드</div><ol>{nav}</ol>
<div class="navfoot"><a href="../outputs/index.html">▶ 결과 대시보드</a></div></nav>
<main class="content"><div class="crumb">Lesson {num} / {total}</div><h1>{title}</h1>
{body}
<div class="pager">{prev}{next}</div></main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script>hljs.highlightAll();document.querySelectorAll('pre').forEach(p=>{{const b=document.createElement('button');b.className='copy';b.textContent='복사';
b.onclick=()=>{{navigator.clipboard.writeText(p.innerText.replace(/복사$/,''));b.textContent='복사됨';setTimeout(()=>b.textContent='복사',1200)}};p.appendChild(b)}});</script>
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
        return "<p class='note'>아직 결과가 없습니다. <code>python run_activity.py E:/Video/cow</code> 실행 후 다시 빌드하세요.</p>"
    return ("<div class='tbl'><table><tr><th>영상</th><th>길이(s)</th><th>추적기 ID 수</th><th>최종 ID 수</th>"
            "<th>추정 두수 N</th><th>병합</th><th>스왑 분리</th><th>제외 검출</th><th>총 이동거리(m)</th><th>평균 활동지수</th><th>처리속도(fps)</th></tr>"
            + "".join(rows) + "</table></div>")


def tracker_compare() -> str:
    p = ROOT / "outputs/tracker_comparison.json"
    if not p.exists():
        return "<p class='note'>비교 실험 결과 없음</p>"
    rows = json.loads(p.read_text(encoding="utf-8"))
    body = "".join(
        f"<tr><td>{r['video']}</td><td>{r['variant']}</td><td>{r['herd']}</td><td>{r['ids']}</td>"
        f"<td><b>{r['ratio']}</b></td><td>{r['mean_len_s']}</td><td>{r['full']}</td></tr>" for r in rows)
    return ("<div class='tbl'><table><tr><th>영상</th><th>방법</th><th>추정 두수</th><th>ID 수</th>"
            "<th>ID/두수</th><th>ID당 평균 지속(s)</th><th>전체지속 ID</th></tr>" + body + "</table></div>")


def videos() -> str:
    items = []
    for p in sorted((ROOT / "outputs").glob("*/*_tracked.mp4")):
        n = p.parent.name
        items.append(f"<figure><video src='../outputs/{n}/{p.name}' controls muted preload='none' "
                     f"poster='../outputs/{n}/{n}_bg.jpg'></video><figcaption>{n} · "
                     f"<a href='../outputs/{n}/dashboard.html'>대시보드</a></figcaption></figure>")
    return "<div class='two'>" + "".join(items) + "</div>"


def train_metrics() -> str:
    p = ROOT / "runs/hanwoo/train/results.csv"
    if not p.exists():
        return "<p class='note'>학습 결과 없음</p>"
    import csv
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    return (f"<div class='kpis'><div><span>학습 epoch</span><b>{len(rows)}</b></div>"
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
