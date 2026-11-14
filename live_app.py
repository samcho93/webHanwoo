"""한우 실시간 추적 프로그램 (파일 / RTSP 카메라 / 웹캠).

  GUI     : python live_app.py
  헤드리스: python live_app.py --headless --source rtsp://user:pw@ip:554/stream --save-video
            python live_app.py --headless --source E:\\Video\\cow\\cow09.mp4 --fast
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hanwoo.live import LiveConfig, LiveEngine  # noqa: E402

DEFAULT_PRESET = ROOT / "configs" / "live_default.json"


def headless(args):
    cfg = LiveConfig.load(args.preset) if args.preset else LiveConfig()
    if args.source:
        cfg.source = args.source
    if args.herd_size is not None:
        cfg.herd_size = args.herd_size
    if args.enroll is not None:
        cfg.enroll_s = args.enroll
    if args.fast:
        cfg.realtime = False
    cfg.save_video = args.save_video
    eng = LiveEngine(cfg)
    eng.start()
    t0 = time.time()
    last = 0
    try:
        while eng.running():
            time.sleep(0.5)
            st = eng.status
            if st and time.time() - last > 10:
                last = time.time()
                print(f"  t={st['t']:7.1f}s  phase={st['phase']}  N={st['N']}  visible={st['visible']}  "
                      f"proc {st['fps']:.1f} fps", flush=True)
            if args.max_seconds and st and st.get("t", 0) >= args.max_seconds:
                eng.stop()
    except KeyboardInterrupt:
        eng.stop()
    st = eng.status
    if eng.error:
        print(eng.error)
        return 1
    print(f"done: frames={st.get('frames')} wall={time.time() - t0:.0f}s  N={st.get('N')} "
          f"(동시검출 {st.get('n_sim')}, 외형군집 {st.get('n_clu')})  avg proc {st.get('fps', 0):.1f} fps")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--source")
    ap.add_argument("--preset")
    ap.add_argument("--herd-size", type=int)
    ap.add_argument("--enroll", type=float)
    ap.add_argument("--fast", action="store_true", help="파일을 원래 속도가 아니라 최대 속도로 처리")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0)
    args = ap.parse_args()
    if args.headless:
        sys.exit(headless(args))
    from hanwoo.live_gui import run_gui
    run_gui(args.preset or (DEFAULT_PRESET if DEFAULT_PRESET.exists() else None), args.source)


if __name__ == "__main__":
    main()
