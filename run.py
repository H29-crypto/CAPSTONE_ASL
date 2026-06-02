"""
run.py — Sign Language Recognition System entry point.

Delegates to realtime_demo/main.py which hosts both modes:
  [1]  Isolated ASL sign recognition  (Phase-aware TCN, live webcam)
  [2]  Continuous German SL           (AdaptSign, PHOENIX-2014-T)
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).parent / "realtime_demo" / "main.py"),
        run_name="__main__",
    )
