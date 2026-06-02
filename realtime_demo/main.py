"""
main.py — Sign Language Recognition System  (Unified Launcher)
--------------------------------------------------------------
  [1]  Isolated Sign Mode (ASL)
       Record a 2-second window → Top-5 predictions.
       Model: Phase-Aware TCN trained on ASL vocabulary.

  [2]  Continuous Translation Mode (German SL)
       Sign freely → live running gloss sentence.
       Model: AdaptSign ViT-B/16 trained on PHOENIX-2014-T.

Usage:
  python realtime_demo/main.py
"""

import sys
from pathlib import Path

# Ensure pipeline.py and sibling modules are importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────

BANNER = r"""
  ╔══════════════════════════════════════════════════════════╗
  ║          Sign Language Recognition System                ║
  ║                   Capstone Project                       ║
  ╠══════════════════════════════════════════════════════════╣
  ║                                                          ║
  ║   [1]   Isolated Sign Mode  (ASL)                        ║
  ║         Press R to record one sign → Top-5 predictions   ║
  ║                                                          ║
  ║   [2]   Continuous German SL  (DGS — CorrNet)            ║
  ║         Auto-detects signing → translated German text    ║
  ║         ~3-5 s per sign on CPU  (ResNet18 backbone)      ║
  ║                                                          ║
  ║   [Q]   Quit                                             ║
  ║                                                          ║
  ╚══════════════════════════════════════════════════════════╝
"""

DIVIDER = "  " + "─" * 56


def show_menu() -> str:
    print(BANNER)
    return input("  Select mode (1 / 2 / Q): ").strip().lower()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    while True:
        choice = show_menu()

        if choice == "1":
            print(f"\n{DIVIDER}")
            print("  Launching Isolated Sign Mode ...")
            print(f"{DIVIDER}\n")
            try:
                from demo import main as run_isolated
                run_isolated()
            except Exception as exc:
                print(f"\n  [ERROR in Isolated Mode] {exc}\n")

        elif choice == "2":
            print(f"\n{DIVIDER}")
            print("  Launching Continuous German SL (CorrNet) ...")
            print("  Loading ResNet18 model — please wait ~20 s ...")
            print(f"{DIVIDER}\n")
            try:
                import importlib.util, pathlib
                spec = importlib.util.spec_from_file_location(
                    "corrnet_webcam",
                    pathlib.Path(__file__).resolve().parent.parent
                    / "corrnet" / "corrnet_webcam.py",
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.main()
            except Exception as exc:
                print(f"\n  [ERROR in German SL Mode] {exc}\n")

        elif choice in ("q", "quit", "exit", ""):
            print("\n  Goodbye!\n")
            break

        else:
            print(f"\n  Unknown choice '{choice}' — please enter 1, 2, or Q.\n")

        # After a mode exits, offer to go back to the menu
        print(f"\n{DIVIDER}")
        again = input("  Return to menu? (Y / N): ").strip().lower()
        if again not in ("y", "yes", ""):
            print("\n  Goodbye!\n")
            break


if __name__ == "__main__":
    main()
