"""
main.py — Sign Language Recognition System  (Isolated ASL)
----------------------------------------------------------
  [1]  Isolated Sign Mode (ASL)
       Record a 2-second window → Top-5 predictions.
       Model: Phase-Aware TCN trained on ASL vocabulary.

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

BANNER = """
  +----------------------------------------------------------+
  |          Sign Language Recognition System                |
  |                   Capstone Project                       |
  +----------------------------------------------------------+
  |                                                          |
  |   [1]   Isolated Sign Mode  (ASL)                        |
  |         Press R to record one sign -> Top-5 predictions  |
  |                                                          |
  |   [Q]   Quit                                             |
  |                                                          |
  +----------------------------------------------------------+
"""

DIVIDER = "  " + "─" * 56


def show_menu() -> str:
    print(BANNER)
    return input("  Select mode (1 / Q): ").strip().lower()


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
