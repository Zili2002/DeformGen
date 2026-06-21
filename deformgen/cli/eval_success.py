from __future__ import annotations

import argparse

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate last-frame success for replay outputs.")
    parser.add_argument("--case", choices=["rope", "sloth", "cloth3"], required=True)
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print("DeformGen eval_success wrapper")
    print(f"case={args.case} replay_root={args.replay_root} out={args.out}")
    print("Use experiments/utils/calculate_success_*_lastframe.py for case-specific metrics.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
