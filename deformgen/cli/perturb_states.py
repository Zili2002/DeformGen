from __future__ import annotations

import argparse

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate perturbed deformable-object states (public DeformGen wrapper).")
    parser.add_argument("--case", choices=["rope", "sloth", "cloth3"], required=True)
    parser.add_argument("--out", required=True, help="Output directory for generated states/manifests.")
    parser.add_argument("--num-states", type=int, default=1)
    parser.add_argument("--mode", default="grid-rigid", choices=["random", "grid", "grid-rigid", "default"])
    parser.add_argument("--config", default=None, help="Optional Hydra/YAML config path.")
    args = parser.parse_args()
    print("DeformGen perturb_states wrapper")
    print(f"case={args.case} mode={args.mode} num_states={args.num_states} out={args.out}")
    print("For full simulation-backed generation, see docs/state_perturbation.md and scripts/perturb/.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
