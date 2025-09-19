# nbvbench/core.py
import argparse, os, csv, random, json, yaml
import numpy as np
from .data import load_scene_from_yaml, hemisphere_candidates_auto
from .geo import compute_auc, RayCoverage, set_ray_coverage, update_coverage_ray
from .selectors import Greedy, RandomSel
from .scorers import GTScorer


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _pick_selector(name: str):
    name = name.lower()
    if name == "greedy": return Greedy()
    if name == "random": return RandomSel()
    # future: add PSO, SMA, etc.
    raise ValueError(f"Unknown selector: {name}")


def main():
    ap = argparse.ArgumentParser(description="NBV-Bench (MVP)")
    ap.add_argument("--cfg", type=str, default="experiments/configs/baseline.yaml")
    ap.add_argument("--selector", type=str, default="greedy", choices=["greedy", "random"])
    args = ap.parse_args()

    cfg = _load_cfg(args.cfg)
    seed = int(cfg.get("seed", 0))
    random.seed(seed); np.random.seed(seed)

    # Scene & candidates
    scene = load_scene_from_yaml(cfg)
    centers, dirs, radius, (fx, fy, W, H), origin = hemisphere_candidates_auto(cfg, scene)

    # Realistic ray-coverage (strict; no fallbacks)
    # after:
    # centers, dirs, radius, (fx, fy, W, H), origin = hemisphere_candidates_auto(cfg, scene)

    cov_cfg = cfg.get("coverage", {})
    near_m = float(cov_cfg.get("near_m", 0.10))
    far_cfg = float(cov_cfg.get("far_m", 2.0))
    far_m  = max(far_cfg, 1.25 * float(radius))   # ensure far plane > camera→origin

    rc = RayCoverage(
        mesh=scene.mesh, origin=origin,
        n_samples=int(cov_cfg.get("n_samples", 100000)),
        eps=float(cov_cfg.get("eps", 0.002)),
        seed=seed,
        intrinsics=(fx, fy, W, H),
        near_m=float(cov_cfg.get("near_m", 0.10)),
        far_m=float(cov_cfg.get("far_m", 2.00)),
        require_front_facing=bool(cov_cfg.get("require_front_facing", True)),
    )
    set_ray_coverage(rc, centers)

    # Selector wires through GTScorer -> geo.delta_coverage_stub -> RayCoverage
    selector = _pick_selector(args.selector)
    scorer = GTScorer(directions=centers, cache=True)

    # Stopping policy
    stop = cfg["stopping"]
    K = int(stop["max_views"])
    cov_target = float(stop.get("cov_target", 1.0))

    # Minimal outputs only
    out_dir = cfg["report"]["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    tag = cfg["report"].get("tag", "run")
    metrics_csv = os.path.join(out_dir, f"metrics_{tag}_{args.selector}.csv")
    sel_csv     = os.path.join(out_dir, f"{tag}_{args.selector}_selected.csv")

    remaining = list(range(len(centers)))
    history, cov_per_step = [], []

    with open(sel_csv, "w", newline="") as fsel:
        wsel = csv.writer(fsel); wsel.writerow(["step","view_index"])
        for step in range(1, K+1):
            v = selector.choose_next(remaining, history, scorer)
            remaining.remove(v); history.append(v)
            wsel.writerow([step, v])

            cov = update_coverage_ray(history)
            cov_per_step.append(cov)
            print(f"[step {step:02d}] v={v} cov={cov:.4f}")
            if cov >= cov_target: break

    auc = compute_auc(cov_per_step)
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step","coverage"])
        for i, c in enumerate(cov_per_step, 1): w.writerow([i, c])

    print(f"[nbv-bench] {args.selector} steps={len(history)} AUC@K={auc:.4f} cov_end={cov_per_step[-1]:.3f}")
    print(f"Saved: {metrics_csv}  |  {sel_csv}")


if __name__ == "__main__":
    main()
