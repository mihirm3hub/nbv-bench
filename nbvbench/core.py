import argparse, os, csv, random, json, yaml
import numpy as np
from .data import load_scene_from_yaml, hemisphere_candidates_auto
from .geo import update_coverage_proxy, compute_auc, fuse_tsdf_stub
from .selectors import Greedy, RandomSel
from .scorers import GTScorer

def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _pick_selector(name: str, cfg: dict):
    name = name.lower()
    if name == "greedy": return Greedy()
    if name == "random": return RandomSel()
    # TODO: add SMA/PSO/RL later
    raise ValueError(f"Unknown selector: {name}")

def main():
    ap = argparse.ArgumentParser(description="nbv-bench runner (scene-centric YAML)")
    ap.add_argument("--cfg", type=str, default="experiments/configs/baseline.yaml")
    ap.add_argument("--selector", type=str, default="greedy",
                    choices=["greedy","random"])
    args = ap.parse_args()

    cfg = _load_cfg(args.cfg)
    seed = int(cfg.get("seed", 0))
    random.seed(seed); np.random.seed(seed)

    # Scene (keeps mesh scale)
    scene = load_scene_from_yaml(cfg)

    # Candidate cameras (auto-scales from mesh extent + FOV if enabled)
    centers, dirs, radius, (fx, fy, W, H) = hemisphere_candidates_auto(cfg, scene)

    # Save candidates + metadata for exact reproducible viz
    out_dir = cfg["report"]["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    tag = cfg["report"].get("tag", "run")
    cand_npy = os.path.join(out_dir, f"{tag}_candidates.npy")
    meta_json = os.path.join(out_dir, f"{tag}_candidates_meta.json")
    np.save(cand_npy, centers.astype(np.float32))
    meta = {
        "M": int(len(centers)),
        "phi_deg_min": float(cfg["hemisphere"]["phi_deg_min"]),
        "phi_deg_max": float(cfg["hemisphere"]["phi_deg_max"]),
        "auto_radius": bool(cfg["hemisphere"].get("auto_radius", False)),
        "radius": float(radius),
        "radius_factor": float(cfg["hemisphere"].get("radius_factor", 3.0)),
        "fit_margin": float(cfg["hemisphere"].get("fit_margin", 1.10)),
        "z_margin_m": float(cfg["hemisphere"].get("z_margin_m", 0.0)),
        "intrinsics": {"W": int(W), "H": int(H), "fx": float(fx), "fy": float(fy)},
        "mesh": scene.name,
        "seed": seed,
    }
    with open(meta_json, "w") as f: json.dump(meta, f, indent=2)
    print(f"[nbv-bench] Saved candidates: {cand_npy}")
    print(f"[nbv-bench] Saved meta:       {meta_json}")

    # Selector & scorer (proxy for now)
    selector = _pick_selector(args.selector, cfg)
    scorer = GTScorer(directions=dirs, cache=True)

    # Stopping policy
    stop = cfg["stopping"]; K = int(stop["max_views"])
    cov_target = float(stop.get("cov_target", 1.0))

    # Output CSVs
    metrics_csv = os.path.join(out_dir, f"metrics_{tag}_{args.selector}.csv")
    sel_csv     = os.path.join(out_dir, f"{tag}_{args.selector}_selected.csv")

    remaining = list(range(len(dirs)))
    history, cov_per_step = [], []

    with open(sel_csv, "w", newline="") as fsel:
        wsel = csv.writer(fsel); wsel.writerow(["step","view_index"])
        for step in range(1, K+1):
            v = selector.choose_next(remaining, history, scorer)
            remaining.remove(v); history.append(v)
            wsel.writerow([step, v])

            cov = update_coverage_proxy(dirs, history)
            cov_per_step.append(cov)
            # TEMP: disable early stop until real coverage is in place
            # if cov >= cov_target: break

    auc = compute_auc(cov_per_step)
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step","coverage"])
        for i, c in enumerate(cov_per_step, 1): w.writerow([i, c])

    if cfg["report"].get("save_mesh", False):
        fuse_tsdf_stub()

    print(f"[nbv-bench] {args.selector} steps={len(history)} AUC@K={auc:.4f} cov_end={cov_per_step[-1]:.3f}")
    print(f"Saved: {metrics_csv}  |  {sel_csv}")

if __name__ == "__main__":
    main()
