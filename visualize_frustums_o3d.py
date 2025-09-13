#!/usr/bin/env python3
# visualize_frustums_o3d.py
import argparse, csv, json, os, math
from typing import List
import numpy as np
import open3d as o3d
import yaml

from nbvbench.data import load_scene_from_yaml, hemisphere_candidates_auto

# -------- CSV (selected indices) --------
def read_selection_indices_csv(path: str) -> List[int]:
    with open(path, "r", newline="") as f:
        try:
            rdr = csv.DictReader(f)
            if rdr.fieldnames is None:
                f.seek(0); return [int(r[0]) for r in csv.reader(f) if r]
            key = None
            for k in ["view_index","index","view","v","candidate","id"]:
                if k in rdr.fieldnames: key = k; break
            if key is None and "step" in rdr.fieldnames:
                others = [c for c in rdr.fieldnames if c!="step"]
                if others: key = others[0]
            if key is None: raise ValueError(f"No index column in headers {rdr.fieldnames}")
            return [int(r[key]) for r in rdr if r.get(key,"").strip()]
        except Exception:
            f.seek(0); return [int(r[0]) for r in csv.reader(f) if r]

# -------- Camera helpers (cam->world, forward=-Z) --------
def look_at_extrinsic(eye: np.ndarray,
                      center: np.ndarray,
                      up=np.array([0,0,1.0], dtype=np.float32)) -> np.ndarray:
    # Open3D expects world->camera extrinsic here.
    f = center - eye
    f = f / (np.linalg.norm(f) + 1e-12)
    u = up / (np.linalg.norm(up) + 1e-12)
    s = np.cross(f, u); s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    R = np.stack([s, u, f], axis=0)     # rows
    t = -R @ eye.reshape(3,1)
    T_w2c = np.eye(4)
    T_w2c[:3,:3] = R
    T_w2c[:3, 3] = t[:,0]
    return T_w2c


def frustum_lineset(W, H, fx, fy, cx, cy, extrinsic, scale=0.1, color=(0,1,0)):
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    ls = o3d.geometry.LineSet.create_camera_visualization(W, H, K, extrinsic, scale)
    cols = np.tile(np.array(color, dtype=float)[None,:], (len(ls.lines),1))
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


def color_by_step(i: int, total: int, base: np.ndarray):
    a = 0.3 + 0.7*(i/max(1,total-1))
    return tuple(a*base + (1-a)*np.array([0.7,0.7,0.7]))

# -------- Intrinsics helper (if YAML uses fov_deg) --------
def fx_fy_from_fov(W, H, fov_deg):
    f = 0.5 * W / math.tan(math.radians(fov_deg)/2.0)
    return f, f

# -------- Main --------
def main():
    ap = argparse.ArgumentParser(description="NBV frustum visualizer (loads saved candidates if present)")
    ap.add_argument("--cfg", type=str, default="experiments/configs/baseline.yaml")
    ap.add_argument("--indices_csv", type=str, required=True)
    ap.add_argument("--recenter", action="store_true")
    ap.add_argument("--show_candidates", type=lambda s: s.lower()!="false", default=True)
    ap.add_argument("--limit_frustums", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--frustum_scale_factor", type=float, default=0.30)
    ap.add_argument("--base_color", type=float, nargs=3, default=[0.1,0.7,0.2])
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # Intrinsics (support fov_deg or fx/fy)
    W = int(cfg["intrinsics"]["width"]); H = int(cfg["intrinsics"]["height"])
    intr = cfg["intrinsics"]
    if "fx" in intr and "fy" in intr: fx, fy = float(intr["fx"]), float(intr["fy"])
    else: fx, fy = fx_fy_from_fov(W, H, float(intr["fov_deg"]))
    cx, cy = W/2.0, H/2.0

    # Scene
    scene = load_scene_from_yaml(cfg)
    hemi = cfg["hemisphere"]
    anchor = str(hemi.get("anchor", "center")).lower()
    z_off = float(hemi.get("z_offset_m", 0.0))
    z_margin = float(hemi.get("z_margin_m", 0.0))  # still supported

    if anchor == "base" and scene.mesh is not None:
        aabb = scene.mesh.get_axis_aligned_bounding_box()
        z_min = float(aabb.get_min_bound()[2])
        obj_center = np.array([scene.center[0], scene.center[1], z_min + z_off], dtype=np.float32)
    else:
        obj_center = scene.center + np.array([0,0,z_margin], dtype=np.float32)


    # Prefer saved candidates from the run (exact match)
    tag = cfg.get("report", {}).get("tag", "run")
    out_dir = cfg["report"]["out_dir"]
    cand_npy  = os.path.join(out_dir, f"{tag}_candidates.npy")
    meta_json = os.path.join(out_dir, f"{tag}_candidates_meta.json")

    if os.path.exists(cand_npy):
        centers = np.load(cand_npy)  # (M,3)
        # Re-derive mean radius and dirs for orientation
        radius = np.mean(np.linalg.norm(centers - obj_center, axis=1))
        dirs = centers - obj_center
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
        meta = {}
        if os.path.exists(meta_json):
            with open(meta_json, "r") as f: meta = json.load(f)
            # Soft checks: warn if YAML and meta disagree
            warn = []
            if int(meta.get("M", len(centers))) != len(centers): warn.append("M")
            if abs(meta.get("radius", radius) - radius) > 1e-5: warn.append("radius")
            if meta.get("intrinsics", {}).get("W") != W or meta.get("intrinsics", {}).get("H") != H: warn.append("W/H")
            if warn: print(f"[viz] WARNING: meta mismatch on {warn}; using saved candidates anyway.")
        print(f"[viz] Loaded saved candidates: {cand_npy}")
    else:
        # Fall back to recomputing (should still match if cfg hasn't changed)
        centers, dirs, radius, _ = hemisphere_candidates_auto(cfg, scene)
        print("[viz] No saved candidates found; recomputed from YAML.")

    # Optional recenter for drawing only
    shift = -scene.center if args.recenter else np.zeros(3, dtype=np.float32)
    obj_center_vis = obj_center + shift
    scale_factor = 1.5   # 1.5x bigger hemisphere
    centers_vis = (centers - obj_center) * scale_factor + obj_center + shift


    # Selections (declutter)
    sel = read_selection_indices_csv(args.indices_csv)
    if args.stride > 1: sel = sel[::args.stride]
    if args.limit_frustums > 0: sel = sel[:args.limit_frustums]
    print(f"[viz] frustums to draw: {len(sel)}")

    # Build scene geoms
    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)]
    if scene.mesh is not None:
        mesh = scene.mesh
        if args.recenter: mesh = mesh.translate(shift)
        mesh.compute_vertex_normals(); mesh.paint_uniform_color([0.82,0.82,0.82])
        geoms.append(mesh)

    if args.show_candidates:
        pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(centers_vis.astype(np.float64)))
        pc.paint_uniform_color([0.75,0.75,0.75]); geoms.append(pc)

    # Frustums
    frustum_scale = max(args.frustum_scale_factor * max(scene.max_extent, 1e-6), 0.005)
    base = np.array(args.base_color, dtype=float)

    for i, idx in enumerate(sel):
        eye = centers_vis[idx]
        extr = look_at_extrinsic(eye, obj_center_vis)
        col = color_by_step(i, len(sel), base)
        geoms.append(frustum_lineset(int(W), int(H), float(fx), float(fy), float(cx), float(cy),
                                     extrinsic=extr, scale=frustum_scale, color=tuple(col)))

    print(f"[viz] mesh={scene.name} center={obj_center_vis} max_extent={scene.max_extent:.4f} "
          f"radius≈{radius:.4f} frustum_scale={frustum_scale:.4f}")

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geoms: vis.add_geometry(g)
    ctr = vis.get_view_control()
    ctr.set_lookat(obj_center_vis.astype(float).tolist())
    ctr.set_front([0.5, -0.5, 0.5])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(1.0)
    vis.run(); vis.destroy_window()

if __name__ == "__main__":
    main()
