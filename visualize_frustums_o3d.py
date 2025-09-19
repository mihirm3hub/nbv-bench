#!/usr/bin/env python3
"""
NBV-Bench visualizer

Default look (mode=aim + spheres):
  • gray candidate spheres (size ∝ scene extent)
  • larger green selected spheres
  • green aim lines to origin
  • white background

Modes:
  - aim      : aim lines + selected spheres (default)
  - frusta   : camera frusta only (with selected spheres)
  - both     : overlay frusta + aim lines

Examples:
python visualize_frustums_o3d.py \
  --cfg experiments/configs/baseline.yaml \
  --indices_csv experiments/results/baseline_greedy_selected.csv \
  --mode both --cand_glyph spheres --recenter --axes --auto_fit \
  --cand_radius_factor 0.015 --sel_radius_factor 0.03 --frustum_scale_factor 0.8
"""
import argparse, csv, os, json, numpy as np, yaml

try:
    import open3d as o3d
except Exception as e:
    raise RuntimeError("This visualizer requires Open3D. Try: pip install open3d") from e

from nbvbench.data import load_scene_from_yaml, hemisphere_candidates_auto


# -------------------- helpers --------------------

def read_selected_indices(csv_path: str) -> np.ndarray:
    idx = []
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)  # header
        for row in r:
            if len(row) >= 2:
                idx.append(int(row[1]))
    return np.array(idx, dtype=np.int32)


def lookat_basis(c: np.ndarray, origin: np.ndarray):
    fwd = origin - c
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    up_world = np.array([0, 0, 1], np.float32)
    if abs(float(np.dot(fwd, up_world))) > 0.98:
        up_world = np.array([1, 0, 0], np.float32)
    right = np.cross(fwd, up_world); right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, fwd)
    Rwc = np.stack([right, up, fwd], axis=1)
    return Rwc


def make_frustum_lineset(c: np.ndarray, origin: np.ndarray, intr: tuple, scale: float, color=(0,0.6,0)):
    """Wire frustum at camera center c looking at origin."""
    fx, fy, W, H = intr
    cx, cy = 0.5 * W, 0.5 * H
    Rwc = lookat_basis(c, origin)
    # image corners in cam frame (z=1)
    us = np.array([0.0, W, W, 0.0], dtype=np.float32)
    vs = np.array([0.0, 0.0, H, H], dtype=np.float32)
    x = (us - cx) / fx; y = (vs - cy) / fy; z = np.ones_like(x)
    D_cam = np.stack([x, y, z], axis=1)
    D_cam = D_cam / (np.linalg.norm(D_cam, axis=1, keepdims=True) + 1e-9)
    D_world = (Rwc @ D_cam.T).T
    corners = c[None, :] + scale * D_world
    pts = np.vstack([c[None, :], corners]).astype(np.float32)
    lines = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]], dtype=np.int32)
    col = np.tile(np.asarray(color, dtype=np.float32), (len(lines), 1))
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(col)
    return ls


def make_aim_lines(centers: np.ndarray, origin: np.ndarray, sel_idx: np.ndarray, color=(0,0.6,0)):
    """LineSet with lines from each selected camera to origin."""
    if len(sel_idx) == 0:
        return None
    points = [origin]
    lines, colors = [], []
    for i, v in enumerate(sel_idx, start=1):
        points.append(centers[v])
        lines.append([0, i])
        colors.append(color)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32))
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float32))
    return ls


def mesh_aabb_lines(mesh, color=(1,0,0)):
    aabb = mesh.get_axis_aligned_bounding_box()
    aabb.color = np.asarray(color, dtype=np.float64)
    return aabb


def add_spheres(centers: np.ndarray, idx: np.ndarray, radius: float, color=(0.1,0.8,0.1), stride: int = 1, limit: int = 0):
    """Add sphere glyphs at centers[idx]."""
    out = []
    if idx is None:
        iterable = centers[::max(1, stride)]
    else:
        iterable = centers[idx][::max(1, stride)]
    count = 0
    for c in iterable:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sph.compute_vertex_normals()
        sph.paint_uniform_color(color)
        sph.translate(c)
        out.append(sph)
        count += 1
        if limit > 0 and count >= limit:
            break
    return out


def combined_aabb(geoms):
    mins, maxs = [], []
    for g in geoms:
        if hasattr(g, "get_axis_aligned_bounding_box"):
            bb = g.get_axis_aligned_bounding_box()
            mins.append(np.asarray(bb.get_min_bound()))
            maxs.append(np.asarray(bb.get_max_bound()))
    if not mins: return None
    mins = np.vstack(mins).min(axis=0); maxs = np.vstack(maxs).max(axis=0)
    return o3d.geometry.AxisAlignedBoundingBox(min_bound=mins, max_bound=maxs)


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser("NBV-Bench Visualizer")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--indices_csv", required=True)
    # Optional saved candidates/meta (not required)
    ap.add_argument("--candidates_npy", default="")
    ap.add_argument("--meta_json", default="")
    # Modes
    ap.add_argument("--mode", choices=["aim","frusta","both"], default="aim")
    # Candidate glyphs
    ap.add_argument("--cand_glyph", choices=["spheres","points"], default="spheres")
    ap.add_argument("--cand_stride", type=int, default=1, help="Draw every N-th candidate (for performance)")
    ap.add_argument("--cand_limit", type=int, default=0, help="Max candidate glyphs (0 = no cap)")
    # Toggles
    ap.add_argument("--show_candidates", type=str, default="true")  # true/false
    ap.add_argument("--show_bbox", action="store_true")
    ap.add_argument("--axes", action="store_true")
    ap.add_argument("--recenter", action="store_true")
    # Scale & sizes
    ap.add_argument("--unit_scale", type=float, default=1.0, help="e.g., 0.001 for mm→m")
    ap.add_argument("--cand_point_size", type=float, default=2.5)  # only used if cand_glyph=points
    ap.add_argument("--cand_radius_factor", type=float, default=0.010, help="candidate sphere radius = factor * scene_extent")
    ap.add_argument("--cand_radius_abs", type=float, default=0.0, help="absolute candidate sphere radius (overrides factor)")
    ap.add_argument("--sel_radius_factor", type=float, default=0.025, help="selected sphere radius = factor * scene_extent")
    ap.add_argument("--frustum_scale_factor", type=float, default=0.6)
    ap.add_argument("--frustum_scale_abs", type=float, default=0.0)
    ap.add_argument("--frustum_color", type=str, default="0,0.6,0", help="r,g,b in [0,1] for frusta")
    # Selection sampling
    ap.add_argument("--stride", type=int, default=1, help="Subsample selected indices")
    ap.add_argument("--limit_frustums", type=int, default=0)
    # Viewer behavior
    ap.add_argument("--auto_fit", action="store_true")
    ap.add_argument("--line_width", type=float, default=1.8)
    args = ap.parse_args()

    show_candidates = str(args.show_candidates).lower() not in ["false","0","no"]
    fr_col = tuple(float(v) for v in args.frustum_color.split(",")) if args.frustum_color else (0,0.6,0)

    # Load cfg & scene
    cfg = yaml.safe_load(open(args.cfg, "r"))
    scene = load_scene_from_yaml(cfg)
    if scene.mesh is None:
        raise RuntimeError("Could not load mesh from config.")

    # Get candidates, intrinsics, origin (5-return API)
    centers = None; intr = None; origin = None
    if args.candidates_npy and os.path.exists(args.candidates_npy):
        centers = np.load(args.candidates_npy).astype(np.float32)
        if args.meta_json and os.path.exists(args.meta_json):
            meta = json.load(open(args.meta_json, "r"))
            intr = (float(meta["intrinsics"]["fx"]), float(meta["intrinsics"]["fy"]),
                    int(meta["intrinsics"]["W"]), int(meta["intrinsics"]["H"]))
            origin = np.array(meta.get("origin", scene.center.tolist()), dtype=np.float32)
        else:
            _, _, _, intr, origin = hemisphere_candidates_auto(cfg, scene)
    else:
        centers, _, _, intr, origin = hemisphere_candidates_auto(cfg, scene)

    fx, fy, W, H = intr

    # Selected indices
    sel = read_selected_indices(args.indices_csv)
    if args.stride > 1: sel = sel[::max(1,int(args.stride))]
    if args.limit_frustums > 0 and len(sel) > args.limit_frustums: sel = sel[:args.limit_frustums]

    # Scale / recenter
    s = float(args.unit_scale)
    scene_extent = max(scene.max_extent * s, 1e-6)
    origin_vis = origin * s
    shift = -origin_vis if args.recenter else np.zeros(3, dtype=np.float32)

    # Mesh (light gray)
    mesh = scene.mesh
    if s != 1.0: mesh = mesh.scale(s, center=origin)  # keep origin fixed
    if args.recenter: mesh = mesh.translate(shift)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.85, 0.85, 0.85])

    geoms = [mesh]

    # BBox
    if args.show_bbox:
        geoms.append(mesh_aabb_lines(mesh, color=(1,0,0)))

    # --- Candidates (GRAY) ---
    if show_candidates and centers is not None:
        centers_vis = (centers * s) + shift
        if args.cand_glyph == "spheres":
            cand_r = args.cand_radius_abs if args.cand_radius_abs > 0.0 else max(args.cand_radius_factor * scene_extent, 1e-5)
            geoms += add_spheres(centers_vis, idx=None, radius=cand_r, color=(0.70, 0.70, 0.70),
                                 stride=max(1, int(args.cand_stride)), limit=int(args.cand_limit))
        else:
            pts = o3d.geometry.PointCloud()
            pts.points = o3d.utility.Vector3dVector(centers_vis)
            pts.paint_uniform_color([0.70, 0.70, 0.70])  # gray
            geoms.append(pts)

    # --- Selected spheres (GREEN) ---
    centers_vis = (centers * s) + shift
    sel_r = max(args.sel_radius_factor * scene_extent, 1e-4)
    geoms += add_spheres(centers_vis, idx=sel, radius=sel_r, color=(0.1, 0.8, 0.1))

    # --- AIM lines / Frusta / Both ---
    want_aim = (args.mode in ("aim","both"))
    want_frusta = (args.mode in ("frusta","both"))

    if want_aim:
        ls = make_aim_lines(centers_vis, origin_vis + shift, sel, color=(0, 0.6, 0))
        if ls is not None: geoms.append(ls)

    if want_frusta and len(sel) > 0:
        fr_len = float(args.frustum_scale_abs) if args.frustum_scale_abs > 0.0 else max(args.frustum_scale_factor * scene_extent, 1e-3)
        frusta = []
        for v in sel:
            c = centers[v] * s + shift
            frusta.append(make_frustum_lineset(c, origin_vis + shift, (fx, fy, W, H), fr_len, color=fr_col))
        fused = frusta[0]
        for f in frusta[1:]: fused += f
        geoms.append(fused)

    # Axes
    if args.axes:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 * scene_extent, origin=origin_vis + shift))

    # ----- Show (WHITE BACKGROUND) -----
    vis = o3d.visualization.Visualizer()
    vis.create_window("NBV-Bench Visualizer")
    for g in geoms: vis.add_geometry(g)

    opt = vis.get_render_option()
    if hasattr(opt, "background_color"):
        opt.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # white background
    if hasattr(opt, "line_width"): opt.line_width = float(args.line_width)
    if hasattr(opt, "point_size"): opt.point_size = float(args.cand_point_size)  # used if cand_glyph=points

    # Auto-fit (simple)
    bbox = combined_aabb(geoms)
    if args.auto_fit and bbox is not None:
        ctr = vis.get_view_control()
        ctr.set_lookat(bbox.get_center())
        ctr.set_up([0, 0, 1])
        ctr.set_front([0, -1, -0.3])
        ctr.set_zoom(0.7)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
