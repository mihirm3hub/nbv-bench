# nbvbench/data.py
import math, os
import numpy as np
from dataclasses import dataclass

try:
    import open3d as o3d
except Exception:
    o3d = None


@dataclass
class Scene:
    name: str
    mesh: "object | None"
    center: np.ndarray
    max_extent: float


def _mesh_center_and_extent(mesh):
    aabb = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(aabb.get_center(), dtype=np.float32)
    extent = float(np.max(aabb.get_extent()))
    return center, extent


def load_scene_from_yaml(cfg: dict) -> Scene:
    scene_cfg = cfg.get("scene", {})
    path = scene_cfg.get("mesh", "") if isinstance(scene_cfg, dict) else cfg.get("mesh_path", "")
    mesh = None
    center = np.zeros(3, dtype=np.float32)
    extent = 1.0
    if path and os.path.exists(path) and o3d is not None:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) == 0 and hasattr(o3d.io, "read_triangle_model"):
            model = o3d.io.read_triangle_model(path)
            if len(model.meshes) > 0:
                mesh = model.meshes[0].mesh
        if hasattr(mesh, "compute_vertex_normals"):
            mesh.compute_vertex_normals()
        center, extent = _mesh_center_and_extent(mesh)
    name = os.path.basename(path) if path else "scene"
    return Scene(name=name, mesh=mesh, center=center, max_extent=extent)


def _fx_fy_from_fov(W, H, fov_deg):
    f = 0.5 * W / math.tan(math.radians(fov_deg) / 2.0)
    return f, f


def _fov_from_fx_fy(W, H, fx, fy):
    hfov = 2.0 * math.atan(0.5 * W / fx)
    vfov = 2.0 * math.atan(0.5 * H / fy)
    return hfov, vfov


def hemisphere_candidates_auto(cfg: dict, scene: Scene):
    """
    Build hemisphere candidate cameras that auto-scale with mesh + intrinsics.
    Returns: centers[M,3], dirs[M,3], radius, (fx,fy,W,H), origin[3]
    """
    # intrinsics
    W = int(cfg["intrinsics"]["width"])
    H = int(cfg["intrinsics"]["height"])
    intr = cfg["intrinsics"]
    if "fx" in intr and "fy" in intr:
        fx = float(intr["fx"]); fy = float(intr["fy"])
    else:
        fx, fy = _fx_fy_from_fov(W, H, float(intr["fov_deg"]))

    hemi = cfg["hemisphere"]
    phi_min = math.radians(float(hemi["phi_deg_min"]))
    phi_max = math.radians(float(hemi["phi_deg_max"]))
    # Accept both names; treat equivalently
    z_margin = float(hemi.get("z_offset_m", hemi.get("z_margin_m", 0.0)))
    M = int(hemi.get("M", 256))
    auto_radius = bool(hemi.get("auto_radius", False))
    radius_factor = float(hemi.get("radius_factor", 3.0))
    fit_margin = float(hemi.get("fit_margin", 1.10))

    half = 0.5 * scene.max_extent
    if auto_radius:
        hfov, vfov = _fov_from_fx_fy(W, H, fx, fy)
        d_fit_h = (half / math.tan(0.5 * hfov)) if hfov > 1e-6 else 1e9
        d_fit_v = (half / math.tan(0.5 * vfov)) if vfov > 1e-6 else 1e9
        d_fit = max(d_fit_h, d_fit_v) * fit_margin
        d_base = radius_factor * half
        radius = max(d_fit, d_base)
    else:
        radius = float(hemi.get("r0_m", 0.30))

    # Fibonacci hemisphere directions
    # --- Area-uniform spherical Fibonacci sampling over [phi_min, phi_max] ---
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z_hi = math.cos(phi_min)   # closer to +Z
    z_lo = math.cos(phi_max)   # closer to equator
    dirs = []
    for i in range(M):
        # uniform in z between [z_lo, z_hi] yields near-uniform surface area
        t = (i + 0.5) / M
        z = z_lo + (z_hi - z_lo) * t
        rxy = max(0.0, 1.0 - z*z) ** 0.5
        theta = i * golden_angle
        v = np.array([math.cos(theta) * rxy,
                      math.sin(theta) * rxy,
                      z], dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        dirs.append(v)
    dirs = np.stack(dirs, axis=0)


    # Origin choice
    anchor = str(hemi.get("anchor", "center")).lower()
    if anchor == "base" and scene.mesh is not None:
        aabb = scene.mesh.get_axis_aligned_bounding_box()
        z_min = float(aabb.get_min_bound()[2])
        base = np.array([scene.center[0], scene.center[1], z_min], dtype=np.float32)
        origin = base + np.array([0, 0, z_margin], dtype=np.float32)
    else:
        origin = scene.center + np.array([0, 0, z_margin], dtype=np.float32)

    centers = dirs * float(radius) + origin
    return centers.astype(np.float32), dirs.astype(np.float32), float(radius), (float(fx), float(fy), int(W), int(H)), origin.astype(np.float32)


__all__ = [
    "Scene",
    "load_scene_from_yaml",
    "hemisphere_candidates_auto",
]
