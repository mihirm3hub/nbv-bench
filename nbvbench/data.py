# nbvbench/data.py
import math, os
import numpy as np
from dataclasses import dataclass

try:
    import open3d as o3d
except Exception:
    o3d = None  # mesh loading becomes no-op if Open3D isn't installed


@dataclass
class Scene:
    name: str
    mesh: "object | None"
    center: np.ndarray     # (3,)
    max_extent: float      # scalar


def _mesh_center_and_extent(mesh):
    aabb = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(aabb.get_center(), dtype=np.float32)
    extent = float(np.max(aabb.get_extent()))
    return center, extent


def load_scene_from_yaml(cfg: dict) -> Scene:
    """
    Load a mesh (if mesh_path exists) and compute its center+extent.
    Returns a Scene with mesh (or None), center, and max_extent.
    """
    path = cfg.get("mesh_path", "")
    mesh = None
    center = np.zeros(3, dtype=np.float32)
    extent = 1.0
    if path and os.path.exists(path) and o3d is not None:
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        center, extent = _mesh_center_and_extent(mesh)
    name = os.path.basename(path) if path else "toy"
    return Scene(name=name, mesh=mesh, center=center, max_extent=extent)


def _fx_fy_from_fov(W, H, fov_deg):
    # simple symmetric FOV → fx=fy
    f = 0.5 * W / math.tan(math.radians(fov_deg) / 2.0)
    return f, f


def _fov_from_fx_fy(W, H, fx, fy):
    hfov = 2.0 * math.atan(0.5 * W / fx)
    vfov = 2.0 * math.atan(0.5 * H / fy)
    return hfov, vfov


def hemisphere_candidates_auto(cfg: dict, scene: Scene):
    """
    Build hemisphere candidate camera centers that auto-scale with the mesh + intrinsics.
    Returns (centers[M,3], dirs[M,3], radius, (fx,fy,W,H)).
    """
    # intrinsics -> fx, fy
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
    z_margin = float(hemi.get("z_margin_m", 0.0))
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
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    dirs = []
    for i in range(M):
        t = i / max(1, M - 1)
        phi = phi_min + t * (phi_max - phi_min)
        theta = i * golden_angle
        v = np.array([math.cos(theta) * math.sin(phi),
                      math.sin(theta) * math.sin(phi),
                      math.cos(phi)], dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        dirs.append(v)
    dirs = np.stack(dirs, axis=0)

    # choose hemisphere origin
    anchor = str(hemi.get("anchor", "center")).lower()
    z_off = float(hemi.get("z_offset_m", 0.0))
    if anchor == "base" and scene.mesh is not None:
        aabb = scene.mesh.get_axis_aligned_bounding_box()
        z_min = float(aabb.get_min_bound()[2])
        origin = np.array([scene.center[0], scene.center[1], z_min + z_off], dtype=np.float32)
    else:
        origin = scene.center + np.array([0, 0, z_margin], dtype=np.float32)

    centers = dirs * radius + origin
    return centers, dirs, radius, (fx, fy, W, H)


__all__ = [
    "Scene",
    "load_scene_from_yaml",
    "hemisphere_candidates_auto",
]
