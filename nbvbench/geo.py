# nbvbench/geo.py
import numpy as np
from typing import Sequence, Optional, Dict

try:
    import open3d as o3d
except Exception:
    o3d = None


# ---------------- Minimal angular fallback (only used if RayCoverage not set) ----------------
def angular_diversity_gain(directions: np.ndarray, v: int, history: Sequence[int]) -> float:
    if len(history) == 0:
        return 1.0
    dv = directions[v] / (np.linalg.norm(directions[v]) + 1e-9)
    sims = []
    for u in history:
        du = directions[u] / (np.linalg.norm(directions[u]) + 1e-9)
        sims.append(abs(float(np.dot(dv, du))))
    return float(1.0 - max(sims))


# ---------------- Realistic ray-visibility coverage (strict; no synthetic path) ----------------
class RayCoverage:
    """
    Ray visibility coverage with realistic constraints:
      • frustum gating via pinhole intrinsics (fx, fy, W, H)
      • near/far clipping
      • front-facing filter (dot(n, view_dir) > 0)

    Strict MVP: requires Open3D and a real mesh; raises if unavailable.
    """
    def __init__(
        self,
        mesh,
        origin: np.ndarray,
        n_samples: int = 100_000,
        eps: float = 2e-3,
        seed: int = 0,
        intrinsics: Optional[tuple] = None,  # (fx, fy, W, H)
        near_m: float = 0.10,
        far_m: float = 2.00,
        require_front_facing: bool = True,
    ):
        if o3d is None or mesh is None:
            raise RuntimeError("RayCoverage: Open3D and a valid mesh are required.")

        self.origin = np.asarray(origin, dtype=np.float32)
        self.n = int(n_samples)
        self.eps = float(eps)
        self.intr = intrinsics
        self.near_m = float(near_m)
        self.far_m = float(far_m)
        self.require_front_facing = bool(require_front_facing)

        # Tensor raycaster
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(tmesh)
        self._scene = scene

        # GT surface samples
        pcd = mesh.sample_points_uniformly(self.n)
        if not pcd.has_normals():
            pcd.estimate_normals()
        self._points = np.asarray(pcd.points, dtype=np.float32)
        self._normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None

        self._per_view_visible: Dict[int, np.ndarray] = {}

    # --- camera model helpers ---
    def _frustum_mask(self, centers: np.ndarray, v: int) -> np.ndarray:
        if self.intr is None:
            return np.ones(len(self._points), dtype=bool)
        fx, fy, W, H = self.intr
        c = centers[v].astype(np.float32)
        P = self._points

        # Look-at basis (+Z forward to origin)
        fwd = self.origin - c; fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        up_world = np.array([0, 0, 1], np.float32)
        if abs(float(np.dot(fwd, up_world))) > 0.98:
            up_world = np.array([1, 0, 0], np.float32)
        right = np.cross(fwd, up_world); right /= (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, fwd)
        Rcw = np.stack([right, up, fwd], axis=0)  # world->cam

        X = (P - c) @ Rcw.T
        Z = X[:, 2]
        m = (Z > self.near_m) & (Z < self.far_m)
        x = fx * (X[:, 0] / Z) + (W * 0.5)
        y = fy * (X[:, 1] / Z) + (H * 0.5)
        m &= (x >= 0) & (x < W) & (y >= 0) & (y < H)

        if self.require_front_facing and self._normals is not None:
            view_dir = (c - P)
            view_dir = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-9)
            m &= (np.einsum("ij,ij->i", self._normals, view_dir) > 0.0)
        return m

    # --- visibility ---
    def _visible_mask_for_view(self, centers: np.ndarray, v: int) -> np.ndarray:
        if v in self._per_view_visible:
            return self._per_view_visible[v]
        c = centers[v].astype(np.float32)
        fr = self._frustum_mask(centers, v)
        vis = np.zeros(len(self._points), dtype=bool)
        if not np.any(fr):
            self._per_view_visible[v] = vis
            return vis

        idx = np.where(fr)[0]
        Pf = self._points[fr]
        V = Pf - c
        d = np.linalg.norm(V, axis=1) + 1e-9
        dirs = V / d[:, None]

        rays = o3d.core.Tensor(np.hstack([np.repeat(c[None, :], len(Pf), axis=0), dirs]).astype(np.float32))
        t_hit = self._scene.cast_rays(rays)["t_hit"].numpy()
        vis_f = (t_hit > 0) & (np.abs(t_hit - d) <= self.eps)

        vis[idx] = vis_f
        self._per_view_visible[v] = vis
        return vis

    # --- public API ---
    def delta(self, centers: np.ndarray, v: int, history: Sequence[int]) -> float:
        vis_v = self._visible_mask_for_view(centers, v)
        if len(history) == 0:
            return float(np.mean(vis_v))
        union = np.zeros(self.n, dtype=bool)
        for u in history:
            union |= self._visible_mask_for_view(centers, u)
        gain = np.logical_and(vis_v, ~union)
        return float(np.mean(gain))

    def coverage(self, centers: np.ndarray, history: Sequence[int]) -> float:
        if len(history) == 0:
            return 0.0
        union = np.zeros(self.n, dtype=bool)
        for u in history:
            union |= self._visible_mask_for_view(centers, u)
        return float(np.mean(union))


# ---------------- Singleton wiring used by the existing scorer ----------------
_RC: Optional[RayCoverage] = None
_centers_cache: Optional[np.ndarray] = None

def set_ray_coverage(rc: RayCoverage, centers: np.ndarray):
    global _RC, _centers_cache
    _RC = rc
    _centers_cache = centers.astype(np.float32)

def delta_coverage_stub(directions_or_centers: np.ndarray, v: int, history: Sequence[int]) -> float:
    if _RC is not None and _centers_cache is not None:
        return _RC.delta(_centers_cache, v, history)
    return angular_diversity_gain(directions_or_centers, v, history)

def update_coverage_ray(history: Sequence[int]) -> float:
    if _RC is None or _centers_cache is None:
        return 0.0
    return _RC.coverage(_centers_cache, history)

def compute_auc(cov_list: Sequence[float]) -> float:
    return float(np.mean(cov_list)) if len(cov_list) > 0 else 0.0

def fuse_tsdf_stub(*args, **kwargs):
    pass
