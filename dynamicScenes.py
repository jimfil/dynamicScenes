from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D_, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)

import heapq
import numpy as np
import csv
from pathlib import Path
import time
import src.readlazfiles as rl
import src.octnode as octN
import src.kdnode as kdN
import src.quadnode as quadN

WIDTH = 1000
HEIGHT = 800

COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.MAGENTA, Color.YELLOWGREEN, Color.CYAN]

class DynamicScenes(Scene3D_):
    # ================= CLASS-LEVEL DEFAULTS (tree construction) =================
    # Octree parameters – adjustable without touching individual method calls
    OCT_MAX_DEPTH: int = 6
    OCT_CAPACITY:  int = 50

    # Quadtree parameters – adjustable without touching individual method calls
    QUAD_MAX_DEPTH: int = 7
    QUAD_CAPACITY:  int = 40

    # ================= INITIALIZATION =================
    def __init__(self,
                 crop_percentage: int = 88,
                 downsample_step: int = 15):
        """Initialise the scene and pre-compute all expensive data.

        Parameters
        ----------
        crop_percentage : int
            X/Y percentile cut-off used by ``getPointsPartial`` to restrict
            the loaded point cloud to a spatial sub-region (default 88 %).
        downsample_step : int
            Stride used when thinning point clouds for ICP and KD-tree
            classification inside ``process_data`` (default every 15th point).
        """
        super().__init__(WIDTH, HEIGHT, "DynamicScenes")

        # Store parametric values as instance attributes for later inspection
        self.crop_percentage  = crop_percentage
        self.downsample_step  = downsample_step

        # Load raw datasets using the configurable crop percentage
        self.pts2016 = self.getPointsPartial("0_5D4KVPBP.laz", 2016, remove_ground=True, percentage=crop_percentage)
        self.pts2020 = self.getPointsPartial("0_WE1NZ71I.laz", 2020, remove_ground=True, percentage=crop_percentage)

        # Pre-compute everything expensive once, forwarding the downsample stride
        self.process_data(kd_threshold=1.5, icp_iterations=20, icp_tolerance=1e-4,
                          downsample_step=downsample_step)

        print("Datasets loaded. Use keys 1-4 to switch between tasks.")
        self.on_key_press(Key._1, 0)

    def process_data(self, kd_threshold: float = 1.5,
                     icp_iterations: int = 20,
                     icp_tolerance: float = 1e-4,
                     downsample_step: int = 15):
        """Run all expensive computations once and cache results as attributes.

        Parameters
        ----------
        kd_threshold : float
            Distance threshold for KD-tree scene classification.
        icp_iterations : int
            Maximum number of ICP alignment iterations.
        icp_tolerance : float
            Convergence tolerance for ICP (stops when MSE improvement < this).
        downsample_step : int
            Stride for thinning point clouds before ICP and KD-tree queries.
            A value of 15 keeps every 15th point; increase for faster (coarser)
            processing or decrease for denser (slower) processing.

        Cached attributes
        -----------------
        self.pts2016_ds / self.pts2020_ds  : downsampled raw clouds
        self.pts2016_aligned / self.pts2020_aligned : ICP+PCA aligned clouds
        self.global_center, self.global_size, self.max_dist : shared geometry
        self.mean_pca, self.R_pca          : PCA frame of pts2020
        self.static_world, self.added_world, self.removed_world : scaled world-space
        self.static_aligned                : aligned static points (for tree building)
        self.task1_static, self.task1_added, self.task1_removed : Task-1 (kd threshold=3)
        """
        print("Pre-processing: ICP alignment + scene classification (runs once)...")
        step = downsample_step          # parametric – set via __init__ or direct call
        pts16 = self.pts2016[::step]
        pts20 = self.pts2020[::step]

        # --- ICP alignment ---
        pts16_icp = self.icp_align(pts16, pts20,
                                   max_iterations=icp_iterations,
                                   tolerance=icp_tolerance)

        # --- PCA world frame from the fixed (2020) cloud ---
        mean, R = self._pca_transform(pts20)
        pts20_al = self._apply_transform(pts20, mean, R)
        pts16_al = self._apply_transform(pts16_icp, mean, R)

        all_pts = np.vstack((pts16_al, pts20_al))
        g_center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2.0
        g_size   = float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2.0)
        g_maxd   = np.percentile(np.linalg.norm(all_pts - g_center, axis=1), 99)

        # --- KD-tree scene classification (vectorised) ---
        static_al, new_al, removed_al = self.compareScenes(
            pts16_al, pts20_al, threshold=kd_threshold)

        # Store aligned clouds and shared geometry
        self.pts2016_ds       = pts16
        self.pts2020_ds       = pts20
        self.pts2016_aligned  = pts16_al
        self.pts2020_aligned  = pts20_al
        self.static_aligned   = static_al
        self.global_center    = g_center
        self.global_size      = g_size
        self.max_dist         = g_maxd
        self.mean_pca         = mean
        self.R_pca            = R

        # World-space point clouds for render_scene
        self.static_world  = self._inverse_transform(self.scale(static_al,    g_center, g_maxd), mean, R)
        self.added_world   = self._inverse_transform(self.scale(new_al,       g_center, g_maxd), mean, R)
        self.removed_world = self._inverse_transform(self.scale(removed_al,   g_center, g_maxd), mean, R)

        # Task 1: coarser kd-threshold=3; downsample_step forwarded so it is
        # also configurable (find_dynamic_objects defaults to step=40 when not
        # supplied, preserving the original coarser thinning for Task 1).
        self.task1_static, self.task1_added, self.task1_removed = \
            self.find_dynamic_objects(self.pts2016, self.pts2020, threshold=3)

        print("Pre-processing complete.")

    def setup(self):
        """Original lab setup placeholder."""
        # This was used in earlier versions of the lab
        pass

    # ================= DATA LOADING / PROCESSING =================
    def remove_ground_grid(self, pts: np.ndarray, grid_size: float = 5.0, z_threshold: float = 0.4) -> np.ndarray:
        """Grid-based Local Minimum Filter for robust ground removal.

        Divides the X-Y plane into a regular grid of cells of side ``grid_size``.
        For each cell the lowest Z value (local ground estimate) is found.  Any
        point that lies within ``z_threshold`` metres of that local minimum is
        considered ground and discarded.

        This handles slopes and uneven terrain correctly because the ground
        reference is *local* rather than a single global percentile.

        Parameters
        ----------
        pts : np.ndarray, shape (N, 3)
            Raw point cloud in original sensor coordinates (X, Y, Z).
        grid_size : float
            Side length of each grid cell in the same units as the point cloud.
            Smaller values give finer ground resolution but are slower.
        z_threshold : float
            Vertical clearance above the local ground minimum to keep a point.
            Points with Z <= local_min + z_threshold are removed as ground.

        Returns
        -------
        np.ndarray
            Filtered point cloud with ground points removed.
        """
        if len(pts) == 0:
            return pts
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # Discretise X and Y into integer cell indices
        x_idx = np.floor((x - x.min()) / grid_size).astype(np.int32)
        y_idx = np.floor((y - y.min()) / grid_size).astype(np.int32)

        # Build a flat cell key so we can use np.unique for grouping
        n_cols = int(x_idx.max()) + 1
        cell_key = y_idx * n_cols + x_idx  # unique integer per cell

        # For every point compute the minimum Z in its cell
        unique_keys, inverse = np.unique(cell_key, return_inverse=True)
        cell_min_z = np.zeros(len(unique_keys), dtype=np.float32)
        for k in range(len(unique_keys)):
            cell_min_z[k] = z[inverse == k].min()

        # Map back: each point gets its cell's minimum Z
        local_ground_z = cell_min_z[inverse]

        # Keep only points that are sufficiently above the local ground
        mask = z > (local_ground_z + z_threshold)
        return pts[mask]

    def getPoints(self, filename, year, remove_ground=True,
                  grid_size: float = 5.0, z_threshold: float = 0.4):
        """Load a point cloud, optionally removing ground via a grid-based filter.

        Parameters
        ----------
        grid_size : float
            Grid cell side length for the local minimum ground filter.
        z_threshold : float
            Vertical clearance above the local ground to retain a point.
        """
        pts = rl.readpoints(filename, year=year)
        pts = np.array(pts, dtype=np.float32)
        if remove_ground:
            pts = self.remove_ground_grid(pts, grid_size=grid_size, z_threshold=z_threshold)
        # Reorder to [X, Height, Y]
        pts = pts[:, [0, 2, 1]]
        return pts

    def getPointsPartial(self, filename, year, remove_ground, percentage,
                         grid_size: float = 5.0, z_threshold: float = 0.4):
        """Load a spatial subset of a point cloud with ground removed.

        Parameters
        ----------
        percentage : int
            X/Y percentile cut-off to keep only a spatial sub-region.
        grid_size : float
            Grid cell side length for the local minimum ground filter.
        z_threshold : float
            Vertical clearance above the local ground to retain a point.
        """
        pts = rl.readpoints(filename, year=year)
        pts = np.array(pts, dtype=np.float32)
        if remove_ground:
            pts = self.remove_ground_grid(pts, grid_size=grid_size, z_threshold=z_threshold)

        x_axis = np.percentile(pts[:, 0], percentage)
        pts = pts[pts[:, 0] < x_axis]
        y_axis = np.percentile(pts[:, 1], percentage)
        pts = pts[pts[:, 1] < y_axis]

        # Reorder to [X, Height, Y]
        pts = pts[:, [0, 2, 1]]
        return pts

    def centerAndNormalize(self, pts):
        """Centers points and scales them to fit within a [-10, 10] range."""
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        pts_centered = pts - center
        
        distances = np.linalg.norm(pts_centered, axis=1)
        max_dist = np.percentile(distances, 99)
        
        if max_dist > 0:
            pts_scaled = np.clip((pts_centered / max_dist) * 10.0, -10.0, 10.0)
        else:
            pts_scaled = pts_centered
        return pts_scaled

    # ================= MATH & TRANSFORMS =================
    def _pca_transform(self, points: np.ndarray):
        """Compute PCA rotation matrix constrained to the vertical axis (index 1).
        This ensures rotation only happens around the height axis.
        """
        mean = points.mean(axis=0)
        # Indices 0 and 2 are horizontal in our [X, Z(Height), Y] system
        horizontal_pts = points[:, [0, 2]] - mean[[0, 2]]
        cov = np.cov(horizontal_pts.T)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        R2d = evecs[:, idx]
        
        # Construct 3D rotation matrix that keeps vertical axis (index 1) fixed
        R = np.eye(3)
        R[0, 0] = R2d[0, 0]
        R[0, 2] = R2d[0, 1]
        R[2, 0] = R2d[1, 0]
        R[2, 2] = R2d[1, 1]
        return mean, R

    def _apply_transform(self, pts: np.ndarray, mean: np.ndarray, R: np.ndarray):
        """Rotate points into PCA-aligned space: (pts - mean) @ R"""
        return (pts - mean) @ R

    def _inverse_transform(self, pts: np.ndarray, mean: np.ndarray, R: np.ndarray):
        """Inverse of _apply_transform: rotate back and re‑add mean."""
        return pts @ R.T

    # ---- fast numpy-only nearest-neighbour (no Point3D overhead) ----
    @staticmethod
    def _nn_numpy(query: np.ndarray, cloud: np.ndarray) -> np.ndarray:
        """For each row in query find the index of its nearest row in cloud.
        Fully vectorised; no external libraries."""
        # Split into batches to avoid huge memory allocation
        batch = 2000
        indices = np.empty(len(query), dtype=np.int64)
        for start in range(0, len(query), batch):
            q = query[start:start + batch]          # (B, D)
            # squared distances (B, N)
            diff = q[:, np.newaxis, :] - cloud[np.newaxis, :, :]
            d2 = (diff * diff).sum(axis=2)
            indices[start:start + batch] = d2.argmin(axis=1)
        return indices

    def icp_align(self, src: np.ndarray, tgt: np.ndarray,
                  max_iterations: int = 30,
                  tolerance: float = 1e-4,
                  subsample: int = 2000) -> np.ndarray:
        """Iterative Closest Point – aligns *src* onto *tgt*.

        Uses the combined-cloud PCA as a robust first-guess warm-start, then
        refines with an SVD-based point-to-point ICP loop.  Nearest-neighbour
        queries use the pure-numpy batched _nn_numpy helper.

        Parameters
        ----------
        src : np.ndarray (N, 3)   – cloud to move (e.g. pts2016).
        tgt : np.ndarray (M, 3)   – fixed reference cloud (e.g. pts2020 static).
        max_iterations : int      – ICP iteration cap.
        tolerance : float         – convergence: stop when MSE improvement
                                    drops below this value.
        subsample : int           – number of src points sampled per iteration.

        Returns
        -------
        np.ndarray (N, 3)  – src transformed to best align with tgt.
        """
        # --- warm-start: PCA on COMBINED cloud so both are in the same frame ---
        # Using the combined cloud avoids the ±eigenvector flip that occurs when
        # each cloud is PCA-aligned independently.
        combined = np.vstack((src, tgt))
        mean_c, R_c = self._pca_transform(combined)
        src_work = self._apply_transform(src, mean_c, R_c)
        tgt_work = self._apply_transform(tgt, mean_c, R_c)

        prev_err = np.inf
        for it in range(max_iterations):
            # subsample src for speed
            if len(src_work) > subsample:
                idx = np.random.choice(len(src_work), subsample, replace=False)
                src_sub = src_work[idx]
            else:
                src_sub = src_work

            # find nearest neighbour in tgt for every src point
            nn_idx = self._nn_numpy(src_sub, tgt_work)
            matched_tgt = tgt_work[nn_idx]

            # compute mean squared error
            mse = float(np.mean(np.sum((src_sub - matched_tgt) ** 2, axis=1)))

            # SVD-based optimal rotation + translation
            c_src = src_sub.mean(axis=0)
            c_tgt = matched_tgt.mean(axis=0)
            H = (src_sub - c_src).T @ (matched_tgt - c_tgt)
            U, _, Vt = np.linalg.svd(H)
            R_icp = Vt.T @ U.T
            # fix reflection
            if np.linalg.det(R_icp) < 0:
                Vt[-1, :] *= -1
                R_icp = Vt.T @ U.T
            t_icp = c_tgt - R_icp @ c_src

            # apply to the FULL src_work
            src_work = (R_icp @ src_work.T).T + t_icp

            improvement = prev_err - mse
            print(f"  ICP iter {it+1}: MSE={mse:.6f}, Δ={improvement:.6f}")
            if improvement < tolerance and it > 0:
                print(f"  ICP converged at iteration {it+1}.")
                break
            prev_err = mse

        return src_work

    def scale(self, pts, center, max_dist):
        """Consolidated scaling helper."""
        if len(pts) == 0:
            return pts          # preserve shape (0, D) so matmul stays valid
        if max_dist > 0:
            return ((pts - center) / max_dist) * 10.0
        return pts - center

    # ================= CLASSIFICATION LOGIC =================
    def compareScenes(self, pts2016, pts2020, threshold=1.5, downsample=False):
        """Vectorised KD-tree based classification using _nn_numpy.
        Returns (static_pts, new_pts, removed_pts) in input space."""
        if downsample:
            pts2016 = pts2016[::40]
            pts2020 = pts2020[::40]

        threshold_sq = threshold ** 2

        # Pass 1: Find new and static points (2020 vs 2016)
        print("Classifying points (vectorised NN)...")
        nn_idx_fwd = self._nn_numpy(pts2020, pts2016)      # each 2020 pt -> nearest 2016 pt
        d_sq_fwd = np.sum((pts2020 - pts2016[nn_idx_fwd]) ** 2, axis=1)
        is_new = d_sq_fwd > threshold_sq
        new_pts = pts2020[is_new]
        static_pts = pts2020[~is_new]

        # Pass 2: Find removed points (2016 vs 2020)
        nn_idx_rev = self._nn_numpy(pts2016, pts2020)      # each 2016 pt -> nearest 2020 pt
        d_sq_rev = np.sum((pts2016 - pts2020[nn_idx_rev]) ** 2, axis=1)
        removed_pts = pts2016[d_sq_rev > threshold_sq]

        print(f"Classification result: {len(new_pts)} added, {len(removed_pts)} removed, {len(static_pts)} static.")
        return static_pts, new_pts, removed_pts

    def _node_geometry_signature(self, points: np.ndarray) -> np.ndarray:
        """Compute the PCA eigenvalue signature of a point set.

        Returns the three sorted eigenvalues of the covariance matrix,
        normalised so their sum equals 1.  These values compactly encode
        the local shape: a flat plane has one dominant eigenvalue near zero
        while a diffuse cloud has three roughly equal values.

        Parameters
        ----------
        points : np.ndarray, shape (N, D)  – works for both 2-D and 3-D nodes.

        Returns
        -------
        np.ndarray, shape (D,)
            Normalised eigenvalues sorted in descending order.
        """
        if len(points) < 2:
            return np.zeros(points.shape[1])
        cov = np.cov(points.T)
        if cov.ndim == 0:            # scalar fallback for 1-D degenerate case
            return np.array([1.0])
        evals = np.linalg.eigvalsh(cov)
        evals = np.sort(np.abs(evals))[::-1]   # descending
        total = evals.sum()
        return evals / total if total > 0 else evals

    def _hausdorff_distance_approx(self, pts_a: np.ndarray, pts_b: np.ndarray,
                                   sample: int = 200) -> float:
        """Approximate one-sided Hausdorff distance from pts_a to pts_b.

        For each (sampled) point in ``pts_a`` the nearest neighbour in
        ``pts_b`` is found with a simple vectorised nearest-neighbour search
        (O(n·m) in the sample).  Returns the maximum of those nearest-
        neighbour distances, which is the directed Hausdorff distance.

        Only numpy is used; no heavy external libraries.

        Parameters
        ----------
        pts_a, pts_b : np.ndarray, shape (N, D) / (M, D)
        sample : int
            Maximum number of points drawn from each set for the check.
            Reduces cost when point sets are large.

        Returns
        -------
        float  – directed Hausdorff distance (max over pts_a of min-dist-to-pts_b).
        """
        if len(pts_a) == 0 or len(pts_b) == 0:
            return float('inf')
        # Sub-sample if necessary
        if len(pts_a) > sample:
            idx = np.random.choice(len(pts_a), sample, replace=False)
            pts_a = pts_a[idx]
        if len(pts_b) > sample:
            idx = np.random.choice(len(pts_b), sample, replace=False)
            pts_b = pts_b[idx]
        # Vectorised pairwise distances: shape (|pts_a|, |pts_b|)
        diff = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]   # (N, M, D)
        dists = np.sqrt((diff ** 2).sum(axis=-1))                   # (N, M)
        return float(dists.min(axis=1).max())

    def compare_trees(self, node_a, node_b, threshold=1,
                      geo_threshold: float = 0.25,
                      hausdorff_threshold: float = 0.5,
                      min_pts: int = 5):
        """Generic tree comparison for Octree and Quadtree.

        The *changed* condition now uses **true geometric comparison**:

        1. **PCA eigenvalue signature** – encodes whether the local point
           distribution is planar, linear or volumetric.  If the L1 distance
           between the two normalised eigenvalue vectors exceeds
           ``geo_threshold``, the structural shape has changed.
        2. **Approximate Hausdorff distance** – measures how far the two
           point sets are from each other geometrically.  If the directed
           distance exceeds ``hausdorff_threshold``, the geometry has shifted.

        Either condition alone is sufficient to flag a node as *changed*.
        Pure appearance / disappearance is still handled by the point-count
        zero checks (added / removed).

        Parameters
        ----------
        threshold : int  (legacy, kept for API compatibility)
            Minimum point count that must be present in a non-empty leaf
            before it is eligible for geometric comparison.
        geo_threshold : float
            Maximum tolerated L1 distance between normalised PCA eigenvalue
            signatures.  Values in [0, 1]; a good default is 0.25.
        hausdorff_threshold : float
            Maximum tolerated directed Hausdorff distance (in the same units
            as the point coordinates) before flagging as changed.
        min_pts : int
            Minimum number of points that must be present in *both* nodes
            to attempt a geometric comparison.  Nodes below this count are
            treated as empty for classification purposes.
        """
        changes = []
        if node_a.is_leaf and node_b.is_leaf:
            pts_a = node_a.points if node_a.points is not None else np.empty((0, 1))
            pts_b = node_b.points if node_b.points is not None else np.empty((0, 1))
            len_a, len_b = len(pts_a), len(pts_b)

            if len_a >= min_pts and len_b == 0:
                changes.append((node_a, 'removed'))
            elif len_a == 0 and len_b >= min_pts:
                changes.append((node_b, 'added'))
            elif len_a >= min_pts and len_b >= min_pts:
                # --- Geometric comparison ---
                # 1. PCA eigenvalue signature distance
                sig_a = self._node_geometry_signature(pts_a)
                sig_b = self._node_geometry_signature(pts_b)
                # Pad to the same length in case dims differ (2-D vs 3-D nodes)
                n = max(len(sig_a), len(sig_b))
                sig_a = np.pad(sig_a, (0, n - len(sig_a)))
                sig_b = np.pad(sig_b, (0, n - len(sig_b)))
                geo_dist = float(np.abs(sig_a - sig_b).sum())

                # 2. Approximate directed Hausdorff distance
                h_dist = self._hausdorff_distance_approx(pts_a, pts_b)

                if geo_dist > geo_threshold or h_dist > hausdorff_threshold:
                    changes.append((node_b, 'changed'))
        elif node_a.is_leaf:
            changes.append((node_b, 'added'))
        elif node_b.is_leaf:
            changes.append((node_a, 'removed'))
        else:
            for child_a, child_b in zip(node_a.children, node_b.children):
                changes.extend(self.compare_trees(
                    child_a, child_b,
                    threshold=threshold,
                    geo_threshold=geo_threshold,
                    hausdorff_threshold=hausdorff_threshold,
                    min_pts=min_pts,
                ))
        return changes

    def find_dynamic_objects(self, pts_2016, pts_2020, threshold=0.5,
                             downsample_step: int = 40):
        """Lab Task 4 logic: vectorised nearest-neighbour comparison.

        Parameters
        ----------
        pts_2016, pts_2020 : np.ndarray (N, 3)
            Full (undownsampled) point clouds for the two epochs.
        threshold : float
            Distance threshold: points farther than this from their nearest
            neighbour in the opposite epoch are classified as dynamic.
        downsample_step : int
            Stride for thinning the clouds before comparison (default 40).
            Decrease for denser results at the cost of higher compute time.
        """
        print("Finding dynamic objects (vectorised NN)...")
        pts_2016 = pts_2016[::downsample_step]
        pts_2020 = pts_2020[::downsample_step]

        # Vectorised: each 2016 point -> nearest 2020 point
        nn_idx_16 = self._nn_numpy(pts_2016, pts_2020)
        d_16 = np.linalg.norm(pts_2016 - pts_2020[nn_idx_16], axis=1)
        dynamic_removed = pts_2016[d_16 > threshold]
        static_map = pts_2016[d_16 <= threshold]

        # Vectorised: each 2020 point -> nearest 2016 point
        nn_idx_20 = self._nn_numpy(pts_2020, pts_2016)
        d_20 = np.linalg.norm(pts_2020 - pts_2016[nn_idx_20], axis=1)
        dynamic_added = pts_2020[d_20 > threshold]

        # Centering and normalization for this specific task
        all_pts = np.vstack((pts_2016, pts_2020))
        center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2.0
        max_dist = np.percentile(np.linalg.norm(all_pts - center, axis=1), 99)

        def local_scale(p):
            if len(p) == 0: return np.array([])
            return np.clip(((p - center) / max_dist) * 10.0, -10.0, 10.0) if max_dist > 0 else p - center

        return local_scale(static_map), local_scale(dynamic_added), local_scale(dynamic_removed)

    # ================= VISUALIZERS & LAB TASKS =================
    def compareScenesOctTree(self,
                             max_depth: int = None,
                             capacity:  int = None):
        """Detect volumetric changes using an Octree (uses cached ICP-aligned data).

        Parameters
        ----------
        max_depth : int, optional
            Maximum recursion depth of each OctNode tree.
            Defaults to the class attribute ``OCT_MAX_DEPTH`` (currently 6).
        capacity : int, optional
            Maximum number of points per leaf before the node is split.
            Defaults to the class attribute ``OCT_CAPACITY`` (currently 50).
        """
        # Fall back to class-level defaults when not explicitly supplied
        if max_depth is None:
            max_depth = self.OCT_MAX_DEPTH
        if capacity is None:
            capacity = self.OCT_CAPACITY

        mean, R = self.mean_pca, self.R_pca
        g_center, g_size, g_maxd = self.global_center, self.global_size, self.max_dist

        print(f"Building OctTrees (max_depth={max_depth}, capacity={capacity})...")
        tree_2016 = octN.OctNode(self.pts2016_aligned, g_center, g_size, depth=0, max_depth=max_depth, capacity=capacity)
        tree_2020 = octN.OctNode(self.pts2020_aligned, g_center, g_size, depth=0, max_depth=max_depth, capacity=capacity)

        changed_nodes = self.compare_trees(tree_2016, tree_2020, threshold=1)
        boxes_world = []
        for node, status in changed_nodes:
            p1_aligned = node.center - node.size
            p2_aligned = node.center + node.size
            boxes_world.append({
                'p1_aligned': p1_aligned,
                'p2_aligned': p2_aligned,
                'status': status,
                'mean': mean, 'R': R,
                'max_dist': g_maxd,
                'global_center': g_center
            })

        self.render_scene(self.static_world, self.added_world, self.removed_world,
                          boxes_world, box_type='generalized')

    def compareScenesQuadTree(self,
                              max_depth: int = None,
                              capacity:  int = None):
        """Top-down change detection using 2D QuadTrees (uses cached ICP-aligned data).

        Parameters
        ----------
        max_depth : int, optional
            Maximum recursion depth of each QuadNode tree.
            Defaults to the class attribute ``QUAD_MAX_DEPTH`` (currently 7).
        capacity : int, optional
            Maximum number of points per leaf before the node is split.
            Defaults to the class attribute ``QUAD_CAPACITY`` (currently 40).
        """
        # Fall back to class-level defaults when not explicitly supplied
        if max_depth is None:
            max_depth = self.QUAD_MAX_DEPTH
        if capacity is None:
            capacity = self.QUAD_CAPACITY

        mean, R = self.mean_pca, self.R_pca
        g_center, g_size, g_maxd = self.global_center, self.global_size, self.max_dist

        all_min = np.vstack((self.pts2016_aligned, self.pts2020_aligned)).min(axis=0)
        all_max = np.vstack((self.pts2016_aligned, self.pts2020_aligned)).max(axis=0)
        h_min, h_max = float(all_min[1]), float(all_max[1])

        print(f"Building QuadTrees (max_depth={max_depth}, capacity={capacity})...")
        pts16_2d = self.pts2016_aligned[:, [0, 2]]
        pts20_2d = self.pts2020_aligned[:, [0, 2]]
        center_2d = g_center[[0, 2]]
        tree_2016 = quadN.QuadNode(pts16_2d, center_2d, g_size, depth=0, max_depth=max_depth, capacity=capacity)
        tree_2020 = quadN.QuadNode(pts20_2d, center_2d, g_size, depth=0, max_depth=max_depth, capacity=capacity)

        changed_nodes = self.compare_trees(tree_2016, tree_2020, threshold=5)
        boxes_world = []
        for node, status in changed_nodes:
            cx, cy = node.center
            sz = node.size
            p1_aligned = np.array([cx - sz, h_min, cy - sz])
            p2_aligned = np.array([cx + sz, h_max, cy + sz])
            boxes_world.append({
                'p1_aligned': p1_aligned,
                'p2_aligned': p2_aligned,
                'status': status,
                'mean': mean, 'R': R,
                'max_dist': g_maxd,
                'global_center': g_center
            })

        self.render_scene(self.static_world, self.added_world, self.removed_world,
                          boxes_world, box_type='generalized')

    def printSceneDifferences(self,
                              max_depth: int = None,
                              capacity:  int = None):
        """Renders change detection as colored points from Octree nodes (uses cache).

        Parameters
        ----------
        max_depth : int, optional
            Forwarded to OctNode; defaults to ``OCT_MAX_DEPTH``.
        capacity : int, optional
            Forwarded to OctNode; defaults to ``OCT_CAPACITY``.
        """
        if max_depth is None:
            max_depth = self.OCT_MAX_DEPTH
        if capacity is None:
            capacity = self.OCT_CAPACITY

        mean, R = self.mean_pca, self.R_pca
        g_center, g_size, g_maxd = self.global_center, self.global_size, self.max_dist

        tree_2016 = octN.OctNode(self.pts2016_aligned, g_center, g_size, depth=0, max_depth=max_depth, capacity=capacity)
        tree_2020 = octN.OctNode(self.pts2020_aligned, g_center, g_size, depth=0, max_depth=max_depth, capacity=capacity)

        changed_nodes = self.compare_trees(tree_2016, tree_2020, threshold=1)
        added_list, removed_list = [], []

        for node, status in changed_nodes:
            if node.points is None or len(node.points) == 0:
                continue
            node_pts_world = self._inverse_transform(
                self.scale(node.points, g_center, g_maxd), mean, R)
            if status == 'added':
                added_list.append(node_pts_world)
            elif status == 'removed':
                removed_list.append(node_pts_world)

        added_world   = np.vstack(added_list)   if added_list   else np.array([])
        removed_world = np.vstack(removed_list) if removed_list else np.array([])

        self.render_scene(self.static_world, added_world, removed_world)

    # ================= CLUSTERING & CATEGORIZATION =================
    def cluster_and_categorize(self, pts: np.ndarray,
                               eps: float = 0.8,
                               min_points: int = 15,
                               min_building_area: float = 8.0,
                               min_building_height: float = 1.0,
                               min_tree_height: float = 3.0,
                               max_tree_area: float = 6.0,
                               max_pole_width: float = 0.5,
                               min_pole_height: float = 1.5) -> list:
        """Euclidean cluster extraction (DBSCAN-style) with geometric categorization.

        Uses a pure-numpy radius search to group points into connected components
        (no external libraries).  Each cluster is then categorized by a simple
        bounding-box heuristic:

        * **building**  – footprint area >= min_building_area AND height >= min_building_height
        * **tree**      – height >= min_tree_height AND footprint area < max_tree_area
        * **pole/mast** – both footprint dims < max_pole_width AND height >= min_pole_height
        * **object**    – everything else

        Parameters
        ----------
        pts : np.ndarray (N, 3)  – points in visualizer space [X, Height, Y].
        eps : float              – neighbourhood radius for cluster growth.
        min_points : int         – minimum cluster size (noise gate).
        min_building_area : float  – footprint area threshold to classify as building.
        min_building_height : float – minimum height to classify as building.
        min_tree_height : float    – minimum vertical extent to classify as tree.
        max_tree_area : float      – maximum footprint area to still classify as tree.
        max_pole_width : float     – maximum X and Y extent to classify as pole.
        min_pole_height : float    – minimum height to classify as pole.

        Returns
        -------
        list of (np.ndarray, str)  – [(cluster_points, category_label), ...]
        """
        if len(pts) == 0:
            return []

        n = len(pts)
        labels = -np.ones(n, dtype=np.int32)   # -1 = unvisited
        cluster_id = 0

        # --- voxel-grid spatial index for fast radius queries ---
        # Map each point to an integer cell (cx, cy, cz)
        cell_size = eps  # one voxel = one eps-radius
        coords = np.floor(pts / cell_size).astype(np.int64)

        # Build dict: cell_key -> list of point indices
        from collections import defaultdict
        grid: dict = defaultdict(list)
        for i, c in enumerate(coords):
            grid[(c[0], c[1], c[2])].append(i)

        def region_query(idx: int) -> np.ndarray:
            """Return indices of all points within eps of pts[idx] using voxel grid."""
            cx, cy, cz = coords[idx]
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        candidates.extend(grid.get((cx+dx, cy+dy, cz+dz), []))
            if not candidates:
                return np.array([idx], dtype=np.int64)
            cand = np.array(candidates, dtype=np.int64)
            diff = pts[cand] - pts[idx]
            d2 = (diff * diff).sum(axis=1)
            return cand[d2 <= eps * eps]

        for i in range(n):
            if labels[i] != -1:
                continue
            neighbours = region_query(i)
            if len(neighbours) < min_points:
                labels[i] = -2   # noise
                continue
            # Start a new cluster
            labels[i] = cluster_id
            seed_set = list(neighbours)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                if labels[q] == -2:        # previously marked noise → border point
                    labels[q] = cluster_id
                if labels[q] != -1:        # already assigned
                    j += 1
                    continue
                labels[q] = cluster_id
                q_neighbours = region_query(q)
                if len(q_neighbours) >= min_points:
                    seed_set.extend(q_neighbours.tolist())
                j += 1
            cluster_id += 1

        # --- categorize each cluster ---
        results = []
        for cid in range(cluster_id):
            mask = labels == cid
            cluster_pts = pts[mask]
            if len(cluster_pts) < min_points:
                continue

            bb_min = cluster_pts.min(axis=0)   # [X, H, Y]
            bb_max = cluster_pts.max(axis=0)
            extent = bb_max - bb_min           # [dX, dH, dY]

            height = float(extent[1])          # vertical extent
            footprint_x = float(extent[0])
            footprint_y = float(extent[2])
            footprint_area = footprint_x * footprint_y

            if footprint_area >= min_building_area and height >= min_building_height:
                category = 'building'
            elif height >= min_tree_height and footprint_area < max_tree_area:
                category = 'tree'
            elif footprint_x < max_pole_width and footprint_y < max_pole_width and height >= min_pole_height:
                category = 'pole'
            else:
                category = 'object'

            results.append((cluster_pts, category))

        print(f"Clustering: {cluster_id} clusters found "
              f"({sum(1 for _,c in results if c=='building')} buildings, "
              f"{sum(1 for _,c in results if c=='tree')} trees, "
              f"{sum(1 for _,c in results if c=='pole')} poles, "
              f"{sum(1 for _,c in results if c=='object')} objects)")
        return results

    def colorize_clusters(self, clusters: list):
        """Add each cluster to the scene with a category-specific color.

        Category color mapping:
          building → CYAN,  tree → GREEN,  pole → YELLOW,  object → ORANGE
        Noise points (unclustered) are drawn in dim white at half size.
        """
        CAT_COLOR = {
            'building': (0.0, 0.9, 1.0, 1.0),    # cyan
            'tree':     (0.1, 0.9, 0.1, 1.0),    # green
            'pole':     (1.0, 1.0, 0.0, 1.0),    # yellow
            'object':   (1.0, 0.5, 0.0, 1.0),    # orange
        }
        for i, (pts, category) in enumerate(clusters):
            color = CAT_COLOR.get(category, (1.0, 1.0, 1.0, 1.0))
            self.addShape(
                PointSet3D(pts, size=2.5, color=color),
                name=f"cluster_{i}_{category}"
            )

    def displayTask4(self, static, added, removed):
        """Displays results for Task 4 with geometric cluster coloring."""
        if len(static) > 0:
            print("Clustering static points...")
            clusters = self.cluster_and_categorize(static, eps=0.8, min_points=15)
            if clusters:
                self.colorize_clusters(clusters)
            else:
                self.addShape(PointSet3D(static, size=1.0, color=Color.WHITE), name="static")
        if len(added) > 0:
            self.addShape(PointSet3D(added, size=3.0, color=Color.GREEN), name="added")
        if len(removed) > 0:
            self.addShape(PointSet3D(removed, size=3.0, color=Color.RED), name="removed")

    # ================= MASTER RENDERER =================
    def clear_scene(self):
        """Clears all shapes from the scene."""
        for name in list(self._shapeDict.keys()):
            self.removeShape(name)

    def on_key_press(self, symbol, modifiers):
        """Triggers different lab tasks dynamically."""
        if symbol in [Key._1, Key.NUM_1]:
            self.clear_scene()
            print("Rendering Task 4: KD-Tree Dynamic Objects...")
            self.displayTask4(self.task1_static, self.task1_added, self.task1_removed)
        elif symbol in [Key._2, Key.NUM_2]:
            self.clear_scene()
            print("Rendering Scene Differences...")
            self.printSceneDifferences()
        elif symbol in [Key._3, Key.NUM_3]:
            self.clear_scene()
            print("Rendering OctTree Comparison...")
            self.compareScenesOctTree()
        elif symbol in [Key._4, Key.NUM_4]:
            self.clear_scene()
            print("Rendering QuadTree Comparison...")
            self.compareScenesQuadTree()

    def render_scene(self, static_world, added_world, removed_world, boxes_world=None, box_type='cuboid'):
        """Master visualization method."""
        # 1. Static Points (Height-colored)
        if len(static_world) > 0:
            ps_static = PointSet3D(static_world, size=1)
            ps_static.colors = self.colorPointsByHeight(static_world).tolist()
            self.addShape(ps_static, name="Static Points")

        # 2. Added Points (Red)
        if len(added_world) > 0:
            ps_added = PointSet3D(added_world, size=3, color=(1.0, 0.0, 0.0, 1.0))
            self.addShape(ps_added, name="Added Points")

        # 3. Removed Points (Blue)
        if len(removed_world) > 0:
            ps_removed = PointSet3D(removed_world, size=3, color=(0.0, 0.0, 1.0, 1.0))
            self.addShape(ps_removed, name="Removed Points")

        # 4. Changed Area Boxes
        if boxes_world:
            for box in boxes_world:
                status = box['status']
                if status == 'removed': color = (0.0, 0.0, 1.0) 
                elif status == 'added': color = (1.0, 0.0, 0.0) 
                else: color = (1.0, 0.5, 0.0) 
                
                if box_type == 'cuboid':
                    self.addShape(Cuboid3D(box['p1'], box['p2'], color=color, width=2, filled=False))
                elif box_type == 'generalized':
                    p1, p2 = box['p1_aligned'], box['p2_aligned']
                    # Create 8 vertices in PCA space
                    v = np.array([[p1[0], p1[1], p1[2]], [p2[0], p1[1], p1[2]], [p2[0], p2[1], p1[2]], [p1[0], p2[1], p1[2]],
                                  [p1[0], p1[1], p2[2]], [p2[0], p1[1], p2[2]], [p2[0], p2[1], p2[2]], [p1[0], p2[1], p2[2]]])
                    # Transform to world space
                    v_scaled = self.scale(v, box['global_center'], box['max_dist'])
                    v_world = self._inverse_transform(v_scaled, box['mean'], box['R'])
                    # Initialize generalized cuboid with dummy and override vertices
                    gc = Cuboid3DGeneralized(Cuboid3D([0,0,0], [1,1,1], color=color, filled=False))
                    gc._vertices = v_world
                    self.addShape(gc)
                elif box_type == 'line':
                    corners = box['corners']
                    for i in range(4):
                        self.addShape(Line3D(corners[i], corners[(i+1)%4], color=color, width=2))

    def colorPointsByHeight(self, pts):
        """Generates color mapping based on Y-axis (height)."""
        height_vals = pts[:, 1]
        h_min, h_max = float(height_vals.min()), float(height_vals.max())
        znorm = ((height_vals - h_min) / (h_max - h_min)).astype(np.float32) if h_max != h_min else np.zeros_like(height_vals)
        return np.column_stack((znorm, 0.2 * np.ones_like(znorm), 1.0 - znorm, np.ones_like(znorm)))

if __name__ == "__main__":
    app = DynamicScenes()
    app.mainLoop()