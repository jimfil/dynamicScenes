import numpy as np

def pca_transform(points: np.ndarray):
    """Compute PCA rotation matrix constrained to the vertical axis (index 1).
    This ensures rotation only happens around the height axis.
    """
    if len(points) == 0:
        return np.zeros(3), np.eye(3)
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

def apply_transform(pts: np.ndarray, mean: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate points into PCA-aligned space: (pts - mean) @ R"""
    if len(pts) == 0:
        return pts
    return (pts - mean) @ R

def inverse_transform(pts: np.ndarray, mean: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Inverse of apply_transform: rotate back and re‑add mean."""
    if len(pts) == 0:
        return pts
    return pts @ R.T + mean

def nn_numpy(query: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """For each row in query find the index of its nearest row in cloud.
    Uses scipy.spatial.cKDTree if available for massive speedup,
    otherwise falls back to the batch vectorised pure-numpy approach."""
    if len(cloud) == 0:
        # Return zeros if cloud is empty to avoid index out of bounds/value error
        return np.zeros(len(query), dtype=np.int64)
    if len(query) == 0:
        return np.empty(0, dtype=np.int64)
        
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(cloud)
        _, indices = tree.query(query, k=1, workers=-1)
        return indices
    except ImportError:
        # Fallback to batch vectorised pure-numpy approach
        batch = 2000
        indices = np.empty(len(query), dtype=np.int64)
        for start in range(0, len(query), batch):
            q = query[start:start + batch]          # (B, D)
            diff = q[:, np.newaxis, :] - cloud[np.newaxis, :, :]
            d2 = (diff * diff).sum(axis=2)
            indices[start:start + batch] = d2.argmin(axis=1)
        return indices

def icp_align(src: np.ndarray, tgt: np.ndarray,
              max_iterations: int = 30,
              tolerance: float = 1e-4,
              subsample: int = 2000) -> np.ndarray:
    """Iterative Closest Point – aligns *src* onto *tgt*.
    Uses the combined-cloud PCA as a robust first-guess warm-start, then
    refines with an SVD-based point-to-point ICP loop.
    """
    if len(src) == 0 or len(tgt) == 0:
        return src

    # --- warm-start: PCA on COMBINED cloud so both are in the same frame ---
    combined = np.vstack((src, tgt))
    mean_c, R_c = pca_transform(combined)
    src_work = apply_transform(src, mean_c, R_c)
    tgt_work = apply_transform(tgt, mean_c, R_c)

    prev_err = np.inf
    for it in range(max_iterations):
        if len(src_work) > subsample:
            idx = np.random.choice(len(src_work), subsample, replace=False)
            src_sub = src_work[idx]
        else:
            src_sub = src_work

        # find nearest neighbour in tgt for every src point
        nn_idx = nn_numpy(src_sub, tgt_work)
        matched_tgt = tgt_work[nn_idx]

        # compute mean squared error
        mse = float(np.mean(np.sum((src_sub - matched_tgt) ** 2, axis=1)))

        # SVD-based optimal rotation + translation
        c_src = src_sub.mean(axis=0)
        c_tgt = matched_tgt.mean(axis=0)
        H = (src_sub - c_src).T @ (matched_tgt - c_tgt)
        U, S, Vt = np.linalg.svd(H)
        R_icp = Vt.T @ U.T
        # fix reflection
        if np.linalg.det(R_icp) < 0:
            Vt[-1, :] *= -1
            R_icp = Vt.T @ U.T
        t_icp = c_tgt - R_icp @ c_src

        # apply to the FULL src_work
        src_work = (R_icp @ src_work.T).T + t_icp

        improvement = prev_err - mse
        print(f"  ICP iter {it+1}: MSE={mse:.6f}, Delta={improvement:.6f}")
        if improvement < tolerance and it > 0:
            print(f"  ICP converged at iteration {it+1}.")
            break
        prev_err = mse

    # Apply inverse PCA transform to return back to original frame
    return inverse_transform(src_work, mean_c, R_c)

def scale(pts: np.ndarray, center: np.ndarray, max_dist: float) -> np.ndarray:
    """Consolidated scaling helper."""
    if len(pts) == 0:
        return pts
    if max_dist > 0:
        return ((pts - center) / max_dist) * 10.0
    return pts - center

def node_geometry_signature(points: np.ndarray) -> np.ndarray:
    """Compute the PCA eigenvalue signature of a point set."""
    if len(points) < 2:
        return np.zeros(points.shape[1])
    cov = np.cov(points.T)
    if cov.ndim == 0:            # scalar fallback for 1-D degenerate case
        return np.array([1.0])
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(np.abs(evals))[::-1]   # descending
    total = evals.sum()
    return evals / total if total > 0 else evals

def hausdorff_distance_approx(pts_a: np.ndarray, pts_b: np.ndarray,
                              sample: int = 200) -> float:
    """Approximate one-sided Hausdorff distance from pts_a to pts_b."""
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

def compare_trees(node_a, node_b, threshold=1,
                  geo_threshold: float = 0.25,
                  hausdorff_threshold: float = 0.5,
                  min_pts: int = 5):
    """Generic tree comparison for Octree and Quadtree."""
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
            sig_a = node_geometry_signature(pts_a)
            sig_b = node_geometry_signature(pts_b)
            # Pad to the same length in case dims differ (2-D vs 3-D nodes)
            n = max(len(sig_a), len(sig_b))
            sig_a = np.pad(sig_a, (0, n - len(sig_a)))
            sig_b = np.pad(sig_b, (0, n - len(sig_b)))
            geo_dist = float(np.abs(sig_a - sig_b).sum())

            # 2. Approximate directed Hausdorff distance
            h_dist = hausdorff_distance_approx(pts_a, pts_b)

            if geo_dist > geo_threshold or h_dist > hausdorff_threshold:
                changes.append((node_b, 'changed'))
    elif node_a.is_leaf:
        changes.append((node_b, 'added'))
    elif node_b.is_leaf:
        changes.append((node_a, 'removed'))
    else:
        for child_a, child_b in zip(node_a.children, node_b.children):
            changes.extend(compare_trees(
                child_a, child_b,
                threshold=threshold,
                geo_threshold=geo_threshold,
                hausdorff_threshold=hausdorff_threshold,
                min_pts=min_pts,
            ))
    return changes

def color_points_by_height(pts: np.ndarray) -> np.ndarray:
    """Generates color mapping based on Y-axis (height)."""
    if len(pts) == 0:
        return np.empty((0, 4), dtype=np.float32)
    height_vals = pts[:, 1]
    h_min, h_max = float(height_vals.min()), float(height_vals.max())
    znorm = ((height_vals - h_min) / (h_max - h_min)).astype(np.float32) if h_max != h_min else np.zeros_like(height_vals)
    return np.column_stack((znorm, 0.2 * np.ones_like(znorm), 1.0 - znorm, np.ones_like(znorm)))

