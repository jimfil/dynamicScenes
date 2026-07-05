import numpy as np
from collections import defaultdict
from src.geometry_utils import nn_numpy

def remove_reflections(pts: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """Filter out self-reflections (ego-vehicle points close to the sensor origin)."""
    if len(pts) == 0:
        return pts
    local_dist = np.linalg.norm(pts, axis=1)
    return pts[local_dist > threshold]

def remove_ground_grid(pts: np.ndarray, grid_size: float = 5.0, z_threshold: float = 0.4) -> np.ndarray:
    """Grid-based Local Minimum/Percentile Filter for robust ground removal.
    Divides the X-Y plane into a regular grid of cells of side ``grid_size``.
    For each cell, the local ground Z is estimated. If a cell contains no ground
    (local minimum is significantly higher than global ground level), the local
    ground is set to the global ground to prevent building facade slicing.
    
    Optimised using contiguous array sorting for O(1) cell slicing.
    """
    if len(pts) == 0:
        return pts
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Compute dynamic global ground level (2nd percentile to ignore noise)
    global_ground_z = np.percentile(z, 2.0)

    # Discretise X and Y into integer cell indices
    x_idx = np.floor((x - x.min()) / grid_size).astype(np.int32)
    y_idx = np.floor((y - y.min()) / grid_size).astype(np.int32)

    # Build a flat cell key
    n_cols = int(x_idx.max()) + 1
    cell_key = y_idx * n_cols + x_idx

    # Sort indices by cell key to make cells contiguous in memory
    sort_idx = np.argsort(cell_key)
    sorted_cell_key = cell_key[sort_idx]
    sorted_z = z[sort_idx]

    # Find starting boundaries of unique cell groups
    unique_keys, first_indices = np.unique(sorted_cell_key, return_index=True)
    boundaries = np.append(first_indices, len(sorted_z))
    
    n_cells = len(unique_keys)
    cell_min_z = np.zeros(n_cells, dtype=np.float32)

    for k in range(n_cells):
        start = boundaries[k]
        end = boundaries[k+1]
        cell_z = sorted_z[start:end]
        
        # 5th percentile Z in this cell
        local_min = np.percentile(cell_z, 5.0) if len(cell_z) > 0 else global_ground_z
        if local_min > global_ground_z + 1.5:
            cell_min_z[k] = global_ground_z
        else:
            cell_min_z[k] = local_min

    # Map back cell Z values
    cell_min_z_sorted = cell_min_z[np.searchsorted(unique_keys, sorted_cell_key)]
    
    # Restore original unsorted order
    local_ground_z = np.empty_like(z)
    local_ground_z[sort_idx] = cell_min_z_sorted

    # Keep only points sufficiently above local ground
    mask = z > (local_ground_z + z_threshold)
    return pts[mask]

def compare_scenes(pts2016: np.ndarray, pts2020: np.ndarray, threshold: float = 1.5, downsample: bool = False):
    """Vectorised nearest-neighbour comparison.
    Returns (static_pts, new_pts, removed_pts) in input space.
    """
    if downsample:
        pts2016 = pts2016[::40]
        pts2020 = pts2020[::40]

    if len(pts2016) == 0 or len(pts2020) == 0:
        # Gracefully handle empty arrays
        return np.empty((0, 3)), pts2020.copy(), pts2016.copy()

    threshold_sq = threshold ** 2

    # Pass 1: Find new and static points (2020 vs 2016)
    # print("Classifying points (vectorised NN)...")
    nn_idx_fwd = nn_numpy(pts2020, pts2016)      # each 2020 pt -> nearest 2016 pt
    d_sq_fwd = np.sum((pts2020 - pts2016[nn_idx_fwd]) ** 2, axis=1)
    is_new = d_sq_fwd > threshold_sq
    new_pts = pts2020[is_new]
    static_pts = pts2020[~is_new]

    # Pass 2: Find removed points (2016 vs 2020)
    nn_idx_rev = nn_numpy(pts2016, pts2020)      # each 2016 pt -> nearest 2020 pt
    d_sq_rev = np.sum((pts2016 - pts2020[nn_idx_rev]) ** 2, axis=1)
    removed_pts = pts2016[d_sq_rev > threshold_sq]

    # print(f"Classification result: {len(new_pts)} added, {len(removed_pts)} removed, {len(static_pts)} static.")
    return static_pts, new_pts, removed_pts

def cluster_and_categorize(pts: np.ndarray,
                          eps: float = 0.6,
                          min_points: int = 15,
                          min_building_area: float = 20.0,
                          min_building_height: float = 2.0,
                          min_tree_height: float = 2.0,
                          max_tree_area: float = 25.0,
                          max_pole_width: float = 1.5,
                          min_pole_height: float = 1.5) -> list:
    """Euclidean cluster extraction using fast Scipy cKDTree with physical geometry rules.
    Optimised via 15cm voxel grid downsampling prior to clustering.
    """
    if len(pts) == 0:
        return []

    from scipy.spatial import cKDTree
    
    # 1. Voxel downsample (15cm) to compress points for DBSCAN
    voxel_size = 0.15
    voxel_coords = np.floor(pts / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
    ds_pts = pts[unique_idx]

    # 2. Run DBSCAN on downsampled points
    n_ds = len(ds_pts)
    ds_labels = -np.ones(n_ds, dtype=np.int32)
    cluster_id = 0
    
    ds_tree = cKDTree(ds_pts)
    neighbors_list = ds_tree.query_ball_point(ds_pts, eps)
    
    # Scale min points for downsampled density
    min_pts_ds = max(3, int(min_points * 0.2))

    for i in range(n_ds):
        if ds_labels[i] != -1:
            continue
        neighbors = neighbors_list[i]
        if len(neighbors) < min_pts_ds:
            ds_labels[i] = -2  # noise
            continue
        ds_labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set_set = set(neighbors)
        
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if ds_labels[q] == -2:
                ds_labels[q] = cluster_id
            if ds_labels[q] != -1:
                j += 1
                continue
            ds_labels[q] = cluster_id
            q_neighbors = neighbors_list[q]
            if len(q_neighbors) >= min_pts_ds:
                for idx in q_neighbors:
                    if idx not in seed_set_set:
                        seed_set_set.add(idx)
                        seed_set.append(idx)
            j += 1
        cluster_id += 1

    # 3. Map labels back to original points via 1-NN KDTree query
    _, nn_idx = ds_tree.query(pts, k=1)
    labels = ds_labels[nn_idx]

    # --- categorize each cluster using original full resolution ---
    results = []
    category_counts = {'building': 0, 'tree': 0, 'pole': 0, 'human': 0, 'object': 0, 'ground_noise': 0}
    
    for cid in range(cluster_id):
        mask = labels == cid
        cluster_pts = pts[mask]
        if len(cluster_pts) < min_points:
            continue

        bb_min = cluster_pts.min(axis=0)   # [X, H, Y]
        bb_max = cluster_pts.max(axis=0)
        extent = bb_max - bb_min           # [dX, dH, dY]

        width_x = float(extent[0])
        height = float(extent[1])
        width_y = float(extent[2])
        area = width_x * width_y
        max_width = max(width_x, width_y)
        aspect = height / max_width if max_width > 0 else 0

        # Heuristic rules
        # 1. Filter out flat ground residues
        if height < 0.5 and aspect < 0.2:
            category = 'ground_noise'
        # 2. Building: large area, tall
        elif area >= min_building_area and height >= min_building_height:
            category = 'building'
        # 3. Human: tall-ish, very small horizontal footprint, vertical aspect ratio
        elif 0.8 <= height <= 2.1 and width_x <= 1.0 and width_y <= 1.0 and area <= 0.8 and aspect >= 1.2:
            category = 'human'
        # 4. Pole: very thin, vertically elongated
        elif width_x < max_pole_width and width_y < max_pole_width and height >= min_pole_height and aspect >= 1.5:
            category = 'pole'
        # 5. Tree: tall, medium area
        elif height >= min_tree_height and area < max_tree_area:
            category = 'tree'
        # 6. Generic Object (cars, barriers, etc.)
        else:
            category = 'object'

        category_counts[category] += 1
        
        # We completely ignore ground noise clusters to keep the scene clean
        if category != 'ground_noise':
            results.append((cluster_pts, category))
            
    return results

    # print(f"Clustering: {cluster_id} raw clusters found "
    #       f"({category_counts['building']} buildings, "
    #       f"{category_counts['tree']} trees, "
    #       f"{category_counts['pole']} poles, "
    #       f"{category_counts['object']} objects, "
    #       f"{category_counts['ground_noise']} ground noise filtered)")
          
    return results
