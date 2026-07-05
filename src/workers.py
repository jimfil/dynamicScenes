import traceback
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import src.octnode as octN
import src.quadnode as quadN
from src.pipeline import compare_scenes, cluster_and_categorize
from src.geometry_utils import color_points_by_height, compare_trees

class FrameProcessingWorker(QThread):
    result_ready = pyqtSignal(dict)
    
    def __init__(self, loader, frame_idx: int, pcd_mode: str, mode: str = None):
        super().__init__()
        self.loader = loader
        self.frame_idx = frame_idx
        self.pcd_mode = pcd_mode  # raw, compare, classify, octree, quadtree
        
    def run(self):
        try:
            def get_frame(idx):
                pts, _ = self.loader.load_raw_frame(idx)
                return pts

            if self.pcd_mode == "raw":
                pts = get_frame(self.frame_idx)
                colors = color_points_by_height(pts)
                self.result_ready.emit({
                    "success": True,
                    "mode": "raw",
                    "index": self.frame_idx,
                    "pts": pts,
                    "colors": colors
                })
                
            elif self.pcd_mode in ["compare", "classify"]:
                pts_curr = get_frame(self.frame_idx)
                pts_prev = pts_curr if self.frame_idx == 1 else get_frame(self.frame_idx - 1)
                
                static, added, removed = compare_scenes(pts_prev, pts_curr, threshold=1.0)
                
                if self.pcd_mode == "compare":
                    static_colors = color_points_by_height(static)
                    self.result_ready.emit({
                        "success": True,
                        "mode": "compare",
                        "index": self.frame_idx,
                        "static": static,
                        "static_colors": static_colors,
                        "added": added,
                        "removed": removed
                    })
                else:
                    # Classify mode
                    clusters = cluster_and_categorize(static, eps=0.6, min_points=15)
                    self.result_ready.emit({
                        "success": True,
                        "mode": "classify",
                        "index": self.frame_idx,
                        "clusters": clusters,
                        "added": added,
                        "removed": removed,
                        "static": static,
                        "raw": pts_curr
                    })
                    
            elif self.pcd_mode in ["octree", "quadtree"]:
                pts_curr = get_frame(self.frame_idx)
                pts_prev = pts_curr if self.frame_idx == 1 else get_frame(self.frame_idx - 1)
                
                static, added, removed = compare_scenes(pts_prev, pts_curr, threshold=1.0)
                
                all_pts = np.vstack((pts_prev, pts_curr))
                g_center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2.0
                g_size = float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2.0)
                
                boxes = []
                
                if self.pcd_mode == "octree":
                    print("Building OctTrees...")
                    tree_prev = octN.OctNode(pts_prev, g_center, g_size, depth=0, max_depth=6, capacity=50)
                    tree_curr = octN.OctNode(pts_curr, g_center, g_size, depth=0, max_depth=6, capacity=50)
                    
                    changed_nodes = compare_trees(tree_prev, tree_curr, threshold=1)
                    
                    for node, status in changed_nodes:
                        boxes.append({
                            'p1': node.center - node.size,
                            'p2': node.center + node.size,
                            'status': status
                        })
                        
                elif self.pcd_mode == "quadtree":
                    print("Building QuadTrees...")
                    pts_prev_2d = pts_prev[:, [0, 2]]
                    pts_curr_2d = pts_curr[:, [0, 2]]
                    center_2d = g_center[[0, 2]]
                    
                    tree_prev = quadN.QuadNode(pts_prev_2d, center_2d, g_size, depth=0, max_depth=7, capacity=40)
                    tree_curr = quadN.QuadNode(pts_curr_2d, center_2d, g_size, depth=0, max_depth=7, capacity=40)
                    
                    changed_nodes = compare_trees(tree_prev, tree_curr, threshold=5)
                    
                    all_min = np.vstack((pts_prev, pts_curr)).min(axis=0)
                    all_max = np.vstack((pts_prev, pts_curr)).max(axis=0)
                    h_min, h_max = float(all_min[1]), float(all_max[1])
                    
                    for node, status in changed_nodes:
                        cx, cy = node.center
                        sz = node.size
                        p1 = np.array([cx - sz, h_min, cy - sz])
                        p2 = np.array([cx + sz, h_max, cy + sz])
                        boxes.append({
                            'p1': p1,
                            'p2': p2,
                            'status': status
                        })
                        
                static_colors = color_points_by_height(static)
                self.result_ready.emit({
                    "success": True,
                    "mode": self.pcd_mode,
                    "index": self.frame_idx,
                    "static": static,
                    "static_colors": static_colors,
                    "added": added,
                    "removed": removed,
                    "boxes": boxes
                })
                
        except Exception as e:
            traceback.print_exc()
            self.result_ready.emit({
                "success": False,
                "index": self.frame_idx,
                "error": str(e)
            })


class SequencePrecomputer(QThread):
    frame_done = pyqtSignal(int)

    def __init__(self, loader, viewer):
        super().__init__()
        self.loader = loader
        self.viewer = viewer
        self.num_frames = loader.get_num_frames()

    def run(self):
        pts_prev = None
        for idx in range(1, self.num_frames + 1):
            try:
                pts_curr, _ = self.loader.load_raw_frame(idx)
                if idx == 1:
                    pts_prev = pts_curr

                # 1. Change detection
                static, added, removed = compare_scenes(pts_prev, pts_curr, threshold=1.0)
                colors_raw = color_points_by_height(pts_curr)
                colors_static = color_points_by_height(static)

                # 2. DBSCAN Clustering
                clusters = cluster_and_categorize(static, eps=0.6, min_points=15)

                # 3. Dynamic Track Association (tightened 4.0m limit)
                associated_clusters = self.associate_tracks(clusters, idx)

                # 4. Save to global cache
                self.viewer.frame_data_cache[idx] = {
                    "raw": pts_curr,
                    "colors_raw": colors_raw,
                    "static": static,
                    "colors_static": colors_static,
                    "added": added,
                    "removed": removed,
                    "clusters": associated_clusters
                }

                pts_prev = pts_curr
                self.frame_done.emit(idx)
                if idx % 50 == 0 or idx == self.num_frames:
                    print(f"Background pre-caching progress: {idx}/{self.num_frames} frames processed.")
            except Exception as e:
                import traceback
                print(f"Error precomputing frame {idx}: {e}")
                traceback.print_exc()

    def associate_tracks(self, clusters, frame_idx):
        associated = []
        for cluster_pts, category in clusters:
            centroid = cluster_pts.mean(axis=0)
            best_track_id = -1
            min_dist = float('inf')

            # Find closest active track of the same category
            for track_id, track_info in self.viewer.global_tracks.items():
                if track_info["category"] != category:
                    continue
                dist = float(np.linalg.norm(centroid - track_info["last_centroid"]))
                if dist < min_dist and dist < 4.0:  # Tightened tracking distance
                    if track_info["last_seen_frame"] < frame_idx:
                        best_track_id = track_id
                        min_dist = dist

            if best_track_id != -1:
                # Update existing track
                self.viewer.global_tracks[best_track_id]["last_centroid"] = centroid
                self.viewer.global_tracks[best_track_id]["last_seen_frame"] = frame_idx
                track_id = best_track_id
            else:
                # Create a new track
                track_id = self.viewer.next_track_id
                self.viewer.global_tracks[track_id] = {
                    "category": category,
                    "last_centroid": centroid,
                    "last_seen_frame": frame_idx
                }
                self.viewer.next_track_id += 1

            associated.append((cluster_pts, category, track_id))
        return associated


