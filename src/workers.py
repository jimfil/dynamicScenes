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
                    clusters = cluster_and_categorize(static, eps=0.8, min_points=15)
                    self.result_ready.emit({
                        "success": True,
                        "mode": "classify",
                        "index": self.frame_idx,
                        "clusters": clusters,
                        "added": added,
                        "removed": removed
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


