from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene3D_
from vvrpywork.shapes import (
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D
)

import numpy as np
import time
import argparse
from pathlib import Path
from PyQt5.QtWidgets import QCheckBox, QComboBox, QLabel, QFrame
from PyQt5.QtCore import QCoreApplication

from src.loaders import PCDSequenceLoader, LIDARSequenceLoader
from src.workers import FrameProcessingWorker
import src.octnode as octN
import src.quadnode as quadN
from src.geometry_utils import compare_trees

WIDTH = 1000
HEIGHT = 800

class DynamicSceneViewer(Scene3D_):
    COLOR_ADDED = (0.0, 1.0, 0.0, 1.0)      # Green
    COLOR_REMOVED = (1.0, 0.0, 0.0, 1.0)    # Red
    COLOR_CHANGED = (1.0, 0.5, 0.0, 1.0)    # Orange
    
    CAT_COLOR = {
        'building': (0.0, 0.9, 1.0, 1.0),    # Cyan
        'tree':     (0.1, 0.9, 0.1, 1.0),    # Green
        'pole':     (1.0, 1.0, 0.0, 1.0),    # Yellow
        'human':    (1.0, 0.3, 0.8, 1.0),    # Vibrant Magenta/Pink
        'object':   (1.0, 0.5, 0.0, 1.0),    # Orange
    }
    
    SIZE_STATIC = 1.0
    SIZE_ADDED = 1.0
    SIZE_REMOVED = 1.0
    SIZE_RAW = 1.0
    SIZE_CLUSTER = 1.0

    def __init__(self, loader, title="Dynamic Point Cloud Viewer"):
        super().__init__(WIDTH, HEIGHT, title, output=True)
        self.loader = loader
        self.frame_idx = 1
        self.viewer_mode = "raw"  # raw, compare, classify, octree, quadtree
        self.animate_mode = False
        self.last_animation_time = 0.0
        
        self.active_workers = set()
        self.is_processing = False
        
        self.frame_data_cache = {}
        self.global_tracks = {}
        self.next_track_id = 1
        self.focused_track_id = -1
        
        # Enable interactive left-click selection on 3D objects
        self._plotter.enable_mesh_picking(
            callback=self.on_mesh_picked,
            use_actor=True,
            show=False,
            show_message=False,
            left_clicking=True,
            picker='point'
        )
        
        self.setup_ui_panel()
        self.num_frames = self.loader.get_num_frames()
        print(f"Loaded dataset with {self.num_frames} frames.")
        
        from src.workers import SequencePrecomputer
        self.precomputer = SequencePrecomputer(self.loader, self)
        self.precomputer.frame_done.connect(self.on_frame_precomputed)
        self.precomputer.start()
        
        self.load_and_display_frame(self.frame_idx)
        
    def setup_ui_panel(self):
        left_widget = None
        if hasattr(self, "_scroll") and self._scroll is not None:
            left_widget = self._scroll.parentWidget()
        if left_widget is None:
            left_widget = self._window.centralWidget().layout().itemAt(0).widget()
            
        left_layout = left_widget.layout()
        
        # Dataset Selector
        dataset_label = QLabel("<b>Dataset:</b>")
        left_layout.addWidget(dataset_label)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("Dynamic PCD Sequence")
        self.dataset_combo.addItem("Aerial LIDAR Comparison")
        
        initial_idx = 1 if isinstance(self.loader, LIDARSequenceLoader) else 0
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.setCurrentIndex(initial_idx)
        self.dataset_combo.blockSignals(False)
        self.dataset_combo.currentIndexChanged.connect(self.on_dataset_changed)
        left_layout.addWidget(self.dataset_combo)
        
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        left_layout.addWidget(line)
        
        self.print("Controls:\n"
                   "  SPACE: Play/Pause animation\n"
                   "  N: Next frame\n"
                   "  P: Previous frame\n"
                   "  R: Reset to frame 1\n"
                   "  1: Raw Height mode\n"
                   "  2: Change Detection mode\n"
                   "  3: Semantic Classification\n"
                   "  4: Octree comparison\n"
                   "  5: Quadtree comparison\n\n"
                   "Colors (Classification):\n"
                   "  Cyan: Buildings\n"
                   "  Green: Trees\n"
                   "  Yellow: Poles\n"
                   "  Pink/Magenta: Humans\n"
                   "  Orange: Generic Objects\n\n"
                   "Colors (Comparison):\n"
                   "  Green: Added points\n"
                   "  Red: Removed points\n"
                   "  Blue -> Red: Static points (Height coded)\n"
                  )
        
        category_label = QLabel("<b>Focus Category:</b>")
        left_layout.addWidget(category_label)
        self.category_focus_combo = QComboBox()
        self.category_focus_combo.addItems(["All Categories", "Buildings", "Trees", "Poles", "Humans", "Objects"])
        self.category_focus_combo.currentIndexChanged.connect(self.on_focus_category_changed)
        left_layout.addWidget(self.category_focus_combo)

        combo_label = QLabel("<b>Single Object Focus:</b>")
        left_layout.addWidget(combo_label)
        
        self.cluster_combo = QComboBox()
        self.cluster_combo.addItem("All Objects")
        self.cluster_combo.currentIndexChanged.connect(self.on_object_focus_changed)
        left_layout.addWidget(self.cluster_combo)

    def on_dataset_changed(self, index):
        if self.is_processing:
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.setCurrentIndex(1 - index)
            self.dataset_combo.blockSignals(False)
            return

        self.animate_mode = False

        # Stop existing precomputer
        if hasattr(self, "precomputer") and self.precomputer.isRunning():
            self.precomputer.terminate()
            self.precomputer.wait()

        for name in list(self._shapeDict.keys()):
            self.removeShape(name)

        self.frame_idx = 1
        self.frame_data_cache.clear()
        self.global_tracks.clear()
        self.next_track_id = 1
        self.focused_track_id = -1

        if index == 0:
            self.loader = PCDSequenceLoader()
            self.viewer_mode = "raw"
            self.num_frames = self.loader.get_num_frames()
            print("Switched dynamically to Dynamic PCD Sequence.")
        else:
            self.loader = LIDARSequenceLoader()
            self.viewer_mode = "compare"
            self.num_frames = self.loader.get_num_frames()
            print("Switched dynamically to Aerial LIDAR Comparison.")

        # Re-start precomputer
        from src.workers import SequencePrecomputer
        self.precomputer = SequencePrecomputer(self.loader, self)
        self.precomputer.frame_done.connect(self.on_frame_precomputed)
        self.precomputer.start()

        # Reset selection dropdowns
        self.category_focus_combo.blockSignals(True)
        self.category_focus_combo.setCurrentIndex(0)
        self.category_focus_combo.blockSignals(False)
        self.rebuild_object_combo()

        self.load_and_display_frame(self.frame_idx)

    def on_focus_category_changed(self):
        self.rebuild_object_combo()
        self.update_task_visibility()

    def on_object_focus_changed(self):
        selected_idx = self.cluster_combo.currentIndex()
        if selected_idx > 0:
            text = self.cluster_combo.itemText(selected_idx)
            parts = text.split(" ")
            self.focused_track_id = int(parts[1])
        else:
            self.focused_track_id = -1
        self.update_task_visibility()

    def on_mesh_picked(self, actor):
        if actor is None:
            return
        
        picked_name = None
        for name, shape in self._shapeDict.items():
            if hasattr(shape, "_actor") and shape._actor == actor:
                picked_name = name
                break
                
        if picked_name is None:
            return
            
        if picked_name.startswith("cluster_") or picked_name.startswith("box_"):
            parts = picked_name.split("_")
            try:
                track_id = int(parts[1])
                category = parts[2]
                
                cat_map_rev = {
                    "building": "Buildings",
                    "tree": "Trees",
                    "pole": "Poles",
                    "human": "Humans",
                    "object": "Objects"
                }
                
                new_cat_text = cat_map_rev.get(category, "All Categories")
                current_cat_text = self.category_focus_combo.currentText()
                
                if new_cat_text != current_cat_text:
                    self.category_focus_combo.blockSignals(True)
                    idx = self.category_focus_combo.findText(new_cat_text)
                    if idx != -1:
                        self.category_focus_combo.setCurrentIndex(idx)
                    self.category_focus_combo.blockSignals(False)
                    self.rebuild_object_combo()

                item_text = f"Object {track_id} ({category})"
                self.cluster_combo.blockSignals(True)
                idx = self.cluster_combo.findText(item_text)
                if idx != -1:
                    self.cluster_combo.setCurrentIndex(idx)
                    self.focused_track_id = track_id
                else:
                    self.focused_track_id = -1
                self.cluster_combo.blockSignals(False)
                
                self.update_task_visibility()
                print(f"Focused on: Object {track_id} ({category})")
                
            except Exception as e:
                print(f"Error handling pick: {e}")

    def rebuild_object_combo(self):
        cat_filter = self.category_focus_combo.currentText().lower()
        if cat_filter == "all categories":
            cat_map = None
        elif cat_filter == "buildings":
            cat_map = "building"
        elif cat_filter == "trees":
            cat_map = "tree"
        elif cat_filter == "poles":
            cat_map = "pole"
        elif cat_filter == "humans":
            cat_map = "human"
        elif cat_filter == "objects":
            cat_map = "object"
        else:
            cat_map = None

        self.cluster_combo.blockSignals(True)
        self.cluster_combo.clear()
        self.cluster_combo.addItem("All Objects")
        
        # Populate matching tracks
        for track_id in sorted(self.global_tracks.keys()):
            track_cat = self.global_tracks[track_id]["category"]
            if cat_map is None or track_cat == cat_map:
                self.cluster_combo.addItem(f"Object {track_id} ({track_cat})")
                
        # Try to restore selection
        if self.focused_track_id > 0:
            track_cat = self.global_tracks.get(self.focused_track_id, {}).get("category", "")
            item_text = f"Object {self.focused_track_id} ({track_cat})"
            idx = self.cluster_combo.findText(item_text)
            if idx != -1:
                self.cluster_combo.setCurrentIndex(idx)
            else:
                self.focused_track_id = -1
                self.cluster_combo.setCurrentIndex(0)
        else:
            self.cluster_combo.setCurrentIndex(0)
        self.cluster_combo.blockSignals(False)

    def on_frame_precomputed(self, index):
        self.rebuild_object_combo()

    def update_task_visibility(self):
        if self.frame_idx in self.frame_data_cache:
            self.display_cached_frame(self.frame_idx)
            return

        selected_idx = self.cluster_combo.currentIndex()
        
        # Get category filter mapping
        cat_filter = self.category_focus_combo.currentText().lower()
        cat_map = None
        if cat_filter == "buildings":
            cat_map = "building"
        elif cat_filter == "trees":
            cat_map = "tree"
        elif cat_filter == "poles":
            cat_map = "pole"
        elif cat_filter == "humans":
            cat_map = "human"
        elif cat_filter == "objects":
            cat_map = "object"

        for name, shape in list(self._shapeDict.items()):
            if name.startswith("cluster_") or name.startswith("box_"):
                parts = name.split("_")
                track_id = int(parts[1])
                category = parts[2]
                
                visible = True
                if cat_map is not None and category != cat_map:
                    visible = False
                    
                if self.focused_track_id > 0:
                    if track_id != self.focused_track_id:
                        visible = False
                elif selected_idx > 0:
                    if track_id != (selected_idx - 1):
                        visible = False
                        
                if hasattr(shape, "_actor") and shape._actor is not None:
                    shape._actor.SetVisibility(visible)
        self._plotter.render()

    def remove_shapes_silent(self, names_list):
        for name in names_list:
            if name in self._shapeDict:
                shape = self._shapeDict[name]
                if hasattr(shape, "_actor") and shape._actor is not None:
                    try:
                        self._plotter.remove_actor(shape._actor, render=False)
                    except Exception:
                        pass
                del self._shapeDict[name]

    def clear_dynamic_shapes(self):
        # Remove old cluster and box shapes in a single pass without rendering intermediate frames
        names_to_remove = []
        for name in list(self._shapeDict.keys()):
            if name.startswith("cluster_") or name.startswith("box_") or name.startswith("tree_box_"):
                names_to_remove.append(name)
        self.remove_shapes_silent(names_to_remove)

    def load_and_display_frame(self, index):
        if index in self.frame_data_cache:
            self.display_cached_frame(index)
        else:
            if self.is_processing:
                return
                
            self.is_processing = True
            worker = FrameProcessingWorker(self.loader, index, self.viewer_mode)
            self.active_workers.add(worker)
            
            worker.result_ready.connect(self.on_frame_processed)
            worker.finished.connect(lambda w=worker: self.active_workers.discard(w))
            worker.finished.connect(worker.deleteLater)
            
            worker.start()

    def display_cached_frame(self, index):
        self.clear_dynamic_shapes()
        data = self.frame_data_cache[index]

        if self.viewer_mode == "raw":
            pts = data["raw"]
            colors = data["colors_raw"]
            self.remove_shapes_silent(["PCD_Static", "PCD_Added", "PCD_Removed"])
            if pts is not None and len(pts) > 0:
                print(f"Rendering Frame {index} (Raw Mode)...")
                if "PCD_Raw" in self._shapeDict:
                    ps = self._shapeDict["PCD_Raw"]
                    ps.points = pts
                    ps.colors = colors
                    self.updateShape("PCD_Raw")
                else:
                    ps = PointSet3D(pts, size=self.SIZE_RAW)
                    ps.colors = colors.tolist()
                    self.addShape(ps, name="PCD_Raw")

        elif self.viewer_mode == "compare":
            static_pts = data["static"]
            static_colors = data["colors_static"]
            added_pts = data["added"]
            removed_pts = data["removed"]
            
            self.remove_shapes_silent(["PCD_Raw"])
            print(f"Rendering Frame {index} Comparison...")

            if len(static_pts) > 0:
                if "PCD_Static" in self._shapeDict:
                    ps = self._shapeDict["PCD_Static"]
                    ps.points = static_pts
                    ps.colors = static_colors
                    self.updateShape("PCD_Static")
                else:
                    ps = PointSet3D(static_pts, size=self.SIZE_STATIC)
                    ps.colors = static_colors.tolist()
                    self.addShape(ps, name="PCD_Static")
            elif "PCD_Static" in self._shapeDict:
                self.remove_shapes_silent(["PCD_Static"])

            if len(added_pts) > 0:
                if "PCD_Added" in self._shapeDict:
                    ps = self._shapeDict["PCD_Added"]
                    ps.points = added_pts
                    ps.colors = np.tile(self.COLOR_ADDED, (len(added_pts), 1))
                    self.updateShape("PCD_Added")
                else:
                    ps = PointSet3D(added_pts, size=self.SIZE_ADDED, color=self.COLOR_ADDED)
                    self.addShape(ps, name="PCD_Added")
            elif "PCD_Added" in self._shapeDict:
                self.remove_shapes_silent(["PCD_Added"])

            if len(removed_pts) > 0:
                if "PCD_Removed" in self._shapeDict:
                    ps = self._shapeDict["PCD_Removed"]
                    ps.points = removed_pts
                    ps.colors = np.tile(self.COLOR_REMOVED, (len(removed_pts), 1))
                    self.updateShape("PCD_Removed")
                else:
                    ps = PointSet3D(removed_pts, size=self.SIZE_REMOVED, color=self.COLOR_REMOVED)
                    self.addShape(ps, name="PCD_Removed")
            elif "PCD_Removed" in self._shapeDict:
                self.remove_shapes_silent(["PCD_Removed"])

        elif self.viewer_mode == "classify":
            self._plotter.suppress_rendering = True
            try:
                clusters = data["clusters"]
                self.remove_shapes_silent(["PCD_Raw", "PCD_Static", "PCD_Added", "PCD_Removed"])

                # print(f"Segmenting Frame {index} (Classification)...")

                static_pts = data["static"]
                if len(static_pts) > 0:
                    ps = PointSet3D(static_pts, size=1.0, color=(0.5, 0.5, 0.5, 0.15))
                    self.addShape(ps, name="PCD_Static")

                selected_idx = self.cluster_combo.currentIndex()

                # Get category filter mapping
                cat_filter = self.category_focus_combo.currentText().lower()
                cat_map = None
                if cat_filter == "buildings":
                    cat_map = "building"
                elif cat_filter == "trees":
                    cat_map = "tree"
                elif cat_filter == "poles":
                    cat_map = "pole"
                elif cat_filter == "humans":
                    cat_map = "human"
                elif cat_filter == "objects":
                    cat_map = "object"

                for cluster_pts, category, track_id in clusters:
                    visible = True

                    if cat_map is not None and category != cat_map:
                        visible = False

                    if self.focused_track_id > 0:
                        if track_id != self.focused_track_id:
                            visible = False
                    elif selected_idx > 0:
                        if track_id != (selected_idx - 1):
                            visible = False

                    if visible:
                        color = self.CAT_COLOR.get(category, (1.0, 1.0, 1.0, 1.0))
                        self.addShape(PointSet3D(cluster_pts, size=self.SIZE_CLUSTER, color=color), name=f"cluster_{track_id}_{category}")
                        bb_min = cluster_pts.min(axis=0)
                        bb_max = cluster_pts.max(axis=0)
                        self.addShape(Cuboid3D(bb_min, bb_max, color=color, filled=False), name=f"box_{track_id}_{category}")
            finally:
                self._plotter.suppress_rendering = False
                self._plotter.render()

        elif self.viewer_mode in ["octree", "quadtree"]:
            static_pts = data["static"]
            static_colors = data["colors_static"]
            added_pts = data["added"]
            removed_pts = data["removed"]
            
            self.remove_shapes_silent(["PCD_Raw"])
                
            print(f"Rendering Frame {index} Subdivision ({self.viewer_mode.upper()})...")
            
            if len(static_pts) > 0:
                ps = PointSet3D(static_pts, size=self.SIZE_STATIC)
                ps.colors = static_colors.tolist()
                self.addShape(ps, name="PCD_Static")
                
            if len(added_pts) > 0:
                ps = PointSet3D(added_pts, size=self.SIZE_ADDED, color=self.COLOR_ADDED)
                self.addShape(ps, name="PCD_Added")
                
            if len(removed_pts) > 0:
                ps = PointSet3D(removed_pts, size=self.SIZE_REMOVED, color=self.COLOR_REMOVED)
                self.addShape(ps, name="PCD_Removed")
                
            pts_curr = data["raw"]
            pts_prev = self.frame_data_cache[index - 1]["raw"] if index > 1 else pts_curr
            all_pts = np.vstack((pts_prev, pts_curr))
            g_center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2.0
            g_size = float(np.max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2.0)
            
            boxes = []
            if self.viewer_mode == "octree":
                tree_prev = octN.OctNode(pts_prev, g_center, g_size, depth=0, max_depth=6, capacity=50)
                tree_curr = octN.OctNode(pts_curr, g_center, g_size, depth=0, max_depth=6, capacity=50)
                changed_nodes = compare_trees(tree_prev, tree_curr, threshold=1)
                for node, status in changed_nodes:
                    boxes.append({
                        'p1': node.center - node.size,
                        'p2': node.center + node.size,
                        'status': status
                    })
            else:
                pts_prev_2d = pts_prev[:, [0, 2]]
                pts_curr_2d = pts_curr[:, [0, 2]]
                center_2d = g_center[[0, 2]]
                tree_prev = quadN.QuadNode(pts_prev_2d, center_2d, g_size, depth=0, max_depth=7, capacity=40)
                tree_curr = quadN.QuadNode(pts_curr_2d, center_2d, g_size, depth=0, max_depth=7, capacity=40)
                changed_nodes = compare_trees(tree_prev, tree_curr, threshold=5)
                all_min = all_pts.min(axis=0)
                all_max = all_pts.max(axis=0)
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
                    
            for i, box in enumerate(boxes):
                p1, p2, status = box['p1'], box['p2'], box['status']
                color = self.COLOR_ADDED if status == 'added' else (self.COLOR_REMOVED if status == 'removed' else self.COLOR_CHANGED)
                self.addShape(Cuboid3D(p1, p2, color=color, filled=False), name=f"tree_box_{i}")

    def on_frame_processed(self, result):
        self.is_processing = False
        
        if not result["success"]:
            print(f"Error: {result.get('error', 'Unknown error')}")
            if self.animate_mode:
                self.animate_mode = False
                print("Animation paused due to error.")
            return
            
        index = result["index"]
        mode = result["mode"]
        
        if mode in ["raw", "compare", "octree", "quadtree"]:
            self.cluster_combo.blockSignals(True)
            self.cluster_combo.clear()
            self.cluster_combo.addItem("All Clusters")
            self.cluster_combo.blockSignals(False)
            
        self.clear_dynamic_shapes()
        
        if mode == "raw":
            pts = result["pts"]
            colors = result["colors"]
            
            for name in ["PCD_Static", "PCD_Added", "PCD_Removed"]:
                if name in self._shapeDict:
                    self.removeShape(name)
                    
            if pts is not None and len(pts) > 0:
                print(f"Rendering Frame {index} (Raw Mode)...")
                if "PCD_Raw" in self._shapeDict:
                    ps = self._shapeDict["PCD_Raw"]
                    ps.points = pts
                    ps.colors = colors
                    self.updateShape("PCD_Raw")
                else:
                    ps = PointSet3D(pts, size=self.SIZE_RAW)
                    ps.colors = colors.tolist()
                    self.addShape(ps, name="PCD_Raw")
            else:
                print(f"Frame {index} has no points.")
                
        elif mode == "compare":
            static_pts = result["static"]
            static_colors = result["static_colors"]
            added_pts = result["added"]
            removed_pts = result["removed"]
            
            if "PCD_Raw" in self._shapeDict:
                self.removeShape("PCD_Raw")
                
            print(f"Rendering Frame {index} Comparison...")
            
            if len(static_pts) > 0:
                if "PCD_Static" in self._shapeDict:
                    ps = self._shapeDict["PCD_Static"]
                    ps.points = static_pts
                    ps.colors = static_colors
                    self.updateShape("PCD_Static")
                else:
                    ps = PointSet3D(static_pts, size=self.SIZE_STATIC)
                    ps.colors = static_colors.tolist()
                    self.addShape(ps, name="PCD_Static")
            elif "PCD_Static" in self._shapeDict:
                self.removeShape("PCD_Static")
                
            if len(added_pts) > 0:
                if "PCD_Added" in self._shapeDict:
                    ps = self._shapeDict["PCD_Added"]
                    ps.points = added_pts
                    ps.colors = np.tile(self.COLOR_ADDED, (len(added_pts), 1))
                    self.updateShape("PCD_Added")
                else:
                    ps = PointSet3D(added_pts, size=self.SIZE_ADDED, color=self.COLOR_ADDED)
                    self.addShape(ps, name="PCD_Added")
            elif "PCD_Added" in self._shapeDict:
                self.removeShape("PCD_Added")
                
            if len(removed_pts) > 0:
                if "PCD_Removed" in self._shapeDict:
                    ps = self._shapeDict["PCD_Removed"]
                    ps.points = removed_pts
                    ps.colors = np.tile(self.COLOR_REMOVED, (len(removed_pts), 1))
                    self.updateShape("PCD_Removed")
                else:
                    ps = PointSet3D(removed_pts, size=self.SIZE_REMOVED, color=self.COLOR_REMOVED)
                    self.addShape(ps, name="PCD_Removed")
            elif "PCD_Removed" in self._shapeDict:
                self.removeShape("PCD_Removed")
                
        elif mode == "classify":
            self._plotter.suppress_rendering = True
            try:
                clusters = result["clusters"]
                
                self.remove_shapes_silent(["PCD_Raw", "PCD_Static", "PCD_Added", "PCD_Removed"])
                
                # Perform track association on-the-fly for on-demand computed frames
                associated_clusters = []
                for cluster_pts, category in clusters:
                    centroid = cluster_pts.mean(axis=0)
                    best_track_id = -1
                    min_dist = float('inf')
                    for track_id, track_info in self.global_tracks.items():
                        if track_info["category"] != category:
                            continue
                        dist = float(np.linalg.norm(centroid - track_info["last_centroid"]))
                        if dist < min_dist and dist < 4.0:
                            if track_info["last_seen_frame"] < index:
                                best_track_id = track_id
                                min_dist = dist
                                
                    if best_track_id != -1:
                        self.global_tracks[best_track_id]["last_centroid"] = centroid
                        self.global_tracks[best_track_id]["last_seen_frame"] = index
                        track_id = best_track_id
                    else:
                        track_id = self.next_track_id
                        self.global_tracks[track_id] = {
                            "category": category,
                            "last_centroid": centroid,
                            "last_seen_frame": index
                        }
                        self.next_track_id += 1
                    associated_clusters.append((cluster_pts, category, track_id))

                # Store in cache so that subsequent checkbox toggles hit the cache
                static_pts = result["static"] if "static" in result else np.empty((0,3))
                self.frame_data_cache[index] = {
                    "raw": result["raw"],
                    "static": static_pts,
                    "clusters": associated_clusters
                }

                # Render static points as transparent background
                if len(static_pts) > 0:
                    ps = PointSet3D(static_pts, size=1.0, color=(0.5, 0.5, 0.5, 0.15))
                    self.addShape(ps, name="PCD_Static")
                
                selected_idx = self.cluster_combo.currentIndex()

                # Get category filter mapping
                cat_filter = self.category_focus_combo.currentText().lower()
                cat_map = None
                if cat_filter == "buildings":
                    cat_map = "building"
                elif cat_filter == "trees":
                    cat_map = "tree"
                elif cat_filter == "poles":
                    cat_map = "pole"
                elif cat_filter == "humans":
                    cat_map = "human"
                elif cat_filter == "objects":
                    cat_map = "object"

                for cluster_pts, category, track_id in associated_clusters:
                    visible = True

                    if cat_map is not None and category != cat_map:
                        visible = False

                    if self.focused_track_id > 0:
                        if track_id != self.focused_track_id:
                            visible = False
                    elif selected_idx > 0:
                        if track_id != (selected_idx - 1):
                            visible = False

                    if visible:
                        color = self.CAT_COLOR.get(category, (1.0, 1.0, 1.0, 1.0))
                        self.addShape(PointSet3D(cluster_pts, size=self.SIZE_CLUSTER, color=color), name=f"cluster_{track_id}_{category}")
                        bb_min = cluster_pts.min(axis=0)
                        bb_max = cluster_pts.max(axis=0)
                        self.addShape(Cuboid3D(bb_min, bb_max, color=color, filled=False), name=f"box_{track_id}_{category}")
            finally:
                self._plotter.suppress_rendering = False
                self._plotter.render()

            self.rebuild_object_combo()
            
        elif mode in ["octree", "quadtree"]:
            static_pts = result["static"]
            static_colors = result["static_colors"]
            added_pts = result["added"]
            removed_pts = result["removed"]
            boxes = result["boxes"]
            
            if "PCD_Raw" in self._shapeDict:
                self.removeShape("PCD_Raw")
                
            print(f"Rendering Frame {index} Subdivision ({mode.upper()})...")
            
            if len(static_pts) > 0:
                if "PCD_Static" in self._shapeDict:
                    ps = self._shapeDict["PCD_Static"]
                    ps.points = static_pts
                    ps.colors = static_colors
                    self.updateShape("PCD_Static")
                else:
                    ps = PointSet3D(static_pts, size=self.SIZE_STATIC)
                    ps.colors = static_colors.tolist()
                    self.addShape(ps, name="PCD_Static")
            elif "PCD_Static" in self._shapeDict:
                self.removeShape("PCD_Static")
                
            if len(added_pts) > 0:
                if "PCD_Added" in self._shapeDict:
                    ps = self._shapeDict["PCD_Added"]
                    ps.points = added_pts
                    ps.colors = np.tile(self.COLOR_ADDED, (len(added_pts), 1))
                    self.updateShape("PCD_Added")
                else:
                    ps = PointSet3D(added_pts, size=self.SIZE_ADDED, color=self.COLOR_ADDED)
                    self.addShape(ps, name="PCD_Added")
            elif "PCD_Added" in self._shapeDict:
                self.removeShape("PCD_Added")
                
            if len(removed_pts) > 0:
                if "PCD_Removed" in self._shapeDict:
                    ps = self._shapeDict["PCD_Removed"]
                    ps.points = removed_pts
                    ps.colors = np.tile(self.COLOR_REMOVED, (len(removed_pts), 1))
                    self.updateShape("PCD_Removed")
                else:
                    ps = PointSet3D(removed_pts, size=self.SIZE_REMOVED, color=self.COLOR_REMOVED)
                    self.addShape(ps, name="PCD_Removed")
            elif "PCD_Removed" in self._shapeDict:
                self.removeShape("PCD_Removed")
                
            for i, box in enumerate(boxes):
                status = box['status']
                if status == 'added':
                    color = self.COLOR_ADDED[:3]
                elif status == 'removed':
                    color = self.COLOR_REMOVED[:3]
                else:
                    color = self.COLOR_CHANGED[:3]
                    
                self.addShape(Cuboid3D(box['p1'], box['p2'], color=color, filled=False), name=f"box_{i}")

        self._plotter.render()

    def on_key_press(self, symbol, modifiers):
        if symbol == Key.N:
            if self.frame_idx < self.num_frames:
                self.frame_idx += 1
                self.load_and_display_frame(self.frame_idx)
            else:
                print("Reached last frame.")
        elif symbol == Key.P:
            if self.frame_idx > 1:
                self.frame_idx -= 1
                self.load_and_display_frame(self.frame_idx)
            else:
                print("Already at first frame.")
        elif symbol == Key.R:
            self.frame_idx = 1
            self.load_and_display_frame(self.frame_idx)
        elif symbol in [Key._1, Key.NUM_1]:
            self.viewer_mode = "raw"
            print("Switched to Raw Height Mode (No Classification)")
            self.load_and_display_frame(self.frame_idx)
        elif symbol in [Key._2, Key.NUM_2]:
            self.viewer_mode = "compare"
            print("Switched to Change Detection Mode")
            self.load_and_display_frame(self.frame_idx)
        elif symbol in [Key._3, Key.NUM_3]:
            self.viewer_mode = "classify"
            self.category_focus_combo.blockSignals(True)
            self.category_focus_combo.setCurrentIndex(0)
            self.category_focus_combo.blockSignals(False)
            self.focused_track_id = -1
            self.rebuild_object_combo()
            print("Switched to Semantic Classification Mode")
            self.load_and_display_frame(self.frame_idx)
        elif symbol in [Key._4, Key.NUM_4]:
            self.viewer_mode = "octree"
            print("Switched to Octree Comparison Mode")
            self.load_and_display_frame(self.frame_idx)
        elif symbol in [Key._5, Key.NUM_5]:
            self.viewer_mode = "quadtree"
            print("Switched to Quadtree Comparison Mode")
            self.load_and_display_frame(self.frame_idx)
        elif symbol == Key.SPACE:
            self.animate_mode = not self.animate_mode
            status = "playing" if self.animate_mode else "paused"
            print(f"Animation is now {status}.")
            self.load_and_display_frame(self.frame_idx)

    def on_idle(self):
        QCoreApplication.processEvents()

        if self.animate_mode and not self.is_processing:
            current_time = time.time()
            if current_time - self.last_animation_time > 0.1:
                if self.frame_idx < self.num_frames:
                    self.frame_idx += 1
                else:
                    self.frame_idx = 1
                self.load_and_display_frame(self.frame_idx)
                self.last_animation_time = current_time
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Scenes Point Cloud Visualizer")
    parser.add_argument("files", nargs="*", help="One calibration CSV file, or two .laz/.las files.")
    args = parser.parse_args()

    if len(args.files) >= 2:
        loader = LIDARSequenceLoader(file2016=args.files[0], file2020=args.files[1])
        app = DynamicSceneViewer(loader, title="Aerial Lidar Viewer")
    elif len(args.files) == 1:
        loader = PCDSequenceLoader(csv_path=args.files[0])
        app = DynamicSceneViewer(loader, title="PCD Dynamic Scenes Viewer")
    else:
        loader = PCDSequenceLoader()
        app = DynamicSceneViewer(loader, title="PCD Dynamic Scenes Viewer")
        
    app.mainLoop()