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
from PyQt5.QtWidgets import QCheckBox, QComboBox, QLabel

from src.loaders import PCDSequenceLoader, LIDARSequenceLoader
from src.workers import FrameProcessingWorker

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
        'object':   (1.0, 0.5, 0.0, 1.0),    # Orange
    }
    
    SIZE_STATIC = 1.0
    SIZE_ADDED = 3.0
    SIZE_REMOVED = 3.0
    SIZE_RAW = 1.5
    SIZE_CLUSTER = 2.5

    def __init__(self, loader, title="Dynamic Point Cloud Viewer"):
        super().__init__(WIDTH, HEIGHT, title, output=True)
        self.loader = loader
        self.frame_idx = 1
        self.viewer_mode = "raw"  # raw, compare, classify, octree, quadtree
        self.animate_mode = False
        self.last_animation_time = 0.0
        
        self.active_workers = set()
        self.is_processing = False
        
        self.setup_ui_panel()
        self.num_frames = self.loader.get_num_frames()
        print(f"Loaded dataset with {self.num_frames} frames.")
        
        self.load_and_display_frame(self.frame_idx)
        
    def setup_ui_panel(self):
        left_widget = None
        if hasattr(self, "_scroll") and self._scroll is not None:
            left_widget = self._scroll.parentWidget()
        if left_widget is None:
            left_widget = self._window.centralWidget().layout().itemAt(0).widget()
            
        left_layout = left_widget.layout()
        
        self.print("Controls:\n")
        
        header = QLabel("<b>Filter Categories:</b>")
        left_layout.addWidget(header)
        
        self.cb_buildings = QCheckBox("Buildings (Cyan)")
        self.cb_buildings.setChecked(True)
        self.cb_buildings.stateChanged.connect(self.update_task_visibility)
        left_layout.addWidget(self.cb_buildings)
        
        self.cb_trees = QCheckBox("Trees (Green)")
        self.cb_trees.setChecked(True)
        self.cb_trees.stateChanged.connect(self.update_task_visibility)
        left_layout.addWidget(self.cb_trees)
        
        self.cb_poles = QCheckBox("Poles (Yellow)")
        self.cb_poles.setChecked(True)
        self.cb_poles.stateChanged.connect(self.update_task_visibility)
        left_layout.addWidget(self.cb_poles)
        
        self.cb_objects = QCheckBox("Objects (Orange)")
        self.cb_objects.setChecked(True)
        self.cb_objects.stateChanged.connect(self.update_task_visibility)
        left_layout.addWidget(self.cb_objects)
        
        spacer = QLabel("")
        left_layout.addWidget(spacer)
        
        combo_label = QLabel("<b>Single Cluster Focus:</b>")
        left_layout.addWidget(combo_label)
        
        self.cluster_combo = QComboBox()
        self.cluster_combo.addItem("All Clusters")
        self.cluster_combo.currentIndexChanged.connect(self.update_task_visibility)
        left_layout.addWidget(self.cluster_combo)

    def update_task_visibility(self):
        show_buildings = self.cb_buildings.isChecked()
        show_trees = self.cb_trees.isChecked()
        show_poles = self.cb_poles.isChecked()
        show_objects = self.cb_objects.isChecked()
        
        selected_idx = self.cluster_combo.currentIndex()
        
        for name, shape in list(self._shapeDict.items()):
            if name.startswith("cluster_"):
                parts = name.split("_")
                cluster_idx = int(parts[1])
                category = parts[2]
                
                visible = False
                if category == "building" and show_buildings:
                    visible = True
                elif category == "tree" and show_trees:
                    visible = True
                elif category == "pole" and show_poles:
                    visible = True
                elif category == "object" and show_objects:
                    visible = True
                    
                if selected_idx > 0:
                    if cluster_idx != (selected_idx - 1):
                        visible = False
                        
                if hasattr(shape, "_actor") and shape._actor is not None:
                    shape._actor.SetVisibility(visible)
        self._plotter.render()

    def clear_dynamic_shapes(self):
        # Remove old cluster and box shapes
        for name in list(self._shapeDict.keys()):
            if name.startswith("cluster_") or name.startswith("box_"):
                self.removeShape(name)

    def load_and_display_frame(self, index):
        if self.is_processing:
            return
            
        self.is_processing = True
        worker = FrameProcessingWorker(self.loader, index, self.viewer_mode)
        self.active_workers.add(worker)
        
        worker.result_ready.connect(self.on_frame_processed)
        worker.finished.connect(lambda w=worker: self.active_workers.discard(w))
        worker.finished.connect(worker.deleteLater)
        
        worker.start()

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
            clusters = result["clusters"]
            
            for name in ["PCD_Raw", "PCD_Static", "PCD_Added", "PCD_Removed"]:
                if name in self._shapeDict:
                    self.removeShape(name)
                    
            print(f"Segmenting Frame {index} (Semantic Classification)...")
            
            self.cluster_combo.blockSignals(True)
            self.cluster_combo.clear()
            self.cluster_combo.addItem("All Clusters")
            
            for i, (pts, category) in enumerate(clusters):
                color = self.CAT_COLOR.get(category, (1.0, 1.0, 1.0, 1.0))
                name = f"cluster_{i}_{category}"
                self.addShape(PointSet3D(pts, size=self.SIZE_CLUSTER, color=color), name=name)
                self.cluster_combo.addItem(f"Cluster #{i} ({category})")
                
            self.cluster_combo.blockSignals(False)
            self.update_task_visibility()
            
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