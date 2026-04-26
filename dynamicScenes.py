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

WIDTH = 1000
HEIGHT = 800

COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.MAGENTA, Color.YELLOWGREEN, Color.CYAN]

haveCsv = False

class DynamicScenes(Scene3D_):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "DynamicScenes", output=True, n_sliders=4)
        self.compareScenes("0_5D4KVPBP.laz", "0_WE1NZ71I.laz")
        #self.setup()
    
    #0_5D4KVPBP
    #0_WE1NZ71I

    def getPoints(self, filename, year, remove_ground=True):
        pts = rl.readpoints(filename, year=year)
        pts = np.array(pts, dtype=np.float32)
        if remove_ground:
            ground_z = np.percentile(pts[:, 2], 5) 
            pts = pts[pts[:, 2] > ground_z + 0.5]
        pts = pts[:, [0, 2, 1]]

        return pts
        
    def compareScenes(self, filename2016, filename2020):
        print("Loading scenes...")
        # Load points 
        pts2016 = self.getPoints(filename2016, 2016, remove_ground=True)
        pts2020 = self.getPoints(filename2020, 2020, remove_ground=True)

        # Downsampling
        pts2016 = pts2016[::25]
        pts2020 = pts2020[::25]

        print(f"Building KD-Tree with {len(pts2016)} points from 2016...")
        tree2016 = kdN.KdNode(pts2016, depth=0)

        print("Comparing 2020 points against 2016 KD-Tree...")
        threshold_sq = 1.0 ** 2 # Points further than 1 meter are considered "changed" / "dynamic"
        
        is_new = []
        for p in pts2020:
            test_pt = Point3D([float(p[0]), float(p[1]), float(p[2])])
            nn = kdN.KdNode.nearestNeighbor(test_pt, tree2016)
            
            d_sq = (p[0] - nn.pivot[0])**2 + (p[1] - nn.pivot[1])**2 + (p[2] - nn.pivot[2])**2
            is_new.append(d_sq > threshold_sq)

        is_new = np.array(is_new)
        new_pts = pts2020[is_new]
        static_pts = pts2020[~is_new]

        print(f"Found {len(new_pts)} dynamic points and {len(static_pts)} static points.")

        # Center and scale based on the entire 2020 scene to keep alignment intact
        center = (pts2020.min(axis=0) + pts2020.max(axis=0)) / 2.0
        distances = np.linalg.norm(pts2020 - center, axis=1)
        max_dist = np.percentile(distances, 99)
        
        def scale(pts):
            if max_dist > 0:
                return np.clip(((pts - center) / max_dist) * 10.0, -10.0, 10.0)
            return pts - center

        static_pts_scaled = scale(static_pts)
        new_pts_scaled = scale(new_pts)

        # Visualize static points normally
        if len(static_pts_scaled) > 0:
            colors = self.colorPointsByHeight(static_pts_scaled)
            ps_static = PointSet3D(static_pts_scaled, size=1)
            ps_static.colors = colors.tolist()
            self.addShape(ps_static, name="Static Points")

        # Visualize dynamic/new points in bold red
        if len(new_pts_scaled) > 0:
            ps_new = PointSet3D(new_pts_scaled, size=3)
            ps_new.color = (1.0, 0.0, 0.0, 1.0) # Solid red
            self.addShape(ps_new, name="Dynamic Points")
        
        print("Comparison rendering complete!")

    def centerAndNormalize(self, pts):
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        pts_centered = pts - center
        
        distances = np.linalg.norm(pts_centered, axis=1)
        max_dist = np.percentile(distances, 99)
        
        if max_dist > 0:
            pts_scaled = np.clip((pts_centered / max_dist) * 10.0, -10.0, 10.0)
        else:
            pts_scaled = pts_centered
        return pts_scaled

    def colorPointsByHeight(self, pts):
        height_vals = pts[:, 1]
        h_min, h_max = float(height_vals.min()), float(height_vals.max())
        
        if h_max == h_min:
            znorm = np.zeros_like(height_vals, dtype=np.float32)
        else:
            znorm = ((height_vals - h_min) / (h_max - h_min)).astype(np.float32)

        colors = np.column_stack((znorm, 0.2 * np.ones_like(znorm), 1.0 - znorm, np.ones_like(znorm)))
        return colors


    def setup(self):
        
        
        pts2016 = self.getPoints("0_5D4KVPBP.laz", 2016, remove_ground=True) #2016
        #pts = self.getPoints("0_WE1NZ71I.laz", 2020) #2020

        pts2016 = self.centerAndNormalize(pts2016)
        colors2016 = self.colorPointsByHeight(pts2016)
        
        ps = PointSet3D(pts2016, size=1, color=(1, 1, 1, 1)) 
        ps.colors = colors2016.tolist()
        self.addShape(ps, name="points2016")


if __name__ == "__main__":
    app = DynamicScenes()
    app.mainLoop()

    