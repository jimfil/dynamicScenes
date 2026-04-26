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


WIDTH = 1000
HEIGHT = 800

COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.MAGENTA, Color.YELLOWGREEN, Color.CYAN]

haveCsv = False

class DynamicScenes(Scene3D_):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "DynamicScenes", output=True, n_sliders=4)
        self.setup()
   

    def opencsv(self):
        if haveCsv:
            base = Path(__file__).resolve().parent
            csv_path = base / "Data" / "output_scene1.csv"
            if not csv_path.exists():
                self.print(f"CSV not found: {csv_path}")
                return

            pts = []
            with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    nums = []
                    for v in row:
                        s = v.strip()
                        if s == "":
                            continue
                        try:
                            nums.append(float(s))
                        except:
                            # skip non-numeric tokens
                            pass
                    if not nums:
                        continue
                    if len(nums) >= 3:
                        pts.append(nums[:3])
                    if len(pts) >= 2000000:
                        break
        return pts
        
    def setup(self):
        if haveCsv:
            pts = self.opencsv()
        else:
            pts = rl.readpoints("0_WE1NZ71I.laz", year= 2020, test=True)
            #0_5D4KVPBP
            #0_WE1NZ71I
        if len(pts) == 0:
            self.print(f"No numeric points parsed from {csv_path}")
            return

        pts = np.array(pts, dtype=np.float32)
        # remove lowest points (ground) using a robust percentile to ignore outliers
        # this provides a stable ground elevation so you don't have to tune the offset per file
        ground_z = np.percentile(pts[:, 2], 5) 
        pts = pts[pts[:, 2] > ground_z + 0.5]


        # every 20 points keep 1 
        #pts = pts[::10]

        pts = pts[:, [0, 2, 1]]

        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        pts_centered = pts - center
        
        distances = np.linalg.norm(pts_centered, axis=1)
        max_dist = np.percentile(distances, 99)
        
        if max_dist > 0:
            pts_scaled = np.clip((pts_centered / max_dist) * 10.0, -10.0, 10.0)
        else:
            pts_scaled = pts_centered

        height_vals = pts_scaled[:, 1]
        h_min, h_max = float(height_vals.min()), float(height_vals.max())
        
        if h_max == h_min:
            znorm = np.zeros_like(height_vals, dtype=np.float32)
        else:
            znorm = ((height_vals - h_min) / (h_max - h_min)).astype(np.float32)

        colors = np.column_stack((znorm, 0.2 * np.ones_like(znorm), 1.0 - znorm, np.ones_like(znorm)))

        ps = PointSet3D(pts_scaled, size=1, color=(1, 1, 1, 1)) 
        ps.colors = colors.tolist()
        self.addShape(ps, name="scene1_points")


if __name__ == "__main__":
    app = DynamicScenes()
    app.mainLoop()

    