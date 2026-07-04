import csv
from pathlib import Path
import numpy as np
import src.readpcdfiles as rp
import src.readlazfiles as rl
from src.pipeline import remove_reflections, remove_ground_grid
from src.geometry_utils import icp_align, pca_transform, apply_transform, scale

class PCDSequenceLoader:
    def __init__(self, csv_path: str = "Data/pcd/calibration.csv", data_dir: str = "Data/pcd/velodyne_points", cache_limit: int = 30):
        self.csv_path = Path(csv_path)
        self.data_dir = Path(data_dir)
        self.cache_limit = cache_limit
        self.calibration_data = {}
        self.cache = {}
        self.cache_order = []
        self.load_calibration()
        
    def load_calibration(self):
        if not self.csv_path.exists():
            print(f"Error: Calibration file not found at {self.csv_path}")
            return
        with open(self.csv_path, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader) # skip header
            for row in reader:
                if len(row) < 18:
                    continue
                idx = int(row[0])
                filename = row[1]
                m = np.array([float(val) for val in row[2:18]], dtype=np.float32).reshape(4, 4)
                self.calibration_data[idx] = (filename, m)
                
    def get_num_frames(self) -> int:
        return len(self.calibration_data)
        
    def load_raw_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Loads and returns the preprocessed point cloud and transformation matrix T."""
        if index in self.cache:
            self.cache_order.remove(index)
            self.cache_order.append(index)
            return self.cache[index]
            
        if index not in self.calibration_data:
            raise KeyError(f"Frame index {index} not found in calibration data.")
            
        filename, T = self.calibration_data[index]
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"PCD file not found at {filepath}")
            
        pts = rp.read_pcd_points(str(filepath))
        
        # Preprocess frame points
        # 1. Filter out ego-reflections
        pts = remove_reflections(pts, threshold=2.0)
        
        # 2. Homogeneous transformation (local sensor frame -> global map frame)
        if len(pts) > 0:
            pts_h = np.hstack((pts, np.ones((len(pts), 1), dtype=np.float32)))
            pts_g = (pts_h @ T.T)[:, :3]
        else:
            pts_g = np.empty((0, 3), dtype=np.float32)
            
        # 3. Remove ground (while in [X, Y, Z] global map format)
        pts_clean = remove_ground_grid(pts_g, grid_size=5.0, z_threshold=0.4)
        
        # 4. Coordinate axis reordering to [X, Height, Y]
        if len(pts_clean) > 0:
            pts_reordered = pts_clean[:, [0, 2, 1]]
        else:
            pts_reordered = np.empty((0, 3), dtype=np.float32)
        
        # Cache management
        if len(self.cache) >= self.cache_limit:
            oldest_key = self.cache_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
            
        self.cache[index] = (pts_reordered, T)
        self.cache_order.append(index)
        return pts_reordered, T

class LIDARSequenceLoader:
    def __init__(self, file2016="Data/laz/2016.laz", file2020="Data/laz/2020.laz", crop_percentage=80, downsample_step=30):
        self.file2016 = file2016
        self.file2020 = file2020
        self.crop_percentage = crop_percentage
        self.downsample_step = downsample_step
        self.cache = {}
        self.prealign_data()
        
    def prealign_data(self):
        print("LIDAR: Pre-loading and aligning epochs (once)...")
        # 1. Load raw datasets
        pts1 = rl.readpoints(self.file2016)
        pts2 = rl.readpoints(self.file2020)
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)
        
        # 2. Ground grid removal
        pts1_clean = remove_ground_grid(pts1, grid_size=5.0, z_threshold=0.4)
        pts2_clean = remove_ground_grid(pts2, grid_size=5.0, z_threshold=0.4)
        
        # 3. Crop based on crop percentage
        def crop_points(pts):
            if len(pts) == 0:
                return pts
            x_axis = np.percentile(pts[:, 0], self.crop_percentage)
            pts = pts[pts[:, 0] < x_axis]
            if len(pts) == 0:
                return pts
            y_axis = np.percentile(pts[:, 1], self.crop_percentage)
            return pts[pts[:, 1] < y_axis]
            
        pts1_cropped = crop_points(pts1_clean)
        pts2_cropped = crop_points(pts2_clean)
        
        # 4. Coordinate axis reordering to [X, Height, Y]
        pts1_reordered = pts1_cropped[:, [0, 2, 1]] if len(pts1_cropped) > 0 else np.empty((0, 3), dtype=np.float32)
        pts2_reordered = pts2_cropped[:, [0, 2, 1]] if len(pts2_cropped) > 0 else np.empty((0, 3), dtype=np.float32)
        
        # 5. Downsample
        step = self.downsample_step
        pts1_ds = pts1_reordered[::step]
        pts2_ds = pts2_reordered[::step]
        
        # 6. SVD-based ICP alignment
        pts1_aligned = icp_align(pts1_ds, pts2_ds, max_iterations=20, tolerance=1e-4)
        
        # 7. PCA transform
        mean, R = pca_transform(pts2_ds)
        pts2_al = apply_transform(pts2_ds, mean, R)
        pts1_al = apply_transform(pts1_aligned, mean, R)
        
        # 8. Scale to fit within standard visualizer viewport limits [-10, 10]
        all_pts = np.vstack((pts1_al, pts2_al))
        g_center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2.0
        g_maxd = np.percentile(np.linalg.norm(all_pts - g_center, axis=1), 99)
        
        pts1_final = scale(pts1_al, g_center, g_maxd)
        pts2_final = scale(pts2_al, g_center, g_maxd)
        
        self.cache[1] = (pts1_final, np.eye(4, dtype=np.float32))
        self.cache[2] = (pts2_final, np.eye(4, dtype=np.float32))
        print("LIDAR: Pre-alignment complete.")
        
    def get_num_frames(self) -> int:
        return 2
        
    def load_raw_frame(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.cache[index]


