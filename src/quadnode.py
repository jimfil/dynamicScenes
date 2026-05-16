import numpy as np

class QuadNode:
    def __init__(self, pts: np.ndarray, center: np.ndarray, size: float, depth: int, max_depth: int = 6, capacity: int = 50):
        # center is a 2D array: [x, y]
        self.center = center
        self.size = size
        self.depth = depth
        self.children = []
        
        # Base case
        if len(pts) <= capacity or depth >= max_depth:
            self.is_leaf = True
            self.points = pts
            return
            
        self.is_leaf = False
        self.points = None 
        
        # Split points based only on X and Y
        x_mask = (pts[:, 0] >= center[0]).astype(int)
        y_mask = (pts[:, 1] >= center[1]).astype(int)
        
        # 0 to 3 for the 4 quadrants
        quad_indices = x_mask * 1 + y_mask * 2
        
        new_size = size / 2.0
        
        # 4 directional multipliers for the new 2D centers
        offsets = np.array([
            [-1, -1], [ 1, -1], [-1,  1], [ 1,  1]
        ])
        
        # Create the 4 children
        for i in range(4):
            child_pts = pts[quad_indices == i]
            child_center = center + (offsets[i] * new_size)
            child = QuadNode(child_pts, child_center, new_size, depth + 1, max_depth, capacity)
            self.children.append(child)

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"QuadLeaf @ {self.center}, depth = {self.depth}, pts = {len(self.points)}"
        return f"QuadNode @ {self.center}, depth = {self.depth}"