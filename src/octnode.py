import numpy as np

class OctNode:
    def __init__(self, pts: np.ndarray, center: np.ndarray, size: float, depth: int, max_depth: int = 5, capacity: int = 50):
        self.center = center
        self.size = size
        self.depth = depth
        self.children = []
        
        # Base case: If we have few enough points, or we hit the depth limit, become a Leaf Node.
        if len(pts) <= capacity or depth >= max_depth:
            self.is_leaf = True
            self.points = pts
            return
            
        self.is_leaf = False
        self.points = None # Only leaf nodes hold points
        
        # To split the points instantly using numpy, we check if they are greater than the center.
        # This creates a boolean mask. We multiply by 1, 2, and 4 to create a unique ID (0 to 7) for each octant.
        x_mask = (pts[:, 0] >= center[0]).astype(int)
        y_mask = (pts[:, 1] >= center[1]).astype(int)
        z_mask = (pts[:, 2] >= center[2]).astype(int)
        
        octant_indices = x_mask * 1 + y_mask * 2 + z_mask * 4
        
        new_size = size / 2.0
        
        # The 8 directional multipliers to calculate the new centers
        offsets = np.array([
            [-1, -1, -1], [ 1, -1, -1], [-1,  1, -1], [ 1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [-1,  1,  1], [ 1,  1,  1]
        ])
        
        # Create the 8 children
        for i in range(8):
            # Slice the numpy array instantly to get only the points for this octant
            child_pts = pts[octant_indices == i]
            
            # Calculate the physical center of the new smaller box
            child_center = center + (offsets[i] * new_size)
            
            # Recursively create the child node
            child = OctNode(child_pts, child_center, new_size, depth + 1, max_depth, capacity)
            self.children.append(child)

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"OctLeaf @ {self.center}, depth = {self.depth}, pts = {len(self.points)}"
        return f"OctNode @ {self.center}, depth = {self.depth}"

    @staticmethod
    def getMaxDepth(node: 'OctNode') -> int:
        if node.is_leaf:
            return node.depth
        return max(OctNode.getMaxDepth(child) for child in node.children)

    @staticmethod
    def getLeafNodes(node: 'OctNode') -> list:
        """Returns all leaf nodes (boxes that actually contain points)."""
        def _getLeafNodes(node: 'OctNode', leaves: list):
            if node.is_leaf:
                if len(node.points) > 0: # Only return boxes that aren't empty
                    leaves.append(node)
                return leaves
            for child in node.children:
                leaves = _getLeafNodes(child, leaves)
            return leaves
            
        return _getLeafNodes(node, [])