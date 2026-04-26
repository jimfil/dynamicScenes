import numpy as np
import heapq
from vvrpywork.shapes import Point3D, Sphere3D
   
class KdNode:
    def __init__(self, pts: np.ndarray, depth: int):

        if len(pts) < 1:
            return
        
        # Select the axis to do the split
        axis = depth % 3 

        # Find the median of the points along the selected axis (pivot)
        indices = np.argsort(pts[:, axis])
        sorted_pts = pts[indices]

        median_idx = len(sorted_pts) // 2 
        pts_left = sorted_pts[:median_idx]
        pts_right = sorted_pts[(median_idx + 1):]


        self.pivot = sorted_pts[median_idx]
        self.depth = depth
        self.left_child = KdNode(pts_left, depth + 1) if len(pts_left) > 0 else None
        self.right_child = KdNode(pts_right, depth + 1) if len(pts_right) > 0 else None

    def __repr__(self) -> str:
        return f"k-d node @ {self.pivot}, depth = {self.depth}"
    
    @staticmethod
    def getMaxDepth(node: 'KdNode') -> int:

        if node.left_child is None and node.right_child is None:
            return node.depth
        elif node.right_child is None:
            return KdNode.getMaxDepth(node.left_child)
        elif node.left_child is None:
            return KdNode.getMaxDepth(node.right_child)
        else:
            return max(KdNode.getMaxDepth(node.left_child), KdNode.getMaxDepth(node.right_child))  

    
    @staticmethod
    def getNodesBelow(node: 'KdNode') -> np.ndarray:

        def _getNodesBelow(node: 'KdNode', pts: list):

            if node.left_child is not None:
                pts.append(node.left_child.pivot)
                pts = _getNodesBelow(node.left_child, pts)

            if node.right_child is not None:
                pts.append(node.right_child.pivot)
                pts = _getNodesBelow(node.right_child, pts)

            return pts

        pts = _getNodesBelow(node, [])
        return np.array(pts)
    
    @staticmethod
    def getNodesAtDepth(node: 'KdNode', depth: int) -> list:

        def _getNodesAtDepth(node: 'KdNode', depth: int, pts: list):

            if node.depth == depth:
                pts.append(node)
                return pts

            if node.left_child is not None:
                pts = _getNodesAtDepth(node.left_child,depth)

            if node.right_child is not None:
                pts = _getNodesAtDepth(node.right_child,depth)

            return pts

        pts = _getNodesAtDepth(node, depth, [])
        return pts
    
    @staticmethod
    def inSphere(sphere: Sphere3D, node: 'KdNode') -> np.ndarray:
        
        def _inSphere(sphere: Sphere3D, node: 'KdNode', pts: list):

            d_sq = (sphere.x - node.pivot[0]) ** 2 + (sphere.y - node.pivot[1]) ** 2 + (sphere.z - node.pivot[2]) ** 2
            if d_sq <= sphere.radius ** 2:
                pts.append(node.pivot)

            axis = node.depth % 3
            d_pivot = (node.pivot - (sphere.x, sphere.y, sphere.z))[axis]

            if d_pivot <= 0: # The sphere center is on the right side of the pivot
                check_first = node.right_child
                check_second = node.left_child
            else:
                check_first = node.left_child
                check_second = node.right_child
            
            if check_first is not None:
                pts = _inSphere(sphere, check_first, pts)

            # Only search the farther side if the sphere crosses the splitting plane.
            if abs(d_pivot) >= sphere.radius:
                return pts
            
            if check_second is not None:
                pts = _inSphere(sphere, check_second, pts)

            return pts

        pts = _inSphere(sphere, node, [])
        return np.array(pts)
    
    @staticmethod
    def nearestNeighbor(test_pt:Point3D, node:'KdNode') -> 'KdNode':

        def _nearestNeighbor(test_pt:Point3D, node:'KdNode', nn:'KdNode', min_dist_sq:float):

            d_sq = (test_pt.x - node.pivot[0]) ** 2 + (test_pt.y - node.pivot[1]) ** 2 + (test_pt.z - node.pivot[2]) ** 2

            if d_sq < min_dist_sq:
                nn = node
                min_dist_sq = d_sq

            # Check which side of the splitting plane the test point is on
            d_pivot = (node.pivot - (test_pt.x, test_pt.y, test_pt.z))[node.depth % 3]

            if d_pivot <= 0:
                check_first = node.right_child
                check_second = node.left_child
            else:
                check_first = node.left_child
                check_second = node.right_child
            
            if check_first is not None:
                nn, min_dist_sq = _nearestNeighbor(test_pt, check_first, nn, min_dist_sq)

            if d_pivot ** 2 >= min_dist_sq:
                return nn, min_dist_sq

            if check_second is not None:
                nn, min_dist_sq = _nearestNeighbor(test_pt, check_second, nn, min_dist_sq)

            return nn, min_dist_sq

        nn, _ = _nearestNeighbor(test_pt, node, None, np.inf)
        return nn
    
    @staticmethod
    def nearestK(test_pt:Point3D, node:'KdNode', k:int) -> list['KdNode']:

        def _nearestK(test_pt:Point3D, node:'KdNode', k:int, heap:list[tuple[int,'KdNode']], d_threshold:float):

            d_sq = (test_pt.x - node.pivot[0]) ** 2 + (test_pt.y - node.pivot[1]) ** 2 + (test_pt.z - node.pivot[2]) ** 2
            
            if d_sq < d_threshold:
                if len(heap) < k:
                    # NOTE: heapq uses min heap, so we negate the distance to get max heap
                    heapq.heappush(heap, (-d_sq, node))
                else:
                    heapq.heapreplace(heap, (-d_sq, node))
                
                # If the heap is full, calculate the new threshold
                if len(heap) == k: 
                    d_threshold = -heap[0][0]

            d_pivot = (node.pivot - (test_pt.x, test_pt.y, test_pt.z))[node.depth % 3]
                
            if d_pivot <= 0:
                check_first = node.right_child
                check_second = node.left_child
            else:
                check_first = node.left_child
                check_second = node.right_child

            if check_first is not None:
                heap, d_threshold = _nearestK(test_pt, check_first, k, heap, d_threshold)

            if d_pivot ** 2 >= d_threshold:
                return heap, d_threshold
            
            if check_second is not None:
                heap, d_threshold = _nearestK(test_pt, check_second, k, heap, d_threshold)

            return heap, d_threshold
        
        if k == 0:
            return []

        heap, _ = _nearestK(test_pt, node, k, [], np.inf)
        return [n for _, n in heap]

        