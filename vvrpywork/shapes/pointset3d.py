from .abstract import ShapeSet
from .types import NDArray, List, List3, Tuple, ColorType, Number
from .point3d import Point3D
from .cuboid3d import Cuboid3D
from vvrpywork.scene import Scene3D, Scene3D_

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


class PointSet3D(ShapeSet):
    '''A class used to represent a set of points in 3D space.'''

    def __init__(self, points:None|NDArray|List|Tuple=None, size:Number=1, color:ColorType=(0, 0, 0)):
        '''Inits PointSet3D.

        Inits a PointSet3D containing `points`. If `points` is `None`,
        the pointset will be initialized empty.

        Args:
            points: The points of the pointset.
            size: The size of the displayed points.
            color: The color of the displayed points (RGB or RGBA).
        '''
        self._points:list[List3] = []
        self.size = size
        self._opacity = color[3] if len(color) == 4 else 1
        self._colors:list[ColorType] = []

        if points is not None:
            if isinstance(points, (np.ndarray, list, tuple)):
                self._points = [list(_) for _ in points]
                self._colors = [[*color, 1] if len(color) == 3 else [*color] for _ in points]
            else:
                raise TypeError(f"Unsupported type for points: {type(points)}")
            
    def __len__(self):
        return len(self._points)

    def __getitem__(self, idx):
        return self.getPointAt(idx)

    def _addToScene(self, scene:Scene3D, name:None|str):
        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        if len(self) == 0:
            shape = o3d.geometry.PointCloud()
        else:
            shape = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
            shape.colors = o3d.utility.Vector3dVector(self.colors[:,:3])
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 5 * self.size
        material.base_color = (1, 1, 1, self._opacity)
        scene._scene_widget.scene.add_geometry(name, shape, material)

        self._shape = shape
        self._material = material

    def _update(self, name:str, scene:Scene3D):
        # or use https://www.open3d.org/docs/latest/python_api/open3d.visualization.rendering.Scene.html#open3d.visualization.rendering.Scene.update_geometry
        scene.removeShape(name)
        scene._shapeDict[name] = self
        self._shape.points = o3d.utility.Vector3dVector(self.points) if len(self._points) > 0 else o3d.utility.Vector3dVector()
        self._shape.colors = o3d.utility.Vector3dVector(self.colors[:,:3]) if len(self._colors) > 0 else o3d.utility.Vector3dVector()
        self._material.point_size = 5 * self.size
        self._material.base_color = (1, 1, 1, self._opacity)
        scene._scene_widget.scene.add_geometry(name, self._shape, self._material)
        
    def _addToScene_PyVista(self, scene:Scene3D_, name:None|str):
        from pyvista import PolyData

        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        if len(self) == 0:
            shape = PolyData(np.array((0.0, 0.0, 0.0)))
            shape.point_data["colors"] = np.zeros((1, 4))
        else:
            shape = PolyData(self.points)
            shape.point_data["colors"] = self.colors
        self._actor = scene._plotter.add_mesh(shape, point_size=5 * self.size, name=name, scalars="colors", rgba=True)
        self._shape_pv = shape

    def _update_PyVista(self, name:str, scene:Scene3D_):
        if len(self) == 0:
            self._shape_pv.points = np.array((0.0, 0.0, 0.0))
            self._shape_pv.verts = np.array((1, 0))
            self._shape_pv.point_data["colors"] = np.zeros((1, 4))
        else:
            self._shape_pv.points = self.points
            n = len(self)
            # If the PolyData contains only points, they are considered lone vertices
            # Thus, pd.verts should still be set as [1 0  1 1  1 2  1 3  1 4 ... 1 (n-1)]
            self._shape_pv.verts = np.column_stack((np.ones(n, dtype=np.int32), np.arange(0, n))).flatten()
            self._shape_pv.point_data["colors"] = self.colors
        self._actor.GetProperty().SetPointSize(5 * self.size)

    @property
    def points(self) -> NDArray:
        '''The points of the pointset.'''
        return np.array(self._points)
    
    @points.setter
    def points(self, points:NDArray):
        if isinstance(points, (np.ndarray, list, tuple)):
            self._points = [list(_) for _ in points]

    @property
    def size(self) -> Number:
        '''The size of the displayed points.'''
        return self._size
    
    @size.setter
    def size(self, size:Number):
        try:
            size = size.item()
        except:
            pass
        finally:
            self._size = size

    @property
    def colors(self) -> NDArray:
        '''The points' colors in RGBA format.'''
        return np.array(self._colors, dtype=np.float32)
    
    @colors.setter
    def colors(self, colors:NDArray|List|Tuple):
        if isinstance(colors, (np.ndarray, list, tuple)):
            self._colors = [list(_) for _ in colors]

    def getPointAt(self, index:int) -> Point3D:
        '''Returns the point at the specified index.

        Args:
            index: The index at which the desired point is placed
                inside the pointset.
        
        Returns:
            The point at the specified index as a `Point3D` object. It
                retains its size and color.
        '''
        return Point3D(self._points[index], self.size, color=self._colors[index])
        
    def add(self, point:Point3D):
        '''Appends a point to the pointset.
        
        Args:
            point: The `Point3D` object to append.
        '''
        self._points.append([point.x, point.y, point.z])
        self._colors.append(point.color)

    def createRandom(self, bound:Cuboid3D, num_points:int, seed:None|int|str=None, color:ColorType=(0, 0, 0)):
        '''Appends random points to the pointset.

        Uniformly generates random points inside a region and appends
        them to the pointset.
        
        Args:
            bound: The area inside of which the random points will be
                generated.
            num_points: How many points to generate.
            seed: An optional seed for the RNG.
            color: The color of the generated points.
        '''
        if len(self.points) > 0:
            print('Point Set is not empty; random points will be appended to the existing ones.')

        if seed:
            if isinstance(seed, str):
                seed = seed.encode()
                seed = int.from_bytes(seed, "big") % (2<<32)
            np.random.seed(seed)

        if issubclass(type(bound), Cuboid3D):
            x1 = bound.x_min
            y1 = bound.y_min
            z1 = bound.z_min
            x2 = bound.x_max
            y2 = bound.y_max
            z2 = bound.z_max

            random_array = np.random.random_sample((num_points, 3))
            pts = random_array * np.array((x2-x1, y2-y1, z2-z1)) + np.array((x1, y1, z1))
            for p in pts:
                self._points.append(p)
                self._colors.append([*color, 1] if len(color) == 3 else [*color])

        else:
            raise TypeError
        
    def remove(self, index:int):
        '''Removes a point from the pointset.

        Removes a point from the pointset's specified index.
        
        Args:
            index: The index at which the to-be-removed point is placed
                inside the pointset.
        '''
        self._points.pop(index)
        self._colors.pop(index)
        
    def clear(self):
        '''Clears the pointset.

        Clears the pointset, completely removing all points and
        information about them (e.g., color).
        '''
        self._points.clear()
        self._colors.clear()
        
    def getAABB(self) -> Cuboid3D:
        '''Returns the AABB of the pointset.

        Returns the Axis Aligned Bounding Box of the points in the
        pointset.

        Returns:
            The AABB of the pointset.
        '''
        if len(self) > 1:
            points = self.points
            return Cuboid3D(points.min(axis=0), points.max(axis=0))
        else:
            raise RuntimeError("PointSet3D object must contain at least 2 points to define the AABB")
        
    def remove_duplicated_points(self):
        '''Removes points that exist multiple times in the pointset.'''
        try:
            self._shape.remove_duplicated_points()
        except:
            raise NotImplementedError("Currently only works after adding the pointset to a scene!")
