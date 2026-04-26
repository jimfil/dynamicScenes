from .abstract import ShapeSet
from .types import NDArray, List, List2, List3, Tuple, ColorType, Number
from .pointset3d import PointSet3D
from .line3d import Line3D
from .mesh3d import Mesh3D
from vvrpywork.scene import Scene3D, Scene3D_

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


class LineSet3D(ShapeSet):
    '''A class used to represent a set of lines in 3D space.'''

    def __init__(self, points:None|NDArray|List|Tuple=None, lines:None|NDArray|List|Tuple=None, width:Number=1, color:ColorType=(0, 0, 0)):
        '''Inits LineSet3D.

        Inits a LineSet3D containing `points` connected according to
        `lines`.
        
        If `lines` is `None`, the `points` will be connected in pairs
        i.e., (0, 1), (2, 3), etc.
        
        If `points` is `None`, the lineset will be initialized empty.

        Args:
            points: The points of the lineset.
            lines: The indices in `points` that are connected by a
                line.
            width: The width of the displayed lines.
            color: The color of the displayed lines (RGB or RGBA).
        '''
        self._points:list[List3] = []
        self._lines:list[List2] = []

        self.width = width
        self._opacity = color[3] if len(color) == 4 else 1
        self._colors:list[ColorType] = []

        if isinstance(points, PointSet3D):
            points = points.points

        if points is not None and lines is None:
            if isinstance(points, (np.ndarray, list, tuple)) and len(points) % 2 == 0:
                lines = [[i,i+1] for i in range(0, len(points), 2)]

        if points is not None and lines is not None:
            if isinstance(points, (np.ndarray, list, tuple)):
                self._points = [list(_) for _ in points]
            if isinstance(lines, (np.ndarray, list, tuple)):
                self._lines = [list(_) for _ in lines]

            self._colors = [[*color, 1] if len(color) == 3 else [*color] for _ in lines]
    
    def __len__(self):
        return len(self._lines)
    
    def __getitem__(self, idx):
        return self.getLineAt(idx)

    def _addToScene(self, scene:Scene3D, name:None|str):
        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        if len(self) == 0:
            shape = o3d.geometry.LineSet()
        else:
            shape = o3d.geometry.LineSet(o3d.utility.Vector3dVector(self.points), o3d.utility.Vector2iVector(self.lines))
            shape.colors = o3d.utility.Vector3dVector(self.colors[:,:3])
        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 2 * self.width
        material.base_color = (1, 1, 1, self._opacity)
        scene._scene_widget.scene.add_geometry(name, shape, material)

        self._shape = shape
        self._material = material

    def _update(self, name:str, scene:Scene3D):
        # unfortunately, open3d does not support updating linesets yet; do it the ol' fashioned way
        scene.removeShape(name)
        scene._shapeDict[name] = self
        if len(self) == 0:
            self._shape.clear()
        else:
            self._shape.lines = o3d.utility.Vector2iVector(self.lines)
            self._shape.points = o3d.utility.Vector3dVector(self.points)
            self._shape.colors = o3d.utility.Vector3dVector(self.colors[:,:3])
        self._material.line_width = 2 * self.width
        self._material.base_color = (1, 1, 1, self._opacity)
        scene._scene_widget.scene.add_geometry(name, self._shape, self._material)

    def _addToScene_PyVista(self, scene:Scene3D_, name:None|str):
        from pyvista import PolyData

        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        if len(self) == 0:
            shape = PolyData(np.array(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))), lines=np.array((2, 0, 1)))
            shape.cell_data["colors"] = np.zeros((1, 4))
        else:
            points = self.points
            lines = self.lines
            lines = np.hstack([np.full((lines.shape[0], 1), 2), lines])
            shape = PolyData(points, lines=lines)
            shape.cell_data["colors"] = self.colors
        self._actor = scene._plotter.add_mesh(shape, line_width=2 * self.width, name=name, scalars="colors", rgba=True)
        self._shape_pv = shape

    def _update_PyVista(self, name:str, scene:Scene3D_):
        if len(self) == 0:
            self._shape_pv.points = np.array(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
            self._shape_pv.lines = np.array((2, 0, 1))
            self._shape_pv.cell_data["colors"] = np.zeros((1, 4))
        else:
            self._shape_pv.points = self.points
            lines = self.lines
            self._shape_pv.lines = np.hstack([np.full((lines.shape[0], 1), 2), lines])
            self._shape_pv.cell_data["colors"] = self.colors
        self._actor.GetProperty().SetLineWidth(2 * self.width)

    @property
    def points(self) -> NDArray:
        '''The points of the lineset.'''
        return np.array(self._points)
    
    @points.setter
    def points(self, points:NDArray|List|Tuple):
        if isinstance(points, (np.ndarray, list, tuple)):
            self._points = [list(_) for _ in points]

    @property
    def lines(self) -> NDArray:
        '''The point indices indicating lines of the lineset.'''
        return np.array(self._lines)
    
    @lines.setter
    def lines(self, lines:NDArray|List|Tuple):
        if isinstance(lines, (np.ndarray, list, tuple)):
            self._lines = [list(_) for _ in lines]

    @property
    def width(self) -> Number:
        '''The width of the displayed lines.'''
        return self._width
    
    @width.setter
    def width(self, width:Number):
        try:
            width = width.item()
        except:
            pass
        finally:
            self._width = width

    @property
    def colors(self) -> NDArray:
        '''The lines' colors in RGBA format.'''
        return np.array(self._colors, dtype=np.float32)
    
    @colors.setter
    def colors(self, colors:NDArray|List|Tuple):
        if isinstance(colors, (np.ndarray, list, tuple)):
            self._colors = [list(_) for _ in colors]

    def getLineAt(self, index:int) -> Line3D:
        '''Returns the line at the specified index.

        Args:
            index: The index at which the desired line is placed
                inside the lineset.
        
        Returns:
            The line at the specified index as a `Line3D` object. It
                retains its width and color.
        '''
        return Line3D(self._points[self._lines[index][0]], self._points[self._lines[index][1]], self.width, color=self.colors[index])

    def add(self, line:Line3D):
        '''Appends a line to the lineset.
        
        Args:
            line: The `Line3D` object to append.
        '''
        idx1 = idx2 = None
        if len(self._points) == 0:
            self._points.append([line.x1, line.y1, line.z1])

        for i, p in enumerate(self._points):
            if p[0] == line.x1 and p[1] == line.y1 and p[2] == line.z1:
                idx1 = i
            if p[0] == line.x2 and p[1] == line.y2 and p[2] == line.z2:
                idx2 = i

        if idx1 == None:
            self._points.append([line.x1, line.y1, line.z1])
            idx1 = len(self._points) - 1
        if idx2 == None:
            self._points.append([line.x2, line.y2, line.z2])
            idx2 = len(self._points) - 1

        self._lines.append([idx1, idx2])
        self._colors.append([*line.color])

    def remove(self, index:int):
        '''Removes a line from the lineset.

        Removes a line from the lineset's specified index (does not
        affect LineSet3D.points in any way).
        
        Args:
            index: The index at which the to-be-removed line is placed
                inside the lineset.
        '''
        self._lines.pop(index)
        self._colors.pop(index)

    def clear(self):
        '''Clears the lineset.

        Clears the lineset, completely removing all points, lines and
        information about them (e.g., color).
        '''
        self._points.clear()
        self._lines.clear()
        self._colors.clear()

    @staticmethod
    def create_from_mesh(mesh:Mesh3D, width:Number=1, color:ColorType=(0, 0, 0)) -> "LineSet3D":
        '''Creates a Lineset3D object from a Mesh3D object.

        Args:
            mesh: The mesh to be turned into a lineset.
            width: The width of the displayed lines.
            color: The color of the displayed lines (RGB or RGBA).

        Returns:
            The Lineset3D object extracted from the mesh.
        '''
        tm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.triangles))
        ls = o3d.geometry.LineSet.create_from_triangle_mesh(tm)
        return LineSet3D(np.asarray(ls.points), np.asarray(ls.lines), width, color)
