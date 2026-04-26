from .abstract import Shape
from .types import NDArray3, List3, Tuple3, ColorType, Number
from .point3d import Point3D
from vvrpywork.scene import Scene3D, Scene3D_

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


class Cuboid3D(Shape):
    '''A class used to represent a cuboid in 3D space.
    
    A class used to represent a cuboid in 3D space. The cuboid has to
    be axis-aligned. If you need a cuboid that supports rotation, use
    the more flexible (but less robust) `Cuboid3DGeneralized`.
    '''

    def __init__(self, p1:Point3D|NDArray3|List3|Tuple3, p2:Point3D|NDArray3|List3|Tuple3, width:Number=1, color:ColorType=(0, 0, 0), filled:bool=False):
        '''Inits Cuboid3D given 2 vertices of the cuboid.

        Args:
            p1: The coordinates of one vertex.
            p2: The coordinates of another vertex.
            width: The width of the displayed cuboid (if not filled).
            color: The color of the displayed cuboid (RGB or RGBA).
            filled: Whether to fill in the cuboid or draw only its
                outline.
        '''
        
        if isinstance(p1, Point3D):
            x1 = p1.x
            y1 = p1.y
            z1 = p1.z
        elif isinstance(p1, (list, tuple)):
            x1 = p1[0]
            y1 = p1[1]
            z1 = p1[2]
        elif isinstance(p1, np.ndarray):
            x1 = p1[0].item()
            y1 = p1[1].item()
            z1 = p1[2].item()
        else:
            raise TypeError("Incorrect type for p1")
        
        if isinstance(p2, Point3D):
            x2 = p2.x
            y2 = p2.y
            z2 = p2.z
        elif isinstance(p2, (list, tuple)):
            x2 = p2[0]
            y2 = p2[1]
            z2 = p2[2]
        elif isinstance(p2, np.ndarray):
            x2 = p2[0].item()
            y2 = p2[1].item()
            z2 = p2[2].item()
        else:
            raise TypeError("Incorrect type for p2")
        
        self._x_min = min(x1, x2)
        self._y_min = min(y1, y2)
        self._z_min = min(z1, z2)
        self._x_max = max(x1, x2)
        self._y_max = max(y1, y2)
        self._z_max = max(z1, z2)

        self.width = width
        self._color = [*color, 1] if len(color) == 3 else [*color]
        self._filled = filled

    def _addToScene(self, scene:Scene3D, name:None|str):
        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        if self.filled:
            shape = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
            shape.translate((-0.5, -0.5, -0.5))
            shape.compute_vertex_normals()
        else:
            vertices = np.array(((-0.5, -0.5, -0.5),
                                 (0.5, -0.5, -0.5),
                                 (0.5, 0.5, -0.5),
                                 (-0.5, 0.5, -0.5),
                                 (-0.5, -0.5, 0.5),
                                 (0.5, -0.5, 0.5),
                                 (0.5, 0.5, 0.5),
                                 (-0.5, 0.5, 0.5)))
            lines = np.array(((0, 1), (1, 2), (2, 3), (3, 0),
                              (4, 5), (5, 6), (6, 7), (7, 4),
                              (0, 4), (1, 5), (2, 6), (3, 7)))
            shape = o3d.geometry.LineSet(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(lines))
            material.shader = "unlitLine"
            material.line_width = 2 * self.width
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        material.base_color = color
        scene._scene_widget.scene.add_geometry(name, shape, material)
        scene._scene_widget.scene.set_geometry_transform(name, ((self._x_max - self._x_min, 0, 0, (self.x_max + self.x_min)/2), (0, self._y_max - self._y_min, 0, (self.y_max + self.y_min)/2), (0, 0, self._z_max - self._z_min, (self.z_max + self.z_min)/2), (0, 0, 0, 1)))

    def _update(self, name:str, scene:Scene3D):
        scene._scene_widget.scene.set_geometry_transform(name, ((self._x_max - self._x_min, 0, 0, (self.x_max + self.x_min)/2), (0, self._y_max - self._y_min, 0, (self.y_max + self.y_min)/2), (0, 0, self._z_max - self._z_min, (self.z_max + self.z_min)/2), (0, 0, 0, 1)))
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        material = rendering.MaterialRecord()
        if self.filled:
            material.shader = "defaultLitTransparency"
        else:
            material.shader = "unlitLine"
            material.line_width = 2 * self.width
        material.base_color = color
        scene._scene_widget.scene.modify_geometry_material(name, material)

    def _addToScene_PyVista(self, scene:Scene3D_, name:None|str):
        from pyvista import Cube
        from vtk import vtkMatrix4x4

        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        shape = Cube() # default is fine
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        if self.filled:
            self._actor = scene._plotter.add_mesh(shape, color=tuple(int(255 * _ + 0.5) for _ in color[:3]), opacity=0, name=name, smooth_shading=True, split_sharp_edges=True)
        else:
            self._actor = scene._plotter.add_mesh(shape, color=tuple(int(255 * _ + 0.5) for _ in color[:3]), opacity=0, line_width=self.width, name=name, style="wireframe")
        transform = np.array(((self._x_max - self._x_min, 0, 0, (self.x_max + self.x_min)/2), (0, self._y_max - self._y_min, 0, (self.y_max + self.y_min)/2), (0, 0, self._z_max - self._z_min, (self.z_max + self.z_min)/2), (0, 0, 0, 1)))
        mat = vtkMatrix4x4()
        mat.DeepCopy(transform.ravel())
        self._actor.SetUserMatrix(mat)
        self._actor.GetProperty().SetOpacity(color[3])

    def _update_PyVista(self, name:str, scene:Scene3D_):
        from vtk import vtkMatrix4x4
        
        transform = np.array(((self._x_max - self._x_min, 0, 0, (self.x_max + self.x_min)/2), (0, self._y_max - self._y_min, 0, (self.y_max + self.y_min)/2), (0, 0, self._z_max - self._z_min, (self.z_max + self.z_min)/2), (0, 0, 0, 1)))
        mat = vtkMatrix4x4()
        mat.DeepCopy(transform.ravel())
        self._actor.SetUserMatrix(mat)
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        self._actor.GetProperty().SetColor(color[0], color[1], color[2])
        self._actor.GetProperty().SetOpacity(color[3])
        if not self.filled:
            self._actor.GetProperty().SetLineWidth(self.width)

    @property
    def x_min(self) -> Number:
        '''The x-coordinate of the bottom-left-back vertex.'''
        return self._x_min
    
    @x_min.setter
    def x_min(self, x:Number):
        try:
            x = x.item()
        except:
            pass
        finally:
            self._x_min = x
            if self._x_min > self._x_max:
                self._x_min, self._x_max = self._x_max, self._x_min

    @property
    def y_min(self) -> Number:
        '''The y-coordinate of the bottom-left-back vertex.'''
        return self._y_min
    
    @y_min.setter
    def y_min(self, y:Number):
        try:
            y = y.item()
        except:
            pass
        finally:
            self._y_min = y
            if self._y_min > self._y_max:
                self._y_min, self._y_max = self._y_max, self._y_min

    @property
    def z_min(self) -> Number:
        '''The z-coordinate of the bottom-left-back vertex.'''
        return self._z_min
    
    @z_min.setter
    def z_min(self, z:Number):
        try:
            z = z.item()
        except:
            pass
        finally:
            self._z_min = z
            if self._z_min > self._z_max:
                self._z_min, self._z_max = self._z_max, self._z_min
    
    @property
    def x_max(self) -> Number:
        '''The x-coordinate of the top-right-front vertex.'''
        return self._x_max
    
    @x_max.setter
    def x_max(self, x:Number):
        try:
            x = x.item()
        except:
            pass
        finally:
            self._x_max = x
            if self._x_max < self._x_min:
                self._x_max, self._x_min = self._x_min, self._x_max

    @property
    def y_max(self) -> Number:
        '''The y-coordinate of the top-right-front vertex.'''
        return self._y_max
    
    @y_max.setter
    def y_max(self, y:Number):
        try:
            y = y.item()
        except:
            pass
        finally:
            self._y_max = y
            if self._y_max < self._y_min:
                self._y_max, self._y_min = self._y_min, self._y_max

    @property
    def z_max(self) -> Number:
        '''The z-coordinate of the top-right-front vertex.'''
        return self._z_max
    
    @z_max.setter
    def z_max(self, z:Number):
        try:
            z = z.item()
        except:
            pass
        finally:
            self._z_max = z
            if self._z_max < self._z_min:
                self._z_max, self._z_min = self._z_min, self._z_max

    @property
    def width(self) -> Number:
        '''The cuboid's width (if not filled).'''
        return self._width
    
    @width.setter
    def width(self, w:Number):
        try:
            w = w.item()
        except:
            pass
        finally:
            self._width = w

    @property
    def color(self) -> ColorType:
        '''The cuboid's color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = [*color, 1] if len(color) == 3 else [*color]

    @property
    def filled(self) -> bool:
        '''Whether to fill in the cuboid or draw only its outline.'''
        return self._filled

    def translate(self, translation:Point3D|NDArray3|List3|Tuple3):
        '''Translates the cuboid by a vector.

        Translates the cuboid by `translation`. This is mostly useful
        when the min/max coordinate values would switch if you applied
        the translation individually to each vertex.

        Args:
            translation: The translation vector. Its coordinates will
                be added to the cuboid's coordinates.
        '''
        if isinstance(translation, Point3D):
            x = translation.x
            y = translation.y
            z = translation.z
        elif isinstance(translation, (list, tuple)):
            x = translation[0]
            y = translation[1]
            z = translation[2]
        elif isinstance(translation, np.ndarray):
            x = translation[0].item()
            y = translation[1].item()
            z = translation[2].item()
        else:
            raise TypeError("Incorrect type for translation")
        
        self._x_min += x
        self._y_min += y
        self._z_min += z
        self._x_max += x
        self._y_max += y
        self._z_max += z
