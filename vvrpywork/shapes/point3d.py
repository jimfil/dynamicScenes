from .abstract import Shape
from .types import NDArray3, List3, Tuple3, ColorType, Number
from vvrpywork.scene import Scene3D, Scene3D_

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


class Point3D(Shape):
    '''A class used to represent a point in 3D space.'''

    def __init__(self, p:"Point3D"|NDArray3|List3|Tuple3, size:Number=1, resolution:int=20, color:ColorType=(0, 0, 0)):
        '''Inits Point3D from (x,y,z) coordinates.
        
        Args:
            p: The coordinates of the point.
            size: The size of the displayed point.
            resolution: The resolution of the displayed point.
            color: The color of the displayed point (RGB or RGBA).
        '''
        if isinstance(p, Point3D):
            self._x = p.x
            self._y = p.y
            self._z = p.z
        elif isinstance(p, (list, tuple)):
            self._x = p[0]
            self._y = p[1]
            self._z = p[2]
        elif isinstance(p, np.ndarray):
            self._x = p[0].item()
            self._y = p[1].item()
            self._z = p[2].item()
        else:
            raise TypeError("Incorrect type for p")
        
        self.size = size
        self._resolution = resolution

        self._color = [*color, 1] if len(color) == 3 else [*color]

    def _addToScene(self, scene:Scene3D, name:None|str):
        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        shape = o3d.geometry.TriangleMesh.create_sphere(0.02, self.resolution)
        shape.compute_vertex_normals()
        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        material.base_color = color
        scene._scene_widget.scene.add_geometry(name, shape, material)
        scene._scene_widget.scene.set_geometry_transform(name, ((self.size, 0, 0, self.x), (0, self.size, 0, self.y), (0, 0, self.size, self.z), (0, 0, 0, 1)))

    def _update(self, name:str, scene:Scene3D):
        translation = np.array(((self.size, 0, 0, self._x), (0, self.size, 0, self._y), (0, 0, self.size, self._z), (0, 0, 0, 1)))
        scene._scene_widget.scene.set_geometry_transform(name, translation)
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color = color
        scene._scene_widget.scene.modify_geometry_material(name, material)

    def _addToScene_PyVista(self, scene:Scene3D_, name:None|str):
        from pyvista import Sphere
        from vtk import vtkMatrix4x4

        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        shape = Sphere(radius=0.02, theta_resolution=self.resolution, phi_resolution=self.resolution)
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        self._actor = scene._plotter.add_mesh(shape, color=tuple(int(255 * _ + 0.5) for _ in color[:3]), opacity=0, name=name, smooth_shading=True, split_sharp_edges=True)
        transform = np.array(((self.size, 0, 0, self._x), (0, self.size, 0, self._y), (0, 0, self.size, self._z), (0, 0, 0, 1)))
        mat = vtkMatrix4x4()
        mat.DeepCopy(transform.ravel())
        self._actor.SetUserMatrix(mat)
        self._actor.GetProperty().SetOpacity(color[3])

    def _update_PyVista(self, name:str, scene:Scene3D_):
        from vtk import vtkMatrix4x4
        
        transform = np.array(((self.size, 0, 0, self._x), (0, self.size, 0, self._y), (0, 0, self.size, self._z), (0, 0, 0, 1)))
        mat = vtkMatrix4x4()
        mat.DeepCopy(transform.ravel())
        self._actor.SetUserMatrix(mat)
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        self._actor.GetProperty().SetColor(color[0], color[1], color[2])
        self._actor.GetProperty().SetOpacity(color[3])

    def __eq__(self, other:"Point3D"|NDArray3|List3|Tuple3):
        if isinstance(other, Point3D):
            return self.x == other.x and self.y == other.y and self.z == other.z
        elif isinstance(other, np.ndarray, list, tuple):
            return self.x == other[0] and self.y == other[1] and self.x == other[2]
        else:
            return False

    @property
    def x(self) -> Number:
        '''The point's position on the x-axis.'''
        return self._x
    
    @x.setter
    def x(self, x:Number):
        try:
            x = x.item()
        except:
            pass
        finally:
            self._x = x

    @property
    def y(self) -> Number:
        '''The point's position on the y-axis.'''
        return self._y
    
    @y.setter
    def y(self, y:Number):
        try:
            y = y.item()
        except:
            pass
        finally:
            self._y = y

    @property
    def z(self) -> Number:
        '''The point's position on the z-axis.'''
        return self._z
    
    @z.setter
    def z(self, z:Number):
        try:
            z = z.item()
        except:
            pass
        finally:
            self._z = z

    @property
    def size(self) -> Number:
        '''The point's size.'''
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
    def resolution(self) -> int:
        '''The point's resolution.
        
        The point is drawn using triangles. `resolution` represents
        the amount of triangles that will be used.
        '''
        return self._resolution

    @property
    def color(self) -> ColorType:
        '''The point's color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = [*color, 1] if len(color) == 3 else [*color]

    def distanceSq(self, p:"Point3D") -> Number:
        '''Calculates the squared distance from a second point.
        
        Calculates the squared Euclidean distance between this and
        another point. It doesn't take the square root of the result
        and is, therefore, faster than calling `distance`.

        Args:
            p: The second point, the squared distance to which will
                be calculated.

        Returns:
            The squared distance between this point and `p`.
        '''
        return (self.x - p.x) ** 2 + (self.y - p.y) ** 2 + (self.z - p.z) ** 2
    
    def distance(self, p:"Point3D") -> float:
        '''Calculates the distance from a second point.
        
        Calculates the Euclidean distance between this and another
        point. If you do not need the exact distance, you may want
        to look into using `distanceSq` instead.

        Args:
            p: The second point, the distance to which will be
                calculated.

        Returns:
            The distance between this point and `p`.
        '''
        return self.distanceSq(p) ** 0.5
