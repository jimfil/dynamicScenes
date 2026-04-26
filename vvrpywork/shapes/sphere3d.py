from .abstract import Shape
from .types import NDArray3, List3, Tuple3, ColorType, Number
from .point3d import Point3D
from vvrpywork.scene import Scene3D, Scene3D_

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


class Sphere3D(Shape):
    '''A class used to represent a sphere in 3D space.'''

    def __init__(self, p:Point3D|NDArray3|List3|Tuple3, radius:Number=1, resolution:int=20, width:Number=1, color:ColorType=(0, 0, 0), filled:bool=False):
        '''Inits Sphere3D given the sphere's center and radius.

        Args:
            p: The coordinates of the center.
            radius: The sphere's radius.
            resolution: The resolution of the displayed sphere.
            width: The width of the displayed sphere (if not filled).
            color: The color of the displayed sphere (RGB or RGBA).
            filled: Whether to fill in the sphere or draw only its
                outline.
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
        
        self.radius = radius
        self._resolution = resolution
        self.width = width
        self._color = [*color, 1] if len(color) == 3 else [*color]
        self._filled = filled

    def _addToScene(self, scene:Scene3D, name:None|str):
        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        shape = o3d.geometry.TriangleMesh.create_sphere(1, self.resolution)
        shape.compute_vertex_normals()
        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        if not self.filled:
            shape = o3d.geometry.LineSet.create_from_triangle_mesh(shape)
            material.shader = "unlitLine"
            material.line_width = 2 * self.width
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        material.base_color = color
        scene._scene_widget.scene.add_geometry(name, shape, material)
        scene._scene_widget.scene.set_geometry_transform(name, ((self.radius, 0, 0, self.x), (0, self.radius, 0, self.y), (0, 0, self.radius, self.z), (0, 0, 0, 1)))

    def _update(self, name:str, scene:Scene3D):
        scene._scene_widget.scene.set_geometry_transform(name, ((self.radius, 0, 0, self.x), (0, self.radius, 0, self.y), (0, 0, self.radius, self.z), (0, 0, 0, 1)))
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
        from pyvista import Sphere
        from vtk import vtkMatrix4x4

        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        shape = Sphere(radius=1, theta_resolution=self.resolution, phi_resolution=self.resolution)
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        if self.filled:
            self._actor = scene._plotter.add_mesh(shape, color=tuple(int(255 * _ + 0.5) for _ in color[:3]), opacity=0, name=name, smooth_shading=True, split_sharp_edges=True)
        else:
            self._actor = scene._plotter.add_mesh(shape, color=tuple(int(255 * _ + 0.5) for _ in color[:3]), opacity=0, line_width=self.width, name=name, style="wireframe")
        transform = np.array(((self.radius, 0, 0, self._x), (0, self.radius, 0, self._y), (0, 0, self.radius, self._z), (0, 0, 0, 1)))
        mat = vtkMatrix4x4()
        mat.DeepCopy(transform.ravel())
        self._actor.SetUserMatrix(mat)
        self._actor.GetProperty().SetOpacity(color[3])

    def _update_PyVista(self, name:str, scene:Scene3D_):
        from vtk import vtkMatrix4x4
        
        transform = np.array(((self.radius, 0, 0, self._x), (0, self.radius, 0, self._y), (0, 0, self.radius, self._z), (0, 0, 0, 1)))
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
    def x(self) -> Number:
        '''The x-coordinate of the sphere's center point.'''
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
        '''The y-coordinate of the sphere's center point.'''
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
        '''The z-coordinate of the sphere's center point.'''
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
    def radius(self) -> Number:
        '''The sphere's radius.'''
        return self._radius
    
    @radius.setter
    def radius(self, r:Number):
        try:
            r = r.item()
        except:
            pass
        finally:
            self._radius = r
    
    @property
    def resolution(self) -> int:
        '''The sphere's resolution.
        
        The sphere is drawn using triangles. `resolution` represents
        the amount of triangles that will be used.
        '''
        return self._resolution
    
    @property
    def width(self) -> Number:
        '''The sphere's width (if not filled).'''
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
    def color(self) -> ColorType:
        '''The sphere's color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = [*color, 1] if len(color) == 3 else [*color]

    @property
    def filled(self) -> bool:
        '''Whether to fill in the sphere or draw only its outline.'''
        return self._filled

    def getPointCenter(self) -> Point3D:
        '''Returns the sphere's center.
        
        Returns:
            The sphere's center point as a `Point3D` object.
        '''
        return Point3D((self.x, self.y, self.z))
    
    def contains(self, point:Point3D) -> bool:
        '''Determines whether a point is inside the sphere.

        Args:
            point: The point to check (if it's inside the sphere).

        Returns:
            `True` if the point is inside the sphere (incl. the
                outline), `False` otherwise.
        '''
        return (self.x - point.x) ** 2 + (self.y - point.y) ** 2 + (self.z - point.z) ** 2 <= self.radius ** 2
