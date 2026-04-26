from .abstract import Shape
from .types import NDArray3, List3, Tuple3, ColorType, Number
from .point3d import Point3D
from vvrpywork.scene import Scene3D, Scene3D_

from math import acos
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


class Line3D(Shape):
    '''A class used to represent a line segment in 3D space.'''

    def __init__(self, p1:Point3D|NDArray3|List3|Tuple3, p2:Point3D|NDArray3|List3|Tuple3, width:Number=1, resolution:int=20, color:ColorType=(0, 0, 0)):
        '''Inits Line3D given the line segment's 2 endpoints.

        Args:
            p1: The coordinates of the first endpoint.
            p2: The coordinates of the second endpoint.
            width: The width of the displayed line segment.
            resolution: The resolution of the displayed line.
            color: The color of the displayed line segment (RGB or
                RGBA).
        '''
        
        if isinstance(p1, Point3D):
            self._x1 = p1.x
            self._y1 = p1.y
            self._z1 = p1.z
        elif isinstance(p1, (list, tuple)):
            self._x1 = p1[0]
            self._y1 = p1[1]
            self._z1 = p1[2]
        elif isinstance(p1, np.ndarray):
            self._x1 = p1[0].item()
            self._y1 = p1[1].item()
            self._z1 = p1[2].item()
        else:
            raise TypeError("Incorrect type for p1")
        
        if isinstance(p2, Point3D):
            self._x2 = p2.x
            self._y2 = p2.y
            self._z2 = p2.z
        elif isinstance(p2, (list, tuple)):
            self._x2 = p2[0]
            self._y2 = p2[1]
            self._z2 = p2[2]
        elif isinstance(p2, np.ndarray):
            self._x2 = p2[0].item()
            self._y2 = p2[1].item()
            self._z2 = p2[2].item()
        else:
            raise TypeError("Incorrect type for p2")

        self._width = width
        self._resolution = resolution

        self._color = [*color, 1] if len(color) == 3 else [*color]

        self._length = self.length()
        self._rot = None
    
    def _addToScene(self, scene:Scene3D, name:None|str):
        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        shape = o3d.geometry.TriangleMesh.create_cylinder(0.005 * self.width, self.length(), self.resolution)
        shape.compute_vertex_normals()

        v = np.array(((self.x2-self.x1)/self.length(), (self.y2-self.y1)/self.length(), (self.z2-self.z1)/self.length()))
        z = np.array((0., 0., 1.))
        a = np.cross(z, v)
        div = (a**2).sum()**0.5
        
        if div != 0:
            a /= div
            aa = a * acos(np.dot(z, v))

            self._rot = o3d.geometry.get_rotation_matrix_from_axis_angle(aa)
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = self._rot
            rotation[3, 3] = 1
        else:
            self._rot = None
            rotation = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

        translation = np.array(((1, 0, 0, (self.x1+self.x2)/2), (0, 1, 0, (self.y1+self.y2)/2), (0, 0, 1, (self.z1+self.z2)/2), (0, 0, 0, 1)))

        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        material.base_color = color
        scene._scene_widget.scene.add_geometry(name, shape, material)
        scene._scene_widget.scene.set_geometry_transform(name, translation @ rotation)

    def _update(self, name:str, scene:Scene3D):
        old_length = self._length
        new_length = ((self.x2-self.x1)**2 + (self.y2-self.y1)**2 + (self.z2-self.z1)**2)**.5
        self._length = new_length
        scale = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, new_length/old_length, 0), (0, 0, 0, 1)))
        v = np.array(((self.x2-self.x1)/new_length, (self.y2-self.y1)/new_length, (self.z2-self.z1)/new_length))
        z = np.array((0., 0., 1.))
        a = np.cross(z, v)
        div = (a**2).sum()**0.5
        
        if div != 0:
            a /= div
            aa = a * acos(np.dot(z, v))

            self._rot = o3d.geometry.get_rotation_matrix_from_axis_angle(aa)
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = self._rot
            rotation[3, 3] = 1
        else:
            self._rot = None
            rotation = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

        translation = np.array(((1, 0, 0, (self.x1+self.x2)/2), (0, 1, 0, (self.y1+self.y2)/2), (0, 0, 1, (self.z1+self.z2)/2), (0, 0, 0, 1)))

        scene._scene_widget.scene.set_geometry_transform(name, translation @ rotation @ scale)
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color = color
        scene._scene_widget.scene.modify_geometry_material(name, material)

    def _addToScene_PyVista(self, scene:Scene3D_, name:None|str):
        from pyvista import Cylinder
        from vtk import vtkMatrix4x4

        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        shape = Cylinder(radius=0.005 * self.width, height=self.length(), resolution=self.resolution, direction=(0, 0, 1))

        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        self._actor = scene._plotter.add_mesh(shape, color=tuple(int(255 * _ + 0.5) for _ in color[:3]), opacity=0, name=name, smooth_shading=True, split_sharp_edges=True)

        v = np.array(((self.x2-self.x1)/self.length(), (self.y2-self.y1)/self.length(), (self.z2-self.z1)/self.length()))
        z = np.array((0., 0., 1.))
        a = np.cross(z, v)
        div = (a**2).sum()**0.5
        
        if div != 0:
            a /= div
            aa = a * acos(np.dot(z, v))

            self._rot = o3d.geometry.get_rotation_matrix_from_axis_angle(aa)
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = self._rot
            rotation[3, 3] = 1
        else:
            self._rot = None
            rotation = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

        translation = np.array(((1, 0, 0, (self.x1+self.x2)/2), (0, 1, 0, (self.y1+self.y2)/2), (0, 0, 1, (self.z1+self.z2)/2), (0, 0, 0, 1)))

        transform = translation @ rotation

        mat = vtkMatrix4x4()
        mat.DeepCopy(transform.ravel())
        self._actor.SetUserMatrix(mat)
        self._actor.GetProperty().SetOpacity(color[3])

    def _update_PyVista(self, name:str, scene:Scene3D_):
        from vtk import vtkMatrix4x4
        
        old_length = self._length
        new_length = ((self.x2-self.x1)**2 + (self.y2-self.y1)**2 + (self.z2-self.z1)**2)**.5
        self._length = new_length
        scale = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, new_length/old_length, 0), (0, 0, 0, 1)))
        v = np.array(((self.x2-self.x1)/new_length, (self.y2-self.y1)/new_length, (self.z2-self.z1)/new_length))
        z = np.array((0., 0., 1.))
        a = np.cross(z, v)
        div = (a**2).sum()**0.5
        
        if div != 0:
            a /= div
            aa = a * acos(np.dot(z, v))

            self._rot = o3d.geometry.get_rotation_matrix_from_axis_angle(aa)
            rotation = np.zeros((4, 4))
            rotation[:3, :3] = self._rot
            rotation[3, 3] = 1
        else:
            self._rot = None
            rotation = np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))

        translation = np.array(((1, 0, 0, (self.x1+self.x2)/2), (0, 1, 0, (self.y1+self.y2)/2), (0, 0, 1, (self.z1+self.z2)/2), (0, 0, 0, 1)))

        transform = translation @ rotation @ scale

        mat = vtkMatrix4x4()
        mat.DeepCopy(transform.ravel())
        self._actor.SetUserMatrix(mat)

        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        self._actor.GetProperty().SetColor(color[0], color[1], color[2])
        self._actor.GetProperty().SetOpacity(color[3])

    @property
    def x1(self) -> Number:
        '''The x-coordinate of the first endpoint.'''
        return self._x1
    
    @x1.setter
    def x1(self, x1:Number):
        try:
            x1 = x1.item()
        except:
            pass
        finally:
            self._x1 = x1

    @property
    def y1(self) -> Number:
        '''The y-coordinate of the first endpoint.'''
        return self._y1
    
    @y1.setter
    def y1(self, y1:Number):
        try:
            y1 = y1.item()
        except:
            pass
        finally:
            self._y1 = y1

    @property
    def z1(self) -> Number:
        '''The z-coordinate of the first endpoint.'''
        return self._z1
    
    @z1.setter
    def z1(self, z1:Number):
        try:
            z1 = z1.item()
        except:
            pass
        finally:
            self._z1 = z1

    @property
    def x2(self) -> Number:
        '''The x-coordinate of the second endpoint.'''
        return self._x2
    
    @x2.setter
    def x2(self, x2:Number):
        try:
            x2 = x2.item()
        except:
            pass
        finally:
            self._x2 = x2

    @property
    def y2(self) -> Number:
        '''The y-coordinate of the second endpoint.'''
        return self._y2
    
    @y2.setter
    def y2(self, y2:Number):
        try:
            y2 = y2.item()
        except:
            pass
        finally:
            self._y2 = y2

    @property
    def z2(self) -> Number:
        '''The z-coordinate of the second endpoint.'''
        return self._z2
    
    @z2.setter
    def z2(self, z2:Number):
        try:
            z2 = z2.item()
        except:
            pass
        finally:
            self._z2 = z2

    @property
    def width(self) -> Number:
        '''The line segment's width.'''
        return self._width
    
    @property
    def resolution(self) -> int:
        '''The line's resolution.
        
        The line is drawn as a small cylinder using triangles.
        `resolution` represents the amount of triangles that will be
        used.
        '''
        return self._resolution

    @property
    def color(self) -> ColorType:
        '''The line segment's color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = [*color, 1] if len(color) == 3 else [*color]

    def getPointFrom(self) -> Point3D:
        '''Returns the line segment's first endpoint.
        
        Returns:
            The line segment's first endpoint as a `Point3D` object.
        '''
        return Point3D((self.x1, self.y1, self.z1))
    
    def getPointTo(self) -> Point3D:
        '''Returns the line segment's second endpoint.
        
        Returns:
            The line segment's second endpoint as a `Point3D` object.
        '''
        return Point3D((self.x2, self.y2, self.z2))

    def length(self) -> float:
        '''Calculates the length of the line segment.
        
        Returns:
            The length of the line segment.
        '''
        return ((self._x2-self._x1)**2 + (self._y2-self._y1)**2 + (self._z2-self._z1)**2)**.5
