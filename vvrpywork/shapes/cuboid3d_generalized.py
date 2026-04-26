from .abstract import Shape
from .types import NDArray3, List3, Tuple3, ColorType, Number
from .cuboid3d import Cuboid3D
from .point3d import Point3D
from vvrpywork.scene import Scene3D, Scene3D_

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


class Cuboid3DGeneralized(Shape):
    '''A class used to represent a cuboid in 3D space.
    
    A class used to represent a cuboid in 3D space. The cuboid may be
    translated and rotated. If you need a cuboid that supports more
    complex deformations, use a `Mesh3D` instead.
    '''

    def __init__(self, cuboid:Cuboid3D):
        '''Inits Cuboid3DGeneralized from a Cuboid3D object.

        Args:
            cuboid: The `Cuboid3D` object to copy.
        '''
        if isinstance(cuboid, Cuboid3D):
            self._vertices = np.array([[cuboid.x_min, cuboid.y_min, cuboid.z_min],
                                       [cuboid.x_max, cuboid.y_min, cuboid.z_min],
                                       [cuboid.x_max, cuboid.y_max, cuboid.z_min],
                                       [cuboid.x_min, cuboid.y_max, cuboid.z_min],
                                       [cuboid.x_min, cuboid.y_min, cuboid.z_max],
                                       [cuboid.x_max, cuboid.y_min, cuboid.z_max],
                                       [cuboid.x_max, cuboid.y_max, cuboid.z_max],
                                       [cuboid.x_min, cuboid.y_max, cuboid.z_max]])
            self._triangles = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                                        [0, 4, 7], [0, 7, 3], [5, 1, 2], [5, 2, 6],
                                        [7, 6, 2], [7, 2, 3], [0, 1, 5], [0, 5, 4]])
            
        else:
            raise TypeError("Incorrect type for cuboid")

        self.width = cuboid.width
        self._color = cuboid.color
        self._filled = cuboid.filled

    def _addToScene(self, scene:Scene3D, name:None|str):
        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        material = rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        if self.filled:
            shape = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self._vertices), o3d.utility.Vector3iVector(self._triangles))
            shape.compute_vertex_normals()
        else:
            lines = np.array(((0, 1), (1, 2), (2, 3), (3, 0),
                              (4, 5), (5, 6), (6, 7), (7, 4),
                              (0, 4), (1, 5), (2, 6), (3, 7)))
            shape = o3d.geometry.LineSet(o3d.utility.Vector3dVector(self._vertices), o3d.utility.Vector2iVector(lines))
            material.shader = "unlitLine"
            material.line_width = 2 * self.width
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        material.base_color = color
        scene._scene_widget.scene.add_geometry(name, shape, material)

    def _update(self, name:str, scene:Scene3D):
        scene.removeShape(name)
        self._addToScene(scene, name)

    def _addToScene_PyVista(self, scene:Scene3D_, name:None|str):
        from pyvista import PolyData

        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        if self.filled:
            vertices = self._vertices
            faces = self._triangles
            faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
            self._shape_pv = PolyData(vertices, faces=faces)
            self._actor = scene._plotter.add_mesh(self._shape_pv, color=tuple(int(255 * _ + 0.5) for _ in color[:3]), opacity=color[3], name=name)
        else:
            vertices = self._vertices
            lines = np.array(((0, 1), (1, 2), (2, 3), (3, 0),
                              (4, 5), (5, 6), (6, 7), (7, 4),
                              (0, 4), (1, 5), (2, 6), (3, 7)))
            lines = np.hstack([np.full((lines.shape[0], 1), 2), lines])
            self._shape_pv = PolyData(vertices, lines=lines)
            self._actor = scene._plotter.add_mesh(self._shape_pv, color=tuple(int(255 * _ + 0.5) for _ in color[:3]), opacity=color[3], name=name, line_width=2 * self.width)

    def _update_PyVista(self, name:str, scene:Scene3D_):
        self._shape_pv.points = self._vertices
        # faces or lines shouldn't change
        color = self.color
        color = tuple((*color, 1)) if len(color) == 3 else color
        self._actor.GetProperty().SetColor(color[0], color[1], color[2])
        self._actor.GetProperty().SetOpacity(color[3])
        if not self.filled:
            self._actor.GetProperty().SetLineWidth(self.width)

    @property
    def width(self) -> Number:
        '''The cuboid's width (if not filled).'''
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

        Translates the cuboid by `translation`.

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
        
        self._vertices += np.array((x, y, z))

    def rotate(self, angle:Number, axis:NDArray3|List3|Tuple3):
        '''Rotates the cuboid.

        Rotates the cuboid using a rotation represented as axis-angle.

        Args:
            angle: The angle to rotate the cuboid.
            axis: The axis about which to rotate the cuboid.
        
        '''
        if isinstance(axis, (np.ndarray, list, tuple)):
            axis = np.array(axis)
            axis = axis / np.linalg.norm(axis)
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(angle * axis)
            self._vertices = (rot @ self._vertices.T).T

        else:
            raise TypeError("Incorrect type for axis")
