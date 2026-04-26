from .abstract import ShapeSet
from .types import NDArray, List, Tuple, ColorType, Number
from vvrpywork.scene import Scene3D, Scene3D_

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import os


class Mesh3D(ShapeSet):
    '''A class used to represent a triangle mesh in 3D space.'''

    def __init__(self, path:None|str=None, color:ColorType=(0, 0, 0), smooth_shading=True):
        '''Inits Mesh3D.

        Inits a Mesh3D from a specified path.

        Args:
            path: The path to a file describing a triangle mesh.
            color: The color of the displayed mesh (RGB or RGBA).
            snooth_shading: Only for the PyVista backend. In smooth shading
                mode, normals are ignored. Set this to `False` if you are
                using custom normals.
        '''
        self._color = [*color, 1] if len(color) == 3 else [*color]

        self._smooth_shading = smooth_shading

        if path is not None:
            self._shape = o3d.io.read_triangle_mesh(path)
        else:
            self._shape = o3d.geometry.TriangleMesh()
        self._material = rendering.MaterialRecord()
        self._material.shader = "defaultLitTransparency"
        self._material.base_color = (*color[:3], color[3] if len(color) == 4 else 1)
        if not self._shape.has_vertex_normals():
            self._shape.compute_vertex_normals()
        if not self._shape.has_triangle_normals():
            self._shape.compute_triangle_normals()

    def _addToScene(self, scene:Scene3D, name:None|str):
        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        if not self._shape.has_vertex_normals():
            self._shape.compute_vertex_normals()
        if not self._shape.has_triangle_normals():
            self._shape.compute_triangle_normals()
        scene._scene_widget.scene.add_geometry(name, self._shape, self._material)

    def _update(self, name:str, scene:Scene3D):
        scene.removeShape(name)
        self._addToScene(scene, name)

    def _addToScene_PyVista(self, scene:Scene3D_, name:None|str):
        from pyvista import PolyData

        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        if len(self.vertices) == 0 or len(self.triangles) == 0:
            self._shape_pv = PolyData(np.array(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))), np.array((3, 0, 1, 2)))
            self._shape_pv.point_data["colors"] = np.zeros((3, 4))
        else:
            vertices = np.asarray(self._shape.vertices)
            faces = np.asarray(self._shape.triangles)
            faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
            self._shape_pv = PolyData(vertices, faces)
            self._shape_pv.point_data.active_normals = self.vertex_normals
            self._shape_pv.cell_data.active_normals = self.triangle_normals
            self._shape_pv.point_data["colors"] = self.vertex_colors
        self._actor = scene._plotter.add_mesh(self._shape_pv, name=name, smooth_shading=self._smooth_shading, split_sharp_edges=self._smooth_shading, scalars="colors", rgb=True)

    def _update_PyVista(self, name:str, scene:Scene3D_):
        if len(self.vertices) == 0 or len(self.triangles) == 0:
            self._shape_pv.points = np.array(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
            self._shape_pv.faces = np.array((3, 0, 1, 2))
            self._shape_pv.point_data["colors"] = np.zeros((3, 4))
        else:
            self._shape_pv.points = self.vertices
            self._shape_pv.faces = np.hstack([np.full((self.triangles.shape[0], 1), 3), self.triangles])
            self._shape_pv.point_data.active_normals = self.vertex_normals
            self._shape_pv.cell_data.active_normals = self.triangle_normals
            self._shape_pv.point_data["colors"] = self.vertex_colors
        # scene._plotter.update()
        self._actor = scene._plotter.add_mesh(self._shape_pv, name=name, smooth_shading=self._smooth_shading, split_sharp_edges=self._smooth_shading, scalars="colors", rgb=True)

    @property
    def vertices(self) -> NDArray:
        '''The vertices of the mesh.'''
        return np.copy(np.asarray(self._shape.vertices))
    
    @vertices.setter
    def vertices(self, vertices:NDArray|List|Tuple):
        self._shape.vertices = o3d.utility.Vector3dVector(vertices)

    @property
    def triangles(self) -> NDArray:
        '''The triangles (as indices to `points`) of the mesh.'''
        return np.copy(np.asarray(self._shape.triangles))
    
    @triangles.setter
    def triangles(self, triangles:NDArray|List|Tuple):
        self._shape.triangles = o3d.utility.Vector3iVector(triangles)

    @property
    def vertex_normals(self) -> NDArray:
        '''The normals of each vertex.'''
        if not self._shape.has_vertex_normals():
            self._shape.compute_vertex_normals()
        return np.copy(np.asarray(self._shape.vertex_normals))
    
    @vertex_normals.setter
    def vertex_normals(self, normals:NDArray|List|Tuple):
        self._shape.vertex_normals = o3d.utility.Vector3dVector(normals)

    @property
    def triangle_normals(self) -> NDArray:
        '''The normals of each triangle.'''
        if not self._shape.has_triangle_normals():
            self._shape.compute_triangle_normals()
        return np.copy(np.asarray(self._shape.triangle_normals))
    
    @triangle_normals.setter
    def triangle_normals(self, normals:NDArray|List|Tuple):
        self._shape.triangle_normals = o3d.utility.Vector3dVector(normals)

    @property
    def color(self) -> ColorType:
        '''The mesh's color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = color
        self._material.base_color = (*color[:3], color[3] if len(color) == 4 else 1)

    @property
    def vertex_colors(self) -> NDArray:
        '''A specific color for each vertex.'''
        if not self._shape.has_vertex_colors():
            self._shape.paint_uniform_color(self._color[:3])
        return np.copy(np.asarray(self._shape.vertex_colors))
    
    @vertex_colors.setter
    def vertex_colors(self, colors:NDArray|List|Tuple):
        self._shape.vertex_colors = o3d.utility.Vector3dVector(colors)

    @property
    def smooth_shading(self) -> bool:
        '''Whether smooth shaing is enabled (for Scene3D_).'''
        return self._smooth_shading
    
    @smooth_shading.setter
    def smooth_shading(self, smooth:bool):
        self._smooth_shading = smooth

    def remove_duplicated_vertices(self):
        '''Removes duplicated vertices.'''
        self._shape.remove_duplicated_vertices()
        self._shape.compute_vertex_normals()

    def remove_unreferenced_vertices(self):
        '''Removes unreferenced vertices.'''
        self._shape.remove_unreferenced_vertices()
        self._shape.compute_vertex_normals()

    @staticmethod
    def create_bunny(color:ColorType=(0, 0, 0)) -> "Mesh3D":
        '''Creates a mesh of the Stanford Bunny.
        
        Returns:
            The `Mesh3D` object of the Stanford Bunny.
        '''
        m = Mesh3D(o3d.data.BunnyMesh(os.path.join(os.path.abspath(os.sep), "vvrpywork_data", "open3d_data")).path, color)
        m.remove_unreferenced_vertices()
        return m
    
    @staticmethod
    def create_armadillo(color:ColorType=(0, 0, 0)) -> "Mesh3D":
        '''Creates a mesh of the Stanford Armadillo.
        
        Returns:
            The `Mesh3D` object of the Stanford Armadillo.
        '''
        m = Mesh3D(o3d.data.ArmadilloMesh(os.path.join(os.path.abspath(os.sep), "vvrpywork_data", "open3d_data")).path, color)
        m.vertices = (((-1, 0, 0), (0, 1, 0), (0, 0, -1)) @ m.vertices.T).T
        m.vertex_normals = (((-1, 0, 0), (0, 1, 0), (0, 0, -1)) @ m.vertex_normals.T).T
        return m
