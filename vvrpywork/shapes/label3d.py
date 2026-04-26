from .abstract import Shape
from .types import NDArray3, List3, Tuple3, ColorType, Number
from .point3d import Point3D
from vvrpywork.scene import Scene3D, Scene3D_

import numpy as np
import open3d.visualization.gui as gui


class Label3D(Shape):
    '''A class used to represent a text label in 3D space.'''

    def __init__(self, p:Point3D|NDArray3|List3|Tuple3, text:str, size:Number=1, color:ColorType=(0, 0, 0)):
        '''Inits Label3D from (x,y,z) coordinates and text.
        
        Args:
            p: The coordinates of the label's anchor (Open3D -> top-left,
                PyVista -> bottom-left).
            text: The text of the label.
            size: The font size of the label.
            color: The color of the text (RGB or RGBA).
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
        
        self.text = text
        self.size = size
        self.color = color

    def _addToScene(self, scene:Scene3D, name:None|str):
        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        self._shape = scene._scene_widget.add_3d_label((self.x, self.y, self.z), self.text)
        self._shape.scale = self.size
        self._shape.color = gui.Color(*self.color)

    def _update(self, name:str, scene:Scene3D):
        shape = scene._shapeDict[name]._shape
        shape.position = (self.x, self.y, self.z)
        shape.text = self.text
        shape.scale = self.size
        shape.color = gui.Color(*self.color)

    def _addToScene_PyVista(self, scene:Scene3D_, name:None|str):
        from pyvista import Label

        name = str(id(self)) if name is None else name
        scene._shapeDict[name] = self
        # color = self.color
        # color = tuple((*color, 1)) if len(color) == 3 else color
        self._shape_pv = Label(self.text, (self.x, self.y, self.z), size=16 * self.size)
        self._actor = scene._plotter.add_actor(self._shape_pv)

    def _update_PyVista(self, name:str, scene:Scene3D_):
        self._shape_pv.input = self.text
        self._shape_pv.position = (self.x, self.y, self.z)
        self._shape_pv.size = 16 * self.size


    @property
    def x(self) -> Number:
        '''The label's anchor x-coordinate.'''
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
        '''The label's anchor y-coordinate.'''
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
        '''The label's anchor z-coordinate.'''
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
    def text(self) -> str:
        '''The label's text.'''
        return self._text
    
    @text.setter
    def text(self, text:str):
        self._text = text

    @property
    def size(self) -> Number:
        '''The label's font size.'''
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
    def color(self) -> ColorType:
        '''The label's font color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = [*color, 1] if len(color) == 3 else [*color]
