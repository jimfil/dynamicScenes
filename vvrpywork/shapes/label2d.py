from .abstract import Shape
from .types import NDArray2, List2, Tuple2, ColorType, Number
from .point2d import Point2D
from vvrpywork.scene import Scene2D

from numpy import ndarray
from pyglet.text import Label
from pyglet.graphics import Group


class Label2D(Shape):
    '''A class used to represent a text label in 2D space.'''

    def __init__(self, p:Point2D|NDArray2|List2|Tuple2, text:str, size:Number=32, font:str="", bold:bool=False, italic:bool=False, color:ColorType=(0, 0, 0)):
        '''Inits Label2D from (x,y) coordinates and text.
        
        Args:
            p: The coordinates of the label's anchor (center).
            text: The text of the label.
            size: The font size of the label.
            font: The name of an installed font.
            bold: Whether the text should be in bold.
            italic: Whether the text should be in italic.
            color: The color of the text (RGB or RGBA).
        '''
        
        if isinstance(p, Point2D):
            self._x = p.x
            self._y = p.y
        elif isinstance(p, (list, tuple)):
            self._x = p[0]
            self._y = p[1]
        elif isinstance(p, ndarray):
            self._x = p[0].item()
            self._y = p[1].item()
        else:
            raise TypeError("Incorrect type for p")
        
        self.text = text
        self.size = size
        self.font = font
        self.bold = bold
        self.italic = italic
        self.color = color

    def _addToScene(self, scene:Scene2D, name:None|str):
        name = str(id(self)) if name is None else name
        shape = Label(self.text, self.font, self.size, self.bold, self.italic, color=tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer), x=100 * self.x, y=100 * self.y, anchor_x="center", anchor_y="center", program=scene._text_shader)
        scene._shapeDict[name] = {"class": self, "shape": shape}

    def _update(self, shape:Label, scene:Scene2D):
        shape.x = 100 * self.x
        shape.y = 100 * self.y
        shape.text = self.text
        shape.font_size = self.size
        shape.font_name = self.font
        shape.bold = self.bold
        shape.italic = self.italic
        shape.color = tuple(int(255 * _ + 0.5) for _ in self.color)

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
    def font(self) -> str:
        '''The label's font.'''
        return self._font
    
    @font.setter
    def font(self, font:str):
        self._font = font

    @property
    def bold(self) -> bool:
        '''Whether the text should be in bold.'''
        return self._bold
    
    @bold.setter
    def bold(self, bold:bool):
        self._bold = bold

    @property
    def italic(self) -> bool:
        '''Whether the text should be in italic.'''
        return self._italic
    
    @italic.setter
    def italic(self, italic:bool):
        self._italic = italic

    @property
    def color(self) -> ColorType:
        '''The label's font color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = [*color, 1] if len(color) == 3 else [*color]
