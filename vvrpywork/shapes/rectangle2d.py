from .abstract import Shape
from .types import NDArray2, List2, Tuple2, ColorType, Number
from .point2d import Point2D
from vvrpywork.scene import Scene2D

from numpy import ndarray
from pyglet.shapes import Rectangle, Box
from pyglet.graphics import Group


class Rectangle2D(Shape):
    '''A class used to represent a rectangle in 2D space.'''

    def __init__(self, p1:Point2D|NDArray2|List2|Tuple2, p2:Point2D|NDArray2|List2|Tuple2, width:Number=1, color:ColorType=(0, 0, 0), filled:bool=False):
        '''Inits Rectangle2D given 2 vertices of the rectangle.

        Args:
            p1: The coordinates of one vertex.
            p2: The coordinates of another vertex.
            width: The width of the displayed rectangle (if not
                filled).
            color: The color of the displayed rectangle (RGB or RGBA).
            filled: Whether to fill in the rectangle or draw only its
                outline.
        '''

        if isinstance(p1, Point2D):
            x1 = p1.x
            y1 = p1.y
        elif isinstance(p1, (list, tuple)):
            x1 = p1[0]
            y1 = p1[1]
        elif isinstance(p1, ndarray):
            x1 = p1[0].item()
            y1 = p1[1].item()
        else:
            raise TypeError("Incorrect type for p1")
        
        if isinstance(p2, Point2D):
            x2 = p2.x
            y2 = p2.y
        elif isinstance(p2, (list, tuple)):
            x2 = p2[0]
            y2 = p2[1]
        elif isinstance(p2, ndarray):
            x2 = p2[0].item()
            y2 = p2[1].item()
        else:
            raise TypeError("Incorrect type for p2")

        self._x_min = min(x1, x2)
        self._y_min = min(y1, y2)
        self._x_max = max(x1, x2)
        self._y_max = max(y1, y2)

        self.width = width
        self._color = [*color, 1] if len(color) == 3 else [*color]
        self._filled = filled

    def _addToScene(self, scene:Scene2D, name:None|str):
        if self.filled:
            name = str(id(self)) if name is None else name
            shape = Rectangle(100 * self.x_min, 100 * self.y_min, 100 * (self.x_max-self.x_min), 100 * (self.y_max-self.y_min), tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer))
            scene._shapeDict[name] = {"class": self, "shape": shape}
        else:
            name = str(id(self)) if name is None else name
            shape = Box(100 * self.x_min, 100 * self.y_min, 100 * (self.x_max-self.x_min), 100 * (self.y_max-self.y_min), self.width, tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer))
            scene._shapeDict[name] = {"class": self, "shape": shape}

    def _update(self, shape:Rectangle|Box, scene:Scene2D):
        if self.filled:
            shape.position = (100 * self.x_min, 100 * self.y_min)
            shape.width = 100 * (self.x_max-self.x_min)
            shape.height = 100 * (self.y_max-self.y_min)
            shape.color = tuple(int(255 * _ + 0.5) for _ in self.color)
        else:
            shape.position = (100 * self.x_min, 100 * self.y_min)
            shape.width = 100 * (self.x_max-self.x_min)
            shape.height = 100 * (self.y_max-self.y_min)
            shape._thickness = self.width
            shape.color = tuple(int(255 * _ + 0.5) for _ in self.color)

    @property
    def x_min(self) -> Number:
        '''The x-coordinate of the bottom-left vertex.'''
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
        '''The y-coordinate of the bottom-left vertex.'''
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
    def x_max(self) -> Number:
        '''The x-coordinate of the top-right vertex.'''
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
        '''The y-coordinate of the top-right vertex.'''
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
    def width(self) -> Number:
        '''The rectangle's width (if not filled).'''
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
        '''The rectangle's color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = [*color, 1] if len(color) == 3 else [*color]

    @property
    def filled(self) -> bool:
        '''Whether to fill in the rectangle or draw only its outline.'''
        return self._filled
    
    def translate(self, translation:Point2D|NDArray2|List2|Tuple2):
        '''Translates the rectangle by a vector.

        Translates the rectangle by `translation`. This is mostly
        useful when the min/max coordinate values would switch if you
        applied the translation individually to each vertex.

        Args:
            translation: The translation vector. Its coordinates will
                be added to the rectangle's coordinates.
        '''
        if isinstance(translation, Point2D):
            x = translation.x
            y = translation.y
        elif isinstance(translation, (list, tuple)):
            x = translation[0]
            y = translation[1]
        elif isinstance(translation, ndarray):
            x = translation[0].item()
            y = translation[1].item()
        else:
            raise TypeError("Incorrect type for translation")
        
        self._x_min += x
        self._y_min += y
        self._x_max += x
        self._y_max += y
