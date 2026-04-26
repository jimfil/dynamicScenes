from .abstract import Shape
from .types import NDArray2, List2, Tuple2, ColorType, Number
from .point2d import Point2D
from vvrpywork.scene import Scene2D

from numpy import ndarray
from pyglet.shapes import Circle, Arc
from pyglet.graphics import Group


class Circle2D(Shape):
    '''A class used to represent a circle in 2D space.'''

    def __init__(self, p:Point2D|NDArray2|List2|Tuple2, radius:Number, resolution:None|int=None, width:Number=1, color:ColorType=(0, 0, 0), filled:bool=False):
        '''Inits Circle2D given the circle's center and radius.

        Args:
            p: The coordinates of the center.
            radius: The circle's radius.
            resolution: The resolution of the displayed circle.
                If `None`, it will be calculated automatically.
            width: The width of the displayed circle (if not filled).
            color: The color of the displayed circle (RGB or RGBA).
            filled: Whether to fill in the circle or draw only its
                outline.
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
        
        self.radius = radius
        self._resolution = resolution
        self.width = width
        self._color = [*color, 1] if len(color) == 3 else [*color]
        self._filled = filled

    def _addToScene(self, scene:Scene2D, name:None|str):
        if self.filled:
            name = str(id(self)) if name is None else name
            shape = Circle(100 * self.x, 100 * self.y, 100 * self.radius, self.resolution, tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer))
            scene._shapeDict[name] = {"class": self, "shape": shape}
        else:
            name = str(id(self)) if name is None else name
            shape = Arc(100 * self.x, 100 * self.y, 100 * self.radius, self.resolution, thickness=self.width, color=tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer))
            scene._shapeDict[name] = {"class": self, "shape": shape}

    def _update(self, shape:Circle|Arc, scene:Scene2D):
        if self.filled:
            shape.position = (100 * self.x, 100 * self.y)
            shape.radius = 100 * self.radius
            shape.color = tuple(int(255 * _ + 0.5) for _ in self.color)
        else:
            shape.position = (100 * self.x, 100 * self.y)
            shape.radius = 100 * self.radius
            shape.thickness = self.width
            shape.color = tuple(int(255 * _ + 0.5) for _ in self.color)

    @property
    def x(self) -> Number:
        '''The x-coordinate of the circle's center point.'''
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
        '''The y-coordinate of the circle's center point.'''
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
    def radius(self) -> Number:
        '''The circle's radius.'''
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
    def resolution(self) -> None|int:
        '''The circle's resolution.
        
        The circle is drawn using triangles. `resolution` represents
        the amount of triangles that will be used.
        '''
        return self._resolution
    
    @property
    def width(self) -> Number:
        '''The circle's width (if not filled).'''
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
        '''The circle's color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = [*color, 1] if len(color) == 3 else [*color]

    @property
    def filled(self) -> bool:
        '''Whether to fill in the circle or draw only its outline.'''
        return self._filled
    
    def getPointCenter(self) -> Point2D:
        '''Returns the circle's center.
        
        Returns:
            The circle's center point as a `Point2D` object.
        '''
        return Point2D((self.x, self.y))
    
    def contains(self, point:Point2D) -> bool:
        '''Determines whether a point is inside the circle.

        Args:
            point: The point to check (if it's inside the circle).

        Returns:
            `True` if the point is inside the circle (incl. the
                outline), `False` otherwise.
        '''
        return (self.x - point.x) ** 2 + (self.y - point.y) ** 2 <= self.radius ** 2
