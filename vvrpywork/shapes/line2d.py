from .abstract import Shape
from .types import NDArray2, List2, Tuple2, ColorType, Number
from .point2d import Point2D
from vvrpywork.scene import Scene2D

from numpy import ndarray
from pyglet.shapes import Line
from pyglet.graphics import Group


class Line2D(Shape):
    '''A class used to represent a line segment in 2D space.'''

    def __init__(self, p1:Point2D|NDArray2|List2|Tuple2, p2:Point2D|NDArray2|List2|Tuple2, width:Number=1, color:ColorType=(0, 0, 0)):
        '''Inits Line2D given the line segment's 2 endpoints.

        Args:
            p1: The coordinates of the first endpoint.
            p2: The coordinates of the second endpoint.
            width: The width of the displayed line segment.
            color: The color of the displayed line segment (RGB or
                RGBA).
        '''
        
        if isinstance(p1, Point2D):
            self._x1 = p1.x
            self._y1 = p1.y
        elif isinstance(p1, (list, tuple)):
            self._x1 = p1[0]
            self._y1 = p1[1]
        elif isinstance(p1, ndarray):
            self._x1 = p1[0].item()
            self._y1 = p1[1].item()
        else:
            raise TypeError("Incorrect type for p1")
        
        if isinstance(p2, Point2D):
            self._x2 = p2.x
            self._y2 = p2.y
        elif isinstance(p2, (list, tuple)):
            self._x2 = p2[0]
            self._y2 = p2[1]
        elif isinstance(p2, ndarray):
            self._x2 = p2[0].item()
            self._y2 = p2[1].item()
        else:
            raise TypeError("Incorrect type for p2")
        
        self.width = width
        self._color = [*color, 1] if len(color) == 3 else [*color]
    
    def _addToScene(self, scene:Scene2D, name:None|str):
        name = str(id(self)) if name is None else name
        shape = Line(100 * self.x1, 100 * self.y1, 100 * self.x2, 100 * self.y2, self.width, tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer))
        scene._shapeDict[name] = {"class": self, "shape": shape}

    def _update(self, shape:Line, scene:Scene2D):
        shape.position = (100 * self.x1, 100 * self.y1)
        shape.x2 = 100 * self.x2
        shape.y2 = 100 * self.y2
        shape.width = self.width
        shape.color = tuple(int(255 * _ + 0.5) for _ in self.color)

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
    def width(self) -> Number:
        '''The line segment's width.'''
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
        '''The line segment's color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = [*color, 1] if len(color) == 3 else [*color]

    def getPointFrom(self) -> Point2D:
        '''Returns the line segment's first endpoint.
        
        Returns:
            The line segment's first endpoint as a `Point2D` object.
        '''
        return Point2D((self.x1, self.y1))
    
    def getPointTo(self) -> Point2D:
        '''Returns the line segment's second endpoint.
        
        Returns:
            The line segment's second endpoint as a `Point2D` object.
        '''
        return Point2D((self.x2, self.y2))

    def length(self) -> float:
        '''Calculates the length of the line segment.
        
        Returns:
            The length of the line segment.
        '''
        return ((self.x2-self.x1)**2 + (self.y2-self.y1)**2)**.5
    
    def isOnRight(self, point:Point2D) -> bool:
        '''Determines whether a point is to the right of the line.
        
        Determines whether a point is to the right of the line defined
        by this line segment.

        Args:
            point: The point to check (if it's on the right).

        Returns:
            `True` if the point is on the right, `False` otherwise
                (incl. if it's on the line itself).
        '''
        return ((self.x2 - self.x1)*(point.y - self.y1) - (self.y2 - self.y1)*(point.x - self.x1)) < 0
