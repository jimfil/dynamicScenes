from .abstract import Shape
from .types import NDArray2, List2, Tuple2, ColorType, Number
from vvrpywork.scene import Scene2D

from numpy import ndarray
from pyglet.shapes import Circle
from pyglet.graphics import Group


class Point2D(Shape):
    '''A class used to represent a point in 2D space.'''

    def __init__(self, p:"Point2D"|NDArray2|List2|Tuple2, size:Number=1.0, resolution:None|int=None, color:ColorType=(0, 0, 0)):
        '''Inits Point2D from (x,y) coordinates.
        
        Args:
            p: The coordinates of the point.
            size: The size of the displayed point.
            resolution: The resolution of the displayed point.
                If `None`, it will be calculated automatically.
            color: The color of the displayed point (RGB or RGBA).
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

        self.size = size
        self._resolution = resolution

        self._color = [*color, 1] if len(color) == 3 else [*color]

    def _addToScene(self, scene:Scene2D, name:None|str):
        name = str(id(self)) if name is None else name
        shape = Circle(100 * self.x, 100 * self.y, self.size, self.resolution, tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer))
        self._resolution = shape._segments
        scene._shapeDict[name] = {"class": self, "shape": shape}

    def _update(self, shape:Circle, scene:Scene2D):
        shape.position = (100 * self.x, 100 * self.y)
        shape.radius = self.size
        shape.color = tuple(int(255 * _ + 0.5) for _ in self.color)

    def __eq__(self, other:"Point2D"|NDArray2|List2|Tuple2):
        if isinstance(other, Point2D):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, ndarray, list, tuple):
            return self.x == other[0] and self.y == other[1]
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
    def resolution(self) -> None|int:
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

    def distanceSq(self, p:"Point2D") -> Number:
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
        return (self.x - p.x) ** 2 + (self.y - p.y) ** 2
    
    def distance(self, p:"Point2D") -> float:
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
