from .abstract import Shape
from .types import NDArray2, List2, Tuple2, ColorType, Number
from .point2d import Point2D
from .circle2d import Circle2D
from vvrpywork.scene import Scene2D

from numpy import ndarray
from pyglet.shapes import Triangle, Line
from pyglet.graphics import Group


class Triangle2D(Shape):
    '''A class used to represent a triangle in 2D space.'''

    def __init__(self, p1:Point2D|NDArray2|List2|Tuple2, p2:Point2D|NDArray2|List2|Tuple2, p3:Point2D|NDArray2|List2|Tuple2, width:Number=1, color:ColorType=(0, 0, 0), filled:bool=False):
        '''Inits Triangle2D given the triangle's 3 vertices.

        Args:
            p1: The coordinates of the first vertex.
            p2: The coordinates of the second vertex.
            p3: The coordinates of the third vertex.
            width: The width of the displayed triangle (if not filled).
            color: The color of the displayed triangle (RGB or RGBA).
            filled: Whether to fill in the triangle or draw only its
                outline.
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
        
        if isinstance(p3, Point2D):
            self._x3 = p3.x
            self._y3 = p3.y
        elif isinstance(p3, (list, tuple)):
            self._x3 = p3[0]
            self._y3 = p3[1]
        elif isinstance(p3, ndarray):
            self._x3 = p3[0].item()
            self._y3 = p3[1].item()
        else:
            raise TypeError("Incorrect type for p3")

        self.width = width
        self._color = [*color, 1] if len(color) == 3 else [*color]
        self._filled = filled

    def _addToScene(self, scene:Scene2D, name:None|str):
        if self.filled:
            name = str(id(self)) if name is None else name
            shape = Triangle(100 * self.x1, 100 * self.y1, 100 * self.x2, 100 * self.y2, 100 * self.x3, 100 * self.y3, tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer))
            scene._shapeDict[name] = {"class": self, "shape": shape}
        else:
            line1 = Line(100 * self.x1, 100 * self.y1, 100 * self.x2, 100 * self.y2, self.width, tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer))
            line2 = Line(100 * self.x2, 100 * self.y2, 100 * self.x3, 100 * self.y3, self.width, tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer))
            line3 = Line(100 * self.x3, 100 * self.y3, 100 * self.x1, 100 * self.y1, self.width, tuple(int(255 * _ + 0.5) for _ in self.color), batch=scene._shapeBatch, group=Group(scene._layer))
            scene._shapeDict[name] = {"class": self, "shape": (line1, line2, line3)}

    def _update(self, shape:Triangle|tuple[Line, Line, Line], scene:Scene2D):
        if self.filled:
            shape.position = (100 * self.x1, 100 * self.y1)
            shape.x2 = 100 * self.x2
            shape.y2 = 100 * self.y2
            shape.x3 = 100 * self.x3
            shape.y3 = 100 * self.y3
            shape.color = tuple(int(255 * _ + 0.5) for _ in self.color)
        else:
            line1, line2, line3 = shape

            line1.position = (100 * self.x1, 100 * self.y1)
            line1.x2 = 100 * self.x2
            line1.y2 = 100 * self.y2
            line1.width = self.width
            line1.color = tuple(int(255 * _ + 0.5) for _ in self.color)

            line2.position = (100 * self.x2, 100 * self.y2)
            line2.x2 = 100 * self.x3
            line2.y2 = 100 * self.y3
            line2.width = self.width
            line2.color = tuple(int(255 * _ + 0.5) for _ in self.color)

            line3.position = (100 * self.x3, 100 * self.y3)
            line3.x2 = 100 * self.x1
            line3.y2 = 100 * self.y1
            line3.width = self.width
            line3.color = tuple(int(255 * _ + 0.5) for _ in self.color)

    @property
    def x1(self) -> Number:
        '''The x-coordinate of the first vertex.'''
        return self._x1
    
    @x1.setter
    def x1(self, x:Number):
        try:
            x = x.item()
        except:
            pass
        finally:
            self._x1 = x

    @property
    def y1(self) -> Number:
        '''The y-coordinate of the first vertex.'''
        return self._y1
    
    @y1.setter
    def y1(self, y:Number):
        try:
            y = y.item()
        except:
            pass
        finally:
            self._y1 = y
    
    @property
    def x2(self) -> Number:
        '''The x-coordinate of the second vertex.'''
        return self._x2
    
    @x2.setter
    def x2(self, x:Number):
        try:
            x = x.item()
        except:
            pass
        finally:
            self._x2 = x

    @property
    def y2(self) -> Number:
        '''The y-coordinate of the second vertex.'''
        return self._y2
    
    @y2.setter
    def y2(self, y:Number):
        try:
            y = y.item()
        except:
            pass
        finally:
            self._y2 = y
    
    @property
    def x3(self) -> Number:
        '''The x-coordinate of the third vertex.'''
        return self._x3
    
    @x3.setter
    def x3(self, x:Number):
        try:
            x = x.item()
        except:
            pass
        finally:
            self._x3 = x

    @property
    def y3(self) -> Number:
        '''The y-coordinate of the third vertex.'''
        return self._y3
    
    @y3.setter
    def y3(self, y:Number):
        try:
            y = y.item()
        except:
            pass
        finally:
            self._y3 = y
    
    @property
    def width(self) -> Number:
        '''The triangle's width (if not filled).'''
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
        '''The triangle's color in RGBA format.'''
        return self._color
    
    @color.setter
    def color(self, color:ColorType):
        self._color = [*color, 1] if len(color) == 3 else [*color]

    @property
    def filled(self) -> bool:
        '''Whether to fill in the triangle or draw only its outline.'''
        return self._filled

    def getPoint1(self) -> Point2D:
        '''Returns the triangle's first vertex.
        
        Returns:
            The triangle's first vertex as a `Point2D` object.
        '''
        return Point2D((self.x1, self.y1))
    
    def getPoint2(self) -> Point2D:
        '''Returns the triangle's second vertex.
        
        Returns:
            The triangle's second vertex as a `Point2D` object.
        '''
        return Point2D((self.x2, self.y2))
    
    def getPoint3(self) -> Point2D:
        '''Returns the triangle's third vertex.
        
        Returns:
            The triangle's third vertex as a `Point2D` object.
        '''
        return Point2D((self.x3, self.y3))
    
    def getCircumCircle(self) -> Circle2D:
        '''Returns the triangle's circumcircle.
        
        Returns:
            The triangle's circumcircle as a `Circle2D` object.
        '''
        # https://en.wikipedia.org/wiki/Circumcircle#Circumcenter_coordinates
        ax = self.x1
        ay = self.y1
        bx = self.x2
        by = self.y2
        cx = self.x3
        cy = self.y3

        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = 1 / D * ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by **2) * (cy - ay) + (cx ** 2 + cy **2) * (ay - by))
        uy = 1 / D * ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by **2) * (ax - cx) + (cx ** 2 + cy **2) * (bx - ax))
        r = ((ax - ux) ** 2 + (ay - uy) ** 2) ** .5

        return Circle2D((ux, uy), r)
    
    def contains(self, point:Point2D) -> bool:
        '''Determines whether a point is inside the triangle.

        Args:
            point: The point to check (if it's inside the triangle).

        Returns:
            `True` if the point is inside the triangle (incl. the
                edges), `False` otherwise.
        '''
        # https://totologic.blogspot.com/2014/01/accurate-point-in-triangle-test.html
        x = point.x
        y = point.y
        x1 = self.x1
        y1 = self.y1
        x2 = self.x2
        y2 = self.y2
        x3 = self.x3
        y3 = self.y3

        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        c = 1 - a - b

        return (a >= 0 and b >= 0 and c >= 0 and a <= 1 and b <= 1 and c <= 1)
