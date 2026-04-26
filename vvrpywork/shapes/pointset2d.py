from .abstract import ShapeSet
from .types import NDArray, List, List2, Tuple, ColorType, Number
from .point2d import Point2D
from .rectangle2d import Rectangle2D
from .circle2d import Circle2D
from vvrpywork.scene import Scene2D

from math import sin, cos
import numpy as np
from pyglet.shapes import Circle
from pyglet.graphics import Group
import random


class PointSet2D(ShapeSet):
    '''A class used to represent a set of points in 2D space.'''

    def __init__(self, points:None|NDArray|List|Tuple=None, size:Number=1, color:ColorType=(0, 0, 0)):
        '''Inits PointSet2D.

        Inits a PointSet2D containing `points`. If `points` is `None`,
        the pointset will be initialized empty.

        Args:
            points: The points of the pointset.
            size: The size of the displayed points.
            color: The color of the displayed points (RGB or RGBA).
        '''
        self._points:list[List2] = []
        self.size = size
        self._colors:list[ColorType] = []

        if points is not None:
            if isinstance(points, (np.ndarray, list, tuple)):
                self._points = [list(_) for _ in points]
                self._colors = [[*color, 1] if len(color) == 3 else [*color] for _ in points]
            else:
                raise TypeError(f"Unsupported type for points: {type(points)}")

    def __len__(self):
        return len(self._points)
    
    def __getitem__(self, idx):
        return self.getPointAt(idx)

    def _addToScene(self, scene:Scene2D, name:None|str):
        name = str(id(self)) if name is None else name
        points = []
        for p, c in zip(self._points, self._colors):
            points.append(Circle(100 * p[0], 100 * p[1], self.size, color=tuple(int(255 * _ + 0.5) for _ in c), batch=scene._shapeBatch, group=Group(scene._layer)))
        scene._shapeDict[name] = {"class": self, "shape": points}

    def _update(self, shape:list[Circle], scene:Scene2D):
            if len(shape) == len(self._points):
                for i, p in enumerate(shape):
                    p.position = (100 * self._points[i][0], 100 * self._points[i][1])
                    p.radius = self.size
                    p.color = tuple(int(255 * _ + 0.5) for _ in self._colors[i])

            elif len(shape) < len(self._points):
                for i, p in enumerate(shape):
                    p.position = (100 * self._points[i][0], 100 * self._points[i][1])
                    p.radius = self.size
                    p.color = tuple(int(255 * _ + 0.5) for _ in self._colors[i])
                for p, c in zip(self._points[len(shape):], self._colors[len(shape):]):
                    shape.append(Circle(100 * p[0], 100 * p[1], self.size, color=tuple(int(255 * _ + 0.5) for _ in c), batch=scene._shapeBatch, group=Group(scene._layer)))
            
            else:  # len(shape) > len(self._points)
                for i, p in enumerate(self._points):
                    shape[i].position = (100 * p[0], 100 * p[1])
                    shape[i].radius = self.size
                    shape[i].color = tuple(int(255 * _ + 0.5) for _ in self._colors[i])
                del shape[len(self._points):]

    @property
    def points(self) -> NDArray:
        '''The points of the pointset.'''
        return np.array(self._points)
    
    @points.setter
    def points(self, points:NDArray|List|Tuple):
        if isinstance(points, (np.ndarray, list, tuple)):
            self._points = [list(_) for _ in points]
    
    @property
    def size(self) -> Number:
        '''The size of the displayed points.'''
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
    def colors(self) -> NDArray:
        '''The points' colors in RGBA format.'''
        return np.array(self._colors)
    
    @colors.setter
    def colors(self, colors:NDArray|List|Tuple):
        if isinstance(colors, (np.ndarray, list, tuple)):
            self._colors = [list(_) for _ in colors]

    def getPointAt(self, index:int) -> Point2D:
        '''Returns the point at the specified index.

        Args:
            index: The index at which the desired point is placed
                inside the pointset.
        
        Returns:
            The point at the specified index as a `Point2D` object. It
                retains its size and color.
        '''
        return Point2D(self._points[index], self.size, color=self._colors[index])

    def add(self, point:Point2D):
        '''Appends a point to the pointset.
        
        Args:
            point: The `Point2D` object to append.
        '''
        self._points.append([point.x, point.y])
        self._colors.append([*point.color])

    def createRandom(self, bound:Rectangle2D|Circle2D, num_points:int, seed:None|int|str=None, color:ColorType=(0, 0, 0)):
        '''Appends random points to the pointset.

        Uniformly generates random points inside a region and appends
        them to the pointset.
        
        Args:
            bound: The area inside of which the random points will be
                generated.
            num_points: How many points to generate.
            seed: An optional seed for the RNG.
            color: The color of the generated points.
        '''
        if len(self._points) > 0:
            print('Point Set is not empty; random points will be appended to the existing ones.')

        if seed:
            random.seed(seed)

        if isinstance(bound, Rectangle2D):
            x1 = bound.x_min
            y1 = bound.y_min
            x2 = bound.x_max
            y2 = bound.y_max
            
            for _ in range(num_points):
                x = random.uniform(x1, x2)
                y = random.uniform(y1, y2)
                self._points.append([x, y])
                self._colors.append([*color, 1] if len(color) == 3 else [*color])

        elif isinstance(bound, Circle2D):
            # https://stackoverflow.com/a/50746409
            centerX = bound.x
            centerY = bound.y
            R = bound.radius

            for _ in range(num_points):
                r = R * random.uniform(0, 1) ** .5
                theta = random.uniform(0, 1) * 2 * np.pi

                x = centerX + r * cos(theta)
                y = centerY + r * sin(theta)
                self._points.append([x, y])
                self._colors.append([*color, 1] if len(color) == 3 else [*color])

        else:
            raise TypeError("Unsupported bound type")
        
    def remove(self, index:int):
        '''Removes a point from the pointset.

        Removes a point from the pointset's specified index.
        
        Args:
            index: The index at which the to-be-removed point is placed
                inside the pointset.
        '''
        self._points.pop(index)
        self._colors.pop(index)

    def clear(self):
        '''Clears the pointset.

        Clears the pointset, completely removing all points and
        information about them (e.g., color).
        '''
        self._points.clear()
        self._colors.clear()
