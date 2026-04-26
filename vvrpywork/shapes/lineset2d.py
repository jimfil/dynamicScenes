from .abstract import ShapeSet
from .types import NDArray, List, List2, Tuple, ColorType, Number
from .pointset2d import PointSet2D
from .line2d import Line2D
from vvrpywork.scene import Scene2D

import numpy as np
from pyglet.shapes import Line
from pyglet.graphics import Group


class LineSet2D(ShapeSet):
    '''A class used to represent a set of lines in 2D space.'''

    def __init__(self, points:None|NDArray|List|Tuple=None, lines:None|NDArray|List|Tuple=None, width:Number=1, color:ColorType=(0, 0, 0)):
        '''Inits LineSet2D.

        Inits a LineSet2D containing `points` connected according to
        `lines`.
        
        If `lines` is `None`, the `points` will be connected in pairs
        i.e., (0, 1), (2, 3), etc.
        
        If `points` is `None`, the lineset will be initialized empty.

        Args:
            points: The points of the lineset.
            lines: The indices in `points` that are connected by a
                line.
            width: The width of the displayed lines.
            color: The color of the displayed lines (RGB or RGBA).
        '''
        self._points:list[List2] = []
        self._lines:list[List2] = []

        self.width = width
        self._colors:list[ColorType] = []

        if isinstance(points, PointSet2D):
            points = points.points

        if points is not None and lines is None:
            if isinstance(points, (np.ndarray, list, tuple)) and len(points) % 2 == 0:
                lines = [[i,i+1] for i in range(0, len(points), 2)]
            else:
                raise RuntimeError("Attempted connecting point pairs, but points is not divisible by 2.")

        if points is not None and lines is not None:
            if isinstance(points, (np.ndarray, list, tuple)):
                self._points = [list(_) for _ in points]
            if isinstance(lines, (np.ndarray, list, tuple)):
                self._lines = [list(_) for _ in lines]

            self._colors = [[*color, 1] if len(color) == 3 else [*color] for _ in lines]

    def __len__(self):
        return len(self._lines)
    
    def __getitem__(self, idx):
        return self.getLineAt(idx)

    def _addToScene(self, scene:Scene2D, name:None|str):
        name = str(id(self)) if name is None else name
        lines = []
        for i, l in enumerate(self._lines):
            lines.append(Line(100 * self._points[l[0]][0], 100 * self._points[l[0]][1], 100 * self._points[l[1]][0], 100 * self._points[l[1]][1], self.width, color=tuple(int(255 * _ + 0.5) for _ in self.colors[i]), batch=scene._shapeBatch, group=Group(scene._layer)))
        scene._shapeDict[name] = {"class": self, "shape": lines}

    def _update(self, shape:list[Line], scene:Scene2D):
            if len(shape) == len(self._lines):
                for i, l in enumerate(shape):
                    l.position = (100 * self._points[self._lines[i][0]][0], 100 * self._points[self._lines[i][0]][1])
                    l.x2 = 100 * self._points[self._lines[i][1]][0]
                    l.y2 = 100 * self._points[self._lines[i][1]][1]
                    l.width = self.width
                    l.color = tuple(int(255 * _ + 0.5) for _ in self.colors[i])

            elif len(shape) < len(self._lines):
                for i, l in enumerate(shape):
                    l.position = (100 * self._points[self._lines[i][0]][0], 100 * self._points[self._lines[i][0]][1])
                    l.x2 = 100 * self._points[self._lines[i][1]][0]
                    l.y2 = 100 * self._points[self._lines[i][1]][1]
                    l.width = self.width
                    l.color = tuple(int(255 * _ + 0.5) for _ in self.colors[i])
                for i, l in enumerate(self._lines[len(shape):]):
                    
                    shape.append(Line(100 * self._points[l[0]][0], 100 * self._points[l[0]][1], 100 * self._points[l[1]][0], 100 * self._points[l[1]][1], self.width, color=tuple(int(255 * _ + 0.5) for _ in self.colors[len(shape) + i]), batch=scene._shapeBatch, group=Group(scene._layer)))
            
            else:  # len(shape) > len(self._lines)
                for i, l in enumerate(self._lines):
                    shape[i].position = (100 * self._points[l[0]][0], 100 * self._points[l[0]][1])
                    shape[i].x2 = 100 * self._points[l[1]][0]
                    shape[i].y2 = 100 * self._points[l[1]][1]
                    shape[i].width = self.width
                    shape[i].color = tuple(int(255 * _ + 0.5) for _ in self.colors[i])
                del shape[len(self._lines):]

    @property
    def points(self) -> NDArray:
        '''The points of the lineset.'''
        return np.array(self._points)
    
    @points.setter
    def points(self, pts:NDArray|List|Tuple):
        if isinstance(pts, (np.ndarray, list, tuple)):
            self._points = [list(_) for _ in pts]
    
    @property
    def lines(self) -> NDArray:
        '''The point indices indicating lines of the lineset.'''
        return np.array(self._lines)
    
    @lines.setter
    def lines(self, lns:NDArray|List|Tuple):
        if isinstance(lns, (np.ndarray, list, tuple)):
            self._lines = [list(_) for _ in lns]

    @property
    def width(self) -> Number:
        '''The width of the displayed lines.'''
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
    def colors(self) -> NDArray:
        '''The lines' colors in RGBA format.'''
        return np.array(self._colors)
    
    @colors.setter
    def colors(self, colors:NDArray|List|Tuple):
        if isinstance(colors, (np.ndarray, list, tuple)):
            self._colors = [list(_) for _ in colors]
        
    def getLineAt(self, index:int) -> Line2D:
        '''Returns the line at the specified index.

        Args:
            index: The index at which the desired line is placed
                inside the lineset.
        
        Returns:
            The line at the specified index as a `Line2D` object. It
                retains its width and color.
        '''
        return Line2D(self._points[self._lines[index][0]], self._points[self._lines[index][1]], self.width, self.colors[index])
        
    def add(self, line:Line2D):
        '''Appends a line to the lineset.
        
        Args:
            line: The `Line2D` object to append.
        '''
        idx1 = idx2 = None
        if len(self._points) == 0:
            self._points.append([line.x1, line.y1])

        for i, p in enumerate(self._points):
            if p[0] == line.x1 and p[1] == line.y1:
                idx1 = i
            if p[0] == line.x2 and p[1] == line.y2:
                idx2 = i

        if idx1 == None:
            self._points.append([line.x1, line.y1])
            idx1 = len(self._points) - 1
        if idx2 == None:
            self._points.append([line.x2, line.y2])
            idx2 = len(self._points) - 1

        self._lines.append([idx1, idx2])
        self._colors.append([*line.color])

    def remove(self, index:int):
        '''Removes a line from the lineset.

        Removes a line from the lineset's specified index (does not
        affect LineSet2D.points in any way).
        
        Args:
            index: The index at which the to-be-removed line is placed
                inside the lineset.
        '''
        self._lines.pop(index)
        self._colors.pop(index)

    def clear(self):
        '''Clears the lineset.

        Clears the lineset, completely removing all points, lines and
        information about them (e.g., color).
        '''
        self._points.clear()
        self._lines.clear()
        self._colors.clear()
