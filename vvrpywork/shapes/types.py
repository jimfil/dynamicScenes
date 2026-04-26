from beartype.vale import Is
from numbers import Number
from numpy import ndarray
from typing import Annotated

NDArray = ndarray
List = list
Tuple = tuple

NDArray2 = Annotated[ndarray, Is[lambda array: array.shape == (2,)]]
NDArray3 = Annotated[ndarray, Is[lambda array: array.shape == (3,)]]
NDArray4 = Annotated[ndarray, Is[lambda array: array.shape == (4,)]]
List2 = Annotated[list[Number], 2]
List3 = Annotated[list[Number], 3]
List4 = Annotated[list[Number], 4]
Tuple2 = tuple[Number, Number]
Tuple3 = tuple[Number, Number, Number]
Tuple4 = tuple[Number, Number, Number, Number]

ColorType = NDArray3|NDArray4|List3|List4|Tuple3|Tuple4