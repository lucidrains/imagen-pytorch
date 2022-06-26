from typing import List, Tuple, Union
from .type_vars import T

_TupleOrList = Union[Tuple[T, ...], List[T]]
_TupleOrListOrSingle = Union[_TupleOrList[T], T]
