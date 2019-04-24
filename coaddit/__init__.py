__version__='0.1.0'

from . import polyclip
from .polyclip import (
    is_simple_poly,
    poly_area,
    clip_poly,
)

from . import rasterize
from .rasterize import rasterize_poly
