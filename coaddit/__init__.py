__version__ = '0.1.0'  # noqa

from . import polyclip  # noqa
from .polyclip import (  # noqa
    is_simple_poly,
    poly_area,
    clip_poly,
)

from . import rasterize  # noqa
from .rasterize import rasterize_poly  # noqa
