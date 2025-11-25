"""
GFRAM - Geometric Face Recognition and Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A professional face recognition library based on geometric features and custom AI models.

Basic usage:
   >>> import gfram
   >>> recognizer = gfram.Recognizer.from_pretrained('gfram-base')
   >>> result = recognizer.recognize('photo.jpg')

:copyright: (c) 2024 by Ortiqova F.S.
:license: MIT, see LICENSE for more details.
"""

from .version import __version__, __author__, __license__, __description__

# Detectors
from .detectors import FaceDetector, LandmarkNormalizer

# Geometry
from .geometry.features import GeometricFeatureExtractor


# Models
from .models import (
    GeometricTransformer,
    GeometricGNN,
    create_geometric_transformer,
    create_geometric_gnn,
    TripletLoss,
    ArcFaceLoss,
    CosFaceLoss,
    CombinedLoss
)
from .api.online_recognizer import OnlineRecognizer

# Matching
from .matching import FaceIndex, DistanceMetrics

# High-level API
from .api import Recognizer

__all__ = [
    # Version
    '__version__',
    '__author__',
    '__license__',
    '__description__',

    # Detectors
    'FaceDetector',
    'LandmarkNormalizer',

    # Geometry
    'GeometricFeatureExtractor',

    # Models
    'GeometricTransformer',
    'GeometricGNN',
    'create_geometric_transformer',
    'create_geometric_gnn',
    'TripletLoss',
    'ArcFaceLoss',
    'CosFaceLoss',
    'CombinedLoss',

    # Matching
    'FaceIndex',
    'DistanceMetrics',

    # High-level API
    'Recognizer',
    'OnlineRecognizer',

]