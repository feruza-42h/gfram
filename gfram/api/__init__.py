"""
High-level API for face recognition.
"""

from .recognizer import Recognizer
from .online_recognizer import OnlineRecognizer


__all__ = [
    'Recognizer',
    'OnlineRecognizer',
]