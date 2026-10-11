"""
GFRAM Studio: desktop application of the gfram hybrid face recognition system.

Install with ``pip install "gfram[studio]"`` and start with ``gfram-studio``
(or ``python -m gfram.studio``).
"""

from pathlib import Path

from ..version import __version__  # the studio ships with the library and shares its version

ROOT = Path(__file__).resolve().parents[2]          # repository root when run from source
STUDIO_HOME = Path.home() / '.gfram' / 'studio'     # studio database, key, cards and journal

__all__ = ['__version__', 'ROOT', 'STUDIO_HOME']
