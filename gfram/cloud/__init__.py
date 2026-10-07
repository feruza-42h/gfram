"""
GFRAM Cloud Module
==================

Server integration for https://gfram.uz

Author: Ortiqova F.S.
"""

from .server_client import (
    GFRAMClient,
    download_model,
    contribute,
    server_health,
    server_stats,
    get_cache_dir
)

from .consent import (
    get_contribution_consent,
    set_contribution_consent,
    installation_id,
)

from .model_loader import (
    ensure_model_available,
    get_model_path,
    clear_cache
)

__all__ = [
    'GFRAMClient',
    'download_model',
    'contribute',
    'server_health',
    'server_stats',
    'get_cache_dir',
    'ensure_model_available',
    'get_model_path',
    'clear_cache',
    'get_contribution_consent',
    'set_contribution_consent',
    'installation_id',
]
