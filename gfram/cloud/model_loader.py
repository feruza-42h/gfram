"""
GFRAM Model Loader
==================

Resolves which model file to use, in order:
1. cached model (~/.gfram/cache/gfram_model.pth) if valid and not older than the server's
2. latest model downloaded from https://gfram.uz
3. model bundled with the package (gfram/pretrained/gfram_model.pth)

Every candidate is validated as a gfram-model/2 package before use, so a stale
or corrupt file can never silently load as random weights.

Author: Ortiqova F.S.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BUNDLED_MODEL = Path(__file__).parent.parent / 'pretrained' / 'gfram_model.pth'
CHECK_INTERVAL = 24 * 3600  # ask the server for a newer model at most once a day


def get_model_path() -> Path:
    """Get model path in cache"""
    from .server_client import get_cache_dir
    return get_cache_dir() / 'gfram_model.pth'


def _package_version(path: Path) -> Optional[str]:
    """Model version if path is a valid package, else None."""
    from ..models.package import read_package, ModelPackageError
    if not path.exists():
        return None
    try:
        return read_package(path)['model_version']
    except ModelPackageError as e:
        logger.warning(f'Ignoring invalid model file: {e}')
        return None


def _version_tuple(v: str):
    return tuple(int(p) if p.isdigit() else 0 for p in str(v).split('.'))


def _server_version() -> Optional[str]:
    from .server_client import get_client
    info = get_client(timeout=5).model_info()
    if not isinstance(info, dict):
        return None
    return info.get('model_version') or info.get('version')


def _check_due(stamp: Path) -> bool:
    try:
        return time.time() - json.loads(stamp.read_text())['checked_at'] > CHECK_INTERVAL
    except Exception:
        return True


def _download_validated(dest: Path) -> Optional[str]:
    """Download into a temp file and only replace dest if the result is a valid package."""
    from .server_client import download_model
    tmp = dest.with_suffix('.download')
    try:
        if download_model(force=True, output_path=tmp) is None:
            return None
        version = _package_version(tmp)
        if version is None:
            return None
        tmp.replace(dest)
        return version
    finally:
        if tmp.exists():
            tmp.unlink()


def ensure_model_available(force_download: bool = False) -> Optional[Path]:
    """
    Return the path of the best available valid model package.

    Args:
        force_download: Ask the server even if the cache was checked recently.
    """
    cached = get_model_path()
    stamp = cached.with_name('model_check.json')
    cached_version = _package_version(cached)
    bundled_version = _package_version(BUNDLED_MODEL)
    local_version = max([v for v in (cached_version, bundled_version) if v], key=_version_tuple, default=None)

    # Contact the server at most once per CHECK_INTERVAL, unless there is no usable model at all
    if force_download or local_version is None or _check_due(stamp):
        server_version = None
        try:
            server_version = _server_version()
        except Exception as e:
            logger.debug(f'Model server unreachable: {e}')

        newer = (server_version is not None and
                 (local_version is None or _version_tuple(server_version) > _version_tuple(local_version)))
        if force_download or newer or local_version is None:
            try:
                downloaded = _download_validated(cached)
                if downloaded:
                    cached_version = downloaded
                    logger.info(f'Model {downloaded} downloaded from server')
            except Exception as e:
                logger.warning(f'Could not download model: {e}')

        try:
            stamp.write_text(json.dumps({'checked_at': time.time(), 'server_version': server_version}))
        except OSError:
            pass

    # Prefer whichever valid model is newer: a fresh install can ship a newer
    # bundled model than an old cache, and the server can publish newer than both
    if cached_version and (bundled_version is None or
                           _version_tuple(cached_version) >= _version_tuple(bundled_version)):
        return cached
    if bundled_version:
        logger.info(f'Using bundled model {bundled_version}')
        return BUNDLED_MODEL

    logger.error('No valid GFRAM model found (server unreachable and no bundled model)')
    return None


def clear_cache():
    """Clear model cache"""
    from .server_client import get_cache_dir

    cache_dir = get_cache_dir()

    for f in cache_dir.glob('*.pth'):
        f.unlink()
        logger.info(f"Deleted: {f}")

    print(f"✅ Cache cleared: {cache_dir}")
