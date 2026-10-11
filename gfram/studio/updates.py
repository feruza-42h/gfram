"""Installed versions, update checks (PyPI, gfram.uz model server) and the self-update command."""

import json
import platform
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from . import ROOT, STUDIO_HOME, __version__

PYPI_URL = 'https://pypi.org/pypi/gfram/json'
PACKAGE_SPEC = 'gfram[studio]'


def version_tuple(v: str):
    out = []
    for part in str(v).split('.'):
        num = ''.join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def is_newer(latest: Optional[str], current: str) -> bool:
    return bool(latest) and version_tuple(latest) > version_tuple(current)


def source_checkout() -> bool:
    """True when running from the git repository (developer machine) instead of a pip install."""
    import gfram
    return (ROOT / '.git').exists() and Path(gfram.__file__).resolve().is_relative_to(ROOT)


def system_name() -> str:
    if sys.platform == 'darwin':
        return f'macOS {platform.mac_ver()[0]} ({platform.machine()})'
    if sys.platform == 'win32':
        return f'Windows {platform.release()} ({platform.version()}, {platform.machine()})'
    return f'{platform.system()} {platform.release()} ({platform.machine()})'


def installed_info(engine=None) -> Dict:
    import gfram
    info = {
        'version': __version__,
        'python': platform.python_version(),
        'system': system_name(),
        'location': str(Path(gfram.__file__).resolve().parent),
        'interpreter': sys.executable,
        'data': str(STUDIO_HOME),
        'source': source_checkout(),
    }
    if engine is not None:
        info['geometry_model'] = engine.embedder.version
        info['appearance_model'] = engine.fusion.appearance_model
    return info


def _ssl_context():
    try:
        from ..cloud.server_client import _ssl_context as ctx
        return ctx()
    except Exception:
        return None


def latest_pypi(timeout: float = 6.0) -> Dict:
    """{'version', 'date', 'url'} of the newest gfram release on PyPI."""
    req = urllib.request.Request(PYPI_URL, headers={'User-Agent': f'GFRAM-Studio/{__version__}'})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        data = json.loads(r.read())
    version = data['info']['version']
    files = data.get('releases', {}).get(version) or []
    date = files[0]['upload_time'][:10] if files else ''
    return {'version': version, 'date': date, 'url': f'https://pypi.org/project/gfram/{version}/'}


def model_server_status() -> Dict:
    """Newest geometric model on gfram.uz and whether the server answers."""
    from ..cloud.model_loader import _server_version
    try:
        return {'online': True, 'version': _server_version()}
    except Exception as e:
        return {'online': False, 'error': str(e)}


def update_model() -> Optional[str]:
    """Ask gfram.uz for a newer geometric model now; returns the version in use afterwards."""
    from ..cloud.model_loader import _package_version, ensure_model_available
    path = ensure_model_available(force_download=True)
    return _package_version(path) if path else None


def pip_command():
    return [sys.executable, '-m', 'pip', 'install', '--upgrade', '--disable-pip-version-check', PACKAGE_SPEC]
