"""
GFRAM Contribution Consent
==========================

Enrolled faces are sent to the GFRAM server only with the operator's consent.
The decision is stored once per installation and can be changed at any time:

    >>> import gfram
    >>> gfram.set_contribution_consent(True)    # share enrolled face data
    >>> gfram.set_contribution_consent(False)   # stop sharing

Each installation also gets a random installation id (not derived from any
person or machine data), so the server can tell identical names on different
installations apart.

Author: Ortiqova F.S.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

NOTICE = (
    "GFRAM: enrolled faces are NOT being shared with the GFRAM server. "
    "With your consent, each gfram.add() sends the person's name, 478 face landmarks, "
    "geometric features, geometric and appearance embeddings, photo quality and head pose "
    "(never the photo) to https://gfram.uz to improve the shared model. "
    "Make sure the people you enrol agree, then call gfram.set_contribution_consent(True)."
)
_notice_shown = False


def _consent_file():
    from .server_client import get_cache_dir
    return get_cache_dir() / 'consent.json'


def get_contribution_consent() -> Optional[bool]:
    """True / False once decided, None if the operator has not decided yet."""
    try:
        return bool(json.loads(_consent_file().read_text())['consent'])
    except Exception:
        return None


def set_contribution_consent(consent: bool) -> None:
    """Record the operator's decision about sharing enrolled face data."""
    _consent_file().write_text(json.dumps({
        'consent': bool(consent),
        'decided_at': datetime.now(timezone.utc).isoformat(),
    }))
    logger.info(f"Contribution consent set to {bool(consent)}")


def installation_id() -> str:
    """Random, persistent id of this installation (created on first use)."""
    from .server_client import get_cache_dir
    path = get_cache_dir() / 'installation_id'
    try:
        value = path.read_text().strip()
        if value:
            return value
    except OSError:
        pass
    value = uuid.uuid4().hex
    path.write_text(value)
    return value


def should_contribute(explicit: Optional[bool]) -> bool:
    """
    Explicit per-recognizer choice wins; otherwise the stored consent decides.
    Without any decision nothing is sent and a one-time notice explains how to opt in.
    """
    global _notice_shown
    if explicit is not None:
        return explicit
    consent = get_contribution_consent()
    if consent is None:
        if not _notice_shown:
            _notice_shown = True
            logger.warning(NOTICE)
            print(NOTICE)
        return False
    return consent
