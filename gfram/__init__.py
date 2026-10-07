"""
GFRAM - Geometric Face Recognition and Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Professional face recognition library.

Simple usage:
    >>> import gfram
    >>> 
    >>> # Add person (model ships with the package; data is shared only with consent)
    >>> gfram.add("John", "john.jpg")
    >>> 
    >>> # Recognize
    >>> result = gfram.recognize("test.jpg")
    >>> print(result)
    {'name': 'John', 'confidence': 0.95}

Author: Ortiqova F.S.
License: MIT
Server: https://gfram.uz
"""

from .version import __version__, __author__

# ============================================================
# SIMPLE PUBLIC API
# ============================================================

# Global recognizer instance
_recognizer = None


def _get_recognizer():
    """Get or create global recognizer"""
    global _recognizer
    if _recognizer is None:
        from .api.simple_recognizer import SimpleRecognizer
        _recognizer = SimpleRecognizer()
    return _recognizer


def add(name: str, image):
    """
    Add a person for recognition.
    
    Automatically:
    - Downloads model from server (first time)
    - Detects face and extracts features
    - Adds to local database
    - With consent (gfram.set_contribution_consent), shares the face data with the server
    
    Args:
        name: Person's name
        image: Image path (str) or numpy array
    
    Returns:
        dict with result info
    
    Example:
        >>> gfram.add("John", "john.jpg")
        {'success': True, 'name': 'John', 'person_id': 0}
    """
    return _get_recognizer().add(name, image)


def recognize(image):
    """
    Recognize a person in image.
    
    Hybrid: geometric module + deep appearance model, combined by adaptive fusion.
    
    Args:
        image: Image path (str) or numpy array
    
    Returns:
        dict with recognition result
    
    Example:
        >>> result = gfram.recognize("test.jpg")
        >>> print(result)
        {'name': 'John', 'confidence': 0.95, 'recognized': True}
        
        >>> # Unknown person
        {'name': 'Unknown', 'confidence': 0.0, 'recognized': False}
    """
    return _get_recognizer().recognize(image)


def verify(image1, image2):
    """
    Check whether two photos show the same person.

    Does not use or change the database.

    Args:
        image1: Image path (str) or numpy array
        image2: Image path (str) or numpy array

    Returns:
        dict with verification result

    Example:
        >>> gfram.verify("photo_a.jpg", "photo_b.jpg")
        {'same_person': True, 'confidence': 0.97, 'threshold': 0.53}
    """
    return _get_recognizer().verify(image1, image2)


def list_persons():
    """
    List all added persons.
    
    Returns:
        List of person names
    
    Example:
        >>> gfram.list_persons()
        ['John', 'Alice', 'Bob']
    """
    return _get_recognizer().list_persons()


def remove(name: str):
    """
    Remove a person from recognition database.
    
    Args:
        name: Person's name
    
    Returns:
        True if removed, False if not found
    """
    return _get_recognizer().remove(name)


def clear():
    """
    Clear all persons from local database.
    """
    return _get_recognizer().clear()


def stats():
    """
    Get statistics.
    
    Returns:
        dict with stats
    
    Example:
        >>> gfram.stats()
        {'persons': 3, 'total_recognitions': 10, 'device': 'cpu'}
    """
    return _get_recognizer().stats()


def set_contribution_consent(consent: bool):
    """
    Allow (True) or stop (False) sharing enrolled face data with the GFRAM server.

    With consent, each gfram.add() sends the person's name, 478 face landmarks,
    geometric features, geometric and appearance embeddings, photo quality and
    head pose (never the photo) to https://gfram.uz to improve the shared model.
    Make sure the people you enrol agree. The decision is remembered.
    """
    from .cloud.consent import set_contribution_consent as _set
    _set(consent)


def contribution_consent():
    """
    Current sharing decision: True, False, or None if not decided yet
    (nothing is sent until consent is given).
    """
    from .cloud.consent import get_contribution_consent
    return get_contribution_consent()


def rotate_template_key():
    """
    Revoke the stored face templates: re-protect the local database under a new
    secret key (no photos needed). Copies of the database taken before the rotation
    can no longer be matched. Returns the new key fingerprint.
    """
    return _get_recognizer().rotate_template_key()


def server_status():
    """
    Check server status.
    
    Returns:
        dict with server info
    """
    from .cloud import server_health
    return server_health()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Simple API
    'add',
    'recognize',
    'verify',
    'list_persons',
    'remove',
    'clear',
    'stats',
    'set_contribution_consent',
    'contribution_consent',
    'rotate_template_key',
    'server_status',
]
