"""
GFRAM Protected Templates
=========================

Cancelable, key-bound face templates for the local database.

* Embeddings (geometric 128-d, appearance 512-d) are stored only after a secret
  orthogonal rotation R derived from the installation key. Rotations preserve dot
  products, so every similarity — and therefore the adaptive fusion's match
  probability, which depends on embeddings only through similarities and stored
  scalar quality cues — is exactly unchanged.
* The 478 landmarks (kept so templates can be re-computed after a model update)
  are encrypted with AES-256-GCM under a key derived from the same secret.
* Revocation: rotating the key re-protects the whole database without photos;
  templates stolen before the rotation no longer match anything.

Protection relies on the secrecy of the key: the key file is kept outside the
database directory with owner-only permissions. A stolen database without the key
reveals neither face shape nor comparable embeddings.

Author: Ortiqova F.S.
"""

import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Optional

import numpy as np

KEY_BYTES = 32
LANDMARKS_AAD = b'gfram-landmarks-v1'


# ---------------------------------------------------------------- key storage

def default_key_path() -> Path:
    """~/.gfram/keys/template.key (or $GFRAM_KEY_DIR/template.key)."""
    from .cloud.server_client import get_cache_dir
    key_dir = Path(os.environ.get('GFRAM_KEY_DIR', get_cache_dir().parent / 'keys'))
    return key_dir / 'template.key'


def load_or_create_key(path: Optional[Path] = None) -> bytes:
    """Read the installation key, creating a random one with owner-only permissions."""
    path = Path(path) if path else default_key_path()
    if path.exists():
        key = bytes.fromhex(path.read_text().strip())
        if len(key) != KEY_BYTES:
            raise ValueError(f'Template key in {path} is corrupt')
        return key
    key = secrets.token_bytes(KEY_BYTES)
    write_key(key, path)
    return key


def write_key(key: bytes, path: Optional[Path] = None) -> None:
    path = Path(path) if path else default_key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    tmp = path.with_suffix('.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(key.hex())
    tmp.replace(path)


# ---------------------------------------------------------------- derivation

def _subkey(key: bytes, label: str) -> bytes:
    return hmac.new(key, label.encode(), hashlib.sha256).digest()


def _keystream_normals(seed: bytes, count: int) -> np.ndarray:
    """
    Standard normals from SHA-256 in counter mode (Box-Muller). Deterministic across
    numpy versions, unlike numpy's random generators, so a key always maps to the
    same rotation.
    """
    n_words = count + (count % 2)
    raw = bytearray()
    block = 0
    while len(raw) < 4 * n_words:
        raw += hashlib.sha256(seed + block.to_bytes(8, 'big')).digest()
        block += 1
    words = np.frombuffer(bytes(raw[:4 * n_words]), dtype='>u4').astype(np.float64)
    u = (words + 0.5) / 2.0 ** 32  # in (0, 1)
    u1, u2 = u[0::2], u[1::2]
    r = np.sqrt(-2.0 * np.log(u1))
    z = np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])
    return z[:count]


def keyed_rotation(key: bytes, dim: int, label: str) -> np.ndarray:
    """Haar-distributed random orthogonal matrix (dim x dim) determined by key and label."""
    a = _keystream_normals(_subkey(key, f'rotation:{label}:{dim}'), dim * dim).reshape(dim, dim)
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))  # unique Q for a full-rank matrix: sign convention of R fixed
    return q.astype(np.float32)


# ---------------------------------------------------------------- protector

class TemplateProtector:
    """Applies the installation key to embeddings and landmarks."""

    def __init__(self, key: bytes):
        if len(key) != KEY_BYTES:
            raise ValueError('Template key must be 32 bytes')
        self._key = key
        self._rotations = {}
        self.fingerprint = _subkey(key, 'fingerprint').hex()[:16]

    def _rotation(self, kind: str, dim: int) -> np.ndarray:
        if (kind, dim) not in self._rotations:
            self._rotations[(kind, dim)] = keyed_rotation(self._key, dim, kind)
        return self._rotations[(kind, dim)]

    def protect(self, embedding: np.ndarray, kind: str) -> np.ndarray:
        """kind: 'geo' or 'app'. Works on (D,) or (N, D)."""
        e = np.asarray(embedding, dtype=np.float32)
        return (e @ self._rotation(kind, e.shape[-1]).T).astype(np.float32)

    def unprotect(self, template: np.ndarray, kind: str) -> np.ndarray:
        t = np.asarray(template, dtype=np.float32)
        return (t @ self._rotation(kind, t.shape[-1])).astype(np.float32)

    def encrypt_landmarks(self, landmarks: np.ndarray) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = secrets.token_bytes(12)
        data = np.asarray(landmarks, dtype=np.float32).tobytes()
        return nonce + AESGCM(_subkey(self._key, 'landmarks-aes')).encrypt(nonce, data, LANDMARKS_AAD)

    def decrypt_landmarks(self, blob: bytes, shape=(478, 3)) -> np.ndarray:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce, ciphertext = blob[:12], blob[12:]
        data = AESGCM(_subkey(self._key, 'landmarks-aes')).decrypt(nonce, ciphertext, LANDMARKS_AAD)
        return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
