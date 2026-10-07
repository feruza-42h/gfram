"""
GFRAM Server Client
===================

Client for GFRAM Model Server (https://gfram.uz)

Author: Ortiqova F.S.
"""

import os
import ssl
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Server URL
SERVER_URL = "https://gfram.uz"

# Cache directory
CACHE_DIR = Path.home() / '.gfram' / 'cache'


def get_cache_dir() -> Path:
    """Get cache directory"""
    cache_dir = Path(os.environ.get('GFRAM_CACHE_DIR', CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_server_url() -> str:
    """Get server URL"""
    return os.environ.get('GFRAM_SERVER_URL', SERVER_URL)


def _ssl_context():
    """
    SSL context backed by certifi when available: python.org builds on macOS ship
    without a CA store, so the default context fails with CERTIFICATE_VERIFY_FAILED.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class GFRAMClient:
    """GFRAM Server API Client"""
    
    def __init__(self, server_url: Optional[str] = None, timeout: float = 60):
        self.server_url = (server_url or get_server_url()).rstrip('/')
        self.timeout = timeout
        self._ssl = _ssl_context()
    
    def _request(
        self, 
        endpoint: str, 
        method: str = "GET",
        data: Optional[bytes] = None
    ) -> Dict:
        """Make HTTP request"""
        url = f"{self.server_url}{endpoint}"
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'GFRAM-Client/3.2'
        }
        
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            
            with urllib.request.urlopen(request, timeout=self.timeout, context=self._ssl) as response:
                content = response.read()
                try:
                    return json.loads(content.decode('utf-8'))
                except json.JSONDecodeError:
                    return {'status': 'success', 'raw': content}
                    
        except urllib.error.HTTPError as e:
            return {'status': 'error', 'code': e.code, 'message': e.reason}
        except urllib.error.URLError as e:
            return {'status': 'error', 'message': str(e.reason)}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def health(self) -> Dict:
        """Check server health"""
        return self._request('/api/health')
    
    def stats(self) -> Dict:
        """Get server stats"""
        return self._request('/api/stats')
    
    def model_info(self) -> Dict:
        """Get model info"""
        return self._request('/api/model')
    
    def download_model(self, output_path: Optional[Path] = None, force: bool = False) -> Optional[Path]:
        """Download model from server"""
        if output_path is None:
            output_path = get_cache_dir() / 'gfram_model.pth'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check cache
        if output_path.exists() and not force:
            logger.info(f"Model in cache: {output_path}")
            return output_path
        
        url = f"{self.server_url}/api/model/download"
        
        print(f"📥 Downloading GFRAM model...")
        print(f"   Server: {self.server_url}")
        
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'GFRAM-Client/3.2'})
            with urllib.request.urlopen(request, timeout=self.timeout, context=self._ssl) as response, \
                    open(output_path, 'wb') as f:
                total = int(response.headers.get('Content-Length') or 0)
                done = 0
                while True:
                    block = response.read(1 << 16)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    if total > 0:
                        percent = min(100, int(100 * done / total))
                        bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
                        print(f'\r   [{bar}] {percent}% ({done / 2**20:.1f}/{total / 2**20:.1f} MB)',
                              end='', flush=True)
            print(f"\n   ✅ Model saved: {output_path}")

            return output_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            print(f"\n   ❌ Error: {e}")
            if output_path.exists():
                output_path.unlink()
            return None

    def contribute(
        self,
        person_id: str,
        landmarks: np.ndarray,
        geometric_features: np.ndarray,
        embedding: Optional[np.ndarray] = None
    ) -> Dict:
        """Send data to server for training"""
        data = {
            "person_id": str(person_id),
            "landmarks": landmarks.tolist() if isinstance(landmarks, np.ndarray) else landmarks,
            "geometric_features": geometric_features.tolist() if isinstance(geometric_features, np.ndarray) else geometric_features,
        }
        
        if embedding is not None:
            data["embedding"] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        return self._request('/api/contribute', method='POST', data=json.dumps(data).encode('utf-8'))


# Global client
_client: Optional[GFRAMClient] = None


def get_client(timeout: Optional[float] = None) -> GFRAMClient:
    """Get global client (a custom timeout returns a dedicated client)"""
    global _client
    if timeout is not None:
        return GFRAMClient(timeout=timeout)
    if _client is None:
        _client = GFRAMClient()
    return _client


# Convenience functions
def download_model(force: bool = False, output_path: Optional[Path] = None) -> Optional[Path]:
    """Download model from server"""
    return get_client().download_model(output_path=output_path, force=force)


def contribute(
    person_id: str,
    landmarks: np.ndarray,
    geometric_features: np.ndarray,
    embedding: Optional[np.ndarray] = None
) -> Dict:
    """Send data to server"""
    return get_client().contribute(person_id, landmarks, geometric_features, embedding)


def server_health() -> Dict:
    """Check server health"""
    return get_client().health()


def server_stats() -> Dict:
    """Get server stats"""
    return get_client().stats()
