"""
Pytest fixtures for GFRAM tests
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check for optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@pytest.fixture
def sample_landmarks_478():
    """Generate sample 478 landmarks (3D)"""
    np.random.seed(42)
    landmarks = np.random.randn(478, 3).astype(np.float32)
    # Normalize to reasonable range
    landmarks = landmarks * 0.1 + 0.5
    return landmarks


@pytest.fixture
def sample_landmarks_468():
    """Generate sample 468 landmarks (3D)"""
    np.random.seed(42)
    landmarks = np.random.randn(468, 3).astype(np.float32)
    landmarks = landmarks * 0.1 + 0.5
    return landmarks


@pytest.fixture
def sample_image():
    """Generate sample RGB image"""
    np.random.seed(42)
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def geometric_extractor():
    """Create GeometricFeatureExtractor instance"""
    from gfram.geometry.features import GeometricFeatureExtractor
    return GeometricFeatureExtractor(num_landmarks=478)


# Conditional fixtures for PyTorch
if HAS_TORCH:
    import torch
    
    @pytest.fixture
    def batch_landmarks():
        """Generate batch of landmarks for model testing"""
        np.random.seed(42)
        batch_size = 4
        landmarks = np.random.randn(batch_size, 478, 3).astype(np.float32)
        return torch.FloatTensor(landmarks)
    
    @pytest.fixture
    def device():
        """Get available device"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def transformer_model():
        """Create GeometricTransformer model"""
        from gfram.models import create_geometric_transformer
        model = create_geometric_transformer(config_name='tiny', num_landmarks=478)
        model.eval()
        return model
