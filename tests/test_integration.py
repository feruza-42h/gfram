"""
Integration tests for GFRAM
"""

import pytest
import numpy as np
import torch


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for full pipeline"""
    
    def test_full_feature_extraction_pipeline(self, sample_landmarks_478):
        """Test full feature extraction pipeline"""
        from gfram.geometry.features import GeometricFeatureExtractor
        from gfram.geometry.landmarks import LandmarkNormalizer
        
        # Normalize
        normalizer = LandmarkNormalizer()
        normalized = normalizer.normalize(sample_landmarks_478)
        
        # Extract features
        extractor = GeometricFeatureExtractor()
        features = extractor.extract(normalized)
        
        assert features is not None
        assert len(features) > 100
        assert np.all(np.isfinite(features))
    
    def test_model_with_extracted_features(self, sample_landmarks_478):
        """Test model works with real landmarks"""
        from gfram.models import create_geometric_transformer
        from gfram.geometry.landmarks import LandmarkNormalizer
        
        # Normalize landmarks
        normalizer = LandmarkNormalizer()
        normalized = normalizer.normalize(sample_landmarks_478)
        
        # Create model
        model = create_geometric_transformer(config_name='tiny')
        model.eval()
        
        # Get embedding
        with torch.no_grad():
            landmarks_tensor = torch.FloatTensor(normalized).unsqueeze(0)
            embedding = model(landmarks_tensor)
        
        assert embedding is not None
        assert embedding.shape[0] == 1
    
    def test_hybrid_matching_pipeline(self, sample_landmarks_478):
        """Test hybrid matching with geometric and deep features"""
        from gfram.geometry.features import GeometricFeatureExtractor
        from gfram.geometry.landmarks import LandmarkNormalizer
        from gfram.models import create_geometric_transformer
        from gfram.matching import FaceIndex
        
        # Normalize
        normalizer = LandmarkNormalizer()
        normalized = normalizer.normalize(sample_landmarks_478)
        
        # Extract geometric features
        extractor = GeometricFeatureExtractor()
        geo_features = extractor.extract(normalized)
        
        # Get deep embedding
        model = create_geometric_transformer(config_name='tiny')
        model.eval()
        
        with torch.no_grad():
            landmarks_tensor = torch.FloatTensor(normalized).unsqueeze(0)
            deep_embedding = model(landmarks_tensor).numpy().squeeze()
        
        # Create indices
        geo_index = FaceIndex(metric='cosine')
        deep_index = FaceIndex(metric='cosine')
        
        # Add to indices
        metadata = {'name': 'test', 'person_id': 0}
        geo_index.add(geo_features, metadata=metadata)
        deep_index.add(deep_embedding, metadata=metadata)
        
        # Search
        geo_results = geo_index.search(geo_features, k=1)
        deep_results = deep_index.search(deep_embedding, k=1)
        
        assert len(geo_results) > 0 or len(deep_results) > 0


@pytest.mark.integration
class TestModelTraining:
    """Tests for model training components"""
    
    def test_loss_computation(self):
        """Test loss can be computed"""
        from gfram.models.losses import TripletLoss
        
        loss_fn = TripletLoss(margin=0.2)
        
        anchor = torch.randn(8, 256)
        positive = torch.randn(8, 256)
        negative = torch.randn(8, 256)
        
        loss = loss_fn(anchor, positive, negative)
        
        assert loss.item() >= 0
        assert torch.isfinite(loss)
    
    def test_model_backward_pass(self):
        """Test model supports backward pass"""
        from gfram.models import create_geometric_transformer
        
        model = create_geometric_transformer(config_name='tiny')
        model.train()
        
        landmarks = torch.randn(4, 478, 3, requires_grad=True)
        output = model(landmarks)
        
        # Compute dummy loss
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert landmarks.grad is not None


@pytest.mark.integration  
class TestCloudComponents:
    """Tests for cloud components"""
    
    def test_model_loader_import(self):
        """Test model loader can be imported"""
        from gfram.cloud.model_loader import get_model_path, ensure_model_available
        assert get_model_path is not None
        assert ensure_model_available is not None
    
    def test_server_client_import(self):
        """Test server client can be imported"""
        from gfram.cloud.server_client import server_health
        assert server_health is not None
    
    def test_model_cache_path(self):
        """Test model cache path is valid"""
        from gfram.cloud.model_loader import get_model_path
        
        path = get_model_path()
        
        assert path is not None
        # Path should be in user's home directory
        assert '.gfram' in str(path) or 'gfram' in str(path).lower()
