"""
Tests for gfram.geometry module
"""

import pytest
import numpy as np


class TestGeometricFeatureExtractor:
    """Tests for GeometricFeatureExtractor class"""
    
    def test_extractor_initialization(self):
        """Test extractor initializes correctly"""
        from gfram.geometry.features import GeometricFeatureExtractor
        
        extractor = GeometricFeatureExtractor(num_landmarks=478)
        assert extractor.num_landmarks == 478
        assert extractor.extract_iris == True
    
    def test_extractor_468_mode(self):
        """Test extractor in 468 landmark mode"""
        from gfram.geometry.features import GeometricFeatureExtractor
        
        extractor = GeometricFeatureExtractor(num_landmarks=468, extract_iris=False)
        assert extractor.num_landmarks == 468
        assert extractor.extract_iris == False
    
    def test_extract_features_478(self, sample_landmarks_478):
        """Test feature extraction with 478 landmarks"""
        from gfram.geometry.features import GeometricFeatureExtractor
        
        extractor = GeometricFeatureExtractor(num_landmarks=478)
        features = extractor.extract(sample_landmarks_478)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) > 100  # Should have 150+ features
    
    def test_extract_features_468(self, sample_landmarks_468):
        """Test feature extraction with 468 landmarks"""
        from gfram.geometry.features import GeometricFeatureExtractor
        
        extractor = GeometricFeatureExtractor(num_landmarks=468, extract_iris=False)
        features = extractor.extract(sample_landmarks_468)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features) >= 150
    
    def test_feature_count(self):
        """Test feature count method"""
        from gfram.geometry.features import GeometricFeatureExtractor
        
        extractor = GeometricFeatureExtractor(num_landmarks=478)
        count = extractor.get_feature_count()
        
        # 30 + 40 + 20 + 30 + 15 + 15 + 3 = 153
        assert count == 153
    
    def test_feature_names(self):
        """Test feature names method"""
        from gfram.geometry.features import GeometricFeatureExtractor
        
        extractor = GeometricFeatureExtractor(num_landmarks=478)
        names = extractor.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) == extractor.get_feature_count()
        assert 'left_iris_radius' in names
        assert 'right_iris_radius' in names
    
    def test_invalid_landmarks_shape(self):
        """Test error on invalid landmark shape"""
        from gfram.geometry.features import GeometricFeatureExtractor
        
        extractor = GeometricFeatureExtractor()
        invalid_landmarks = np.random.randn(100, 3).astype(np.float32)
        
        with pytest.raises(ValueError):
            extractor.extract(invalid_landmarks)
    
    def test_features_are_finite(self, sample_landmarks_478):
        """Test that all extracted features are finite"""
        from gfram.geometry.features import GeometricFeatureExtractor
        
        extractor = GeometricFeatureExtractor()
        features = extractor.extract(sample_landmarks_478)
        
        assert np.all(np.isfinite(features))
    
    def test_features_reproducible(self, sample_landmarks_478):
        """Test that feature extraction is deterministic"""
        from gfram.geometry.features import GeometricFeatureExtractor
        
        extractor = GeometricFeatureExtractor()
        
        features1 = extractor.extract(sample_landmarks_478)
        features2 = extractor.extract(sample_landmarks_478)
        
        np.testing.assert_array_equal(features1, features2)


class TestCreateFeatureExtractor:
    """Tests for factory function"""
    
    def test_create_478_mode(self):
        """Test creating extractor in 478 mode"""
        from gfram.geometry.features import create_feature_extractor
        
        extractor = create_feature_extractor(mode='478')
        assert extractor.num_landmarks == 478
        assert extractor.extract_iris == True
    
    def test_create_468_mode(self):
        """Test creating extractor in 468 mode"""
        from gfram.geometry.features import create_feature_extractor
        
        extractor = create_feature_extractor(mode='468')
        assert extractor.num_landmarks == 468
        assert extractor.extract_iris == False


class TestLandmarkNormalizer:
    """Tests for landmark normalization"""
    
    def test_normalizer_import(self):
        """Test LandmarkNormalizer can be imported"""
        from gfram.geometry.landmarks import LandmarkNormalizer
        normalizer = LandmarkNormalizer()
        assert normalizer is not None
    
    def test_normalize_landmarks(self, sample_landmarks_478):
        """Test landmark normalization"""
        from gfram.geometry.landmarks import LandmarkNormalizer
        
        normalizer = LandmarkNormalizer()
        normalized = normalizer.normalize(sample_landmarks_478)
        
        assert normalized.shape == sample_landmarks_478.shape
        assert isinstance(normalized, np.ndarray)


class TestSymmetryFeatures:
    """Tests for symmetry feature extraction"""
    
    def test_symmetry_import(self):
        """Test symmetry module can be imported"""
        from gfram.geometry.symmetry import compute_symmetry_features
        assert compute_symmetry_features is not None
    
    def test_compute_symmetry(self, sample_landmarks_478):
        """Test symmetry computation"""
        from gfram.geometry.symmetry import compute_symmetry_features
        
        features = compute_symmetry_features(sample_landmarks_478)
        assert isinstance(features, (list, np.ndarray))


class TestMoments:
    """Tests for Hu moments"""
    
    def test_moments_import(self):
        """Test moments module can be imported"""
        from gfram.geometry.moments import compute_hu_moments
        assert compute_hu_moments is not None
    
    def test_compute_moments(self, sample_landmarks_478):
        """Test Hu moments computation"""
        from gfram.geometry.moments import compute_hu_moments
        
        moments = compute_hu_moments(sample_landmarks_478[:, :2])  # 2D
        assert isinstance(moments, (list, np.ndarray))
        assert len(moments) == 7  # 7 Hu moments
