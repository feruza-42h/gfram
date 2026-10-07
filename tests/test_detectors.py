"""
Tests for gfram.detectors module
"""

import pytest
import numpy as np


class TestFaceDetector:
    """Tests for FaceDetector class"""
    
    def test_detector_import(self):
        """Test FaceDetector can be imported"""
        from gfram.detectors import FaceDetector
        assert FaceDetector is not None
    
    def test_detector_creation(self):
        """Test FaceDetector can be created"""
        from gfram.detectors import FaceDetector
        
        detector = FaceDetector()
        assert detector is not None
    
    def test_detect_no_face(self, sample_image):
        """Test detection on image with no face"""
        from gfram.detectors import FaceDetector
        
        detector = FaceDetector()
        faces = detector.detect(sample_image)
        
        # Random image should have no faces
        assert isinstance(faces, list)
        # May or may not detect (depends on random pattern)
    
    def test_detect_returns_list(self, sample_image):
        """Test detect returns a list"""
        from gfram.detectors import FaceDetector
        
        detector = FaceDetector()
        result = detector.detect(sample_image)
        
        assert isinstance(result, list)
    
    def test_detector_with_different_image_sizes(self):
        """Test detector works with different image sizes"""
        from gfram.detectors import FaceDetector
        
        detector = FaceDetector()
        
        sizes = [(240, 320, 3), (480, 640, 3), (720, 1280, 3)]
        
        for size in sizes:
            img = np.random.randint(0, 255, size, dtype=np.uint8)
            result = detector.detect(img)
            assert isinstance(result, list)


class TestLandmarkNormalizer:
    """Tests for LandmarkNormalizer"""
    
    def test_normalizer_import(self):
        """Test LandmarkNormalizer can be imported"""
        from gfram.detectors import LandmarkNormalizer
        assert LandmarkNormalizer is not None
    
    def test_normalizer_creation(self):
        """Test LandmarkNormalizer can be created"""
        from gfram.detectors import LandmarkNormalizer
        
        normalizer = LandmarkNormalizer()
        assert normalizer is not None
    
    def test_normalize_landmarks(self, sample_landmarks_478):
        """Test landmark normalization"""
        from gfram.detectors import LandmarkNormalizer
        
        normalizer = LandmarkNormalizer()
        normalized = normalizer.normalize(sample_landmarks_478)
        
        assert normalized.shape == sample_landmarks_478.shape
        assert normalized.dtype == sample_landmarks_478.dtype
    
    def test_normalized_centered(self, sample_landmarks_478):
        """Test normalized landmarks are centered"""
        from gfram.detectors import LandmarkNormalizer
        
        normalizer = LandmarkNormalizer()
        normalized = normalizer.normalize(sample_landmarks_478)
        
        # Centroid should be close to origin
        centroid = np.mean(normalized, axis=0)
        assert np.allclose(centroid, 0, atol=0.1)
    
    def test_normalize_468(self, sample_landmarks_468):
        """Test normalization works with 468 landmarks"""
        from gfram.detectors import LandmarkNormalizer
        
        normalizer = LandmarkNormalizer()
        normalized = normalizer.normalize(sample_landmarks_468)
        
        assert normalized.shape == sample_landmarks_468.shape


class TestDetectorExports:
    """Test module exports"""
    
    def test_detectors_init_exports(self):
        """Test detectors __init__ exports"""
        from gfram import detectors
        
        assert hasattr(detectors, 'FaceDetector')
        assert hasattr(detectors, 'LandmarkNormalizer')
