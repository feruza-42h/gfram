"""
Tests for gfram.matching module
"""

import pytest
import numpy as np


class TestFaceIndex:
    """Tests for FaceIndex class"""
    
    def test_index_import(self):
        """Test FaceIndex can be imported"""
        from gfram.matching import FaceIndex
        assert FaceIndex is not None
    
    def test_index_creation(self):
        """Test FaceIndex can be created"""
        from gfram.matching import FaceIndex
        
        index = FaceIndex(dimension=256)
        assert index is not None
    
    def test_index_with_metric(self):
        """Test FaceIndex with different metrics"""
        from gfram.matching import FaceIndex
        
        for metric in ['cosine', 'euclidean']:
            index = FaceIndex(dimension=256, metric=metric)
            assert index is not None
    
    def test_add_single_vector(self):
        """Test adding single vector to index"""
        from gfram.matching import FaceIndex
        
        index = FaceIndex(dimension=256)
        vector = np.random.randn(256).astype(np.float32)
        metadata = {'name': 'test', 'person_id': 0}
        
        index.add(metadata['name'], vector, metadata=metadata)
        
        # Should not raise
        assert True
    
    def test_add_multiple_vectors(self):
        """Test adding multiple vectors"""
        from gfram.matching import FaceIndex
        
        index = FaceIndex(dimension=256)
        
        for i in range(10):
            vector = np.random.randn(256).astype(np.float32)
            metadata = {'name': f'person_{i}', 'person_id': i}
            index.add(metadata['name'], vector, metadata=metadata)
        
        assert True
    
    def test_search(self):
        """Test searching in index"""
        from gfram.matching import FaceIndex
        
        index = FaceIndex(dimension=256)
        
        # Add vectors
        for i in range(5):
            vector = np.random.randn(256).astype(np.float32)
            metadata = {'name': f'person_{i}', 'person_id': i}
            index.add(metadata['name'], vector, metadata=metadata)
        
        # Search
        query = np.random.randn(256).astype(np.float32)
        results = index.search(query, k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
    
    def test_search_returns_metadata(self):
        """Test search returns metadata"""
        from gfram.matching import FaceIndex
        
        index = FaceIndex(dimension=256)
        
        # Add vector with metadata
        vector = np.random.randn(256).astype(np.float32)
        metadata = {'name': 'test_person', 'person_id': 42}
        index.add(metadata['name'], vector, metadata=metadata)
        
        # Search with same vector should return it
        results = index.search(vector, k=1)
        
        assert len(results) == 1
        assert results[0]['name'] == 'test_person'
        assert results[0]['metadata']['person_id'] == 42
        assert results[0]['similarity'] > 0.999
    
    def test_search_empty_index(self):
        """Test searching in empty index"""
        from gfram.matching import FaceIndex
        
        index = FaceIndex(dimension=256)
        query = np.random.randn(256).astype(np.float32)
        
        results = index.search(query, k=5)
        
        assert isinstance(results, list)
        assert len(results) == 0


class TestMatchingExports:
    """Test matching module exports"""
    
    def test_matching_init_exports(self):
        """Test matching __init__ exports"""
        from gfram import matching
        
        assert hasattr(matching, 'FaceIndex')
