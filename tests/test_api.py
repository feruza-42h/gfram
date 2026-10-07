"""
Tests for gfram public API
"""

import pytest
import numpy as np


class TestPublicAPI:
    """Tests for public gfram API"""
    
    def test_import_gfram(self):
        """Test gfram can be imported"""
        import gfram
        assert gfram is not None
    
    def test_version(self):
        """Test version is available"""
        import gfram
        assert hasattr(gfram, '__version__')
        assert isinstance(gfram.__version__, str)
    
    def test_author(self):
        """Test author is available"""
        import gfram
        assert hasattr(gfram, '__author__')
    
    def test_api_functions_exist(self):
        """Test all API functions exist"""
        import gfram
        
        functions = ['add', 'recognize', 'list_persons', 'remove', 'clear', 'stats']
        
        for func in functions:
            assert hasattr(gfram, func), f"Missing function: {func}"
    
    def test_stats(self):
        """Test stats function works"""
        import gfram
        
        stats = gfram.stats()
        
        assert isinstance(stats, dict)
        assert 'persons' in stats
        assert 'device' in stats
    
    def test_list_persons_empty(self):
        """Test list_persons on empty database"""
        import gfram
        
        gfram.clear()  # Ensure empty
        persons = gfram.list_persons()
        
        assert isinstance(persons, list)
        assert len(persons) == 0
    
    def test_clear(self):
        """Test clear function"""
        import gfram
        
        gfram.clear()
        persons = gfram.list_persons()
        
        assert len(persons) == 0


class TestSimpleRecognizer:
    """Tests for SimpleRecognizer class"""
    
    def test_recognizer_import(self):
        """Test SimpleRecognizer can be imported"""
        from gfram.api.simple_recognizer import SimpleRecognizer
        assert SimpleRecognizer is not None
    
    def test_recognizer_creation(self):
        """Test SimpleRecognizer can be created"""
        from gfram.api.simple_recognizer import SimpleRecognizer
        
        recognizer = SimpleRecognizer()
        assert recognizer is not None
    
    def test_recognizer_stats(self):
        """Test recognizer stats"""
        from gfram.api.simple_recognizer import SimpleRecognizer
        
        recognizer = SimpleRecognizer()
        stats = recognizer.stats()
        
        assert isinstance(stats, dict)
        assert 'persons' in stats
        assert 'device' in stats
    
    def test_recognizer_list_persons(self):
        """Test listing persons"""
        from gfram.api.simple_recognizer import SimpleRecognizer
        
        recognizer = SimpleRecognizer()
        recognizer.clear()
        
        persons = recognizer.list_persons()
        
        assert isinstance(persons, list)
    
    def test_recognizer_clear(self):
        """Test clearing database"""
        from gfram.api.simple_recognizer import SimpleRecognizer
        
        recognizer = SimpleRecognizer()
        recognizer.clear()
        
        assert len(recognizer.list_persons()) == 0


class TestRecognizeWithoutFace:
    """Tests for recognition without real faces"""
    
    def test_recognize_no_face(self):
        """Test recognition on image without face"""
        import gfram
        
        # Random image with no face
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = gfram.recognize(img)
        
        assert isinstance(result, dict)
        assert 'recognized' in result
        assert result['recognized'] == False


class TestExports:
    """Test module exports"""
    
    def test_all_exports(self):
        """Test __all__ exports"""
        import gfram
        
        for name in gfram.__all__:
            assert hasattr(gfram, name), f"Missing export: {name}"
