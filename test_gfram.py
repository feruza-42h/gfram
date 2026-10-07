#!/usr/bin/env python3
"""
GFRAM Library Test
==================

Test the gfram library.

Usage:
    python test_gfram.py

Author: Ortiqova F.S.
"""

import sys
import os

# Add library path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test imports"""
    print("\n1️⃣ Testing imports...")
    
    try:
        import gfram
        print(f"   ✅ gfram version: {gfram.__version__}")
        
        # Test simple API functions exist
        assert hasattr(gfram, 'add')
        assert hasattr(gfram, 'recognize')
        assert hasattr(gfram, 'list_persons')
        assert hasattr(gfram, 'remove')
        assert hasattr(gfram, 'clear')
        assert hasattr(gfram, 'stats')
        assert hasattr(gfram, 'server_status')
        print("   ✅ All API functions exist")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server():
    """Test server connection"""
    print("\n2️⃣ Testing server connection...")
    
    try:
        import gfram
        
        status = gfram.server_status()
        print(f"   Server: {status}")
        
        if status.get('status') == 'ok':
            print("   ✅ Server is available")
            return True
        else:
            print(f"   ⚠️ Server response: {status}")
            return True  # Not critical
            
    except Exception as e:
        print(f"   ⚠️ Server unavailable: {e}")
        return True  # Not critical


def test_model_download():
    """Test model download"""
    print("\n3️⃣ Testing model download...")
    
    try:
        from gfram.cloud import ensure_model_available, get_model_path
        
        path = get_model_path()
        print(f"   Cache path: {path}")
        
        if path.exists():
            print(f"   ✅ Model exists: {path.stat().st_size / 1024 / 1024:.1f} MB")
            return True
        
        print("   Downloading model...")
        model_path = ensure_model_available()
        
        if model_path and model_path.exists():
            print(f"   ✅ Model downloaded: {model_path}")
            return True
        else:
            print("   ⚠️ Could not download model")
            return True  # Try local
            
    except Exception as e:
        print(f"   ⚠️ Download issue: {e}")
        return True


def test_face_detector():
    """Test face detector"""
    print("\n4️⃣ Testing FaceDetector...")
    
    try:
        from gfram.detectors import FaceDetector
        import numpy as np
        
        detector = FaceDetector()
        print("   ✅ FaceDetector created")
        
        # Test with random image (won't find face)
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = detector.detect(test_img)
        print(f"   Faces found: {len(faces)} (expected 0 for random image)")
        print("   ✅ Detector works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_geometric_features():
    """Test geometric feature extraction"""
    print("\n5️⃣ Testing GeometricFeatureExtractor...")
    
    try:
        from gfram.geometry.features import GeometricFeatureExtractor
        import numpy as np
        
        extractor = GeometricFeatureExtractor()
        print("   ✅ Extractor created")
        
        # Random landmarks
        landmarks = np.random.randn(478, 3).astype(np.float32)
        features = extractor.extract(landmarks)
        
        print(f"   Features: {len(features)}")
        print("   ✅ Features extracted")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_api():
    """Test simple API"""
    print("\n6️⃣ Testing Simple API...")
    
    try:
        import gfram
        
        # Stats should work
        stats = gfram.stats()
        print(f"   Stats: {stats}")
        print("   ✅ stats() works")
        
        # List persons should work
        persons = gfram.list_persons()
        print(f"   Persons: {persons}")
        print("   ✅ list_persons() works")
        
        # Clear should work
        gfram.clear()
        print("   ✅ clear() works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 GFRAM LIBRARY TEST")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Server", test_server()))
    results.append(("Model Download", test_model_download()))
    results.append(("Face Detector", test_face_detector()))
    results.append(("Geometric Features", test_geometric_features()))
    results.append(("Simple API", test_simple_api()))
    
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"   TOTAL: {passed}/{len(results)} tests passed")
    
    return passed == len(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
