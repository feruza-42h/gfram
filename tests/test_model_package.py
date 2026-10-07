"""
Tests for the gfram-model/2 package, Procrustes alignment, the model loader's
update logic and the landmark-backed recognizer database.
"""

import json
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import pytest
import torch

from gfram.cloud import model_loader
from gfram.geometry.alignment import procrustes_align
from gfram.models.package import FaceEmbedder, ModelPackageError, read_package

BUNDLED = model_loader.BUNDLED_MODEL


def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


@pytest.fixture
def face(rng=np.random.default_rng(0)):
    return rng.normal(size=(478, 3)).astype(np.float32)


# ---------------------------------------------------------------- alignment

def test_alignment_removes_pose_scale_and_translation(face):
    rng = np.random.default_rng(1)
    moved = face @ random_rotation(rng).T * 3.7 + np.array([10, -4, 2], np.float32)
    ref = procrustes_align(face, face)
    np.testing.assert_allclose(procrustes_align(moved, ref), ref, atol=1e-5)


def test_alignment_batch_matches_single(face):
    batch = np.stack([face, face * 2])
    out = procrustes_align(batch, face)
    np.testing.assert_allclose(out[0], procrustes_align(face, face), atol=1e-6)
    assert out.shape == batch.shape


# ---------------------------------------------------------------- package

@pytest.mark.skipif(not BUNDLED.exists(), reason='bundled model not present')
def test_bundled_model_loads_strictly():
    pkg = read_package(BUNDLED)
    embedder = FaceEmbedder(pkg)
    assert embedder.thresholds['best_accuracy'] > 0
    emb = embedder.embed(np.random.default_rng(0).normal(size=(3, 478, 3)).astype(np.float32))
    assert emb.shape == (3, embedder.dim)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


@pytest.mark.skipif(not BUNDLED.exists(), reason='bundled model not present')
def test_embedding_is_pose_invariant(face):
    embedder = FaceEmbedder.from_file(BUNDLED)
    moved = face @ random_rotation(np.random.default_rng(2)).T * 120 + 300
    assert float(embedder.embed(face) @ embedder.embed(moved)) > 0.999


def test_legacy_checkpoint_is_rejected(tmp_path):
    old = tmp_path / 'old.pth'
    torch.save({'model_state_dict': {}, 'config': {}}, old)
    with pytest.raises(ModelPackageError):
        read_package(old)


@pytest.mark.skipif(not BUNDLED.exists(), reason='bundled model not present')
def test_mismatched_weights_fail_loudly():
    pkg = read_package(BUNDLED)
    pkg['config_name'] = 'base'  # weights are 'tiny'
    with pytest.raises(RuntimeError):
        FaceEmbedder(pkg)


# ---------------------------------------------------------------- loader

class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def model_server(tmp_path, monkeypatch):
    """Local stand-in for gfram.uz serving /api/model and /api/model/download."""
    root = tmp_path / 'srv'
    (root / 'api' / 'model').mkdir(parents=True)
    server = HTTPServer(('127.0.0.1', 0), partial(_Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    cache = tmp_path / 'cache'
    monkeypatch.setenv('GFRAM_SERVER_URL', f'http://127.0.0.1:{server.server_port}')
    monkeypatch.setenv('GFRAM_CACHE_DIR', str(cache))
    import gfram.cloud.server_client as sc
    monkeypatch.setattr(sc, '_client', None)

    def publish(version, payload=None):
        # SimpleHTTPRequestHandler serves directories via index.html, so the
        # endpoints are files named index.html inside api/model and api/model/download
        (root / 'api' / 'model' / 'index.html').write_text(json.dumps({'version': version}))
        dl = root / 'api' / 'model' / 'download'
        dl.mkdir(exist_ok=True)
        if payload is None:
            pkg = read_package(BUNDLED)
            pkg['model_version'] = version
            torch.save(pkg, dl / 'index.html')
        else:
            (dl / 'index.html').write_bytes(payload)

    yield publish, cache
    server.shutdown()


@pytest.mark.skipif(not BUNDLED.exists(), reason='bundled model not present')
def test_newer_server_model_is_downloaded(model_server):
    publish, cache = model_server
    publish('9.0.0')
    path = model_loader.ensure_model_available()
    assert path == cache / 'gfram_model.pth'
    assert read_package(path)['model_version'] == '9.0.0'


@pytest.mark.skipif(not BUNDLED.exists(), reason='bundled model not present')
def test_corrupt_server_model_is_rejected(model_server):
    publish, cache = model_server
    publish('9.0.0', payload=b'not a model')
    assert model_loader.ensure_model_available() == BUNDLED
    assert not (cache / 'gfram_model.pth').exists()


@pytest.mark.skipif(not BUNDLED.exists(), reason='bundled model not present')
def test_older_server_model_is_ignored(model_server):
    publish, _ = model_server
    publish('1.0.0')
    assert model_loader.ensure_model_available() == BUNDLED


# ---------------------------------------------------------------- hybrid recognizer

class FakeAppearance:
    """Stands in for the downloaded MobileFaceNet: returns whatever vector the test sets."""
    vector = np.eye(512, dtype=np.float32)[0]

    def __init__(self, *args, **kwargs):
        pass

    def embed(self, crops):
        return np.repeat(FakeAppearance.vector[None], len(crops), 0), np.full(len(crops), 25.0)


@pytest.fixture
def recognizer(tmp_path, monkeypatch, face):
    if not BUNDLED.exists():
        pytest.skip('bundled model not present')
    import gfram.models.hybrid as hybrid
    from gfram.api.simple_recognizer import SimpleRecognizer
    monkeypatch.setattr(model_loader, 'ensure_model_available', lambda *a, **k: BUNDLED)
    monkeypatch.setattr(hybrid, 'AppearanceEmbedder', FakeAppearance)
    FakeAppearance.vector = np.eye(512, dtype=np.float32)[0]

    def make():
        r = SimpleRecognizer(db_path=str(tmp_path), contribute=False)
        r._detect = lambda img: {'landmarks': face, 'bbox': (0, 0, 1, 1)}
        return r
    return make


IMG = np.zeros((4, 4, 3), np.uint8)


def test_bundled_package_is_hybrid():
    if not BUNDLED.exists():
        pytest.skip('bundled model not present')
    pkg = read_package(BUNDLED)
    assert pkg['format'] == 'gfram-model/3'
    assert pkg['fusion']['appearance_model'] == 'w600k_mbf'
    assert 0 < pkg['fusion']['thresholds']['best_accuracy'] < 1


def test_hybrid_recognition_and_geometry_refresh(recognizer):
    r = recognizer()
    assert r.add('alice', IMG)['success']
    assert r.stats()['mode'] == 'hybrid'

    # Simulate a database whose geometry came from an older geometry model
    r._geo_version, r._geo = '0.0.1', np.zeros_like(r._geo)
    r._save_database()

    r2 = recognizer()
    result = r2.recognize(IMG)
    assert result['recognized'] and result['name'] == 'alice'
    assert result['confidence'] > r2.threshold
    assert r2._geo_version == r2.embedder.version


def test_stranger_is_rejected(recognizer):
    r = recognizer()
    r.add('alice', IMG)
    FakeAppearance.vector = np.eye(512, dtype=np.float32)[1]  # orthogonal appearance
    result = r.recognize(IMG)
    assert not result['recognized'] and result['name'] == 'Unknown'
    assert result['best_candidate'] == 'alice' and result['confidence'] < r.threshold


def test_verify_same_and_missing_face(recognizer):
    r = recognizer()
    same = r.verify(IMG, IMG)
    assert same['same_person'] and same['confidence'] > r.threshold

    r._detect = lambda image: None
    assert 'error' in r.verify(IMG, IMG)
    assert r.list_persons() == []  # verify never touches the database


def test_old_database_is_archived(tmp_path, recognizer):
    import pickle
    pickle.dump({'format': 2, 'persons': {'bob': 0}}, open(tmp_path / 'persons.pkl', 'wb'))
    r = recognizer()
    assert r.list_persons() == []
    assert (tmp_path / 'legacy_v3.1' / 'persons.pkl').exists()
