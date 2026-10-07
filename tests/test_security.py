"""
Protected templates: what is stored, exact score preservation, migration,
revocation by key rotation, wrong-key handling and rotation crash recovery.
"""

import pickle
import secrets

import numpy as np
import pytest

from tests.test_model_package import BUNDLED, FakeAppearance, IMG, face, recognizer  # noqa: F401  (fixtures)


def db_contents(tmp_path):
    return pickle.load(open(tmp_path / 'persons.pkl', 'rb'))


def test_database_stores_no_plain_landmarks_or_embeddings(recognizer, tmp_path, face):
    r = recognizer()
    r.add('alice', IMG)
    data = db_contents(tmp_path)

    assert data['format'] == 4 and 'sample_landmarks' not in data
    blob = data['sample_landmarks_enc'][0]
    assert isinstance(blob, bytes) and face.astype(np.float32).tobytes() not in blob

    raw_geo = r.embedder.embed(face)
    assert not np.allclose(data['geo'][0], raw_geo, atol=1e-3)          # stored template is rotated
    assert abs(float(data['geo'][0] @ data['geo'][0]) - float(raw_geo @ raw_geo)) < 1e-4  # norm kept


def test_fusion_probability_is_exactly_preserved():
    from gfram.models.hybrid import AdaptiveFusion
    from gfram.models.package import read_package
    from gfram.security import TemplateProtector

    if not BUNDLED.exists():
        pytest.skip('bundled model not present')
    fusion = AdaptiveFusion(read_package(BUNDLED)['fusion'])
    rng = np.random.default_rng(1)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    gallery = {'geo': unit(rng.normal(size=(50, 128))), 'app': unit(rng.normal(size=(50, 512))),
               'app_norm': rng.uniform(15, 30, 50), 'sharp': rng.uniform(3, 8, 50), 'pose': rng.normal(0, 15, (50, 3))}
    query = {'geo': gallery['geo'][3] * 0.9 + 0.1 * unit(rng.normal(size=128)),
             'app': gallery['app'][3] * 0.9 + 0.1 * unit(rng.normal(size=512)),
             'app_norm': 22.0, 'sharp': 5.0, 'pose': np.array([5.0, 1.0, 0.0])}

    p = TemplateProtector(secrets.token_bytes(32))
    protect = lambda d: dict(d, geo=p.protect(d['geo'], 'geo'), app=p.protect(d['app'], 'app'))
    plain, protected = fusion.probability(query, gallery), fusion.probability(protect(query), protect(gallery))
    assert np.abs(plain - protected).max() < 1e-5
    assert int(np.argmax(plain)) == int(np.argmax(protected)) == 3


def test_unprotected_32_database_is_migrated_without_loss(recognizer, tmp_path, face):
    r = recognizer()
    geo = r.embedder.embed(face)
    pickle.dump({'format': 3, 'persons': {'alice': 0}, 'next_id': 1, 'sample_names': ['alice'],
                 'sample_landmarks': [face.astype(np.float32)], 'sample_app': [FakeAppearance.vector],
                 'sample_quality': [{'app_norm': 25.0, 'sharp': 5.0}], 'appearance_model': 'w600k_mbf',
                 'geo': geo[None], 'pose': np.zeros((1, 3), np.float32), 'geo_version': r.embedder.version},
                open(tmp_path / 'persons.pkl', 'wb'))

    r2 = recognizer()
    assert r2.list_persons() == ['alice']
    assert db_contents(tmp_path)['format'] == 4
    result = r2.recognize(IMG)
    assert result['recognized'] and result['name'] == 'alice'


def test_rotation_revokes_old_templates(recognizer, tmp_path):
    r = recognizer()
    r.add('alice', IMG)
    stolen = (tmp_path / 'persons.pkl').read_bytes()
    old_fp = r.protector.fingerprint

    new_fp = r.rotate_template_key()
    assert new_fp != old_fp
    assert r.recognize(IMG)['name'] == 'alice'                       # still works after rotation

    (tmp_path / 'persons.pkl').write_bytes(stolen)                   # an attacker's old copy
    with pytest.raises(RuntimeError, match='different template key'):
        recognizer()


def test_wrong_key_gives_clear_error(recognizer, tmp_path):
    from gfram.security import default_key_path, write_key
    recognizer().add('alice', IMG)
    write_key(secrets.token_bytes(32), default_key_path())
    with pytest.raises(RuntimeError, match='different template key'):
        recognizer()


def test_interrupted_rotation_recovers_the_matching_key(recognizer, tmp_path):
    from gfram.security import default_key_path, load_or_create_key, write_key
    r = recognizer()
    r.add('alice', IMG)
    path = default_key_path()
    old_key = load_or_create_key(path)

    r.rotate_template_key()
    new_key = load_or_create_key(path)
    # Simulate a crash after the database was saved but before the main key file was replaced
    write_key(old_key, path)
    write_key(new_key, path.with_suffix('.key.new'))

    r2 = recognizer()
    assert r2.recognize(IMG)['name'] == 'alice'
    assert load_or_create_key(path) == new_key                       # main key file repaired


def test_rotation_refused_for_caller_managed_key(tmp_path, monkeypatch, face):
    import gfram.models.hybrid as hybrid
    from gfram.api.simple_recognizer import SimpleRecognizer
    from gfram.cloud import model_loader
    if not BUNDLED.exists():
        pytest.skip('bundled model not present')
    monkeypatch.setattr(model_loader, 'ensure_model_available', lambda *a, **k: BUNDLED)
    monkeypatch.setattr(hybrid, 'AppearanceEmbedder', FakeAppearance)
    r = SimpleRecognizer(db_path=str(tmp_path), contribute=False, template_key=secrets.token_bytes(32))
    with pytest.raises(RuntimeError, match='key you manage'):
        r.rotate_template_key()
