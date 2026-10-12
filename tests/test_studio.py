"""GFRAM Studio helpers that need no GUI, camera or models."""

import pytest


def test_version_comparison():
    from gfram.studio.updates import is_newer, version_tuple
    assert version_tuple('3.4.0') == (3, 4, 0)
    assert is_newer('3.10.0', '3.9.9')
    assert not is_newer('3.4.0', '3.4.0')
    assert not is_newer(None, '3.4.0')
    assert is_newer('3.4.1rc1', '3.4.0')


def test_packaged_results_are_readable():
    from gfram.studio import data
    assert data.run('baselines')['current_pipeline']['accuracy'] == pytest.approx(0.5366, abs=1e-3)
    assert data.run('hybrid_lfw')['w600k_mbf']['accuracy'] > 0.99
    assert len(data.training_history()) == 30
    assert data.run('does_not_exist') == {}


def test_studio_version_is_library_version():
    import gfram
    from gfram.studio import __version__
    assert __version__ == gfram.__version__
