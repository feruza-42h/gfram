#!/usr/bin/env python3
"""
GFRAM Basic Usage Example
=========================

Enrol one person from a photo, then check other photos against them.
Runs in a temporary database, so your real GFRAM database is never touched.

Usage:
    python basic_usage.py PERSON_PHOTO [OTHER_PHOTO ...]

Example:
    python basic_usage.py john1.jpg john2.jpg stranger.jpg

Requirements:
    pip install gfram
"""

import sys
import tempfile
from pathlib import Path

import gfram
from gfram.api.simple_recognizer import SimpleRecognizer


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    enrol, queries = sys.argv[1], sys.argv[2:]

    print("=" * 50)
    print("🎯 GFRAM Basic Usage Example")
    print(f"📦 GFRAM Version: {gfram.__version__}")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as db:
        # contribute=False keeps the example fully local
        recognizer = SimpleRecognizer(db_path=db, contribute=False)

        name = Path(enrol).stem
        result = recognizer.add(name, enrol)
        if not result['success']:
            print(f"❌ {enrol}: {result['error']}")
            sys.exit(1)
        print(f"\n✅ Enrolled '{name}' from {enrol}")
        print(f"   Model {recognizer.stats()['model_version']} ({recognizer.stats()['mode']}), "
              f"threshold {recognizer.threshold:.3f}")

        for query in queries:
            r = recognizer.recognize(query)
            if r.get('error'):
                print(f"\n❓ {query}: {r['error']}")
            elif r['recognized']:
                print(f"\n✅ {query}: {r['name']} (match probability {r['confidence']:.3f})")
            else:
                print(f"\n❌ {query}: not '{r['best_candidate']}' (match probability {r['confidence']:.3f})")

    print("\n" + "=" * 50)
    print("With the default database the same steps are:")
    print("  gfram.add('John', 'john.jpg')")
    print("  gfram.recognize('test.jpg')")
    print("=" * 50)


if __name__ == '__main__':
    main()
