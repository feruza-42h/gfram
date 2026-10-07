#!/usr/bin/env python3
"""
GFRAM Face Recognition Example
==============================

Command-line face recognition workflow:
1. Add persons to a database
2. Recognize unknown faces
3. Verify whether two photos show the same person

Usage:
    python face_recognition.py --add NAME IMAGE [IMAGE ...]
    python face_recognition.py --recognize IMAGE
    python face_recognition.py --verify IMAGE1 IMAGE2
    python face_recognition.py --list | --remove NAME | --clear

Options:
    --db DIR          database directory (default: ./gfram_example_db)
    --threshold T     match probability needed to accept a match (default: calibrated, ~0.53)
    --contribute      share enrolled faces with the GFRAM server for this run
                      (otherwise nothing is sent from this example)

Requirements:
    pip install gfram
"""

import argparse
import sys

import gfram
from gfram.api.simple_recognizer import SimpleRecognizer


def add_person(recognizer, name: str, image_path: str):
    """Add a photo of a person to the database."""
    result = recognizer.add(name, image_path)
    if result['success']:
        print(f"✅ {image_path}: added to '{name}' ({result['faces']} photo(s) enrolled)")
    else:
        print(f"❌ {image_path}: {result['error']}")
    return result


def recognize_face(recognizer, image_path: str):
    """Recognize a face in an image."""
    print(f"\n🔍 Recognizing face in: {image_path}")
    result = recognizer.recognize(image_path)

    if result.get('error'):
        print(f"❓ {result['error']}")
    elif result['recognized']:
        print(f"✅ Recognized: {result['name']}")
        print(f"   Match probability: {result['confidence']:.3f} (threshold {result['threshold']:.3f})")
    else:
        print(f"❓ Unknown person")
        print(f"   Closest: {result['best_candidate']}, match probability {result['confidence']:.3f} "
              f"(threshold {result['threshold']:.3f})")
    return result


def verify_faces(recognizer, image1_path: str, image2_path: str):
    """Verify if two images show the same person."""
    print(f"\n🔐 Verifying: {image1_path} vs {image2_path}")
    result = recognizer.verify(image1_path, image2_path)

    if result.get('error'):
        print(f"❌ {result['error']}")
    else:
        verdict = "✅ SAME PERSON" if result['same_person'] else "❌ DIFFERENT PERSONS"
        print(f"   Match probability: {result['confidence']:.3f} (threshold {result['threshold']:.3f})")
        print(f"   {verdict}")
    return result


def list_all(recognizer):
    """List all persons in database."""
    stats = recognizer.stats()
    print(f"\n📋 Database {stats['db_path']}")
    print(f"   {stats['persons']} person(s), {stats['faces']} photo(s)")
    for i, name in enumerate(recognizer.list_persons(), 1):
        print(f"   {i}. {name}")


def main():
    parser = argparse.ArgumentParser(
        description='GFRAM Face Recognition Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python face_recognition.py --add John john1.jpg john2.jpg
    python face_recognition.py --recognize test.jpg
    python face_recognition.py --verify photo1.jpg photo2.jpg
    python face_recognition.py --list
    python face_recognition.py --remove John
    python face_recognition.py --clear
        """
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--add', nargs='+', metavar='ARG', help='Add person: --add NAME IMAGE [IMAGE ...]')
    action.add_argument('--recognize', metavar='IMAGE', help='Recognize face in image')
    action.add_argument('--verify', nargs=2, metavar=('IMAGE1', 'IMAGE2'), help='Verify two images')
    action.add_argument('--list', action='store_true', help='List all persons in database')
    action.add_argument('--remove', metavar='NAME', help='Remove a person')
    action.add_argument('--clear', action='store_true', help='Delete every person in the database')
    parser.add_argument('--db', default='gfram_example_db', help='Database directory')
    parser.add_argument('--threshold', type=float, default=None, help='Match threshold')
    parser.add_argument('--contribute', action='store_true', help='Send enrolled faces to the server')
    args = parser.parse_args()

    if not any([args.add, args.recognize, args.verify, args.list, args.remove, args.clear]):
        parser.print_help()
        return

    print("=" * 50)
    print(f"🎯 GFRAM Face Recognition (v{gfram.__version__})")
    print("=" * 50)

    recognizer = SimpleRecognizer(db_path=args.db, threshold=args.threshold, contribute=args.contribute)

    if args.add:
        if len(args.add) < 2:
            print("❌ Usage: --add NAME IMAGE [IMAGE ...]")
            sys.exit(1)
        for image in args.add[1:]:
            add_person(recognizer, args.add[0], image)
    elif args.recognize:
        recognize_face(recognizer, args.recognize)
    elif args.verify:
        verify_faces(recognizer, *args.verify)
    elif args.list:
        list_all(recognizer)
    elif args.remove:
        print(f"✅ Removed {args.remove}" if recognizer.remove(args.remove) else f"❓ {args.remove} not found")
    elif args.clear:
        recognizer.clear()
        print(f"\n✅ Database {args.db} cleared")


if __name__ == '__main__':
    main()
