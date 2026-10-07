# GFRAM Examples

Example scripts for GFRAM 3.1. None of them touches your default database
(`~/.gfram/database`) or sends data to the server unless you ask for it.

## 1. Basic Usage (`basic_usage.py`)

Enrol one person from a photo and check other photos against them, in a
temporary database.

```bash
python basic_usage.py john1.jpg john2.jpg stranger.jpg
```

## 2. Face Recognition (`face_recognition.py`)

Command-line workflow with a persistent database (`./gfram_example_db` by default).

```bash
# Add a person (several photos improve accuracy)
python face_recognition.py --add John john1.jpg john2.jpg john3.jpg

# Recognize a face
python face_recognition.py --recognize unknown.jpg

# Verify two photos (does not use the database)
python face_recognition.py --verify photo1.jpg photo2.jpg

# Manage the database
python face_recognition.py --list
python face_recognition.py --remove John
python face_recognition.py --clear

# Options
python face_recognition.py --recognize unknown.jpg --threshold 0.9    # stricter matching
python face_recognition.py --add John john.jpg --db my_db --contribute
```

## 3. Geometric Features (`geometric_features.py`)

Shows the 153 hand-crafted geometric features and the 128-value learned
geometric embedding used by the hybrid recogniser, and demonstrates that the embedding ignores
head rotation and scale.

```bash
python geometric_features.py face.jpg
python geometric_features.py            # synthetic landmarks, no image needed
```

## Quick Start

```python
import gfram

gfram.add("Alice", "alice1.jpg")
gfram.add("Alice", "alice2.jpg")
gfram.add("Bob", "bob.jpg")

result = gfram.recognize("test.jpg")
print(f"{result['name']} (match probability {result['confidence']:.3f})")

print(gfram.verify("photo1.jpg", "photo2.jpg"))
```

## Notes

- The geometric model (1.9 MB) ships with the package; newer versions are downloaded from gfram.uz automatically.
- Runs on CPU (~15 ms per face); no GPU required.
- `confidence` is the probability that two faces belong to the same person; the default threshold (~0.53) is calibrated during training. Raise it (e.g. 0.9) for fewer false matches.
- Recognition is hybrid (geometry + appearance model): 99.5% on LFW. The appearance model (13 MB) is downloaded from InsightFace on first run.
