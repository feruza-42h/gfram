# GFRAM - Geometric Face Recognition and Matching

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/gfram.svg)](https://badge.fury.io/py/gfram)

**Hybrid face recognition built on facial geometry and AI.** GFRAM's geometric module reads 478 3D facial landmarks, aligns the face in 3D, estimates head pose and computes a learned geometric identity embedding (GeometricTransformer). A deep appearance model then works on the face crop that the geometric module aligned, and a quality-aware adaptive fusion combines both into the probability that two faces belong to the same person.

> **Status:** research library (v3.2), 99.5% on LFW. Do not use it as the sole basis for security or legal decisions.

## 🌟 Key Features

- **🔷 Geometric module**: 478 MediaPipe 3D landmarks, 3D Procrustes alignment, head pose, GeometricTransformer embedding
- **🤖 AI on top of geometry**: deep appearance model (MobileFaceNet) on geometry-aligned crops, no separate face detector
- **⚖️ Adaptive fusion**: a calibrated model weighs appearance, geometry, image quality and pose for each comparison and outputs a match probability
- **⚡ CPU-friendly**: ~25 ms per face on a laptop CPU; 1.9 MB geometric model + 13 MB appearance model
- **🔄 Update-safe database**: geometry is re-computed automatically when a new model arrives from gfram.uz
- **🌐 Cross-platform**: Windows, macOS, Linux

## 📊 Results

**LFW** (standard 6,000-pair verification benchmark):

| Method | LFW accuracy |
|--------|-------------:|
| Hand-crafted geometric features (GFRAM 3.0) | 53.7% |
| 3D Procrustes-aligned landmarks, no learning | 67.5% |
| Geometry + AI: GeometricTransformer trained on CelebA | 78.0% |
| **Hybrid: geometric module + appearance model + adaptive fusion** | **99.5–99.7%** |

- 99.67%: research protocol (10-fold, fusion and threshold learned on 9 folds, tested on the 10th).
- 99.53%: the shipped library via `gfram.verify()` on all 6,000 pairs with one fixed threshold, counting the 9 pairs with no detected face as errors.

**CPLFW** (cross-pose: frontal vs. profile faces, 6,000 pairs), with the ArcFace-R50 appearance model:

| Method | Accuracy | TAR @ FAR=1% |
|--------|---------:|-------------:|
| Appearance model only (stock InsightFace pipeline) | 95.75% | 92.8% |
| + quality-aware fusion | 96.10% | 92.3% |
| **+ quality + geometric pose and identity cues** | **96.28%** | **93.4%** |

With landmark-aligned crops (the lightweight pipeline used by the library), adding the geometric cues to the fusion is statistically significant on CPLFW (McNemar p < 0.002; TAR@FAR=1% +6 to +11 points). Against the strongest stock pipeline the geometric gain in accuracy is not significant (p = 0.18), while the TAR@FAR=1% gain is (+1.1 points, 95% CI [+0.1, +2.0]).

Cross-dataset check: a fusion trained only on LFW improves CPLFW from 91.8% (appearance only) to 93.9%.

Notes: the geometric embedding is trained on CelebA; LFW and CPLFW are never used to train it. The shipped fusion (11 parameters) is fitted on LFW and CPLFW pairs, so the library numbers above are not fully held-out; the cross-validated and cross-dataset numbers are. CelebA identities are anonymised, so overlap with LFW celebrities cannot be ruled out.

## 📦 Installation

```bash
pip install gfram
```

From source:
```bash
git clone https://github.com/feruza-42h/gfram.git
cd gfram
pip install -e .
```

On first use GFRAM downloads the appearance model (MobileFaceNet `w600k_mbf`, 13 MB, from the official InsightFace release, verified by SHA-256) into `~/.gfram/cache`.

Requirements: Python 3.10–3.14, PyTorch 2.0+, MediaPipe 0.10+, ONNX Runtime, NumPy, SciPy, OpenCV (installed automatically).

## 🚀 Quick Start

```python
import gfram

# Enrol people (several photos per person make recognition more reliable)
gfram.add("john", "john1.jpg")
gfram.add("john", "john2.jpg")
gfram.add("jane", "jane1.jpg")

# Recognise
result = gfram.recognize("unknown.jpg")
print(result)
# {'name': 'john', 'confidence': 0.998, 'recognized': True, 'person_id': 0,
#  'best_candidate': 'john', 'threshold': 0.526, 'bbox': (x, y, w, h)}

gfram.verify("a.jpg", "b.jpg")  # {'same_person': True, 'confidence': 0.999, 'threshold': 0.526}
gfram.list_persons()   # ['john', 'jane']
gfram.remove("jane")   # True
gfram.stats()          # {'persons': 1, 'faces': 2, 'model_version': '3.2.0', 'mode': 'hybrid', ...}
gfram.clear()          # delete the local database
gfram.server_status()  # {'status': 'ok', ...}

gfram.set_contribution_consent(True)   # opt in to sharing enrolled face data (see Privacy)
```

`confidence` is the probability that the query face and the closest enrolled face belong to the same person (enrolled faces are stored as landmarks and embeddings, never as photos); a face is `recognized` when it reaches `threshold`. Unrecognised faces return `'name': 'Unknown'` while `best_candidate` still shows the closest person.

### Choosing a threshold

```python
from gfram.api.simple_recognizer import SimpleRecognizer

recognizer = SimpleRecognizer(
    db_path="my_db",      # default: ~/.gfram/database
    threshold=0.9,        # default: calibrated threshold stored in the model (0.526)
    contribute=False,     # never share enrolled faces, whatever the stored consent
)
```

| Threshold | Behaviour |
|----------:|-----------|
| 0.526 (default) | best accuracy over clean, degraded and cross-pose training pairs; below 1% false accepts on LFW |
| 0.90 | ~0.1% false accepts on LFW; strict |

## 🏗️ How It Works

```
Image
  ↓  MediaPipe FaceMesh
478 3D landmarks
  ├─ GEOMETRIC MODULE ──────────────────────────────────────────────┐
  │   3D Procrustes alignment → GeometricTransformer → 128-d         │
  │   geometric embedding; head pose (yaw/pitch/roll)                │
  │   5 anchor points → face crop aligned to the ArcFace template    │
  │                                                                  │
  ├─ APPEARANCE MODEL (AI) ─────────────────────────────────────────┤
  │   MobileFaceNet on the aligned crop → 512-d embedding + its      │
  │   norm (image-quality cue); crop sharpness                       │
  │                                                                  │
  └─ ADAPTIVE FUSION ───────────────────────────────────────────────┘
      appearance similarity, geometric similarity, quality cues,
      pose difference → calibrated logistic model → match probability
```

**Geometric model** (`scripts/celeba/train.py`): GeometricTransformer (3 layers, 4 heads, d=128, 423K parameters), trained on 198,803 CelebA faces of 9,325 people with ArcFace loss on Procrustes-aligned, standardised landmarks.

**Fusion** (`scripts/hybrid/train_fusion.py`): logistic model over 10 pair features, fitted on ~51,000 LFW pairs under eight capture conditions (clean, low resolution, blur, darkness, JPEG artefacts, mask, sunglasses) and CPLFW cross-pose pairs.

**Model package** (`gfram-model/3`): one `.pth` file with the geometric model, its preprocessing, the fusion parameters and calibrated thresholds. Models are loaded with strict checks, so an incompatible file fails loudly instead of silently running with random weights.

**Model updates**: on start-up GFRAM uses the newest valid model among the local cache, the gfram.uz server (checked at most once a day) and the copy bundled with the package. Downloads are validated before they replace the cache.

## ⚖️ Licences

- GFRAM code and the geometric model: MIT.
- The appearance model (InsightFace `w600k_mbf`, trained on WebFace600K) is distributed by InsightFace under its own terms; GFRAM downloads it from the official InsightFace release.

## 🔒 Privacy

- Photos never leave the device. Appearance embeddings stay in the local database.
- **Sharing with the server is opt-in.** Until you call `gfram.set_contribution_consent(True)`, nothing is sent and GFRAM prints a one-time notice. With consent, each `gfram.add()` sends the **person's name, 478 landmarks, geometric features, geometric and appearance embeddings, photo quality, head pose, model versions and a random installation id** (never the photo) to gfram.uz to improve the shared model. Make sure the people you enrol agree.
- `gfram.set_contribution_consent(False)` stops sharing; `gfram.contribution_consent()` shows the current decision; `SimpleRecognizer(contribute=True/False)` overrides it for one recognizer. The server rejects contributions sent without consent.
- Landmarks and embeddings are biometric data; treat them as sensitive.

## 🔬 Reproducing the Results

```bash
# Geometry: landmarks, baselines, GeometricTransformer
python scripts/lfw/extract_landmarks.py
python scripts/lfw/baseline.py
python scripts/celeba/extract_landmarks.py
python scripts/celeba/train.py --config tiny --epochs 30 --steps-per-epoch 2000

# Hybrid: appearance benchmark, robustness, cross-pose, significance, fusion
python scripts/hybrid/lfw_benchmark.py
python scripts/hybrid/robustness.py
python scripts/hybrid/ablation.py
python scripts/hybrid/cplfw_benchmark.py
python scripts/hybrid/cplfw_insightface.py && python scripts/hybrid/significance.py --crop insightface
python scripts/hybrid/train_fusion.py

# Package
python scripts/export_model.py runs/celeba_tiny/best.pt dist/gfram_model_v3.2.0.pth \
    --version 3.2.0 --fusion data/models/fusion_mbf.json
```

The appearance experiments expect `w600k_mbf.onnx` / `w600k_r50.onnx` in `data/models/` (from InsightFace `buffalo_s` / `buffalo_l`). All scripts are resumable and sized for an 8 GB laptop.

## 🔧 Configuration

```bash
export GFRAM_CACHE_DIR=/path/to/cache      # models (default: ~/.gfram/cache)
export GFRAM_SERVER_URL=https://gfram.uz   # model server
```

## ⚠️ Limitations

- Geometry alone reaches 78% on LFW; the 99.5% comes from the hybrid. On frontal photos the geometric identity cue adds little to the appearance model; its measurable benefit is with large pose differences.
- MediaPipe needs a reasonably visible face: it finds landmarks on ~83% of CPLFW images, failing mostly on extreme profiles. Without landmarks GFRAM reports "No face detected".
- Accuracy drops with heavy blur, strong compression and very low resolution (e.g. 93–96% on LFW with strong blur/JPEG artefacts).

## 📁 Project Structure

```
gfram/
├── __init__.py              # Simple API: add, recognize, verify, list_persons, remove, clear, stats
├── api/simple_recognizer.py # SimpleRecognizer: database, hybrid matching, thresholds
├── cloud/                   # gfram.uz client and model loader (cache → server → bundled)
├── detectors/               # MediaPipe FaceMesh wrapper (478 landmarks)
├── geometry/                # Procrustes alignment, hand-crafted geometric features
├── models/
│   ├── geometric_transformer.py
│   ├── package.py           # model package loading, FaceEmbedder (geometric module)
│   ├── hybrid.py            # crop alignment, head pose, appearance model, adaptive fusion
│   └── losses.py            # ArcFace, CosFace, Triplet, Contrastive, Center
├── matching/, evaluation/, training/, utils/
└── pretrained/gfram_model.pth   # bundled model package (v3.2.0)
scripts/                     # data preparation, training, evaluation, export
tests/
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
pip install -e ".[dev]"
pytest
```

## 🔬 Citation

A paper describing GFRAM is in preparation. Until then, please cite the repository:

```bibtex
@software{gfram,
  title  = {GFRAM: Geometric Face Recognition and Matching},
  author = {Ortiqova, Feruza S.},
  year   = {2026},
  url    = {https://github.com/feruza-42h/gfram},
  note   = {Tashkent University of Information Technologies}
}
```

If you use the hybrid mode, also cite InsightFace / ArcFace (Deng et al., CVPR 2019) and WebFace260M (Zhu et al., CVPR 2021).

## 📄 License

MIT License - see [LICENSE](LICENSE). See [Licences](#-licences) for the appearance model.

## 🙏 Acknowledgments

- **MediaPipe** for facial landmark detection
- **InsightFace** for the pretrained appearance models
- **PyTorch** and **ONNX Runtime**
- **LFW**, **CPLFW** and **CelebA** datasets for evaluation and training

## 📞 Contact

- **Author**: Ortiqova Feruza Sardor qizi
- **Email**: feruzaortiqova42@gmail.com
- **GitHub**: [@feruza-42h](https://github.com/feruza-42h)
- **Issues**: [GitHub Issues](https://github.com/feruza-42h/gfram/issues)
