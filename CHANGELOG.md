# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.3.0] - 2026-10-07

### Added
- Protected, cancelable templates: embeddings stored only after secret keyed orthogonal
  rotations, landmarks encrypted with AES-256-GCM; recognition results are exactly unchanged.
- `gfram.rotate_template_key()` revokes all stored templates without photos; interrupted
  rotations recover automatically. `SimpleRecognizer(template_key=...)` for external key stores.
- `scripts/hybrid/template_protection.py`: score preservation, unlinkability and revocation experiment.

### Changed
- Database format 4. Databases from 3.2.0 are protected in place automatically; nobody is lost.
- New dependency: `cryptography`.
- `FaceIndex` uses numpy exact search; `faiss-cpu` is no longer a dependency (it aborted
  processes on macOS when loaded next to PyTorch).
- Linux needs `libegl1 libgles2 libgl1` for MediaPipe; `FaceDetector` now says so explicitly.

## [3.2.0] - 2026-10-07

### Added
- Hybrid recognition by default: geometric module (478 3D landmarks, GeometricTransformer,
  head pose, crop alignment) + MobileFaceNet appearance model + quality-aware adaptive fusion.
  `confidence` is now a calibrated match probability. LFW: 99.53% through `gfram.verify()`.
- `gfram.verify(image1, image2)`.
- Opt-in sharing with the server: `gfram.set_contribution_consent()`, `gfram.contribution_consent()`.
  Shared records include appearance embedding, quality, pose and a random installation id.
- Model package format `gfram-model/3` (geometry model, preprocessing, fusion, thresholds).

### Changed
- Supported Python versions: 3.10–3.14 (3.8 and 3.9 dropped).
- Database format 3 stores landmarks and appearance embeddings; older databases are moved to
  `legacy_v3.x/` and people must be enrolled again.
- Appearance model is downloaded once from the official InsightFace release and verified by SHA-256.

### Removed
- Unused extras `gfram[gpu]`, `gfram[geometric]`, `gfram[all]`.
- Modules from 2.x that the 3.x line no longer used (still available in git history).

### Fixed
- `gfram.models` failed to import (`MultiHeadGeometricAttention` did not exist).
- Test suite aligned with the real API and isolated from the user's `~/.gfram`.

## [3.1.0] - 2026-10-07

### Added
- First trained GeometricTransformer (CelebA, 198,803 faces): 78.0% on LFW, up from 53.7%
  with the 3.0 hand-crafted pipeline. Procrustes alignment and per-coordinate standardisation.
- Strict model loading, bundled model, validated server updates, certifi-based SSL.

### Fixed
- 3.0 shipped a checkpoint that matched none of the model's weights and was loaded with
  `strict=False`, so recognition ran on random weights.

## [2.0.0] - 2025-11-21

### Added
- Initial public release 🎉
- Face detection using MediaPipe with 468 landmarks
- Geometric feature extraction (150+ features):
  - Euclidean distances between landmarks
  - Differential geometry features
  - Topological features (persistent homology)
  - Statistical features
  - Symmetry analysis (15 features)
  - Graph-based features
  - Delaunay triangulation features (15 features)
  - Hu moments (7 invariants)
- AI-powered models:
  - Geometric Transformer architecture
  - Graph Neural Network (GNN) for landmark relationships
  - Multiple loss functions (Triplet, ArcFace, CosFace, Combined)
- Training pipeline:
  - Custom dataset classes for landmarks
  - Triplet learning support
  - Data augmentation for landmarks
  - Complete trainer with validation
- FAISS-based face indexing:
  - Multiple index types (Flat, HNSW, IVF)
  - Cosine and Euclidean metrics
  - Efficient similarity search
- Utility modules:
  - Configuration management (YAML support)
  - Image I/O operations
  - Visualization tools (landmark drawing, face mesh)
  - Landmark processing and normalization
- High-level API:
  - `Recognizer` class for easy face recognition
  - Pretrained model support
- Comprehensive documentation:
  - Full API documentation
  - Usage examples
  - Testing guide
  - PyPI publishing guide

### Features
- Python 3.8+ support
- CPU and GPU support (optional)
- Cross-platform compatibility (Windows, Linux, macOS)
- Type hints throughout the codebase
- Extensive docstrings
- Professional code structure

### Dependencies
- numpy >= 1.21.0
- scipy >= 1.7.0
- opencv-python >= 4.5.0
- mediapipe >= 0.10.0
- scikit-learn >= 1.0.0
- torch >= 2.0.0
- faiss-cpu >= 1.7.0
- tqdm >= 4.62.0
- pyyaml >= 6.0
- pillow >= 9.0.0

### Optional Dependencies
- faiss-gpu >= 1.7.0 (for GPU acceleration)
- torch-geometric >= 2.3.0 (for advanced GNN features)

### Documentation
- README with quickstart guide
- API reference
- Examples directory with sample code
- Comprehensive docstrings

### Testing
- Unit tests for all modules
- Integration tests
- Example scripts
- Coverage > 80%

### Known Issues
- macOS: OpenMP conflict warning (use `export KMP_DUPLICATE_LIB_OK=TRUE` as workaround)
- GPU support requires CUDA-compatible hardware

---

## [Unreleased]

### Planned Features
- Pre-trained models for different datasets
- Web interface for face recognition
- Real-time video processing
- Mobile deployment support (TensorFlow Lite, ONNX)
- Additional distance metrics
- Advanced augmentation techniques
- Distributed training support
- Model compression techniques
- API server with REST endpoints
- Docker container
- Benchmark suite

---

## Version History

- **v2.0.0** (2025-11-21): Initial public release

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.