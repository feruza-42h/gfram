# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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