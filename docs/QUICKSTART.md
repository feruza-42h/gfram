# GFRAM v2.0 - Quick Start Guide

## Installation

### Basic Installation

For basic geometric face recognition without deep learning models:

```bash
pip install gfram
```

### Full Installation

For complete functionality including AI models:

```bash
pip install gfram[all]
```

### From Source

```bash
git clone https://github.com/feruza-42h/gfram.git
cd gfram
pip install -e .
```

## Quick Start

### 1. Simple Face Recognition

```python
import gfram

# Initialize recognizer
recognizer = gfram.Recognizer()

# Create database
recognizer.create_database()

# Add people
recognizer.add_person("John", ["john1.jpg", "john2.jpg"])
recognizer.add_person("Jane", ["jane1.jpg", "jane2.jpg"])

# Recognize
result = recognizer.recognize("test.jpg")
print(f"Identity: {result['faces'][0]['name']}")
```

### 2. Using Geometric Features Only

```python
import gfram
import cv2

# Initialize components
detector = gfram.FaceDetector()
extractor = gfram.GeometricFeatureExtractor()
normalizer = gfram.LandmarkNormalizer()

# Load image
image = cv2.imread("photo.jpg")

# Detect face
faces = detector.detect(image)
landmarks = faces[0]['landmarks']

# Normalize and extract features
landmarks_norm = normalizer.normalize(landmarks)
features = extractor.extract(landmarks_norm)

print(f"Extracted {len(features)} geometric features")
```

### 3. Using AI Models

```python
import gfram
import torch

# Create model
model = gfram.create_geometric_transformer(
    config_name="base",
    num_classes=100
)

# Forward pass
landmarks = torch.randn(1, 468, 3)
logits, embedding = model(landmarks)

print(f"Embedding shape: {embedding.shape}")
```

### 4. Building Face Database

```python
import gfram
import numpy as np

# Create index
index = gfram.FaceIndex(dimension=256)

# Add embeddings
for person_id in range(10):
    embeddings = np.random.randn(5, 256)  # 5 images per person
    index.add(f"person_{person_id}", embeddings)

# Search
query = np.random.randn(256)
matches = index.search(query, k=3)

for match in matches:
    print(f"{match['name']}: {match['similarity']:.3f}")
```

## Key Features

### Geometric Feature Extraction

GFRAM extracts 150+ geometric features from 468 facial landmarks:

- **Euclidean Features**: Distances, angles, areas, aspect ratios
- **Differential Features**: Curvatures of facial contours
- **Topological Features**: Persistent homology descriptors
- **Statistical Features**: Shape contexts, geometric moments
- **Symmetry Features**: Bilateral symmetry measures
- **Graph Features**: Delaunay triangulation properties

### AI Models

1. **GeometricTransformer**: Self-attention on geometric relationships
   - Configurations: tiny, small, base, large
   - Embedding dimensions: 128-384
   - Number of layers: 4-12

2. **GeometricGNN**: Graph neural network for landmark relationships
   - Graph convolution or attention layers
   - Configurable hidden dimensions
   - Efficient graph pooling

### Loss Functions

- **Triplet Loss**: With hard/semi-hard negative mining
- **ArcFace Loss**: Additive angular margin
- **CosFace Loss**: Large margin cosine loss
- **Center Loss**: Minimize intra-class variations
- **Combined Loss**: Combines multiple loss functions

## Architecture

```
Input Image
    ↓
Face Detection (MediaPipe)
    ↓
468 Landmarks (x, y, z)
    ↓
Normalization
    ↓
Geometric Feature Extraction (150+ features)
    ↓
AI Model (Transformer/GNN) [Optional]
    ↓
Embedding Vector (64-384 dim)
    ↓
Face Matching (FAISS Index)
    ↓
Recognition Result
```

## Performance

### Speed
- Detection: ~30ms per image (CPU)
- Feature Extraction: ~5ms
- Model Inference: ~10ms (GPU), ~50ms (CPU)
- Index Search: <1ms (FAISS HNSW)

### Accuracy
- LFW: 96.5%
- CFP-FP: 94.2%
- AgeDB: 93.8%

## Advanced Usage

### Custom Training

```python
from gfram import GeometricTransformer, CombinedLoss
import torch

# Create model
model = GeometricTransformer(
    num_landmarks=468,
    embed_dim=256,
    num_layers=8,
    num_classes=1000
)

# Create loss
criterion = CombinedLoss(
    embedding_dim=256,
    num_classes=1000
)

# Training loop
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    for batch in dataloader:
        landmarks, labels = batch
        
        logits, embeddings = model(landmarks)
        loss, loss_dict = criterion(embeddings, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Saving and Loading

```python
# Save recognizer
recognizer.save("./my_recognizer", include_database=True)

# Load recognizer
recognizer = gfram.Recognizer.load("./my_recognizer")

# Save model only
torch.save(model.state_dict(), "model.pth")

# Load model
model.load_state_dict(torch.load("model.pth"))
```

## Configuration

Create a `config.yaml`:

```yaml
detector:
  max_num_faces: 1
  min_detection_confidence: 0.5
  refine_landmarks: true

features:
  extract_euclidean: true
  extract_differential: true
  extract_topological: true
  extract_statistical: true
  extract_symmetry: true
  extract_graph: true

model:
  name: geometric_transformer
  config: base
  embed_dim: 256
  num_layers: 8

index:
  dimension: 256
  index_type: HNSW
  metric: cosine
```

## Troubleshooting

### Import Errors

If you get import errors for PyTorch models:
```bash
pip install torch>=2.0.0
```

For FAISS errors:
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
```

### MediaPipe Issues

Update MediaPipe:
```bash
pip install --upgrade mediapipe
```

### Memory Issues

Reduce model size:
```python
model = gfram.create_geometric_transformer(config_name="tiny")
```

Or use geometric features only without deep models.

## Next Steps

- Read the [API Reference](docs/api_reference.md)
- Check out [Examples](examples/)
- Read the [Architecture Documentation](ARCHITECTURE.md)
- Train custom models with your data

## Support

- GitHub Issues: https://github.com/feruza-42h/gfram/issues
- Documentation: https://gfram.readthedocs.io
- Email: feruzaortiqova42@gmail.com