"""
GFRAM - Basic Usage Example

This example demonstrates the basic usage of GFRAM library for face recognition.
"""

import gfram
import cv2

# Example 1: Simple Face Recognition
print("=" * 50)
print("Example 1: Simple Face Recognition")
print("=" * 50)

# Initialize recognizer
recognizer = gfram.Recognizer()

# Create a new face database
recognizer.create_database()

# Add people to database
print("\n1. Adding people to database...")
recognizer.add_person(
    name="John Doe",
    images=["path/to/john_1.jpg", "path/to/john_2.jpg", "path/to/john_3.jpg"]
)

recognizer.add_person(
    name="Jane Smith",
    images=["path/to/jane_1.jpg", "path/to/jane_2.jpg"]
)

print(f"Database stats: {recognizer.face_index.get_stats()}")

# Recognize a face from a new image
print("\n2. Recognizing face from image...")
result = recognizer.recognize("path/to/test_image.jpg")

print(f"Number of faces detected: {result['num_faces']}")
for i, face in enumerate(result['faces']):
    print(f"\nFace {i + 1}:")
    print(f"  Name: {face['name']}")
    print(f"  Confidence: {face['confidence']:.2%}")
    print(f"  Bounding box: {face['bbox']}")

# Save the recognizer
print("\n3. Saving recognizer...")
recognizer.save("./my_recognizer")
print("Recognizer saved successfully!")

# Example 2: Low-level API Usage
print("\n" + "=" * 50)
print("Example 2: Low-level API Usage")
print("=" * 50)

# Initialize components separately
detector = gfram.FaceDetector()
feature_extractor = gfram.GeometricFeatureExtractor()

# Load and process image
image = cv2.imread("path/to/image.jpg")

# Detect faces
faces = detector.detect(image)
print(f"Detected {len(faces)} face(s)")

if faces:
    # Extract features from first face
    landmarks = faces[0]['landmarks']

    # Normalize landmarks
    normalizer = gfram.LandmarkNormalizer()
    landmarks_norm = normalizer.normalize(landmarks)

    # Extract geometric features
    features = feature_extractor.extract(landmarks_norm)
    print(f"Extracted {len(features)} geometric features")

    # Visualize detection
    vis_image = detector.visualize(
        image,
        faces,
        draw_landmarks=True,
        draw_bbox=True
    )
    cv2.imwrite("output.jpg", vis_image)
    print("Visualization saved to output.jpg")

# Example 3: Working with AI Models
print("\n" + "=" * 50)
print("Example 3: Working with AI Models")
print("=" * 50)

import torch

# Create a Geometric Transformer model
model = gfram.create_geometric_transformer(
    config_name="base",
    num_classes=100  # Number of identities
)

print(f"Model created: {model}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Example forward pass
landmarks_tensor = torch.randn(1, 468, 3)  # Batch of 1, 468 landmarks, 3D coordinates
logits, embedding = model(landmarks_tensor)

print(f"\nModel output shapes:")
print(f"  Logits: {logits.shape if logits is not None else None}")
print(f"  Embedding: {embedding.shape if embedding is not None else None}")

print("\n" + "=" * 50)
print("Examples completed successfully!")
print("=" * 50)