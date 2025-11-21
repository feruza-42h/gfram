"""
High-level Face Recognition API.

Provides simple interface for face recognition tasks.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
import logging
import pickle
import json

from ..detectors import FaceDetector, LandmarkNormalizer
from ..geometry.features import GeometricFeatureExtractor
from ..models import GeometricTransformer, GeometricGNN
from ..matching import FaceIndex

logger = logging.getLogger(__name__)


class Recognizer:
    """
    High-level face recognition interface.

    Combines detection, feature extraction, and matching in a simple API.

    Example:
        >>> recognizer = Recognizer.from_pretrained("gfram-base")
        >>> result = recognizer.recognize("photo.jpg")
        >>> print(f"Identity: {result['name']}, Confidence: {result['confidence']}")
    """

    def __init__(
            self,
            detector: Optional[FaceDetector] = None,
            feature_extractor: Optional[GeometricFeatureExtractor] = None,
            model: Optional[torch.nn.Module] = None,
            face_index: Optional['FaceIndex'] = None,
            device: str = "cpu",
    ):
        """
        Initialize Recognizer.

        Args:
            detector: Face detector instance.
            feature_extractor: Geometric feature extractor.
            model: Deep learning model for embeddings.
            face_index: Face database index for matching.
            device: Device to run models on ('cpu' or 'cuda').
        """
        self.device = device

        # Initialize components
        self.detector = detector or FaceDetector()
        self.feature_extractor = feature_extractor or GeometricFeatureExtractor()
        self.model = model
        self.face_index = face_index
        self.normalizer = LandmarkNormalizer()

        # Move model to device
        if self.model is not None:
            self.model.to(device)
            self.model.eval()

        logger.info(f"Recognizer initialized on device: {device}")

    @classmethod
    def from_pretrained(
            cls,
            model_name: str = "gfram-base",
            device: str = "cpu",
            **kwargs
    ) -> 'Recognizer':
        """
        Load a pretrained recognizer.

        Args:
            model_name: Name of pretrained model.
            device: Device to load model on.
            **kwargs: Additional arguments for model loading.

        Returns:
            Initialized Recognizer.
        """
        # TODO: Implement model loading from hub or local cache
        logger.info(f"Loading pretrained model: {model_name}")

        # For now, create with default configuration
        return cls(device=device)

    def recognize(
            self,
            image: Union[str, np.ndarray],
            threshold: float = 0.5,
            top_k: int = 1,
    ) -> Dict:
        """
        Recognize faces in an image.

        Args:
            image: Image path or numpy array.
            threshold: Confidence threshold for recognition.
            top_k: Number of top matches to return.

        Returns:
            Dictionary with recognition results:
            {
                'faces': List of detected faces with identities,
                'num_faces': Number of faces detected
            }
        """
        # Load image if path provided
        if isinstance(image, str):
            import cv2
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")

        # Detect faces
        detected_faces = self.detector.detect(image)

        if not detected_faces:
            return {
                'faces': [],
                'num_faces': 0
            }

        # Process each detected face
        results = []
        for face_data in detected_faces:
            result = self._process_face(face_data, threshold, top_k)
            results.append(result)

        return {
            'faces': results,
            'num_faces': len(results)
        }

    def _process_face(
            self,
            face_data: Dict,
            threshold: float,
            top_k: int
    ) -> Dict:
        """
        Process a single detected face.

        Args:
            face_data: Dictionary with face landmarks and bbox.
            threshold: Confidence threshold.
            top_k: Number of matches to return.

        Returns:
            Dictionary with recognition results for this face.
        """
        landmarks = face_data['landmarks']

        # Normalize landmarks
        landmarks_norm = self.normalizer.normalize(landmarks)

        # Extract geometric features
        geometric_features = self.feature_extractor.extract(landmarks_norm)

        # Get deep learning embedding if model available
        embedding = None
        if self.model is not None:
            landmarks_tensor = torch.from_numpy(landmarks_norm).unsqueeze(0).float()
            landmarks_tensor = landmarks_tensor.to(self.device)

            with torch.no_grad():
                _, embedding = self.model(landmarks_tensor, return_embedding=True)
                if embedding is not None:
                    embedding = embedding.cpu().numpy().flatten()

        # Match against database
        matches = []
        if self.face_index is not None:
            query_vector = embedding if embedding is not None else geometric_features
            matches = self.face_index.search(query_vector, k=top_k, threshold=threshold)

        # Build result
        result = {
            'bbox': face_data.get('bbox'),
            'landmarks': landmarks,
            'geometric_features': geometric_features,
            'embedding': embedding.tolist() if embedding is not None else None,
            'matches': matches,
        }

        # Add identity if confident match found
        if matches and len(matches) > 0:
            best_match = matches[0]
            if best_match['similarity'] >= threshold:
                result['name'] = best_match['name']
                result['confidence'] = best_match['similarity']
            else:
                result['name'] = 'Unknown'
                result['confidence'] = 0.0
        else:
            result['name'] = 'Unknown'
            result['confidence'] = 0.0

        return result

    def add_person(
            self,
            name: str,
            images: Union[str, List[str], np.ndarray, List[np.ndarray]],
    ) -> bool:
        """
        Add a person to the recognition database.

        Args:
            name: Person's name/identifier.
            images: Image(s) of the person (paths or arrays).

        Returns:
            True if successful, False otherwise.
        """
        if self.face_index is None:
            raise RuntimeError("No face index initialized. Create one first.")

        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]

        # Process each image
        embeddings = []
        for img in images:
            # Load if path
            if isinstance(img, str):
                import cv2
                img = cv2.imread(img)
                if img is None:
                    logger.warning(f"Failed to load image: {img}")
                    continue

            # Detect face
            faces = self.detector.detect(img)
            if not faces:
                logger.warning(f"No face detected in image")
                continue

            # Use first detected face
            face_data = faces[0]
            landmarks = face_data['landmarks']
            landmarks_norm = self.normalizer.normalize(landmarks)

            # Extract features
            if self.model is not None:
                landmarks_tensor = torch.from_numpy(landmarks_norm).unsqueeze(0).float()
                landmarks_tensor = landmarks_tensor.to(self.device)

                with torch.no_grad():
                    _, embedding = self.model(landmarks_tensor, return_embedding=True)
                    if embedding is not None:
                        embeddings.append(embedding.cpu().numpy().flatten())
            else:
                # Use geometric features
                features = self.feature_extractor.extract(landmarks_norm)
                embeddings.append(features)

        if not embeddings:
            logger.error(f"Failed to extract features for person: {name}")
            return False

        # Add to index
        embeddings_array = np.array(embeddings)
        self.face_index.add(name, embeddings_array)

        logger.info(f"Added person '{name}' with {len(embeddings)} images")
        return True

    def create_database(
            self,
            save_path: Optional[str] = None
    ) -> 'FaceIndex':
        """
        Create a new face database.

        Args:
            save_path: Path to save the database.

        Returns:
            FaceIndex instance.
        """
        from ..matching import FaceIndex

        # Determine embedding dimension
        if self.model is not None:
            embed_dim = self.model.embed_dim
        else:
            # Use geometric feature dimension
            embed_dim = 150  # Default from GeometricFeatureExtractor

        self.face_index = FaceIndex(dimension=embed_dim)

        if save_path:
            self.face_index.save(save_path)

        return self.face_index

    def load_database(
            self,
            path: str
    ) -> 'FaceIndex':
        """
        Load an existing face database.

        Args:
            path: Path to saved database.

        Returns:
            Loaded FaceIndex instance.
        """
        from ..matching import FaceIndex

        self.face_index = FaceIndex.load(path)
        logger.info(f"Loaded face database from: {path}")

        return self.face_index

    def save(
            self,
            path: str,
            include_database: bool = True
    ):
        """
        Save the recognizer state.

        Args:
            path: Directory to save recognizer.
            include_database: Whether to include face database.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model if present
        if self.model is not None:
            model_path = path / "model.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_class': self.model.__class__.__name__,
            }, model_path)
            logger.info(f"Saved model to {model_path}")

        # Save database if present and requested
        if include_database and self.face_index is not None:
            db_path = path / "face_database"
            self.face_index.save(str(db_path))
            logger.info(f"Saved database to {db_path}")

        # Save config
        config = {
            'device': self.device,
            'has_model': self.model is not None,
            'has_database': self.face_index is not None,
        }

        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Recognizer saved to {path}")

    @classmethod
    def load(
            cls,
            path: str,
            device: str = "cpu"
    ) -> 'Recognizer':
        """
        Load a saved recognizer.

        Args:
            path: Directory containing saved recognizer.
            device: Device to load on.

        Returns:
            Loaded Recognizer instance.
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Initialize recognizer
        recognizer = cls(device=device)

        # Load model if present
        model_path = path / "model.pth"
        if model_path.exists() and config['has_model']:
            checkpoint = torch.load(model_path, map_location=device)
            # TODO: Reconstruct model from class name
            logger.info(f"Loaded model from {model_path}")

        # Load database if present
        db_path = path / "face_database"
        if db_path.exists() and config['has_database']:
            recognizer.load_database(str(db_path))

        logger.info(f"Recognizer loaded from {path}")
        return recognizer

    def get_embedding(
            self,
            image: Union[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Extract embedding vector from an image.

        Args:
            image: Image path or array.

        Returns:
            Embedding vector or None if no face detected.
        """
        # Load image if needed
        if isinstance(image, str):
            import cv2
            image = cv2.imread(image)

        # Detect face
        faces = self.detector.detect(image)
        if not faces:
            return None

        # Process first face
        landmarks = faces[0]['landmarks']
        landmarks_norm = self.normalizer.normalize(landmarks)

        # Extract embedding
        if self.model is not None:
            landmarks_tensor = torch.from_numpy(landmarks_norm).unsqueeze(0).float()
            landmarks_tensor = landmarks_tensor.to(self.device)

            with torch.no_grad():
                _, embedding = self.model(landmarks_tensor, return_embedding=True)
                return embedding.cpu().numpy().flatten() if embedding is not None else None
        else:
            # Use geometric features
            return self.feature_extractor.extract(landmarks_norm)