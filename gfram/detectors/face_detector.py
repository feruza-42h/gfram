"""
Face Detection and Landmark Extraction using MediaPipe.

UPDATED: Full support for both 468 and 478 landmarks
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Professional face detector using MediaPipe Face Mesh.

    UPDATED: Supports both 468 and 478 landmarks
    - 468 landmarks: Standard face mesh
    - 478 landmarks: Face mesh with iris landmarks

    Features:
    - Automatic version detection
    - Face detection
    - 3D facial landmarks
    - Face bounding box
    - Face confidence score
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        landmark_mode: str = 'auto'  # NEW: 'auto', '468', '478'
    ):
        """
        Initialize face detector.

        Args:
            static_image_mode: If True, treats each image independently.
            max_num_faces: Maximum number of faces to detect.
            refine_landmarks: Whether to refine landmarks around eyes and lips.
                             When True, MediaPipe returns 478 landmarks (with iris)
                             When False, returns 468 landmarks
            min_detection_confidence: Minimum confidence for face detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            landmark_mode: Landmark mode:
                          - 'auto': Auto-detect (468 or 478)
                          - '468': Force 468 landmarks
                          - '478': Use 478 landmarks (requires refine_landmarks=True)
        """
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.landmark_mode = landmark_mode

        # Detected landmark count
        self.detected_landmark_count = None

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info(f"FaceDetector initialized")
        logger.info(f"  Refine landmarks: {refine_landmarks}")
        logger.info(f"  Expected landmarks: 478 if refine=True, 468 if refine=False")
        logger.info(f"  Landmark mode: {landmark_mode}")

    def detect(
        self,
        image: np.ndarray,
        return_landmarks: bool = True,
        return_bbox: bool = True
    ) -> List[Dict]:
        """
        Detect faces and extract landmarks from an image.

        Args:
            image: Input image (BGR format, as from cv2.imread).
            return_landmarks: Whether to return facial landmarks.
            return_bbox: Whether to return bounding box.

        Returns:
            List of dictionaries, one per detected face, containing:
            - 'landmarks': (N, 3) array of landmark coordinates
            - 'bbox': (x, y, w, h) bounding box
            - 'confidence': detection confidence score
            - 'num_landmarks': number of landmarks (468 or 478)
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to detector")
            return []

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        height, width, _ = image.shape

        # Process image
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return []

        # Extract results for each detected face
        detected_faces = []

        for face_landmarks in results.multi_face_landmarks:
            face_data = {}

            # Extract landmarks
            if return_landmarks:
                landmarks = self._extract_landmarks(face_landmarks, width, height)

                # Auto-detect landmark count on first detection
                if self.detected_landmark_count is None:
                    self.detected_landmark_count = len(landmarks)
                    logger.info(f"Detected {self.detected_landmark_count} landmarks")

                # Handle landmark mode
                if self.landmark_mode == '468' and len(landmarks) == 478:
                    # Force 468 by taking first 468 points
                    landmarks = landmarks[:468]
                    logger.debug("Converted 478 landmarks to 468")
                elif self.landmark_mode == '478' and len(landmarks) == 468:
                    logger.warning("Requested 478 landmarks but got 468. Set refine_landmarks=True")

                face_data['landmarks'] = landmarks
                face_data['num_landmarks'] = len(landmarks)

            # Compute bounding box
            if return_bbox:
                bbox = self._compute_bbox(face_landmarks, width, height)
                face_data['bbox'] = bbox

            # Add confidence
            face_data['confidence'] = 1.0

            detected_faces.append(face_data)

        return detected_faces

    def detect_single(
        self,
        image: np.ndarray,
        return_landmarks: bool = True,
        return_bbox: bool = True
    ) -> Optional[Dict]:
        """
        Detect a single face (the first one found).

        Args:
            image: Input image.
            return_landmarks: Whether to return facial landmarks.
            return_bbox: Whether to return bounding box.

        Returns:
            Dictionary with face data or None if no face detected.
        """
        faces = self.detect(image, return_landmarks, return_bbox)
        return faces[0] if faces else None

    def _extract_landmarks(
        self,
        face_landmarks,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Extract landmark coordinates as numpy array.

        Args:
            face_landmarks: MediaPipe face landmarks object.
            width: Image width.
            height: Image height.

        Returns:
            Array of shape (468, 3) or (478, 3) with (x, y, z) coordinates.
        """
        landmarks = []

        for landmark in face_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z * width  # z is also normalized relative to width

            landmarks.append([x, y, z])

        return np.array(landmarks, dtype=np.float32)

    def _compute_bbox(
        self,
        face_landmarks,
        width: int,
        height: int
    ) -> Tuple[int, int, int, int]:
        """
        Compute bounding box from landmarks.

        Args:
            face_landmarks: MediaPipe face landmarks object.
            width: Image width.
            height: Image height.

        Returns:
            Tuple of (x, y, w, h) for bounding box.
        """
        # Extract all x and y coordinates
        x_coords = [landmark.x * width for landmark in face_landmarks.landmark]
        y_coords = [landmark.y * height for landmark in face_landmarks.landmark]

        # Compute bounding box
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))

        # Add some padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        w = x_max - x_min
        h = y_max - y_min

        return (x_min, y_min, w, h)

    def get_landmark_info(self) -> Dict:
        """
        Get information about landmark configuration.

        Returns:
            Dictionary with landmark info
        """
        return {
            'refine_landmarks': self.refine_landmarks,
            'landmark_mode': self.landmark_mode,
            'detected_count': self.detected_landmark_count,
            'expected_count': 478 if self.refine_landmarks else 468
        }

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


class LandmarkNormalizer:
    """
    Normalize landmarks for consistent representation.

    UPDATED: Supports both 468 and 478 landmarks
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize normalizer.

        Args:
            target_size: Target size for normalized landmarks.
        """
        self.target_size = target_size

    def normalize(
        self,
        landmarks: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Normalize landmarks to standard coordinate system.

        Args:
            landmarks: Input landmarks (468, 3) or (478, 3).
            bbox: Optional bounding box (x, y, w, h).

        Returns:
            Normalized landmarks with same shape as input.
        """
        num_landmarks = len(landmarks)

        if num_landmarks not in [468, 478]:
            logger.warning(f"Unexpected landmark count: {num_landmarks}")

        # Copy to avoid modifying original
        normalized = landmarks.copy()

        # Center landmarks
        centroid = np.mean(normalized[:, :2], axis=0)  # Use x, y only
        normalized[:, :2] -= centroid

        # Scale to unit size
        scale = np.max(np.abs(normalized[:, :2]))
        if scale > 0:
            normalized[:, :2] /= scale

        # Scale z coordinate similarly
        z_scale = np.max(np.abs(normalized[:, 2]))
        if z_scale > 0:
            normalized[:, 2] /= z_scale

        return normalized

    def denormalize(
        self,
        normalized_landmarks: np.ndarray,
        original_landmarks: np.ndarray
    ) -> np.ndarray:
        """
        Denormalize landmarks back to original coordinate system.

        Args:
            normalized_landmarks: Normalized landmarks.
            original_landmarks: Original landmarks for reference.

        Returns:
            Denormalized landmarks.
        """
        # Compute original scale and centroid
        centroid = np.mean(original_landmarks[:, :2], axis=0)
        scale = np.max(np.abs(original_landmarks[:, :2] - centroid))

        # Denormalize
        denormalized = normalized_landmarks.copy()
        denormalized[:, :2] *= scale
        denormalized[:, :2] += centroid

        # Z coordinate
        z_scale = np.max(np.abs(original_landmarks[:, 2]))
        denormalized[:, 2] *= z_scale

        return denormalized


def load_image(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None
) -> Optional[np.ndarray]:
    """
    Load and optionally resize an image.

    Args:
        image_path: Path to image file.
        target_size: Optional target size (width, height).

    Returns:
        Image array in BGR format or None if loading fails.
    """
    try:
        image_path = Path(image_path)

        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        # Load image
        image = cv2.imread(str(image_path))

        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Resize if requested
        if target_size is not None:
            image = cv2.resize(image, target_size)

        return image

    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def create_face_detector(
    mode: str = '478',
    **kwargs
) -> FaceDetector:
    """
    Factory function to create face detector with specific configuration.

    Args:
        mode: Landmark mode:
              - '468': Standard 468 landmarks
              - '478': Enhanced 478 landmarks (with iris)
              - 'auto': Auto-detect
        **kwargs: Additional arguments for FaceDetector

    Returns:
        Configured FaceDetector instance
    """
    if mode == '468':
        return FaceDetector(
            refine_landmarks=False,
            landmark_mode='468',
            **kwargs
        )
    elif mode == '478':
        return FaceDetector(
            refine_landmarks=True,
            landmark_mode='478',
            **kwargs
        )
    else:  # auto
        return FaceDetector(
            refine_landmarks=True,  # Use 478 by default
            landmark_mode='auto',
            **kwargs
        )


# Backward compatibility
def get_face_detector(**kwargs) -> FaceDetector:
    """
    Get face detector with default configuration.

    Returns 478 landmarks by default for best accuracy.
    """
    return create_face_detector(mode='478', **kwargs)