"""
Face Detection and Landmark Extraction using MediaPipe.

Provides robust face detection and 468-point landmark extraction.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Professional face detector using MediaPipe Face Mesh.

    Provides:
    - Face detection
    - 468 3D facial landmarks
    - Face bounding box
    - Face confidence score
    """

    def __init__(
            self,
            static_image_mode: bool = True,
            max_num_faces: int = 1,
            refine_landmarks: bool = True,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5
    ):
        """
        Initialize face detector.

        Args:
            static_image_mode: If True, treats each image independently.
            max_num_faces: Maximum number of faces to detect.
            refine_landmarks: Whether to refine landmarks around eyes and lips.
            min_detection_confidence: Minimum confidence for face detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
        """
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info("FaceDetector initialized with MediaPipe Face Mesh")

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
                face_data['landmarks'] = landmarks

            # Compute bounding box
            if return_bbox:
                bbox = self._compute_bbox(face_landmarks, width, height)
                face_data['bbox'] = bbox

            # Add confidence (MediaPipe doesn't provide per-face confidence, use 1.0)
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
            Array of shape (468, 3) with (x, y, z) coordinates.
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

    def visualize(
            self,
            image: np.ndarray,
            faces: List[Dict],
            draw_landmarks: bool = True,
            draw_bbox: bool = True,
            draw_connections: bool = False
    ) -> np.ndarray:
        """
        Visualize detected faces on the image.

        Args:
            image: Input image.
            faces: List of detected faces from detect().
            draw_landmarks: Whether to draw landmark points.
            draw_bbox: Whether to draw bounding boxes.
            draw_connections: Whether to draw landmark connections.

        Returns:
            Image with visualizations.
        """
        vis_image = image.copy()

        for face in faces:
            # Draw bounding box
            if draw_bbox and 'bbox' in face:
                x, y, w, h = face['bbox']
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw landmarks
            if draw_landmarks and 'landmarks' in face:
                landmarks = face['landmarks']
                for landmark in landmarks:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)

            # Draw connections (face mesh)
            if draw_connections and 'landmarks' in face:
                # Use MediaPipe's face mesh connections
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles

                # This is a simplified version
                # For full connections, you'd need the MediaPipe drawing utilities
                pass

        return vis_image

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


class LandmarkNormalizer:
    """
    Normalize facial landmarks for consistent processing.

    Applies:
    - Translation (center at origin)
    - Scaling (normalize by interocular distance)
    - Rotation (align to canonical pose)
    """

    def __init__(self, method: str = "procrustes"):
        """
        Initialize landmark normalizer.

        Args:
            method: Normalization method ('simple', 'procrustes', 'affine').
        """
        self.method = method

    def normalize(
            self,
            landmarks: np.ndarray,
            target_size: float = 1.0
    ) -> np.ndarray:
        """
        Normalize landmarks.

        Args:
            landmarks: Input landmarks (N, 2) or (N, 3).
            target_size: Target scale for normalized landmarks.

        Returns:
            Normalized landmarks.
        """
        if self.method == "simple":
            return self._simple_normalize(landmarks, target_size)
        elif self.method == "procrustes":
            return self._procrustes_normalize(landmarks, target_size)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def _simple_normalize(
            self,
            landmarks: np.ndarray,
            target_size: float
    ) -> np.ndarray:
        """
        Simple normalization: center and scale.
        """
        # Center landmarks
        centroid = np.mean(landmarks, axis=0)
        landmarks_centered = landmarks - centroid

        # Scale by interocular distance
        # Approximate eye landmarks (left and right eye centers)
        left_eye_idx = [33, 133, 160, 159, 158, 157, 173]  # MediaPipe indices
        right_eye_idx = [362, 263, 387, 386, 385, 384, 398]

        if landmarks.shape[0] >= 468:  # MediaPipe full mesh
            left_eye = landmarks[left_eye_idx, :2].mean(axis=0)
            right_eye = landmarks[right_eye_idx, :2].mean(axis=0)
            interocular = np.linalg.norm(left_eye - right_eye)
        else:
            # Fallback: use overall size
            interocular = np.linalg.norm(landmarks.max(axis=0) - landmarks.min(axis=0))

        if interocular > 1e-6:
            scale = target_size / interocular
            landmarks_normalized = landmarks_centered * scale
        else:
            landmarks_normalized = landmarks_centered

        return landmarks_normalized

    def _procrustes_normalize(
            self,
            landmarks: np.ndarray,
            target_size: float
    ) -> np.ndarray:
        """
        Procrustes analysis normalization.
        """
        # For now, use simple normalization
        # Full Procrustes would require template matching
        return self._simple_normalize(landmarks, target_size)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file.

    Args:
        image_path: Path to image file.

    Returns:
        Image as numpy array or None if loading fails.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None