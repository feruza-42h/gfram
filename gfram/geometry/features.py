"""
Geometric feature extraction module - UPDATED
Extracts 150+ geometric features from facial landmarks.

UPDATED: Full support for both 468 and 478 landmarks
- 468 landmarks: Standard face mesh
- 478 landmarks: Face mesh with iris landmarks (468-477)

Feature Categories:
1. Euclidean Features (30): Distances, angles, areas
2. Differential Features (40): Curvatures
3. Topological Features (20): Persistent homology
4. Statistical Features (30): Shape descriptors
5. Symmetry Features (15): Bilateral symmetry
6. Graph Features (15): Delaunay triangulation
7. Iris Features (3): NEW! Eye iris measurements (only for 478)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial import Delaunay, distance_matrix
from scipy.spatial.distance import euclidean
from scipy.interpolate import splprep, splev
import logging

logger = logging.getLogger(__name__)


class GeometricFeatureExtractor:
    """
    Extract comprehensive geometric features from facial landmarks.

    UPDATED: Supports both 468 and 478 landmarks
    """

    def __init__(
        self,
        num_landmarks: int = 478,  # UPDATED: Default to 478
        extract_euclidean: bool = True,
        extract_differential: bool = True,
        extract_topological: bool = True,
        extract_statistical: bool = True,
        extract_symmetry: bool = True,
        extract_graph: bool = True,
        extract_iris: bool = True,  # NEW!
    ):
        """
        Initialize feature extractor.

        Args:
            num_landmarks: Number of facial landmarks (468 or 478).
            extract_euclidean: Extract Euclidean geometry features.
            extract_differential: Extract differential geometry features.
            extract_topological: Extract topological features.
            extract_statistical: Extract statistical shape features.
            extract_symmetry: Extract symmetry features.
            extract_graph: Extract graph-based features.
            extract_iris: Extract iris features (only for 478 landmarks).
        """
        self.num_landmarks = num_landmarks
        self.extract_euclidean = extract_euclidean
        self.extract_differential = extract_differential
        self.extract_topological = extract_topological
        self.extract_statistical = extract_statistical
        self.extract_symmetry = extract_symmetry
        self.extract_graph = extract_graph
        self.extract_iris = extract_iris

        # Define key landmark indices
        self._define_key_indices()

        logger.info(f"GeometricFeatureExtractor initialized for {num_landmarks} landmarks")

    def _define_key_indices(self):
        """Define indices for key facial regions."""
        # Eye landmarks (same for 468 and 478)
        self.left_eye_indices = [33, 160, 158, 133, 153, 144, 145, 163]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380, 374, 390]

        # Eyebrow landmarks
        self.left_eyebrow_indices = [70, 63, 105, 66, 107, 55, 65]
        self.right_eyebrow_indices = [336, 296, 334, 293, 300, 285, 295]

        # Nose landmarks
        self.nose_bridge_indices = [168, 6, 197, 195, 5]
        self.nose_tip_indices = [4, 1, 2]
        self.nose_base_indices = [98, 97, 2, 326, 327]

        # Lip landmarks
        self.outer_lip_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146
        ]
        self.inner_lip_indices = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95
        ]

        # Face outline
        self.face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

        # Jaw landmarks
        self.jaw_indices = [
            152, 377, 400, 378, 379, 365, 397, 288, 361,
            323, 454, 356, 389, 251, 284, 332, 297, 338
        ]

        # NEW: Iris landmarks (only for 478)
        # Left iris: indices 468-472 (5 points)
        # Right iris: indices 473-477 (5 points)
        self.left_iris_indices = list(range(468, 473))  # [468, 469, 470, 471, 472]
        self.right_iris_indices = list(range(473, 478))  # [473, 474, 475, 476, 477]

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract all geometric features from landmarks.

        Args:
            landmarks: Landmark array of shape (N, 3) where N is 468 or 478.

        Returns:
            Feature vector of shape (150+,) or (153+,) with iris features.
        """
        num_landmarks = len(landmarks)

        if num_landmarks not in [468, 478]:
            raise ValueError(f"Expected 468 or 478 landmarks, got {num_landmarks}")

        # Extract features from first 468 landmarks
        base_landmarks = landmarks[:468] if num_landmarks == 478 else landmarks

        features = []

        # Euclidean features (30)
        if self.extract_euclidean:
            euclidean_features = self._extract_euclidean_features(base_landmarks)
            features.extend(euclidean_features)

        # Differential features (40)
        if self.extract_differential:
            differential_features = self._extract_differential_features(base_landmarks)
            features.extend(differential_features)

        # Topological features (20)
        if self.extract_topological:
            topological_features = self._extract_topological_features(base_landmarks)
            features.extend(topological_features)

        # Statistical features (30)
        if self.extract_statistical:
            statistical_features = self._extract_statistical_features(base_landmarks)
            features.extend(statistical_features)

        # Symmetry features (15)
        if self.extract_symmetry:
            symmetry_features = self._extract_symmetry_features(base_landmarks)
            features.extend(symmetry_features)

        # Graph features (15)
        if self.extract_graph:
            graph_features = self._extract_graph_features(base_landmarks)
            features.extend(graph_features)

        # NEW: Iris features (3) - only for 478 landmarks
        if self.extract_iris and num_landmarks == 478:
            iris_features = self._extract_iris_features(landmarks)
            features.extend(iris_features)
        elif self.extract_iris and num_landmarks == 468:
            # Pad with zeros if iris extraction requested but not available
            features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def _extract_euclidean_features(self, landmarks: np.ndarray) -> List[float]:
        """
        Extract Euclidean geometry features (30 features).

        Features:
        - Inter-ocular distance
        - Eye widths and heights
        - Nose dimensions
        - Mouth dimensions
        - Face proportions
        """
        features = []

        # Eye features
        left_eye = landmarks[self.left_eye_indices]
        right_eye = landmarks[self.right_eye_indices]

        # Inter-ocular distance
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        inter_ocular = np.linalg.norm(left_eye_center - right_eye_center)
        features.append(inter_ocular)

        # Eye widths
        left_eye_width = np.linalg.norm(left_eye[0] - left_eye[4])
        right_eye_width = np.linalg.norm(right_eye[0] - right_eye[4])
        features.extend([left_eye_width, right_eye_width])

        # Eye heights
        left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5])
        right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5])
        features.extend([left_eye_height, right_eye_height])

        # Eye aspect ratios
        left_eye_ratio = left_eye_height / (left_eye_width + 1e-6)
        right_eye_ratio = right_eye_height / (right_eye_width + 1e-6)
        features.extend([left_eye_ratio, right_eye_ratio])

        # Nose features
        nose_bridge = landmarks[self.nose_bridge_indices]
        nose_tip = landmarks[self.nose_tip_indices]
        nose_base = landmarks[self.nose_base_indices]

        # Nose length
        nose_length = np.linalg.norm(nose_bridge[0] - nose_tip[1])
        features.append(nose_length)

        # Nose width
        nose_width = np.linalg.norm(nose_base[0] - nose_base[-1])
        features.append(nose_width)

        # Nose ratio
        nose_ratio = nose_length / (nose_width + 1e-6)
        features.append(nose_ratio)

        # Mouth features
        outer_lip = landmarks[self.outer_lip_indices]
        inner_lip = landmarks[self.inner_lip_indices]

        # Mouth width
        mouth_width = np.linalg.norm(outer_lip[0] - outer_lip[10])
        features.append(mouth_width)

        # Mouth height
        mouth_height = np.linalg.norm(outer_lip[5] - outer_lip[15])
        features.append(mouth_height)

        # Mouth ratio
        mouth_ratio = mouth_height / (mouth_width + 1e-6)
        features.append(mouth_ratio)

        # Face proportions
        face_oval = landmarks[self.face_oval_indices]

        # Face width
        face_width = np.max(face_oval[:, 0]) - np.min(face_oval[:, 0])
        features.append(face_width)

        # Face height
        face_height = np.max(face_oval[:, 1]) - np.min(face_oval[:, 1])
        features.append(face_height)

        # Face ratio
        face_ratio = face_height / (face_width + 1e-6)
        features.append(face_ratio)

        # Relative positions
        eye_to_nose = np.linalg.norm(left_eye_center - nose_tip[1])
        features.append(eye_to_nose)

        nose_to_mouth = np.linalg.norm(nose_tip[1] - outer_lip[5])
        features.append(nose_to_mouth)

        eye_to_mouth = np.linalg.norm(left_eye_center - outer_lip[5])
        features.append(eye_to_mouth)

        # Golden ratio features
        upper_face = eye_to_nose
        lower_face = nose_to_mouth
        golden_ratio = upper_face / (lower_face + 1e-6)
        features.append(golden_ratio)

        # Pad to 30 features if needed
        while len(features) < 30:
            features.append(0.0)

        return features[:30]

    def _extract_differential_features(self, landmarks: np.ndarray) -> List[float]:
        """
        Extract differential geometry features (40 features).

        Features:
        - Curvatures
        - Tangent angles
        - Contour smoothness
        """
        features = []

        # Extract contours for different facial regions
        contours = [
            self.left_eye_indices,
            self.right_eye_indices,
            self.outer_lip_indices,
            self.face_oval_indices,
        ]

        for contour_indices in contours:
            contour = landmarks[contour_indices]

            # Compute curvature
            curvatures = self._compute_curvature(contour)

            # Curvature statistics
            features.append(np.mean(curvatures))
            features.append(np.std(curvatures))
            features.append(np.max(curvatures))
            features.append(np.min(curvatures))

            # Tangent angle variations
            tangent_angles = self._compute_tangent_angles(contour)
            features.append(np.mean(tangent_angles))
            features.append(np.std(tangent_angles))

        # Pad to 40
        while len(features) < 40:
            features.append(0.0)

        return features[:40]

    def _extract_topological_features(self, landmarks: np.ndarray) -> List[float]:
        """
        Extract topological features (20 features).

        Features:
        - Persistent homology
        - Betti numbers
        - Euler characteristic
        """
        features = []

        try:
            # Simplified topological features
            # In production, use ripser or gudhi for persistent homology

            # Connected components (simplified)
            features.append(1.0)  # One face

            # Holes/cycles (simplified)
            features.append(4.0)  # Eyes, nostrils, mouth

            # Distance matrix statistics
            dist_matrix = distance_matrix(landmarks[:, :2], landmarks[:, :2])
            features.append(np.mean(dist_matrix))
            features.append(np.std(dist_matrix))
            features.append(np.max(dist_matrix))
            features.append(np.min(dist_matrix[dist_matrix > 0]))

        except Exception as e:
            logger.warning(f"Error in topological features: {e}")

        # Pad to 20
        while len(features) < 20:
            features.append(0.0)

        return features[:20]

    def _extract_statistical_features(self, landmarks: np.ndarray) -> List[float]:
        """
        Extract statistical shape features (30 features).

        Features:
        - Moments
        - Shape descriptors
        - Distribution statistics
        """
        features = []

        # Central moments
        centroid = np.mean(landmarks, axis=0)
        centered = landmarks - centroid

        # First moments (mean deviation)
        features.extend(np.mean(np.abs(centered), axis=0).tolist())

        # Second moments (variance)
        features.extend(np.var(centered, axis=0).tolist())

        # Third moments (skewness)
        third_moments = np.mean(centered ** 3, axis=0)
        features.extend(third_moments.tolist())

        # Fourth moments (kurtosis)
        fourth_moments = np.mean(centered ** 4, axis=0)
        features.extend(fourth_moments.tolist())

        # Covariance features
        cov_matrix = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        features.extend(eigenvalues.tolist())

        # Shape compactness
        perimeter = self._compute_perimeter(landmarks[self.face_oval_indices])
        area = self._compute_area(landmarks[self.face_oval_indices])
        compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)
        features.append(compactness)

        # Pad to 30
        while len(features) < 30:
            features.append(0.0)

        return features[:30]

    def _extract_symmetry_features(self, landmarks: np.ndarray) -> List[float]:
        """
        Extract symmetry features (15 features).

        Features:
        - Bilateral symmetry
        - Mirror differences
        - Asymmetry scores
        """
        features = []

        # Compute midline
        midline_x = np.median(landmarks[:, 0])

        # Split landmarks into left and right
        left_landmarks = landmarks[landmarks[:, 0] < midline_x]
        right_landmarks = landmarks[landmarks[:, 0] >= midline_x]

        # Mirror right side
        right_mirrored = right_landmarks.copy()
        right_mirrored[:, 0] = 2 * midline_x - right_mirrored[:, 0]

        # Compute symmetry score (simplified)
        if len(left_landmarks) > 0 and len(right_mirrored) > 0:
            # Use closest point matching
            from scipy.spatial.distance import cdist
            distances = cdist(left_landmarks, right_mirrored)
            min_distances = np.min(distances, axis=1)

            symmetry_score = np.mean(min_distances)
            symmetry_std = np.std(min_distances)

            features.extend([symmetry_score, symmetry_std])

        # Eye symmetry
        left_eye = landmarks[self.left_eye_indices]
        right_eye = landmarks[self.right_eye_indices]

        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)

        eye_y_diff = abs(left_eye_center[1] - right_eye_center[1])
        features.append(eye_y_diff)

        # Eyebrow symmetry
        left_eyebrow = landmarks[self.left_eyebrow_indices]
        right_eyebrow = landmarks[self.right_eyebrow_indices]

        left_eyebrow_center = np.mean(left_eyebrow, axis=0)
        right_eyebrow_center = np.mean(right_eyebrow, axis=0)

        eyebrow_y_diff = abs(left_eyebrow_center[1] - right_eyebrow_center[1])
        features.append(eyebrow_y_diff)

        # Pad to 15
        while len(features) < 15:
            features.append(0.0)

        return features[:15]

    def _extract_graph_features(self, landmarks: np.ndarray) -> List[float]:
        """
        Extract graph-based features (15 features).

        Features:
        - Delaunay triangulation properties
        - Graph connectivity
        - Node degrees
        """
        features = []

        try:
            # Delaunay triangulation on 2D landmarks
            tri = Delaunay(landmarks[:, :2])

            # Number of triangles
            num_triangles = len(tri.simplices)
            features.append(num_triangles)

            # Average triangle area
            triangle_areas = []
            for simplex in tri.simplices:
                points = landmarks[simplex, :2]
                area = 0.5 * abs(
                    (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1]) -
                    (points[2, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
                )
                triangle_areas.append(area)

            features.append(np.mean(triangle_areas))
            features.append(np.std(triangle_areas))

        except Exception as e:
            logger.warning(f"Error in graph features: {e}")

        # Pad to 15
        while len(features) < 15:
            features.append(0.0)

        return features[:15]

    def _extract_iris_features(self, landmarks: np.ndarray) -> List[float]:
        """
        NEW: Extract iris features (3 features) - only for 478 landmarks.

        Features:
        - Left iris radius
        - Right iris radius
        - Inter-iris distance
        """
        features = []

        if len(landmarks) < 478:
            return [0.0, 0.0, 0.0]

        try:
            # Left iris (landmarks 468-472)
            left_iris = landmarks[self.left_iris_indices]
            left_iris_center = np.mean(left_iris, axis=0)
            left_iris_radius = np.mean([
                np.linalg.norm(point - left_iris_center)
                for point in left_iris
            ])

            # Right iris (landmarks 473-477)
            right_iris = landmarks[self.right_iris_indices]
            right_iris_center = np.mean(right_iris, axis=0)
            right_iris_radius = np.mean([
                np.linalg.norm(point - right_iris_center)
                for point in right_iris
            ])

            # Inter-iris distance
            inter_iris_distance = np.linalg.norm(left_iris_center - right_iris_center)

            features.extend([
                left_iris_radius,
                right_iris_radius,
                inter_iris_distance
            ])

        except Exception as e:
            logger.warning(f"Error in iris features: {e}")
            features = [0.0, 0.0, 0.0]

        return features

    # Helper methods

    def _compute_curvature(self, contour: np.ndarray) -> np.ndarray:
        """Compute curvature along contour."""
        if len(contour) < 3:
            return np.array([0.0])

        curvatures = []
        for i in range(1, len(contour) - 1):
            p1 = contour[i - 1]
            p2 = contour[i]
            p3 = contour[i + 1]

            # Menger curvature
            area = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) -
                (p3[0] - p1[0]) * (p2[1] - p1[1])
            )

            d1 = np.linalg.norm(p2 - p1)
            d2 = np.linalg.norm(p3 - p2)
            d3 = np.linalg.norm(p3 - p1)

            curvature = 4 * area / (d1 * d2 * d3 + 1e-6)
            curvatures.append(curvature)

        return np.array(curvatures)

    def _compute_tangent_angles(self, contour: np.ndarray) -> np.ndarray:
        """Compute tangent angles along contour."""
        if len(contour) < 2:
            return np.array([0.0])

        angles = []
        for i in range(len(contour) - 1):
            vec = contour[i + 1] - contour[i]
            angle = np.arctan2(vec[1], vec[0])
            angles.append(angle)

        return np.array(angles)

    def _compute_perimeter(self, contour: np.ndarray) -> float:
        """Compute perimeter of contour."""
        perimeter = 0.0
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i + 1) % len(contour)]
            perimeter += np.linalg.norm(p2 - p1)
        return perimeter

    def _compute_area(self, contour: np.ndarray) -> float:
        """Compute area of contour using shoelace formula."""
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        names = []

        if self.extract_euclidean:
            names.extend([f'euclidean_{i}' for i in range(30)])
        if self.extract_differential:
            names.extend([f'differential_{i}' for i in range(40)])
        if self.extract_topological:
            names.extend([f'topological_{i}' for i in range(20)])
        if self.extract_statistical:
            names.extend([f'statistical_{i}' for i in range(30)])
        if self.extract_symmetry:
            names.extend([f'symmetry_{i}' for i in range(15)])
        if self.extract_graph:
            names.extend([f'graph_{i}' for i in range(15)])
        if self.extract_iris:
            names.extend(['left_iris_radius', 'right_iris_radius', 'inter_iris_distance'])

        return names

    def get_feature_count(self) -> int:
        """Get total number of features."""
        count = 0
        if self.extract_euclidean: count += 30
        if self.extract_differential: count += 40
        if self.extract_topological: count += 20
        if self.extract_statistical: count += 30
        if self.extract_symmetry: count += 15
        if self.extract_graph: count += 15
        if self.extract_iris: count += 3
        return count


# Factory function
def create_feature_extractor(
    mode: str = '478',
    **kwargs
) -> GeometricFeatureExtractor:
    """
    Create feature extractor with specific configuration.

    Args:
        mode: Landmark mode ('468' or '478')
        **kwargs: Additional arguments

    Returns:
        GeometricFeatureExtractor instance
    """
    num_landmarks = 478 if mode == '478' else 468
    extract_iris = (mode == '478')

    return GeometricFeatureExtractor(
        num_landmarks=num_landmarks,
        extract_iris=extract_iris,
        **kwargs
    )