"""
Geometric feature extraction module.
Extracts 150+ geometric features from facial landmarks.

Feature Categories:
1. Euclidean Features (30): Distances, angles, areas
2. Differential Features (40): Curvatures
3. Topological Features (20): Persistent homology
4. Statistical Features (30): Shape descriptors
5. Symmetry Features (15): Bilateral symmetry
6. Graph Features (15): Delaunay triangulation
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
    """

    def __init__(
            self,
            num_landmarks: int = 468,
            extract_euclidean: bool = True,
            extract_differential: bool = True,
            extract_topological: bool = True,
            extract_statistical: bool = True,
            extract_symmetry: bool = True,
            extract_graph: bool = True,
    ):
        """
        Initialize feature extractor.

        Args:
            num_landmarks: Number of facial landmarks (468 for MediaPipe).
            extract_euclidean: Extract Euclidean geometry features.
            extract_differential: Extract differential geometry features.
            extract_topological: Extract topological features.
            extract_statistical: Extract statistical shape features.
            extract_symmetry: Extract symmetry features.
            extract_graph: Extract graph-based features.
        """
        self.num_landmarks = num_landmarks
        self.extract_euclidean = extract_euclidean
        self.extract_differential = extract_differential
        self.extract_topological = extract_topological
        self.extract_statistical = extract_statistical
        self.extract_symmetry = extract_symmetry
        self.extract_graph = extract_graph

        # Define key landmark indices for MediaPipe (468 points)
        self._define_key_indices()

    def _define_key_indices(self):
        """Define indices for key facial regions."""
        # Eye landmarks
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
        self.outer_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91,
                                  146]
        self.inner_lip_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88,
                                  95]

        # Face outline
        self.face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

        # Jaw landmarks
        self.jaw_indices = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338]

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract all geometric features from landmarks.

        Args:
            landmarks: Landmark array of shape (N, 3) where N is number of landmarks.

        Returns:
            Feature vector as numpy array.
        """
        if landmarks.shape[0] != self.num_landmarks:
            raise ValueError(
                f"Expected {self.num_landmarks} landmarks, got {landmarks.shape[0]}"
            )

        features = []

        # 1. Euclidean features (30)
        if self.extract_euclidean:
            euclidean_feats = self._extract_euclidean_features(landmarks)
            features.append(euclidean_feats)

        # 2. Differential features (40)
        if self.extract_differential:
            differential_feats = self._extract_differential_features(landmarks)
            features.append(differential_feats)

        # 3. Topological features (20)
        if self.extract_topological:
            topological_feats = self._extract_topological_features(landmarks)
            features.append(topological_feats)

        # 4. Statistical features (30)
        if self.extract_statistical:
            statistical_feats = self._extract_statistical_features(landmarks)
            features.append(statistical_feats)

        # 5. Symmetry features (15)
        if self.extract_symmetry:
            symmetry_feats = self._extract_symmetry_features(landmarks)
            features.append(symmetry_feats)

        # 6. Graph features (15)
        if self.extract_graph:
            graph_feats = self._extract_graph_features(landmarks)
            features.append(graph_feats)

        # Concatenate all features
        feature_vector = np.concatenate(features)

        # Handle NaN and Inf values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_vector.astype(np.float32)

    def _extract_euclidean_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract Euclidean geometry features.

        Features:
        - Inter-landmark distances (normalized by interocular distance)
        - Angles between landmark triplets
        - Triangle areas
        - Aspect ratios

        Returns 30 features.
        """
        features = []

        # Get 2D landmarks (x, y)
        points_2d = landmarks[:, :2]

        # Calculate interocular distance for normalization
        left_eye_center = np.mean(landmarks[self.left_eye_indices, :2], axis=0)
        right_eye_center = np.mean(landmarks[self.right_eye_indices, :2], axis=0)
        interocular_dist = np.linalg.norm(left_eye_center - right_eye_center)

        if interocular_dist < 1e-6:
            interocular_dist = 1.0  # Avoid division by zero

        # 1. Key distances (10 features)
        key_distances = []

        # Eye widths
        left_eye_width = self._region_width(landmarks, self.left_eye_indices) / interocular_dist
        right_eye_width = self._region_width(landmarks, self.right_eye_indices) / interocular_dist
        key_distances.extend([left_eye_width, right_eye_width])

        # Eye heights
        left_eye_height = self._region_height(landmarks, self.left_eye_indices) / interocular_dist
        right_eye_height = self._region_height(landmarks, self.right_eye_indices) / interocular_dist
        key_distances.extend([left_eye_height, right_eye_height])

        # Nose dimensions
        nose_length = self._region_height(landmarks, self.nose_bridge_indices) / interocular_dist
        nose_width = self._region_width(landmarks, self.nose_base_indices) / interocular_dist
        key_distances.extend([nose_length, nose_width])

        # Mouth dimensions
        mouth_width = self._region_width(landmarks, self.outer_lip_indices) / interocular_dist
        mouth_height = self._region_height(landmarks, self.outer_lip_indices) / interocular_dist
        key_distances.extend([mouth_width, mouth_height])

        # Face dimensions
        face_width = self._region_width(landmarks, self.face_oval_indices) / interocular_dist
        face_height = self._region_height(landmarks, self.face_oval_indices) / interocular_dist
        key_distances.extend([face_width, face_height])

        features.extend(key_distances)

        # 2. Angles (10 features)
        angles = []

        # Eye angles
        left_eye_angle = self._compute_region_angle(landmarks, self.left_eye_indices)
        right_eye_angle = self._compute_region_angle(landmarks, self.right_eye_indices)
        angles.extend([left_eye_angle, right_eye_angle])

        # Eyebrow angles
        left_brow_angle = self._compute_region_angle(landmarks, self.left_eyebrow_indices)
        right_brow_angle = self._compute_region_angle(landmarks, self.right_eyebrow_indices)
        angles.extend([left_brow_angle, right_brow_angle])

        # Nose angle
        nose_angle = self._compute_region_angle(landmarks, self.nose_bridge_indices)
        angles.append(nose_angle)

        # Mouth angle
        mouth_angle = self._compute_region_angle(landmarks, self.outer_lip_indices)
        angles.append(mouth_angle)

        # Jaw angle
        jaw_angle = self._compute_region_angle(landmarks, self.jaw_indices)
        angles.append(jaw_angle)

        # Additional geometric angles
        # Angle between eyes and nose
        eyes_nose_angle = self._angle_between_points(
            left_eye_center, right_eye_center,
            np.mean(landmarks[self.nose_tip_indices, :2], axis=0)
        )
        angles.append(eyes_nose_angle)

        # Angle between eyes and mouth
        mouth_center = np.mean(landmarks[self.outer_lip_indices, :2], axis=0)
        eyes_mouth_angle = self._angle_between_points(
            left_eye_center, right_eye_center, mouth_center
        )
        angles.append(eyes_mouth_angle)

        # Face tilt angle
        face_tilt = np.arctan2(
            right_eye_center[1] - left_eye_center[1],
            right_eye_center[0] - left_eye_center[0]
        )
        angles.append(face_tilt)

        features.extend(angles)

        # 3. Area ratios (5 features)
        area_ratios = []

        # Eye area ratio
        left_eye_area = self._compute_polygon_area(landmarks[self.left_eye_indices, :2])
        right_eye_area = self._compute_polygon_area(landmarks[self.right_eye_indices, :2])
        eye_area_ratio = left_eye_area / (right_eye_area + 1e-6)
        area_ratios.append(eye_area_ratio)

        # Mouth area
        outer_lip_area = self._compute_polygon_area(landmarks[self.outer_lip_indices, :2])
        inner_lip_area = self._compute_polygon_area(landmarks[self.inner_lip_indices, :2])
        mouth_area_ratio = inner_lip_area / (outer_lip_area + 1e-6)
        area_ratios.append(mouth_area_ratio)

        # Face area
        face_area = self._compute_polygon_area(landmarks[self.face_oval_indices, :2])
        normalized_face_area = face_area / (interocular_dist ** 2)
        area_ratios.append(normalized_face_area)

        # Nose to face ratio
        nose_area = self._compute_polygon_area(landmarks[self.nose_base_indices, :2])
        nose_face_ratio = nose_area / (face_area + 1e-6)
        area_ratios.append(nose_face_ratio)

        # Mouth to face ratio
        mouth_face_ratio = outer_lip_area / (face_area + 1e-6)
        area_ratios.append(mouth_face_ratio)

        features.extend(area_ratios)

        # 4. Aspect ratios (5 features)
        aspect_ratios = []

        # Eye aspect ratios
        left_eye_aspect = left_eye_height / (left_eye_width + 1e-6)
        right_eye_aspect = right_eye_height / (right_eye_width + 1e-6)
        aspect_ratios.extend([left_eye_aspect, right_eye_aspect])

        # Nose aspect ratio
        nose_aspect = nose_length / (nose_width + 1e-6)
        aspect_ratios.append(nose_aspect)

        # Mouth aspect ratio
        mouth_aspect = mouth_height / (mouth_width + 1e-6)
        aspect_ratios.append(mouth_aspect)

        # Face aspect ratio
        face_aspect = face_height / (face_width + 1e-6)
        aspect_ratios.append(face_aspect)

        features.extend(aspect_ratios)

        return np.array(features, dtype=np.float32)

    def _region_width(self, landmarks: np.ndarray, indices: List[int]) -> float:
        """Calculate width of a region."""
        points = landmarks[indices, :2]
        return np.max(points[:, 0]) - np.min(points[:, 0])

    def _region_height(self, landmarks: np.ndarray, indices: List[int]) -> float:
        """Calculate height of a region."""
        points = landmarks[indices, :2]
        return np.max(points[:, 1]) - np.min(points[:, 1])

    def _compute_region_angle(self, landmarks: np.ndarray, indices: List[int]) -> float:
        """Compute orientation angle of a region using PCA."""
        points = landmarks[indices, :2]

        if len(points) < 2:
            return 0.0

        # Center the points
        centered = points - np.mean(points, axis=0)

        # Compute covariance matrix
        cov = np.cov(centered.T)

        # Get principal component (eigenvector with largest eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        principal_component = eigenvectors[:, np.argmax(eigenvalues)]

        # Compute angle
        angle = np.arctan2(principal_component[1], principal_component[0])

        return float(angle)

    def _angle_between_points(
            self,
            p1: np.ndarray,
            p2: np.ndarray,
            p3: np.ndarray
    ) -> float:
        """Compute angle at p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.arccos(cos_angle))

    def _compute_polygon_area(self, points: np.ndarray) -> float:
        """Compute area of polygon using shoelace formula."""
        if len(points) < 3:
            return 0.0

        x = points[:, 0]
        y = points[:, 1]

        area = 0.5 * np.abs(
            np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
        )

        return float(area)

    def _extract_differential_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract differential geometry features (curvatures).

        Features:
        - Curvature of facial contours
        - Mean and Gaussian curvature estimates
        - Curvature statistics

        Returns 40 features.
        """
        features = []

        # Define contours to analyze
        contours = {
            'left_eye': self.left_eye_indices,
            'right_eye': self.right_eye_indices,
            'left_eyebrow': self.left_eyebrow_indices,
            'right_eyebrow': self.right_eyebrow_indices,
            'nose': self.nose_bridge_indices,
            'outer_lip': self.outer_lip_indices,
            'inner_lip': self.inner_lip_indices,
            'jaw': self.jaw_indices,
        }

        for contour_name, indices in contours.items():
            if len(indices) < 3:
                # Not enough points for curvature
                features.extend([0.0] * 5)
                continue

            points = landmarks[indices, :2]
            curvatures = self._compute_curvature(points)

            if len(curvatures) > 0:
                # Statistics of curvature
                mean_curv = np.mean(curvatures)
                std_curv = np.std(curvatures)
                max_curv = np.max(np.abs(curvatures))
                min_curv = np.min(curvatures)
                median_curv = np.median(curvatures)

                features.extend([mean_curv, std_curv, max_curv, min_curv, median_curv])
            else:
                features.extend([0.0] * 5)

        return np.array(features, dtype=np.float32)

    def _compute_curvature(self, points: np.ndarray, window: int = 3) -> np.ndarray:
        """
        Compute curvature along a contour using finite differences.

        Args:
            points: Array of shape (N, 2) representing contour points.
            window: Window size for smoothing.

        Returns:
            Array of curvature values.
        """
        if len(points) < window:
            return np.array([])

        # Smooth the contour using moving average
        smoothed = np.copy(points).astype(float)
        for i in range(len(points)):
            start = max(0, i - window // 2)
            end = min(len(points), i + window // 2 + 1)
            smoothed[i] = np.mean(points[start:end], axis=0)

        # Compute first and second derivatives
        dx = np.gradient(smoothed[:, 0])
        dy = np.gradient(smoothed[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula: k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        numerator = dx * ddy - dy * ddx
        denominator = (dx ** 2 + dy ** 2) ** (3 / 2) + 1e-8
        curvature = numerator / denominator

        return curvature

    def _extract_topological_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract topological features using persistent homology.

        Features:
        - Persistence diagrams statistics
        - Betti numbers
        - Topological signatures

        Returns 20 features.
        """
        features = []

        # For now, use distance-based topological features
        # Full persistent homology requires ripser library

        points_2d = landmarks[:, :2]

        # Compute distance matrix
        dist_matrix = distance_matrix(points_2d, points_2d)

        # 1. Distance statistics (10 features)
        # These capture global shape properties
        dist_mean = np.mean(dist_matrix)
        dist_std = np.std(dist_matrix)
        dist_max = np.max(dist_matrix)
        dist_min = np.min(dist_matrix[dist_matrix > 0])

        # Percentiles
        dist_25 = np.percentile(dist_matrix, 25)
        dist_50 = np.percentile(dist_matrix, 50)
        dist_75 = np.percentile(dist_matrix, 75)
        dist_90 = np.percentile(dist_matrix, 90)

        # Sparsity (ratio of small distances)
        sparsity = np.mean(dist_matrix < dist_mean)

        # Connectivity (average of smallest k distances per point)
        k = 5
        sorted_dists = np.sort(dist_matrix, axis=1)
        connectivity = np.mean(sorted_dists[:, 1:k + 1])

        features.extend([
            dist_mean, dist_std, dist_max, dist_min,
            dist_25, dist_50, dist_75, dist_90,
            sparsity, connectivity
        ])

        # 2. Local topology (10 features)
        # Compute local density and clustering
        for region_indices in [
            self.left_eye_indices, self.right_eye_indices,
            self.nose_bridge_indices, self.outer_lip_indices,
            self.jaw_indices
        ]:
            if len(region_indices) > 0:
                region_points = landmarks[region_indices, :2]
                region_dist = distance_matrix(region_points, region_points)

                # Local density
                local_density = 1.0 / (np.mean(region_dist[region_dist > 0]) + 1e-6)
                features.append(local_density)

                # Local clustering coefficient (simplified)
                local_cluster = np.std(region_dist) / (np.mean(region_dist) + 1e-6)
                features.append(local_cluster)
            else:
                features.extend([0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def _extract_statistical_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract statistical shape features.

        Features:
        - Hu moments (invariant moments)
        - Shape context histograms
        - Distribution statistics

        Returns 30 features.
        """
        features = []

        points_2d = landmarks[:, :2]

        # 1. Hu moments (7 features)
        # These are invariant to translation, scale, and rotation
        hu_moments = self._compute_hu_moments(points_2d)
        features.extend(hu_moments)

        # 2. Spatial distribution statistics (15 features)
        # X-coordinate statistics
        x_coords = points_2d[:, 0]
        x_mean = np.mean(x_coords)
        x_std = np.std(x_coords)
        x_skew = self._compute_skewness(x_coords)
        x_kurt = self._compute_kurtosis(x_coords)
        x_range = np.ptp(x_coords)

        features.extend([x_mean, x_std, x_skew, x_kurt, x_range])

        # Y-coordinate statistics
        y_coords = points_2d[:, 1]
        y_mean = np.mean(y_coords)
        y_std = np.std(y_coords)
        y_skew = self._compute_skewness(y_coords)
        y_kurt = self._compute_kurtosis(y_coords)
        y_range = np.ptp(y_coords)

        features.extend([y_mean, y_std, y_skew, y_kurt, y_range])

        # Z-coordinate statistics (depth)
        z_coords = landmarks[:, 2]
        z_mean = np.mean(z_coords)
        z_std = np.std(z_coords)
        z_skew = self._compute_skewness(z_coords)
        z_kurt = self._compute_kurtosis(z_coords)
        z_range = np.ptp(z_coords)

        features.extend([z_mean, z_std, z_skew, z_kurt, z_range])

        # 3. Shape compactness and eccentricity (8 features)
        # Compute bounding box
        bbox_width = np.ptp(x_coords)
        bbox_height = np.ptp(y_coords)
        bbox_area = bbox_width * bbox_height

        # Convex hull area approximation
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points_2d)
            hull_area = hull.volume  # In 2D, volume is area
            convexity = hull_area / (bbox_area + 1e-6)
        except:
            hull_area = bbox_area
            convexity = 1.0

        # Compactness (circularity)
        perimeter = self._compute_perimeter(points_2d)
        compactness = (4 * np.pi * hull_area) / (perimeter ** 2 + 1e-6)

        # Eccentricity
        eccentricity = bbox_height / (bbox_width + 1e-6)

        # Extent (ratio of points area to bounding box area)
        extent = hull_area / (bbox_area + 1e-6)

        # Solidity (ratio of points area to convex hull area)
        solidity = hull_area / (hull_area + 1e-6)  # Simplified

        # Orientation from PCA
        centered = points_2d - np.mean(points_2d, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        orientation = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

        # Major and minor axis lengths
        major_axis = 2 * np.sqrt(np.max(eigenvalues))
        minor_axis = 2 * np.sqrt(np.min(eigenvalues))

        features.extend([
            compactness, eccentricity, extent, solidity,
            orientation, major_axis, minor_axis, hull_area
        ])

        return np.array(features, dtype=np.float32)

    def _compute_hu_moments(self, points: np.ndarray) -> List[float]:
        """Compute Hu's 7 invariant moments."""
        # Convert points to binary image for moment computation
        # Normalize coordinates
        points_norm = points - np.min(points, axis=0)
        points_norm = points_norm / (np.max(points_norm) + 1e-6)
        points_norm = (points_norm * 100).astype(int)

        # Create binary image
        img_size = 128
        img = np.zeros((img_size, img_size), dtype=np.uint8)

        for pt in points_norm:
            x, y = pt
            if 0 <= x < img_size and 0 <= y < img_size:
                img[y, x] = 255

        # Compute moments using OpenCV
        import cv2
        moments = cv2.moments(img)

        # Compute Hu moments
        hu_moments = cv2.HuMoments(moments).flatten()

        # Log transform for better scale
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        return hu_moments.tolist()

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-6:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-6:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3.0)

    def _compute_perimeter(self, points: np.ndarray) -> float:
        """Compute perimeter of point cloud (using convex hull approximation)."""
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            perimeter = 0.0
            for simplex in hull.simplices:
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                perimeter += np.linalg.norm(p2 - p1)
            return float(perimeter)
        except:
            return 0.0

    def _extract_symmetry_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract bilateral symmetry features.

        Features:
        - Left-right symmetry measures
        - Asymmetry coefficients

        Returns 15 features.
        """
        features = []

        points_2d = landmarks[:, :2]

        # 1. Global symmetry (5 features)
        # Find vertical axis of symmetry (approximate)
        x_center = np.mean(points_2d[:, 0])

        # Mirror points across vertical axis
        mirrored = points_2d.copy()
        mirrored[:, 0] = 2 * x_center - mirrored[:, 0]

        # Compute symmetry error
        symmetry_error = np.mean(np.linalg.norm(points_2d - mirrored, axis=1))
        symmetry_std = np.std(np.linalg.norm(points_2d - mirrored, axis=1))

        # Asymmetry coefficient
        asymmetry_coef = symmetry_error / (x_center + 1e-6)

        # Left-right balance (center of mass deviation)
        com_x = np.mean(points_2d[:, 0])
        lr_balance = (com_x - x_center) / (x_center + 1e-6)

        # Variance asymmetry
        left_mask = points_2d[:, 0] < x_center
        right_mask = points_2d[:, 0] >= x_center

        if np.any(left_mask) and np.any(right_mask):
            left_var = np.var(points_2d[left_mask])
            right_var = np.var(points_2d[right_mask])
            var_asymmetry = (left_var - right_var) / (left_var + right_var + 1e-6)
        else:
            var_asymmetry = 0.0

        features.extend([
            symmetry_error, symmetry_std, asymmetry_coef,
            lr_balance, var_asymmetry
        ])

        # 2. Paired feature symmetry (10 features)
        # Compare left and right features
        paired_regions = [
            (self.left_eye_indices, self.right_eye_indices, "eyes"),
            (self.left_eyebrow_indices, self.right_eyebrow_indices, "eyebrows"),
        ]

        for left_indices, right_indices, name in paired_regions:
            left_points = landmarks[left_indices, :2]
            right_points = landmarks[right_indices, :2]

            # Width difference
            left_width = np.ptp(left_points[:, 0])
            right_width = np.ptp(right_points[:, 0])
            width_diff = (left_width - right_width) / (left_width + right_width + 1e-6)

            # Height difference
            left_height = np.ptp(left_points[:, 1])
            right_height = np.ptp(right_points[:, 1])
            height_diff = (left_height - right_height) / (left_height + right_height + 1e-6)

            # Area difference
            left_area = self._compute_polygon_area(left_points)
            right_area = self._compute_polygon_area(right_points)
            area_diff = (left_area - right_area) / (left_area + right_area + 1e-6)

            # Position difference (Y-coordinate)
            left_y = np.mean(left_points[:, 1])
            right_y = np.mean(right_points[:, 1])
            pos_diff = (left_y - right_y) / (np.mean(points_2d[:, 1]) + 1e-6)

            # Shape difference (using moments)
            left_mom = np.mean(np.var(left_points, axis=0))
            right_mom = np.mean(np.var(right_points, axis=0))
            shape_diff = (left_mom - right_mom) / (left_mom + right_mom + 1e-6)

            features.extend([
                width_diff, height_diff, area_diff, pos_diff, shape_diff
            ])

        return np.array(features, dtype=np.float32)

    def _extract_graph_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract graph-based features using Delaunay triangulation.

        Features:
        - Delaunay triangulation properties
        - Graph spectral features
        - Connectivity measures

        Returns 15 features.
        """
        features = []

        points_2d = landmarks[:, :2]

        # Compute Delaunay triangulation
        try:
            tri = Delaunay(points_2d)

            # 1. Triangulation statistics (7 features)
            num_triangles = len(tri.simplices)
            num_points = len(points_2d)

            # Average triangle area
            triangle_areas = []
            for simplex in tri.simplices:
                pts = points_2d[simplex]
                area = self._compute_polygon_area(pts)
                triangle_areas.append(area)

            avg_triangle_area = np.mean(triangle_areas)
            std_triangle_area = np.std(triangle_areas)
            max_triangle_area = np.max(triangle_areas)
            min_triangle_area = np.min(triangle_areas)

            # Triangle regularity (how close to equilateral)
            regularities = []
            for simplex in tri.simplices:
                pts = points_2d[simplex]
                sides = [
                    np.linalg.norm(pts[1] - pts[0]),
                    np.linalg.norm(pts[2] - pts[1]),
                    np.linalg.norm(pts[0] - pts[2])
                ]
                regularity = np.std(sides) / (np.mean(sides) + 1e-6)
                regularities.append(regularity)

            avg_regularity = np.mean(regularities)

            # Edge statistics
            edges = set()
            for simplex in tri.simplices:
                edges.add(tuple(sorted([simplex[0], simplex[1]])))
                edges.add(tuple(sorted([simplex[1], simplex[2]])))
                edges.add(tuple(sorted([simplex[2], simplex[0]])))

            num_edges = len(edges)

            features.extend([
                num_triangles / num_points,  # Normalized triangle count
                avg_triangle_area,
                std_triangle_area,
                max_triangle_area,
                min_triangle_area,
                avg_regularity,
                num_edges / num_points  # Normalized edge count
            ])

        except Exception as e:
            # If triangulation fails, use default values
            logger.warning(f"Delaunay triangulation failed: {e}")
            features.extend([0.0] * 7)

        # 2. Graph connectivity features (8 features)
        # Build adjacency matrix from Delaunay
        try:
            adjacency = np.zeros((len(points_2d), len(points_2d)))
            for simplex in tri.simplices:
                adjacency[simplex[0], simplex[1]] = 1
                adjacency[simplex[1], simplex[0]] = 1
                adjacency[simplex[1], simplex[2]] = 1
                adjacency[simplex[2], simplex[1]] = 1
                adjacency[simplex[2], simplex[0]] = 1
                adjacency[simplex[0], simplex[2]] = 1

            # Degree statistics
            degrees = np.sum(adjacency, axis=1)
            avg_degree = np.mean(degrees)
            std_degree = np.std(degrees)
            max_degree = np.max(degrees)
            min_degree = np.min(degrees)

            # Clustering coefficient (simplified)
            clustering = np.mean(degrees > avg_degree)

            # Graph density
            density = num_edges / (num_points * (num_points - 1) / 2 + 1e-6)

            # Spectral features (using Laplacian)
            degree_matrix = np.diag(degrees)
            laplacian = degree_matrix - adjacency

            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(laplacian)

            # Algebraic connectivity (second smallest eigenvalue)
            algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0.0

            features.extend([
                avg_degree,
                std_degree,
                max_degree,
                min_degree,
                clustering,
                density,
                algebraic_connectivity,
                eigenvalues[-1]  # Largest eigenvalue
            ])

        except Exception as e:
            logger.warning(f"Graph features computation failed: {e}")
            features.extend([0.0] * 8)

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        names = []

        if self.extract_euclidean:
            names.extend([
                # Key distances (10)
                "left_eye_width", "right_eye_width",
                "left_eye_height", "right_eye_height",
                "nose_length", "nose_width",
                "mouth_width", "mouth_height",
                "face_width", "face_height",

                # Angles (10)
                "left_eye_angle", "right_eye_angle",
                "left_brow_angle", "right_brow_angle",
                "nose_angle", "mouth_angle", "jaw_angle",
                "eyes_nose_angle", "eyes_mouth_angle", "face_tilt",

                # Area ratios (5)
                "eye_area_ratio", "mouth_area_ratio",
                "normalized_face_area", "nose_face_ratio", "mouth_face_ratio",

                # Aspect ratios (5)
                "left_eye_aspect", "right_eye_aspect",
                "nose_aspect", "mouth_aspect", "face_aspect"
            ])

        if self.extract_differential:
            contours = ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow',
                        'nose', 'outer_lip', 'inner_lip', 'jaw']
            for contour in contours:
                names.extend([
                    f"{contour}_curv_mean", f"{contour}_curv_std",
                    f"{contour}_curv_max", f"{contour}_curv_min",
                    f"{contour}_curv_median"
                ])

        # Add other feature names...
        # (truncated for brevity)

        return names