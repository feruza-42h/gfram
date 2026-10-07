"""
Realistic image degradations for robustness experiments.

Occlusions are placed using the landmarks of the clean image, so a "mask"
covers the real lower face and "sunglasses" the real eyes.
"""

import cv2
import numpy as np

# MediaPipe face-oval contour (clockwise from forehead)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE_BRIDGE = 6
IRIS_A, IRIS_B = slice(468, 473), slice(473, 478)


def low_resolution(img, lm, face_px):
    """Downscale so the face is ~face_px wide, then back up (CCTV-like)."""
    width = np.ptp(lm[:, 0])
    f = min(1.0, face_px / max(width, 1))
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def blur(img, lm, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def dark(img, lm, gain, rng):
    """Underexposure with sensor noise."""
    x = img.astype(np.float32) * gain + rng.normal(0, 4, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def jpeg(img, lm, quality):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def mask(img, lm):
    """Surgical-mask style occlusion of everything below the nose bridge."""
    out = img.copy()
    y_top = lm[NOSE_BRIDGE, 1] + 0.35 * (lm[152, 1] - lm[NOSE_BRIDGE, 1])
    pts = [lm[i, :2] for i in FACE_OVAL if lm[i, 1] >= y_top]
    pts = np.array(sorted(pts, key=lambda p: np.arctan2(p[1] - lm[1, 1], p[0] - lm[1, 0])), np.int32)
    hull = cv2.convexHull(np.vstack([pts, [[pts[:, 0].min(), y_top], [pts[:, 0].max(), y_top]]]).astype(np.int32))
    cv2.fillPoly(out, [hull], (200, 170, 120))
    return out


def sunglasses(img, lm):
    out = img.copy()
    a, b = lm[IRIS_A, :2].mean(0), lm[IRIS_B, :2].mean(0)
    d = np.linalg.norm(a - b)
    for c in (a, b):
        cv2.ellipse(out, (int(c[0]), int(c[1])), (int(0.42 * d), int(0.3 * d)), 0, 0, 360, (15, 15, 15), -1)
    cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), (15, 15, 15), max(2, int(0.05 * d)))
    return out


def conditions(rng):
    """name -> function(img, lm)"""
    return {
        'clean': lambda img, lm: img,
        'lowres_24px': lambda img, lm: low_resolution(img, lm, 24),
        'lowres_16px': lambda img, lm: low_resolution(img, lm, 16),
        'blur_s4': lambda img, lm: blur(img, lm, 4),
        'dark': lambda img, lm: dark(img, lm, 0.12, rng),
        'jpeg_q5': lambda img, lm: jpeg(img, lm, 5),
        'mask': mask,
        'sunglasses': sunglasses,
    }
