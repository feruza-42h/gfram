"""Shared widgets: face view with landmark overlays, cards, charts, background tasks."""

import traceback
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

from . import theme
from .i18n import L

IMAGE_FILTER = 'Images (*.jpg *.jpeg *.png *.bmp *.webp *.heic)'


# ---------------------------------------------------------------- image helpers

def bgr_to_qimage(img: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8, count=img.bytesPerLine() * h).reshape(h, img.bytesPerLine())
    arr = arr[:, :w * 3].reshape(h, w, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def read_image(path) -> Optional[np.ndarray]:
    """BGR image from any path (unicode-safe), large photos downscaled to 1600 px."""
    path = str(path)
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        q = QImage(path)
        if q.isNull():
            return None
        img = qimage_to_bgr(q)
    h, w = img.shape[:2]
    s = 1600 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def pick_images(parent, multiple=False) -> List[str]:
    if multiple:
        files, _ = QFileDialog.getOpenFileNames(parent, L('Rasmlarni tanlang', 'Выберите фото'), str(Path.home()), IMAGE_FILTER)
        return files
    f, _ = QFileDialog.getOpenFileName(parent, L('Rasmni tanlang', 'Выберите фото'), str(Path.home()), IMAGE_FILTER)
    return [f] if f else []


@lru_cache(None)
def mesh_edges() -> np.ndarray:
    try:
        from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
        return np.array([(c.start, c.end) for c in C.FACE_LANDMARKS_TESSELATION], np.int32)
    except Exception:
        return np.zeros((0, 2), np.int32)


@lru_cache(None)
def contour_edges() -> np.ndarray:
    try:
        from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
        return np.array([(c.start, c.end) for c in C.FACE_LANDMARKS_CONTOURS], np.int32)
    except Exception:
        return np.zeros((0, 2), np.int32)


# ---------------------------------------------------------------- layout helpers

def card(title: Optional[str] = None, subtitle: Optional[str] = None, spacing: int = 10):
    """QFrame card and its inner layout."""
    frame = QFrame()
    frame.setObjectName('card')
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 14, 16, 16)
    lay.setSpacing(spacing)
    if title:
        t = QLabel(title)
        t.setObjectName('h2')
        lay.addWidget(t)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName('muted')
        s.setWordWrap(True)
        lay.addWidget(s)
    return frame, lay


def heading(text: str, sub: Optional[str] = None) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 4)
    lay.setSpacing(4)
    t = QLabel(text)
    t.setObjectName('h1')
    lay.addWidget(t)
    if sub:
        s = QLabel(sub)
        s.setObjectName('muted')
        s.setWordWrap(True)
        lay.addWidget(s)
    return w


def label(text: str, muted=False, wrap=True, size=None, bold=False, color=None) -> QLabel:
    lb = QLabel(text)
    lb.setWordWrap(wrap)
    if muted:
        lb.setObjectName('muted')
    style = []
    if size:
        style.append(f'font-size:{size}px')
    if bold:
        style.append('font-weight:700')
    if color:
        style.append(f'color:{color}')
    if style:
        lb.setStyleSheet(';'.join(style))
    lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lb


def button(text: str, kind: str = '', slot: Optional[Callable] = None) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    b.setCursor(Qt.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    return b


class StatTile(QFrame):
    def __init__(self, value: str, caption: str, color: str = theme.TEXT, hint: str = ''):
        super().__init__()
        self.setObjectName('card')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        self.value = QLabel(value)
        self.value.setObjectName('statValue')
        self.value.setStyleSheet(f'color:{color}')
        cap = QLabel(caption)
        cap.setObjectName('statLabel')
        cap.setWordWrap(True)
        lay.addWidget(self.value)
        lay.addWidget(cap)
        if hint:
            self.setToolTip(hint)

    def set(self, value: str, color: Optional[str] = None):
        self.value.setText(value)
        if color:
            self.value.setStyleSheet(f'color:{color}')


class Chart(QWidget):
    """Matplotlib figure embedded in Qt."""

    def __init__(self, width=6.0, height=3.6, projection=None):
        super().__init__()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(width, height), dpi=100, layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet('background: transparent')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.setMinimumHeight(int(height * 80))
        self.projection = projection

    def axes(self, *args, **kw):
        self.figure.clear()
        if self.projection:
            kw.setdefault('projection', self.projection)
        return self.figure.add_subplot(*args, **kw) if args or kw else self.figure.add_subplot(111)

    def draw(self):
        self.canvas.draw_idle()


# ---------------------------------------------------------------- face view

class FaceView(QWidget):
    """
    Shows a BGR image (kept in aspect) with face overlays:
    478 landmarks, triangulated mesh, contours, box and a caption per face.
    """

    def __init__(self, placeholder: str = '', min_h: int = 240):
        super().__init__()
        self.setMinimumHeight(min_h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pix: Optional[QPixmap] = None
        self._size = (1, 1)
        self.faces: List[Dict] = []
        self.placeholder = placeholder or L('Rasm yo‘q', 'Нет изображения')
        self.show_points = True
        self.show_mesh = False
        self.show_contours = True
        self.show_box = True
        self.show_pose = True

    def set_image(self, img: Optional[np.ndarray], faces: Optional[List[Dict]] = None):
        if img is None:
            self._pix = None
        else:
            self._pix = QPixmap.fromImage(bgr_to_qimage(img))
            self._size = (img.shape[1], img.shape[0])
        self.faces = faces or []
        self.update()

    def set_qimage(self, qimg: QImage, size, faces: List[Dict]):
        self._pix = QPixmap.fromImage(qimg)
        self._size = size
        self.faces = faces
        self.update()

    def set_faces(self, faces: List[Dict]):
        self.faces = faces
        self.update()

    def _geometry(self):
        w, h = self._size
        s = min(self.width() / w, self.height() / h)
        ox, oy = (self.width() - w * s) / 2, (self.height() - h * s) / 2
        return s, ox, oy

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor('#0b1016'))
        if self._pix is None:
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        s, ox, oy = self._geometry()
        w, h = self._size
        p.drawPixmap(QRectF(ox, oy, w * s, h * s), self._pix, QRectF(0, 0, self._pix.width(), self._pix.height()))
        for face in self.faces:
            self._paint_face(p, face, s, ox, oy)

    def _paint_face(self, p: QPainter, face: Dict, s, ox, oy):
        color = QColor(face.get('color', theme.ACCENT))
        lm = face.get('landmarks')
        if lm is not None:
            pts = lm[:, :2] * s + np.array([ox, oy])
            if self.show_mesh:
                edges = mesh_edges()
                if len(edges):
                    c = QColor(color)
                    c.setAlpha(70)
                    p.setPen(QPen(c, 0.8))
                    path = QPainterPath()
                    for a, b in edges:
                        path.moveTo(*pts[a])
                        path.lineTo(*pts[b])
                    p.drawPath(path)
            if self.show_contours:
                edges = contour_edges()
                if len(edges):
                    c = QColor(color)
                    c.setAlpha(200)
                    p.setPen(QPen(c, 1.6))
                    path = QPainterPath()
                    for a, b in edges:
                        path.moveTo(*pts[a])
                        path.lineTo(*pts[b])
                    p.drawPath(path)
            if self.show_points:
                c = QColor(color)
                c.setAlpha(230)
                p.setPen(Qt.NoPen)
                p.setBrush(c)
                r = max(1.0, min(2.2, s * 1.3))
                for x, y in pts:
                    p.drawEllipse(QPointF(x, y), r, r)
            if self.show_pose and face.get('pose') is not None:
                self._paint_axes(p, pts, face['pose'])
        box = face.get('bbox')
        if box is not None and self.show_box:
            x, y, bw, bh = box
            rect = QRectF(ox + x * s, oy + y * s, bw * s, bh * s)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(color, 2.5))
            p.drawRoundedRect(rect, 8, 8)
            text = face.get('label')
            if text:
                f = QFont()
                f.setPointSize(15)
                f.setBold(True)
                p.setFont(f)
                lines = text.split('\n')
                fm = p.fontMetrics()
                tw = max(fm.horizontalAdvance(t) for t in lines) + 16
                th = fm.height() * len(lines) + 8
                top = rect.top() - th - 4
                if top < 0:
                    top = rect.bottom() + 4
                bg = QRectF(rect.left(), top, tw, th)
                c = QColor(color)
                c.setAlpha(225)
                p.setPen(Qt.NoPen)
                p.setBrush(c)
                p.drawRoundedRect(bg, 6, 6)
                p.setPen(QColor('#ffffff'))
                for i, t in enumerate(lines):
                    p.drawText(QPointF(bg.left() + 8, bg.top() + 4 + fm.ascent() + i * fm.height()), t)

    @staticmethod
    def _paint_axes(p: QPainter, pts: np.ndarray, pose):
        """Head orientation arrows from the nose tip (yaw/pitch/roll from the 3D landmarks)."""
        yaw, pitch, roll = np.radians(pose)
        nose = pts[1]
        size = np.linalg.norm(pts[33] - pts[263]) * 0.9
        cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
        rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        r = rz @ ry @ rx
        for vec, col in (((1, 0, 0), theme.BAD), ((0, -1, 0), theme.GOOD), ((0, 0, -1), theme.WARN)):
            d = r @ np.array(vec, float)
            end = nose + d[:2] * size
            p.setPen(QPen(QColor(col), 3))
            p.drawLine(QPointF(*nose), QPointF(*end))


class ImageSlot(QFrame):
    """A face image slot: open file, drag & drop, or set programmatically."""

    changed = Signal(object, str)   # image (BGR), source description

    def __init__(self, title: str, min_h: int = 300):
        super().__init__()
        self.setObjectName('card')
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 12)
        top = QHBoxLayout()
        t = QLabel(title)
        t.setObjectName('h2')
        top.addWidget(t)
        top.addStretch(1)
        self.open_btn = button(L('Rasm ochish…', 'Открыть фото…'), slot=self.open_file)
        top.addWidget(self.open_btn)
        lay.addLayout(top)
        self.view = FaceView(L('Rasmni shu yerga torting yoki «Rasm ochish» tugmasini bosing',
                               'Перетащите фото сюда или нажмите «Открыть фото»'), min_h)
        lay.addWidget(self.view, 1)
        self.source = QLabel('')
        self.source.setObjectName('muted')
        lay.addWidget(self.source)
        self.image: Optional[np.ndarray] = None

    def add_button(self, b: QPushButton):
        self.layout().itemAt(0).layout().insertWidget(2, b)

    def open_file(self):
        files = pick_images(self)
        if files:
            self.load(files[0])

    def load(self, path):
        img = read_image(path)
        if img is not None:
            self.set(img, Path(path).name)

    def set(self, img: np.ndarray, source: str):
        self.image = img
        self.view.set_image(img)
        self.source.setText(source)
        self.changed.emit(img, source)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.load(urls[0].toLocalFile())


# ---------------------------------------------------------------- background work

class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(object)


class Task(QRunnable):
    """Run fn(progress_callback) on the thread pool; results come back on the GUI thread."""

    def __init__(self, fn: Callable, on_done=None, on_fail=None, on_progress=None):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()
        self.cancelled = False
        if on_done:
            self.signals.done.connect(on_done)
        if on_progress:
            self.signals.progress.connect(on_progress)
        self.signals.failed.connect(on_fail or (lambda msg: print(msg)))

    def run(self):
        try:
            result = self.fn(self)
            self.signals.done.emit(result)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())

    def report(self, value):
        self.signals.progress.emit(value)


_pool = QThreadPool()
_pool.setMaxThreadCount(2)
_alive = set()


def run_task(fn: Callable, on_done=None, on_fail=None, on_progress=None) -> Task:
    t = Task(fn, on_done, on_fail, on_progress)
    t.setAutoDelete(False)
    _alive.add(t)
    t.signals.done.connect(lambda *_: _alive.discard(t))
    t.signals.failed.connect(lambda *_: _alive.discard(t))
    _pool.start(t)
    return t


def prob_color(p: float, threshold: float) -> str:
    return theme.GOOD if p >= threshold else theme.BAD
